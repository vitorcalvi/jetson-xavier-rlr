"""Negative integration tests for ``TraderLoop`` + ``OrderController`` + killfiles.

These tests pin the fail-closed surface area of the durable trader:
restart safety, UTC rollover semantics, partial fills, broker outages,
delayed cancel ACKs, late fills after cancel, unknown-POST recovery,
``EMERGENCY``/``FLATTEN``/``KILL`` files, kill-epoch switches, mixed-wire
symbol canonicalisation, sticky HARD incidents, and truncated open-order
snapshots.

The tests reuse the permit / ToolBroker / Ledger wiring from
``tests/trading/test_loop_controller.py`` and drive every loop-level
scenario through the real ``TraderLoop.tick`` entry point so the same
``begin_day`` -> ``killmodes.apply`` -> ``reconcile`` -> ``controller gate``
-> adapter chain is exercised.  Controller-only tests pin the per-call
invariants the loop cannot reach in isolation.

NOTES on current main semantics (asserted by these tests, never weakened):

* :meth:`OrderController.mark_submitting` refuses a BUY unless
  ``mode == RUN`` and the order's stored ``epoch`` matches ``kill_epoch``;
  refused orders become ``CANCELED_LOCAL`` and never hit the broker.
* ``mark_submitting`` itself trips ``HALT_ENTRIES`` (each POST requires
  re-reconcile and a fresh operator ``set_mode(RUN, operator_approved=True)``
  before the next entry is admitted).
* ``HARD`` incidents (``asset_fee_exceeds_filled_quantity`` and the other
  surprises from the controller's hard kernel) survive any matching broker
  snapshot; only ``clear_incident(operator_approved=True)`` clears them.
* :func:`canonical_symbol` rewrites ``"BTCUSD"`` to ``"BTC/USD"`` so a
  reconcile against the ledger does not trip a spurious mismatch.
* :meth:`OrderController.reconcile` with ``complete_snapshot=False`` fails
  closed (trips the controller) on an empty snapshot.
* ``OrderController.begin_day`` records a forward UTC rollover without
  clearing an operator halt; a backward roll trips the controller.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import reflective_loop_runtime.trading.loop as loop_module
from reflective_loop_runtime.kernel.audit import AuditLog
from reflective_loop_runtime.kernel.permit import PermitIssuer, PermitService
from reflective_loop_runtime.kernel.policy import MissionPolicyStore, sign_envelope
from reflective_loop_runtime.kernel.toolbroker import ToolBroker
from reflective_loop_runtime.trading.killmodes import apply as apply_killmodes
from reflective_loop_runtime.trading.ledger import Ledger
from reflective_loop_runtime.trading.loop import (
    ControllerFailStop,
    TraderLoop,
    trading_manifest,
)
from reflective_loop_runtime.trading.model import Decision, Limits
from reflective_loop_runtime.trading.orders import (
    CANCEL_AND_FLATTEN,
    EMERGENCY_STOP,
    HALT_ENTRIES,
    RUN,
    OrderController,
    OrderSafetyError,
    deterministic_order_id,
)

from .fault_broker import FaultBroker

DEV_SECRET = "dev-only-shared-secret"
ISSUER_KEY = b"governor-owned-hmac-key-0123456789"


# ---------------------------------------------------------------------------
# Permit / tool-broker wiring (mirrors tests/trading/test_loop_controller.py)
# ---------------------------------------------------------------------------


def _bundle(**overrides: object) -> dict[str, object]:
    """A minimal mission bundle; ``MissionPolicyStore`` rejects empty ones."""
    bundle: dict[str, object] = {
        "mission_id": "mission-7",
        "objective": "invoke tools under a permit",
        "invariants": ["never_shell_out_unsandboxed"],
        "allowed_domains": [],
        "capabilities": [],
        "budgets": {},
        "standing_mandates": [],
        "policy_version": "1.0.0",
    }
    bundle.update(overrides)
    return bundle


def _policy(
    tmp_path: Path,
    *,
    mandates: tuple[object, ...] = (),
    version: str = "1.0.0",
) -> MissionPolicyStore:
    """Write a signed bundle and load it as a ``MissionPolicyStore``."""
    path = tmp_path / f"policy-{version}-{len(mandates)}.json"
    envelope = sign_envelope(
        _bundle(policy_version=version, standing_mandates=list(mandates)),
        secret=DEV_SECRET,
    )
    path.write_text(json.dumps(envelope), encoding="utf-8")
    return MissionPolicyStore(secret=DEV_SECRET, allow_dev_hmac=True).load(path)


def _service(
    tmp_path: Path,
    manifest: object,
    *,
    mandates: tuple[object, ...] = (),
    version: str = "1.0.0",
) -> PermitService:
    issuer = PermitIssuer(ISSUER_KEY)
    policy = _policy(tmp_path, mandates=mandates, version=version)
    return PermitService(issuer, policy, manifest=manifest)


def _build_broker(
    tmp_path: Path,
    adapter: Callable[[Mapping[str, object]], dict[str, object]] | None,
) -> tuple[PermitService, ToolBroker]:
    """Build a real ``PermitService`` + ``ToolBroker`` for the trader manifest."""
    service = _service(tmp_path, trading_manifest())
    adapters: dict[
        tuple[str, str], Callable[[Mapping[str, object]], dict[str, object]]
    ] = {}
    if adapter is not None:
        adapters[("alpaca_paper", "order")] = adapter
    broker = ToolBroker(
        service,
        audit=AuditLog(":memory:"),
        workspace=tmp_path / "ws",
        manifest=trading_manifest(),
        adapters=adapters,
    )
    return service, broker


# ---------------------------------------------------------------------------
# Feature stand-in
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeFeatures:
    price: float
    atr14: float = 0.0


def _install_fake_features(
    monkeypatch: pytest.MonkeyPatch,
    *,
    price: float = 100.0,
) -> dict[str, float]:
    """Replace ``loop_module.compute_features`` with a price-driven stand-in."""
    state: dict[str, float] = {"price": float(price)}

    def _compute(snap: object, *, now: float | None = None) -> _FakeFeatures:
        return _FakeFeatures(price=state["price"])

    monkeypatch.setattr(loop_module, "compute_features", _compute)
    return state


def _approve(
    _decision: object,
    _features: object,
    _state: object,
    _limits: object,
    *,
    now: float,
) -> tuple[bool, None]:
    return (True, None)


def _noop_market(
    symbols: tuple[str, ...], now: float | None = None, **kw: object
) -> dict[str, object]:
    """Market stand-in: returns a sentinel object per symbol so the loop proceeds."""
    return {sym: object() for sym in symbols}


# ---------------------------------------------------------------------------
# Adapter / controller wiring helpers
# ---------------------------------------------------------------------------


@dataclass
class LoopConfig:
    """A bundle of TraderLoop kwargs for the integration tests."""

    broker: Any
    tool_broker: ToolBroker
    permits: PermitService
    ledger: Ledger
    kill_path: Path
    heartbeat_path: Path
    decide_fn: Callable[..., Any]
    review_fn: Callable[..., Any] = _approve
    symbols: tuple[str, ...] = ("BTC/USD",)
    controller: Any | None = None
    data_dir: Path | None = None
    history_start_iso: str | None = None


def _build_loop(cfg: LoopConfig) -> TraderLoop:
    """Build a TraderLoop from a ``LoopConfig``."""
    return TraderLoop(
        market_fn=_noop_market,
        broker=cfg.broker,
        permits=cfg.permits,
        tool_broker=cfg.tool_broker,
        ledger=cfg.ledger,
        limits=Limits(),
        heartbeat_path=cfg.heartbeat_path,
        kill_path=cfg.kill_path,
        decide_fn=cfg.decide_fn,
        review_fn=cfg.review_fn,
        symbols=cfg.symbols,
        controller=cfg.controller,
        data_dir=cfg.data_dir,
        history_start_iso=cfg.history_start_iso,
    )


def _make_buy_decide(
    *, symbol: str = "BTC/USD", notional: float = 100.0
) -> Callable[..., Decision]:
    def decide_fn(
        features: object, position: object, limits: object, *, now: float
    ) -> Decision:
        return Decision(
            symbol=symbol,
            action="enter_long",
            reason="t",
            stop=90.0,
            target=110.0,
            notional=notional,
        )

    return decide_fn


def _make_exit_decide(
    *, symbol: str = "BTC/USD"
) -> Callable[..., Decision]:
    def decide_fn(
        features: object, position: object, limits: object, *, now: float
    ) -> Decision:
        return Decision(symbol=symbol, action="exit", reason="target")

    return decide_fn


def _make_hold_decide(
    *, symbol: str = "BTC/USD"
) -> Callable[..., Decision]:
    """A decide_fn that returns ``hold`` so no execution happens."""

    def decide_fn(
        features: object, position: object, limits: object, *, now: float
    ) -> Decision:
        return Decision(symbol=symbol, action="hold", reason="no-trade")

    return decide_fn


def _new_controller(tmp_path: Path) -> OrderController:
    """Build a fresh OrderController rooted at ``tmp_path/orders.db``."""
    return OrderController(tmp_path / "orders.db")


def _reconcile_clean(c: OrderController, *, positions: dict[str, Any] | None = None) -> bool:
    """Reconcile with an empty broker view (clean and complete)."""
    return c.reconcile(
        positions or {}, [], account_active=True, complete_snapshot=True
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_restart_starts_halted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new OrderController on the same db starts in HALT_ENTRIES with an incident."""
    _install_fake_features(monkeypatch, price=100.0)

    # First controller: reconcile and approve RUN, then close.
    c1 = _new_controller(tmp_path)
    assert _reconcile_clean(c1)
    c1.set_mode(RUN, operator_approved=True)
    assert c1.snapshot()["mode"] == RUN
    c1.close()

    # Re-open on the same db: must come up HALT_ENTRIES, uncertain, with an incident.
    c2 = _new_controller(tmp_path)
    try:
        snap = c2.snapshot()
        assert snap["mode"] == HALT_ENTRIES
        assert snap["uncertain"] is True
        assert snap["reconciled"] is False
        assert snap["incident"], "a fresh controller must have an incident"

        # prepare_order for a BUY must be refused.
        with pytest.raises(OrderSafetyError):
            c2.prepare_order("BTC/USD", "buy", 1, 1)

        # Drive a loop tick with the re-opened controller; no submit may happen.
        fault = FaultBroker()
        permits, tool_broker = _build_broker(tmp_path, fault.adapter())
        ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")
        cfg = LoopConfig(
            broker=fault,
            tool_broker=tool_broker,
            permits=permits,
            ledger=ledger,
            kill_path=tmp_path / "KILL",
            heartbeat_path=tmp_path / "hb.json",
            decide_fn=_make_buy_decide(),
            controller=c2,
            data_dir=tmp_path,
            history_start_iso="2026-10-05T00:00:00+00:00",
        )
        loop = _build_loop(cfg)
        loop.tick(1000.0)

        assert fault.submit_calls == [], (
            f"no entry may submit on a restarted controller, got {fault.submit_calls!r}"
        )
        ledger.close()
    finally:
        c2.close()


def test_utc_rollover_keeps_halt(tmp_path: Path) -> None:
    """begin_day to the next UTC day keeps a halt in place; no submits."""
    c = _new_controller(tmp_path)
    try:
        assert _reconcile_clean(c)
        c.set_mode(RUN, operator_approved=True)
        c.set_mode(HALT_ENTRIES)
        assert c.snapshot()["mode"] == HALT_ENTRIES

        # Forward UTC roll: must not clear the halt.
        c.begin_day("2026-10-13")
        snap = c.snapshot()
        assert snap["mode"] == HALT_ENTRIES
        assert snap["utc_day"] == "2026-10-13"

        # Entries must still be refused.
        with pytest.raises(OrderSafetyError):
            c.prepare_order("BTC/USD", "buy", 1, 1)
    finally:
        c.close()


def test_partial_fill_recorded_and_reconciled(tmp_path: Path) -> None:
    """A partial fill is recorded; reconcile matches broker qty; no oversell on exit."""
    c = _new_controller(tmp_path)
    try:
        assert _reconcile_clean(c)
        c.set_mode(RUN, operator_approved=True)

        cid = c.prepare_order("BTC/USD", "buy", 1, 2)
        c.mark_submitting(cid)
        c.acknowledge_order(cid, broker_status="partially_filled")
        assert c.record_fill(cid, "partial-fill", 1, fee_in_asset="0.0025")

        snap = c.snapshot()
        assert snap["orders"][cid]["status"] == "PARTIAL"
        assert snap["orders"][cid]["filled_qty"] == "1"
        # 1 - 0.0025 fee
        assert snap["positions"]["BTC/USD"] == "0.9975"

        # Reconcile with the broker reporting the partial fill.
        assert c.reconcile(
            {"BTC/USD": "0.9975"},
            [
                {
                    "client_order_id": cid,
                    "symbol": "BTC/USD",
                    "side": "buy",
                    "filled_qty": "1",
                    "status": "partially_filled",
                }
            ],
            account_active=True,
            complete_snapshot=True,
        )
        assert c.snapshot()["reconciled"]
        assert not c.snapshot()["uncertain"]

        # An exit larger than the held qty must be refused (no oversell).
        with pytest.raises(OrderSafetyError, match="sell exceeds"):
            c.prepare_order("BTC/USD", "sell", 1, 2)
        # A sell within the held quantity is admitted.
        sell_cid = c.prepare_order("BTC/USD", "sell", 2, "0.5")
        assert c.snapshot()["orders"][sell_cid]["side"] == "sell"
    finally:
        c.close()


def test_broker_outage_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``orders_since`` raising trips the controller; entries are blocked."""
    _install_fake_features(monkeypatch, price=100.0)
    fault = FaultBroker(raise_on_history=True)
    permits, tool_broker = _build_broker(tmp_path, fault.adapter())
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    controller = _new_controller(tmp_path)
    assert _reconcile_clean(controller)
    controller.set_mode(RUN, operator_approved=True)

    cfg = LoopConfig(
        broker=fault,
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=_make_buy_decide(),
        controller=controller,
        data_dir=tmp_path,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)
    loop.tick(1000.0)

    try:
        assert fault.submit_calls == [], (
            f"outage must block entries, got {fault.submit_calls!r}"
        )
        snap = controller.snapshot()
        assert snap["mode"] == HALT_ENTRIES
        assert snap["uncertain"] is True
        assert snap["reconciled"] is False
        assert snap["incident"], "an outage must leave an incident on the controller"

        # Subsequent prepare_order must still be refused.
        with pytest.raises(OrderSafetyError):
            controller.prepare_order("BTC/USD", "buy", 2, 1)
    finally:
        controller.close()
        ledger.close()


def test_delayed_cancel_ack_blocks_entries(tmp_path: Path) -> None:
    """Cancel requested but not confirmed -> new entries on the symbol are refused."""
    c = _new_controller(tmp_path)
    try:
        assert _reconcile_clean(c)
        c.set_mode(RUN, operator_approved=True)

        cid = c.prepare_order("BTC/USD", "buy", 1, 1)
        c.mark_submitting(cid)
        c.acknowledge_order(cid)
        c.request_cancel(cid)
        snap = c.snapshot()
        assert snap["orders"][cid]["status"] == "CANCEL_REQUESTED"
        assert snap["orders"][cid]["cancel_requested"] is True

        # Until confirm_canceled, the order is still ACTIVE; new entries on the
        # same symbol must be refused by the existing-unresolved gate.
        with pytest.raises(OrderSafetyError, match="existing unresolved|broker reconciliation required"):
            c.prepare_order("BTC/USD", "buy", 2, 1)

        # Reconcile (broker view: order still in pending_cancel, no broker
        # cancel ACK). The controller stays reconciled-but-active.
        assert c.reconcile(
            {},
            [
                {
                    "client_order_id": cid,
                    "symbol": "BTC/USD",
                    "side": "buy",
                    "filled_qty": "0",
                    "status": "pending_cancel",
                }
            ],
            account_active=True,
            complete_snapshot=True,
        )

        # Cancel finally confirmed -> reconcile a clean CANCELED view, then a
        # fresh entry on the symbol must be admitted.
        c.confirm_canceled(cid)
        assert c.reconcile(
            {},
            [
                {
                    "client_order_id": cid,
                    "symbol": "BTC/USD",
                    "side": "buy",
                    "filled_qty": "0",
                    "status": "canceled",
                }
            ],
            account_active=True,
            complete_snapshot=True,
        )
        c.set_mode(RUN, operator_approved=True)
        fresh = c.prepare_order("BTC/USD", "buy", 3, 1)
        assert c.snapshot()["orders"][fresh]["status"] == "PREPARED"
    finally:
        c.close()


def test_late_fill_after_cancel_trips_halt(tmp_path: Path) -> None:
    """A fill after cancel confirmation trips the controller; entries blocked."""
    c = _new_controller(tmp_path)
    try:
        assert _reconcile_clean(c)
        c.set_mode(RUN, operator_approved=True)
        cid = c.prepare_order("BTC/USD", "buy", 1, 1)
        c.mark_submitting(cid)
        c.acknowledge_order(cid)
        c.request_cancel(cid)
        c.confirm_canceled(cid)
        # A late fill arrives at the broker after the cancel ACK; record it.
        assert c.record_fill(cid, "late-fill", "0.5", fee_in_asset="0.00125")
        snap = c.snapshot()
        assert snap["orders"][cid]["status"] == "PARTIAL"
        assert snap["mode"] == HALT_ENTRIES
        assert snap["incident"] == "late_or_oversize_fill_requires_reconciliation"
        assert snap["uncertain"] is True

        # A later reconcile against a broker view that knows about the cancel +
        # fill must keep the controller tripped (uncertain + an incident).
        assert not c.reconcile(
            {"BTC/USD": "0.49875"},
            [
                {
                    "client_order_id": cid,
                    "symbol": "BTC/USD",
                    "side": "buy",
                    "filled_qty": "0.5",
                    "status": "filled",
                }
            ],
            account_active=True,
            complete_snapshot=True,
        )
        snap2 = c.snapshot()
        assert snap2["mode"] == HALT_ENTRIES
        assert not snap2["reconciled"]
        assert snap2["incident"]

        # Entries stay blocked.
        with pytest.raises(OrderSafetyError):
            c.prepare_order("BTC/USD", "buy", 2, 1)
    finally:
        c.close()


def test_unknown_post_not_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown POST is recovered via lookup; exactly one broker order exists."""
    _install_fake_features(monkeypatch, price=100.0)
    controller = _new_controller(tmp_path)
    assert _reconcile_clean(controller)
    controller.set_mode(RUN, operator_approved=True)

    now = 1000.0
    decision_tick = int(now // 60)
    expected_cid = deterministic_order_id("BTC/USD", "buy", decision_tick)

    # submit_raises_after_accept: records the submit, then raises.
    # The post-raise recovery uses get_order_by_client_id to find the broker
    # record and ack+record it on the controller.
    fault = FaultBroker(submit_raises_after_accept=True)
    permits, tool_broker = _build_broker(tmp_path, fault.adapter())
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    cfg = LoopConfig(
        broker=fault,
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=_make_buy_decide(),
        controller=controller,
        data_dir=tmp_path,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)

    try:
        loop.tick(now)

        # Exactly one POST attempt, even though it raised.
        assert len(fault.submit_calls) == 1, (
            f"expected exactly one submit, got {fault.submit_calls!r}"
        )
        assert fault.submit_calls[0]["client_order_id"] == expected_cid

        # The broker has the order in its lookup; lookup must find it.
        record = fault.get_order_by_client_id(expected_cid)
        assert record is not None
        assert record["client_order_id"] == expected_cid

        # The controller recorded the recovered fill.
        state = controller.snapshot()
        assert expected_cid in state["orders"]
        order_status = state["orders"][expected_cid]["status"]
        assert order_status in {"FILLED", "PARTIAL", "ACKED"}, (
            f"unexpected order status after recovery: {order_status!r}"
        )
    finally:
        controller.close()
        ledger.close()


def test_emergency_blocks_automated_sends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``EMERGENCY`` file blocks every automated send (entries and exits)."""
    _install_fake_features(monkeypatch, price=110.0)
    fault = FaultBroker()
    permits, tool_broker = _build_broker(tmp_path, fault.adapter())
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    # Seed an open position so an exit decision has something to close.
    ledger.open_trade(
        symbol="BTC/USD",
        entry_ts=999.0,
        entry_px=100.0,
        qty=1.0,
        stop=90.0,
        target=110.0,
        atr=0.0,
        entry_order_id="seed-1",
    )

    # The controller only learns positions from its own fills (there is no
    # adoption path for pre-existing broker positions), so seed via a fill.
    controller = _new_controller(tmp_path)
    assert _reconcile_clean(controller)
    controller.set_mode(RUN, operator_approved=True)
    seed = controller.prepare_order("BTC/USD", "buy", 1, 1)
    controller.mark_submitting(seed)
    controller.acknowledge_order(seed, broker_status="new")
    controller.record_fill(seed, "seed-fill", 1, fee_in_asset="0")
    assert controller.reconcile(
        {"BTC/USD": 1.0},
        [{"client_order_id": seed, "symbol": "BTC/USD", "side": "buy",
          "filled_qty": "1", "status": "filled"}],
        account_active=True,
        complete_snapshot=True,
    )
    controller.set_mode(EMERGENCY_STOP)

    (tmp_path / "EMERGENCY").write_text("halt", encoding="utf-8")

    cfg = LoopConfig(
        broker=fault,
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=_make_exit_decide(),
        controller=controller,
        data_dir=tmp_path,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)

    try:
        loop.tick(1000.0)

        # No automated sends at all.
        assert fault.submit_calls == [], (
            f"EMERGENCY must block every automated send, got {fault.submit_calls!r}"
        )
        assert controller.snapshot()["mode"] == EMERGENCY_STOP

        journal = ledger.journal_path().read_text(encoding="utf-8")
        assert any('"kind": "emergency_no_auto_send"' in line for line in journal.splitlines()), (
            "expected an emergency_no_auto_send journal entry"
        )
    finally:
        controller.close()
        ledger.close()


def test_flatten_allows_exits_blocks_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``FLATTEN`` file: exits are admitted, entries are blocked."""
    _install_fake_features(monkeypatch, price=110.0)
    fault = FaultBroker(
        initial_positions=[{"symbol": "BTCUSD", "qty": "1.0"}],
    )
    permits, tool_broker = _build_broker(tmp_path, fault.adapter())
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    # Seed an open trade so an exit has something to close.
    ledger.open_trade(
        symbol="BTC/USD",
        entry_ts=999.0,
        entry_px=100.0,
        qty=1.0,
        stop=90.0,
        target=110.0,
        atr=0.0,
        entry_order_id="seed-1",
    )

    controller = _new_controller(tmp_path)
    assert _reconcile_clean(controller)
    controller.set_mode(RUN, operator_approved=True)
    # Seed a buy that filled, then reconcile against the broker so the
    # controller has a non-zero BTC/USD position.
    buy_id = controller.prepare_order("BTC/USD", "buy", 1, 1)  # integer tick: the prelude proves history coverage from it
    controller.mark_submitting(buy_id)
    controller.acknowledge_order(buy_id, broker_status="accepted")
    controller.record_fill(buy_id, "seed-buy-exec", 1, fee_in_asset="0")
    assert controller.reconcile(
        {"BTC/USD": 1.0},
        [
            {
                "client_order_id": buy_id,
                "symbol": "BTC/USD",
                "side": "buy",
                "filled_qty": "1",
                "status": "filled",
            }
        ],
        account_active=True,
        complete_snapshot=True,
    )
    controller.set_mode(RUN, operator_approved=True)

    (tmp_path / "FLATTEN").write_text("halt", encoding="utf-8")

    cfg = LoopConfig(
        broker=fault,
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=_make_exit_decide(),
        controller=controller,
        data_dir=tmp_path,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)

    try:
        loop.tick(1000.0)

        # FLATTEN escalated the controller to CANCEL_AND_FLATTEN.
        assert controller.snapshot()["mode"] == CANCEL_AND_FLATTEN
        # Exactly one sell POST; no entries (we did not select the buy decision
        # this tick, but the gate would have refused a BUY anyway).
        assert len(fault.submit_calls) == 1, (
            f"expected exactly one sell submit, got {fault.submit_calls!r}"
        )
        call = fault.submit_calls[0]
        assert call["side"] == "sell"
        assert call["symbol"] == "BTC/USD"

        # Now ask the controller to admit a BUY under FLATTEN: refused.
        with pytest.raises(OrderSafetyError, match="buy blocked|broker reconciliation required|entry not authorized"):
            controller.prepare_order("BTC/USD", "buy", 99, 1)
    finally:
        controller.close()
        ledger.close()


def test_kill_file_removal_does_not_deescalate(tmp_path: Path) -> None:
    """Removing a kill file does NOT lower the controller's current mode."""
    c = _new_controller(tmp_path)
    try:
        assert _reconcile_clean(c)
        c.set_mode(RUN, operator_approved=True)

        # Drop a FLATTEN file and apply -> CANCEL_AND_FLATTEN.
        (tmp_path / "FLATTEN").write_text("x", encoding="utf-8")
        assert apply_killmodes(c, tmp_path) == CANCEL_AND_FLATTEN
        assert c.snapshot()["mode"] == CANCEL_AND_FLATTEN

        # Remove the file; apply must not lower the mode.
        (tmp_path / "FLATTEN").unlink()
        assert apply_killmodes(c, tmp_path) is None
        assert c.snapshot()["mode"] == CANCEL_AND_FLATTEN, (
            "removing the FLATTEN file must not lower the controller's mode"
        )

        # The operator must explicitly approve any move down the restrictiveness
        # ladder (this is the post-de-escalation gate the brief calls out).
        with pytest.raises(OrderSafetyError, match="de-escalation"):
            c.set_mode(HALT_ENTRIES)
        with pytest.raises(OrderSafetyError, match="de-escalation"):
            c.set_mode(RUN)
        c.set_mode(HALT_ENTRIES, operator_approved=True)
        assert c.snapshot()["mode"] == HALT_ENTRIES
    finally:
        c.close()


def test_halt_between_prepare_and_post_blocks_entry(tmp_path: Path) -> None:
    """KILL file between prepare_order and the POST bumps kill_epoch; mark_submitting refused."""
    c = _new_controller(tmp_path)
    try:
        assert _reconcile_clean(c)
        c.set_mode(RUN, operator_approved=True)

        # 1. Prepare the buy while RUN: stored epoch == current kill_epoch.
        cid = c.prepare_order("BTC/USD", "buy", 1, 1)
        stored_epoch = c.snapshot()["orders"][cid]["epoch"]
        assert c.snapshot()["orders"][cid]["status"] == "PREPARED"

        # 2. KILL file appears; a killmodes.apply escalates RUN -> HALT_ENTRIES,
        #    bumping kill_epoch. The stored epoch is now stale.
        (tmp_path / "KILL").write_text("x", encoding="utf-8")
        apply_killmodes(c, tmp_path)
        assert c.snapshot()["mode"] == HALT_ENTRIES
        assert c.snapshot()["kill_epoch"] > stored_epoch

        # 3. POST attempt: operator bumps back to RUN, but the epoch gate in
        # mark_submitting still refuses the stale cid. The cid is CANCELED_LOCAL
        # and never reached the broker.
        c.set_mode(RUN, operator_approved=True)
        with pytest.raises(OrderSafetyError, match="entry not authorized"):
            c.mark_submitting(cid)
        assert c.snapshot()["orders"][cid]["status"] == "CANCELED_LOCAL"

        # A fresh cid prepared under the new epoch is submittable.
        cid2 = c.prepare_order("BTC/USD", "buy", 2, 1)
        c.mark_submitting(cid2)
        assert c.snapshot()["orders"][cid2]["status"] == "SUBMITTING"
    finally:
        c.close()


def test_mixed_wire_symbols_reconcile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Broker returns ``BTCUSD``; canonicalised reconcile matches ``BTC/USD`` (no halt)."""
    _install_fake_features(monkeypatch, price=100.0)
    controller = _new_controller(tmp_path)
    assert _reconcile_clean(controller)
    controller.set_mode(RUN, operator_approved=True)

    cid = controller.prepare_order("BTC/USD", "buy", 1, 1)
    controller.mark_submitting(cid)
    controller.acknowledge_order(cid, broker_status="new")
    controller.record_fill(cid, "buy-fill", 1, fee_in_asset="0.0025")
    assert controller.snapshot()["positions"]["BTC/USD"] == "0.9975"
    # Bring the controller back to RUN after the seed tripped it.
    assert controller.reconcile(
        {"BTC/USD": "0.9975"},
        [
            {
                "client_order_id": cid,
                "symbol": "BTC/USD",
                "side": "buy",
                "filled_qty": "1",
                "status": "filled",
            }
        ],
        account_active=True,
        complete_snapshot=True,
    )
    controller.set_mode(RUN, operator_approved=True)

    # The broker's wire format is the bare ``BTCUSD`` (Alpaca's pre-canonical
    # symbol). The FaultBroker must canonicalise on the way out, just like
    # the production PaperBroker.
    fault = FaultBroker(
        initial_positions=[{"symbol": "BTCUSD", "qty": "0.9975"}],
        initial_lookup={
            cid: {
                "id": "seed-filled-1",
                "client_order_id": cid,
                "symbol": "BTCUSD",
                "side": "buy",
                "filled_qty": "1",
                "status": "filled",
            }
        },
    )
    permits, tool_broker = _build_broker(tmp_path, fault.adapter())
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    cfg = LoopConfig(
        broker=fault,
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=_make_hold_decide(),
        controller=controller,
        data_dir=tmp_path,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)

    try:
        loop.tick(1000.0)

        # Reconcile succeeded: no spurious halt, controller still RUN.
        snap = controller.snapshot()
        assert snap["mode"] == RUN
        assert snap["reconciled"] is True
        assert snap["uncertain"] is False
        assert not snap["incident"], (
            f"mixed-wire canonicalisation must not leave an incident; got {snap['incident']!r}"
        )
        # Position survived the canonical rewrite.
        assert snap["positions"]["BTC/USD"] == "0.9975"
    finally:
        controller.close()
        ledger.close()


def test_hard_incident_survives_clean_snapshot(tmp_path: Path) -> None:
    """A fee>fill anomaly is HARD; a later matching snapshot does NOT clear it."""
    c = _new_controller(tmp_path)
    try:
        assert _reconcile_clean(c)
        c.set_mode(RUN, operator_approved=True)
        cid = c.prepare_order("BTC/USD", "buy", 1, 2)
        c.mark_submitting(cid)
        c.acknowledge_order(cid, broker_status="partially_filled")
        # Force a fee > fill to trip the HARD incident on record_fill.
        assert c.record_fill(cid, "huge-fee", 1, fee_in_asset=2)
        snap = c.snapshot()
        assert snap["incident"] == "asset_fee_exceeds_filled_quantity"
        assert snap["mode"] == HALT_ENTRIES

        # A matching broker snapshot must NOT auto-clear this HARD incident.
        assert c.reconcile(
            {"BTC/USD": "0"},
            [
                {
                    "client_order_id": cid,
                    "symbol": "BTC/USD",
                    "side": "buy",
                    "filled_qty": "1",
                    "status": "partially_filled",
                }
            ],
            account_active=True,
            complete_snapshot=True,
        )
        snap = c.snapshot()
        assert snap["incident"] == "asset_fee_exceeds_filled_quantity"
        assert snap["mode"] == HALT_ENTRIES

        # Entries stay blocked.
        with pytest.raises(OrderSafetyError):
            c.prepare_order("BTC/USD", "buy", 2, 1)

        # Only clear_incident(operator_approved=True) after a clean reconcile
        # can clear a HARD incident.
        c.clear_incident(operator_approved=True)
        assert c.snapshot()["incident"] == ""
    finally:
        c.close()


def test_open_orders_truncated_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``open_orders``/``orders_since`` raising ``incomplete`` blocks entries."""
    _install_fake_features(monkeypatch, price=100.0)
    fault = FaultBroker(raise_on_history=True)
    permits, tool_broker = _build_broker(tmp_path, fault.adapter())
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    controller = _new_controller(tmp_path)
    assert _reconcile_clean(controller)
    controller.set_mode(RUN, operator_approved=True)

    cfg = LoopConfig(
        broker=fault,
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=_make_buy_decide(),
        controller=controller,
        data_dir=tmp_path,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)

    try:
        loop.tick(1000.0)

        # No entries submitted.
        assert fault.submit_calls == [], (
            f"truncated history must block entries; got {fault.submit_calls!r}"
        )
        snap = controller.snapshot()
        assert snap["mode"] == HALT_ENTRIES
        assert snap["reconciled"] is False
        assert snap["uncertain"] is True
        assert snap["incident"], "truncated history must leave an incident"

        journal = ledger.journal_path().read_text(encoding="utf-8")
        assert any('"kind": "reconcile_failed"' in line for line in journal.splitlines()), (
            "expected a reconcile_failed journal entry on truncated history"
        )

        # A subsequent prepare_order must still be refused.
        with pytest.raises(OrderSafetyError):
            controller.prepare_order("BTC/USD", "buy", 2, 1)
    finally:
        controller.close()
        ledger.close()


def test_killmode_error_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``killmodes.apply`` ``OSError`` trips the prelude; no entry submits.

    The controller leaves RUN, the journal records ``prelude_failed`` with
    ``stage="killmodes"`` and the subsequent ``prepare_order`` is refused
    because the controller is no longer clean.
    """
    _install_fake_features(monkeypatch, price=100.0)

    def _boom(_ctrl: object, _data_dir: object) -> None:
        raise OSError("kill dir unreadable")

    # The loop's prelude imports the killmodes module and then resolves
    # ``killmodes.apply`` at call time, so patching the module attribute
    # actually changes what the prelude sees.
    import reflective_loop_runtime.trading.killmodes as _killmodes_module
    monkeypatch.setattr(_killmodes_module, "apply", _boom)

    fault = FaultBroker()
    permits, tool_broker = _build_broker(tmp_path, fault.adapter())
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    controller = _new_controller(tmp_path)
    assert _reconcile_clean(controller)
    controller.set_mode(RUN, operator_approved=True)

    cfg = LoopConfig(
        broker=fault,
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=_make_buy_decide(),
        controller=controller,
        data_dir=tmp_path,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)
    try:
        loop.tick(1000.0)

        assert fault.submit_calls == [], (
            f"killmode error must block entries; got {fault.submit_calls!r}"
        )
        snap = controller.snapshot()
        assert snap["mode"] == HALT_ENTRIES
        assert snap["reconciled"] is False
        assert snap["uncertain"] is True
        assert snap["incident"], (
            "controller must carry an incident after killmodes.apply raised"
        )

        journal = ledger.journal_path().read_text(encoding="utf-8")
        assert any(
            '"kind": "prelude_failed"' in line and '"stage": "killmodes"' in line
            for line in journal.splitlines()
        ), "expected a prelude_failed journal entry for the killmodes stage"

        with pytest.raises(OrderSafetyError):
            controller.prepare_order("BTC/USD", "buy", 2, 1)
    finally:
        controller.close()
        ledger.close()


def test_unknown_open_order_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An open order whose cid is not on the controller fails the prelude."""
    _install_fake_features(monkeypatch, price=100.0)
    # Seed one open order that the controller has never seen.
    foreign_cid = "rlr-foreign0000000000000000unknown"
    fault = FaultBroker(
        initial_orders=[
            {
                "id": "broker-order-foreign",
                "client_order_id": foreign_cid,
                "symbol": "BTC/USD",
                "side": "buy",
                "status": "new",
                "filled_qty": "0",
            }
        ]
    )
    permits, tool_broker = _build_broker(tmp_path, fault.adapter())
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    controller = _new_controller(tmp_path)
    assert _reconcile_clean(controller)
    controller.set_mode(RUN, operator_approved=True)

    cfg = LoopConfig(
        broker=fault,
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=_make_buy_decide(),
        controller=controller,
        data_dir=tmp_path,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)
    try:
        loop.tick(1000.0)

        assert fault.submit_calls == [], (
            f"unknown open order must block entries; got {fault.submit_calls!r}"
        )
        snap = controller.snapshot()
        assert snap["mode"] == HALT_ENTRIES
        assert snap["reconciled"] is False
        assert snap["uncertain"] is True
        assert snap["incident"], (
            "controller must carry an incident when an unknown open order surfaces"
        )

        journal = ledger.journal_path().read_text(encoding="utf-8")
        assert any(
            '"kind": "prelude_failed"' in line
            and '"stage": "unknown_open_order"' in line
            for line in journal.splitlines()
        ), "expected a prelude_failed journal entry for the unknown_open_order stage"

        with pytest.raises(OrderSafetyError):
            controller.prepare_order("BTC/USD", "buy", 2, 1)
    finally:
        controller.close()
        ledger.close()


def test_inactive_account_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An inactive account trips the prelude; no entry submits."""
    _install_fake_features(monkeypatch, price=100.0)
    fault = FaultBroker(account_active=False)
    permits, tool_broker = _build_broker(tmp_path, fault.adapter())
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    controller = _new_controller(tmp_path)
    assert _reconcile_clean(controller)
    controller.set_mode(RUN, operator_approved=True)

    cfg = LoopConfig(
        broker=fault,
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=_make_buy_decide(),
        controller=controller,
        data_dir=tmp_path,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)
    try:
        loop.tick(1000.0)

        assert fault.submit_calls == [], (
            f"inactive account must block entries; got {fault.submit_calls!r}"
        )
        snap = controller.snapshot()
        assert snap["mode"] == HALT_ENTRIES
        assert snap["reconciled"] is False
        assert snap["uncertain"] is True
        assert snap["incident"], (
            "controller must carry an incident when the account is inactive"
        )

        with pytest.raises(OrderSafetyError):
            controller.prepare_order("BTC/USD", "buy", 2, 1)
    finally:
        controller.close()
        ledger.close()


def test_missing_kill_dir_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructing the loop without a ``data_dir`` trips the prelude."""
    _install_fake_features(monkeypatch, price=100.0)
    fault = FaultBroker()
    permits, tool_broker = _build_broker(tmp_path, fault.adapter())
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    controller = _new_controller(tmp_path)
    assert _reconcile_clean(controller)
    controller.set_mode(RUN, operator_approved=True)

    cfg = LoopConfig(
        broker=fault,
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=_make_buy_decide(),
        controller=controller,
        data_dir=None,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)
    try:
        loop.tick(1000.0)

        assert fault.submit_calls == [], (
            f"missing kill dir must block entries; got {fault.submit_calls!r}"
        )
        snap = controller.snapshot()
        assert snap["mode"] == HALT_ENTRIES
        assert snap["reconciled"] is False
        assert snap["uncertain"] is True
        assert snap["incident"], (
            "controller must carry an incident when the kill dir is missing"
        )

        journal = ledger.journal_path().read_text(encoding="utf-8")
        assert any(
            '"kind": "prelude_failed"' in line and '"stage": "no_kill_dir"' in line
            for line in journal.splitlines()
        ), (
            "expected a prelude_failed journal entry for the no_kill_dir stage"
        )

        with pytest.raises(OrderSafetyError):
            controller.prepare_order("BTC/USD", "buy", 2, 1)
    finally:
        controller.close()
        ledger.close()

"""A fill recovered after an unknown POST must be persisted to the Ledger.

Mirrors :func:`test_unknown_post_not_retried`, but additionally asserts that the
``_record_fill_in_ledger`` path was taken on the recovery branch, so RiskState
and the exit logic see the broker's exposure.
"""


def test_recovered_fill_persisted_to_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recovered fill writes the Ledger; the trade's qty/px mirror the broker."""
    _install_fake_features(monkeypatch, price=100.0)
    controller = _new_controller(tmp_path)
    assert _reconcile_clean(controller)
    controller.set_mode(RUN, operator_approved=True)

    now = 1000.0
    decision_tick = int(now // 60)
    expected_cid = deterministic_order_id("BTC/USD", "buy", decision_tick)

    # The POST raises after the broker accepted the order; recovery looks it up.
    fault = FaultBroker(submit_raises_after_accept=True)
    permits, tool_broker = _build_broker(tmp_path, fault.adapter())
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    # Defaults: account_active=True, open_orders=[].
    assert fault.account_status()["active"] is True
    assert fault.open_orders() == []

    cfg = LoopConfig(
        broker=fault,
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=_make_buy_decide(),
        controller=controller,
        data_dir=tmp_path,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)

    try:
        loop.tick(now)

        # Exactly one POST attempt, even though it raised.
        assert len(fault.submit_calls) == 1, (
            f"expected exactly one submit, got {fault.submit_calls!r}"
        )
        assert fault.submit_calls[0]["client_order_id"] == expected_cid

        # The broker has the order in its lookup; it is fully filled.
        record = fault.get_order_by_client_id(expected_cid)
        assert record is not None
        broker_filled_qty = float(record["filled_qty"])
        broker_filled_px = float(record["filled_avg_price"])
        assert broker_filled_qty > 0
        assert broker_filled_px > 0

        # The recovery also wrote the Ledger: an open trade must now exist.
        assert ledger.has_open("BTC/USD") is True, (
            "recovery did not persist the fill to the Ledger"
        )
        positions = ledger.open_positions()
        assert "BTC/USD" in positions
        trade = positions["BTC/USD"]
        assert float(trade.qty) == broker_filled_qty, (
            f"ledger qty {trade.qty} != broker filled_qty {broker_filled_qty}"
        )
        assert float(trade.entry_px) == broker_filled_px, (
            f"ledger entry_px {trade.entry_px} != broker filled_avg_price {broker_filled_px}"
        )
    finally:
        controller.close()
        ledger.close()


def _new_controller(tmp_path: Path) -> OrderController:
    """Build a fresh OrderController rooted at ``tmp_path/orders.db``."""
    return OrderController(tmp_path / "orders.db")


def _reconcile_clean(c: OrderController) -> bool:
    return c.reconcile({}, [], account_active=True, complete_snapshot=True)


def test_nonexistent_kill_dir_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``data_dir`` pointing at a missing path must fail closed (D1)."""
    fault = FaultBroker()
    permits, tool_broker = _build_broker(tmp_path, fault.adapter())
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    controller = _new_controller(tmp_path)
    assert _reconcile_clean(controller)
    controller.set_mode(RUN, operator_approved=True)
    controller_snap_before = controller.snapshot()
    assert controller_snap_before["mode"] == RUN
    assert controller_snap_before["incident"] == ""

    missing_dir = tmp_path / "missing"
    assert not missing_dir.exists()

    cfg = LoopConfig(
        broker=fault,
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=_make_buy_decide(),
        controller=controller,
        data_dir=missing_dir,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)
    try:
        loop.tick(1000.0)

        assert fault.submit_calls == [], (
            f"missing kill dir must block entries; got {fault.submit_calls!r}"
        )
        snap = controller.snapshot()
        assert snap["mode"] != RUN or snap["incident"], (
            "controller must no longer be RUN-clean when kill dir is missing"
        )
        assert snap["incident"], (
            "controller must carry an incident when the kill dir is missing"
        )
        assert snap["mode"] == HALT_ENTRIES

        journal = ledger.journal_path().read_text(encoding="utf-8")
        assert any(
            '"kind": "prelude_failed"' in line
            and '"stage": "kill_dir_unavailable"' in line
            for line in journal.splitlines()
        ), (
            "expected a prelude_failed journal entry for the "
            "kill_dir_unavailable stage"
        )
    finally:
        controller.close()
        ledger.close()


def test_symlinked_kill_dir_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A symlinked data_dir must fail closed (D1)."""
    fault = FaultBroker()
    permits, tool_broker = _build_broker(tmp_path, fault.adapter())
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    controller = _new_controller(tmp_path)
    assert _reconcile_clean(controller)
    controller.set_mode(RUN, operator_approved=True)
    assert controller.snapshot()["mode"] == RUN

    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real_dir)

    cfg = LoopConfig(
        broker=fault,
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=_make_buy_decide(),
        controller=controller,
        data_dir=link,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)
    try:
        loop.tick(1000.0)

        assert fault.submit_calls == [], (
            f"symlinked kill dir must block entries; got {fault.submit_calls!r}"
        )
        snap = controller.snapshot()
        assert snap["incident"], (
            "symlinked kill dir must trip the controller with an incident"
        )
        assert snap["mode"] == HALT_ENTRIES
        assert snap["reconciled"] is False

        journal = ledger.journal_path().read_text(encoding="utf-8")
        assert any(
            '"kind": "prelude_failed"' in line
            and '"stage": "kill_dir_unavailable"' in line
            for line in journal.splitlines()
        ), (
            "expected a prelude_failed journal entry for the "
            "kill_dir_unavailable stage"
        )
    finally:
        controller.close()
        ledger.close()


def test_journal_error_still_trips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing ``ledger.journal`` must not stop the controller from being tripped (D2)."""
    fault = FaultBroker()
    permits, tool_broker = _build_broker(tmp_path, fault.adapter())
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    controller = _new_controller(tmp_path)
    assert _reconcile_clean(controller)
    controller.set_mode(RUN, operator_approved=True)
    assert controller.snapshot()["mode"] == RUN

    real_dir = tmp_path / "aux"  # not "kill": collides with KILL on case-insensitive filesystems
    real_dir.mkdir()

    journal_calls: list[dict[str, object]] = []

    def _raising_journal(payload: dict[str, object]) -> None:
        journal_calls.append(dict(payload))
        raise OSError("disk full: journal write failed")

    monkeypatch.setattr(ledger, "journal", _raising_journal)

    cfg = LoopConfig(
        broker=fault,
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=_make_buy_decide(),
        controller=controller,
        data_dir=None,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)
    try:
        # The loop's own later journal writes also raise (disk full); the
        # safety property is that the controller was halted BEFORE that.
        with pytest.raises(OSError):
            loop.tick(1000.0)
        assert loop.tick_count == 1

        # No broker submit must have happened.
        assert fault.submit_calls == [], (
            f"journal failure must still block entries; got {fault.submit_calls!r}"
        )

        snap = controller.snapshot()
        assert snap["mode"] == HALT_ENTRIES, (
            "controller must be tripped into HALT_ENTRIES even when journal "
            "fails; got snapshot: " + repr(snap)
        )
        assert snap["incident"], (
            "controller must carry an incident even when journal fails"
        )
        assert snap["reconciled"] is False
        assert snap["uncertain"] is True

        # The journal call was actually attempted (the trip-first ordering
        # requires the safety signal to fire regardless of journal IO).
        assert journal_calls, "expected the loop to attempt journaling"
    finally:
        controller.close()
        ledger.close()


def test_recovered_fill_ledger_exception_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recovered fill whose ledger write raises OSError must fail closed (D3)."""
    _install_fake_features(monkeypatch, price=100.0)
    controller = _new_controller(tmp_path)
    assert _reconcile_clean(controller)
    controller.set_mode(RUN, operator_approved=True)
    assert controller.snapshot()["mode"] == RUN

    now = 1000.0
    decision_tick = int(now // 60)
    expected_cid = deterministic_order_id("BTC/USD", "buy", decision_tick)

    fault = FaultBroker(submit_raises_after_accept=True)
    permits, tool_broker = _build_broker(tmp_path, fault.adapter())
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    # Confirm the broker's view: account active, no open orders.
    assert fault.account_status()["active"] is True
    assert fault.open_orders() == []

    # Make ledger.open_trade raise OSError so the recovery's
    # _record_fill_in_ledger -> ledger.open_trade path explodes.
    def _raising_open_trade(**kwargs: Any) -> None:
        raise OSError("disk full: ledger.open_trade failed")

    monkeypatch.setattr(ledger, "open_trade", _raising_open_trade)

    cfg = LoopConfig(
        broker=fault,
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=_make_buy_decide(),
        controller=controller,
        data_dir=tmp_path,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)

    try:
        # The tick must complete without raising out.
        loop.tick(now)

        assert loop.tick_count == 1

        # Exactly one POST attempt: the broker accepts then raises.
        assert len(fault.submit_calls) == 1, (
            f"expected exactly one submit, got {fault.submit_calls!r}"
        )
        assert fault.submit_calls[0]["client_order_id"] == expected_cid

        # The ledger never received a successful open_trade call.
        assert ledger.has_open("BTC/USD") is False

        snap = controller.snapshot()
        assert snap["incident"], (
            "controller must carry an incident when recovered_fill ledger write "
            f"fails; got snapshot: {snap!r}"
        )
        assert snap["mode"] == HALT_ENTRIES
        assert snap["reconciled"] is False
        # The recovery branch itself caught the ledger failure (not the
        # generic per-symbol error handler).
        journal = ledger.journal_path().read_text(encoding="utf-8")
        assert '"kind": "recovered_fill_ledger_failed"' in journal
    finally:
        controller.close()
        ledger.close()


def test_kill_between_prepare_and_submit_blocks_send(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A KILL file appearing between prepare_order and mark_submitting must abort the send."""
    _install_fake_features(monkeypatch, price=100.0)
    controller = _new_controller(tmp_path)
    assert _reconcile_clean(controller)
    controller.set_mode(RUN, operator_approved=True)
    assert controller.snapshot()["mode"] == RUN

    fault = FaultBroker()
    permits, tool_broker = _build_broker(tmp_path, fault.adapter())
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    real_dir = tmp_path / "aux"  # not "kill": collides with KILL on case-insensitive filesystems
    real_dir.mkdir()

    def _prepare_wrapper(
        symbol: str, side: str, decision_tick: int, qty: float
    ) -> str:
        # The real prepare_order is monkey-patched below; call it first,
        # then drop the KILL file.  The KILL file must be visible to
        # ``_kill_blocks`` by the time mark_submitting runs.
        cid = original_prepare(symbol, side, decision_tick, qty)
        (tmp_path / "KILL").write_text("halt", encoding="utf-8")
        return cid

    original_prepare = controller.prepare_order
    monkeypatch.setattr(controller, "prepare_order", _prepare_wrapper)

    cfg = LoopConfig(
        broker=fault,
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=_make_buy_decide(),
        controller=controller,
        data_dir=tmp_path,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)
    try:
        loop.tick(1000.0)

        assert fault.submit_calls == [], (
            f"pre_submit kill must block the send; got {fault.submit_calls!r}"
        )

        journal = ledger.journal_path().read_text(encoding="utf-8")
        assert any(
            '"kind": "kill_at_dispatch"' in line and '"stage": "pre_submit"' in line
            for line in journal.splitlines()
        ), "expected a kill_at_dispatch journal entry with stage pre_submit"
    finally:
        controller.close()
        ledger.close()


def test_kill_between_submitting_and_transport_blocks_send(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A KILL file appearing between mark_submitting and the tool-broker invoke must abort the send."""
    _install_fake_features(monkeypatch, price=100.0)
    controller = _new_controller(tmp_path)
    assert _reconcile_clean(controller)
    controller.set_mode(RUN, operator_approved=True)
    assert controller.snapshot()["mode"] == RUN

    fault = FaultBroker()
    permits, tool_broker = _build_broker(tmp_path, fault.adapter())
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    real_dir = tmp_path / "aux"  # not "kill": collides with KILL on case-insensitive filesystems
    real_dir.mkdir()

    def _mark_submitting_wrapper(cid: str) -> None:
        original_mark(cid)
        (tmp_path / "KILL").write_text("halt", encoding="utf-8")

    original_mark = controller.mark_submitting
    monkeypatch.setattr(controller, "mark_submitting", _mark_submitting_wrapper)

    cfg = LoopConfig(
        broker=fault,
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=_make_buy_decide(),
        controller=controller,
        data_dir=tmp_path,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)
    try:
        loop.tick(1000.0)

        assert fault.submit_calls == [], (
            f"pre_transport kill must block the send; got {fault.submit_calls!r}"
        )

        journal = ledger.journal_path().read_text(encoding="utf-8")
        assert any(
            '"kind": "kill_at_dispatch"' in line and '"stage": "pre_transport"' in line
            for line in journal.splitlines()
        ), "expected a kill_at_dispatch journal entry with stage pre_transport"
    finally:
        controller.close()
        ledger.close()


def test_partial_fill_recovered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown POST whose broker record is ``partially_filled`` must record the partial fill."""
    _install_fake_features(monkeypatch, price=100.0)
    controller = _new_controller(tmp_path)
    assert _reconcile_clean(controller)
    controller.set_mode(RUN, operator_approved=True)
    assert controller.snapshot()["mode"] == RUN

    now = 1000.0
    decision_tick = int(now // 60)
    expected_cid = deterministic_order_id("BTC/USD", "buy", decision_tick)

    fault = FaultBroker(submit_raises_after_accept=True)
    permits, tool_broker = _build_broker(tmp_path, fault.adapter())
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    # Override the broker's view of the just-attempted order so the loop's
    # unknown-POST recovery sees a partial fill rather than the default
    # ``filled`` record.  The broker's ``submit`` overwrites ``_lookup``
    # when it runs, so we wrap ``get_order_by_client_id`` instead and
    # return the partial-fill record for our cid.
    partial_record: dict[str, object] = {
        "id": "fault-partial",
        "client_order_id": expected_cid,
        "symbol": "BTC/USD",
        "side": "buy",
        "status": "partially_filled",
        "filled_qty": "0.5",
        "filled_avg_price": "100",
    }
    _original_get = fault.get_order_by_client_id

    def _get_partial(cid: str) -> dict[str, object] | None:
        if cid == expected_cid:
            return dict(partial_record)
        return _original_get(cid)

    monkeypatch.setattr(fault, "get_order_by_client_id", _get_partial)

    cfg = LoopConfig(
        broker=fault,
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=_make_buy_decide(),
        controller=controller,
        data_dir=tmp_path,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)

    try:
        loop.tick(now)

        assert len(fault.submit_calls) == 1, (
            f"expected exactly one submit, got {fault.submit_calls!r}"
        )

        state = controller.snapshot()
        order = state["orders"].get(expected_cid)
        assert order is not None
        # Controller stores filled_qty as the canonical string form.
        assert str(order["filled_qty"]) == "0.5", (
            f"expected controller filled_qty == 0.5, got {order['filled_qty']!r}"
        )

        open_trades = ledger.open_positions()
        assert "BTC/USD" in open_trades, (
            f"expected BTC/USD to be open after recovery, got {list(open_trades)!r}"
        )
        assert float(open_trades["BTC/USD"].qty) == 0.5, (
            f"expected ledger qty 0.5, got {open_trades['BTC/USD']!r}"
        )
    finally:
        controller.close()
        ledger.close()

# ---------------------------------------------------------------------------
# G5 closure tests: I0 (controller_required) + I2 (controller storage fail-stop)
# ---------------------------------------------------------------------------


def test_controller_storage_failure_fail_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Controller ``reconcile`` raising a SQLite error trips fail-stop; second tick raises immediately."""
    _install_fake_features(monkeypatch, price=100.0)
    controller = _new_controller(tmp_path)
    assert _reconcile_clean(controller)
    controller.set_mode(RUN, operator_approved=True)

    fault = FaultBroker()
    permits, tool_broker = _build_broker(tmp_path, fault.adapter())
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    def _explode(*args: object, **kwargs: object) -> bool:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(controller, "reconcile", _explode)

    cfg = LoopConfig(
        broker=fault,
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=_make_buy_decide(),
        controller=controller,
        data_dir=tmp_path,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)

    try:
        with pytest.raises(ControllerFailStop):
            loop.tick(1000.0)
        assert loop._fail_stopped is True, (
            "tick() must remember the fail-stop for subsequent calls"
        )
        assert fault.submit_calls == [], (
            f"no order may be admitted on a fail-stopped tick; got {fault.submit_calls!r}"
        )

        # A second tick raises immediately without reaching the prelude.
        with pytest.raises(ControllerFailStop, match="fail-stopped earlier"):
            loop.tick(1000.0)
        assert fault.submit_calls == [], (
            "second fail-stopped tick must not submit anything either"
        )
    finally:
        controller.close()
        ledger.close()


def test_mark_submitting_sqlite_error_fail_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``sqlite3`` error from ``controller.mark_submitting`` trips fail-stop."""
    _install_fake_features(monkeypatch, price=100.0)
    controller = _new_controller(tmp_path)
    assert _reconcile_clean(controller)
    controller.set_mode(RUN, operator_approved=True)

    fault = FaultBroker()
    permits, tool_broker = _build_broker(tmp_path, fault.adapter())
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    def _explode(cid: str) -> None:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(controller, "mark_submitting", _explode)

    cfg = LoopConfig(
        broker=fault,
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=_make_buy_decide(),
        controller=controller,
        data_dir=tmp_path,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)

    try:
        with pytest.raises(ControllerFailStop):
            loop.tick(1000.0)
        assert loop._fail_stopped is True
        assert fault.submit_calls == [], (
            "mark_submitting raising must not let the broker POST through"
        )

        journal_text = ledger.journal_path().read_text(encoding="utf-8")
        assert any(
            '"kind": "controller_storage_failed"' in line
            and '"stage": "mark_submitting"' in line
            for line in journal_text.splitlines()
        ), "expected a controller_storage_failed journal entry for mark_submitting"

        with pytest.raises(ControllerFailStop):
            loop.tick(1000.0)
    finally:
        controller.close()
        ledger.close()


def test_require_controller_refuses_without_controller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Safety-enabled mode refuses any order effect unless a controller is wired in."""
    _install_fake_features(monkeypatch, price=100.0)
    permits, tool_broker = _build_broker(tmp_path, FaultBroker().adapter())
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    # No controller, no dry-run, require_controller=True -> the buy decision
    # must produce a controller_required journal entry and zero submits.
    cfg = LoopConfig(
        broker=FaultBroker(),
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=_make_buy_decide(),
        controller=None,
        data_dir=tmp_path,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)
    loop._require_controller = True
    loop._dry_run = False

    try:
        loop.tick(1000.0)
        assert cfg.broker.submit_calls == [], (
            f"safety-enabled mode must refuse without a controller; got {cfg.broker.submit_calls!r}"
        )
        journal_text = ledger.journal_path().read_text(encoding="utf-8")
        assert any(
            '"kind": "controller_required"' in line for line in journal_text.splitlines()
        ), "expected a controller_required journal entry"
    finally:
        ledger.close()


def test_partial_then_filled_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two recoveries on the same cid record only the delta; no double count."""
    _install_fake_features(monkeypatch, price=100.0)
    controller = _new_controller(tmp_path)
    assert _reconcile_clean(controller)
    controller.set_mode(RUN, operator_approved=True)

    now = 1000.0
    decision_tick = int(now // 60)
    expected_cid = deterministic_order_id("BTC/USD", "buy", decision_tick)

    # submit_raises_after_accept: records the submit then raises; recovery
    # consults ``get_order_by_client_id`` to find the broker record.
    fault = FaultBroker(submit_raises_after_accept=True)
    permits, tool_broker = _build_broker(tmp_path, fault.adapter())
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    call_count = {"n": 0}

    def _get_by_call(cid: str) -> dict[str, object] | None:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {
                "id": "fault-1",
                "client_order_id": cid,
                "symbol": "BTC/USD",
                "side": "buy",
                "status": "partially_filled",
                "filled_qty": "0.5",
                "filled_avg_price": "100",
            }
        return {
            "id": "fault-2",
            "client_order_id": cid,
            "symbol": "BTC/USD",
            "side": "buy",
            "status": "filled",
            "filled_qty": "1",
            "filled_avg_price": "100",
        }

    monkeypatch.setattr(fault, "get_order_by_client_id", _get_by_call)

    cfg = LoopConfig(
        broker=fault,
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=_make_buy_decide(),
        controller=controller,
        data_dir=tmp_path,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)

    try:
        loop.tick(now)
        assert call_count["n"] == 1
        assert str(controller.snapshot()["orders"][expected_cid]["filled_qty"]) == "0.5"

        # Later the broker reports the order fully filled. By design there is
        # no second submit/recovery (the controller halts entries after a
        # POST); the next prelude reconcile sees broker 1.0 vs controller 0.5
        # and must flag the unresolved exposure with entries blocked (I5).
        full = {
            "id": "fault-1",
            "client_order_id": expected_cid,
            "symbol": "BTC/USD",
            "side": "buy",
            "status": "filled",
            "filled_qty": "1",
            "filled_avg_price": "100",
            "submitted_at": "2026-10-12T00:00:00Z",
        }
        monkeypatch.setattr(fault, "orders_since", lambda *a, **k: [full])
        monkeypatch.setattr(fault, "open_orders", lambda: [])
        monkeypatch.setattr(
            fault, "positions", lambda: [{"symbol": "BTC/USD", "qty": "0.9975"}]
        )
        loop.tick(now + 60)

        snap = controller.snapshot()
        # no silent double count, and the exposure is flagged
        assert str(snap["orders"][expected_cid]["filled_qty"]) == "0.5"
        assert snap["incident"]
        assert snap["mode"] != RUN
        assert snap["reconciled"] is False
        assert call_count["n"] == 1
        assert len(fault.submit_calls) == 1
    finally:
        controller.close()
        ledger.close()


def test_duplicate_recovery_snapshot_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recovering the same partial-fill snapshot twice records no extra qty."""
    _install_fake_features(monkeypatch, price=100.0)
    controller = _new_controller(tmp_path)
    assert _reconcile_clean(controller)
    controller.set_mode(RUN, operator_approved=True)

    now = 1000.0
    decision_tick = int(now // 60)
    expected_cid = deterministic_order_id("BTC/USD", "buy", decision_tick)

    fault = FaultBroker(submit_raises_after_accept=True)
    permits, tool_broker = _build_broker(tmp_path, fault.adapter())
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    def _get_same_partial(cid: str) -> dict[str, object] | None:
        return {
            "id": "fault-1",
            "client_order_id": cid,
            "symbol": "BTC/USD",
            "side": "buy",
            "status": "partially_filled",
            "filled_qty": "0.5",
            "filled_avg_price": "100",
        }

    monkeypatch.setattr(fault, "get_order_by_client_id", _get_same_partial)

    cfg = LoopConfig(
        broker=fault,
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=_make_buy_decide(),
        controller=controller,
        data_dir=tmp_path,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)

    try:
        loop.tick(now)
        loop.tick(now)

        state = controller.snapshot()
        order = state["orders"].get(expected_cid)
        assert order is not None
        assert str(order["filled_qty"]) == "0.5", (
            f"expected controller filled_qty to remain 0.5 after duplicate recovery; "
            f"got {order['filled_qty']!r}"
        )
        assert order["status"] == "PARTIAL", (
            f"expected status PARTIAL after duplicate recovery; got {order['status']!r}"
        )
    finally:
        controller.close()
        ledger.close()


def test_kill_dir_vanishing_after_prelude_blocks_send(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The kill dir disappearing between prelude and dispatch must block the BUY (I1)."""
    import shutil

    _install_fake_features(monkeypatch, price=100.0)
    controller = _new_controller(tmp_path)
    assert _reconcile_clean(controller)
    controller.set_mode(RUN, operator_approved=True)

    fault = FaultBroker()
    permits, tool_broker = _build_broker(tmp_path, fault.adapter())
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")
    kill_dir = tmp_path / "operator"
    kill_dir.mkdir()

    original_prepare = controller.prepare_order

    def _prepare_then_remove_dir(symbol, side, decision_tick, qty):
        cid = original_prepare(symbol, side, decision_tick, qty)
        shutil.rmtree(kill_dir)
        return cid

    monkeypatch.setattr(controller, "prepare_order", _prepare_then_remove_dir)
    cfg = LoopConfig(
        broker=fault,
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=_make_buy_decide(),
        controller=controller,
        data_dir=kill_dir,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)
    try:
        loop.tick(1000.0)
        assert fault.submit_calls == []
        journal = ledger.journal_path().read_text(encoding="utf-8")
        assert any(
            '"kind": "kill_at_dispatch"' in line and '"stage": "pre_submit"' in line
            for line in journal.splitlines()
        )
    finally:
        controller.close()
        ledger.close()


def test_prepare_order_sqlite_error_fail_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sqlite3 error from prepare_order fail-stops the trader, sticky across ticks (I2)."""
    import sqlite3

    _install_fake_features(monkeypatch, price=100.0)
    controller = _new_controller(tmp_path)
    assert _reconcile_clean(controller)
    controller.set_mode(RUN, operator_approved=True)

    fault = FaultBroker()
    permits, tool_broker = _build_broker(tmp_path, fault.adapter())
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    def _broken_prepare(*args, **kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(controller, "prepare_order", _broken_prepare)
    cfg = LoopConfig(
        broker=fault,
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=_make_buy_decide(),
        controller=controller,
        data_dir=tmp_path,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)
    try:
        with pytest.raises(ControllerFailStop):
            loop.tick(1000.0)
        with pytest.raises(ControllerFailStop):
            loop.tick(1060.0)
        assert fault.submit_calls == []
    finally:
        controller.close()
        ledger.close()


def test_vanished_kill_dir_halt_is_sticky_after_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dispatch refusal from a vanished kill dir persists a halt; restoring it allows no BUY (I2)."""
    import shutil

    _install_fake_features(monkeypatch, price=100.0)
    controller = _new_controller(tmp_path)
    assert _reconcile_clean(controller)
    controller.set_mode(RUN, operator_approved=True)
    fault = FaultBroker()
    permits, tool_broker = _build_broker(tmp_path, fault.adapter())
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")
    kill_dir = tmp_path / "operator"
    kill_dir.mkdir()
    original_prepare = controller.prepare_order
    state = {"removed": False}

    def _prepare_then_remove_dir(symbol, side, decision_tick, qty):
        cid = original_prepare(symbol, side, decision_tick, qty)
        if not state["removed"]:
            shutil.rmtree(kill_dir)
            state["removed"] = True
        return cid

    monkeypatch.setattr(controller, "prepare_order", _prepare_then_remove_dir)
    cfg = LoopConfig(
        broker=fault,
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=_make_buy_decide(),
        controller=controller,
        data_dir=kill_dir,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)
    try:
        loop.tick(1000.0)
        snap = controller.snapshot()
        assert snap["mode"] != RUN and snap["incident"]
        kill_dir.mkdir()  # the directory comes back
        loop.tick(1060.0)
        assert fault.submit_calls == []
        assert controller.snapshot()["incident"]
    finally:
        controller.close()
        ledger.close()


def test_controller_snapshot_sqlite_error_fail_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sqlite3 error from controller.snapshot in _execute fail-stops, sticky (I2)."""
    import sqlite3

    _install_fake_features(monkeypatch, price=100.0)
    controller = _new_controller(tmp_path)
    assert _reconcile_clean(controller)
    controller.set_mode(RUN, operator_approved=True)
    fault = FaultBroker()
    permits, tool_broker = _build_broker(tmp_path, fault.adapter())
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")
    cfg = LoopConfig(
        broker=fault,
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=_make_buy_decide(),
        controller=controller,
        data_dir=tmp_path,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)
    real_snapshot = controller.snapshot
    calls = {"n": 0}

    def flaky_snapshot():
        calls["n"] += 1
        # let the prelude's reads pass; fail the dispatch-time read
        if calls["n"] > 2:
            raise sqlite3.OperationalError("disk I/O error")
        return real_snapshot()

    monkeypatch.setattr(controller, "snapshot", flaky_snapshot)
    try:
        with pytest.raises(ControllerFailStop):
            loop.tick(1000.0)
        with pytest.raises(ControllerFailStop):
            loop.tick(1060.0)
        assert fault.submit_calls == []
    finally:
        controller.close()
        ledger.close()


def _shadow_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, autorun: bool, run_first: bool):
    _install_fake_features(monkeypatch, price=100.0)
    controller = _new_controller(tmp_path)
    if run_first:
        assert _reconcile_clean(controller)
        controller.set_mode(RUN, operator_approved=True)
    fault = FaultBroker()
    permits, tool_broker = _build_broker(tmp_path, fault.adapter())
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")
    loop = TraderLoop(
        market_fn=_noop_market, broker=fault, permits=permits, tool_broker=tool_broker,
        ledger=ledger, limits=Limits(), heartbeat_path=tmp_path / "hb.json",
        kill_path=tmp_path / "KILL", decide_fn=_make_buy_decide(), review_fn=_approve,
        symbols=("BTC/USD",), controller=controller, data_dir=tmp_path,
        history_start_iso="2026-10-05T00:00:00+00:00", require_controller=True,
        shadow=True, shadow_autorun=autorun,
    )
    return loop, controller, fault, ledger


def test_shadow_never_sends_orders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loop, controller, fault, ledger = _shadow_loop(tmp_path, monkeypatch, autorun=False, run_first=True)
    try:
        loop.tick(1000.0)
        assert fault.submit_calls == []
        journal = ledger.journal_path().read_text(encoding="utf-8")
        assert '"kind": "shadow_order"' in journal and '"controller_would_allow": true' in journal
        assert controller.snapshot()["orders"] == {}  # nothing prepared in the controller
    finally:
        controller.close()
        ledger.close()


def test_shadow_autorun_enters_run_after_clean_reconcile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loop, controller, fault, ledger = _shadow_loop(tmp_path, monkeypatch, autorun=True, run_first=False)
    try:
        assert controller.snapshot()["mode"] != RUN
        loop.tick(1000.0)
        assert controller.snapshot()["mode"] == RUN
        assert fault.submit_calls == []
    finally:
        controller.close()
        ledger.close()


def test_autorun_ignored_without_shadow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_features(monkeypatch, price=100.0)
    controller = _new_controller(tmp_path)
    fault = FaultBroker()
    permits, tool_broker = _build_broker(tmp_path, fault.adapter())
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")
    loop = TraderLoop(
        market_fn=_noop_market, broker=fault, permits=permits, tool_broker=tool_broker,
        ledger=ledger, limits=Limits(), heartbeat_path=tmp_path / "hb.json",
        kill_path=tmp_path / "KILL", decide_fn=_make_buy_decide(), review_fn=_approve,
        symbols=("BTC/USD",), controller=controller, data_dir=tmp_path,
        history_start_iso="2026-10-05T00:00:00+00:00", shadow=False, shadow_autorun=True,
    )
    try:
        loop.tick(1000.0)
        assert controller.snapshot()["mode"] != RUN  # never self-approves outside shadow
    finally:
        controller.close()
        ledger.close()


def test_paper_autorun_reenters_run_but_never_over_hard_incident(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_features(monkeypatch, price=100.0)
    controller = _new_controller(tmp_path)
    fault = FaultBroker()
    permits, tool_broker = _build_broker(tmp_path, fault.adapter())
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")
    loop = TraderLoop(
        market_fn=_noop_market, broker=fault, permits=permits, tool_broker=tool_broker,
        ledger=ledger, limits=Limits(), heartbeat_path=tmp_path / "hb.json",
        kill_path=tmp_path / "KILL", decide_fn=lambda *a, **k: Decision("BTC/USD", "hold", "x"),
        review_fn=_approve, symbols=("BTC/USD",), controller=controller, data_dir=tmp_path,
        history_start_iso="2026-10-05T00:00:00+00:00", require_controller=True,
        paper_autorun=True, buy_fee_in_asset=0.0,
    )
    try:
        loop.tick(1000.0)
        assert controller.snapshot()["mode"] == RUN
        # a hard incident must keep it halted
        controller.reconcile({}, [], account_active=False, complete_snapshot=False)
        loop._data_dir = tmp_path / "missing"  # prelude fails closed too
        loop.tick(1060.0)
        assert controller.snapshot()["mode"] != RUN
    finally:
        controller.close()
        ledger.close()

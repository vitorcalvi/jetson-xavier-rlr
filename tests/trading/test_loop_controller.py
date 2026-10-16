"""Tests for the OrderController integration in :class:`TraderLoop`.

The OrderController, the kill-modes mapper, and the broker reconciliation are
opt-in.  When ``controller=None`` the loop is bit-for-bit compatible with the
pre-controller behaviour; when a controller is injected the loop performs a
durable prelude on every tick and gates every entry / exit through the
controller's mode.  The tests below pin the opt-in surface called out by the
brief:

* ``test_controller_none_keeps_old_behaviour`` -- ``controller=None`` does
  not change the loop's behaviour.
* ``test_entry_blocked_when_controller_halted`` -- a fresh controller starts
  in ``HALT_ENTRIES`` and refuses to submit any entry.
* ``test_entry_allowed_after_operator_run`` -- after a clean reconcile and
  an operator-approved ``RUN`` transition the entry is submitted and the
  adapter receives the controller's client order id.
* ``test_reconcile_failure_blocks_entry`` -- a broker ``orders_since``
  exception trips the controller and the entry is blocked.
* ``test_unknown_submit_is_looked_up_not_retried`` -- an exception from
  ``broker.submit`` is recovered via ``get_order_by_client_id``; exactly
  one submit attempt is made.
* ``test_exit_blocked_in_emergency`` -- an exit in ``EMERGENCY_STOP`` is
  journaled and not submitted.
* ``test_kill_file_escalates_controller`` -- an operator ``KILL`` file in
  ``data_dir`` escalates the controller from ``RUN`` to ``HALT_ENTRIES``.
* ``test_positions_error_fails_closed`` -- a ``positions()`` error trips
  the controller even when the operator already approved ``RUN``.
* ``test_missing_history_start_is_incomplete`` -- no ``history_start_iso``
  means the reconcile is incomplete and the controller refuses to enter.
* ``test_sell_fill_fee_in_usd`` -- a sell fill records ``fee_in_asset=0``
  because Alpaca takes the sell fee in USD.
* ``test_prepare_order_refusal_does_not_crash`` -- a ``prepare_order``
  refusal is journaled and does not crash the tick.

The fixtures below mirror the permit / tool-broker wiring from
``tests/trading/test_loop.py``; the only new addition is a ``FakeBroker``
that implements the four methods the controller prelude talks to
(``positions``, ``orders_since``, ``get_order_by_client_id``) plus the
``submit`` the production adapter calls.  The ``LoopConfig`` dataclass is
a tiny aggregator so every test can pin the same TraderLoop wiring without
copy-pasting the same eight keyword arguments.
"""

from __future__ import annotations

import json
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
from reflective_loop_runtime.trading.broker import client_order_id
from reflective_loop_runtime.trading.ledger import Ledger
from reflective_loop_runtime.trading.loop import TraderLoop, trading_manifest
from reflective_loop_runtime.trading.model import Decision, Limits
from reflective_loop_runtime.trading.orders import (
    EMERGENCY_STOP,
    HALT_ENTRIES,
    RUN,
    OrderController,
    deterministic_order_id,
)

DEV_SECRET = "dev-only-shared-secret"
ISSUER_KEY = b"governor-owned-hmac-key-0123456789"


# ---------------------------------------------------------------------------
# Permit / tool-broker wiring (mirrors tests/trading/test_loop.py)
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
    fake_order: Callable[[Mapping[str, object]], dict[str, object]] | None,
) -> tuple[PermitService, ToolBroker]:
    """Build a real ``PermitService`` + ``ToolBroker`` for the trader manifest."""
    service = _service(tmp_path, trading_manifest())
    adapters: dict[
        tuple[str, str], Callable[[Mapping[str, object]], dict[str, object]]
    ] = {}
    if fake_order is not None:
        adapters[("alpaca_paper", "order")] = fake_order
    broker = ToolBroker(
        service,
        audit=AuditLog(":memory:"),
        workspace=tmp_path / "ws",
        manifest=trading_manifest(),
        adapters=adapters,
    )
    return service, broker


# ---------------------------------------------------------------------------
# Feature stand-in (also from tests/trading/test_loop.py)
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


# ---------------------------------------------------------------------------
# Fake broker (positions / orders_since / get_order_by_client_id / submit)
# ---------------------------------------------------------------------------


class FakeBroker:
    """A combined broker + loop-broker stand-in.

    In production the same ``PaperBroker`` instance is passed to both
    ``TraderLoop`` (as ``broker=...``) and to ``make_order_adapter`` (so the
    adapter can call ``broker.submit``).  These tests follow the same
    convention with a single object that fulfils both roles, which makes
    ``test_unknown_submit_is_looked_up_not_retried`` a single-broker test.
    """

    def __init__(
        self,
        *,
        positions: list[dict[str, object]] | None = None,
        orders: list[dict[str, object]] | None = None,
        open_orders: list[dict[str, object]] | None = None,
        lookup: dict[str, dict[str, object]] | None = None,
        submit_raises: bool = False,
        orders_since_raises: bool = False,
        positions_raises: bool = False,
        account_active: bool = True,
        account_status_raises: bool = False,
    ) -> None:
        self._positions = [dict(p) for p in (positions or [])]
        self._orders = [dict(o) for o in (orders or [])]
        self._open_orders = [dict(o) for o in (open_orders or [])]
        self._lookup = dict(lookup or {})
        self._submit_raises = bool(submit_raises)
        self._orders_since_raises = bool(orders_since_raises)
        self._positions_raises = bool(positions_raises)
        self._account_active = bool(account_active)
        self._account_status_raises = bool(account_status_raises)
        self.submit_calls: list[dict[str, object]] = []
        self.positions_calls = 0
        self.orders_since_calls = 0
        self.account_status_calls = 0
        self.open_orders_calls = 0

    def positions(self) -> list[dict[str, object]]:
        self.positions_calls += 1
        if self._positions_raises:
            raise RuntimeError("positions boom")
        return [dict(p) for p in self._positions]

    def open_orders(self) -> list[dict[str, object]]:
        """Return the currently-open orders the broker has on the books."""
        self.open_orders_calls += 1
        return [dict(o) for o in self._open_orders]

    def account_status(self) -> dict[str, object]:
        """Return the Alpaca-style account status snapshot.

        Default is active; tests opt out via ``account_active=False`` or
        request failure with ``account_status_raises=True``.
        """
        self.account_status_calls += 1
        if self._account_status_raises:
            raise RuntimeError("account status boom")
        status = "ACTIVE" if self._account_active else "INACTIVE"
        return {
            "active": bool(self._account_active),
            "status": status,
            "crypto_status": status,
        }

    def orders_since(
        self,
        after_iso: str,
        *,
        page_limit: int = 500,
        max_pages: int = 50,
    ) -> list[dict[str, object]]:
        self.orders_since_calls += 1
        if self._orders_since_raises:
            raise RuntimeError("orders_since boom")
        return [dict(o) for o in self._orders]

    def get_order_by_client_id(self, cid: str) -> dict[str, object] | None:
        result = self._lookup.get(cid)
        return dict(result) if result is not None else None

    def submit(
        self,
        symbol: str,
        side: str,
        *,
        client_order_id: str,
        notional: float | None = None,
        qty: float | None = None,
    ) -> dict[str, object]:
        self.submit_calls.append(
            {
                "symbol": symbol,
                "side": side,
                "client_order_id": client_order_id,
                "notional": notional,
                "qty": qty,
            }
        )
        if self._submit_raises:
            raise RuntimeError("submit boom")
        # Track open orders so open_orders() based reconciles see them.
        open_record: dict[str, object] = {
            "id": f"broker-order-{len(self.submit_calls)}",
            "client_order_id": client_order_id,
            "symbol": symbol,
            "side": side,
            "status": "accepted",
            "filled_qty": "0",
        }
        self._open_orders.append(open_record)
        self._lookup[client_order_id] = dict(open_record)
        return {
            "id": open_record["id"],
            "client_order_id": client_order_id,
            "status": "accepted",
        }


def _make_submit_adapter(
    broker: FakeBroker, calls: list[dict[str, object]]
) -> Callable[[Mapping[str, object]], dict[str, object]]:
    """An adapter that forwards to ``broker.submit`` and records the call."""

    def fake_order(args: Mapping[str, object]) -> dict[str, object]:
        calls.append(dict(args))
        return broker.submit(
            str(args["symbol"]),
            str(args["side"]),
            client_order_id=str(args["client_order_id"]),
            notional=args.get("notional"),
            qty=args.get("qty"),
        )

    return fake_order


def _make_filled_order_adapter(
    calls: list[dict[str, object]],
    *,
    fill_status: str = "filled",
    filled_qty: float = 0.01,
    price: float = 100.0,
    price_box: dict[str, float] | None = None,
) -> Callable[[Mapping[str, object]], dict[str, object]]:
    """An adapter that records the call and returns a fake broker fill.

    When ``price_box`` is supplied, the fill price is read from
    ``price_box['price']`` on every call so callers can simulate a price
    walking between ticks without rebuilding the adapter.
    """

    def fake_order(args: Mapping[str, object]) -> dict[str, object]:
        calls.append(dict(args))
        fill_price = (
            float(price_box["price"]) if price_box is not None else float(price)
        )
        return {
            "order_id": "fake-order-1",
            "status": fill_status,
            "filled_qty": float(filled_qty),
            "filled_avg_price": fill_price,
            "client_order_id": str(args["client_order_id"]),
        }

    return fake_order


def _noop_market(
    symbols: tuple[str, ...], now: float | None = None, **kw: object
) -> dict[str, object]:
    """Market stand-in: returns a sentinel object per symbol so the loop proceeds."""
    return {sym: object() for sym in symbols}


def _approve(
    _decision: object,
    _features: object,
    _state: object,
    _limits: object,
    *,
    now: float,
) -> tuple[bool, None]:
    return (True, None)


# ---------------------------------------------------------------------------
# Wiring helper
# ---------------------------------------------------------------------------


@dataclass
class LoopConfig:
    """A bundle of TraderLoop kwargs; tests pass one or two overrides."""

    broker: Any
    tool_broker: Any
    permits: Any
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_controller_none_keeps_old_behaviour(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``controller=None`` keeps the loop exactly the way the brief asks for.

    The same entry + exit flow exercised by ``test_loop_entry_then_exit_records_r``
    must still close a trade with ``r_multiple == 1.0`` when the controller
    keyword is left at its default (None).
    """
    price_box = _install_fake_features(monkeypatch, price=100.0)
    calls: list[dict[str, object]] = []
    adapter = _make_filled_order_adapter(calls, price_box=price_box)
    permits, tool_broker = _build_broker(tmp_path, adapter)
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")
    fake_broker = FakeBroker()

    def decide_fn(
        features: object, position: object, limits: object, *, now: float
    ) -> Decision:
        if position is None:
            return Decision(
                symbol="BTC/USD",
                action="enter_long",
                reason="t",
                stop=90.0,
                target=110.0,
                notional=100.0,
            )
        return Decision("BTC/USD", "exit", "target")

    cfg = LoopConfig(
        broker=fake_broker,
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=decide_fn,
    )
    loop = _build_loop(cfg)
    # ``controller`` defaults to ``None``; the loop must behave like the
    # pre-controller version.

    price_box["price"] = 100.0
    loop.tick(1000.0)
    assert len(calls) == 1
    # The cid is the old client_order_id() helper (no controller wired).
    assert calls[0]["client_order_id"] == client_order_id("BTC/USD", "buy", 16)

    price_box["price"] = 110.0
    loop.tick(1001.0)
    assert len(calls) == 2

    closed = ledger.closed_trades()
    assert len(closed) == 1
    assert closed[0]["r_multiple"] == pytest.approx(1.0)
    ledger.close()


def test_entry_blocked_when_controller_halted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh controller starts ``HALT_ENTRIES`` -> no submit, journal veto."""
    _install_fake_features(monkeypatch, price=100.0)
    calls: list[dict[str, object]] = []
    adapter = _make_filled_order_adapter(calls)
    permits, tool_broker = _build_broker(tmp_path, adapter)
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")
    controller = OrderController(tmp_path / "orders.db")

    def decide_fn(
        features: object, position: object, limits: object, *, now: float
    ) -> Decision:
        return Decision(
            symbol="BTC/USD",
            action="enter_long",
            reason="t",
            stop=90.0,
            target=110.0,
            notional=100.0,
        )

    cfg = LoopConfig(
        broker=FakeBroker(),
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=decide_fn,
        controller=controller,
        data_dir=tmp_path,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)

    try:
        loop.tick(1000.0)

        assert calls == [], f"expected no submit while HALT_ENTRIES, got {calls!r}"
        assert controller.snapshot()["mode"] == HALT_ENTRIES

        journal_text = ledger.journal_path().read_text(encoding="utf-8")
        assert any(
            '"kind": "controller_veto"' in line
            and "controller_mode:HALT_ENTRIES" in line
            for line in journal_text.splitlines()
        ), "expected a controller_mode veto in the journal"
    finally:
        controller.close()
        ledger.close()


def test_entry_allowed_after_operator_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reconcile + operator RUN -> submit happens with the controller's cid."""
    _install_fake_features(monkeypatch, price=100.0)
    calls: list[dict[str, object]] = []
    adapter = _make_filled_order_adapter(calls)
    permits, tool_broker = _build_broker(tmp_path, adapter)
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")
    controller = OrderController(tmp_path / "orders.db")
    assert controller.reconcile({}, [], account_active=True, complete_snapshot=True)
    controller.set_mode(RUN, operator_approved=True)
    assert controller.snapshot()["mode"] == RUN

    def decide_fn(
        features: object, position: object, limits: object, *, now: float
    ) -> Decision:
        return Decision(
            symbol="BTC/USD",
            action="enter_long",
            reason="t",
            stop=90.0,
            target=110.0,
            notional=100.0,
        )

    cfg = LoopConfig(
        broker=FakeBroker(),
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=decide_fn,
        controller=controller,
        data_dir=tmp_path,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)

    now = 1000.0
    try:
        loop.tick(now)

        assert len(calls) == 1, f"expected exactly one submit, got {calls!r}"
        expected_cid = deterministic_order_id("BTC/USD", "buy", int(now // 60))
        assert calls[0]["client_order_id"] == expected_cid, (
            f"submit must use the controller's cid; got {calls[0]['client_order_id']!r}"
        )
        # The controller is now in ``HALT_ENTRIES`` again because the fill
        # tripped it; the test only cares that the submit happened.
        assert expected_cid in controller.snapshot()["orders"]
    finally:
        controller.close()
        ledger.close()


def test_reconcile_failure_blocks_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``orders_since`` raises -> the controller trips, no submit happens."""
    _install_fake_features(monkeypatch, price=100.0)
    calls: list[dict[str, object]] = []
    adapter = _make_filled_order_adapter(calls)
    permits, tool_broker = _build_broker(tmp_path, adapter)
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")
    controller = OrderController(tmp_path / "orders.db")
    fake_broker = FakeBroker(orders_since_raises=True)

    def decide_fn(
        features: object, position: object, limits: object, *, now: float
    ) -> Decision:
        return Decision(
            symbol="BTC/USD",
            action="enter_long",
            reason="t",
            stop=90.0,
            target=110.0,
            notional=100.0,
        )

    cfg = LoopConfig(
        broker=fake_broker,
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=decide_fn,
        controller=controller,
        data_dir=tmp_path,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)

    try:
        loop.tick(1000.0)

        assert calls == [], f"expected no submit, got {calls!r}"
        assert controller.snapshot()["mode"] == HALT_ENTRIES

        journal_text = ledger.journal_path().read_text(encoding="utf-8")
        assert any(
            '"kind": "reconcile_failed"' in line for line in journal_text.splitlines()
        ), "expected a reconcile_failed journal entry"
    finally:
        controller.close()
        ledger.close()


def test_unknown_submit_is_looked_up_not_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A submit exception is recovered via the broker; exactly one submit."""
    _install_fake_features(monkeypatch, price=100.0)
    calls: list[dict[str, object]] = []

    controller = OrderController(tmp_path / "orders.db")
    assert controller.reconcile({}, [], account_active=True, complete_snapshot=True)
    controller.set_mode(RUN, operator_approved=True)

    now = 1000.0
    decision_tick = int(now // 60)
    expected_cid = deterministic_order_id("BTC/USD", "buy", decision_tick)

    fake_broker = FakeBroker(
        submit_raises=True,
        lookup={
            expected_cid: {
                "id": "broker-order-1",
                "client_order_id": expected_cid,
                "status": "filled",
                "filled_qty": 0.01,
                "filled_avg_price": 100.0,
            }
        },
    )
    adapter = _make_submit_adapter(fake_broker, calls)
    permits, tool_broker = _build_broker(tmp_path, adapter)
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    def decide_fn(
        features: object, position: object, limits: object, *, now: float
    ) -> Decision:
        return Decision(
            symbol="BTC/USD",
            action="enter_long",
            reason="t",
            stop=90.0,
            target=110.0,
            notional=100.0,
        )

    cfg = LoopConfig(
        broker=fake_broker,
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=decide_fn,
        controller=controller,
        data_dir=tmp_path,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)

    try:
        loop.tick(now)

        # Exactly one submit attempt (the controller does not retry).
        assert len(fake_broker.submit_calls) == 1, (
            f"expected exactly one submit, got {fake_broker.submit_calls!r}"
        )
        assert fake_broker.submit_calls[0]["client_order_id"] == expected_cid

        # The controller was recovered: the order is recorded (FILLED or
        # PARTIAL depending on whether the broker's reported qty met the
        # prepared quantity), and the position is now reflected in the
        # controller's state.
        state = controller.snapshot()
        assert expected_cid in state["orders"]
        order_status = state["orders"][expected_cid]["status"]
        assert order_status in {"FILLED", "PARTIAL", "ACKED"}, (
            f"unexpected order status after recovery: {order_status!r}"
        )
        # The controller has come out of uncertainty: the fill was recorded
        # and the position reflects the broker's view.
        assert "BTC/USD" in state["positions"]
    finally:
        controller.close()
        ledger.close()


def test_exit_blocked_in_emergency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exit while in ``EMERGENCY_STOP`` is journaled and not submitted."""
    _install_fake_features(monkeypatch, price=110.0)
    calls: list[dict[str, object]] = []
    adapter = _make_filled_order_adapter(calls)
    permits, tool_broker = _build_broker(tmp_path, adapter)
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    # Seed an open position so the exit decision has a fallback qty.
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

    controller = OrderController(tmp_path / "orders.db")
    assert controller.reconcile({}, [], account_active=True, complete_snapshot=True)
    controller.set_mode(EMERGENCY_STOP)
    assert controller.snapshot()["mode"] == EMERGENCY_STOP

    fake_broker = FakeBroker(positions=[{"symbol": "BTCUSD", "qty": 1.0}])

    def decide_fn(
        features: object, position: object, limits: object, *, now: float
    ) -> Decision:
        return Decision("BTC/USD", "exit", "target")

    cfg = LoopConfig(
        broker=fake_broker,
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=decide_fn,
        controller=controller,
        data_dir=tmp_path,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)

    try:
        loop.tick(1000.0)

        assert calls == [], (
            f"expected no submit in EMERGENCY_STOP, got {calls!r}"
        )
        assert controller.snapshot()["mode"] == EMERGENCY_STOP

        journal_text = ledger.journal_path().read_text(encoding="utf-8")
        assert any(
            '"kind": "emergency_no_auto_send"' in line
            for line in journal_text.splitlines()
        ), "expected an emergency_no_auto_send journal entry"
    finally:
        controller.close()
        ledger.close()


def test_kill_file_escalates_controller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``KILL`` file in ``data_dir`` escalates the controller to ``HALT_ENTRIES``."""
    _install_fake_features(monkeypatch, price=100.0)
    calls: list[dict[str, object]] = []
    adapter = _make_filled_order_adapter(calls)
    permits, tool_broker = _build_broker(tmp_path, adapter)
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    controller = OrderController(tmp_path / "orders.db")
    assert controller.reconcile({}, [], account_active=True, complete_snapshot=True)
    controller.set_mode(RUN, operator_approved=True)
    assert controller.snapshot()["mode"] == RUN

    # Drop a KILL file in the data directory; the loop's kill_path and the
    # controller's data_dir both point at the same directory so a single
    # file is enough.
    (tmp_path / "KILL").write_text("halt", encoding="utf-8")

    fake_broker = FakeBroker()

    def decide_fn(
        features: object, position: object, limits: object, *, now: float
    ) -> Decision:
        return Decision("BTC/USD", "hold", "no-trade")

    cfg = LoopConfig(
        broker=fake_broker,
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=decide_fn,
        controller=controller,
        data_dir=tmp_path,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)

    try:
        loop.tick(1000.0)

        # The controller has been escalated from RUN to HALT_ENTRIES by the
        # killmodes.apply call inside the prelude.
        assert controller.snapshot()["mode"] == HALT_ENTRIES, (
            f"expected HALT_ENTRIES after kill-file escalation, "
            f"got {controller.snapshot()!r}"
        )
    finally:
        controller.close()
        ledger.close()


def test_positions_error_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``positions()`` raising trips the controller; no entry even in RUN."""
    _install_fake_features(monkeypatch, price=100.0)
    calls: list[dict[str, object]] = []
    adapter = _make_filled_order_adapter(calls)
    permits, tool_broker = _build_broker(tmp_path, adapter)
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    controller = OrderController(tmp_path / "orders.db")
    assert controller.reconcile({}, [], account_active=True, complete_snapshot=True)
    controller.set_mode(RUN, operator_approved=True)
    assert controller.snapshot()["mode"] == RUN

    fake_broker = FakeBroker(positions_raises=True)

    def decide_fn(
        features: object, position: object, limits: object, *, now: float
    ) -> Decision:
        return Decision(
            symbol="BTC/USD",
            action="enter_long",
            reason="t",
            stop=90.0,
            target=110.0,
            notional=100.0,
        )

    cfg = LoopConfig(
        broker=fake_broker,
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=decide_fn,
        controller=controller,
        data_dir=tmp_path,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)

    try:
        loop.tick(1000.0)

        assert calls == [], (
            f"expected no submit when positions() raises, got {calls!r}"
        )
        # The controller was tripped by the fail-closed reconcile.
        snap = controller.snapshot()
        assert snap["mode"] == HALT_ENTRIES, (
            f"expected HALT_ENTRIES after positions() error, got {snap!r}"
        )
        assert not snap["reconciled"], "controller must not be reconciled"

        journal_text = ledger.journal_path().read_text(encoding="utf-8")
        assert any(
            '"kind": "reconcile_failed"' in line
            for line in journal_text.splitlines()
        ), "expected a reconcile_failed journal entry when positions() raises"
    finally:
        controller.close()
        ledger.close()


def test_missing_history_start_is_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``history_start_iso=None`` -> incomplete snapshot -> controller trips."""
    _install_fake_features(monkeypatch, price=100.0)
    calls: list[dict[str, object]] = []
    adapter = _make_filled_order_adapter(calls)
    permits, tool_broker = _build_broker(tmp_path, adapter)
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    controller = OrderController(tmp_path / "orders.db")
    assert controller.reconcile({}, [], account_active=True, complete_snapshot=True)
    controller.set_mode(RUN, operator_approved=True)
    assert controller.snapshot()["mode"] == RUN

    fake_broker = FakeBroker()

    def decide_fn(
        features: object, position: object, limits: object, *, now: float
    ) -> Decision:
        return Decision(
            symbol="BTC/USD",
            action="enter_long",
            reason="t",
            stop=90.0,
            target=110.0,
            notional=100.0,
        )

    cfg = LoopConfig(
        broker=fake_broker,
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=decide_fn,
        controller=controller,
        data_dir=tmp_path,
        history_start_iso=None,  # no broker order history available
    )
    loop = _build_loop(cfg)

    try:
        loop.tick(1000.0)

        assert calls == [], (
            f"expected no submit with no history, got {calls!r}"
        )
        # No history means incomplete truth; the controller must not be in
        # a clean state and the entry must be refused.
        snap = controller.snapshot()
        assert snap["mode"] != RUN, (
            f"controller must not be RUN with incomplete history, got {snap!r}"
        )
        assert not snap["reconciled"], (
            "controller must not be reconciled without a complete history"
        )
    finally:
        controller.close()
        ledger.close()


def test_sell_fill_fee_in_usd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sell fill must call ``record_fill`` with ``fee_in_asset=0`` (USD fee)."""
    _install_fake_features(monkeypatch, price=110.0)
    calls: list[dict[str, object]] = []
    adapter = _make_filled_order_adapter(calls, filled_qty=1.0)
    permits, tool_broker = _build_broker(tmp_path, adapter)
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    # Seed an open position so the exit decision has a fallback qty.
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

    controller = OrderController(tmp_path / "orders.db")
    # Take the controller out of its initial ``HALT_ENTRIES`` so the seed
    # ``prepare_order`` below is admitted.
    assert controller.reconcile({}, [], account_active=True, complete_snapshot=True)
    controller.set_mode(RUN, operator_approved=True)
    # Seed the controller with a buy that filled so the controller believes
    # we own 1 BTC. The seed fill is fee-free so the controller's internal
    # position exactly matches the broker's reported position.
    seed_cid = controller.prepare_order("BTC/USD", "buy", 1, 1.0)  # integer tick: the prelude proves history coverage from it
    controller.mark_submitting(seed_cid)
    controller.acknowledge_order(seed_cid, broker_status="accepted")
    controller.record_fill(seed_cid, "seed-exec-id", 1.0, fee_in_asset="0")
    assert controller.reconcile(
        {"BTC/USD": 1.0},
        [
            {
                "client_order_id": seed_cid,
                "symbol": "BTC/USD",
                "side": "buy",
                "filled_qty": "1.0",
                "status": "filled",
            }
        ],
        account_active=True,
        complete_snapshot=True,
    )
    controller.set_mode(RUN, operator_approved=True)

    fake_broker = FakeBroker(positions=[{"symbol": "BTCUSD", "qty": 1.0}])

    record_fill_calls: list[dict[str, object]] = []
    real_record_fill = controller.record_fill

    def spy_record_fill(*args: object, **kwargs: object) -> object:
        record_fill_calls.append({"args": args, "kwargs": dict(kwargs)})
        return real_record_fill(*args, **kwargs)

    monkeypatch.setattr(controller, "record_fill", spy_record_fill)

    def decide_fn(
        features: object, position: object, limits: object, *, now: float
    ) -> Decision:
        return Decision("BTC/USD", "exit", "target")

    cfg = LoopConfig(
        broker=fake_broker,
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=decide_fn,
        controller=controller,
        data_dir=tmp_path,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)

    try:
        loop.tick(1000.0)

        # The adapter saw the sell, so record_fill must have been invoked.
        assert len(record_fill_calls) == 1, (
            f"expected exactly one record_fill call, got {record_fill_calls!r}"
        )
        kwargs = record_fill_calls[0]["kwargs"]
        # Accept any of 0, 0.0, or "0"/"0.0" string variants the controller
        # might serialise.
        fee = kwargs.get("fee_in_asset", "MISSING")
        assert fee in (0, 0.0, "0", "0.0", "0.0000", "0E-10"), (
            f"sell record_fill must use fee_in_asset=0, got {fee!r}"
        )
        # The controller must not have tripped on an unexpected sell fee.
        assert controller.snapshot()["incident"] != "unexpected_asset_fee_on_sell", (
            "controller tripped on an unexpected sell fee: fee was not 0"
        )
    finally:
        controller.close()
        ledger.close()


def test_prepare_order_refusal_does_not_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``prepare_order`` refusal is journaled and the tick survives."""
    _install_fake_features(monkeypatch, price=100.0)
    calls: list[dict[str, object]] = []
    adapter = _make_filled_order_adapter(calls)
    permits, tool_broker = _build_broker(tmp_path, adapter)
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    controller = OrderController(tmp_path / "orders.db")
    assert controller.reconcile({}, [], account_active=True, complete_snapshot=True)
    controller.set_mode(RUN, operator_approved=True)

    fake_broker = FakeBroker()

    from reflective_loop_runtime.trading.orders import OrderSafetyError

    def boom_prepare(*args: object, **kwargs: object) -> str:
        raise OrderSafetyError("controller refused for test")

    monkeypatch.setattr(controller, "prepare_order", boom_prepare)

    def decide_fn(
        features: object, position: object, limits: object, *, now: float
    ) -> Decision:
        return Decision(
            symbol="BTC/USD",
            action="enter_long",
            reason="t",
            stop=90.0,
            target=110.0,
            notional=100.0,
        )

    cfg = LoopConfig(
        broker=fake_broker,
        tool_broker=tool_broker,
        permits=permits,
        ledger=ledger,
        kill_path=tmp_path / "KILL",
        heartbeat_path=tmp_path / "hb.json",
        decide_fn=decide_fn,
        controller=controller,
        data_dir=tmp_path,
        history_start_iso="2026-10-05T00:00:00+00:00",
    )
    loop = _build_loop(cfg)

    try:
        # Must not raise; the refusal must be journaled.
        loop.tick(1000.0)

        assert calls == [], (
            f"expected no submit when prepare_order refuses, got {calls!r}"
        )
        journal_text = ledger.journal_path().read_text(encoding="utf-8")
        assert any(
            '"kind": "controller_refused"' in line
            and '"side": "buy"' in line
            and "OrderSafetyError" in line
            for line in journal_text.splitlines()
        ), (
            "expected a controller_refused journal entry with side=buy and "
            "OrderSafetyError in the error"
        )
    finally:
        controller.close()
        ledger.close()

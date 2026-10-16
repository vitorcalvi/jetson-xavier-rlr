"""Tests for :class:`reflective_loop_runtime.trading.loop.TraderLoop`.

The six tests below exercise the wave-1 permit-gated trader loop end-to-end:

* ``test_trading_manifest_valid``    -- the manifest builds without error.
* ``test_permit_required_for_order`` -- the kernel rejects ``invoke_effect``
                                       calls without a permit.
* ``test_dry_run_never_orders``      -- a single dry-run tick never reaches
                                       the adapter.
* ``test_loop_entry_then_exit_records_r`` -- an entry at the stop distance
                                       followed by an exit at the target
                                       records a closed trade with
                                       ``r_multiple`` close to 1.0.
* ``test_tick_survives_api_error``   -- a market exception is journaled but
                                       does not crash the loop.
* ``test_heartbeat_written``         -- the JSON heartbeat is emitted with
                                       the documented keys after a tick.

The fixtures reuse the :func:`_bundle` / :func:`_policy` / :func:`_service`
pattern from ``tests/kernel/test_toolbroker.py`` so the permit service in
front of the broker is real, not stubbed.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

import reflective_loop_runtime.trading.loop as loop_module
from reflective_loop_runtime.kernel.audit import AuditLog
from reflective_loop_runtime.kernel.permit import (
    PermitIssuer,
    PermitService,
)
from reflective_loop_runtime.kernel.policy import MissionPolicyStore, sign_envelope
from reflective_loop_runtime.kernel.toolbroker import (
    ToolBroker,
    ToolNotPermitted,
)
from reflective_loop_runtime.trading.ledger import Ledger
from reflective_loop_runtime.trading.loop import (
    TraderLoop,
    trading_manifest,
)
from reflective_loop_runtime.trading.model import Decision, Limits

DEV_SECRET = "dev-only-shared-secret"
ISSUER_KEY = b"governor-owned-hmac-key-0123456789"


# ---------------------------------------------------------------------------
# Permit/broker wiring copied from tests/kernel/test_toolbroker.py
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


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


class _FakeFeatures:
    """A minimal stand-in for ``Features`` carrying only ``.price`` and ``.atr14``.

    The real ``compute_features`` lives in
    :mod:`reflective_loop_runtime.trading.features` and depends on bar/quote
    fixtures we do not want to wire up here. Tests below install this on
    ``loop_module.compute_features`` via the ``fake_features`` fixture so the
    loop's ``tick`` can run without touching the real feature pipeline.
    """

    def __init__(self, price: float) -> None:
        self.price = float(price)
        self.atr14 = 0.0


@pytest.fixture
def fake_features(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, float]:
    """Replace ``loop.compute_features`` with a price-driven stand-in.

    The returned dict carries a mutable ``"price"`` entry that tests can update
    between ticks to simulate the broker filling at different prices.
    """

    state: dict[str, float] = {"price": 100.0}

    def _compute(snap: object, *, now: float | None = None) -> _FakeFeatures:
        return _FakeFeatures(price=state["price"])

    monkeypatch.setattr(loop_module, "compute_features", _compute)
    return state


class _FakeBroker:
    """Minimal stand-in for the broker used in ``test_exit_sells_broker_held_qty``.

    The real :class:`ToolBroker` is passed as ``tool_broker`` (it owns
    ``invoke_effect`` and the permit service). This fake only needs to satisfy
    the loop's ``self._broker.positions()`` lookup inside ``_held_qty``. It
    defaults to an empty position list so the existing tests, which do not
    install a fake broker, keep working unchanged -- the helper simply exposes
    a ``positions()`` that returns ``[]`` when the broker is a
    :class:`ToolBroker` that does not implement the call.
    """

    def __init__(self, positions: list[dict[str, object]] | None = None) -> None:
        self._positions = [dict(p) for p in positions] if positions else []

    def positions(self) -> list[dict[str, object]]:
        return [dict(p) for p in self._positions]


def _build_broker(
    tmp_path: Path,
    fake_order: Callable[[Mapping[str, object]], dict[str, object]] | None,
) -> tuple[PermitService, ToolBroker]:
    """Build a real ``PermitService`` + ``ToolBroker`` for the trader manifest."""
    service = _service(tmp_path, trading_manifest())
    adapters: dict[tuple[str, str], Callable[[Mapping[str, object]], dict[str, object]]] = {}
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


def _make_fake_order(
    calls: list[dict[str, object]],
    price_box: dict[str, float] | None,
) -> Callable[[Mapping[str, object]], dict[str, object]]:
    """Adapter that records args and reports ``price_box`` as the fill price."""

    def fake_order(args: Mapping[str, object]) -> dict[str, object]:
        calls.append(dict(args))
        price = price_box["price"] if price_box is not None else 100.0
        return {
            "order_id": "o1",
            "status": "filled",
            "filled_qty": 0.01,
            "filled_avg_price": price,
            "client_order_id": str(args["client_order_id"]),
        }

    return fake_order


def _noop_market(
    symbols: tuple[str, ...], now: float | None = None, **kw: object
) -> dict[str, object]:
    """Market stand-in: returns a sentinel object per symbol so the loop proceeds."""
    return {sym: object() for sym in symbols}


def _approve(
    _decision: object, _features: object, _state: object, _limits: object, *, now: float
) -> tuple[bool, None]:
    return (True, None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_trading_manifest_valid() -> None:
    """The trader manifest builds without raising."""
    trading_manifest()


def test_permit_required_for_order(tmp_path: Path, fake_features: dict[str, float]) -> None:
    """A ``None`` permit is rejected before the adapter is even looked up."""
    calls: list[dict[str, object]] = []
    fake_order = _make_fake_order(calls, price_box=fake_features)
    _service_obj, broker = _build_broker(tmp_path, fake_order)

    with pytest.raises((ToolNotPermitted, Exception)):
        broker.invoke_effect(
            None,
            "alpaca_paper",
            "order",
            {
                "symbol": "BTC/USD",
                "side": "buy",
                "notional": 100.0,
                "client_order_id": "permit-test-1",
            },
        )

    assert calls == [], "adapter must not be invoked without a permit"


def test_dry_run_never_orders(
    tmp_path: Path,
    fake_features: dict[str, float],
) -> None:
    """A dry-run tick journals the decision but never calls the adapter."""
    calls: list[dict[str, object]] = []
    fake_order = _make_fake_order(calls, price_box=fake_features)
    service, broker = _build_broker(tmp_path, fake_order)
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    def decide_fn(features: object, position: object, limits: object, *, now: float) -> Decision:
        return Decision(
            symbol="BTC/USD",
            action="enter_long",
            reason="t",
            stop=90.0,
            target=110.0,
            notional=100.0,
        )

    loop = TraderLoop(
        market_fn=_noop_market,
        broker=broker,
        permits=service,
        tool_broker=broker,
        ledger=ledger,
        limits=Limits(),
        dry_run=True,
        heartbeat_path=tmp_path / "hb.json",
        kill_path=tmp_path / "KILL",
        decide_fn=decide_fn,
        review_fn=_approve,
        symbols=("BTC/USD",),
    )

    loop.tick(1000.0)

    assert calls == [], "dry-run tick must not reach the order adapter"
    ledger.close()


def test_loop_entry_then_exit_records_r(
    tmp_path: Path,
    fake_features: dict[str, float],
) -> None:
    """Enter at 100, exit at 110, against stop 90/target 110: ``r`` is approximately 1.0."""
    calls: list[dict[str, object]] = []
    fake_order = _make_fake_order(calls, price_box=fake_features)
    service, broker = _build_broker(tmp_path, fake_order)
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    def decide_fn(features: object, position: object, limits: object, *, now: float) -> Decision:
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

    loop = TraderLoop(
        market_fn=_noop_market,
        broker=broker,
        permits=service,
        tool_broker=broker,
        ledger=ledger,
        limits=Limits(),
        heartbeat_path=tmp_path / "hb.json",
        kill_path=tmp_path / "KILL",
        decide_fn=decide_fn,
        review_fn=_approve,
        symbols=("BTC/USD",),
    )

    # Entry tick at PRICE = 100.
    fake_features["price"] = 100.0
    loop.tick(1000.0)

    # Exit tick at PRICE = 110 -> r = (110 - 100) / (100 - 90) = 1.0.
    fake_features["price"] = 110.0
    loop.tick(1001.0)

    closed = ledger.closed_trades()
    assert len(closed) == 1
    trade = closed[0]
    assert trade["symbol"] == "BTC/USD"
    assert trade["r_multiple"] == pytest.approx(1.0)
    assert trade["exit_reason"] == "target"
    ledger.close()


def test_tick_survives_api_error(
    tmp_path: Path,
    fake_features: dict[str, float],
) -> None:
    """A market-fn exception is journaled but does not propagate from ``tick``."""
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    def boom(
        symbols: tuple[str, ...],
        now: float | None = None,
        **kw: object,
    ) -> dict[str, object]:
        raise RuntimeError("network down")

    def hold_decide(features: object, position: object, limits: object, *, now: float) -> Decision:
        return Decision("BTC/USD", "hold", "missing-snapshot")

    service, broker = _build_broker(tmp_path, fake_order=None)

    loop = TraderLoop(
        market_fn=boom,
        broker=broker,
        permits=service,
        tool_broker=broker,
        ledger=ledger,
        limits=Limits(),
        heartbeat_path=tmp_path / "hb.json",
        kill_path=tmp_path / "KILL",
        decide_fn=hold_decide,
        review_fn=_approve,
        symbols=("BTC/USD",),
    )

    # Must return normally; the exception is caught inside ``tick``.
    loop.tick(1000.0)

    journal_text = ledger.journal_path().read_text(encoding="utf-8")
    assert any('"kind": "market_error"' in line for line in journal_text.splitlines()), (
        "expected a market_error record in the journal"
    )
    ledger.close()


def test_heartbeat_written(
    tmp_path: Path,
    fake_features: dict[str, float],
) -> None:
    """After a tick the heartbeat JSON has the documented top-level keys."""
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")
    hb = tmp_path / "hb.json"

    def hold_decide(features: object, position: object, limits: object, *, now: float) -> Decision:
        return Decision("BTC/USD", "hold", "no-trade")

    service, broker = _build_broker(tmp_path, fake_order=None)

    loop = TraderLoop(
        market_fn=_noop_market,
        broker=broker,
        permits=service,
        tool_broker=broker,
        ledger=ledger,
        limits=Limits(),
        heartbeat_path=hb,
        kill_path=tmp_path / "KILL",
        decide_fn=hold_decide,
        review_fn=_approve,
        symbols=("BTC/USD",),
    )

    loop.tick(1000.0)

    payload = json.loads(hb.read_text(encoding="utf-8"))
    assert set(payload.keys()) >= {
        "ts",
        "tick",
        "open_positions",
        "realized_r_today",
        "halted",
    }
    ledger.close()


def test_exit_sells_broker_held_qty(
    tmp_path: Path,
    fake_features: dict[str, float],
) -> None:
    """Exit sells the broker's actual held qty (0.9), not the ledger qty (1.0).

    Alpaca deducts the crypto fee in-kind from the bought quantity, so the
    broker holds less than the ledger records. The loop must consult the
    broker via ``_held_qty`` and sell only what is actually there.
    """
    calls: list[dict[str, object]] = []
    fake_order = _make_fake_order(calls, price_box=fake_features)
    service, broker = _build_broker(tmp_path, fake_order)
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")
    fake_broker = _FakeBroker(positions=[{"symbol": "BTCUSD", "qty": "0.9"}])

    def decide_fn(features: object, position: object, limits: object, *, now: float) -> Decision:
        return Decision("BTC/USD", "exit", "target")

    loop = TraderLoop(
        market_fn=_noop_market,
        broker=fake_broker,
        permits=service,
        tool_broker=broker,
        ledger=ledger,
        limits=Limits(),
        heartbeat_path=tmp_path / "hb.json",
        kill_path=tmp_path / "KILL",
        decide_fn=decide_fn,
        review_fn=_approve,
        symbols=("BTC/USD",),
    )

    # Seed an open position in the ledger with qty=1.0 (the loop will treat
    # this as the "fallback" if the broker reports nothing).
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

    fake_features["price"] = 110.0
    loop.tick(1001.0)

    sell_calls = [c for c in calls if c.get("side") == "sell"]
    assert len(sell_calls) == 1, f"expected exactly one sell order, got {calls!r}"
    assert sell_calls[0]["symbol"] == "BTC/USD"
    assert sell_calls[0]["qty"] == pytest.approx(0.9), (
        f"exit must sell the broker's held qty 0.9, got {sell_calls[0]['qty']!r}"
    )
    ledger.close()


def test_kill_file_still_allows_exits(
    tmp_path: Path,
    fake_features: dict[str, float],
) -> None:
    """Kill file present, but a sell decision still goes through the adapter.

    The kill file blocks *entries* via the governor but exits for an
    already-open position must still be executed; otherwise a stranded
    position is worse than the daily cap the kill was meant to enforce.
    """
    calls: list[dict[str, object]] = []
    fake_order = _make_fake_order(calls, price_box=fake_features)
    service, broker = _build_broker(tmp_path, fake_order)
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    # Seed an open position before touching the kill flag.
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

    # Drop the kill file so the loop sees ``halted=True`` this tick.
    kill_path = tmp_path / "KILL"
    kill_path.write_text("halt", encoding="utf-8")

    def decide_fn(features: object, position: object, limits: object, *, now: float) -> Decision:
        return Decision("BTC/USD", "exit", "target")

    loop = TraderLoop(
        market_fn=_noop_market,
        broker=broker,
        permits=service,
        tool_broker=broker,
        ledger=ledger,
        limits=Limits(),
        heartbeat_path=tmp_path / "hb.json",
        kill_path=kill_path,
        decide_fn=decide_fn,
        review_fn=_approve,
        symbols=("BTC/USD",),
    )

    fake_features["price"] = 110.0
    loop.tick(1001.0)

    sell_calls = [c for c in calls if c.get("side") == "sell"]
    assert len(sell_calls) == 1, f"exit must still execute under kill; adapter calls were {calls!r}"
    assert sell_calls[0]["symbol"] == "BTC/USD"

    journal_text = ledger.journal_path().read_text(encoding="utf-8")
    assert any(
        '"kind": "halt"' in line and '"reason": "kill-file present"' in line
        for line in journal_text.splitlines()
    ), "expected one halt record per tick while the kill file is present"

    hb = json.loads((tmp_path / "hb.json").read_text(encoding="utf-8"))
    assert hb["halted"] is True
    assert loop.is_halted() is True
    ledger.close()


def test_stop_target_rebased_on_fill(
    tmp_path: Path,
    fake_features: dict[str, float],
) -> None:
    """After a fill, the ledger stop/target are anchored on the fill price.

    With decision price=100, decision.stop=99, decision.target=101.6 and
    a fill at 100.5 the protective distances must be preserved, giving
    stop=99.5 and target=102.1 -- not the raw decision levels.
    """
    calls: list[dict[str, object]] = []
    fill_box: dict[str, float] = {"price": 100.5}
    fake_order = _make_fake_order(calls, price_box=fill_box)
    service, broker = _build_broker(tmp_path, fake_order)
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")

    fake_features["price"] = 100.0  # the price the decider saw

    def decide_fn(features: object, position: object, limits: object, *, now: float) -> Decision:
        return Decision(
            symbol="BTC/USD",
            action="enter_long",
            reason="t",
            stop=99.0,
            target=101.6,
            notional=100.0,
        )

    loop = TraderLoop(
        market_fn=_noop_market,
        broker=broker,
        permits=service,
        tool_broker=broker,
        ledger=ledger,
        limits=Limits(),
        heartbeat_path=tmp_path / "hb.json",
        kill_path=tmp_path / "KILL",
        decide_fn=decide_fn,
        review_fn=_approve,
        symbols=("BTC/USD",),
    )

    loop.tick(1000.0)

    open_positions = ledger.open_positions()
    assert "BTC/USD" in open_positions, "expected BTC/USD to be open after fill"
    pos = open_positions["BTC/USD"]
    assert pos.entry_px == pytest.approx(100.5)
    assert pos.stop == pytest.approx(99.5)
    assert pos.target == pytest.approx(102.1)
    ledger.close()


@pytest.mark.parametrize(
    "gate,reason",
    [
        (lambda *_: False, "autonomy_no_validated_edge"),
        (lambda *_: "yes", "autonomy_no_validated_edge"),
        (lambda *_: None, "autonomy_no_validated_edge"),
        (lambda *_: 1 / 0, "autonomy_gate_unavailable"),
    ],
)
def test_optional_autonomy_gate_vetoes_new_entries(
    tmp_path: Path,
    fake_features: dict[str, float],
    gate: Callable[..., object],
    reason: str,
) -> None:
    """An unverified result or gate failure must never reach the order adapter."""
    calls: list[dict[str, object]] = []
    service, toolbroker = _build_broker(tmp_path, _make_fake_order(calls, fake_features))
    ledger = Ledger(tmp_path / "trades.db", tmp_path / "journal.jsonl")

    def want_buy(features: object, position: object, limits: object, *, now: float) -> Decision:
        return Decision("BTC/USD", "enter_long", "signal", stop=90, target=110, notional=100)

    loop = TraderLoop(
        market_fn=_noop_market,
        broker=toolbroker,
        permits=service,
        tool_broker=toolbroker,
        ledger=ledger,
        limits=Limits(),
        decide_fn=want_buy,
        review_fn=_approve,
        symbols=("BTC/USD",),
        entry_evidence_gate=gate,
    )
    loop.tick(1000.0)
    assert calls == []
    journal = ledger.journal_path().read_text()
    assert reason in journal
    ledger.close()


def test_optional_autonomy_gate_does_not_intercept_exits(
    tmp_path: Path,
    fake_features: dict[str, float],
) -> None:
    """Even a failing strategy evaluator must not prevent risk-reducing exits."""
    calls: list[dict[str, object]] = []
    service, toolbroker = _build_broker(tmp_path, _make_fake_order(calls, fake_features))
    ledger = Ledger(tmp_path / "trades.db", tmp_path / "journal.jsonl")
    ledger.open_trade(
        symbol="BTC/USD",
        entry_ts=999.0,
        entry_px=100.0,
        qty=0.01,
        stop=90.0,
        target=110.0,
        atr=0.0,
        entry_order_id="seed",
    )

    def exit_now(features: object, position: object, limits: object, *, now: float) -> Decision:
        return Decision("BTC/USD", "exit", "stop")

    def broken_gate(*args: object) -> bool:
        raise RuntimeError("research system failed")

    loop = TraderLoop(
        market_fn=_noop_market,
        broker=toolbroker,
        permits=service,
        tool_broker=toolbroker,
        ledger=ledger,
        limits=Limits(),
        decide_fn=exit_now,
        review_fn=_approve,
        symbols=("BTC/USD",),
        entry_evidence_gate=broken_gate,
    )
    loop.tick(1000.0)
    assert len([c for c in calls if c.get("side") == "sell"]) == 1
    ledger.close()

"""Offline production OrderController invariants; NO network and NO broker keys."""

from __future__ import annotations

import random
from decimal import Decimal

import pytest

from reflective_loop_runtime.trading.orders import (
    CANCEL_AND_FLATTEN,
    EMERGENCY_STOP,
    HALT_ENTRIES,
    RUN,
    OrderController,
    OrderSafetyError,
    deterministic_order_id,
)


def new_controller(tmp_path, *, qty="10"):
    return OrderController(tmp_path / "orders.db", max_symbol_qty=qty)


def reconcile(controller, orders=None, positions=None, **kwargs):
    return controller.reconcile(
        positions or {}, orders or [], account_active=True, complete_snapshot=True, **kwargs
    )


def normal(tmp_path, *, qty="10"):
    c = new_controller(tmp_path, qty=qty)
    assert reconcile(c)
    c.set_mode(RUN, operator_approved=True)
    return c


def snapshot_order(c, cid, status="new"):
    o = c.snapshot()["orders"][cid]
    return {
        "client_order_id": cid,
        "symbol": o["symbol"],
        "side": o["side"],
        "filled_qty": o["filled_qty"],
        "status": status,
    }


def test_default_halt_and_explicit_operator_approval(tmp_path):
    c = new_controller(tmp_path)
    assert c.snapshot()["mode"] == HALT_ENTRIES
    with pytest.raises(OrderSafetyError, match="reconciliation"):
        c.prepare_order("BTC/USD", "buy", "t0", 1)
    with pytest.raises(OrderSafetyError, match="operator approval"):
        c.set_mode(RUN)
    assert reconcile(c)
    with pytest.raises(OrderSafetyError, match="operator approval"):
        c.set_mode(RUN)
    c.set_mode(RUN, operator_approved=True)
    assert c.snapshot()["mode"] == RUN
    c.close()


def test_halt_durable_over_restart_and_midnight(tmp_path):
    c = normal(tmp_path)
    c.set_mode(HALT_ENTRIES)
    c.begin_day("2026-10-12")
    c.begin_day("2026-10-13")
    assert c.snapshot()["mode"] == HALT_ENTRIES
    c.close()
    c = new_controller(tmp_path)
    state = c.snapshot()
    assert (state["mode"], state["reconciled"], state["uncertain"]) == (HALT_ENTRIES, False, True)
    assert reconcile(c)
    with pytest.raises(OrderSafetyError, match="persistent halt"):
        c.prepare_order("ETH/USD", "buy", 3, 1)
    c.close()


def test_clock_rollback_fails_closed(tmp_path):
    c = normal(tmp_path)
    c.begin_day("2026-10-12")
    c.begin_day("2026-10-11")
    assert c.snapshot()["mode"] == HALT_ENTRIES
    assert c.snapshot()["incident"] == "clock_rolled_back"
    c.close()


def test_deterministic_idempotency_and_qty_conflict(tmp_path):
    c = normal(tmp_path)
    a = c.prepare_order("BTC/USD", "buy", "tick:200", "1")
    assert a == deterministic_order_id("BTC/USD", "buy", "tick:200")
    assert c.prepare_order("BTC/USD", "buy", "tick:200", "1.0") == a
    assert c.snapshot()["entries_today"] == 1
    with pytest.raises(OrderSafetyError, match="different size"):
        c.prepare_order("BTC/USD", "buy", "tick:200", "1.1")
    with pytest.raises(OrderSafetyError, match="unresolved order"):
        c.prepare_order("BTC/USD", "buy", "tick:201", "1")
    c.close()


def test_persistent_unknown_post_and_no_blind_retry(tmp_path):
    c = normal(tmp_path)
    cid = c.prepare_order("BTC/USD", "buy", 5, 1)
    c.mark_submitting(cid)
    with pytest.raises(OrderSafetyError, match="lookup by client order id"):
        c.mark_submitting(cid)
    assert c.snapshot()["mode"] == HALT_ENTRIES
    assert c.snapshot()["uncertain"]
    c.close()
    c = new_controller(tmp_path)
    assert not c.reconcile({}, [], account_active=True, complete_snapshot=True)
    assert c.snapshot()["mode"] == HALT_ENTRIES
    assert reconcile(c, [snapshot_order(c, cid)])
    assert c.snapshot()["incident"] == "broker_order_or_position_mismatch"
    c.clear_incident(operator_approved=True)
    assert c.snapshot()["mode"] == HALT_ENTRIES
    c.close()


def test_cancellation_request_is_not_cancel_confirmed(tmp_path):
    c = normal(tmp_path)
    cid = c.prepare_order("ETH/USD", "buy", 4, 2)
    c.mark_submitting(cid)
    c.acknowledge_order(cid)
    c.request_cancel(cid)
    o = c.snapshot()["orders"][cid]
    assert o["status"] == "CANCEL_REQUESTED" and o["cancel_requested"]
    assert reconcile(c, [snapshot_order(c, cid, "pending_cancel")])
    c.confirm_canceled(cid)
    assert c.snapshot()["orders"][cid]["status"] == "CANCELED"
    assert c.snapshot()["uncertain"]
    assert reconcile(c, [snapshot_order(c, cid, "canceled")])
    c.close()


def test_late_fill_after_cancel_is_recorded_and_halts(tmp_path):
    c = normal(tmp_path)
    cid = c.prepare_order("BTC/USD", "buy", 5, 2)
    c.request_cancel(cid)
    c.confirm_canceled(cid)
    assert c.record_fill(cid, "late-fill", "1.25", fee_in_asset="0.0025")
    snap = c.snapshot()
    assert Decimal(snap["positions"]["BTC/USD"]) == Decimal("1.2475")
    assert snap["orders"][cid]["status"] == "PARTIAL"
    assert snap["mode"] == HALT_ENTRIES
    assert snap["incident"] == "late_or_oversize_fill_requires_reconciliation"
    assert not c.record_fill(cid, "late-fill", "1.25", fee_in_asset="0.0025")
    c.close()


def test_oversize_execution_preserved_and_fails_closed(tmp_path):
    c = normal(tmp_path)
    cid = c.prepare_order("BTC/USD", "buy", 3, 1)
    assert c.record_fill(cid, "exec-too-large", 2)
    snap = c.snapshot()
    assert Decimal(snap["positions"]["BTC/USD"]) == 2
    assert snap["incident"] == "late_or_oversize_fill_requires_reconciliation"
    assert snap["mode"] == HALT_ENTRIES
    c.close()


def test_partial_fills_fee_in_kind_and_execution_replay(tmp_path):
    c = normal(tmp_path)
    cid = c.prepare_order("BTC/USD", "buy", 2, 1)
    assert c.record_fill(cid, "fill-1", "0.4", fee_in_asset="0.001")
    assert c.record_fill(cid, "fill-2", "0.6", fee_in_asset="0.0015")
    assert not c.record_fill(cid, "fill-2", "0.6", fee_in_asset="0.0015")
    assert Decimal(c.snapshot()["positions"]["BTC/USD"]) == Decimal("0.9975")
    assert c.snapshot()["orders"][cid]["status"] == "FILLED"
    with pytest.raises(OrderSafetyError, match="quantity"):
        c.record_fill(cid, "fill-invalid", -1)
    c.close()


def test_reconciled_halted_exit_sellable_qty_no_oversell(tmp_path):
    c = normal(tmp_path)
    cid = c.prepare_order("ETH/USD", "buy", 2, 2)
    c.record_fill(cid, "buy-fill", 2, fee_in_asset="0.005")
    assert reconcile(c, [snapshot_order(c, cid, "filled")], {"ETH/USD": "1.995"})
    with pytest.raises(OrderSafetyError, match="sell exceeds"):
        c.prepare_order("ETH/USD", "sell", "x", 2)
    sell_id = c.prepare_order("ETH/USD", "sell", "x", "1.9")
    with pytest.raises(OrderSafetyError, match="sell exceeds"):
        c.prepare_order("ETH/USD", "sell", "y", "0.2")
    assert c.snapshot()["orders"][sell_id]["side"] == "sell"
    c.close()


def test_flatten_requires_pending_buy_cancellation(tmp_path):
    c = normal(tmp_path)
    cid = c.prepare_order("BTC/USD", "buy", 1, 2)
    c.record_fill(cid, "buy-one", 1)
    assert reconcile(c, [snapshot_order(c, cid, "partially_filled")], {"BTC/USD": 1})
    c.set_mode(CANCEL_AND_FLATTEN)
    with pytest.raises(OrderSafetyError, match="must cancel"):
        c.prepare_order("BTC/USD", "sell", 6, 1)
    with pytest.raises(OrderSafetyError, match="de-escalation"):
        c.set_mode(HALT_ENTRIES)
    c.request_cancel(cid)
    c.confirm_canceled(cid)
    assert reconcile(c, [snapshot_order(c, cid, "canceled")], {"BTC/USD": 1})
    assert c.prepare_order("BTC/USD", "sell", 6, 1)
    c.close()


def test_emergency_blocks_automated_exits_and_requires_approval(tmp_path):
    c = normal(tmp_path)
    c.set_mode(EMERGENCY_STOP)
    with pytest.raises(OrderSafetyError, match="de-escalation"):
        c.set_mode(HALT_ENTRIES)
    assert reconcile(c)
    with pytest.raises(OrderSafetyError, match="operator-managed exits"):
        c.prepare_order("BTC/USD", "sell", 2, 1)
    c.set_mode(HALT_ENTRIES, operator_approved=True)
    assert c.snapshot()["mode"] == HALT_ENTRIES
    c.close()


def test_broker_snapshot_incomplete_unknown_and_mismatch(tmp_path):
    c = normal(tmp_path)
    cid = c.prepare_order("BTC/USD", "buy", 4, 2)
    c.mark_submitting(cid)
    assert not c.reconcile(
        {}, [snapshot_order(c, cid)], account_active=True, complete_snapshot=False
    )
    assert not reconcile(c, [snapshot_order(c, cid)], {"BTC/USD": 5})
    assert c.snapshot()["incident"] == "broker_order_or_position_mismatch"
    assert not reconcile(c, [])
    c.close()


def test_unknown_open_broker_order_fails_closed(tmp_path):
    c = normal(tmp_path)
    external = {
        "client_order_id": "external",
        "symbol": "BTC/USD",
        "side": "buy",
        "filled_qty": "0",
        "status": "new",
    }
    assert not reconcile(c, [external])
    assert c.snapshot()["mode"] == HALT_ENTRIES
    c.close()


def test_separate_controllers_serialized_via_sqlite(tmp_path):
    a = normal(tmp_path)
    b = new_controller(tmp_path)  # restart invalidates readiness for both
    with pytest.raises(OrderSafetyError, match="reconciliation"):
        a.prepare_order("BTC/USD", "buy", 4, 1)
    assert reconcile(b)
    b.set_mode(RUN, operator_approved=True)
    first = a.prepare_order("BTC/USD", "buy", 4, 1)
    assert b.prepare_order("BTC/USD", "buy", 4, 1) == first
    assert b.snapshot()["entries_today"] == 1
    a.close()
    b.close()


def test_invalid_quantities_and_no_secret_fields(tmp_path):
    c = normal(tmp_path)
    for value in [float("nan"), float("inf"), -1, "not-a-number", "0"]:
        with pytest.raises(OrderSafetyError):
            c.prepare_order("ETH/USD", "buy", str(value), value)
    assert "APCA-API" not in str(c.snapshot())
    c.close()


def test_randomized_fault_traces_never_reenter_without_operator(tmp_path):
    """Reproducible property exercise against REAL controller state, not toy FSM."""
    c = normal(tmp_path)
    rng = random.Random(20261012)
    c.set_mode(HALT_ENTRIES)
    for i in range(2000):
        op = rng.choice(("halt", "midnight", "reconcile", "entry", "restart", "bad_snapshot"))
        if op == "halt":
            c.set_mode(HALT_ENTRIES)
        elif op == "midnight":
            c.begin_day("2026-10-12" if i % 2 else "2026-10-13")
        elif op == "reconcile":
            reconcile(c)
        elif op == "entry":
            with pytest.raises(OrderSafetyError):
                c.prepare_order("BTC/USD", "buy", i, "0.5")
        elif op == "restart":
            c.close()
            c = new_controller(tmp_path)
        elif op == "bad_snapshot":
            c.reconcile({}, [], account_active=False, complete_snapshot=True)
        assert c.snapshot()["mode"] == HALT_ENTRIES
    c.close()


def test_execution_id_collision_fails_closed_without_double_count(tmp_path):
    c = normal(tmp_path)
    cid = c.prepare_order("BTC/USD", "buy", 1, 2)
    assert c.record_fill(cid, "exec-A", 1)
    before = c.snapshot()["positions"]["BTC/USD"]
    assert not c.record_fill(cid, "exec-A", 1)  # genuine retry
    assert c.snapshot()["positions"]["BTC/USD"] == before
    assert not c.record_fill(cid, "exec-A", "1.5")  # conflicting identity
    assert c.snapshot()["incident"] == "conflicting_replayed_execution_id"
    assert c.snapshot()["mode"] == HALT_ENTRIES
    c.close()


def test_unexpected_unknown_broker_fill_is_preserved_and_halts(tmp_path):
    c = normal(tmp_path)
    assert c.record_fill("unknown-ord", "unknown-fill", "0.01")
    s = c.snapshot()
    assert s["unexpected_fills"] == [
        {"client_order_id": "unknown-ord", "qty": "0.01", "asset_fee": "0"}
    ]
    assert s["mode"] == HALT_ENTRIES and s["uncertain"]
    assert s["incident"] == "unexpected_broker_fill_unknown_order"
    c.close()


def test_mode_does_not_deescalate_on_fill_after_emergency(tmp_path):
    c = normal(tmp_path)
    cid = c.prepare_order("ETH/USD", "buy", 33, 1)
    c.set_mode(EMERGENCY_STOP)
    c.record_fill(cid, "emergency-late", "0.5")
    assert c.snapshot()["mode"] == EMERGENCY_STOP
    with pytest.raises(OrderSafetyError, match="de-escalation"):
        c.set_mode(HALT_ENTRIES)
    c.close()


def test_operator_cannot_clear_incident_before_reconciliation(tmp_path):
    c = normal(tmp_path)
    cid = c.prepare_order("BTC/USD", "buy", 50, 1)
    c.mark_submitting(cid)
    with pytest.raises(OrderSafetyError, match="cannot clear"):
        c.clear_incident(operator_approved=True)
    assert reconcile(c, [snapshot_order(c, cid)])
    c.clear_incident(operator_approved=True)
    c.close()


# ---------------------------------------------------------------------------
# F2: kill_epoch + sticky HARD incident tests
# ---------------------------------------------------------------------------


def test_buy_not_submittable_after_halt(tmp_path):
    """Preparing a BUY in RUN, then halting, must refuse mark_submitting.

    The order never reached the broker: its status must persist as
    CANCELED_LOCAL and no unknown_post incident must have been recorded.
    """
    c = normal(tmp_path)
    cid = c.prepare_order("BTC/USD", "buy", 1, 1)
    assert c.snapshot()["orders"][cid]["status"] == "PREPARED"
    c.set_mode(HALT_ENTRIES)
    with pytest.raises(OrderSafetyError, match="entry not authorized"):
        c.mark_submitting(cid)
    assert c.snapshot()["orders"][cid]["status"] == "CANCELED_LOCAL"
    assert "unknown_post_must_lookup_before_retry" not in c.snapshot()["incident_history"]
    c.close()


def test_buy_not_submittable_after_kill_epoch_change(tmp_path):
    """A kill_epoch_change can be introduced by halting and returning to RUN.

    Even after the operator returns the controller to RUN, the previously
    PREPARED cid must still be refused: its stored epoch is stale.
    """
    c = normal(tmp_path)
    cid = c.prepare_order("BTC/USD", "buy", 1, 1)
    stored_epoch = c.snapshot()["orders"][cid]["epoch"]
    # Halt increments the kill_epoch; returning to RUN keeps the elevated value.
    c.set_mode(HALT_ENTRIES)
    c.set_mode(RUN, operator_approved=True)
    assert c.snapshot()["mode"] == RUN
    assert c.snapshot()["kill_epoch"] > stored_epoch
    with pytest.raises(OrderSafetyError, match="entry not authorized"):
        c.mark_submitting(cid)
    assert c.snapshot()["orders"][cid]["status"] == "CANCELED_LOCAL"
    # A fresh cid prepared after returning to RUN has the new kill_epoch and
    # must be submittable end-to-end.
    cid2 = c.prepare_order("BTC/USD", "buy", 2, 1)
    c.mark_submitting(cid2)
    assert c.snapshot()["orders"][cid2]["status"] == "SUBMITTING"
    c.close()


def test_sell_submittable_in_flatten(tmp_path):
    """Sells remain submittable in CANCEL_AND_FLATTEN; only EMERGENCY_STOP blocks."""
    c = normal(tmp_path)
    buy_id = c.prepare_order("BTC/USD", "buy", 1, 2)
    c.record_fill(buy_id, "buy-fill", 2)
    assert reconcile(c, [snapshot_order(c, buy_id, "filled")], {"BTC/USD": 2})
    c.set_mode(CANCEL_AND_FLATTEN)
    sell_id = c.prepare_order("BTC/USD", "sell", 5, 1)
    assert c.snapshot()["orders"][sell_id]["status"] == "PREPARED"
    # CANCEL_AND_FLATTEN allows sells; the gate must NOT refuse.
    c.mark_submitting(sell_id)
    assert c.snapshot()["orders"][sell_id]["status"] == "SUBMITTING"
    c.close()


def test_fee_exceeds_fill_incident_is_sticky(tmp_path):
    """asset_fee_exceeds_filled_quantity is HARD: a matching snapshot must not clear it."""
    c = normal(tmp_path)
    cid = c.prepare_order("BTC/USD", "buy", 1, 2)
    # Force fee > filled_qty to trip the HARD incident on record_fill.
    assert c.record_fill(cid, "huge-fee", 1, fee_in_asset=2)
    snap = c.snapshot()
    assert snap["incident"] == "asset_fee_exceeds_filled_quantity"
    assert snap["mode"] == HALT_ENTRIES
    assert snap["incident_history"][-1] == "asset_fee_exceeds_filled_quantity"
    # A matching broker snapshot must NOT auto-clear this HARD incident.
    assert reconcile(c, [snapshot_order(c, cid, "partially_filled")], {"BTC/USD": "0"})
    assert c.snapshot()["incident"] == "asset_fee_exceeds_filled_quantity"
    # The record_fill normal path must also not overwrite the HARD incident with
    # the routine reconciliation reason. A subsequent small fill proves this.
    assert c.record_fill(cid, "tiny-extra", 0.01, fee_in_asset=0)
    assert c.snapshot()["incident"] == "asset_fee_exceeds_filled_quantity"
    c.close()


def test_hard_incident_requires_operator_clear(tmp_path):
    """HARD incidents require clear_incident(operator_approved=True)."""
    c = normal(tmp_path)
    cid = c.prepare_order("BTC/USD", "buy", 1, 2)
    assert c.record_fill(cid, "huge-fee", 1, fee_in_asset=2)
    assert c.snapshot()["incident"] == "asset_fee_exceeds_filled_quantity"
    # Even after a clean reconcile, the HARD incident remains.
    assert reconcile(c, [snapshot_order(c, cid, "partially_filled")], {"BTC/USD": "0"})
    assert c.snapshot()["incident"] == "asset_fee_exceeds_filled_quantity"
    # Without operator_approved=True, clear_incident refuses.
    with pytest.raises(OrderSafetyError, match="operator approval"):
        c.clear_incident()
    # With operator_approved=True and a clean reconcile, the incident clears.
    c.clear_incident(operator_approved=True)
    assert c.snapshot()["incident"] == ""
    c.close()


def test_run_blocked_by_live_buy_from_older_epoch(tmp_path):
    c = normal(tmp_path)
    cid = c.prepare_order("BTC/USD", "buy", 1, 1)
    c.mark_submitting(cid)  # trips HALT_ENTRIES and bumps the epoch
    c.acknowledge_order(cid, broker_status="new")
    broker = [{"client_order_id": cid, "symbol": "BTC/USD", "side": "buy",
               "filled_qty": "0", "status": "new"}]
    assert reconcile(c, orders=broker)
    with pytest.raises(OrderSafetyError, match="unresolved buy"):
        c.set_mode(RUN, operator_approved=True)


def test_refused_entry_does_not_jam_reconciliation(tmp_path):
    c = normal(tmp_path)
    cid = c.prepare_order("BTC/USD", "buy", 1, 1)
    c.set_mode(HALT_ENTRIES)
    with pytest.raises(OrderSafetyError, match="entry not authorized"):
        c.mark_submitting(cid)
    assert c.snapshot()["orders"][cid]["status"] == "CANCELED_LOCAL"
    assert reconcile(c)  # never reached the broker: not a mismatch


def test_fill_under_hard_incident_marks_uncertain(tmp_path):
    c = normal(tmp_path)
    cid = c.prepare_order("BTC/USD", "buy", 1, 2)
    c.mark_submitting(cid)
    c.acknowledge_order(cid, broker_status="new")
    c.record_fill(cid, "e1", "1", fee_in_asset="2")  # fee > fill: HARD incident
    hard = c.snapshot()["incident"]
    assert hard not in ("", "execution_requires_reconciliation")
    # a broker snapshot matching local state clears "uncertain" but not the HARD incident
    snap = c.snapshot()
    o = snap["orders"][cid]
    broker = [{"client_order_id": cid, "symbol": "BTC/USD", "side": "buy",
               "filled_qty": o["filled_qty"], "status": "partially_filled"}]
    reconcile(c, orders=broker, positions=dict(snap["positions"]))
    assert c.snapshot()["incident"] == hard and not c.snapshot()["uncertain"]
    c.record_fill(cid, "e2", "1")
    snap = c.snapshot()
    assert snap["incident"] == hard and snap["uncertain"] and not snap["reconciled"]


def test_prepare_order_records_decision_tick(tmp_path):
    c = normal(tmp_path)
    cid = c.prepare_order("BTC/USD", "buy", 29_863_000, 1)
    assert c.snapshot()["orders"][cid]["decision_tick"] == 29_863_000


def test_reconcile_tolerates_broker_9_decimal_rounding(tmp_path):
    """Live 2026-10-12: SOL buy 4.534653824 with 0.25% fee in kind; broker reports 4.523317189."""
    c = normal(tmp_path)
    cid = c.prepare_order("SOL/USD", "buy", 1, "4.534653824")
    c.mark_submitting(cid)
    c.acknowledge_order(cid, broker_status="new")
    fee = Decimal("4.534653824") * Decimal("0.0025")
    c.record_fill(cid, "e1", "4.534653824", fee_in_asset=str(fee))
    broker_orders = [{"client_order_id": cid, "symbol": "SOL/USD", "side": "buy",
                      "filled_qty": "4.534653824", "status": "filled"}]
    assert reconcile(c, orders=broker_orders, positions={"SOL/USD": "4.523317189"})
    assert c.snapshot()["incident"] == ""


def test_reconcile_still_flags_real_quantity_mismatch(tmp_path):
    c = normal(tmp_path)
    assert not reconcile(c, positions={"SOL/USD": "0.01"})


def test_reconcile_adopts_broker_terminal_state_for_notional_buy(tmp_path):
    """Notional buy: size estimate 4.6275 but actual fill 4.534653824 -> PARTIAL locally, filled at broker."""
    c = normal(tmp_path)
    cid = c.prepare_order("SOL/USD", "buy", 1, "4.6275")
    c.mark_submitting(cid)
    c.acknowledge_order(cid, broker_status="new")
    c.record_fill(cid, "e1", "4.534653824", fee_in_asset=str(Decimal("4.534653824") * Decimal("0.0025")))
    assert c.snapshot()["orders"][cid]["status"] == "PARTIAL"
    broker_orders = [{"client_order_id": cid, "symbol": "SOL/USD", "side": "buy",
                      "filled_qty": "4.534653824", "status": "filled"}]
    assert reconcile(c, orders=broker_orders, positions={"SOL/USD": "4.523317189"})
    assert c.snapshot()["orders"][cid]["status"] == "FILLED"


def test_reconcile_terminal_with_different_fill_is_mismatch(tmp_path):
    c = normal(tmp_path)
    cid = c.prepare_order("SOL/USD", "buy", 1, "4.6275")
    c.mark_submitting(cid)
    c.acknowledge_order(cid, broker_status="new")
    c.record_fill(cid, "e1", "4.0", fee_in_asset="0")
    broker_orders = [{"client_order_id": cid, "symbol": "SOL/USD", "side": "buy",
                      "filled_qty": "4.534653824", "status": "filled"}]
    assert not reconcile(c, orders=broker_orders, positions={"SOL/USD": "4.0"})

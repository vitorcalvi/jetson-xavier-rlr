from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pytest

from rlr_stage2_autonomy import (
    ExplicitPaperOrderReject,
    Stage2DurableController,
    StrictSessionQuoteGate,
)


class MockPaperBroker:
    def __init__(self) -> None:
        self.base_url = "https://paper-api.alpaca.markets"
        self.api_key_id = "PKSTAGE2TESTKEY001"
        self.positions: list[dict[str, Any]] = []
        self.orders: dict[str, dict[str, Any]] = {}
        self.coid_map: dict[str, dict[str, Any]] = {}
        self.timeout_next_post = False
        self.reject_next_stop = False
        self.disconnect_next_post = False
        self.fill_stop_during_cancel = False
        self.partial_fill_qty: str | None = None
        self.post_calls = 0

    def get_positions(self) -> list[dict[str, Any]]:
        return list(self.positions)

    def get_orders(self, status: str = "open") -> list[dict[str, Any]]:
        if status == "open":
            return [
                o
                for o in self.orders.values()
                if o["status"] in ("new", "accepted", "partially_filled")
            ]
        return list(self.orders.values())

    def get_order(self, oid: str) -> dict[str, Any]:
        return self.orders[oid]

    def get_order_by_client_order_id(self, coid: str) -> dict[str, Any] | None:
        return self.coid_map.get(coid)

    def post_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.post_calls += 1
        if self.disconnect_next_post:
            raise ConnectionError("Simulated TLS outage")
        if payload["type"] == "stop" and self.reject_next_stop:
            self.reject_next_stop = False
            raise ExplicitPaperOrderReject(422)

        oid = f"ord-{len(self.orders) + 1}"
        coid = payload["client_order_id"]
        if payload["side"] == "buy":
            fqty = self.partial_fill_qty or "0.032455219"
            st = "partially_filled" if self.partial_fill_qty else "filled"
            o = {
                "id": oid,
                "client_order_id": coid,
                "symbol": "SPY",
                "side": "buy",
                "type": "market",
                "time_in_force": "day",
                "status": st,
                "filled_qty": fqty,
                "filled_avg_price": "585.50",
            }
            self.positions = [{"symbol": "SPY", "qty": fqty, "avg_entry_price": "585.50"}]
        elif payload["type"] == "stop":
            o = {
                "id": oid,
                "client_order_id": coid,
                "symbol": "SPY",
                "side": "sell",
                "type": "stop",
                "time_in_force": "day",
                "qty": payload["qty"],
                "stop_price": payload["stop_price"],
                "status": "new",
                "filled_qty": "0",
            }
        else:
            # Enforce real Alpaca fractional share lock: reject market sell if an open stop still locks shares
            open_stops = [x for x in self.get_orders("open") if x["type"] == "stop"]
            if open_stops:
                raise RuntimeError(
                    "42210000 insufficient qty available for order (shares locked by open stop)"
                )
            o = {
                "id": oid,
                "client_order_id": coid,
                "symbol": "SPY",
                "side": "sell",
                "type": "market",
                "time_in_force": "day",
                "qty": payload["qty"],
                "status": "filled",
                "filled_qty": payload["qty"],
                "filled_avg_price": "585.75",
            }
            self.positions = []
        self.orders[oid] = o
        self.coid_map[coid] = o
        if self.timeout_next_post:
            self.timeout_next_post = False
            raise TimeoutError("Simulated HTTP ReadTimeout after broker accepted order")
        return o

    def cancel_order(self, oid: str) -> None:
        o = self.orders[oid]
        if self.fill_stop_during_cancel:
            o["status"] = "filled"
            o["filled_qty"] = o["qty"]
            self.positions = []
        else:
            o["status"] = "canceled"


def _fresh_eval() -> Any:
    gate = StrictSessionQuoteGate()
    now = dt.datetime(2026, 10, 15, 18, 0, 0, tzinfo=dt.timezone.utc)  # 14:00 EDT
    return gate.evaluate(
        585.48, 585.52, now - dt.timedelta(milliseconds=200), now - dt.timedelta(seconds=15), now
    )


def test_f01_restart_flat_survives(tmp_path: Path) -> None:
    b = MockPaperBroker()
    c = Stage2DurableController(tmp_path / "l", tmp_path / "flags", b, max_cycles=1)
    c.execute_entry("c1", _fresh_eval())
    c.cancel_confirm_and_sell()
    assert c.state_row()["state"] == "COMPLETED_DIAGNOSTIC_QUOTA"
    calls = b.post_calls
    c2 = Stage2DurableController(tmp_path / "l", tmp_path / "flags", b, max_cycles=1)
    assert c2.startup_reconcile()["state"] == "COMPLETED_DIAGNOSTIC_QUOTA"
    assert b.post_calls == calls


def test_f02_restart_with_open_position_adopts_and_protects(tmp_path: Path) -> None:
    b = MockPaperBroker()
    b.positions = [{"symbol": "SPY", "qty": "0.032455219", "avg_entry_price": "585.50"}]
    c = Stage2DurableController(tmp_path / "l", tmp_path / "flags", b)
    st = c.startup_reconcile()
    assert st["state"] == "LONG_PROTECTED"
    assert st["held_qty"] == "0.032455219"
    assert len(b.get_orders("open")) == 1


def test_f03_post_timeout_recovers_by_client_order_id(tmp_path: Path) -> None:
    b = MockPaperBroker()
    b.timeout_next_post = True
    c = Stage2DurableController(tmp_path / "l", tmp_path / "flags", b)
    ord_res = c.execute_entry("c-to", _fresh_eval())
    assert ord_res is not None and ord_res["client_order_id"] == "c-to-buy"
    assert b.post_calls == 2  # 1 buy + 1 stop, zero duplicate buy


def test_f04_partial_fill_matches_fractional_stop_qty(tmp_path: Path) -> None:
    b = MockPaperBroker()
    b.partial_fill_qty = "0.012345678"
    c = Stage2DurableController(tmp_path / "l", tmp_path / "flags", b)
    c.execute_entry("c-pf", _fresh_eval())
    st = c.state_row()
    assert b.get_order(st["stop_order_id"])["qty"] == "0.012345678"
    assert b.get_order_by_client_order_id("c-pf-buy")["status"] == "canceled"
    assert st["state"] == "LONG_PROTECTED"


def test_f05_duplicate_trade_update_deduped(tmp_path: Path) -> None:
    b = MockPaperBroker()
    c = Stage2DurableController(tmp_path / "l", tmp_path / "flags", b)
    assert c.ingest_trade_update("exec-1", "c1-buy") is True
    assert c.ingest_trade_update("exec-1", "c1-buy") is False


def test_f06_rejected_stop_flattens_immediately(tmp_path: Path) -> None:
    b = MockPaperBroker()
    b.reject_next_stop = True
    c = Stage2DurableController(tmp_path / "l", tmp_path / "flags", b)
    c.execute_entry("c-rej", _fresh_eval())
    assert c.state_row()["held_qty"] == "0"
    assert b.positions == []


def test_f07_cancel_race_stop_fill_prevents_oversell(tmp_path: Path) -> None:
    b = MockPaperBroker()
    c = Stage2DurableController(tmp_path / "l", tmp_path / "flags", b)
    c.execute_entry("c-race", _fresh_eval())
    b.fill_stop_during_cancel = True
    calls_before = b.post_calls
    c.cancel_confirm_and_sell()
    assert b.post_calls == calls_before
    assert c.state_row()["held_qty"] == "0"


def test_f08_global_kill_while_flat_blocks_buy(tmp_path: Path) -> None:
    b = MockPaperBroker()
    c = Stage2DurableController(tmp_path / "l", tmp_path / "flags", b)
    (tmp_path / "KILL").write_text("1\n", encoding="utf-8")
    assert c.execute_entry("c-kill", _fresh_eval()) is None
    assert (tmp_path / "last_emergency_exposure.json").exists()


def test_f09_global_kill_while_holding_writes_exposure(tmp_path: Path) -> None:
    b = MockPaperBroker()
    c = Stage2DurableController(tmp_path / "l", tmp_path / "flags", b)
    c.execute_entry("c-hold", _fresh_eval())
    (tmp_path / "KILL").write_text("1\n", encoding="utf-8")
    assert c.cancel_confirm_and_sell() is None
    snap = json.loads((tmp_path / "last_emergency_exposure.json").read_text())
    assert snap["held_qty"] == "0.032455219"


def test_f10_strict_5s_quote_and_65s_bar_gate() -> None:
    gate = StrictSessionQuoteGate()
    now = dt.datetime(2026, 10, 15, 18, 0, 0, tzinfo=dt.timezone.utc)
    assert not gate.evaluate(585.0, 0.0, now, now, now).allowed
    assert not gate.evaluate(585.48, 585.52, now - dt.timedelta(seconds=6), now, now).allowed
    assert not gate.evaluate(585.48, 585.52, now, now - dt.timedelta(seconds=70), now).allowed
    assert not gate.evaluate(585.48, 585.52, now, now, now, source="schwab").allowed


def test_f11_broker_disconnect_fails_closed(tmp_path: Path) -> None:
    b = MockPaperBroker()
    b.disconnect_next_post = True
    c = Stage2DurableController(tmp_path / "l", tmp_path / "flags", b)
    assert c.execute_entry("c-disc", _fresh_eval()) is None
    assert c.state_row()["state"] == "HALT_ENTRIES"


def test_f12_crash_during_sell_ack_reconciles_on_restart(tmp_path: Path) -> None:
    b = MockPaperBroker()
    c = Stage2DurableController(tmp_path / "l", tmp_path / "flags", b)
    c.execute_entry("c-sc", _fresh_eval())
    b.cancel_order(c.state_row()["stop_order_id"])
    b.post_order(
        {
            "symbol": "SPY",
            "qty": "0.032455219",
            "side": "sell",
            "type": "market",
            "time_in_force": "day",
            "client_order_id": "c-sc-sell",
        }
    )
    c._transition(state="SELL_WORKING", sell_coid="c-sc-sell")
    c2 = Stage2DurableController(tmp_path / "l", tmp_path / "flags", b)
    st = c2.startup_reconcile()
    assert st["held_qty"] == "0" and st["completed_cycles"] == 1


def test_f13_two_consecutive_cycles_and_synchronous_heartbeat(tmp_path: Path) -> None:
    b = MockPaperBroker()
    c = Stage2DurableController(tmp_path / "l", tmp_path / "flags", b, max_cycles=2)
    c.execute_entry("cycle-1", _fresh_eval())
    # Simulate planned service restart mid-position during Cycle 1
    c_restarted = Stage2DurableController(tmp_path / "l", tmp_path / "flags", b, max_cycles=2)
    assert c_restarted.startup_reconcile()["state"] == "LONG_PROTECTED"
    c_restarted.cancel_confirm_and_sell("CYCLE_1_EXIT")
    assert c_restarted.state_row()["completed_cycles"] == 1
    # Execute Cycle 2
    c_restarted.execute_entry("cycle-2", _fresh_eval())
    c_restarted.cancel_confirm_and_sell("CYCLE_2_EXIT")
    hb = json.loads((tmp_path / "l" / "heartbeat.json").read_text())
    assert hb["controller_state"] == "COMPLETED_DIAGNOSTIC_QUOTA"
    assert hb["completed_cycles"] == 2
    assert (tmp_path / "l" / "AUTONOMY_2CYCLE_TEST_COMPLETE").exists()


def test_stop_timeout_after_broker_acceptance_recovers_existing_id(tmp_path):
    b = MockPaperBroker()
    original = b.post_order

    def lose_stop_response(payload):
        order = original(payload)
        if payload["type"] == "stop":
            raise ConnectionError("response lost after stop accepted")
        return order

    b.post_order = lose_stop_response
    c = Stage2DurableController(tmp_path / "l", tmp_path / "flags", b)
    c.execute_entry("late-ack", _fresh_eval())
    s = c.state_row()
    assert s["state"] == "LONG_PROTECTED"
    assert len(b.get_orders("open")) == 1
    assert len([o for o in b.orders.values() if o["side"] == "sell"]) == 1


def test_ambiguous_stop_not_found_freezes_without_competing_sell(tmp_path):
    b = MockPaperBroker()
    original = b.post_order

    def missing_stop(payload):
        if payload["type"] == "stop":
            raise TimeoutError("socket timeout: order not confirmed")
        return original(payload)

    b.post_order = missing_stop
    c = Stage2DurableController(tmp_path / "l", tmp_path / "flags", b)
    c.execute_entry("missing-stop", _fresh_eval())
    assert c.state_row()["state"] == "RECONCILING_UNCERTAIN_STOP"
    assert len([o for o in b.orders.values() if o["side"] == "sell"]) == 0
    assert len(b.positions) == 1


def test_pending_cancel_never_places_competing_sell(tmp_path):
    b = MockPaperBroker()
    c = Stage2DurableController(tmp_path / "l", tmp_path / "flags", b)
    c.execute_entry("pending-stop", _fresh_eval())
    stopid = c.state_row()["stop_order_id"]

    def cancel_pending(oid):
        b.orders[oid]["status"] = "pending_cancel"

    b.cancel_order = cancel_pending
    assert c.cancel_confirm_and_sell() is None
    assert c.state_row()["state"] == "CANCELING_STOP_FOR_EXIT"
    assert not any(o["type"] == "market" and o["side"] == "sell" for o in b.orders.values())
    b.orders[stopid]["status"] = "canceled"
    c.cancel_confirm_and_sell()
    assert not b.positions


def test_restart_mismatched_existing_stop_blocks_duplicate(tmp_path):
    b = MockPaperBroker()
    c = Stage2DurableController(tmp_path / "l", tmp_path / "flags", b)
    c.execute_entry("mismatch", _fresh_eval())
    stop = b.get_order(c.state_row()["stop_order_id"])
    stop["qty"] = "0.020000000"
    calls = b.post_calls
    again = Stage2DurableController(tmp_path / "l", tmp_path / "flags", b)
    assert again.startup_reconcile()["state"] == "HALT_ENTRIES"
    assert b.post_calls == calls


def test_heartbeat_repaired_from_sqlite_on_restart(tmp_path, monkeypatch):
    import rlr_stage2_autonomy as module

    b = MockPaperBroker()
    c = Stage2DurableController(tmp_path / "l", tmp_path / "flags", b)
    old = module.atomic_write_json

    def fail_once(path, payload, mode=0o600):
        raise OSError("simulated power loss between WAL commit and heartbeat rename")

    monkeypatch.setattr(module, "atomic_write_json", fail_once)
    with pytest.raises(OSError):
        c._transition(state="HALT_ENTRIES")
    monkeypatch.setattr(module, "atomic_write_json", old)
    again = Stage2DurableController(tmp_path / "l", tmp_path / "flags", b)
    db = again.state_row()
    hb = json.loads((tmp_path / "l" / "heartbeat.json").read_text())
    assert db["heartbeat_synced_seq"] == db["transition_seq"] == hb["transition_seq"]
    assert hb["controller_state"] == "HALT_ENTRIES"


def test_actual_404_transport_response_is_missing_order():
    from deploy.stage2_alpaca_paper_adapter import Stage2AlpacaPaperAdapter

    adapter = Stage2AlpacaPaperAdapter("PKSTAGE2TESTKEY001", "test")

    def remote_not_found(url, headers):
        raise RuntimeError("http transport failed for paper endpoint: status=404 body='not found'")

    adapter._api._http_get = remote_not_found
    assert adapter.get_order_by_client_order_id("unknown-client-id") is None


def test_p0_halt_survives_flat_restart_without_flag(tmp_path):
    b = MockPaperBroker()
    c = Stage2DurableController(tmp_path / "l", tmp_path / "flags", b)
    c._transition(state="HALT_ENTRIES")
    again = Stage2DurableController(tmp_path / "l", tmp_path / "flags", b)
    assert again.startup_reconcile()["state"] == "HALT_ENTRIES"
    assert b.post_calls == 0


def test_p0_partial_buy_cancel_not_terminal_never_places_stop(tmp_path):
    b = MockPaperBroker()
    b.partial_fill_qty = "0.012345678"

    def pending(oid):
        b.orders[oid]["status"] = "pending_cancel"

    b.cancel_order = pending
    c = Stage2DurableController(tmp_path / "l", tmp_path / "flags", b)
    c.execute_entry("c-inflight", _fresh_eval())
    assert c.state_row()["state"] == "BUY_PARTIAL_CANCEL_PENDING"
    assert not any(o["type"] == "stop" for o in b.orders.values())
    b.orders["ord-1"]["status"] = "canceled"
    c.reconcile_partial_buy()
    assert c.state_row()["state"] == "LONG_PROTECTED"
    assert b.get_order(c.state_row()["stop_order_id"])["qty"] == "0.012345678"


def test_p0_stop_fill_with_residual_inventory_is_halt_not_completed(tmp_path):
    b = MockPaperBroker()
    c = Stage2DurableController(tmp_path / "l", tmp_path / "flags", b, max_cycles=1)
    c.execute_entry("c-residual", _fresh_eval())
    b.fill_stop_during_cancel = True
    old_cancel = b.cancel_order

    def incomplete_stop_fill(oid):
        old_cancel(oid)
        b.positions = [{"symbol": "SPY", "qty": "0.009", "avg_entry_price": "585.50"}]

    b.cancel_order = incomplete_stop_fill
    assert c.cancel_confirm_and_sell() is None
    assert c.state_row()["state"] == "HALT_ENTRIES"
    assert c.state_row()["completed_cycles"] == 0


def test_p1_exact_end_session_excluded():
    gate = StrictSessionQuoteGate()
    utc = dt.datetime(2026, 10, 16, 19, 50, 0, tzinfo=dt.timezone.utc)
    r = gate.evaluate(585.48, 585.52, utc, utc, utc)
    assert not r.allowed and r.veto_reason == "OUTSIDE_NY_DIAGNOSTIC_SESSION"
    assert not gate.evaluate(585.48, 585.52, utc, utc, utc, source="unknown").allowed


def test_p0_restart_partial_buy_must_resolve_buy_before_stop(tmp_path):
    b = MockPaperBroker()
    b.partial_fill_qty = "0.012345678"
    c = Stage2DurableController(tmp_path / "l", tmp_path / "flags", b)
    o = b.post_order(
        {
            "symbol": "SPY",
            "notional": "25.00",
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
            "client_order_id": "c-crash-buy",
        }
    )
    c._transition(
        state="SUBMITTING_BUY",
        active_cycle_id="c-crash",
        buy_coid=o["client_order_id"],
        held_qty="0",
    )
    again = Stage2DurableController(tmp_path / "l", tmp_path / "flags", b)
    assert again.startup_reconcile()["state"] == "LONG_PROTECTED"
    assert b.get_order_by_client_order_id("c-crash-buy")["status"] == "canceled"
    assert b.get_order(again.state_row()["stop_order_id"])["qty"] == "0.012345678"


def test_p1_two_stops_on_restart_must_halt(tmp_path):
    b = MockPaperBroker()
    c = Stage2DurableController(tmp_path / "l", tmp_path / "flags", b)
    c.execute_entry("two-stops", _fresh_eval())
    b.orders["duplicate"] = {
        "id": "duplicate",
        "client_order_id": "external",
        "symbol": "SPY",
        "side": "sell",
        "type": "stop",
        "qty": "0.032455219",
        "status": "new",
        "stop_price": "570.00",
        "filled_qty": "0",
    }
    again = Stage2DurableController(tmp_path / "l", tmp_path / "flags", b)
    assert again.startup_reconcile()["state"] == "HALT_ENTRIES"

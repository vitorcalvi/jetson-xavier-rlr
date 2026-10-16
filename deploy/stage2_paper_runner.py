#!/usr/bin/env python3
"""Bounded, strictly PAPER Stage2 autonomous one-minute diagnostic supervisor.

EXPERIMENTAL: run only on an isolated ledger after explicit paper-only
account preflight. Two $25 SPY cycles; not an alpha strategy.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from stage2_alpaca_paper_adapter import Stage2AlpacaPaperAdapter
from trader_runtime import _check_alpaca_credentials, _load_trading_policy_or_raise

from reflective_loop_runtime.trading.experiment_registry import load
from reflective_loop_runtime.trading.market import fetch_snapshots
from rlr_stage2_autonomy import Stage2DurableController, StrictSessionQuoteGate

HOME = Path("/home/vitor/rlr")
LEDGER = Path(
    os.environ.get("RLR_STAGE2_LEDGER", str(HOME / "trader-autonomy-1m-2cycle-20261015"))
).resolve()
if not LEDGER.is_relative_to(HOME.resolve()) or LEDGER == HOME.resolve():
    raise RuntimeError("stage2 ledger must be an isolated path within RLR home")
MAX_CYCLES = int(os.environ.get("RLR_STAGE2_MAX_CYCLES", "2"))
if MAX_CYCLES not in (1, 2):
    raise RuntimeError("only bounded 1- or 2-cycle paper diagnostics allowed")
UNIT = "rlr-stage2-paper.service"


def loop():
    os.umask(0o077)
    if load()["ledger"] != str(LEDGER) or load()["service"] != UNIT:
        raise RuntimeError("Stage2 active experiment registry mismatch")
    if os.environ.get("RLR_TRADER_ASSET_CLASS") != "us_equity":
        raise RuntimeError("Stage2 equity-only preflight")
    _load_trading_policy_or_raise(os.environ)
    key, secret = _check_alpaca_credentials(os.environ)
    api = Stage2AlpacaPaperAdapter(key, secret)
    if not api.account_active():
        raise RuntimeError("Alpaca paper account inactive")
    lock_dir = HOME / "run"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / "alpaca_paper_trader.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        LEDGER.mkdir(exist_ok=True)
        os.chmod(LEDGER, 0o700)
        controller = Stage2DurableController(
            LEDGER, HOME / "flags", api, max_cycles=MAX_CYCLES, notional_usd=25, stop_bps=40
        )
        state = controller.startup_reconcile()
        # Observer-only: broker execution remains exclusively Stage2-owned.
        _toon_bridge = None
        try:
            from stage2_toon_bridge import SafetyLockViolation, Stage2ShadowInterceptor

            _toon_bridge = Stage2ShadowInterceptor.from_env(
                ledger=LEDGER, halt_file=HOME / "flags/HALT_ENTRIES"
            )
        except SafetyLockViolation:
            raise  # Explicit override attempt under a halt is fatal.
        except Exception:
            _toon_bridge = None  # Absent/broken shadow code cannot block Stage2.
        gate = StrictSessionQuoteGate()
        last_completed = time.monotonic()
        while True:
            now = time.time()
            # Independent liveness source, not confused with the persisted
            # SQLite controller state-transition heartbeat.
            live = LEDGER / "runtime_tick.json"
            temp = LEDGER / "runtime_tick.tmp"
            temp.write_text(json.dumps({"ts": now, "service": UNIT}) + "\n")
            os.chmod(temp, 0o600)
            temp.replace(live)
            if (HOME / "KILL").exists() or (HOME / "flags/EMERGENCY_STOP").exists():
                controller.evaluate_kill_switch()
                break
            state = controller.state_row()
            mode = state["state"]
            if _toon_bridge is not None:
                try:
                    _toon_bridge.on_controller_tick(
                        mode=mode,
                        timestamp_ms=int(now * 1000),
                        transition_seq=int(state["transition_seq"]),
                    )
                except Exception:
                    pass  # Shadow must not affect legacy control.
            if mode == "COMPLETED_DIAGNOSTIC_QUOTA":
                if api.get_positions() or api.get_orders(status="open"):
                    raise RuntimeError("completion attempted while broker not flat")
                break
            if mode == "BUY_PARTIAL_CANCEL_PENDING":
                controller.reconcile_partial_buy()
            elif mode in ("SUBMITTING_BUY", "LONG_UNPROTECTED", "SUBMITTING_STOP"):
                order_id = state.get("buy_coid")
                buying = api.get_order_by_client_order_id(order_id) if order_id else None
                if buying and buying.get("status") in ("filled", "partially_filled"):
                    if buying.get("status") == "partially_filled":
                        controller._transition(
                            state="BUY_PARTIAL_CANCEL_PENDING",
                            buy_order_id=buying["id"],
                            held_qty=str(buying.get("filled_qty") or "0"),
                        )
                        controller.reconcile_partial_buy()
                    else:
                        controller.startup_reconcile()
                elif buying and buying.get("status") in ("rejected", "canceled", "expired"):
                    controller.startup_reconcile()
            elif mode == "LONG_PROTECTED":
                stop = api.get_order(state["stop_order_id"])
                if stop.get("status") == "filled":
                    controller.startup_reconcile()
                elif stop.get("status") not in (
                    "new",
                    "accepted",
                    "pending_new",
                    "partially_filled",
                ):
                    # A missing stop requires immediate read-only reconciliation,
                    # then fail-closed if it cannot be reattached.
                    controller.startup_reconcile()
                else:
                    fill = api.get_order_by_client_order_id(state["buy_coid"])
                    if fill and fill.get("filled_at"):
                        age = (
                            now
                            - datetime.fromisoformat(
                                fill["filled_at"].replace("Z", "+00:00")
                            ).timestamp()
                        )
                        if age >= 120 or (HOME / "flags/FLATTEN_PAPER").exists():
                            controller.cancel_confirm_and_sell("TIME_EXIT")
            elif mode in ("CANCELING_STOP_FOR_EXIT", "SUBMITTING_SELL", "SELL_WORKING"):
                if mode == "CANCELING_STOP_FOR_EXIT":
                    stop = api.get_order(state["stop_order_id"])
                    if stop.get("status") in ("canceled", "expired", "filled"):
                        controller.cancel_confirm_and_sell("TIME_EXIT_RECOVERY")
                else:
                    result = (
                        api.get_order_by_client_order_id(state.get("sell_coid"))
                        if state.get("sell_coid")
                        else None
                    )
                    if result and result.get("status") == "filled":
                        controller.startup_reconcile()
            elif mode == "IDLE":
                if (HOME / "flags/HALT_ENTRIES").exists():
                    time.sleep(2)
                    continue
                if state["completed_cycles"] and time.monotonic() - last_completed < 90:
                    time.sleep(2)
                    continue
                if not api.market_is_open():
                    time.sleep(5)
                    continue
                snap = fetch_snapshots(
                    ("SPY",),
                    http=__import__(
                        "reflective_loop_runtime.trading.http", fromlist=["_default_http"]
                    )._default_http,
                    now=now,
                    headers=api.market_headers(),
                )["SPY"]
                if _toon_bridge is not None:
                    try:
                        _toon_bridge.on_market_snapshot(
                            symbol="SPY",
                            snapshot=snap,
                            timestamp_ms=int(now * 1000),
                            position={"state": mode, "held_qty": state["held_qty"]},
                        )
                    except Exception:
                        pass  # Observer is strictly advisory.
                closed = [bar for bar in snap.bars if bar.ts + 60 <= now]
                if snap.quote and closed:
                    q = snap.quote
                    b = closed[-1]
                    dnow = datetime.fromtimestamp(now, timezone.utc)
                    gate_value = gate.evaluate(
                        q.bid,
                        q.ask,
                        datetime.fromtimestamp(q.ts, timezone.utc),
                        datetime.fromtimestamp(b.ts + 60, timezone.utc),
                        dnow,
                    )
                    if gate_value.allowed:
                        if api.get_positions() or api.get_orders(status="open"):
                            raise RuntimeError("unknown account exposure blocks entry")
                        cycle_id = (
                            "stage2-"
                            + str(state["completed_cycles"] + 1)
                            + "-"
                            + str(int(now // 60))
                        )
                        if _toon_bridge is not None:
                            try:
                                _toon_bridge.on_stage2_signal(
                                    action="BUY",
                                    qty=25 / max(float(q.ask), 0.01),
                                    price=float(q.ask),
                                    timestamp_ms=int(now * 1000),
                                )
                            except Exception:
                                pass
                        controller.execute_entry(cycle_id, gate_value)
            elif mode == "HALT_ENTRIES":
                # Persisted operator/safety hold: keep read-only liveness alive,
                # NEVER clear the halt or submit a new order automatically.
                time.sleep(5)
                continue
            elif mode in ("EMERGENCY_STOP", "UNPROTECTED_EMERGENCY_FLATTEN"):
                # Stop with no auto-rearm. The independent monitor remains on.
                raise RuntimeError("controller emergency halted; do not auto-clear")
            else:
                raise RuntimeError("unknown controller state: " + str(mode))
            new = controller.state_row()
            if new["completed_cycles"] > state["completed_cycles"]:
                last_completed = time.monotonic()
            time.sleep(2)


if __name__ == "__main__":
    loop()

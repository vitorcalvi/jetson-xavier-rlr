from __future__ import annotations
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")
PAPER_HOST = "paper-api.alpaca.markets"

SECRET_RES = [
    (re.compile(r"(APCA-API-KEY-ID\s*[:=]\s*)([^\s\"',]+)", re.I), r"\1[REDACTED]"),
    (re.compile(r"(APCA-API-SECRET-KEY\s*[:=]\s*)([^\s\"',]+)", re.I), r"\1[REDACTED]"),
    (re.compile(r"(Authorization\s*[:=]\s*(?:Bearer\s+)?)([^\s\"',]+)", re.I), r"\1[REDACTED]"),
    (re.compile(r"\b(PK[A-Z0-9]{12,40})\b"), "[REDACTED_PK]"),
    (re.compile(r"(https?://ntfy\.sh/)([A-Za-z0-9_\-]+)"), r"\1[REDACTED_TOPIC]"),
]

def redact(val: Any) -> str:
    txt = str(val or "")
    for pat, repl in SECRET_RES:
        txt = pat.sub(repl, txt)
    return txt

def assert_paper_gate(base_url: str, key_id: str) -> None:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme != "https" or parsed.netloc != PAPER_HOST:
        raise RuntimeError(f"HARD_CONSTRAINT_1_VIOLATION: host={parsed.netloc}")
    if not key_id or not key_id.startswith("PK"):
        raise RuntimeError("HARD_CONSTRAINT_1_VIOLATION: key_id must start with PK")

def atomic_write_json(path: Path, data: Dict[str, Any], mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass

def format_fractional_stop_price(entry_price: float, stop_bps: float = 40.0) -> str:
    raw = entry_price * (1.0 - stop_bps / 10000.0)
    floored = math.floor(raw * 100.0) / 100.0
    capped = min(floored, round(entry_price - 0.01, 2))
    if capped <= 0.0:
        raise ValueError(f"Invalid stop price {capped} from entry {entry_price}")
    return f"{capped:.2f}"

@dataclass
class QuoteBarEvaluation:
    allowed: bool
    veto_reason: str
    bid: float
    ask: float
    spread_bps: float
    quote_age_ms: float
    bar_age_s: float

class StrictSessionQuoteGate:
    """R08 & R09: Enforces 5.0s quote freshness, 65.0s closed 1m bar freshness, and NY session."""
    def __init__(self, max_quote_age_ms: float = 5000.0, max_bar_age_s: float = 65.0, max_spread_bps: float = 15.0) -> None:
        self.max_quote_age_ms = max_quote_age_ms
        self.max_bar_age_s = max_bar_age_s
        self.max_spread_bps = max_spread_bps
        self.stale_quote_vetoes = 0
        self._last_quote_ns = 0

    def evaluate(
        self,
        bid: float,
        ask: float,
        quote_ts_utc: dt.datetime,
        bar_close_ts_utc: dt.datetime,
        now_utc: dt.datetime,
        source: str = "alpaca_iex",
    ) -> QuoteBarEvaluation:
        if source != "alpaca_iex":
            self.stale_quote_vetoes += 1
            return QuoteBarEvaluation(False, "R08_SCHWAB_UNVERIFIED_FEED_VETO", bid, ask, 0.0, 0.0, 0.0)
        if not (math.isfinite(bid) and math.isfinite(ask)) or bid <= 0.0 or ask <= 0.0:
            self.stale_quote_vetoes += 1
            return QuoteBarEvaluation(False, "R09_ZERO_OR_NONFINITE_QUOTE", bid, ask, 0.0, 0.0, 0.0)
        if bid >= ask:
            self.stale_quote_vetoes += 1
            return QuoteBarEvaluation(False, "R09_CROSSED_OR_LOCKED_QUOTE", bid, ask, 0.0, 0.0, 0.0)

        q_ns = int(quote_ts_utc.timestamp() * 1e9)
        if q_ns < self._last_quote_ns:
            self.stale_quote_vetoes += 1
            return QuoteBarEvaluation(False, "R09_NON_MONOTONIC_QUOTE_TS", bid, ask, 0.0, 0.0, 0.0)

        q_age_ms = (now_utc - quote_ts_utc).total_seconds() * 1000.0
        if q_age_ms < -250.0:
            self.stale_quote_vetoes += 1
            return QuoteBarEvaluation(False, "R09_FUTURE_CLOCK_DRIFT_VETO", bid, ask, 0.0, q_age_ms, 0.0)
        eff_q_age_ms = max(0.0, q_age_ms)
        if eff_q_age_ms > self.max_quote_age_ms:
            self.stale_quote_vetoes += 1
            return QuoteBarEvaluation(False, "R09_QUOTE_OLDER_THAN_5S", bid, ask, 0.0, eff_q_age_ms, 0.0)

        bar_age_s = (now_utc - bar_close_ts_utc).total_seconds()
        if bar_age_s < 0.0:
            self.stale_quote_vetoes += 1
            return QuoteBarEvaluation(False, "R09_INCOMPLETE_FUTURE_1M_BAR", bid, ask, 0.0, eff_q_age_ms, bar_age_s)
        if bar_age_s > self.max_bar_age_s:
            self.stale_quote_vetoes += 1
            return QuoteBarEvaluation(False, "R09_BAR_OLDER_THAN_65S", bid, ask, 0.0, eff_q_age_ms, bar_age_s)

        mid = 0.5 * (bid + ask)
        spread_bps = ((ask - bid) / mid) * 10000.0
        if spread_bps > self.max_spread_bps:
            self.stale_quote_vetoes += 1
            return QuoteBarEvaluation(False, "R09_SPREAD_EXCEEDS_MAX_BPS", bid, ask, spread_bps, eff_q_age_ms, bar_age_s)

        ny_dt = now_utc.astimezone(NY_TZ)
        if ny_dt.weekday() >= 5 or not (dt.time(9, 35) <= ny_dt.time() < dt.time(15, 50)):
            return QuoteBarEvaluation(False, "OUTSIDE_NY_DIAGNOSTIC_SESSION", bid, ask, spread_bps, eff_q_age_ms, bar_age_s)

        self._last_quote_ns = q_ns
        return QuoteBarEvaluation(True, "PERMIT_OK", bid, ask, spread_bps, eff_q_age_ms, bar_age_s)

class ExplicitPaperOrderReject(RuntimeError):
    """Only a definite broker 400/403/422 response, never a timeout."""
    def __init__(self,status:int):
        self.status=status
        super().__init__(f"paper_order_rejected_http_{status}")

class Stage2DurableController:
    """
    Closes R02, R04, R05, R12, R13, F01-F12:
    - Persist-before-POST SQLite order_intents table
    - Two-phase fractional DAY stop with deterministic client_order_id
    - Cancel-confirm-before-sell with stop-fill race reconciliation
    - Synchronous transition heartbeat (zero lag)
    - Clean idle completion (COMPLETED_DIAGNOSTIC_QUOTA) after 2 cycles
    """
    def __init__(
        self,
        ledger_dir: Path,
        flags_dir: Path,
        broker: Any,
        symbol: str = "SPY",
        notional_usd: float = 25.0,
        stop_bps: float = 40.0,
        max_cycles: int = 2,
    ) -> None:
        self.ledger_dir = ledger_dir
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.ledger_dir, 0o700)
        self.db_path = self.ledger_dir / "controller.db"
        self.outbox_path = self.ledger_dir / "alerts_outbox.jsonl"
        self.heartbeat_path = self.ledger_dir / "heartbeat.json"
        self.complete_marker = self.ledger_dir / f"AUTONOMY_{max_cycles}CYCLE_TEST_COMPLETE"
        self.flags_dir = flags_dir
        self.broker = broker
        self.symbol = symbol
        self.notional_usd = notional_usd
        self.stop_bps = stop_bps
        self.max_cycles = max_cycles
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(str(self.db_path),timeout=10)
        os.chmod(self.db_path, 0o600)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=FULL")
        c.execute("PRAGMA busy_timeout=10000")
        return c

    def _init_db(self) -> None:
        with self._conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS controller_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    state TEXT NOT NULL,
                    transition_seq INTEGER NOT NULL DEFAULT 0,
                    heartbeat_synced_seq INTEGER NOT NULL DEFAULT 0,
                    active_cycle_id TEXT,
                    completed_cycles INTEGER NOT NULL DEFAULT 0,
                    held_qty TEXT NOT NULL DEFAULT '0',
                    entry_fill_price REAL NOT NULL DEFAULT 0.0,
                    buy_coid TEXT,
                    buy_order_id TEXT,
                    stop_coid TEXT,
                    stop_order_id TEXT,
                    stop_price TEXT,
                    sell_coid TEXT,
                    sell_order_id TEXT,
                    paper_orders_submitted INTEGER NOT NULL DEFAULT 0,
                    paper_orders_filled INTEGER NOT NULL DEFAULT 0,
                    paper_orders_rejected INTEGER NOT NULL DEFAULT 0,
                    controller_incidents INTEGER NOT NULL DEFAULT 0,
                    unreconciled_orders INTEGER NOT NULL DEFAULT 0,
                    updated_at_utc TEXT NOT NULL
                );
                INSERT OR IGNORE INTO controller_state (id, state, updated_at_utc)
                VALUES (1, 'IDLE', datetime('now'));

                CREATE TABLE IF NOT EXISTS order_intents (
                    client_order_id TEXT PRIMARY KEY,
                    cycle_id TEXT NOT NULL,
                    leg TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    qty_or_notional TEXT NOT NULL,
                    stop_price TEXT,
                    status TEXT NOT NULL,
                    broker_order_id TEXT,
                    filled_qty TEXT DEFAULT '0',
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS processed_executions (
                    execution_id TEXT PRIMARY KEY,
                    client_order_id TEXT NOT NULL,
                    recorded_at_utc TEXT NOT NULL
                );
                """
            )
        self._write_synchronous_heartbeat()

    def state_row(self) -> Dict[str, Any]:
        with self._conn() as c:
            return dict(c.execute("SELECT * FROM controller_state WHERE id = 1").fetchone())

    def _transition(self, **updates: Any) -> Dict[str, Any]:
        now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
        with self._conn() as c:
            cur = dict(c.execute("SELECT * FROM controller_state WHERE id = 1").fetchone())
            updates["transition_seq"] = int(cur["transition_seq"]) + 1
            updates["updated_at_utc"] = now_iso
            cols = ", ".join(f"{k} = ?" for k in updates.keys())
            c.execute(f"UPDATE controller_state SET {cols} WHERE id = 1", list(updates.values()))
        return self._write_synchronous_heartbeat()

    def _write_synchronous_heartbeat(self) -> Dict[str, Any]:
        row = self.state_row()
        hb = {
            "service": "rlr-stage2-paper.service",
            "mode": "AUTONOMY_DIAGNOSTIC_1M",
            "strategy_class": "DIAGNOSTIC_NON_ALPHA",
            "controller_state": row["state"],
            "transition_seq": row["transition_seq"],
            "completed_cycles": row["completed_cycles"],
            "max_cycles": self.max_cycles,
            "held_qty": row["held_qty"],
            "stop_order_id": row["stop_order_id"],
            "stop_price": row["stop_price"],
            "metrics": {
                "paper_orders_submitted": row["paper_orders_submitted"],
                "paper_orders_filled": row["paper_orders_filled"],
                "paper_orders_rejected": row["paper_orders_rejected"],
                "open_positions": 1 if float(row["held_qty"]) > 0 else 0,
                "controller_incidents": row["controller_incidents"],
                "unreconciled_orders": row["unreconciled_orders"],
            },
            "updated_at_utc": row["updated_at_utc"],
        }
        atomic_write_json(self.heartbeat_path, hb, mode=0o600)
        with self._conn() as c:
            c.execute("UPDATE controller_state SET heartbeat_synced_seq=? WHERE id=1",
                      (row["transition_seq"],))
        return row

    def _queue_alert(self, event: str, payload: Dict[str, Any]) -> None:
        now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
        seed = f"{event}:{payload.get('client_order_id', '')}:{payload.get('state', '')}:{now_iso}"
        aid = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]
        line = json.dumps(
            {
                "alert_id": aid,
                "event": event,
                "title": "RLR Stage2 " + event,
                "message": event + ": " + redact(json.dumps(payload, sort_keys=True))[:800],
                "priority": "high",
                "tags": "paper,trading",
                "symbol": self.symbol,
                "timestamp_utc": now_iso,
                "details": json.loads(redact(json.dumps(payload))),
            },
            sort_keys=True,
        ) + "\n"
        fd = os.open(str(self.outbox_path), os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def _persist_intent(self, coid: str, cycle_id: str, leg: str, side: str, otype: str, qty_notional: str, stop_price: Optional[str] = None) -> None:
        now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
        with self._conn() as c:
            c.execute(
                """
                INSERT OR IGNORE INTO order_intents
                (client_order_id, cycle_id, leg, side, order_type, qty_or_notional, stop_price, status, created_at_utc, updated_at_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'INTENT_PERSISTED', ?, ?)
                """,
                (coid, cycle_id, leg, side, otype, qty_notional, stop_price, now_iso, now_iso),
            )

    def _update_intent(self, coid: str, status: str, broker_id: Optional[str] = None, filled_qty: str = "0") -> None:
        now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
        with self._conn() as c:
            c.execute(
                "UPDATE order_intents SET status = ?, broker_order_id = COALESCE(?, broker_order_id), filled_qty = ?, updated_at_utc = ? WHERE client_order_id = ?",
                (status, broker_id, filled_qty, now_iso, coid),
            )

    def evaluate_kill_switch(self) -> Optional[str]:
        kill_file = self.flags_dir.parent / "KILL"
        if kill_file.exists() or (self.flags_dir / "EMERGENCY_STOP").exists():
            row = self.state_row()
            if row["state"] != "EMERGENCY_STOP":
                positions = self.broker.get_positions()
                open_orders = self.broker.get_orders(status="open")
                snap = {
                    "triggered_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "controller_state_before": row["state"],
                    "held_qty": row["held_qty"],
                    "broker_positions": positions,
                    "broker_open_orders": open_orders,
                }
                atomic_write_json(self.flags_dir.parent / "last_emergency_exposure.json", snap, mode=0o600)
                self._transition(state="EMERGENCY_STOP", controller_incidents=int(row["controller_incidents"]) + 1)
                self._queue_alert("controller_halt", {"reason": "EMERGENCY_STOP", "held_qty": row["held_qty"], "state": "EMERGENCY_STOP"})
            return "EMERGENCY_STOP"
        if (self.flags_dir / "FLATTEN_PAPER").exists():
            return "FLATTEN_PAPER"
        if (self.flags_dir / "HALT_ENTRIES").exists():
            row = self.state_row()
            if row["state"] == "IDLE":
                self._transition(state="HALT_ENTRIES")
                self._queue_alert("controller_halt", {"reason": "HALT_ENTRIES", "state": "HALT_ENTRIES"})
            return "HALT_ENTRIES"
        return None

    def _post_with_coid_recovery(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        assert_paper_gate(self.broker.base_url, self.broker.api_key_id)
        coid = payload["client_order_id"]
        # F03: Check if order already exists on broker from a prior interrupted POST
        existing = self.broker.get_order_by_client_order_id(coid)
        if existing is not None:
            return existing
        row = self.state_row()
        try:
            order = self.broker.post_order(payload)
            self._transition(paper_orders_submitted=int(row["paper_orders_submitted"]) + 1)
            return order
        except TimeoutError:
            recovered = self.broker.get_order_by_client_order_id(coid)
            if recovered is not None:
                self._transition(paper_orders_submitted=int(row["paper_orders_submitted"]) + 1)
                return recovered
            raise

    def _place_protective_day_stop(self, cycle_id: str, filled_qty: str, fill_price: float) -> Optional[Dict[str, Any]]:
        stop_coid = f"{cycle_id}-stop"
        stop_price = format_fractional_stop_price(fill_price, self.stop_bps)
        self._persist_intent(stop_coid, cycle_id, "STOP", "sell", "stop", filled_qty, stop_price)
        self._transition(state="SUBMITTING_STOP", stop_coid=stop_coid, stop_price=stop_price)
        try:
            stop_ord = self._post_with_coid_recovery(
                {
                    "symbol": self.symbol,
                    "qty": filled_qty,
                    "side": "sell",
                    "type": "stop",
                    "time_in_force": "day",
                    "stop_price": stop_price,
                    "client_order_id": stop_coid,
                }
            )
            self._update_intent(stop_coid, stop_ord.get("status", "new"), stop_ord["id"])
            self._transition(state="LONG_PROTECTED", stop_order_id=stop_ord["id"])
            return stop_ord
        except Exception as exc:
            # Never interpret an ambiguous network response as a rejection.
            # The broker may have accepted the stop before the response failed.
            recovered = None
            try:
                for attempt in range(3):
                    recovered = self.broker.get_order_by_client_order_id(stop_coid)
                    open_orders = self.broker.get_orders(status="open")
                    if recovered is None:
                        recovered = next((o for o in open_orders
                                          if o.get("client_order_id")==stop_coid),None)
                    if recovered is not None:
                        break
                    time.sleep(0.25)
            except Exception:
                open_orders = None
            if recovered and recovered.get("status") in (
                "new","accepted","pending_new","held","partially_filled"
            ):
                self._update_intent(stop_coid,recovered["status"],recovered["id"])
                self._transition(state="LONG_PROTECTED",stop_order_id=recovered["id"])
                return recovered
            # A confirmed broker 4xx rejection can be risk-reducing
            # only after a fresh broker order+position inventory proves no
            # competing sell order exists for this SPY inventory.
            if isinstance(exc,ExplicitPaperOrderReject) and open_orders is not None:
                if not any(o.get("symbol")==self.symbol and
                           o.get("side")=="sell" for o in open_orders):
                    self._update_intent(stop_coid,"REJECTED")
                    row = self.state_row()
                    self._transition(state="UNPROTECTED_EMERGENCY_FLATTEN",
                          paper_orders_rejected=int(row["paper_orders_rejected"])+1)
                    self._queue_alert("stop_rejected",{"client_order_id":stop_coid,
                        "http_status":exc.status})
                    return self.cancel_confirm_and_sell(reason="STOP_EXPLICIT_REJECT")
            row = self.state_row()
            self._transition(state="RECONCILING_UNCERTAIN_STOP",
                             controller_incidents=int(row["controller_incidents"])+1)
            self._queue_alert("stop_post_requires_reconciliation",{
                "client_order_id": stop_coid, "error_class":type(exc).__name__})
            return None

    def startup_reconcile(self) -> Dict[str, Any]:
        assert_paper_gate(self.broker.base_url, self.broker.api_key_id)
        if self.evaluate_kill_switch() == "EMERGENCY_STOP":
            return self.state_row()

        row = self.state_row()
        positions = self.broker.get_positions()
        open_orders = self.broker.get_orders(status="open")
        spy_pos = next((p for p in positions if p.get("symbol") == self.symbol), None)
        held_qty = str(spy_pos["qty"]) if spy_pos and float(spy_pos.get("qty", 0)) > 0 else "0"
        avg_entry = float(spy_pos.get("avg_entry_price", 0.0)) if spy_pos else 0.0

        # A persisted safety halt is NOT an ephemeral state. Explicit owner
        # intervention is required even if the broker is currently flat.
        if row["state"] in ("HALT_ENTRIES", "EMERGENCY_STOP"):
            if positions or open_orders:
                self._queue_alert("halted_account_exposure", {
                    "state": row["state"], "positions": len(positions),
                    "orders": len(open_orders),
                })
            return self.state_row()

        # Resolve outstanding BUY before any adoption/protective stop.
        if row["state"] == "BUY_PARTIAL_CANCEL_PENDING":
            return self.reconcile_partial_buy()

        # Crash recovery must not protect only a partial BUY while its
        # unfilled remainder could still arrive from Alpaca.
        if row["state"] in ("SUBMITTING_BUY","LONG_UNPROTECTED"):
            buy=self.broker.get_order_by_client_order_id(row["buy_coid"]) if row.get("buy_coid") else None
            if buy and buy.get("status")=="partially_filled":
                self._transition(state="BUY_PARTIAL_CANCEL_PENDING",
                    buy_order_id=buy["id"],
                    held_qty=str(buy.get("filled_qty") or held_qty))
                return self.reconcile_partial_buy()

        # Unknown stop POST is a durable HALT, not permission to send another.
        if row["state"] in ("RECONCILING_UNCERTAIN_STOP", "SUBMITTING_STOP"):
            coid=row.get("stop_coid")
            found=self.broker.get_order_by_client_order_id(coid) if coid else None
            matching=(found is not None and found.get("side")=="sell"
                      and found.get("type")=="stop"
                      and found.get("client_order_id")==coid)
            if matching and found.get("status") in (
                "new","accepted","pending_new","held","partially_filled"
            ) and held_qty!="0" and str(found.get("qty"))==held_qty:
                self._transition(state="LONG_PROTECTED",stop_order_id=found["id"],
                                 held_qty=held_qty,entry_fill_price=avg_entry)
                return self.state_row()
            if matching and found.get("status")=="filled" and held_qty=="0" and not open_orders:
                n=int(row["completed_cycles"])+1
                state="COMPLETED_DIAGNOSTIC_QUOTA" if n>=self.max_cycles else "IDLE"
                self._transition(state=state,held_qty="0",completed_cycles=n,
                                 stop_order_id=None)
                return self.state_row()
            self._transition(state="HALT_ENTRIES",
                             controller_incidents=int(row["controller_incidents"])+1)
            self._queue_alert("unresolved_stop_on_restart",{"client_order_id":coid})
            return self.state_row()

        # A broker-hosted stop may fill while this process is down.
        # Recognize the confirmed stop execution before clearing the position.
        if row["state"] in ("LONG_PROTECTED", "CANCELING_STOP_FOR_EXIT", "SUBMITTING_STOP") and float(held_qty) == 0 and row.get("stop_coid"):
            previous_stop = self.broker.get_order_by_client_order_id(row["stop_coid"])
            if previous_stop and previous_stop.get("status") == "filled" and not open_orders:
                done = int(row["completed_cycles"]) + 1
                target = "COMPLETED_DIAGNOSTIC_QUOTA" if done >= self.max_cycles else "IDLE"
                self._transition(state=target, held_qty="0", stop_order_id=None, completed_cycles=done,
                                 paper_orders_filled=int(row["paper_orders_filled"]) + 1)
                if target == "COMPLETED_DIAGNOSTIC_QUOTA":
                    atomic_write_json(self.complete_marker, {"completed_cycles": done, "flat_verified": True})
                self._queue_alert("fill_sell", {"client_order_id": row["stop_coid"], "via": "recovered_stop"})
                return self.state_row()

        # F12: Crash during SELL_WORKING or SUBMITTING_SELL when broker already filled the sell
        if row["state"] in ("SUBMITTING_SELL", "SELL_WORKING") and float(held_qty) == 0.0 and row.get("sell_coid"):
            terminal_sell = self.broker.get_order_by_client_order_id(row["sell_coid"])
            if terminal_sell and terminal_sell.get("status") == "filled":
                self._update_intent(row["sell_coid"], "filled", terminal_sell["id"], terminal_sell.get("filled_qty", "0"))
                done = int(row["completed_cycles"]) + 1
                target_state = "COMPLETED_DIAGNOSTIC_QUOTA" if done >= self.max_cycles else "IDLE"
                self._transition(
                    state=target_state,
                    held_qty="0",
                    stop_order_id=None,
                    completed_cycles=done,
                    paper_orders_filled=int(row["paper_orders_filled"]) + 1,
                )
                if target_state == "COMPLETED_DIAGNOSTIC_QUOTA":
                    atomic_write_json(self.complete_marker, {"completed_cycles": done, "flat_verified": True})
                self._queue_alert("fill_sell", {"client_order_id": row["sell_coid"], "reconciled_on_startup": True})
                return self.state_row()

        # F02: Broker holds fractional SPY shares on startup.
        # Never adopt one of several stops or a stop with unexpected identity.
        if float(held_qty) > 0.0:
            cycle_id = row.get("active_cycle_id") or f"adopted-{int(time.time())}"
            buys=[o for o in open_orders if o.get("symbol")==self.symbol and o.get("side")=="buy"]
            open_stops=[o for o in open_orders if o.get("symbol")==self.symbol
                and o.get("side")=="sell" and o.get("type")=="stop"]
            if buys or len(open_stops)>1 or any(
                o.get("symbol")==self.symbol and o.get("side")=="sell" and
                o.get("type")!="stop" for o in open_orders
            ):
                self._transition(state="HALT_ENTRIES", held_qty=held_qty,
                    controller_incidents=int(row["controller_incidents"])+1)
                self._queue_alert("unexpected_broker_orders_on_restart",
                    {"open_buys":len(buys),"open_stops":len(open_stops)})
                return self.state_row()
            open_stop = open_stops[0] if open_stops else None
            stop_valid = False
            if open_stop:
                try:
                    stop_valid = (
                        __import__("decimal").Decimal(str(open_stop["qty"])) ==
                        __import__("decimal").Decimal(held_qty)
                        and open_stop.get("status") in
                        ("new","accepted","pending_new","held","partially_filled")
                        and (not row.get("stop_coid") or open_stop.get("client_order_id")==row["stop_coid"])
                        and (not row.get("stop_order_id") or open_stop.get("id")==row["stop_order_id"])
                    )
                except (ValueError, TypeError, KeyError, __import__("decimal").InvalidOperation):
                    stop_valid=False
            if open_stop and stop_valid:
                self._transition(
                    state="LONG_PROTECTED",
                    active_cycle_id=cycle_id,
                    held_qty=held_qty,
                    entry_fill_price=avg_entry,
                    stop_order_id=open_stop["id"],
                    stop_coid=open_stop.get("client_order_id"),
                    stop_price=open_stop.get("stop_price"),
                )
            elif open_stop:
                self._transition(state="HALT_ENTRIES",held_qty=held_qty,
                      controller_incidents=int(row["controller_incidents"])+1)
                self._queue_alert("mismatched_existing_stop",{
                      "stop_order_id":open_stop.get("id"),"broker_qty":held_qty})
                return self.state_row()
            else:
                self._transition(state="LONG_UNPROTECTED", active_cycle_id=cycle_id, held_qty=held_qty, entry_fill_price=avg_entry)
                self._place_protective_day_stop(cycle_id, held_qty, avg_entry)
            self._queue_alert("service_restart", {"adopted_qty": held_qty, "state": self.state_row()["state"]})
            return self.state_row()

        # Never reset a pending BUY/SEND to idle on a flat snapshot.
        if row["state"] in ("SUBMITTING_BUY", "SUBMITTING_STOP", "SUBMITTING_SELL",
                            "SELL_WORKING", "CANCELING_STOP_FOR_EXIT"):
            self._transition(state="HALT_ENTRIES", controller_incidents=int(row["controller_incidents"])+1)
            self._queue_alert("controller_halt", {"reason":"unresolved_order_after_restart"})
            return self.state_row()
        if open_orders:
            self._transition(state="HALT_ENTRIES", controller_incidents=int(row["controller_incidents"])+1)
            return self.state_row()
        # F01: Broker is flat
        if int(row["completed_cycles"]) >= self.max_cycles and not open_orders:
            self._transition(state="COMPLETED_DIAGNOSTIC_QUOTA", held_qty="0", stop_order_id=None)
            atomic_write_json(self.complete_marker, {"completed_cycles": row["completed_cycles"], "flat_verified": True})
        elif self.evaluate_kill_switch() is None:
            self._transition(state="IDLE", held_qty="0", stop_order_id=None)
        self._queue_alert("service_restart", {"adopted_qty": "0", "state": self.state_row()["state"]})
        return self.state_row()

    def execute_entry(self, cycle_id: str, gate_eval: QuoteBarEvaluation) -> Optional[Dict[str, Any]]:
        if self.evaluate_kill_switch() is not None:
            return None
        if not gate_eval.allowed:
            self._queue_alert("stale_market_data", {"reason": gate_eval.veto_reason, "quote_age_ms": gate_eval.quote_age_ms})
            return None

        row = self.state_row()
        if int(row["completed_cycles"]) >= self.max_cycles:
            self._transition(state="COMPLETED_DIAGNOSTIC_QUOTA")
            return None
        if row["state"] != "IDLE" or float(row["held_qty"]) > 0.0:
            return None

        buy_coid = f"{cycle_id}-buy"
        self._persist_intent(buy_coid, cycle_id, "ENTRY", "buy", "market", f"{self.notional_usd:.2f}")
        self._transition(state="SUBMITTING_BUY", active_cycle_id=cycle_id, buy_coid=buy_coid)
        try:
            order = self._post_with_coid_recovery(
                {
                    "symbol": self.symbol,
                    "notional": f"{self.notional_usd:.2f}",
                    "side": "buy",
                    "type": "market",
                    "time_in_force": "day",
                    "client_order_id": buy_coid,
                }
            )
        except ConnectionError as exc:
            row2 = self.state_row()
            self._transition(state="HALT_ENTRIES", controller_incidents=int(row2["controller_incidents"]) + 1)
            self._queue_alert("broker_disconnect", {"client_order_id": buy_coid, "error": redact(exc)})
            return None

        status = order.get("status", "")
        filled_qty = str(order.get("filled_qty") or "0")
        fill_price = float(order.get("filled_avg_price") or gate_eval.ask)
        self._update_intent(buy_coid, status, order["id"], filled_qty)

        if status == "partially_filled" and float(filled_qty) > 0.0:
            self._transition(
                state="BUY_PARTIAL_CANCEL_PENDING",
                buy_order_id=order["id"], held_qty=filled_qty,
                entry_fill_price=fill_price,
            )
            self._queue_alert("partial_fill", {"client_order_id":buy_coid,
                "qty":filled_qty,"action":"cancel_remaining_buy_before_final_stop"})
            self.reconcile_partial_buy()
        elif status == "filled" and float(filled_qty) > 0.0:
            row3 = self.state_row()
            self._transition(
                state="LONG_UNPROTECTED",
                buy_order_id=order["id"], held_qty=filled_qty,
                entry_fill_price=fill_price,
                paper_orders_filled=int(row3["paper_orders_filled"]) + 1,
            )
            self._queue_alert("fill_buy", {"client_order_id": buy_coid, "qty": filled_qty, "price": fill_price})
            self._place_protective_day_stop(cycle_id, filled_qty, fill_price)
        return order

    def reconcile_partial_buy(self) -> Dict[str, Any]:
        """Resolve an outstanding partial BUY without allowing competing stop/sell.

        Remains fail-closed while BUY is open or cancellation is uncertain.
        Only a terminal BUY + broker position inventory permits a final stop.
        """
        row=self.state_row()
        if row["state"] != "BUY_PARTIAL_CANCEL_PENDING":
            return row
        coid=row.get("buy_coid")
        order=self.broker.get_order_by_client_order_id(coid) if coid else None
        if not order or order.get("side")!="buy" or order.get("symbol")!=self.symbol:
            self._transition(state="HALT_ENTRIES",
                controller_incidents=int(row["controller_incidents"])+1)
            self._queue_alert("partial_buy_unknown_order", {"coid":coid})
            return self.state_row()
        status=order.get("status")
        if status not in ("filled","canceled","expired","rejected"):
            if status not in ("pending_cancel",):
                try:
                    self.broker.cancel_order(order["id"])
                except Exception as exc:
                    self._queue_alert("partial_buy_cancel_uncertain",
                        {"coid":coid,"exception":type(exc).__name__})
            order=self.broker.get_order_by_client_order_id(coid)
            status=order.get("status") if order else None
            if status not in ("filled","canceled","expired","rejected"):
                age=(dt.datetime.now(dt.timezone.utc)
                    - dt.datetime.fromisoformat(row["updated_at_utc"])).total_seconds()
                if age>20:
                    self._transition(state="HALT_ENTRIES",
                        controller_incidents=int(row["controller_incidents"])+1)
                    self._queue_alert("partial_buy_cancel_stuck",
                        {"coid":coid,"status":status})
                return self.state_row()

        positions=self.broker.get_positions()
        orders=self.broker.get_orders(status="open")
        held=next((p for p in positions if p.get("symbol")==self.symbol),None)
        held_qty=str(held.get("qty") or "0") if held else "0"
        buy_open=any(o.get("symbol")==self.symbol and o.get("side")=="buy"
                     for o in orders)
        sell_open=any(o.get("symbol")==self.symbol and o.get("side")=="sell"
                      for o in orders)
        if buy_open or sell_open or float(held_qty)<0:
            self._transition(state="HALT_ENTRIES",
                controller_incidents=int(row["controller_incidents"])+1)
            self._queue_alert("partial_buy_order_inventory_mismatch",
                {"coid":coid, "buy_open":buy_open,"sell_open":sell_open})
            return self.state_row()
        if float(held_qty)<=0:
            self._transition(state="HALT_ENTRIES",held_qty="0")
            self._queue_alert("partial_buy_terminal_flat",{"coid":coid,"status":status})
            return self.state_row()
        price=float(held.get("avg_entry_price") or order.get("filled_avg_price") or 0)
        if price<=0:
            self._transition(state="HALT_ENTRIES")
            self._queue_alert("partial_buy_missing_fill_price",{"coid":coid})
            return self.state_row()
        self._transition(state="LONG_UNPROTECTED",held_qty=held_qty,
            entry_fill_price=price,buy_order_id=order["id"])
        self._place_protective_day_stop(row["active_cycle_id"],held_qty,price)
        return self.state_row()

    def cancel_confirm_and_sell(self, reason: str = "TIME_EXIT") -> Optional[Dict[str, Any]]:
        if self.evaluate_kill_switch() == "EMERGENCY_STOP":
            return None
        row = self.state_row()
        held_qty = str(row.get("held_qty") or "0")
        if float(held_qty) <= 0.0:
            return None

        cycle_id = row.get("active_cycle_id") or "cycle"
        stop_oid = row.get("stop_order_id")

        # Step 1: Cancel active DAY stop and confirm terminal status before submitting market SELL
        if stop_oid:
            self._transition(state="CANCELING_STOP_FOR_EXIT")
            original=self.broker.get_order(stop_oid)
            if original.get("status") not in ("canceled","filled","expired"):
                self.broker.cancel_order(stop_oid)
            stop_state=None
            for _ in range(25):
                stop_state=self.broker.get_order(stop_oid)
                if stop_state.get("status") in ("canceled","filled","expired"):
                    break
                time.sleep(0.1)
            if not stop_state or stop_state.get("status") not in ("canceled","filled","expired"):
                self._queue_alert("stop_cancel_pending",{"cycle_id":cycle_id})
                return None
            if stop_state.get("status") == "filled":
                # Stop-filled may represent only a partial exit. Reconcile
                # actual broker inventory before recording a complete cycle.
                remaining = [p for p in self.broker.get_positions()
                             if p.get("symbol")==self.symbol and float(p.get("qty") or 0)>0]
                if remaining:
                    self._transition(state="HALT_ENTRIES",
                        controller_incidents=int(row["controller_incidents"])+1)
                    self._queue_alert("residual_position_after_stop_fill",
                        {"remaining_qty":remaining[0].get("qty")})
                    return None
                done = int(row["completed_cycles"]) + 1
                next_st = "COMPLETED_DIAGNOSTIC_QUOTA" if done >= self.max_cycles else "IDLE"
                self._transition(
                    state=next_st,
                    held_qty="0",
                    stop_order_id=None,
                    completed_cycles=done,
                    paper_orders_filled=int(row["paper_orders_filled"]) + 1,
                )
                if next_st == "COMPLETED_DIAGNOSTIC_QUOTA":
                    atomic_write_json(self.complete_marker, {"completed_cycles": done, "flat_verified": True})
                self._queue_alert("fill_sell", {"client_order_id": stop_state.get("client_order_id"), "qty": stop_state.get("filled_qty"), "via": "protective_stop"})
                return stop_state
            if stop_state.get("status") not in ("canceled", "expired"):
                raise RuntimeError(f"STOP_CANCEL_NOT_CONFIRMED: status={stop_state.get('status')}")

        # Step 2: Re-read the broker's sellable inventory after stop
        # cancellation; a partial stop fill can change the holding quantity.
        current=self.broker.get_positions()
        current_qty=next((str(p["qty"]) for p in current
                          if p.get("symbol")==self.symbol),"0")
        if float(current_qty)<=0:
            self._transition(state="HALT_ENTRIES")
            self._queue_alert("position_flat_without_confirmed_exit",{"cycle_id":cycle_id})
            return None
        if float(current_qty)>float(held_qty)+1e-9:
            self._transition(state="HALT_ENTRIES")
            self._queue_alert("unexpected_position_increase",{"cycle_id":cycle_id})
            return None
        held_qty=current_qty
        sell_coid = f"{cycle_id}-sell"
        self._persist_intent(sell_coid, cycle_id, "EXIT", "sell", "market", held_qty)
        self._transition(state="SUBMITTING_SELL", sell_coid=sell_coid, stop_order_id=None)
        sell_ord = self._post_with_coid_recovery(
            {
                "symbol": self.symbol,
                "qty": held_qty,
                "side": "sell",
                "type": "market",
                "time_in_force": "day",
                "client_order_id": sell_coid,
            }
        )
        self._update_intent(sell_coid, sell_ord.get("status", ""), sell_ord["id"], sell_ord.get("filled_qty", "0"))
        if sell_ord.get("status") == "filled":
            row2 = self.state_row()
            done = int(row2["completed_cycles"]) + 1
            next_st = "COMPLETED_DIAGNOSTIC_QUOTA" if done >= self.max_cycles else "IDLE"
            if self.evaluate_kill_switch() == "HALT_ENTRIES":
                next_st = "HALT_ENTRIES"
            self._transition(
                state=next_st,
                held_qty="0",
                sell_order_id=sell_ord["id"],
                completed_cycles=done,
                paper_orders_filled=int(row2["paper_orders_filled"]) + 1,
            )
            if next_st == "COMPLETED_DIAGNOSTIC_QUOTA":
                atomic_write_json(self.complete_marker, {"completed_cycles": done, "flat_verified": True})
            self._queue_alert("fill_sell", {"client_order_id": sell_coid, "qty": sell_ord.get("filled_qty"), "reason": reason})
        return sell_ord

    def ingest_trade_update(self, exec_id: str, coid: str) -> bool:
        with self._conn() as c:
            if c.execute("SELECT 1 FROM processed_executions WHERE execution_id = ?", (exec_id,)).fetchone():
                return False
            c.execute(
                "INSERT INTO processed_executions (execution_id, client_order_id, recorded_at_utc) VALUES (?, ?, ?)",
                (exec_id, coid, dt.datetime.now(dt.timezone.utc).isoformat()),
            )
        return True


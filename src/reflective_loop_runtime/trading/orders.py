"""Fail-closed, durable PAPER order-state controller (integration not yet wired).

This module has NO network/credential access and cannot transmit an order.
An orchestrator must inject broker-truth reconciliation, signed permit checks,
and HTTP order execution in a separately reviewed change. No assumption of a
cancel acknowledgment, HTTP timeout, or order lookup implies no late fill.

SQLite serializes every transition using BEGIN IMMEDIATE. Persistence is local
to an explicit caller-provided path; on controller startup execution readiness
is invalidated until a complete broker reconciliation is supplied.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path
from typing import Any, TypeVar

HALT_ENTRIES = "HALT_ENTRIES"
CANCEL_AND_FLATTEN = "CANCEL_AND_FLATTEN"
EMERGENCY_STOP = "EMERGENCY_STOP"
RUN = "RUN"
# CANCELED_LOCAL: refused before any POST, so it never reached the broker.
TERMINAL = frozenset({"FILLED", "CANCELED", "REJECTED", "CANCELED_LOCAL"})
ACTIVE = frozenset({"PREPARED", "SUBMITTING", "ACKED", "PARTIAL", "CANCEL_REQUESTED"})
MODES = frozenset({RUN, HALT_ENTRIES, CANCEL_AND_FLATTEN, EMERGENCY_STOP})
# Incidents a matching broker snapshot may auto-clear; anything else is HARD
# and must be cleared explicitly via clear_incident(operator_approved=True).
CLEARABLE_INCIDENTS = frozenset(
    {
        "initial_broker_reconciliation_required",
        "restart_requires_reconciliation",
        "unknown_post_must_lookup_before_retry",
        "cancel_confirmed_requires_broker_reconciliation",
        "execution_requires_reconciliation",
    }
)
T = TypeVar("T")


class OrderSafetyError(RuntimeError):
    """An unsafe transition was refused; never retry blindly."""



# Alpaca reports quantities with 9 decimals; the controller derives positions
# exactly (fee in kind: qty * (1 - fee)), so compare within broker precision.
QTY_TOLERANCE = Decimal("0.000000002")


def _qty_equal(a: Any, b: Any) -> bool:
    return abs(_qty(a) - _qty(b)) <= QTY_TOLERANCE

def _qty(value: Any) -> Decimal:
    try:
        d = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise OrderSafetyError("invalid quantity") from exc
    if not d.is_finite() or d < 0:
        raise OrderSafetyError("quantity must be finite and nonnegative")
    return d


def _positive(value: Any) -> Decimal:
    d = _qty(value)
    if d == 0:
        raise OrderSafetyError("quantity must be positive")
    return d


def _canonical(value: Decimal) -> str:
    return format(value, "f")


def deterministic_order_id(symbol: str, side: str, decision_tick: str | int) -> str:
    """Stable identifier for the *same* intent; no random retry identifiers."""
    if not symbol or side not in {"buy", "sell"} or not str(decision_tick):
        raise OrderSafetyError("missing order identity")
    return "rlr-" + sha256(f"{symbol}|{side}|{decision_tick}".encode()).hexdigest()[:24]


def _initial() -> dict[str, Any]:
    return {
        "version": 1,
        "mode": HALT_ENTRIES,
        "reconciled": False,
        "uncertain": True,
        "incident": "initial_broker_reconciliation_required",
        "utc_day": "",
        "entries_today": 0,
        "orders": {},
        "positions": {},
        "seen_executions": [],
        "execution_details": {},
        "unexpected_fills": [],
        "revision": 0,
        "kill_epoch": 0,
        "incident_history": [],
    }


class OrderController:
    """Durable intent registry and independent safety gate for paper orders.

    - Buy intents require an *explicit* operator RUN transition after complete
      broker reconciliation. HALT_ENTRIES permits reconciled risk-reducing sells.
    - EMERGENCY_STOP disables automated sends; operator escalation is required
      for existing exposure. CANCEL_AND_FLATTEN allows sells only once all BUY
      intents are confirmed terminal and broker positions are reconciled.
    - No callbacks to order placement are implemented here. `mark_submitting`
      must be committed before any external POST; an unknown POST outcome
      requires lookup by client_order_id and complete account reconciliation.
    - Broker fills, including after cancel confirmation, always remain recorded.
    - The `kill_epoch` counter invalidates every previously PREPARED entry as
      soon as the controller halts; a stale cid cannot be POSTed later even
      after an operator returns the system to RUN.
    - HARD incidents (anything outside CLEARABLE_INCIDENTS) require explicit
      operator review via `clear_incident(operator_approved=True)`; matching
      broker snapshots never auto-clear them.
    """

    def __init__(self, database_path: str | Path, *, max_symbol_qty: Any = "1000000"):
        self.path = str(database_path)
        self.max_symbol_qty = _positive(max_symbol_qty)
        self._db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        self._db.execute("PRAGMA busy_timeout=10000")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS order_state (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL)"
        )
        self._db.execute(
            "INSERT OR IGNORE INTO order_state(id,payload) VALUES(1,?)",
            (json.dumps(_initial(), sort_keys=True),),
        )
        # Construction is fail-closed even if last process was reconciled.
        self._modify(lambda s: self._invalidate_on_startup(s))

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> OrderController:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _invalidate_on_startup(state: dict[str, Any]) -> None:
        state["reconciled"] = False
        state["uncertain"] = True
        if state["mode"] == RUN:
            state["mode"] = HALT_ENTRIES
        if not state["incident"]:
            state["incident"] = "restart_requires_reconciliation"

    def snapshot(self) -> dict[str, Any]:
        row = self._db.execute("SELECT payload FROM order_state WHERE id=1").fetchone()
        if row is None:
            raise OrderSafetyError("missing state")
        return deepcopy(json.loads(row[0]))

    def _modify(self, fn: Callable[[dict[str, Any]], T]) -> T:
        self._db.execute("BEGIN IMMEDIATE")
        try:
            row = self._db.execute("SELECT payload FROM order_state WHERE id=1").fetchone()
            if row is None:
                raise OrderSafetyError("missing state")
            data = json.loads(row[0])
            # Tolerate older persisted payloads missing new keys.
            data.setdefault("kill_epoch", 0)
            data.setdefault("incident_history", [])
            result = fn(data)
            self._assert_internal(data)
            data["revision"] += 1
            self._db.execute(
                "UPDATE order_state SET payload=? WHERE id=1",
                (json.dumps(data, sort_keys=True, separators=(",", ":")),),
            )
            self._db.execute("COMMIT")
            return result
        except BaseException:
            self._db.execute("ROLLBACK")
            raise

    @staticmethod
    def _assert_internal(s: dict[str, Any]) -> None:
        if s["mode"] not in MODES:
            raise OrderSafetyError("invalid mode")
        ids = set()
        for cid, order in s["orders"].items():
            if cid in ids or cid != order["client_order_id"]:
                raise OrderSafetyError("duplicated order id")
            ids.add(cid)
            if _qty(order["filled_qty"]) > _qty(order["requested_qty"]) and not s["incident"]:
                raise OrderSafetyError("overfill must halt entries")
        if len(s["seen_executions"]) != len(set(s["seen_executions"])):
            raise OrderSafetyError("execution id replay")
        if set(s["seen_executions"]) != set(s["execution_details"]):
            raise OrderSafetyError("execution metadata incomplete")
        if s["mode"] == RUN and (s["uncertain"] or s["incident"] or not s["reconciled"]):
            raise OrderSafetyError("RUN requires clean reconciled broker state")

    @staticmethod
    def _is_hard_incident(incident: str) -> bool:
        """An incident is HARD if a matching snapshot may not auto-clear it."""
        return bool(incident) and incident not in CLEARABLE_INCIDENTS

    @staticmethod
    def _trip(s: dict[str, Any], reason: str) -> None:
        if s["mode"] == RUN:
            s["mode"] = HALT_ENTRIES
        s["uncertain"] = True
        s["reconciled"] = False
        s["kill_epoch"] = s.get("kill_epoch", 0) + 1
        s.setdefault("incident_history", []).append(reason)
        # A HARD incident must survive a subsequent clearable trigger; HARD
        # anomalies always require operator review (clear_incident).
        if (
            OrderController._is_hard_incident(s["incident"])
            and reason in CLEARABLE_INCIDENTS
        ):
            return
        s["incident"] = reason

    def set_mode(self, mode: str, *, operator_approved: bool = False) -> None:
        if mode not in MODES:
            raise OrderSafetyError("invalid mode")

        def update(s: dict[str, Any]) -> None:
            if s["mode"] == EMERGENCY_STOP and mode != EMERGENCY_STOP and not operator_approved:
                raise OrderSafetyError("emergency de-escalation requires operator approval")
            if (
                s["mode"] == CANCEL_AND_FLATTEN
                and mode in {RUN, HALT_ENTRIES}
                and not operator_approved
            ):
                raise OrderSafetyError("flatten de-escalation requires operator approval")
            if mode == RUN:
                if not operator_approved:
                    raise OrderSafetyError("RUN requires direct operator approval")
                if not s["reconciled"] or s["uncertain"] or s["incident"]:
                    raise OrderSafetyError("cannot RUN before broker reconciliation")
                # A PREPARED buy whose stored epoch matches the current
                # kill_epoch is still a live, in-flight intent and blocks RUN.
                # Stale PREPARED buys (epoch != kill_epoch) are dead because
                # mark_submitting refuses them via the epoch gate, so an
                # operator may return to RUN even with them present.
                if any(
                    o["side"] == "buy"
                    and o["status"] in ACTIVE
                    # a stale PREPARED buy can never be POSTed; anything
                    # further along may be live at the broker in any epoch
                    and (o["status"] != "PREPARED" or o.get("epoch") == s.get("kill_epoch", 0))
                    for o in s["orders"].values()
                ):
                    raise OrderSafetyError("unresolved buy order")
            s["mode"] = mode
            # Any move to a non-RUN mode invalidates all previously authorized
            # PREPARED entries (their stored epoch will no longer match).
            if mode != RUN:
                s["kill_epoch"] = s.get("kill_epoch", 0) + 1

        self._modify(update)

    def begin_day(self, utc_day: str) -> None:
        """The UTC rollover never clears a human or incident halt."""
        if len(utc_day) != 10 or utc_day[4] != "-" or utc_day[7] != "-":
            raise OrderSafetyError("bad UTC date")

        def update(s: dict[str, Any]) -> None:
            if utc_day < s["utc_day"]:
                self._trip(s, "clock_rolled_back")
            elif utc_day > s["utc_day"]:
                s["utc_day"] = utc_day
                s["entries_today"] = 0

        self._modify(update)

    def prepare_order(
        self,
        symbol: str,
        side: str,
        decision_tick: str | int,
        qty: Any,
        *,
        max_entries_per_day: int = 10,
    ) -> str:
        amount = _positive(qty)
        cid = deterministic_order_id(symbol, side, decision_tick)

        def update(s: dict[str, Any]) -> str:
            if cid in s["orders"]:
                old = s["orders"][cid]
                if _qty(old["requested_qty"]) != amount:
                    raise OrderSafetyError("idempotent intent reused with different size")
                return cid  # no new reservation or POST authorization
            if not s["reconciled"] or s["uncertain"] or s["incident"]:
                raise OrderSafetyError("broker reconciliation required")
            if side == "buy":
                if s["mode"] != RUN:
                    raise OrderSafetyError("buy blocked by persistent halt")
                if s["entries_today"] >= max_entries_per_day:
                    raise OrderSafetyError("daily entry cap")
                if any(
                    o["symbol"] == symbol and o["status"] in ACTIVE for o in s["orders"].values()
                ):
                    raise OrderSafetyError("existing unresolved order for symbol")
                if _qty(s["positions"].get(symbol, "0")) + amount > self.max_symbol_qty:
                    raise OrderSafetyError("maximum symbol position exceeded")
            elif side == "sell":
                if s["mode"] == EMERGENCY_STOP:
                    raise OrderSafetyError("emergency mode requires operator-managed exits")
                if s["mode"] == CANCEL_AND_FLATTEN and any(
                    o["side"] == "buy" and o["status"] in ACTIVE for o in s["orders"].values()
                ):
                    raise OrderSafetyError("must cancel and reconcile buy orders before flattening")
                reserved = sum(
                    (
                        _qty(o["requested_qty"]) - _qty(o["filled_qty"])
                        for o in s["orders"].values()
                        if o["symbol"] == symbol and o["side"] == "sell" and o["status"] in ACTIVE
                    ),
                    Decimal(0),
                )
                if amount + reserved > _qty(s["positions"].get(symbol, "0")):
                    raise OrderSafetyError("sell exceeds broker-reconciled sellable quantity")
            s["orders"][cid] = {
                "client_order_id": cid,
                "symbol": symbol,
                "side": side,
                "requested_qty": _canonical(amount),
                "filled_qty": "0",
                "status": "PREPARED",
                "cancel_requested": False,
                "epoch": s.get("kill_epoch", 0),
                # minutes since epoch; lets the loop prove history coverage
                "decision_tick": decision_tick,
            }
            if side == "buy":
                s["entries_today"] += 1
            return cid

        return self._modify(update)

    def mark_submitting(self, client_order_id: str) -> None:
        """Persist uncertainty BEFORE invoking HTTP POST outside this class.

        On refusal the order status is persisted as ``CANCELED_LOCAL`` (it
        never reached the broker) and then an OrderSafetyError is raised.
        The CANCELED_LOCAL write is committed in its own transaction before
        the refusal is signalled, so callers can always audit the local
        decision regardless of how they handle the exception.
        """
        refusal: list[str | None] = [None]

        def update(s: dict[str, Any]) -> None:
            o = s["orders"].get(client_order_id)
            if not o or o["status"] != "PREPARED":
                raise OrderSafetyError("POST retry/refiring refused; lookup by client order id")
            # The kill_epoch / mode / incident gates run BEFORE the side
            # effects of the PREPARED precondition (status -> SUBMITTING) and
            # BEFORE the unknown_post_must_lookup_before_retry trip.
            side = o.get("side")
            if side == "buy":
                if (
                    s["mode"] != RUN
                    or s["uncertain"]
                    or s["incident"] != ""
                    or o.get("epoch") != s.get("kill_epoch", 0)
                ):
                    o["status"] = "CANCELED_LOCAL"  # never reached the broker
                    refusal[0] = "entry not authorized: halted or epoch changed"
                    return
            elif side == "sell":
                if s["mode"] == EMERGENCY_STOP:
                    o["status"] = "CANCELED_LOCAL"
                    refusal[0] = (
                        "automated sell submit blocked: operator-managed exits required"
                    )
                    return
            o["status"] = "SUBMITTING"
            self._trip(s, "unknown_post_must_lookup_before_retry")

        self._modify(update)
        if refusal[0] is not None:
            raise OrderSafetyError(refusal[0])

    def acknowledge_order(self, client_order_id: str, *, broker_status: str = "new") -> None:
        """An ACK does not clear the incident; full reconciliation still required."""
        if broker_status not in {"new", "accepted", "partially_filled"}:
            raise OrderSafetyError("unrecognized broker ACK status")

        def update(s: dict[str, Any]) -> None:
            o = s["orders"].get(client_order_id)
            if not o or o["status"] not in {"SUBMITTING", "ACKED", "PARTIAL"}:
                raise OrderSafetyError("unexpected acknowledgement")
            o["status"] = "PARTIAL" if _qty(o["filled_qty"]) else "ACKED"

        self._modify(update)

    def request_cancel(self, client_order_id: str) -> None:
        """The BUY stays reserved until terminal cancellation plus reconciliation."""

        def update(s: dict[str, Any]) -> None:
            o = s["orders"].get(client_order_id)
            if not o or o["status"] not in ACTIVE:
                raise OrderSafetyError("cannot cancel non-active order")
            o["cancel_requested"] = True
            o["status"] = "CANCEL_REQUESTED"

        self._modify(update)

    def confirm_canceled(self, client_order_id: str) -> None:
        def update(s: dict[str, Any]) -> None:
            o = s["orders"].get(client_order_id)
            if not o or not o["cancel_requested"] or o["status"] not in ACTIVE:
                raise OrderSafetyError("cancellation was not requested or already terminal")
            o["status"] = "CANCELED"
            self._trip(s, "cancel_confirmed_requires_broker_reconciliation")

        self._modify(update)

    def record_fill(
        self,
        client_order_id: str,
        execution_id: str,
        filled_qty: Any,
        *,
        fee_in_asset: Any = "0",
    ) -> bool:
        """Record an independent execution once; preserve even unexpected fills.

        An execution is **not** a client order ID. Duplicate execution IDs are
        idempotent, but differing IDs for the same order count as new fills.
        """
        quantity, fee = _positive(filled_qty), _qty(fee_in_asset)
        if not execution_id:
            raise OrderSafetyError("missing execution identity")

        def update(s: dict[str, Any]) -> bool:
            identity = {
                "client_order_id": client_order_id,
                "qty": _canonical(quantity),
                "asset_fee": _canonical(fee),
            }
            if execution_id in s["execution_details"]:
                if s["execution_details"][execution_id] != identity:
                    self._trip(s, "conflicting_replayed_execution_id")
                return False
            s["execution_details"][execution_id] = identity
            o = s["orders"].get(client_order_id)
            if o is None:
                self._trip(s, "unexpected_broker_fill_unknown_order")
                # Safe storage of unknown fill metadata (not silently dropped).
                s["seen_executions"].append(execution_id)
                s["unexpected_fills"].append(identity)
                return True
            previous_status = o["status"]
            new_filled = _qty(o["filled_qty"]) + quantity
            o["filled_qty"] = _canonical(new_filled)
            o["status"] = "FILLED" if new_filled >= _qty(o["requested_qty"]) else "PARTIAL"
            symbol = o["symbol"]
            position = _qty(s["positions"].get(symbol, "0"))
            if o["side"] == "buy":
                if fee > quantity:
                    self._trip(s, "asset_fee_exceeds_filled_quantity")
                position += quantity - fee
            else:
                if position < quantity:
                    self._trip(s, "sell_fill_exceeds_known_holdings")
                    position = Decimal(0)
                else:
                    position -= quantity
                if fee:
                    self._trip(s, "unexpected_asset_fee_on_sell")
            s["positions"][symbol] = _canonical(max(Decimal(0), position))
            s["seen_executions"].append(execution_id)
            if previous_status in TERMINAL or new_filled > _qty(o["requested_qty"]):
                self._trip(s, "late_or_oversize_fill_requires_reconciliation")
            else:
                # HARD incidents must NOT be silently overwritten by a routine
                # reconciliation flag; operator review is required to clear them.
                s["reconciled"] = False  # position changed; refresh broker truth
                s["uncertain"] = True
                if s["mode"] == RUN:
                    s["mode"] = HALT_ENTRIES
                if not self._is_hard_incident(s["incident"]):
                    s["incident"] = "execution_requires_reconciliation"
            return True

        return self._modify(update)

    def reconcile(
        self,
        broker_positions: dict[str, Any],
        broker_orders: list[dict[str, Any]],
        *,
        account_active: bool,
        complete_snapshot: bool,
    ) -> bool:
        """Fail closed if broker's complete snapshot does not match local state.

        Caller MUST supply complete known-order history (including closed
        orders) and current positions, not merely an unverified empty page.
        Open-order-only or incomplete pagination sets complete_snapshot=False.
        """
        positions = {k: _canonical(_qty(v)) for k, v in broker_positions.items()}
        seen: dict[str, dict[str, Any]] = {}
        for raw in broker_orders:
            cid = raw.get("client_order_id")
            if not isinstance(cid, str) or not cid or cid in seen:
                raise OrderSafetyError("missing or duplicated broker order ID")
            seen[cid] = raw

        def update(s: dict[str, Any]) -> bool:
            if not account_active or not complete_snapshot:
                self._trip(s, "incomplete_broker_truth_or_inactive_account")
                return False
            mismatch = False
            for symbol in set(positions) | set(s["positions"]):
                if not _qty_equal(positions.get(symbol, "0"), s["positions"].get(symbol, "0")):
                    mismatch = True
            for cid, order in s["orders"].items():
                b = seen.get(cid)
                if b is None:
                    # PREPARED is persisted before SUBMITTING, so it was never POSTed.
                    if order["status"] not in TERMINAL and order["status"] != "PREPARED":
                        mismatch = True  # a timeout is NEVER proof of nonexistence
                    continue
                if b.get("symbol") != order["symbol"] or b.get("side") != order["side"]:
                    mismatch = True
                if not _qty_equal(b.get("filled_qty", "0"), order["filled_qty"]):
                    mismatch = True  # missing fee/fill event; must reconstruct
                status = str(b.get("status", "")).lower()
                if status in {"filled", "canceled", "expired", "rejected"}:
                    expected = (
                        "FILLED"
                        if status == "filled"
                        else "CANCELED"
                        if status in {"canceled", "expired"}
                        else "REJECTED"
                    )
                    if order["status"] != expected:
                        if (
                            order["status"] in ACTIVE
                            and not order.get("cancel_requested")
                            and _qty_equal(b.get("filled_qty", "0"), order["filled_qty"])
                        ):
                            # Broker truth is final and the fills already match
                            # (e.g. a notional buy whose size estimate exceeded the
                            # actual fill): adopt the broker's terminal state.
                            # Never after a cancel request (late fills stay a
                            # mismatch for operator review).
                            order["status"] = expected
                        else:
                            mismatch = True
                elif status in {
                    "new",
                    "accepted",
                    "partially_filled",
                    "pending_cancel",
                    "accepted_for_bidding",
                    "pending_new",
                }:
                    if order["status"] in TERMINAL:
                        mismatch = True
                else:
                    mismatch = True
            for cid, order in seen.items():
                if cid not in s["orders"] and str(order.get("status", "")).lower() not in {
                    "filled",
                    "canceled",
                    "expired",
                    "rejected",
                }:
                    mismatch = True  # unknown pending orders could still fill
            if mismatch:
                self._trip(s, "broker_order_or_position_mismatch")
                return False
            s["reconciled"] = True
            s["uncertain"] = False
            # A mismatch or late fill must not be silently cleared merely by
            # a later snapshot; operator must explicitly resolve incident.
            # Only CLEARABLE_INCIDENTS are auto-cleared on a clean snapshot;
            # HARD incidents require clear_incident(operator_approved=True).
            if s["incident"] in CLEARABLE_INCIDENTS:
                s["incident"] = ""
            return True

        return self._modify(update)

    def clear_incident(self, *, operator_approved: bool = False) -> None:
        if not operator_approved:
            raise OrderSafetyError("clearing a safety incident requires operator approval")

        def update(s: dict[str, Any]) -> None:
            if not s["reconciled"] or s["uncertain"]:
                raise OrderSafetyError("cannot clear without clean broker reconciliation")
            s["incident"] = ""

        self._modify(update)

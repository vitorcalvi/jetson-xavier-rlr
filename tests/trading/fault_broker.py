"""Fault-injecting fake broker for the negative integration tests.

The :class:`FaultBroker` mirrors the broker protocol the trader talks to in
production:

* :meth:`positions`, :meth:`open_orders`, :meth:`orders_since`,
  :meth:`get_order_by_client_id` -- the four methods the
  :class:`~reflective_loop_runtime.trading.orders.OrderController` prelude
  consumes.
* :meth:`submit` -- the order POST the production adapter performs.
* :meth:`adapter` -- returns a callable that matches the
  ``(alpaca_paper, order)`` adapter signature used by the
  :class:`~reflective_loop_runtime.kernel.toolbroker.ToolBroker` so the
  trader can drive it through its real permit gate.

Knobs (all default ``False`` / ``None`` / ``0``):

* ``raise_on_history`` -- ``positions``/``open_orders``/``orders_since``
  raise ``RuntimeError("order history incomplete...")`` (a broker outage).
* ``submit_raises_after_accept`` -- ``submit`` records the order in the
  broker's lookup table and then raises; the controller recovers via
  :meth:`get_order_by_client_id`. Exactly one POST is observed.
* ``partial_fill_qty`` -- the broker reports ``status="partially_filled"``
  with ``filled_qty=partial_fill_qty`` for every new order. Lets the tests
  pin the partial-fill reconcile behaviour.
* ``cancel_ack_delay_ticks`` -- the broker keeps a cancelled order in a
  non-terminal state until the test calls :meth:`confirm_cancel_at_broker`.
* ``late_fill_after_cancel`` -- after the broker has acknowledged a cancel,
  the *next* :meth:`orders_since` reports the (formerly cancelled) order
  as ``filled``; the controller must trip on this and never let entries
  through.

Every submit call is recorded in :attr:`submit_calls` so the tests can
assert "exactly one broker order" without breaking the broker abstraction.

Symbol canonicalisation mirrors :mod:`reflective_loop_runtime.trading.broker`:
``"BTCUSD"`` on the wire is rewritten to ``"BTC/USD"`` so a ``reconcile``
that compares the controller's ledger symbol against the broker's view
does not trip a spurious mismatch.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from reflective_loop_runtime.trading.broker import canonical_symbol


def _canonicalise(record: dict[str, Any]) -> dict[str, Any]:
    """Return ``record`` with its ``"symbol"`` key rewritten through canonical_symbol."""
    if isinstance(record.get("symbol"), str):
        record["symbol"] = canonical_symbol(record["symbol"])
    return record


@dataclass
class FaultBroker:
    """A scriptable fake broker that the OrderController can be pointed at."""

    # -- knobs -------------------------------------------------------------
    raise_on_history: bool = False
    submit_raises_after_accept: bool = False
    partial_fill_qty: float | None = None
    cancel_ack_delay_ticks: int = 0
    late_fill_after_cancel: bool = False
    account_active: bool = True
    account_status_raises: bool = False

    # -- seeding (kept verbatim unless the test mutates _lookup) ------------
    initial_positions: list[dict[str, Any]] = field(default_factory=list)
    initial_orders: list[dict[str, Any]] = field(default_factory=list)
    initial_lookup: dict[str, dict[str, Any]] = field(default_factory=dict)

    # -- mutable runtime state ---------------------------------------------
    _submit_calls: list[dict[str, Any]] = field(default_factory=list)
    _positions: list[dict[str, Any]] = field(default_factory=list)
    _open_orders: list[dict[str, Any]] = field(default_factory=list)
    _lookup: dict[str, dict[str, Any]] = field(default_factory=dict)
    _late_fills_published: set[str] = field(default_factory=set)
    _post_cancel_history: dict[str, bool] = field(default_factory=dict)
    _ticks: int = 0

    def __post_init__(self) -> None:
        self._positions = [_canonicalise(dict(p)) for p in self.initial_positions]
        self._open_orders = [_canonicalise(dict(o)) for o in self.initial_orders]
        self._lookup = {k: _canonicalise(dict(v)) for k, v in self.initial_lookup.items()}
        self._submit_calls = []
        self._late_fills_published = set()
        self._post_cancel_history = {}
        self._ticks = 0
        # Every fake order dict must carry a client_order_id so the
        # controller's "unknown_open_order" check has something to match.
        for entry in self._open_orders:
            assert entry.get("client_order_id"), (
                f"FaultBroker initial_order missing client_order_id: {entry!r}"
            )
        for cid, entry in self._lookup.items():
            assert entry.get("client_order_id", cid), (
                f"FaultBroker initial_lookup entry missing client_order_id: {entry!r}"
            )

    # -- helpers used by tests to drive scenario state --------------------
    def advance_tick(self) -> None:
        """Advance the broker's view of time by one loop tick."""
        self._ticks += 1

    def confirm_cancel_at_broker(self, client_order_id: str) -> None:
        """Mark an order as ``canceled`` from the broker's perspective.

        Until this is called the broker still reports the order in a
        non-terminal state (i.e. simulates a delayed cancel ACK).
        """
        rec = self._lookup.get(client_order_id)
        if rec is None:
            return
        rec["status"] = "canceled"
        rec["filled_qty"] = "0"
        self._post_cancel_history[client_order_id] = True
        if self.late_fill_after_cancel:
            # Publish a late fill on the next orders_since/positions query.
            self._late_fills_published.add(client_order_id)

    def publish_late_fill(self, client_order_id: str) -> None:
        """Force-publish a late fill for an order that the broker already cancelled."""
        rec = self._lookup.get(client_order_id)
        if rec is None:
            return
        rec["status"] = "filled"
        # Use a partial qty so subsequent tests can still reason about it.
        rec["filled_qty"] = "0.5"
        rec["filled_avg_price"] = "100.0"
        self._late_fills_published.add(client_order_id)

    @property
    def submit_calls(self) -> list[dict[str, Any]]:
        """Read-only snapshot of every submit attempt the loop has made."""
        return list(self._submit_calls)

    # -- broker protocol the controller prelude uses -----------------------
    def positions(self) -> list[dict[str, Any]]:
        if self.raise_on_history:
            raise RuntimeError("order history incomplete: positions endpoint unavailable")
        return [_canonicalise(dict(p)) for p in self._positions]

    def open_orders(self) -> list[dict[str, Any]]:
        if self.raise_on_history:
            raise RuntimeError("order history incomplete: open orders endpoint unavailable")
        return [_canonicalise(dict(o)) for o in self._open_orders]

    def account_status(self) -> dict[str, Any]:
        """Return the Alpaca-style account status snapshot.

        Mirrors the real broker's response, including the
        ``active`` flag the prelude uses to gate ``account_active`` and
        a ``crypto_status`` so any future crypto gating has a stub.
        """
        if self.account_status_raises:
            raise RuntimeError("account status endpoint unavailable")
        return {
            "active": bool(self.account_active),
            "status": "ACTIVE" if self.account_active else "INACTIVE",
            "crypto_status": "ACTIVE" if self.account_active else "INACTIVE",
        }

    def orders_since(
        self,
        after_iso: str,
        *,
        page_limit: int = 500,
        max_pages: int = 50,
    ) -> list[dict[str, Any]]:
        if self.raise_on_history:
            raise RuntimeError("order history incomplete: orders_since unavailable")
        # Surface both the live lookup and any late fills that have been
        # published since the last tick, so a reconcile detects them.
        out: list[dict[str, Any]] = []
        for cid, rec in self._lookup.items():
            snapshot = _canonicalise(dict(rec))
            if cid in self._late_fills_published:
                snapshot["status"] = "filled"
                snapshot["filled_qty"] = "0.5"
            out.append(snapshot)
        # Drop late-fill flags once the controller has had a chance to see
        # them; subsequent reconciles must keep the fill visible (rec["status"]
        # already pinned above).
        return out

    def get_order_by_client_id(self, cid: str) -> dict[str, Any] | None:
        rec = self._lookup.get(cid)
        return _canonicalise(dict(rec)) if rec is not None else None

    # -- broker protocol the order adapter uses ----------------------------
    def submit(
        self,
        symbol: str,
        side: str,
        *,
        client_order_id: str,
        notional: float | None = None,
        qty: float | None = None,
    ) -> dict[str, Any]:
        call = {
            "symbol": symbol,
            "side": side,
            "client_order_id": client_order_id,
            "notional": notional,
            "qty": qty,
        }
        self._submit_calls.append(call)

        order_id = f"fault-{len(self._submit_calls)}"
        if self.partial_fill_qty is not None:
            status = "partially_filled"
            filled_qty = float(self.partial_fill_qty)
        else:
            status = "filled"
            if qty is not None:
                filled_qty = float(qty)
            elif notional is not None:
                filled_qty = float(notional) / 100.0
            else:
                filled_qty = 0.0
        rec: dict[str, Any] = {
            "id": order_id,
            "client_order_id": client_order_id,
            "symbol": symbol,
            "side": side,
            "status": status,
            "filled_qty": str(filled_qty),
            "filled_avg_price": "100.0",
        }
        self._lookup[client_order_id] = rec
        # Keep open_orders in sync so open_orders() based reconciles see it.
        self._open_orders.append(dict(rec))
        if self.submit_raises_after_accept:
            raise RuntimeError("submit failed after the broker accepted the order")
        return rec

    def adapter(self) -> Callable[[Mapping[str, object]], dict[str, object]]:
        """Return an ``(alpaca_paper, order)`` adapter closing over this broker."""
        broker = self

        def _adapter(args: Mapping[str, object]) -> dict[str, object]:
            rec = broker.submit(
                str(args["symbol"]),
                str(args["side"]),
                client_order_id=str(args["client_order_id"]),
                notional=args.get("notional"),
                qty=args.get("qty"),
            )
            return {
                "order_id": str(rec["id"]),
                "status": str(rec["status"]),
                "filled_qty": float(rec["filled_qty"]),
                "filled_avg_price": float(rec["filled_avg_price"]),
                "client_order_id": str(rec["client_order_id"]),
            }

        return _adapter

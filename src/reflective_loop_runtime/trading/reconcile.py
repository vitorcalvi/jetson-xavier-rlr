"""DIAGNOSTIC ONLY; OrderController.reconcile is the single authoritative
gate; never use .ok to authorize orders.

Broker-truth reconciliation for the paper trader (P0 safety).

The paper trader keeps its own in-memory ledger of positions but executes against
a remote broker (Alpaca paper).  Before opening a new entry the runtime MUST
prove that the broker and the ledger agree on the current position state, or at
least surface the disagreement explicitly.

.. warning::

   This module is **DIAGNOSTIC ONLY**. The single authoritative reconciliation
   gate is :meth:`OrderController.reconcile`. Callers MUST NOT use
   :attr:`ReconcileResult.ok` (or any other field of the value this function
   returns) to authorize new orders. ``reconcile()`` here is a thin helper that
   mirrors the broker state into a frozen dataclass for telemetry and
   debugging; the live trading lane goes through ``OrderController.reconcile``
   which owns its own fail-closed semantics.

Design contract (read this before changing anything):

* ``reconcile()`` MUST fail closed. Any exception from the broker (network
  error, auth error, malformed response, missing method, ``open_orders``
  unavailable) returns an :class:`ReconcileResult` with ``ok=False`` and
  ``uncertain=True``; the diagnostic surface treats that as a hard stop.
* ``open_orders`` is **not** best-effort. A broker that has no
  ``open_orders`` method, or whose ``open_orders()`` call raises, must
  surface ``reason='open_orders_unavailable'`` and mark the result
  uncertain. We cannot certify the ledger against an unknown order book.
* The function is pure given a well-behaved broker: no side effects beyond
  the broker calls it makes (``positions`` and ``open_orders``).
* The result is hashable (``frozen=True``) so callers can cache it in
  policy state without worrying about mutation.

Symbol normalisation: the broker uses Alpaca's ``BTCUSD`` form, the ledger
uses ``BTC/USD``. We strip the slash on both sides so they compare equal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Fee-in-kind rate for Alpaca crypto: 0.25% on each side, so the broker fills
# 0.25% less than the notional/qty the ledger expected.  The ledger records the
# pre-fee quantity, the broker reports the post-fee quantity, so a small
# shortfall on the broker side is expected and not a real disagreement.
_FEE_IN_KIND_RATE = 0.0026


def _normalise_symbol(symbol: str) -> str:
    """Return ``symbol`` with all ``/`` removed (e.g. ``BTC/USD`` -> ``BTCUSD``)."""
    return symbol.replace("/", "")


@dataclass(frozen=True)
class ReconcileResult:
    """Outcome of a broker vs. ledger reconciliation pass.

    Attributes
    ----------
    ok:
        True iff the reconciliation is clean: ``uncertain`` is False.  A result
        with reasons but no uncertainty (e.g. ``dust`` or ``fee_in_kind``) is
        still ``ok`` because the diagnostic surface can proceed.
    uncertain:
        True iff the runtime should refuse to open new entries.  Examples:
        untracked broker position, missing ledger position, qty mismatch, open
        order not yet accounted for, broker error, open_orders unavailable.
    reasons:
        Human-readable, machine-parseable tags for every disagreement found.
        Empty when the broker and ledger agree exactly. Includes
        ``open_orders_unavailable`` when the broker has no ``open_orders``
        method or the call raised.
    broker_positions:
        Symbol -> quantity as reported by the broker (normalised).
    ledger_positions:
        Symbol -> quantity as recorded by the ledger (normalised).
    open_orders:
        Snapshot of ``broker.open_orders()`` at the time of the call.
        Empty when the call failed or the broker does not expose the method;
        in either case ``'open_orders_unavailable'`` is in ``reasons`` and
        ``uncertain`` is True.
    """

    ok: bool
    uncertain: bool
    reasons: tuple[str, ...]
    broker_positions: dict[str, float]
    ledger_positions: dict[str, float]
    open_orders: tuple[dict, ...]


def reconcile(
    broker: Any,
    ledger_positions: dict[str, float],
    *,
    qty_tol: float = 1e-9,
    dust_usd: float = 1.0,
    prices: dict[str, float] | None = None,
) -> ReconcileResult:
    """Compare ``broker.positions()`` against ``ledger_positions``.

    Parameters
    ----------
    broker:
        Any object exposing ``positions()`` (returning a list of dicts with
        ``symbol`` and ``qty`` keys) and ``open_orders()`` (returning a list
        of open-order dicts).
    ledger_positions:
        Mapping from ledger symbol (e.g. ``BTC/USD``) to quantity.  Symbols
        are normalised to strip ``/`` before comparison.
    qty_tol:
        Absolute quantity tolerance for a clean match.  Defaults to ``1e-9``
        which is well below any sub-satoshi precision.
    dust_usd:
        Maximum notional difference (in USD) to accept as harmless dust.  A
        dust position is reported as ``dust:<sym>`` but does NOT mark the
        result uncertain.
    prices:
        Optional symbol -> USD price map.  When supplied, the dust check uses
        ``|broker - ledger| * price`` as the notional difference.

    Returns
    -------
    ReconcileResult
        Always returns a result; never raises.  Broker errors become a
        ``broker_error:<ExcType>`` reason with ``ok=False, uncertain=True``.
        A missing or failing ``open_orders`` method becomes
        ``open_orders_unavailable`` with ``ok=False, uncertain=True`` -- the
        function never silently accepts an unknown order book.
    """
    norm_ledger: dict[str, float] = {
        _normalise_symbol(sym): float(qty) for sym, qty in ledger_positions.items()
    }

    try:
        raw_positions = broker.positions()
    except Exception as exc:  # noqa: BLE001 - we want to catch any broker failure
        # Fail closed.  Never assume the world is flat when the broker is
        # unreachable or returns garbage.
        return ReconcileResult(
            ok=False,
            uncertain=True,
            reasons=(f"broker_error:{type(exc).__name__}",),
            broker_positions={},
            ledger_positions=norm_ledger,
            open_orders=(),
        )

    broker_positions: dict[str, float] = {}
    for entry in raw_positions or ():
        try:
            sym = _normalise_symbol(str(entry["symbol"]))
            qty = float(entry["qty"])
        except (KeyError, TypeError, ValueError):
            # Malformed entry: count as a broker error, fail closed.
            return ReconcileResult(
                ok=False,
                uncertain=True,
                reasons=(f"broker_error:{type(entry).__name__}",),
                broker_positions=broker_positions,
                ledger_positions=norm_ledger,
                open_orders=(),
            )
        broker_positions[sym] = broker_positions.get(sym, 0.0) + qty

    # Open orders are MANDATORY for a clean reconciliation: an unfilled
    # order sitting in the broker's book means the ledger may not yet
    # reflect the corresponding fill. If we cannot see the order book we
    # cannot certify the ledger against it, so we fail closed with
    # ``reason='open_orders_unavailable'``. The previous best-effort
    # behaviour (treating missing ``open_orders`` as empty) was unsafe:
    # an untracked open order could push us past intended exposure while
    # this module reported ``ok=True``.
    open_orders: tuple[dict, ...] = ()
    open_orders_unavailable = False
    if hasattr(broker, "open_orders") and callable(broker.open_orders):
        try:
            raw_open_orders = broker.open_orders() or []
        except Exception:  # noqa: BLE001
            raw_open_orders = []
            open_orders_unavailable = True
        else:
            open_orders = tuple(raw_open_orders)
    else:
        open_orders_unavailable = True

    # The "trading universe" is the set of symbols we are exposed to: every
    # symbol the broker holds OR the ledger tracks.  Open orders outside this
    # universe are ignored here (they will be caught by entry-side guards).
    universe: set[str] = set(broker_positions) | set(norm_ledger)

    reasons: list[str] = []
    uncertain = open_orders_unavailable

    if open_orders_unavailable:
        reasons.append("open_orders_unavailable")

    # Open orders: an unfilled order on a symbol we are trading means the
    # ledger may not yet reflect the fill.  Refuse to add new exposure on top
    # of an unexplained open order.
    for order in open_orders:
        order_sym = _normalise_symbol(str(order.get("symbol", "")))
        if order_sym and order_sym in universe:
            reasons.append(f"open_order:{order_sym}")
            uncertain = True

    all_symbols: set[str] = set(broker_positions) | set(norm_ledger)
    for sym in sorted(all_symbols):
        broker_qty = broker_positions.get(sym, 0.0)
        ledger_qty = norm_ledger.get(sym, 0.0)
        diff = broker_qty - ledger_qty

        if abs(diff) <= qty_tol:
            # Clean match.
            continue

        # Dust tolerance: small notional differences are common on crypto
        # exchanges (rounding, residual fills).  When prices are supplied we
        # compare in notional; otherwise we cannot judge dust and skip.
        if prices is not None and sym in prices:
            notional_diff = abs(diff) * float(prices[sym])
            if notional_diff < dust_usd:
                reasons.append(f"dust:{sym}")
                continue

        # Fee-in-kind: the ledger records the pre-fee quantity, so a small
        # shortfall on the broker side is expected.  Only applies when the
        # ledger actually expected a position.
        if ledger_qty > 0.0:
            lower = ledger_qty * (1.0 - _FEE_IN_KIND_RATE)
            if lower <= broker_qty <= ledger_qty:
                reasons.append(f"fee_in_kind:{sym}")
                continue

        if broker_qty > 0.0 and ledger_qty == 0.0:
            reasons.append(f"untracked_position:{sym}")
            uncertain = True
        elif ledger_qty > 0.0 and broker_qty == 0.0:
            reasons.append(f"missing_position:{sym}")
            uncertain = True
        else:
            reasons.append(f"qty_mismatch:{sym}:{broker_qty}:{ledger_qty}")
            uncertain = True

    return ReconcileResult(
        ok=not uncertain,
        uncertain=uncertain,
        reasons=tuple(reasons),
        broker_positions=broker_positions,
        ledger_positions=norm_ledger,
        open_orders=open_orders,
    )

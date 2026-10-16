"""Tests for :mod:`reflective_loop_runtime.trading.reconcile`.

DIAGNOSTIC ONLY; OrderController.reconcile is the single authoritative gate;
never use .ok to authorize orders.

These cover the named cases the gate requires plus a few defensive checks
around symbol normalisation and result shape.
"""

from __future__ import annotations

import dataclasses
import unittest

from reflective_loop_runtime.trading.reconcile import (
    ReconcileResult,
    reconcile,
)


class _StubBroker:
    """Minimal stand-in for :class:`PaperBroker` used by the reconcile tests.

    The real broker speaks HTTP; here we just hand ``reconcile()`` whatever
    data we want it to see.  Setting ``raise_on_positions`` makes the next
    ``positions()`` call raise, which is the easiest way to test the
    fail-closed broker-error path. Setting ``raise_on_open_orders`` makes
    the next ``open_orders()`` call raise, which is the easiest way to
    test the ``open_orders_unavailable`` path.
    """

    def __init__(
        self,
        positions=None,
        open_orders=None,
        raise_on_positions=None,
        raise_on_open_orders=None,
    ) -> None:
        self._positions = positions if positions is not None else []
        self._open_orders = open_orders if open_orders is not None else []
        self._raise_on_positions = raise_on_positions
        self._raise_on_open_orders = raise_on_open_orders

    def positions(self):
        if self._raise_on_positions is not None:
            raise self._raise_on_positions
        return self._positions

    def open_orders(self):
        if self._raise_on_open_orders is not None:
            raise self._raise_on_open_orders
        return self._open_orders


class _NoOpenOrdersBroker:
    """Broker that has no ``open_orders`` method at all (legacy shape).

    Under the DIAGNOSTIC ONLY contract, a missing ``open_orders`` method
    marks the reconciliation uncertain: we cannot see the broker's order
    book and therefore cannot certify the ledger.
    """

    def __init__(self, positions=None) -> None:
        self._positions = positions if positions is not None else []

    def positions(self):
        return self._positions


class TestReconcile(unittest.TestCase):
    def test_flat_both_ok(self) -> None:
        broker = _StubBroker(positions=[])
        result = reconcile(broker, {})
        self.assertIsInstance(result, ReconcileResult)
        self.assertTrue(result.ok)
        self.assertFalse(result.uncertain)
        self.assertEqual(result.reasons, ())
        self.assertEqual(result.broker_positions, {})
        self.assertEqual(result.ledger_positions, {})
        self.assertEqual(result.open_orders, ())

    def test_fee_in_kind_tolerated(self) -> None:
        # Ledger recorded 1.0 BTC pre-fee; broker reports 0.9974 post-fee.
        # 0.9974 sits inside [0.9974, 1.0] exactly so it is fee_in_kind, not
        # qty_mismatch.
        broker = _StubBroker(positions=[{"symbol": "BTCUSD", "qty": "0.9974"}])
        result = reconcile(broker, {"BTC/USD": 1.0})
        self.assertTrue(result.ok)
        self.assertFalse(result.uncertain)
        self.assertIn("fee_in_kind:BTCUSD", result.reasons)

    def test_untracked_position_uncertain(self) -> None:
        broker = _StubBroker(positions=[{"symbol": "BTCUSD", "qty": "0.5"}])
        result = reconcile(broker, {})
        self.assertFalse(result.ok)
        self.assertTrue(result.uncertain)
        self.assertIn("untracked_position:BTCUSD", result.reasons)

    def test_missing_position_uncertain(self) -> None:
        broker = _StubBroker(positions=[])
        result = reconcile(broker, {"BTC/USD": 0.5})
        self.assertFalse(result.ok)
        self.assertTrue(result.uncertain)
        self.assertIn("missing_position:BTCUSD", result.reasons)

    def test_open_order_uncertain(self) -> None:
        broker = _StubBroker(
            positions=[],
            open_orders=[{"symbol": "BTCUSD", "side": "buy", "qty": "0.1"}],
        )
        # Ledger already accounts for the expected position, but the open
        # order is unfilled and sits in the trading universe.
        result = reconcile(broker, {"BTC/USD": 0.1})
        self.assertFalse(result.ok)
        self.assertTrue(result.uncertain)
        self.assertIn("open_order:BTCUSD", result.reasons)

    def test_broker_error_fails_closed(self) -> None:
        broker = _StubBroker(raise_on_positions=ConnectionError("boom"))
        result = reconcile(broker, {"BTC/USD": 1.0})
        self.assertFalse(result.ok)
        self.assertTrue(result.uncertain)
        self.assertEqual(result.reasons, ("broker_error:ConnectionError",))
        # The ledger is still reported (we did parse it before the call),
        # but the broker side is empty because the call failed.
        self.assertEqual(result.broker_positions, {})
        self.assertEqual(result.ledger_positions, {"BTCUSD": 1.0})
        self.assertEqual(result.open_orders, ())

    def test_dust_allowed(self) -> None:
        # Broker holds 0.00005 BTC, ledger knows nothing.  At $10_000/BTC the
        # notional difference is $0.50, below the default $1.00 dust floor.
        broker = _StubBroker(positions=[{"symbol": "BTCUSD", "qty": "0.00005"}])
        result = reconcile(broker, {}, prices={"BTCUSD": 10_000.0}, dust_usd=1.0)
        self.assertTrue(result.ok)
        self.assertFalse(result.uncertain)
        self.assertIn("dust:BTCUSD", result.reasons)
        # And without prices, the same position is untracked (uncertain).
        no_price = reconcile(broker, {})
        self.assertTrue(no_price.uncertain)
        self.assertIn("untracked_position:BTCUSD", no_price.reasons)

    def test_symbol_normalisation(self) -> None:
        # Broker uses Alpaca's "BTCUSD"; ledger uses "BTC/USD".  The qty is
        # the same so reconciliation must be clean.
        broker = _StubBroker(positions=[{"symbol": "BTCUSD", "qty": "0.25"}])
        result = reconcile(broker, {"BTC/USD": 0.25})
        self.assertTrue(result.ok)
        self.assertFalse(result.uncertain)
        self.assertEqual(result.reasons, ())
        # Both sides are normalised in the result.
        self.assertEqual(result.broker_positions, {"BTCUSD": 0.25})
        self.assertEqual(result.ledger_positions, {"BTCUSD": 0.25})

    def test_legacy_broker_without_open_orders(self) -> None:
        # If the broker has no open_orders method we MUST fail closed under
        # the DIAGNOSTIC ONLY contract: an unknown order book cannot be
        # reconciled against, so ok=False and uncertain=True.
        broker = _NoOpenOrdersBroker(positions=[])
        result = reconcile(broker, {})
        self.assertFalse(result.ok)
        self.assertTrue(result.uncertain)
        self.assertIn("open_orders_unavailable", result.reasons)
        self.assertEqual(result.open_orders, ())

    def test_qty_mismatch_uncertain(self) -> None:
        # Genuine disagreement: ledger says 1.0, broker says 0.5.  This is
        # outside the fee-in-kind window and not dust, so it must be a hard
        # qty_mismatch that marks the result uncertain.
        broker = _StubBroker(positions=[{"symbol": "BTCUSD", "qty": "0.5"}])
        result = reconcile(broker, {"BTC/USD": 1.0})
        self.assertFalse(result.ok)
        self.assertTrue(result.uncertain)
        self.assertIn("qty_mismatch:BTCUSD:0.5:1.0", result.reasons)

    def test_open_order_outside_universe_ignored(self) -> None:
        # An open order on a symbol we don't trade must not poison the
        # reconciliation -- but with no ledger exposure, no broker position,
        # and a clean open_orders call, the result is still ok.
        broker = _StubBroker(
            positions=[],
            open_orders=[{"symbol": "ETHUSD", "qty": "1"}],
        )
        result = reconcile(broker, {})
        self.assertTrue(result.ok)
        self.assertFalse(result.uncertain)
        self.assertEqual(result.reasons, ())

    def test_result_is_frozen(self) -> None:
        # The dataclass is frozen, so callers (and policy caches) can rely
        # on the result not being mutated.
        result = reconcile(_StubBroker(), {})
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.ok = False  # type: ignore[misc]

    def test_reconcile_open_orders_error_is_uncertain(self) -> None:
        # ``broker.open_orders()`` raising is treated exactly like a missing
        # method: the result is uncertain with reason
        # ``open_orders_unavailable``. The previous best-effort behaviour was
        # unsafe because an untracked open order could push exposure past
        # the intended limit while ``ok`` still read True.
        broker = _StubBroker(
            positions=[{"symbol": "BTCUSD", "qty": "0.5"}],
            raise_on_open_orders=ConnectionError("open-orders-down"),
        )
        result = reconcile(broker, {"BTC/USD": 0.5})
        self.assertFalse(result.ok)
        self.assertTrue(result.uncertain)
        self.assertIn("open_orders_unavailable", result.reasons)
        # open_orders snapshot is empty and the broker positions still
        # surface (the failure was scoped to open_orders, not positions).
        self.assertEqual(result.open_orders, ())
        self.assertEqual(result.broker_positions, {"BTCUSD": 0.5})


if __name__ == "__main__":
    unittest.main()

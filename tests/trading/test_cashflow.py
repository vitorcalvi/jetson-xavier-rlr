"""Tests for :mod:`reflective_loop_runtime.trading.cashflow`.

The contract under test is fee-accurate FIFO PnL from Alpaca activities: an
in-kind buy fee must be valued at the *buy* price (never the sell price);
the sell fee comes from the sell CFEE ``net_amount`` (or the sell fill's
implied fee); the bid-ask spread is never separately debited; partial fills
match FIFO; unmatched remainder goes to ``open_lots``.

The named tests below are the contract -- renaming a test is a breaking
change because the task brief enumerates them by name.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from reflective_loop_runtime.trading.cashflow import (
    Activity,
    RoundTrip,
    parse_activities,
    round_trips,
    summary,
)

# --- helpers ---------------------------------------------------------------

_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "activities_sample.json"
)


def _row(
    activity_type: str = "FILL",
    id: str = "",
    symbol: str = "",
    side: str | None = None,
    qty: str | None = None,
    price: str | None = None,
    transaction_time: str = "",
    order_id: str | None = None,
    net_amount: str | None = None,
) -> dict:
    """Build an Alpaca activity row with the fields parse_activities reads.

    All parameters have defaults so the tests can call this helper either
    positionally (matches the Alpaca column order) or by keyword (used by
    the parse test for readability). Only ``net_amount`` is optional on the
    wire; the rest are made defaults here purely to keep call sites compact.
    """

    row: dict = {
        "id": id,
        "activity_type": activity_type,
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "price": price,
        "transaction_time": transaction_time,
        "order_id": order_id,
    }
    if net_amount is not None:
        row["net_amount"] = net_amount
    return row


# -- test_parse_fill_and_cfee -------------------------------------------------

def test_parse_fill_and_cfee() -> None:
    """``parse_activities`` normalises types, symbols and numeric fields.

    The activity rows used here exercise three layers of normalisation: the
    ``activity_type`` filter (only ``FILL``/``CFEE`` survive), the symbol
    rewrite (``"BTCUSD"`` -> ``"BTC/USD"``), and the Decimal coercion
    (string qty/price). ``side`` is preserved when present, ``None``
    otherwise. ``net_amount`` is left ``None`` on FILL rows and only
    populated on CFEE rows that report a USD fee.
    """

    rows = [
        _row(
            "FILL",
            id="f1",
            symbol="BTCUSD",  # no-slash form -> normalised
            side="buy",
            qty="0.001",
            price="50000.0",
            transaction_time="2026-10-11T18:00:00Z",
            order_id="o1",
        ),
        _row(
            "CFEE",
            id="c1",
            symbol="BTC/USD",
            side=None,
            qty="-0.0000025",
            price="0",
            transaction_time="2026-10-11T18:00:01Z",
            order_id="o1",
        ),
        _row(
            "DIV",  # unknown -> filtered
            id="d1",
            symbol="BTC/USD",
            side=None,
            qty="0",
            price="0",
            transaction_time="2026-10-11T18:00:02Z",
            order_id=None,
        ),
    ]
    acts = parse_activities(rows)
    assert len(acts) == 2
    # First record is the FILL.
    fill = acts[0]
    assert isinstance(fill, Activity)
    assert fill.type == "FILL"
    assert fill.id == "f1"
    assert fill.symbol == "BTC/USD"  # normalised
    assert fill.side == "buy"
    assert fill.qty == Decimal("0.001")
    assert fill.price == Decimal("50000.0")
    assert fill.net_amount is None
    assert fill.ts == "2026-10-11T18:00:00Z"
    assert fill.order_id == "o1"
    # Second record is the CFEE with in-kind fee.
    cfee = acts[1]
    assert cfee.type == "CFEE"
    assert cfee.side is None
    assert cfee.qty == Decimal("-0.0000025")
    assert cfee.net_amount is None


# -- test_fifo_round_trip_net -----------------------------------------------

def test_fifo_round_trip_net() -> None:
    """A clean buy-then-sell round trip computes net from cost/proceeds/fees.

    Buy 0.1 BTC at 50000 with no fees. Sell 0.1 BTC at 51000 with a 0.50
    USD fee. The expected cost is 5000, proceeds 5100, fees 0.50, net
    99.50. The test exercises both the FILL parser and the FIFO matcher
    against a hand-built activity list.
    """

    acts = parse_activities([
        _row(
            "FILL", id="b1", symbol="BTC/USD", side="buy",
            qty="0.1", price="50000",
            transaction_time="2026-10-11T18:00:00Z",
            order_id="o-buy",
        ),
        _row(
            "FILL", id="s1", symbol="BTC/USD", side="sell",
            qty="0.1", price="51000",
            transaction_time="2026-10-11T19:00:00Z",
            order_id="o-sell",
        ),
        _row(
            "CFEE", id="f1", symbol="BTC/USD", side="sell",
            qty="0", price="0", net_amount="-0.50",
            transaction_time="2026-10-11T19:00:01Z",
            order_id="o-sell",
        ),
    ])
    trips = round_trips(acts)
    assert len(trips) == 1
    trip = trips[0]
    assert isinstance(trip, RoundTrip)
    assert trip.symbol == "BTC/USD"
    assert trip.qty_sold == Decimal("0.1")
    assert trip.cost_usd == Decimal("5000.0")
    assert trip.proceeds_usd == Decimal("5100.0")
    assert trip.fees_usd == Decimal("0.5")
    assert trip.net_usd == Decimal("99.5")
    # net_pct = net / cost. 99.5 / 5000 = 0.0199.
    assert trip.net_pct == Decimal("99.5") / Decimal("5000.0")
    assert trip.entry_ts == "2026-10-11T18:00:00Z"
    assert trip.exit_ts == "2026-10-11T19:00:00Z"


# -- test_fee_in_kind_valued_at_buy_price ----------------------------------

def test_fee_in_kind_valued_at_buy_price() -> None:
    """An in-kind buy fee is valued at the buy price, not the sell price.

    If a future patch values the in-kind fee at the sell price instead, the
    round-trip cost will quietly grow whenever the sell price differs from
    the buy price. The test pins both directions: a profitable trade (sell
    price > buy price) and a losing trade (sell price < buy price) and
    asserts the fee is the in-kind qty times the *buy* price in both.
    """

    # Profitable case: buy 1 ETH at 2000 (in-kind fee 0.05 ETH),
    # sell 0.95 ETH at 2100, no USD sell fee.
    profit_acts = parse_activities([
        _row("FILL", "p-b", "ETH/USD", "buy",
             "1.0", "2000", "2026-10-11T10:00:00Z", "ob-p"),
        _row("CFEE", "p-c", "ETH/USD", None,
             "-0.05", "0", "2026-10-11T10:00:01Z", "ob-p"),
        _row("FILL", "p-s", "ETH/USD", "sell",
             "0.95", "2100", "2026-10-11T11:00:00Z", "os-p"),
    ])
    [p_trip] = round_trips(profit_acts)
    # Cost = 0.95 * 2000 = 1900. Proceeds = 0.95 * 2100 = 1995. Net =
    # 1995 - 1900 - 0.05*2000 = 95 - 100 = -5. The fee is the in-kind qty
    # at the *buy* price; valuing at the sell price would have given
    # 0.05 * 2100 = 105 and the net would be -10 instead of -5.
    assert p_trip.cost_usd == Decimal("1900.0")
    assert p_trip.proceeds_usd == Decimal("1995.0")
    assert p_trip.fees_usd == Decimal("100.0")
    assert p_trip.net_usd == Decimal("-5.0")

    # Losing case: same buy, sell 0.95 ETH at 1900 (a loss). Net should
    # be 0.95 * 1900 - 0.95 * 2000 - 0.05 * 2000 = 1805 - 1900 - 100 = -195.
    # Valued at the sell price the fee would be 0.05 * 1900 = 95 and the
    # net would be -190 instead of -195.
    loss_acts = parse_activities([
        _row("FILL", "l-b", "ETH/USD", "buy",
             "1.0", "2000", "2026-10-11T10:00:00Z", "ob-l"),
        _row("CFEE", "l-c", "ETH/USD", None,
             "-0.05", "0", "2026-10-11T10:00:01Z", "ob-l"),
        _row("FILL", "l-s", "ETH/USD", "sell",
             "0.95", "1900", "2026-10-11T11:00:00Z", "os-l"),
    ])
    [l_trip] = round_trips(loss_acts)
    assert l_trip.cost_usd == Decimal("1900.0")
    assert l_trip.proceeds_usd == Decimal("1805.0")
    assert l_trip.fees_usd == Decimal("100.0")
    assert l_trip.net_usd == Decimal("-195.0")


# -- test_real_btc_roundtrip_fixture ----------------------------------------

def test_real_btc_roundtrip_fixture() -> None:
    """The 2026-10-11 paper round trip nets ~-0.0504 USD with ~0.0490 fees.

    This is the real paper round trip from the run that motivated the
    module: BUY BTC/USD filled 0.000117439 @ 83486.34 with an in-kind fee
    of 0.000000294 BTC, SELL 0.000117145 @ 83473.12 with a 0.25% USD fee.
    The fixture JSON is the byte-for-byte shape Alpaca returns; the
    expected values are derived from the formula in the module docstring
    and asserted within 0.001 USD so a regression in fee valuation cannot
    survive a refactor.
    """

    rows = json.loads(_FIXTURE_PATH.read_text())
    acts = parse_activities(rows)
    # The fixture has one BUY FILL, one BUY CFEE, one SELL FILL, one SELL CFEE.
    assert len(acts) == 4
    types = [a.type for a in acts]
    assert types == ["FILL", "CFEE", "FILL", "CFEE"]

    trips = round_trips(acts)
    assert len(trips) == 1
    trip = trips[0]
    assert trip.symbol == "BTC/USD"
    # Compute the expected values from the formula:
    #   cost_usd    = matched_qty * buy_price
    #   proceeds_usd = matched_qty * sell_price
    #   buy_fee_usd = in_kind_qty * buy_price
    #   sell_fee_usd = 0.25% of proceeds (from the CFEE net_amount)
    #   net_usd     = proceeds - cost - (buy_fee + sell_fee)
    matched_qty = Decimal("0.000117145")
    buy_price = Decimal("83486.34")
    sell_price = Decimal("83473.12")
    in_kind_qty = Decimal("0.000000294")
    expected_buy_fee = in_kind_qty * buy_price
    expected_proceeds = matched_qty * sell_price
    expected_cost = matched_qty * buy_price
    # expected_sell_fee is whatever the fixture records.
    expected_sell_fee = abs(Decimal("-0.02444614661"))
    expected_fees = expected_buy_fee + expected_sell_fee
    expected_net = expected_proceeds - expected_cost - expected_fees
    assert abs(trip.cost_usd - expected_cost) <= Decimal("0.0000001")
    assert abs(trip.proceeds_usd - expected_proceeds) <= Decimal("0.0000001")
    # The asserted fee and net are the real ones from the run.
    assert abs(trip.fees_usd - Decimal("0.0490")) <= Decimal("0.001")
    assert abs(trip.net_usd - Decimal("-0.0504")) <= Decimal("0.001")
    assert trip.fees_usd == expected_fees
    assert trip.net_usd == expected_net


# -- test_partial_fills_fifo ------------------------------------------------

def test_partial_fills_fifo() -> None:
    """FIFO matches the oldest buy lot first, even across multiple fills.

    Three buys (0.1 @ 100, 0.1 @ 110, 0.1 @ 120) then a single sell of
    0.2. The first sell consumes 0.1 from the 100 lot and 0.1 from the
    110 lot in FIFO order; the cost is 0.1*100 + 0.1*110 = 21 and the
    third lot (120) is left as an open position. A second sell of 0.05
    then drains part of the remaining 120 lot.
    """

    acts = parse_activities([
        _row("FILL", "b1", "BTC/USD", "buy",
             "0.1", "100", "2026-10-11T10:00:00Z", "ob1"),
        _row("FILL", "b2", "BTC/USD", "buy",
             "0.1", "110", "2026-10-11T10:01:00Z", "ob2"),
        _row("FILL", "b3", "BTC/USD", "buy",
             "0.1", "120", "2026-10-11T10:02:00Z", "ob3"),
        _row("FILL", "s1", "BTC/USD", "sell",
             "0.2", "150", "2026-10-11T11:00:00Z", "os1"),
        _row("FILL", "s2", "BTC/USD", "sell",
             "0.05", "160", "2026-10-11T12:00:00Z", "os2"),
    ])
    trips = round_trips(acts)
    assert len(trips) == 2
    # First trip: oldest two buys at cost 0.1*100 + 0.1*110 = 21, proceeds
    # 0.2*150 = 30, no fees. Net = 9.
    t1 = trips[0]
    assert t1.qty_sold == Decimal("0.2")
    assert t1.cost_usd == Decimal("21.0")
    assert t1.proceeds_usd == Decimal("30.0")
    assert t1.fees_usd == Decimal("0")
    assert t1.net_usd == Decimal("9.0")
    assert t1.entry_ts == "2026-10-11T10:00:00Z"
    # Second trip: 0.05 from the 120 lot. Cost = 6, proceeds = 8, net = 2.
    t2 = trips[1]
    assert t2.qty_sold == Decimal("0.05")
    assert t2.cost_usd == Decimal("6.0")
    assert t2.proceeds_usd == Decimal("8.0")
    assert t2.fees_usd == Decimal("0")
    assert t2.net_usd == Decimal("2.0")
    assert t2.entry_ts == "2026-10-11T10:02:00Z"

    # Open lots: 0.1 - 0.05 = 0.05 remaining in the 120 lot.
    info = summary(acts)
    assert info["open_lots"] == {"BTC/USD": Decimal("0.05")}
    assert info["n_round_trips"] == 2
    assert info["realized_net_usd"] == Decimal("11.0")
    assert info["fees_usd"] == Decimal("0")
    assert info["gross_usd"] == Decimal("38.0")
    assert info["incomplete"] is False
    assert info["unmatched_sell_qty"] == {}


# -- test_open_lot_remainder -----------------------------------------------

def test_open_lot_remainder() -> None:
    """Unmatched buys become ``open_lots`` and never produce a round trip.

    A buy of 0.5 BTC with no corresponding sell should not produce a
    round trip (because no proceeds were realised) and the full qty must
    surface in the ``open_lots`` map keyed by symbol. A subsequent partial
    sell closes part of it and the remainder stays open.
    """

    acts = parse_activities([
        _row("FILL", "b1", "BTC/USD", "buy",
             "0.5", "30000", "2026-10-11T10:00:00Z", "ob"),
    ])
    trips = round_trips(acts)
    assert trips == []
    info = summary(acts)
    assert info["n_round_trips"] == 0
    assert info["open_lots"] == {"BTC/USD": Decimal("0.5")}
    assert info["realized_net_usd"] == Decimal("0")
    assert info["fees_usd"] == Decimal("0")
    assert info["gross_usd"] == Decimal("0")
    assert info["incomplete"] is False
    assert info["unmatched_sell_qty"] == {}

    # Add a partial sell of 0.2 BTC at 31000.
    more_acts = acts + parse_activities([
        _row("FILL", "s1", "BTC/USD", "sell",
             "0.2", "31000", "2026-10-11T11:00:00Z", "os"),
    ])
    trips = round_trips(more_acts)
    assert len(trips) == 1
    assert trips[0].cost_usd == Decimal("6000.0")
    assert trips[0].proceeds_usd == Decimal("6200.0")
    assert trips[0].net_usd == Decimal("200.0")
    info = summary(more_acts)
    assert info["n_round_trips"] == 1
    assert info["open_lots"] == {"BTC/USD": Decimal("0.3")}
    assert info["realized_net_usd"] == Decimal("200.0")
    assert info["gross_usd"] == Decimal("6200.0")
    assert info["incomplete"] is False
    assert info["unmatched_sell_qty"] == {}


# -- test_no_spread_double_count -------------------------------------------

def test_no_spread_double_count() -> None:
    """The round trip never subtracts spread separately.

    If a future patch adds an explicit bid-ask spread deduction on top of
    the existing buy/sell fill prices, the net would quietly shrink by
    ``spread * qty`` for every fill -- effectively double-counting spread
    because spread is already implicit in the difference between the buy
    fill price and the sell fill price. The test asserts that net is
    exactly ``proceeds - cost - fees`` for a buy at 100 and a sell at 105
    with no fees: net must be 5 (not 5 minus any spread).
    """

    acts = parse_activities([
        _row("FILL", "b1", "BTC/USD", "buy",
             "1.0", "100", "2026-10-11T10:00:00Z", "ob"),
        _row("FILL", "s1", "BTC/USD", "sell",
             "1.0", "105", "2026-10-11T11:00:00Z", "os"),
    ])
    [trip] = round_trips(acts)
    # No fees configured -> fees_usd must be exactly zero and net_usd
    # must be exactly 5 (proceeds - cost - 0). Any spread subtraction
    # would land between 4.99 and 5 here and fail the equality.
    assert trip.fees_usd == Decimal("0")
    assert trip.net_usd == Decimal("5.0")
    assert trip.net_usd == trip.proceeds_usd - trip.cost_usd - trip.fees_usd
    # Summary must agree with the trip.
    info = summary(acts)
    assert info["realized_net_usd"] == Decimal("5.0")
    assert info["fees_usd"] == Decimal("0")
    assert info["gross_usd"] == Decimal("105.0")

    # And the identity holds even when there are fees, in both directions
    # of profitability.
    acts2 = parse_activities([
        _row("FILL", "b2", "BTC/USD", "buy",
             "1.0", "100", "2026-10-11T10:00:00Z", "ob2"),
        _row("FILL", "s2", "BTC/USD", "sell",
             "1.0", "95", "2026-10-11T11:00:00Z", "os2"),
        _row("CFEE", "c2", "BTC/USD", None,
             "0", "0", net_amount="-1.00",
             transaction_time="2026-10-11T11:00:01Z",
             order_id="os2"),
    ])
    [trip2] = round_trips(acts2)
    # Cost 100, proceeds 95, fee 1 -> net -6.
    assert trip2.net_usd == Decimal("-6.0")
    assert trip2.net_usd == trip2.proceeds_usd - trip2.cost_usd - trip2.fees_usd


# -- test_cfee_applied_once_across_partial_fills ----------------------------

def test_cfee_applied_once_across_partial_fills() -> None:
    """A single CFEE is allocated across partial fills, never re-applied.

    The bug: an order with two BUY partial fills (1 + 1) and one CFEE of
    -0.01 asset was deducting the CFEE twice -- once per fill -- so the
    sell of 1.99 produced net = -2 instead of -1.

    Expected: each partial fill receives its proportional share of the
    order's total CFEE (1/2 of 0.01 each), the buy lots carry
    ``qty_net = 0.995`` and ``buy_fee_usd = 0.5``, the sell of 1.99
    drains both lots and produces a single trip with cost 199, proceeds
    199, fees 1, net -1.
    """

    acts = parse_activities([
        _row("FILL", "f1", "BTC/USD", "buy",
             "1", "100", "2026-10-11T10:00:00Z", "ob"),
        _row("FILL", "f2", "BTC/USD", "buy",
             "1", "100", "2026-10-11T10:00:01Z", "ob"),
        _row("CFEE", "c1", "BTC/USD", None,
             "-0.01", "0", "2026-10-11T10:00:02Z", "ob"),
        _row("FILL", "s1", "BTC/USD", "sell",
             "1.99", "100", "2026-10-11T11:00:00Z", "os"),
    ])
    trips = round_trips(acts)
    assert len(trips) == 1
    trip = trips[0]
    # Matched qty = 1.99 (both lots fully drained); cost = proceeds = 199;
    # buy fee is the single CFEE valued at the buy price (0.01 * 100) split
    # evenly across the two fills.
    assert trip.qty_sold == Decimal("1.99")
    assert trip.cost_usd == Decimal("199")
    assert trip.proceeds_usd == Decimal("199")
    assert trip.fees_usd == Decimal("1")
    assert trip.net_usd == Decimal("-1")

    info = summary(acts)
    assert info["realized_net_usd"] == Decimal("-1")
    assert info["fees_usd"] == Decimal("1")
    assert info["incomplete"] is False
    assert info["unmatched_sell_qty"] == {}

    # And the CFEE arriving *after* the fills still applies exactly once
    # because the matcher looks the order_id up by qty-sum, not by event
    # order. Same numbers, different chronology.
    chronologically_late = parse_activities([
        _row("FILL", "f1", "BTC/USD", "buy",
             "1", "100", "2026-10-11T10:00:00Z", "ob"),
        _row("FILL", "f2", "BTC/USD", "buy",
             "1", "100", "2026-10-11T10:00:01Z", "ob"),
        _row("FILL", "s1", "BTC/USD", "sell",
             "1.99", "100", "2026-10-11T10:00:02Z", "os"),
        _row("CFEE", "c1", "BTC/USD", None,
             "-0.01", "0", "2026-10-11T10:00:03Z", "ob"),
    ])
    trips_late = round_trips(chronologically_late)
    [late_trip] = trips_late
    assert late_trip.qty_sold == Decimal("1.99")
    assert late_trip.cost_usd == Decimal("199")
    assert late_trip.proceeds_usd == Decimal("199")
    assert late_trip.fees_usd == Decimal("1")
    assert late_trip.net_usd == Decimal("-1")


# -- test_duplicate_activity_ids_ignored -----------------------------------

def test_duplicate_activity_ids_ignored() -> None:
    """Activities sharing the same ``id`` are de-duplicated before matching.

    Alpaca can return the same activity row on overlapping paginated
    requests. If the matcher processed each duplicate, every FILL would
    be re-debited and every CFEE would be charged twice. This test
    replays the same BUY fill, the same BUY CFEE and the same SELL fill
    a second time and asserts the realised PnL is exactly the same as
    the un-duplicated version -- no second copy, no second fee.
    """

    single = parse_activities([
        _row("FILL", "b1", "BTC/USD", "buy",
             "1.0", "100", "2026-10-11T10:00:00Z", "ob"),
        _row("CFEE", "c1", "BTC/USD", None,
             "-0.01", "0", "2026-10-11T10:00:01Z", "ob"),
        _row("FILL", "s1", "BTC/USD", "sell",
             "0.99", "100", "2026-10-11T11:00:00Z", "os"),
    ])
    [single_trip] = round_trips(single)
    assert single_trip.qty_sold == Decimal("0.99")
    assert single_trip.cost_usd == Decimal("99")
    assert single_trip.proceeds_usd == Decimal("99")
    assert single_trip.fees_usd == Decimal("1")
    assert single_trip.net_usd == Decimal("-1")

    # Replay every activity; net must still be -1, not -2 / -3.
    replayed = parse_activities([
        _row("FILL", "b1", "BTC/USD", "buy",
             "1.0", "100", "2026-10-11T10:00:00Z", "ob"),
        _row("CFEE", "c1", "BTC/USD", None,
             "-0.01", "0", "2026-10-11T10:00:01Z", "ob"),
        _row("FILL", "s1", "BTC/USD", "sell",
             "0.99", "100", "2026-10-11T11:00:00Z", "os"),
        _row("FILL", "b1", "BTC/USD", "buy",  # duplicate id b1
             "1.0", "100", "2026-10-11T10:00:00Z", "ob"),
        _row("CFEE", "c1", "BTC/USD", None,  # duplicate id c1
             "-0.01", "0", "2026-10-11T10:00:01Z", "ob"),
        _row("FILL", "s1", "BTC/USD", "sell",  # duplicate id s1
             "0.99", "100", "2026-10-11T11:00:00Z", "os"),
    ])
    trips_dup = round_trips(replayed)
    # Exactly one round trip, with the exact same numbers.
    assert len(trips_dup) == 1
    dup_trip = trips_dup[0]
    assert dup_trip.qty_sold == single_trip.qty_sold
    assert dup_trip.cost_usd == single_trip.cost_usd
    assert dup_trip.proceeds_usd == single_trip.proceeds_usd
    assert dup_trip.fees_usd == single_trip.fees_usd
    assert dup_trip.net_usd == single_trip.net_usd

    # Summary also matches; nothing extra got charged.
    info_dup = summary(replayed)
    assert info_dup["realized_net_usd"] == summary(single)["realized_net_usd"]
    assert info_dup["fees_usd"] == summary(single)["fees_usd"]


# -- test_orphan_sell_marks_incomplete -------------------------------------

def test_orphan_sell_marks_incomplete() -> None:
    """An unmatched SELL sets ``incomplete=True`` and surfaces its qty.

    The bug: a SELL fill with no buy lot to consume was silently dropped,
    leaving ``summary`` looking clean while a position had effectively
    walked off-book. The contract is that any unmatched sell qty makes
    the result ``incomplete=True`` and records the symbol-keyed qty in
    ``unmatched_sell_qty`` so the caller must surface the flag rather
    than trust the realised PnL.
    """

    acts = parse_activities([
        _row("FILL", "s1", "BTC/USD", "sell",
             "0.5", "100", "2026-10-11T11:00:00Z", "os"),
    ])
    info = summary(acts)
    # The orphan qty is the full 0.5; the ledger side is empty so there
    # are no realised proceeds and the round-trip count is 0 -- but the
    # summary MUST flag itself incomplete.
    assert info["n_round_trips"] == 0
    assert info["realized_net_usd"] == Decimal("0")
    assert info["fees_usd"] == Decimal("0")
    assert info["incomplete"] is True
    assert info["unmatched_sell_qty"] == {"BTC/USD": Decimal("0.5")}

    # A second sell on a different symbol adds to the orphan map and
    # keeps incomplete True.
    more_acts = acts + parse_activities([
        _row("FILL", "s2", "ETH/USD", "sell",
             "1.0", "2000", "2026-10-11T12:00:00Z", "os-eth"),
    ])
    info_more = summary(more_acts)
    assert info_more["incomplete"] is True
    assert info_more["unmatched_sell_qty"] == {
        "BTC/USD": Decimal("0.5"),
        "ETH/USD": Decimal("1.0"),
    }

    # Adding later buys does NOT cover the orphans -- FIFO is
    # time-ordered and the sells already happened. ``incomplete`` stays
    # True. This is the whole point of the flag: a later buy cannot
    # quietly mask an earlier orphan.
    recovered = more_acts + parse_activities([
        _row("FILL", "b1", "BTC/USD", "buy",
             "1.0", "100", "2026-10-12T10:00:00Z", "ob-btc"),
        _row("FILL", "b2", "ETH/USD", "buy",
             "2.0", "2000", "2026-10-12T10:00:01Z", "ob-eth"),
    ])
    info_recovered = summary(recovered)
    assert info_recovered["incomplete"] is True
    assert info_recovered["unmatched_sell_qty"] == {
        "BTC/USD": Decimal("0.5"),
        "ETH/USD": Decimal("1.0"),
    }

    # And a clean buy-then-sell sequence (no orphans) leaves incomplete
    # False with an empty unmatched map.
    clean = parse_activities([
        _row("FILL", "b1", "BTC/USD", "buy",
             "1.0", "100", "2026-10-11T10:00:00Z", "ob"),
        _row("FILL", "s1", "BTC/USD", "sell",
             "1.0", "110", "2026-10-11T11:00:00Z", "os"),
    ])
    info_clean = summary(clean)
    assert info_clean["incomplete"] is False
    assert info_clean["unmatched_sell_qty"] == {}
    assert info_clean["n_round_trips"] == 1

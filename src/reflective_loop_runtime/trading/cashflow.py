"""Fee-accurate net PnL from Alpaca FILL and CFEE activity rows.

The :class:`Activity` dataclass is a single canonical record extracted from an
Alpaca ``/v2/account/activities`` row, regardless of whether that row reports
a fill (``FILL``) or a fee (``CFEE``). Activities are fed to
:func:`round_trips` which performs per-symbol FIFO matching of buys against
sells and produces one :class:`RoundTrip` per completed round, attributing
the in-kind buy fee (valued at the buy price) and the sell-side USD fee to
each round trip. :func:`summary` aggregates the per-trip results into a
flat dict suitable for telemetry / reporting.

All arithmetic uses :class:`decimal.Decimal`; the module never mixes in a
``float``. The output schema is the contract other lanes (telemetry, the
report generator) read against; renaming or removing a field is a breaking
change and must be coordinated across callers.

Background -- Alpaca's fee model for crypto paper trading
--------------------------------------------------------

* A ``FILL`` row records the gross quantity filled for a market order, the
  fill price, the side, the order id, and the transaction timestamp.
* A ``CFEE`` row for a *buy* in-kind fee has ``qty < 0`` and represents the
  amount of the purchased asset taken as a fee (e.g. ``-0.000000294 BTC``
  for a 25 bps in-kind fee on a BTC buy). The economic cost is the qty
  valued at the buy price, *not* at some later sell price -- this is the
  "fee-in-kind at buy price" rule that prevents spreading the cost across
  both legs of a round trip. If the buy has no in-kind fee but a USD
  fee, ``net_amount < 0`` is used instead and the absolute USD is added to
  fees directly.
* A ``CFEE`` row for a *sell* has ``net_amount`` in USD (negative) and
  represents the cash fee deducted from the sale proceeds. If a sell fee
  is denominated in-kind (rare on Alpaca crypto), ``qty < 0`` is used
  instead and the absolute qty is valued at the sell price.
* A sell ``FILL`` can also carry a fee via its ``net_amount`` field: the
  gross USD proceeds are ``qty * price`` and the net is ``net_amount``;
  the difference is the fee when no CFEE is attached.

The FIFO matcher below implements these rules and never subtracts the bid
ask spread explicitly. Spread is already implicit in the buy price and the
sell price; subtracting it again would double-count.

Fee allocation rule (per order)
-------------------------------

An order can be filled in multiple partial FILL rows that share the same
``order_id``. A single ``CFEE`` row reports the *total* fee for the order,
not the fee per fill. Each fill must receive its proportional share of
that total (``fill_qty / total_order_qty``); otherwise the fee is debited
more than once when several partial fills exist and the realised PnL
silently shrinks by ``fee * (n_fills - 1)``.

Duplicate activity rows (same ``id``) are de-duplicated upstream of the
matcher so a refetched log page never re-applies a fill or fee.

Orphan-sell rule
----------------

A ``FILL`` with ``side == 'sell'`` that has no buy lots to consume is
*never* silently dropped. The unmatched sell qty is recorded per symbol in
the summary's ``unmatched_sell_qty`` map and the summary gains an
``incomplete`` boolean. Any unmatched sell sets ``incomplete = True``;
callers must treat an incomplete summary as unsafe to treat as a closed
ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any


# ``Activity`` is a frozen dataclass with the precise field shape other code
# reads against; do not add or rename fields without coordinating with the
# telemetry/report lanes.
@dataclass(frozen=True)
class Activity:
    """A single Alpaca activity row, normalised.

    ``type`` is the normalised activity type. Only ``FILL`` (a fill of an
    order) and ``CFEE`` (a crypto fee) are produced. ``symbol`` is the
    canonical "BASE/QUOTE" form (e.g. ``"BTC/USD"``); ``"BTCUSD"`` from the
    Alpaca wire is rewritten to the slash form here. ``net_amount`` is set
    on CFEE rows that report a USD-denominated fee, otherwise ``None``.
    """

    id: str
    type: str  # 'FILL' or 'CFEE'
    symbol: str  # normalised "BASE/QUOTE"
    side: str | None  # 'buy' / 'sell' / None
    qty: Decimal
    price: Decimal
    net_amount: Decimal | None
    ts: str
    order_id: str | None


@dataclass(frozen=True)
class RoundTrip:
    """One completed FIFO buy-then-sell round trip for a single symbol.

    ``cost_usd`` is the notional paid for the matched quantity at the buy
    fill price. ``proceeds_usd`` is the notional received at the sell fill
    price. ``fees_usd`` is the total round-trip fee: the in-kind buy fee
    valued at the *buy* price (from the buy CFEE ``qty``, never at the sell
    price) plus the sell-side USD fee (from the sell CFEE ``net_amount`` or
    derived from the sell fill). ``net_usd`` is ``proceeds - cost - fees``.
    ``net_pct`` is ``net_usd / cost_usd`` and is the realised return on the
    matched quantity -- never the equity return.
    """

    symbol: str
    entry_ts: str
    exit_ts: str
    qty_sold: Decimal
    cost_usd: Decimal
    proceeds_usd: Decimal
    fees_usd: Decimal
    net_usd: Decimal
    net_pct: Decimal


# --- helpers ---------------------------------------------------------------


def _to_decimal(value: Any) -> Decimal:
    """Coerce a row field to Decimal. ``None`` becomes ``Decimal(0)``.

    Alpaca returns numeric fields as JSON strings (e.g. ``"0.000117439"``)
    or, occasionally, as floats; both are accepted. The function never
    raises on missing values -- an absent qty/price on a CFEE is fine
    because a CFEE has either qty or net_amount, not both.
    """

    if value is None:
        return Decimal(0)
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        # ``bool`` is a subclass of ``int``; reject so ``True``/``False``
        # never silently round-trips as 1/0.
        return Decimal(0)
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    s = str(value).strip()
    if not s:
        return Decimal(0)
    return Decimal(s)


def _normalise_symbol(symbol: str) -> str:
    """Rewrite Alpaca's ``"BTCUSD"`` form to the canonical ``"BTC/USD"``.

    Alpaca's crypto activities ship with symbols in either the
    ``"BASE/QUOTE"`` form or the no-slash ``"BASEUSD"`` form depending on
    the API version. Normalising here keeps every downstream comparison
    (and the report generator's symbol column) consistent.
    """

    if symbol is None:
        return ""
    s = str(symbol).strip()
    if "/" in s:
        return s
    # Heuristic: split a known quote currency off the tail. We keep this
    # conservative; unknown symbols pass through untouched so we never
    # silently mis-classify an exotic pair.
    for quote in ("USDT", "USDC", "USD", "ETH", "BTC"):
        if s.endswith(quote) and len(s) > len(quote):
            base = s[: -len(quote)]
            if base:
                return f"{base}/{quote}"
    return s


def _dedupe_activities(acts: list[Activity]) -> list[Activity]:
    """Drop activities whose ``id`` has already been seen.

    Alpaca can return the same activity row on overlapping paginated
    requests; the matcher must never re-apply a FILL or CFEE because that
    would double-count a fill or, worse, debit a CFEE twice across
    multiple partial fills of the same order. Activities with an empty ``id``
    are kept (the source either pre-deduped or never had one).
    """

    seen: set[str] = set()
    out: list[Activity] = []
    for act in acts:
        if act.id:
            if act.id in seen:
                continue
            seen.add(act.id)
        out.append(act)
    return out


# --- public API ------------------------------------------------------------


def parse_activities(rows: list[dict]) -> list[Activity]:
    """Parse raw Alpaca activity dicts into :class:`Activity` records.

    Unknown activity types are skipped silently; only ``FILL`` and ``CFEE``
    rows contribute to PnL. ``net_amount`` is only set on CFEE rows that
    report a USD fee; on FILL rows it is left as ``None`` because the
    fills do not carry a fee component on Alpaca crypto. The order of the
    returned list matches the order of ``rows``; callers that care about
    time ordering should sort upstream (the matcher itself sorts).
    """

    activities: list[Activity] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_type = row.get("activity_type")
        if raw_type not in ("FILL", "CFEE"):
            continue
        side_raw = row.get("side")
        if side_raw is None or side_raw == "":
            side: str | None = None
        else:
            candidate = str(side_raw).strip().lower()
            side = candidate if candidate in ("buy", "sell") else None
        qty = _to_decimal(row.get("qty"))
        price = _to_decimal(row.get("price"))
        net_amount_raw = row.get("net_amount")
        net_amount: Decimal | None
        if net_amount_raw is None or net_amount_raw == "":
            net_amount = None
        else:
            net_amount = _to_decimal(net_amount_raw)
        order_id_raw = row.get("order_id")
        order_id: str | None = str(order_id_raw) if order_id_raw else None
        activities.append(
            Activity(
                id=str(row.get("id") or ""),
                type=str(raw_type),
                symbol=_normalise_symbol(row.get("symbol") or ""),
                side=side,
                qty=qty,
                price=price,
                net_amount=net_amount,
                ts=str(row.get("transaction_time") or ""),
                order_id=order_id,
            )
        )
    return activities


def _total_fee_for_order(
    cfee_rows: list[Activity],
) -> tuple[Decimal, Decimal]:
    """Sum the *total* CFEE for an order as ``(in_kind_qty, usd_fee)``.

    The fee is reported once per order by Alpaca, regardless of how many
    partial fills that order produced. The matcher splits this total
    proportionally to fill qty at fill time so that two fills never debit
    the same CFEE twice. ``in_kind_qty`` is the absolute value of the
    negative ``qty`` on a CFEE; ``usd_fee`` is the absolute value of
    ``net_amount``.
    """

    in_kind = Decimal(0)
    usd = Decimal(0)
    for cfee in cfee_rows:
        if cfee.net_amount is not None:
            usd += abs(cfee.net_amount)
        if cfee.qty != 0:
            in_kind += abs(cfee.qty)
    return in_kind, usd


def round_trips(acts: list[Activity]) -> list[RoundTrip]:
    """FIFO-match buys against sells per symbol and produce round trips.

    The matcher is per-symbol and time-ordered. For each buy ``FILL`` we
    open a lot carrying the gross qty, the buy price, and any buy-side
    fee (in-kind qty valued at the buy price, or USD net_amount). For each
    sell ``FILL`` we consume qty from the oldest lot first, attributing
    the sell fee (from a matching CFEE ``net_amount`` or, if absent, from
    the sell fill's ``net_amount`` minus ``qty * price``).

    Lots that cannot be fully closed by the available sells are left in
    ``open_lots`` and do not produce a round trip. The buy fee is
    attributed proportionally to the matched fraction of each lot, so a
    partial fill produces a partial round trip with a partial fee.

    Activities are de-duplicated by ``id`` before matching. An order's
    CFEE is allocated across that order's fills proportionally to fill
    qty, so a CFEE arriving after the fills still applies exactly once.
    Sells that have no buy lots to consume are silently tracked here but
    surface in :func:`summary`'s ``unmatched_sell_qty`` map so they
    cannot go unnoticed.
    """

    unique_acts = _dedupe_activities(acts)

    # Index CFEE rows by order_id so each FILL can attribute its share on
    # demand. We keep them as a list because a single order can have
    # multiple CFEE rows (e.g. an in-kind fee plus a wire fee).
    fees_by_order: dict[str, list[Activity]] = {}
    for act in unique_acts:
        if act.type == "CFEE" and act.order_id:
            fees_by_order.setdefault(act.order_id, []).append(act)

    # Sum total fill qty per order so each fill can be attributed its
    # proportional share of the order's CFEE. Buy and sell sides are
    # tracked separately so a buy CFEE is never applied to a sell fill or
    # vice-versa, even though Alpaca's order_id namespace is shared.
    total_buy_qty_by_order: dict[str, Decimal] = {}
    total_sell_qty_by_order: dict[str, Decimal] = {}
    for act in unique_acts:
        if act.type != "FILL" or not act.order_id:
            continue
        if act.side == "buy":
            total_buy_qty_by_order[act.order_id] = (
                total_buy_qty_by_order.get(act.order_id, Decimal(0)) + act.qty
            )
        elif act.side == "sell":
            total_sell_qty_by_order[act.order_id] = (
                total_sell_qty_by_order.get(act.order_id, Decimal(0)) + act.qty
            )

    # Sort activities by time so FIFO is deterministic. Empty timestamps
    # sort first; ties keep their relative order because the sort is
    # stable.
    sorted_acts = sorted(unique_acts, key=lambda a: a.ts or "")

    # Per-symbol FIFO queues of buy lots. Each lot is a dict so we can
    # mutate qty_remaining as sells come in.
    lots_by_symbol: dict[str, list[dict[str, Any]]] = {}
    trips: list[RoundTrip] = []

    for act in sorted_acts:
        if act.type != "FILL":
            continue
        if act.side == "buy":
            cfee_rows = fees_by_order.get(act.order_id or "", ())
            total_in_kind, total_usd = _total_fee_for_order(cfee_rows)
            total_qty = total_buy_qty_by_order.get(
                act.order_id or "", Decimal(0)
            )
            # Allocate this fill's proportional share of the order's
            # total CFEE. A ratio of zero (no buy qty seen) is harmless:
            # it produces a 0 fee and a lot with gross qty intact.
            ratio = (
                act.qty / total_qty if total_qty > 0 else Decimal(0)
            )
            in_kind = total_in_kind * ratio
            usd = total_usd * ratio
            gross_qty = act.qty
            net_qty = gross_qty - in_kind
            if net_qty < 0:
                # Defensive: a CFEE larger than the FILL is a bad row;
                # clamp so downstream arithmetic never goes negative.
                net_qty = Decimal(0)
            # In-kind fee valued at the buy price -- this is the only
            # place buy fees are converted to USD; never at the sell
            # price. A USD fee (rare for crypto buys) is added directly.
            buy_fee_usd = in_kind * act.price + usd
            lots_by_symbol.setdefault(act.symbol, []).append(
                {
                    "qty_gross": gross_qty,
                    "qty_net": net_qty,
                    "price": act.price,
                    "entry_ts": act.ts,
                    "buy_fee_usd": buy_fee_usd,
                    "order_id": act.order_id,
                }
            )
        elif act.side == "sell":
            cfee_rows = fees_by_order.get(act.order_id or "", ())
            total_in_kind, total_usd = _total_fee_for_order(cfee_rows)
            total_qty = total_sell_qty_by_order.get(
                act.order_id or "", Decimal(0)
            )
            ratio = (
                act.qty / total_qty if total_qty > 0 else Decimal(0)
            )
            in_kind_sell = total_in_kind * ratio
            usd_sell = total_usd * ratio
            if in_kind_sell == 0 and usd_sell == 0 and act.net_amount is not None:
                # The sell FILL itself reports a net_amount; the implied
                # fee is the difference between gross proceeds and net.
                gross_proceeds = act.qty * act.price
                implied_fee = gross_proceeds - act.net_amount
                if implied_fee > 0:
                    usd_sell = implied_fee
            # In-kind sell fee valued at the sell price; USD fee added
            # directly.
            sell_fee_usd = in_kind_sell * act.price + usd_sell
            lots = lots_by_symbol.get(act.symbol, [])
            sell_qty_remaining = act.qty
            proceeds_total = Decimal(0)
            cost_total = Decimal(0)
            buy_fee_total = Decimal(0)
            # ``entry_ts_earliest`` is the entry timestamp of the *first*
            # lot consumed (the FIFO oldest). Default to the sell ts so
            # a degenerate no-lot consumption is at least well-defined.
            entry_ts_earliest: str | None = None
            exit_ts = act.ts
            while sell_qty_remaining > 0 and lots:
                lot = lots[0]
                lot_net = lot["qty_net"]
                if lot_net <= 0:
                    lots.pop(0)
                    continue
                take = min(lot_net, sell_qty_remaining)
                if take <= 0:
                    lots.pop(0)
                    continue
                ratio_lot = take / lot_net if lot_net > 0 else Decimal(0)
                attributed_buy_fee = lot["buy_fee_usd"] * ratio_lot
                cost_total += take * lot["price"]
                proceeds_total += take * act.price
                buy_fee_total += attributed_buy_fee
                if entry_ts_earliest is None:
                    entry_ts_earliest = lot["entry_ts"]
                sell_qty_remaining -= take
                remaining_net = lot_net - take
                remaining_fee = lot["buy_fee_usd"] - attributed_buy_fee
                if remaining_net <= 0:
                    lots.pop(0)
                else:
                    lot["qty_net"] = remaining_net
                    lot["buy_fee_usd"] = remaining_fee
            if proceeds_total > 0 or cost_total > 0:
                total_fees = buy_fee_total + sell_fee_usd
                net_usd = proceeds_total - cost_total - total_fees
                net_pct = (
                    net_usd / cost_total
                    if cost_total > 0
                    else Decimal(0)
                )
                trips.append(
                    RoundTrip(
                        symbol=act.symbol,
                        # ``act.ts`` is a stable fallback if somehow no
                        # lot was consumed (shouldn't happen because of
                        # the proceeds/cost > 0 guard, but keep the type
                        # checker honest).
                        entry_ts=entry_ts_earliest or act.ts,
                        exit_ts=exit_ts,
                        qty_sold=act.qty - sell_qty_remaining,
                        cost_usd=cost_total,
                        proceeds_usd=proceeds_total,
                        fees_usd=total_fees,
                        net_usd=net_usd,
                        net_pct=net_pct,
                    )
                )
        # CFEE rows are consumed by the index above; nothing to do here.
    return trips


def summary(acts: list[Activity]) -> dict[str, Any]:
    """Aggregate the round trips and any unmatched lots into a flat dict.

    The returned dict has the keys ``realized_net_usd``, ``fees_usd``,
    ``gross_usd``, ``n_round_trips``, ``open_lots``, ``incomplete`` and
    ``unmatched_sell_qty``. ``open_lots`` is a ``dict[str, Decimal]``
    mapping symbol to the sum of unmatched net qty across all buy lots.
    ``gross_usd`` is the sum of ``proceeds_usd`` across round trips (the
    notional that changed hands at the sell price); it does not subtract
    cost or fees.

    ``incomplete`` is ``True`` iff any sell ``FILL`` could not be fully
    matched against buy lots. ``unmatched_sell_qty`` maps symbol to the
    sum of unmatched sell qty for that symbol. An incomplete summary is
    not safe to treat as a closed ledger -- callers MUST surface the
    incomplete flag rather than silently trusting the realised PnL.
    """

    trips = round_trips(acts)
    realized_net_usd = sum((t.net_usd for t in trips), Decimal(0))
    fees_usd = sum((t.fees_usd for t in trips), Decimal(0))
    gross_usd = sum((t.proceeds_usd for t in trips), Decimal(0))

    unique_acts = _dedupe_activities(acts)

    # Recompute unmatched lots: walk buys and subtract in-kind fees per
    # the same rules used in round_trips, then subtract the qty the
    # round trips consumed.
    fees_by_order: dict[str, list[Activity]] = {}
    for act in unique_acts:
        if act.type == "CFEE" and act.order_id:
            fees_by_order.setdefault(act.order_id, []).append(act)

    total_buy_qty_by_order: dict[str, Decimal] = {}
    for act in unique_acts:
        if act.type == "FILL" and act.side == "buy" and act.order_id:
            total_buy_qty_by_order[act.order_id] = (
                total_buy_qty_by_order.get(act.order_id, Decimal(0))
                + act.qty
            )

    qty_bought: dict[str, Decimal] = {}
    for act in unique_acts:
        if act.type != "FILL" or act.side != "buy":
            continue
        cfee_rows = fees_by_order.get(act.order_id or "", ())
        total_in_kind, _total_usd = _total_fee_for_order(cfee_rows)
        total_qty = total_buy_qty_by_order.get(
            act.order_id or "", Decimal(0)
        )
        ratio = act.qty / total_qty if total_qty > 0 else Decimal(0)
        in_kind = total_in_kind * ratio
        net = act.qty - in_kind
        if net < 0:
            net = Decimal(0)
        qty_bought[act.symbol] = qty_bought.get(act.symbol, Decimal(0)) + net

    consumed: dict[str, Decimal] = {}
    for t in trips:
        consumed[t.symbol] = consumed.get(t.symbol, Decimal(0)) + t.qty_sold

    open_lots_dict: dict[str, Decimal] = {}
    for sym, bought in qty_bought.items():
        remaining = bought - consumed.get(sym, Decimal(0))
        if remaining > 0:
            open_lots_dict[sym] = remaining

    # Orphan sells: walk the timeline and accumulate the qty the matcher
    # could not consume against any buy lot. ``round_trips`` has stashed
    # these in a side-band; we recompute them here so the summary is
    # correct even if the caller never invoked ``round_trips``.
    unmatched_sell_qty: dict[str, Decimal] = {}
    sorted_acts = sorted(unique_acts, key=lambda a: a.ts or "")
    lots_by_symbol: dict[str, list[dict[str, Any]]] = {}
    for act in sorted_acts:
        if act.type != "FILL":
            continue
        if act.side == "buy":
            cfee_rows = fees_by_order.get(act.order_id or "", ())
            total_in_kind, _total_usd = _total_fee_for_order(cfee_rows)
            total_qty = total_buy_qty_by_order.get(
                act.order_id or "", Decimal(0)
            )
            ratio = act.qty / total_qty if total_qty > 0 else Decimal(0)
            in_kind = total_in_kind * ratio
            gross_qty = act.qty
            net_qty = gross_qty - in_kind
            if net_qty < 0:
                net_qty = Decimal(0)
            lots_by_symbol.setdefault(act.symbol, []).append(
                {
                    "qty_net": net_qty,
                }
            )
        elif act.side == "sell":
            lots = lots_by_symbol.get(act.symbol, [])
            sell_qty_remaining = act.qty
            while sell_qty_remaining > 0 and lots:
                lot = lots[0]
                lot_net = lot["qty_net"]
                if lot_net <= 0:
                    lots.pop(0)
                    continue
                take = min(lot_net, sell_qty_remaining)
                if take <= 0:
                    lots.pop(0)
                    continue
                sell_qty_remaining -= take
                remaining_net = lot_net - take
                if remaining_net <= 0:
                    lots.pop(0)
                else:
                    lot["qty_net"] = remaining_net
            if sell_qty_remaining > 0:
                unmatched_sell_qty[act.symbol] = unmatched_sell_qty.get(
                    act.symbol, Decimal(0)
                ) + sell_qty_remaining

    incomplete = any(q > 0 for q in unmatched_sell_qty.values())

    return {
        "realized_net_usd": realized_net_usd,
        "fees_usd": fees_usd,
        "gross_usd": gross_usd,
        "n_round_trips": len(trips),
        "open_lots": open_lots_dict,
        "incomplete": incomplete,
        "unmatched_sell_qty": unmatched_sell_qty,
    }


__all__ = [
    "Activity",
    "RoundTrip",
    "parse_activities",
    "round_trips",
    "summary",
]

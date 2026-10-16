"""Deterministic long-only decider for the RLR paper trader.

The decider owns the *what* of a trade: given a feature snapshot and the
current open position (or ``None``), it returns a single :class:`Decision`
the governor will then vet. The decider itself performs no network I/O,
no clock reads, no randomness, and no broker calls -- it is a pure
function of its inputs and ``now``.

Algorithm
---------
With a position already open, the decider exits when:

* ``f.price <= position.stop``     -> reason ``"stop"`` (loss-cut)
* ``f.price >= position.target``   -> reason ``"target"`` (profit-take)
* ``now - position.entry_ts >= limits.time_stop_s`` -> ``"time"`` (time-stop)

Otherwise it holds with reason ``"in_position"`` -- the open trade is
still inside its protective levels and its time budget.

Without a position, the decider enters long only when every condition
holds:

* both 5-minute EMAs are filled and ``ema20_5m > ema50_5m`` (uptrend)
* the bar has just crossed *up through* VWAP (reclaim, not just "above")
* ATR(14) and price are finite and positive
* ``atr14 / price`` is within ``limits.atr_band`` (volatility is sane)

When that happens, the protective levels are cost-aware: ``stop_dist``
is the maximum of the ATR-based stop, the ``min_stop_pct`` floor (so
the stop is never tighter than the round-trip cost in calm markets), and
the spread-derived cost term (so a wide-spread entry cannot produce a
losing stop). ``target`` is then ``target_r`` multiples of that stop
distance above the entry. ``notional`` is set to ``limits.notional``.
Any other case is a hold with a precise reason so logs and tests can
distinguish them:

* ``"insufficient_data"`` when any required input is ``None`` / non-finite
* ``"no_trend"`` when the EMAs do not confirm an uptrend
* ``"no_vwap_reclaim"`` when the bar did not just cross VWAP from below
* ``"atr_out_of_band"`` when volatility is too tight or too wide
"""

from __future__ import annotations

import math

from reflective_loop_runtime.trading.model import (
    ACTION_ENTER_LONG,
    ACTION_EXIT,
    ACTION_HOLD,
    Decision,
    Limits,
    Position,
)
from reflective_loop_runtime.trading.types import Features


def _is_finite_positive(value: float | None) -> bool:
    """``True`` only for finite, strictly positive numbers."""
    if value is None:
        return False
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f) and f > 0


def _em_trend(f: Features) -> bool:
    """Both EMAs filled and the short EMA above the long EMA."""
    if f.ema20_5m is None or f.ema50_5m is None:
        return False
    try:
        return math.isfinite(f.ema20_5m) and math.isfinite(f.ema50_5m) and f.ema20_5m > f.ema50_5m
    except TypeError:
        return False


def _atr_in_band(f: Features, band: tuple[float, float]) -> bool:
    """ATR14 / price lies within the inclusive ``(low, high)`` band."""
    if f.atr14 is None:
        return False
    if not (_is_finite_positive(f.price) and math.isfinite(f.atr14)):
        return False
    low, high = band
    ratio = float(f.atr14) / float(f.price)
    return low <= ratio <= high


def decide(
    f: Features,
    position: Position | None,
    limits: Limits,
    *,
    now: float,
) -> Decision:
    """Return the decider's call given the latest features and the open trade.

    The function is pure: it reads no globals, calls no clock, and does
    not touch the broker. Tests inject features and ``position`` to
    exercise every branch.
    """
    symbol = f.symbol

    if position is not None and position.symbol != symbol:
        # Defensive guard: a Position for a different symbol must not be
        # evaluated against this snapshot's features. Treating it as no
        # position would let a stale BTC position silently block an ETH
        # entry through the governor's one-per-symbol check.
        position = None

    if position is not None:
        # During 30-second loops, a recent executable bid protects the
        # stop between bar closes. Never use a stale quote for exits.
        observed = f.price
        if (limits.hardened_entries and f.quote_bid is not None
                and f.quote_age_s is not None
                and -10 <= f.quote_age_s <= limits.max_quote_age_s):
            observed = min(f.price, f.quote_bid)
        if limits.hardened_entries and observed <= position.stop:
            return Decision(symbol=symbol, action=ACTION_EXIT, reason="stop")
        # Stops / target / time are checked in the order the brief lists
        # so the reason field reflects the first trigger that fires.
        if f.price <= position.stop:
            return Decision(symbol=symbol, action=ACTION_EXIT, reason="stop")
        if f.price >= position.target:
            return Decision(symbol=symbol, action=ACTION_EXIT, reason="target")
        if now - position.entry_ts >= limits.time_stop_s:
            return Decision(symbol=symbol, action=ACTION_EXIT, reason="time")
        if (limits.vwap_failure_exit and f.vwap is not None
                and f.below_vwap_bars >= 2
                and f.ema20_5m is not None and f.price < f.ema20_5m
                and f.price < position.entry_px and now - position.entry_ts >= 120):
            return Decision(symbol=symbol, action=ACTION_EXIT, reason="vwap_failure")
        return Decision(symbol=symbol, action=ACTION_HOLD, reason="in_position")

    # No open position -- decide whether to enter long.
    # Each ``hold`` branch carries its own reason so a reader of the log
    # can tell *why* the decider did nothing.
    if not (_is_finite_positive(f.price) and _is_finite_positive(f.atr14)):
        # Both required for the ATR-ratio band; missing either means we
        # cannot yet size a stop/target safely.
        if f.ema20_5m is None or f.ema50_5m is None:
            return Decision(symbol=symbol, action=ACTION_HOLD, reason="insufficient_data")
        # price OR atr14 missing/non-finite -> ATR check will also reject
        # it, so call out the band explicitly.
        return Decision(symbol=symbol, action=ACTION_HOLD, reason="atr_out_of_band")

    if not _em_trend(f):
        # ``insufficient_data`` overrides ``no_trend`` so a quiet market
        # with not-yet-filled EMAs is reported honestly.
        if f.ema20_5m is None or f.ema50_5m is None:
            return Decision(symbol=symbol, action=ACTION_HOLD, reason="insufficient_data")
        return Decision(symbol=symbol, action=ACTION_HOLD, reason="no_trend")

    if not f.crossed_above_vwap:
        return Decision(symbol=symbol, action=ACTION_HOLD, reason="no_vwap_reclaim")

    if not _atr_in_band(f, limits.atr_band):
        return Decision(symbol=symbol, action=ACTION_HOLD, reason="atr_out_of_band")

    if limits.hardened_entries:
        if not f.full_vwap_coverage:
            return Decision(symbol=symbol, action=ACTION_HOLD, reason="vwap_history_incomplete")
        if not f.confirmed_vwap_reclaim:
            return Decision(symbol=symbol, action=ACTION_HOLD, reason="reclaim_not_confirmed")
        if f.ema20_5m is None or f.price <= f.ema20_5m:
            return Decision(symbol=symbol, action=ACTION_HOLD, reason="price_below_ema20")
        if f.ret_5m is None or f.ret_5m < limits.min_ret_5m:
            return Decision(symbol=symbol, action=ACTION_HOLD, reason="weak_5m_momentum")
        if f.ret_60m is None or f.ret_60m <= 0:
            return Decision(symbol=symbol, action=ACTION_HOLD, reason="weak_60m_momentum")
        if (f.spread_pct is None or f.vwap is None
                or (f.price - f.vwap) / f.price <=
                limits.min_reclaim_spread_mult * f.spread_pct / 100.0):
            return Decision(symbol=symbol, action=ACTION_HOLD, reason="reclaim_below_spread")
        if (f.quote_age_s is None or f.quote_age_s < -10
                or f.quote_age_s > limits.max_quote_age_s):
            return Decision(symbol=symbol, action=ACTION_HOLD, reason="quote_stale")

    # Cost-aware stop distance: take the widest of three floors so a
    # calm-market or wide-spread entry cannot produce a stop that loses
    # money before it ever fires.
    price = float(f.price)
    atr14 = float(f.atr14)
    atr_stop = float(limits.stop_atr) * atr14
    pct_stop = float(limits.min_stop_pct) * price
    spread_stop = (
        float(limits.cost_mult) * (float(f.spread_pct) / 100.0) * price
        if f.spread_pct is not None
        else 0.0
    )
    stop_dist = max(atr_stop, pct_stop, spread_stop)
    stop = price - stop_dist
    target = price + float(limits.target_r) * stop_dist
    if limits.hardened_entries:
        fee = float(limits.crypto_fee_rate)
        if not math.isfinite(fee) or fee < 0 or fee >= 1:
            return Decision(symbol=symbol, action=ACTION_HOLD, reason="invalid_fee_model")
        # Conservative modeled round-trip: buy fee in asset and sell fee
        # in cash. Realized reports must use actual broker CFEE activities.
        projected_net_r = (target * (1.0 - fee) ** 2 - price) / stop_dist
        if projected_net_r < limits.min_net_target_r:
            return Decision(symbol=symbol, action=ACTION_HOLD, reason="insufficient_net_reward")
    return Decision(
        symbol=symbol,
        action=ACTION_ENTER_LONG,
        reason="trend_vwap_reclaim",
        stop=stop,
        target=target,
        notional=float(limits.notional),
    )


__all__ = ["decide"]

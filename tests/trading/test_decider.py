"""Tests for the deterministic long-only decider.

The decider is a pure function of ``(features, position, limits, now)``,
so every branch is exercised by injecting those inputs directly. The
tests below cover every named branch in the brief:

* ``test_stop_and_target_exit``     -- price stops / target take profit
* ``test_time_stop_exit``           -- time budget exhausted
* ``test_enter_long_on_trend_vwap_reclaim`` -- the happy path
* ``test_hold_reasons``             -- each ``hold`` reason is reachable
* ``test_stop_distance_covers_costs`` -- cost-aware floor wins on tiny ATR
* ``test_target_is_target_r_times_stop`` -- target is an R-multiple of stop
"""

from __future__ import annotations

import math

from reflective_loop_runtime.trading.decider import decide
from reflective_loop_runtime.trading.model import (
    ACTION_ENTER_LONG,
    ACTION_EXIT,
    ACTION_HOLD,
    Decision,
    Limits,
    Position,
)
from reflective_loop_runtime.trading.types import Features

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _features(
    *,
    symbol: str = "BTC/USD",
    price: float = 100.0,
    ema20: float | None = 101.0,
    ema50: float | None = 100.0,
    atr14: float | None = 1.0,
    spread_pct: float | None = 0.05,
    crossed: bool = True,
    age_s: float = 5.0,
) -> Features:
    """Build a fully-populated :class:`Features` with sane defaults.

    Defaults are tuned so most tests need to override only the field
    that drives their branch; ``ema20 > ema50`` keeps the trend check
    pre-satisfied, ``price=100`` and ``atr14=1`` give ``atr/price=0.01``
    which sits comfortably inside the default ``(0.0005, 0.02)`` band.
    """
    return Features(
        symbol=symbol,
        price=price,
        ema20_5m=ema20,
        ema50_5m=ema50,
        vwap=price - 1.0 if crossed else price + 1.0,
        atr14=atr14,
        ret_60m=0.01,
        spread_pct=spread_pct,
        crossed_above_vwap=crossed,
        last_bar_age_s=age_s,
    )


def _limits(**overrides) -> Limits:
    """Defaults that match the brief; overrides apply field-by-field."""
    base = dict(
        notional=500.0,
        stop_atr=1.5,
        target_atr=2.5,
        time_stop_s=3600.0,
        atr_band=(0.0005, 0.02),
        min_stop_pct=0.004,
        cost_mult=4.0,
        target_r=1.6,
    )
    base.update(overrides)
    return Limits(**base)


def _position(
    *,
    symbol: str = "BTC/USD",
    entry_px: float = 100.0,
    stop: float = 95.0,
    target: float = 110.0,
    entry_ts: float = 1000.0,
    atr: float = 1.0,
) -> Position:
    return Position(
        symbol=symbol,
        qty=1.0,
        entry_px=entry_px,
        stop=stop,
        target=target,
        entry_ts=entry_ts,
        atr=atr,
    )


# ---------------------------------------------------------------------------
# Exit paths
# ---------------------------------------------------------------------------


def test_stop_and_target_exit() -> None:
    """An open position is exited on stop and on target, in that order.

    The order matters because the brief lists stop before target; we
    build two features -- one at the stop and one at the target -- and
    confirm each fires only its own reason.
    """
    pos = _position(entry_ts=1000.0, stop=95.0, target=110.0)
    limits = _limits(time_stop_s=10_000.0)  # large so time-stop never wins

    # Exactly at the stop -> stop wins (the <= check is inclusive).
    f_stop = _features(price=95.0)
    d = decide(f_stop, pos, limits, now=2000.0)
    assert d.action == ACTION_EXIT
    assert d.reason == "stop"

    # Above the target -> target wins (the >= check is inclusive too).
    f_target = _features(price=115.0)
    d = decide(f_target, pos, limits, now=2000.0)
    assert d.action == ACTION_EXIT
    assert d.reason == "target"

    # In the middle -> hold with the in-position reason.
    f_mid = _features(price=102.0)
    d = decide(f_mid, pos, limits, now=2000.0)
    assert d.action == ACTION_HOLD
    assert d.reason == "in_position"


def test_time_stop_exit() -> None:
    """Time-stop fires when ``now - entry_ts`` reaches the limit."""
    pos = _position(entry_ts=1000.0, stop=0.0, target=1e9)
    # Inexpensive time budget; the price is well between stop and target
    # so neither protective exit can fire.
    limits = _limits(time_stop_s=600.0)
    f = _features(price=100.0)
    d = decide(f, pos, limits, now=1600.0)
    assert d.action == ACTION_EXIT
    assert d.reason == "time"

    # Just inside the budget -> still in_position.
    just_inside = decide(f, pos, limits, now=1599.0)
    assert just_inside.action == ACTION_HOLD
    assert just_inside.reason == "in_position"


# ---------------------------------------------------------------------------
# Entry happy path
# ---------------------------------------------------------------------------


def test_enter_long_on_trend_vwap_reclaim() -> None:
    """The decider emits ``enter_long`` with cost-aware protective levels.

    With ``price=100``, ``atr14=1`` and ``spread_pct=0.05``, the three
    stop candidates are ``1.5`` (ATR), ``0.4`` (min_stop_pct floor) and
    ``0.2`` (spread-derived), so the ATR floor wins and ``stop = 98.5``.
    The target is then ``target_r`` times that distance above entry.
    """
    f = _features(price=100.0, ema20=101.0, ema50=100.0, atr14=1.0, crossed=True)
    limits = _limits(notional=500.0, stop_atr=1.5, target_atr=2.5)

    d = decide(f, None, limits, now=0.0)

    assert d.action == ACTION_ENTER_LONG
    assert d.reason == "trend_vwap_reclaim"
    # stop = price - max(stop_atr*atr, min_stop_pct*price, cost_mult*spread*price)
    #      = 100 - max(1.5, 0.4, 0.2) = 98.5
    assert math.isclose(d.stop, 98.5, rel_tol=1e-9)
    # target = price + target_r * stop_dist = 100 + 1.6 * 1.5 = 102.4
    assert math.isclose(d.target, 102.4, rel_tol=1e-9)
    assert d.notional == 500.0
    assert d.symbol == f.symbol


# ---------------------------------------------------------------------------
# Cost-aware protective levels
# ---------------------------------------------------------------------------


def test_stop_distance_covers_costs() -> None:
    """In a calm market the ``min_stop_pct`` floor wins the stop-distance max.

    Live losses came from stops narrower than the round-trip cost; the
    cost-aware formula guarantees ``stop_dist >= min_stop_pct * price``
    even when ATR is tiny and the spread term is small. With a tiny
    ATR inside the band, the floor should win and the stop distance
    must equal ``0.4%`` of the entry price.
    """
    f = _features(
        price=100.0,
        ema20=101.0,
        ema50=100.0,
        atr14=0.05,  # tiny but still inside the default atr_band
        spread_pct=0.01,
        crossed=True,
    )
    limits = _limits()

    d = decide(f, None, limits, now=0.0)

    assert d.action == ACTION_ENTER_LONG
    # Expected stop distance = min_stop_pct * price = 0.004 * 100 = 0.4
    stop_dist = f.price - d.stop
    assert math.isclose(stop_dist, 0.4, rel_tol=1e-9)
    # And the protective stop itself is the price minus that distance.
    assert math.isclose(d.stop, 99.6, rel_tol=1e-9)


def test_target_is_target_r_times_stop() -> None:
    """The take-profit is ``target_r`` multiples of the chosen stop distance.

    The target is no longer an ATR multiple -- it scales with the stop so
    the expected payoff-to-risk ratio stays stable regardless of which
    stop-distance term won the max.
    """
    f = _features(price=100.0, ema20=101.0, ema50=100.0, atr14=1.0, crossed=True)
    limits = _limits()

    d = decide(f, None, limits, now=0.0)

    assert d.action == ACTION_ENTER_LONG
    # target - price == target_r * (price - stop)
    upside = d.target - f.price
    risk = f.price - d.stop
    assert math.isclose(upside, 1.6 * risk, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Hold reasons
# ---------------------------------------------------------------------------


def test_hold_reasons() -> None:
    """Each ``hold`` branch carries the reason the brief defines."""
    limits = _limits()

    # EMAs missing entirely -> insufficient_data
    f_no_ema = _features(ema20=None, ema50=None)
    d = decide(f_no_ema, None, limits, now=0.0)
    assert d.action == ACTION_HOLD
    assert d.reason == "insufficient_data"

    # One EMA missing -> still insufficient_data
    f_one_ema = _features(ema20=101.0, ema50=None)
    d = decide(f_one_ema, None, limits, now=0.0)
    assert d.action == ACTION_HOLD
    assert d.reason == "insufficient_data"

    # EMAs flat or inverted -> no_trend
    f_flat = _features(ema20=100.0, ema50=100.0)
    d = decide(f_flat, None, limits, now=0.0)
    assert d.action == ACTION_HOLD
    assert d.reason == "no_trend"

    f_down = _features(ema20=99.0, ema50=100.0)
    d = decide(f_down, None, limits, now=0.0)
    assert d.action == ACTION_HOLD
    assert d.reason == "no_trend"

    # Trend ok, VWAP not crossed -> no_vwap_reclaim
    f_no_cross = _features(ema20=101.0, ema50=100.0, crossed=False)
    d = decide(f_no_cross, None, limits, now=0.0)
    assert d.action == ACTION_HOLD
    assert d.reason == "no_vwap_reclaim"

    # Trend ok, VWAP crossed, ATR too low -> atr_out_of_band (low side)
    f_thin_atr = _features(ema20=101.0, ema50=100.0, atr14=1e-5, crossed=True)
    d = decide(f_thin_atr, None, limits, now=0.0)
    assert d.action == ACTION_HOLD
    assert d.reason == "atr_out_of_band"

    # Trend ok, VWAP crossed, ATR too high -> atr_out_of_band (high side)
    f_wild_atr = _features(ema20=101.0, ema50=100.0, atr14=10.0, crossed=True)
    d = decide(f_wild_atr, None, limits, now=0.0)
    assert d.action == ACTION_HOLD
    assert d.reason == "atr_out_of_band"

    # ATR missing entirely -> atr_out_of_band (cannot run the ratio)
    f_no_atr = _features(ema20=101.0, ema50=100.0, atr14=None, crossed=True)
    d = decide(f_no_atr, None, limits, now=0.0)
    assert d.action == ACTION_HOLD
    assert d.reason == "atr_out_of_band"


# ---------------------------------------------------------------------------
# Defensive behaviour: a Position for a different symbol is ignored
# ---------------------------------------------------------------------------


def test_position_for_other_symbol_is_ignored() -> None:
    """A stale Position whose symbol does not match ``f.symbol`` is dropped.

    Without this the governor's one-per-symbol check could let a stale
    BTC position silently block an ETH entry.
    """
    f_eth = _features(symbol="ETH/USD")
    pos_btc = _position(symbol="BTC/USD")
    limits = _limits()
    d = decide(f_eth, pos_btc, limits, now=0.0)
    # The position was dropped, so the decider evaluates ETH as flat
    # and -- with crossed=True and EMAs in trend -- enters long.
    assert d.action == ACTION_ENTER_LONG
    assert d.symbol == "ETH/USD"


def test_decision_is_pure_with_no_position() -> None:
    """Calling ``decide`` twice on identical inputs returns an equal Decision.

    The decider is meant to be replay-safe against frozen snapshots.
    """
    f = _features()
    limits = _limits()
    d1 = decide(f, None, limits, now=123.0)
    d2 = decide(f, None, limits, now=123.0)
    assert d1 == d2


# Touch ``Decision`` so the dataclass import is not flagged as unused when
# the file is examined in isolation.
_ = Decision

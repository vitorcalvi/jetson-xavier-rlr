"""Deterministic feature pipeline for the RLR paper trader.

The trading lane's core promise is *deterministic features*: given the same
:class:`Snapshot` and the same ``now``, :func:`compute` must produce the
same :class:`Features` object. This is what makes the slow lane (signal
scoring, telemetry, the held-out scorer) safe to replay against frozen
historical snapshots: there is no clock, randomness, or external lookup
inside this module -- only the snapshot.

Algorithms
----------
* **5-minute resample**: 1-minute bars are grouped by ``floor(ts / 300)``.
  The bucket's ``close`` is the close of the *last* bar in the bucket; this
  is what a typical exchange displays as the 5-minute close.
* **EMA20/EMA50**: standard exponential moving average with smoothing
  ``alpha = 2 / (period + 1)``, seeded with the first value of the window.
  ``EMA50`` is ``None`` until at least 50 five-minute closes are
  available; ``EMA20`` likewise requires 20.
* **VWAP**: rolling intraday VWAP, anchored at 00:00 UTC. ``typical = (h +
  l + c) / 3``. Returns ``None`` when the day's bars carry no volume.
* **ATR(14) (Wilder)**: true range of bar *i* is ``max(h-l, |h-prev_c|,
  |l-prev_c|)``; the first TR uses ``prev_c = bar.o``. ATR is seeded with
  the simple mean of the first 14 true ranges, then Wilder-smoothed.
  ``None`` when there are fewer than 14 bars.
* **ret_60m**: the close-to-close return over the trailing 60 one-minute
  bars, ``c[-1] / c[-61] - 1``. ``None`` when fewer than 61 bars exist.
* **spread_pct**: ``(ask-bid) / mid * 100`` where ``mid = (ask+bid)/2``.
  ``None`` when the snapshot has no quote.
* **crossed_above_vwap**: ``True`` when the last close is above VWAP *and*
  any of the previous 15 closes sat below VWAP -- i.e. the price actually
  crossed up, not merely floats above it.
* **last_bar_age_s**: ``now - last_bar_ts``. ``now`` when ``last_bar_ts``
  is ``0.0``; callers should treat large ages (>> bars of the same minute)
  as stale data.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from reflective_loop_runtime.trading.types import Bar, Features, Snapshot

# Window sizes. Named so the brief's "50 five-minute bars" is not a magic
# number in the body of the function.
_EMA20_PERIOD = 20
_EMA50_PERIOD = 50
_ATR_PERIOD = 14
_RET_60M_WINDOW = 60
_CROSSED_LOOKBACK = 15
_5M_SECONDS = 300


def _bucket_key(ts: float) -> int:
    """Group a 1-minute bar's epoch into its 5-minute bucket."""
    return int(math.floor(float(ts) / _5M_SECONDS))


def _resample_5m_closes(bars: tuple[Bar, ...]) -> list[float]:
    """Collapse 1-minute bars into a list of 5-minute closes (last close per bucket)."""
    if not bars:
        return []
    closes_by_bucket: dict[int, float] = {}
    # ``bars`` is already sorted by ``ts`` ascending (assembled by market.py),
    # but we don't rely on that -- we sort by bucket key and pick the last.
    # A stable iteration by ``ts`` ensures the last bar wins within a bucket.
    for bar in sorted(bars, key=lambda b: b.ts):
        closes_by_bucket[_bucket_key(bar.ts)] = bar.c
    return [closes_by_bucket[key] for key in sorted(closes_by_bucket)]


def _ema(series: list[float], period: int) -> float | None:
    """Standard EMA; returns ``None`` until ``len(series) >= period``."""
    if len(series) < period:
        return None
    alpha = 2.0 / (period + 1)
    # Seed with the simple mean of the first ``period`` values.
    ema = sum(series[:period]) / period
    for value in series[period:]:
        ema = alpha * value + (1 - alpha) * ema
    return ema


def _vwap_day(bars: tuple[Bar, ...], now: float) -> float | None:
    """Rolling intraday VWAP anchored at 00:00 UTC of the ``now`` day."""
    now_day = datetime.fromtimestamp(float(now), tz=timezone.utc).date()
    pv = 0.0
    total_volume = 0.0
    for bar in bars:
        bar_day = datetime.fromtimestamp(bar.ts, tz=timezone.utc).date()
        if bar_day != now_day:
            continue
        volume = bar.v
        if not math.isfinite(volume) or volume <= 0:
            continue
        typical = (bar.h + bar.l + bar.c) / 3.0
        pv += typical * volume
        total_volume += volume
    if total_volume <= 0:
        return None
    return pv / total_volume


def _atr14(bars: tuple[Bar, ...]) -> float | None:
    """Wilder's ATR(14) over 1-minute true range."""
    if len(bars) < _ATR_PERIOD:
        return None
    true_ranges: list[float] = []
    prev_close = bars[0].o  # First TR seeded with the bar's own open.
    for index in range(_ATR_PERIOD):
        bar = bars[index]
        high_low = bar.h - bar.l
        high_prev = abs(bar.h - prev_close)
        low_prev = abs(bar.l - prev_close)
        true_ranges.append(max(high_low, high_prev, low_prev))
        prev_close = bar.c
    atr = sum(true_ranges) / _ATR_PERIOD
    for index in range(_ATR_PERIOD, len(bars)):
        bar = bars[index]
        high_low = bar.h - bar.l
        high_prev = abs(bar.h - prev_close)
        low_prev = abs(bar.l - prev_close)
        tr = max(high_low, high_prev, low_prev)
        atr = (atr * (_ATR_PERIOD - 1) + tr) / _ATR_PERIOD
        prev_close = bar.c
    return atr


def _ret_60m(bars: tuple[Bar, ...]) -> float | None:
    """Close-to-close return over the trailing 60 one-minute bars."""
    if len(bars) <= _RET_60M_WINDOW:
        return None
    last = bars[-1].c
    past = bars[-1 - _RET_60M_WINDOW].c
    if past == 0:
        return None
    return last / past - 1.0


def _spread_pct(snap: Snapshot) -> float | None:
    """Quote spread as a percentage of mid; ``None`` when no quote."""
    if snap.quote is None:
        return None
    bid = snap.quote.bid
    ask = snap.quote.ask
    # An IEX close/overnight quote can carry bid>0 with ask=0 (seen on
    # 2026-10-15). Never turn that into a negative, "cheap" spread.
    # Reject malformed/crossed/non-finite BBO before it reaches risk policy.
    if not (math.isfinite(bid) and math.isfinite(ask) and bid > 0 and ask > 0 and ask >= bid):
        return None
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return None
    return (ask - bid) / mid * 100.0


def _crossed_above_vwap(bars: tuple[Bar, ...], vwap: float | None) -> bool:
    """True iff the last close is above VWAP and the prior 15 closes dipped below."""
    if vwap is None or not bars:
        return False
    last_close = bars[-1].c
    if not (last_close > vwap):
        return False
    start = max(0, len(bars) - 1 - _CROSSED_LOOKBACK)
    return any(bars[i].c < vwap for i in range(start, len(bars) - 1))


def compute(snap: Snapshot, *, now: float) -> Features:
    """Compute deterministic :class:`Features` for ``snap`` at epoch ``now``."""
    bars = snap.bars

    # Do not treat the current still-forming 5m candle as closed.
    complete_bars = tuple(b for b in bars if b.ts < math.floor(float(now) / 300.0) * 300.0)
    closes_5m = _resample_5m_closes(complete_bars)
    ema20_5m = _ema(closes_5m, _EMA20_PERIOD)
    ema50_5m = _ema(closes_5m, _EMA50_PERIOD)
    vwap = _vwap_day(bars, float(now))
    atr14 = _atr14(bars)
    ret_60m = _ret_60m(bars)
    spread = _spread_pct(snap)
    crossed = _crossed_above_vwap(bars, vwap)

    price = bars[-1].c if bars else 0.0
    last_bar_age_s = float(now) - snap.last_bar_ts
    five_min_return = (bars[-1].c / bars[-6].c - 1.0) if len(bars) >= 6 and bars[-6].c > 0 else None
    # Genuine two-bar reclaim: two adjacent closed 1m bars above the
    # current VWAP, immediately preceded by one below. Prevents 15m-old
    # dips from being treated as fresh reclaims.
    confirmed = bool(
        vwap is not None
        and len(bars) >= 3
        and bars[-1].ts - bars[-2].ts == 60
        and bars[-2].ts - bars[-3].ts == 60
        and bars[-3].c < vwap
        and bars[-2].c > vwap
        and bars[-1].c > vwap
    )
    below = 0
    if vwap is not None:
        for bar in reversed(bars):
            if bar.c >= vwap:
                break
            below += 1
    quote_age = float(now) - snap.quote.ts if snap.quote is not None else None
    today = datetime.fromtimestamp(float(now), tz=timezone.utc)
    utc_midnight = datetime(today.year, today.month, today.day, tzinfo=timezone.utc).timestamp()
    today_bars = [b for b in bars if utc_midnight <= b.ts < float(now)]
    vwap_complete = bool(today_bars and today_bars[0].ts <= utc_midnight + 120)

    return Features(
        symbol=snap.symbol,
        price=float(price),
        ema20_5m=ema20_5m,
        ema50_5m=ema50_5m,
        vwap=vwap,
        atr14=atr14,
        ret_60m=ret_60m,
        spread_pct=spread,
        crossed_above_vwap=crossed,
        last_bar_age_s=last_bar_age_s,
        confirmed_vwap_reclaim=confirmed,
        below_vwap_bars=below,
        ret_5m=five_min_return,
        quote_age_s=quote_age,
        quote_bid=snap.quote.bid if snap.quote is not None else None,
        full_vwap_coverage=vwap_complete,
    )


__all__ = ["compute"]

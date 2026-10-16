"""ORB5: 5-minute opening-range breakout above session VWAP (long only).

Pre-registered forward paper hypothesis (docs/prereg/ORB5_IREN_PREREG.toon),
implemented as tested in ChatGPT's RLR_IREN_5M_EDGE_RESEARCH v1.0:

* 5m bars are built from regular-session 1m bars in buckets aligned to
  09:30 ET; only *completed* buckets are used.
* Opening range = high of the 09:30-09:35 bucket.
* Entry signal on a completed bucket whose open is in [09:35, 10:45] ET:
  close > OR high, the previous completed close was NOT above OR high
  (a fresh breakout), and close > VWAP anchored at 09:30 ET.
* Stop distance = max(stop_atr * ATR14(5m), min_stop_pct * price,
  cost_mult * spread * price); target = target_r * stop distance.
* Exits are evaluated on completed 5m buckets: stop, target, 60-min time stop.
  The loop's session layer still flattens from 15:50 ET.

The function is pure: it reads only the snapshot, the position and ``now``.
The loop calls it on every 1m tick; signals fire on the tick one minute after
a bucket completes (data latency), so the order lands ~1 minute after the next
bucket's open: a documented deviation from the backtest's exact next open.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, time

from reflective_loop_runtime.trading.model import (
    ACTION_ENTER_LONG,
    ACTION_EXIT,
    ACTION_HOLD,
    Decision,
    Limits,
    Position,
)
from reflective_loop_runtime.trading.session import ET
from reflective_loop_runtime.trading.types import Bar, Snapshot

BUCKET_S = 300
SESSION_OPEN = time(9, 30)
SESSION_CLOSE = time(16, 0)
SIGNAL_FIRST = time(9, 35)
SIGNAL_LAST = time(10, 45)
SIGNAL_LAST_EXPERIMENTAL = time(14, 30)
ATR_PERIOD = 14
# Parity with the research backtest (ChatGPT parity review M1/M2)
MIN_BUCKET_MINUTES = 4
ATR_PCT_BAND = (0.001, 0.08)
MIN_PRICE = 5.0


@dataclass(frozen=True)
class Bar5:
    start: float
    o: float
    h: float
    l: float  # noqa: E741 - OHLC field name
    c: float
    v: float
    n: int = 0  # 1m bars present in the bucket
    has_last: bool = False  # the bucket's final (4th-offset) minute is present


def _et(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=ET)


def _in_session(ts: float) -> bool:
    t = _et(ts).time()
    return SESSION_OPEN <= t < SESSION_CLOSE


def _bucket_start(ts: float) -> float:
    d = _et(ts)
    anchor = d.replace(hour=9, minute=30, second=0, microsecond=0).timestamp()
    return anchor + math.floor((ts - anchor) / BUCKET_S) * BUCKET_S


def completed_5m_bars(bars: tuple[Bar, ...], now: float) -> list[Bar5]:
    """Regular-session 5m bars whose bucket ended at or before ``now``."""
    buckets: dict[float, list[Bar]] = {}
    for b in bars:
        if not _in_session(b.ts):
            continue
        start = _bucket_start(b.ts)
        if start + BUCKET_S <= now:
            buckets.setdefault(start, []).append(b)
    out = []
    for start in sorted(buckets):
        grp = sorted(buckets[start], key=lambda x: x.ts)
        out.append(Bar5(start, grp[0].o, max(x.h for x in grp), min(x.l for x in grp),
                        grp[-1].c, sum(x.v for x in grp), len(grp),
                        any(abs(x.ts - (start + 240)) < 1 for x in grp)))
    return out


def atr14(bars5: list[Bar5]) -> float | None:
    """Wilder ATR(14) over the 5m sequence (prior sessions included)."""
    if len(bars5) < ATR_PERIOD + 1:
        return None
    trs = [max(b.h - b.l, abs(b.h - p.c), abs(b.l - p.c)) for p, b in zip(bars5, bars5[1:], strict=False)]
    atr = sum(trs[:ATR_PERIOD]) / ATR_PERIOD
    for tr in trs[ATR_PERIOD:]:
        atr = (atr * (ATR_PERIOD - 1) + tr) / ATR_PERIOD
    return atr


def session_vwap(bars: tuple[Bar, ...], day: datetime, upto: float) -> float | None:
    """VWAP of today's regular-session 1m bars that ended at or before ``upto``."""
    pv = vol = 0.0
    for b in bars:
        d = _et(b.ts)
        if d.date() == day.date() and _in_session(b.ts) and b.ts + 60 <= upto:
            pv += (b.h + b.l + b.c) / 3.0 * b.v
            vol += b.v
    return pv / vol if vol > 0 else None


def _spread_frac(snap: Snapshot) -> float:
    q = snap.quote
    if q is None or q.bid <= 0 or q.ask <= 0:
        return 0.0
    mid = (q.bid + q.ask) / 2.0
    return (q.ask - q.bid) / mid if mid > 0 else 0.0


def decide_orb5(
    snap: Snapshot,
    position: Position | None,
    limits: Limits,
    *,
    now: float,
    signal_last: time = SIGNAL_LAST,
) -> Decision:
    symbol = snap.symbol
    bars5 = completed_5m_bars(snap.bars, now)
    if not bars5:
        return Decision(symbol=symbol, action=ACTION_HOLD, reason="insufficient_data")
    last = bars5[-1]
    # Act on the tick one minute after the bucket ends, so the bucket's last
    # 1m bar has been published (Alpaca emits bars shortly after the minute).
    lag = now - (last.start + BUCKET_S)
    just_completed = 60 <= lag < 120
    if position is not None:
        if not just_completed:
            return Decision(symbol=symbol, action=ACTION_HOLD, reason="in_position")
        if last.c <= position.stop:
            return Decision(symbol=symbol, action=ACTION_EXIT, reason="stop")
        if last.c >= position.target:
            return Decision(symbol=symbol, action=ACTION_EXIT, reason="target")
        if now - position.entry_ts >= limits.time_stop_s:
            return Decision(symbol=symbol, action=ACTION_EXIT, reason="time")
        return Decision(symbol=symbol, action=ACTION_HOLD, reason="in_position")
    if not just_completed:
        return Decision(symbol=symbol, action=ACTION_HOLD, reason="await_5m_close")
    day = _et(last.start)
    today = [b for b in bars5 if _et(b.start).date() == day.date()]
    or_bar = next((b for b in today if _et(b.start).time() == SESSION_OPEN), None)
    if or_bar is None:
        return Decision(symbol=symbol, action=ACTION_HOLD, reason="no_opening_range")
    if not (SIGNAL_FIRST <= _et(last.start).time() <= signal_last):
        return Decision(symbol=symbol, action=ACTION_HOLD, reason="outside_signal_window")
    if last.n < MIN_BUCKET_MINUTES or not last.has_last:
        return Decision(symbol=symbol, action=ACTION_HOLD, reason="incomplete_bucket")
    idx = today.index(last)
    prev_close = today[idx - 1].c if idx >= 1 else None
    if prev_close is None or not (last.c > or_bar.h and prev_close <= or_bar.h):
        return Decision(symbol=symbol, action=ACTION_HOLD, reason="no_breakout")
    vwap = session_vwap(snap.bars, day, last.start + BUCKET_S)
    if vwap is None or not last.c > vwap:
        return Decision(symbol=symbol, action=ACTION_HOLD, reason="below_vwap")
    atr = atr14(bars5)
    if atr is None or atr <= 0:
        return Decision(symbol=symbol, action=ACTION_HOLD, reason="insufficient_data")
    price = last.c
    if price < MIN_PRICE or not (ATR_PCT_BAND[0] <= atr / price <= ATR_PCT_BAND[1]):
        return Decision(symbol=symbol, action=ACTION_HOLD, reason="atr_or_price_out_of_band")
    stop_dist = max(float(limits.stop_atr) * atr,
                    float(limits.min_stop_pct) * price,
                    float(limits.cost_mult) * _spread_frac(snap) * price)
    return Decision(
        symbol=symbol,
        action=ACTION_ENTER_LONG,
        reason="orb5_breakout",
        stop=price - stop_dist,
        target=price + float(limits.target_r) * stop_dist,
        notional=float(limits.notional),
    )


# Limits for the pre-registered ORB5 test (differences from the defaults).
ORB5_LIMITS = {"stop_atr": 1.0, "max_entries_per_day": 3}

# Experimental PAPER-only exploration. Not a verified trading edge.
# Smaller notional and extended opportunity window vs preregistered ORB5.
ORB5_EXPLORE_LIMITS = {"stop_atr": 1.0, "max_entries_per_day": 3, "notional": 100.0}


def decide_orb5_explore(
    snap: Snapshot, position: Position | None, limits: Limits, *, now: float,
) -> Decision:
    """Unproven exploratory PAPER strategy with a later last signal time.

    The immutable preregistered ORB5 function remains unchanged by default.
    All signals still require a fresh OR-high breakout, completed 5m bars,
    session VWAP, ATR validity and governor approval.
    """
    return decide_orb5(snap, position, limits, now=now, signal_last=SIGNAL_LAST_EXPERIMENTAL)


__all__ = ["ORB5_LIMITS", "ORB5_EXPLORE_LIMITS", "completed_5m_bars",
           "decide_orb5", "decide_orb5_explore", "atr14", "session_vwap"]

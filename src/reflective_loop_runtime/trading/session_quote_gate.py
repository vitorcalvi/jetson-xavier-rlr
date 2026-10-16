"""Fail-closed quote/bar gate for strictly paper-only one-minute diagnostics."""
from __future__ import annotations
from datetime import datetime, time, timezone
from math import isfinite
from zoneinfo import ZoneInfo

ET = ZoneInfo('America/New_York')

def check(bid: float, ask: float, quote_ts: float, bar_ts: float, now: float, *, source: str = 'alpaca_iex') -> tuple[bool, str]:
    if source != 'alpaca_iex':
        return False, 'unverified_market_source'
    if not all(isfinite(x) for x in (bid, ask, quote_ts, bar_ts, now)):
        return False, 'nonfinite_market_data'
    if bid <= 0 or ask <= bid:
        return False, 'invalid_bid_ask'
    age = now - quote_ts
    if not (-0.25 <= age <= 5.0):
        return False, 'stale_or_future_quote'
    bar_age = now - (bar_ts + 60.0)
    if not (0 <= bar_age <= 65.0):
        return False, 'stale_or_incomplete_1m_bar'
    spread_bps = 10000 * (ask - bid) / ((ask + bid) / 2)
    if spread_bps > 15:
        return False, 'spread_excessive'
    d = datetime.fromtimestamp(now, timezone.utc).astimezone(ET)
    if d.weekday() >= 5 or not time(9,35) <= d.time() < time(15,50):
        return False, 'outside_regular_market_window'
    return True, 'validated_iex_1m'

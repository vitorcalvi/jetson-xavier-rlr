"""US-equity regular-session rules for the paper trader (America/New_York).

Entries are allowed only Mon-Fri 09:35-15:45 ET (skip the opening auction
minutes and leave room before the close); any open position is closed from
15:50 ET so nothing is held overnight. Exchange holidays are covered by the
broker's market clock, which the loop checks separately.
"""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
ENTRY_START = time(9, 35)
ENTRY_END = time(15, 45)
FLATTEN_AT = time(15, 50)  # 10 one-minute retries before the close
SESSION_OPEN = time(9, 30)
SESSION_CLOSE = time(16, 0)


def _et(now_epoch: float) -> datetime:
    return datetime.fromtimestamp(now_epoch, tz=ET)


def _weekday(now_epoch: float) -> bool:
    return _et(now_epoch).weekday() < 5


def entry_allowed(now_epoch: float) -> bool:
    """True only on weekdays with ET time in [09:35, 15:45)."""
    t = _et(now_epoch).time()
    return _weekday(now_epoch) and ENTRY_START <= t < ENTRY_END


def in_regular_session(now_epoch: float) -> bool:
    """True on weekdays with ET time in [09:30, 16:00)."""
    t = _et(now_epoch).time()
    return _weekday(now_epoch) and SESSION_OPEN <= t < SESSION_CLOSE


def must_flatten(now_epoch: float) -> bool:
    """True during the regular session at or after 15:50 ET."""
    return in_regular_session(now_epoch) and _et(now_epoch).time() >= FLATTEN_AT


def is_weekday(now_epoch: float) -> bool:
    """Mon-Fri in America/New_York."""
    return _weekday(now_epoch)

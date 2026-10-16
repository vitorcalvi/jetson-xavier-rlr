"""ORB5 pre-registered strategy: 5m opening-range breakout above session VWAP."""

from __future__ import annotations

from datetime import datetime, timezone

from reflective_loop_runtime.trading.model import Limits, Position
from reflective_loop_runtime.trading.orb5 import (
    ORB5_EXPLORE_LIMITS,
    ORB5_LIMITS,
    completed_5m_bars,
    decide_orb5,
    decide_orb5_explore,
)
from reflective_loop_runtime.trading.types import Bar, Quote, Snapshot


def _ts(iso_utc: str) -> float:
    return datetime.fromisoformat(iso_utc).replace(tzinfo=timezone.utc).timestamp()


# 2026-10-12 is EDT: 13:30Z == 09:30 ET.
OPEN = _ts("2026-10-12T13:30:00")
LIMITS = Limits(**ORB5_LIMITS)


def _prior_day() -> list[Bar]:
    """Friday 2026-10-09 regular session, gently oscillating (ATR history)."""
    start = _ts("2026-10-09T13:30:00")
    return [
        Bar(start + 60 * i, 10.0, 10.02, 9.98, 10.0 + (0.01 if i % 2 else -0.01), 1000)
        for i in range(390)
    ]


def _today(closes_by_minute: dict[int, float], minutes: int, base: float = 10.0) -> list[Bar]:
    out = []
    for i in range(minutes):
        c = closes_by_minute.get(i, base)
        out.append(Bar(OPEN + 60 * i, c, c + 0.01, c - 0.01, c, 1000))
    return out


def _snap(bars: list[Bar]) -> Snapshot:
    return Snapshot(
        symbol="IREN",
        bars=tuple(bars),
        quote=Quote(bid=10.0, ask=10.003, ts=bars[-1].ts),
        fetched_at=bars[-1].ts,
        last_bar_ts=bars[-1].ts,
    )


def _breakout_day() -> list[Bar]:
    # 09:30-09:35 opening range high = 10.01 (closes 10.0); 09:35-09:40 closes ramp to 10.20
    closes = {5: 10.05, 6: 10.10, 7: 10.12, 8: 10.15, 9: 10.20}
    return _prior_day() + _today(closes, 10)


def test_enters_one_minute_after_breakout_bucket_completes() -> None:
    bars = _breakout_day()
    d = decide_orb5(_snap(bars), None, LIMITS, now=OPEN + 11 * 60)  # 09:41 ET
    assert d.action == "enter_long" and d.reason == "orb5_breakout"
    assert d.stop < 10.20 < d.target
    assert abs((d.target - 10.20) - 1.6 * (10.20 - d.stop)) < 1e-9


def test_no_signal_at_bucket_end_minute_or_mid_bucket() -> None:
    bars = _breakout_day()
    assert decide_orb5(_snap(bars), None, LIMITS, now=OPEN + 10 * 60).action == "hold"  # lag 0
    assert decide_orb5(_snap(bars), None, LIMITS, now=OPEN + 13 * 60).action == "hold"  # lag 180 s


def test_not_a_fresh_breakout_when_previous_close_already_above() -> None:
    closes = {i: 10.20 for i in range(5, 15)}
    bars = _prior_day() + _today(closes, 15)
    d = decide_orb5(_snap(bars), None, LIMITS, now=OPEN + 16 * 60)  # bucket 09:40-09:45 done
    assert d.action == "hold" and d.reason == "no_breakout"


def test_outside_signal_window_holds() -> None:
    closes = {i: 10.0 for i in range(0, 95)}
    closes.update({i: 10.30 for i in range(95, 100)})  # breakout in the 11:05 bucket
    bars = _prior_day() + _today(closes, 100)
    d = decide_orb5(_snap(bars), None, LIMITS, now=OPEN + 101 * 60)
    assert d.action == "hold" and d.reason == "outside_signal_window"


def test_exit_on_completed_bucket_stop() -> None:
    bars = _breakout_day() + [Bar(OPEN + 60 * i, 9.5, 9.51, 9.49, 9.5, 1000) for i in range(10, 15)]
    pos = Position(
        symbol="IREN",
        qty=49.0,
        entry_px=10.2,
        stop=10.1,
        target=10.36,
        entry_ts=OPEN + 11 * 60,
        atr=0.02,
    )
    assert decide_orb5(_snap(bars), pos, LIMITS, now=OPEN + 16 * 60).reason == "stop"
    assert decide_orb5(_snap(bars), pos, LIMITS, now=OPEN + 17 * 60).reason == "in_position"


def test_completed_buckets_exclude_partial_and_premarket() -> None:
    pre = [Bar(OPEN - 600, 9.0, 9.0, 9.0, 9.0, 10)]
    bars = pre + _today({}, 7)
    b5 = completed_5m_bars(tuple(bars), OPEN + 7 * 60)
    assert [round(b.start - OPEN) for b in b5] == [0]  # 09:35 bucket still open, pre-market dropped


def test_orb5_limits() -> None:
    assert LIMITS.stop_atr == 1.0 and LIMITS.max_entries_per_day == 3
    assert LIMITS.target_r == 1.6 and LIMITS.time_stop_s == 3600 and LIMITS.notional == 500.0


def test_incomplete_signal_bucket_holds() -> None:
    bars = _breakout_day()
    bars = [b for b in bars if b.ts != OPEN + 9 * 60]  # drop the bucket's last minute
    d = decide_orb5(_snap(bars), None, LIMITS, now=OPEN + 11 * 60)
    assert d.action == "hold" and d.reason == "incomplete_bucket"


def test_price_below_five_holds() -> None:
    scale = 0.4  # same shape at ~$4
    bars = [
        Bar(b.ts, b.o * scale, b.h * scale, b.l * scale, b.c * scale, b.v) for b in _breakout_day()
    ]
    d = decide_orb5(_snap(bars), None, LIMITS, now=OPEN + 11 * 60)
    assert d.action == "hold" and d.reason == "atr_or_price_out_of_band"


def test_exploratory_late_breakout_is_separate_from_prereg_rule() -> None:
    closes = {i: 10.0 for i in range(0, 95)}
    closes.update({i: 10.30 for i in range(95, 100)})
    bars = _prior_day() + _today(closes, 100)
    snap = _snap(bars)
    legacy = decide_orb5(snap, None, LIMITS, now=OPEN + 101 * 60)
    experiment = decide_orb5_explore(snap, None, Limits(**ORB5_EXPLORE_LIMITS), now=OPEN + 101 * 60)
    assert legacy.reason == "outside_signal_window"
    assert experiment.action == "enter_long" and experiment.reason == "orb5_breakout"
    assert experiment.notional == 100.0


def test_exploratory_signal_window_also_closes() -> None:
    closes = {i: 10.0 for i in range(0, 315)}
    closes.update({i: 10.30 for i in range(315, 320)})
    bars = _prior_day() + _today(closes, 320)
    dec = decide_orb5_explore(_snap(bars), None, Limits(**ORB5_EXPLORE_LIMITS), now=OPEN + 321 * 60)
    assert dec.action == "hold" and dec.reason == "outside_signal_window"

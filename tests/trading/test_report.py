"""Tests for the deterministic daily P&L report.

The report module is a pure function of ``(ledger, bars_by_symbol, day,
seed)``: given the same inputs, two calls must return the same payload.
The tests below cover the three guarantees the brief names:

* ``test_report_baselines_deterministic`` -- with a fixed seed, two runs
  over the same bars and same ledger return identical baselines
  (``buy_and_hold_pct`` and ``random_entries_r``).
* ``test_report_counts_and_win_rate`` -- a fake ledger returning a known
  set of closed trades yields the expected ``trades`` count, ``total_r``
  and ``win_rate`` (only ``r > 0`` counts as a win).
* ``test_max_drawdown_r`` -- the drawdown is the largest peak-to-trough
  drop of the *cumulative* R curve, not the minimum of the R series.

The tests use a :class:`_FakeLedger` that exposes just
``closed_trades(day=...)`` so the report module never needs the real
SQLite-backed ledger, and the bars are plain dicts (the report accepts
both dataclass and dict rows).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from reflective_loop_runtime.trading.report import daily_report

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _day_start_utc(year: int, month: int, day: int) -> float:
    """Epoch seconds for the start of a UTC calendar day.

    The report slices bars by the half-open ``[start, start + 86400)``
    interval, so the test fixtures must use the same anchor to land
    their bars inside the report's window.
    """
    return datetime(year, month, day, tzinfo=timezone.utc).timestamp()


def _synthetic_bars(
    day_start: float,
    *,
    n: int = 60,
    base: float = 100.0,
    step: float = 0.10,
) -> list[dict[str, float]]:
    """Build ``n`` 1-minute bars with a mild uptrend so ATR is positive.

    Each bar's high/low are wide enough that the bar-to-bar true range
    is well above zero, which is what the random-baseline simulator
    needs to derive non-zero stop/target distances.
    """
    bars: list[dict[str, float]] = []
    px = base
    for i in range(n):
        ts = day_start + 60.0 * i
        o = px
        c = px + step
        h = c + 0.05
        low = o - 0.05
        bars.append({"ts": ts, "o": o, "h": h, "l": low, "c": c, "v": 1.0})
        px = c
    return bars


class _FakeLedger:
    """A stand-in for the real :class:`Ledger` with only ``closed_trades``.

    The report only ever calls ``ledger.closed_trades(day=...)`` and reads
    ``symbol``, ``entry_ts``, ``exit_ts`` and ``r_multiple`` from each row.
    Returning plain dicts keeps the fake trivial to construct.
    """

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self._rows = list(rows or [])

    def closed_trades(self, day: float | None = None) -> list[dict[str, Any]]:
        return list(self._rows)


def _trade(
    symbol: str,
    entry_ts: float,
    exit_ts: float,
    r_multiple: float,
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "entry_ts": entry_ts,
        "exit_ts": exit_ts,
        "r_multiple": r_multiple,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_report_counts_and_win_rate() -> None:
    """A 3-trade day yields the expected count, total, and win-rate.

    Win-rate is the fraction of closed trades with ``r > 0`` -- a
    scratch (``r == 0``) trade is *not* a win. ``total_r`` is the
    arithmetic sum, so a winner of +1.5 minus a loser of -1.0 plus a
    scratch gives +0.5.
    """
    day = _day_start_utc(2024, 1, 1)
    bars = _synthetic_bars(day)
    trades = [
        _trade("AAA", day + 60.0, day + 600.0, 1.5),
        _trade("AAA", day + 700.0, day + 1300.0, -1.0),
        _trade("AAA", day + 1400.0, day + 2000.0, 0.0),
    ]
    rep = daily_report(_FakeLedger(trades), {"AAA": bars}, day=day)

    # Trades echoed back with the same per-row shape.
    assert len(rep["trades"]) == 3
    assert {t["symbol"] for t in rep["trades"]} == {"AAA"}
    assert math.isclose(rep["total_r"], 0.5, rel_tol=1e-9)
    assert math.isclose(rep["win_rate"], 1.0 / 3.0, rel_tol=1e-9)

    # Baselines are present, named exactly as the spec requires, and the
    # "no trade" baseline is always zero.
    baselines = rep["baselines"]
    assert baselines["no_trade"] == 0.0
    assert isinstance(baselines["buy_and_hold_pct"], float)
    assert isinstance(baselines["random_entries_r"], list)


def test_report_baselines_deterministic() -> None:
    """Two calls with the same seed over the same bars return equal baselines.

    The random baseline must be deterministic so two operators looking at
    the same report agree. We also pin that ``buy_and_hold_pct`` is
    identical across calls because the data is identical and the function
    is a pure fold over the day's bars. As a negative control we change
    the seed and confirm the random list actually moves.
    """
    day = _day_start_utc(2024, 6, 15)
    bars_a = _synthetic_bars(day, n=120)
    bars_b = _synthetic_bars(day, n=120)
    # Two trades so the random baseline samples the same number of
    # entries; with an empty ledger the random list is always empty.
    trades = [
        _trade("AAA", day + 60.0, day + 600.0, 0.5),
        _trade("AAA", day + 700.0, day + 1300.0, -0.25),
    ]
    ledger_a = _FakeLedger(trades)
    ledger_b = _FakeLedger(trades)

    a = daily_report(ledger_a, {"AAA": bars_a}, day=day, seed=7)
    b = daily_report(ledger_b, {"AAA": bars_b}, day=day, seed=7)
    assert a["baselines"]["random_entries_r"] == b["baselines"]["random_entries_r"]
    assert math.isclose(
        a["baselines"]["buy_and_hold_pct"],
        b["baselines"]["buy_and_hold_pct"],
        rel_tol=1e-12,
        abs_tol=1e-12,
    )

    # Different seed -> different random baseline (sanity check that
    # the determinism comes from the seed, not from the data alone).
    c = daily_report(_FakeLedger(trades), {"AAA": bars_a}, day=day, seed=8)
    assert a["baselines"]["random_entries_r"] != c["baselines"]["random_entries_r"]


def test_max_drawdown_r() -> None:
    """Drawdown is the worst peak-to-trough drop of the *cumulative* R curve.

    With trades ``[+1.0, -2.0, +0.5, +0.3]`` the running total is
    ``[1.0, -1.0, -0.5, -0.2]``; the peak reached is 1.0; the lowest
    level after that peak is -1.0, so the drawdown is ``1.0 - (-1.0) =
    2.0``. A naive "min(R)" would be -2.0, which is *not* the metric
    the brief asks for.
    """
    day = _day_start_utc(2024, 3, 1)
    bars = _synthetic_bars(day)
    trades = [
        _trade("AAA", day + 60.0, day + 600.0, 1.0),
        _trade("AAA", day + 700.0, day + 1300.0, -2.0),
        _trade("AAA", day + 1400.0, day + 2000.0, 0.5),
        _trade("AAA", day + 2100.0, day + 2700.0, 0.3),
    ]
    rep = daily_report(_FakeLedger(trades), {"AAA": bars}, day=day)
    assert math.isclose(rep["max_drawdown_r"], 2.0, rel_tol=1e-9)

    # All winning -> the drawdown is zero (the path never falls below
    # its starting level, which the report treats as 0.0).
    winners = [
        _trade("AAA", day + 60.0, day + 600.0, 0.4),
        _trade("AAA", day + 700.0, day + 1300.0, 0.6),
    ]
    rep_up = daily_report(_FakeLedger(winners), {"AAA": bars}, day=day)
    assert rep_up["max_drawdown_r"] == 0.0

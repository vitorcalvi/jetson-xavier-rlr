"""Deterministic daily P&L report for the RLR paper trader.

The report is the slow lane's view of "how did yesterday go?" Three numbers
matter:

* ``total_r`` -- the sum of realised R-multiples for trades closed that day.
* ``win_rate`` -- the fraction of closed trades with ``r_multiple > 0``.
* ``max_drawdown_r`` -- the largest peak-to-trough drop in the running R
  curve across the day's trades, a focused view of how bad the *path* was.

Three baselines back the report against easier models:

* ``no_trade`` is always ``0.0`` R. Anything positive means the trader beat
  sitting on its hands.
* ``buy_and_hold_pct`` is the close-to-close return of an equal-weighted
  basket of the day's symbols (in percent). It is the simplest possible
  passive strategy, and it is the comparison most traders forget: a strategy
  has to beat "buy and do nothing", not "earn zero".
* ``random_entries_r`` resamples entry points across the day's bar stream
  under the *same* stop/target/time-stop rules the live strategy uses, with a
  seeded RNG so two runs over the same data see identical entries. It bounds
  what luck alone would have produced.

Everything here is stdlib-only: the RNG is ``random.Random(seed)`` and the
P&L comes straight from the :class:`Ledger` plus the live ``bars_by_symbol``.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .ledger import Ledger


@dataclass(frozen=True)
class ClosedTrade:
    """One closed trade by R-multiple -- enough for stats, no broker fields."""

    symbol: str
    entry_ts: float
    exit_ts: float
    r_multiple: float


# The fixed entry-fee/distances we apply to every simulated (and real) trade.
# Mirrors the brief's ``Limits`` defaults -- stop_atr=1.5, target_atr=2.5,
# time_stop_s=3600. The simulator uses these against the realised ATR of the
# entry bar so a random entry is held to the same discipline the live trader
# applies. Keeping these duplicated (rather than importing trading.loop.Limits)
# means the report module has no kernel dependency.
_DEFAULT_STOP_ATR = 1.5
_DEFAULT_TARGET_ATR = 2.5
_DEFAULT_TIME_STOP_S = 3600.0
# An arbitrary but fixed intraday slippage assumption for the simulator. It
# keeps the random baseline honest in the case where a synthetic entry fills
# above its stop: in real life the broker does not actually fill at the bar's
# open, so we shave a small amount off the immediate stop hit.
_SLIPPAGE = 0.0005


def _utc_day_window(day: float) -> tuple[float, float]:
    """Return the half-open ``[start, end)`` UTC-day epoch window for ``day``."""
    dt = datetime.fromtimestamp(float(day), tz=timezone.utc)
    start = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc).timestamp()
    return start, start + 86400.0


def _bars_in_window(bars: Sequence[Any], start: float, end: float) -> list[Any]:
    """The ``bars`` whose ``ts`` falls inside the day window, by ascending ``ts``.

    Bars may be plain ``dict`` rows or dataclass instances; we read ``ts``
    via :func:`_attr` so both shapes are supported.
    """
    keep = [b for b in bars if start <= float(_attr(b, "ts")) < end]
    keep.sort(key=lambda b: float(_attr(b, "ts")))
    return keep


def _r(entry: float, exit: float, stop: float) -> float:
    """R-multiple for a long trade: ``(exit-entry)/(entry-stop)``."""
    risk = entry - stop
    if risk == 0:
        return 0.0
    return (exit - entry) / risk


def _max_drawdown(values: Sequence[float]) -> float:
    """Largest peak-to-trough drop of ``values``; ``0.0`` if the path never falls."""
    peak = 0.0
    running = 0.0
    drawdown = 0.0
    for value in values:
        running += value
        if running > peak:
            peak = running
        diff = peak - running
        if diff > drawdown:
            drawdown = diff
    return drawdown


def _attr(row: Any, name: str, default: Any = 0.0) -> Any:
    """``row.name`` for dataclasses, ``row[name]`` for dicts; default when missing.

    Numeric values are normalised to ``float``; any other value (e.g. the
    string a trade-row carries in its ``symbol`` field) is returned as-is
    so callers can decide how to coerce it.
    """
    if isinstance(row, Mapping):
        if name in row:
            value = row[name]
            if isinstance(value, (int, float)):
                return float(value)
            return value
        return default
    value = getattr(row, name, default)
    if isinstance(value, (int, float)):
        return float(value)
    return value


def _buy_and_hold_pct(bars_by_symbol: Mapping[str, Sequence[Any]], day: float) -> float:
    """Equal-weighted basket return (in percent) for the UTC day of ``day``."""
    start, end = _utc_day_window(day)
    rets: list[float] = []
    for _symbol, bars in bars_by_symbol.items():
        window = _bars_in_window(bars, start, end)
        if len(window) < 2:
            continue
        first = _attr(window[0], "o")
        last = _attr(window[-1], "c")
        if first <= 0:
            continue
        rets.append(last / first - 1.0)
    if not rets:
        return 0.0
    return sum(rets) / len(rets) * 100.0


def _atr_proxy(bars: Sequence[Any], index: int) -> float:
    """A simple trailing-mean true-range approximation from 1-minute bars.

    The simulator has no access to the trading lane's Wilder ATR (it would
    create a circular import), so it uses a trailing-14 mean of true range
    seeded with a default. The result is good enough to drive stop/target
    distances for the random baseline.
    """
    start = max(0, index - 13)
    window = list(bars[start : index + 1])
    if len(window) < 2:
        return 0.0
    true_ranges: list[float] = []
    for i in range(1, len(window)):
        high = _attr(window[i], "h")
        low = _attr(window[i], "l")
        prev_close = _attr(window[i - 1], "c")
        true_ranges.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    if not true_ranges:
        return 0.0
    return sum(true_ranges) / len(true_ranges)


def _simulate_random(
    bars_by_symbol: Mapping[str, Sequence[Any]],
    *,
    day: float,
    n_entries: int,
    seed: int,
    stop_atr: float = _DEFAULT_STOP_ATR,
    target_atr: float = _DEFAULT_TARGET_ATR,
    time_stop_s: float = _DEFAULT_TIME_STOP_S,
) -> list[float]:
    """Sample ``n_entries`` random entries across the day and return their R's.

    The simulator must honour the same stop/target/time-stop rules the live
    strategy uses, otherwise the baseline is unfair. The same constants
    appear in :mod:`trading.loop` -- they are duplicated here on purpose to
    keep the report module free of any kernel/lane dependency.
    """
    rng = random.Random(int(seed))
    start, end = _utc_day_window(day)
    pool: list[tuple[str, int, float, float, float]] = []
    for symbol, bars in bars_by_symbol.items():
        window = _bars_in_window(bars, start, end)
        for index, bar in enumerate(window):
            o = _attr(bar, "o")
            if o <= 0:
                continue
            atr = _atr_proxy(window, index)
            if atr <= 0:
                continue
            ts = _attr(bar, "ts")
            pool.append((symbol, index, ts, o, atr))
    if not pool:
        return []
    rng.shuffle(pool)
    sampled = pool[: max(0, int(n_entries))]
    out: list[float] = []
    for symbol, _index, entry_ts, entry_px, atr in sampled:
        bars = _bars_in_window(bars_by_symbol[symbol], start, end)
        if not bars:
            continue
        stop = entry_px - atr * stop_atr
        target = entry_px + atr * target_atr
        r: float = 0.0
        exit_done = False
        for bar in bars:
            ts_bar = _attr(bar, "ts")
            if ts_bar <= entry_ts:
                continue
            low = _attr(bar, "l")
            high = _attr(bar, "h")
            if low <= stop:
                r = _r(entry_px, stop - _SLIPPAGE * entry_px, stop)
                exit_done = True
                break
            if high >= target:
                r = _r(entry_px, target - _SLIPPAGE * entry_px, stop)
                exit_done = True
                break
            if ts_bar - entry_ts >= time_stop_s:
                c = _attr(bar, "c")
                r = _r(entry_px, c, stop)
                exit_done = True
                break
        if not exit_done:
            last = bars[-1]
            r = _r(entry_px, _attr(last, "c"), stop)
        out.append(r)
    return out


def _trades_for_day(ledger: Ledger, day: float) -> list[ClosedTrade]:
    rows = ledger.closed_trades(day=day)
    out: list[ClosedTrade] = []
    for row in rows:
        out.append(
            ClosedTrade(
                symbol=str(_attr(row, "symbol", "")),
                entry_ts=_attr(row, "entry_ts", 0.0),
                exit_ts=_attr(row, "exit_ts", 0.0),
                r_multiple=_attr(row, "r_multiple", 0.0),
            )
        )
    return out


def daily_report(
    ledger: Ledger,
    bars_by_symbol: Mapping[str, Sequence[Any]],
    *,
    day: float,
    seed: int = 7,
    n_random_entries: int | None = None,
) -> dict[str, Any]:
    """Build the JSON-friendly daily P&L report.

    ``bars_by_symbol`` is a ``symbol -> bars`` mapping; ``bars`` may be a
    sequence of either :class:`Bar` instances (the canonical shape) or plain
    dicts with ``ts``, ``o``, ``h``, ``l``, ``c``, ``v``. The function
    accepts either so a unit test can avoid the dataclass import.
    """
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    trades = _trades_for_day(ledger, day)
    r_values = [t.r_multiple for t in trades]
    wins = sum(1 for r in r_values if r > 0)
    win_rate = wins / len(r_values) if r_values else 0.0
    total_r = sum(r_values)
    drawdown = _max_drawdown(r_values)

    # Random baseline: one synthetic entry per closed real trade, so the
    # baseline has the same sample-size feel. Tests override
    # ``n_random_entries`` to keep the snapshot stable.
    n_entries = len(trades) if n_random_entries is None else int(n_random_entries)
    random_r = _simulate_random(bars_by_symbol, day=day, n_entries=n_entries, seed=seed)
    buy_hold_pct = _buy_and_hold_pct(bars_by_symbol, day)

    return {
        "trades": [
            {
                "symbol": t.symbol,
                "entry_ts": t.entry_ts,
                "exit_ts": t.exit_ts,
                "r_multiple": t.r_multiple,
            }
            for t in trades
        ],
        "total_r": float(total_r),
        "win_rate": float(win_rate),
        "max_drawdown_r": float(drawdown),
        "baselines": {
            "no_trade": 0.0,
            "buy_and_hold_pct": float(buy_hold_pct),
            "random_entries_r": [float(x) for x in random_r],
        },
    }


__all__ = [
    "ClosedTrade",
    "daily_report",
]

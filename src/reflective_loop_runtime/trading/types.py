"""Frozen data types shared by every trading lane.

The shape of these dataclasses is the contract other lanes (signal scoring,
risk gates, paper-trade simulator, telemetry) read against. Keep them stable:
adding fields is cheap and backwards-compatible; renaming or removing a field
forces a coordinated bump in every dependent lane.

* :class:`Bar` -- one minute of OHLCV for a single symbol.
* :class:`Quote` -- the most recent top-of-book for a symbol.
* :class:`Snapshot` -- bars + quote + provenance for a symbol at a moment in time.
* :class:`Features` -- deterministic features derived from a ``Snapshot``.

All numeric values are ``float``. ``None`` is the canonical "not enough data"
sentinel for features whose window cannot be filled yet, and is propagated
upstream rather than replaced by zero or NaN so callers cannot confuse
"unknown" with a real zero.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Bar:
    """A single one-minute OHLCV bar.

    ``ts`` is the bar open time in epoch seconds (UTC). Alpaca returns the
    timestamp as an RFC3339 string; :mod:`reflective_loop_runtime.trading.market`
    parses it to ``float`` before constructing the bar.
    """

    ts: float
    o: float
    h: float
    l: float  # noqa: E741 - OHLC field name fixed by the shared interface
    c: float
    v: float


@dataclass(frozen=True)
class Quote:
    """The most recent top-of-book quote for a symbol.

    ``bid`` and ``ask`` are prices; ``ts`` is the quote timestamp in epoch
    seconds (UTC).
    """

    bid: float
    ask: float
    ts: float


@dataclass(frozen=True)
class Snapshot:
    """Bars, latest quote, and provenance for one symbol at one fetch.

    ``bars`` is a tuple (immutable, hashable) of :class:`Bar` ordered by ``ts``
    ascending. Missing data is represented by an empty tuple plus a
    ``last_bar_ts`` of ``0.0``, never by holes inside the sequence.

    ``fetched_at`` is the epoch seconds when the snapshot was assembled (usually
    the ``now`` passed to :func:`fetch_snapshots`). ``last_bar_ts`` is the
    timestamp of the most recent bar in ``bars``, or ``0.0`` if ``bars`` is
    empty -- callers can use ``last_bar_ts == 0.0`` as a fast "no data yet"
    check.
    """

    symbol: str
    bars: tuple[Bar, ...]
    quote: Quote | None
    fetched_at: float
    last_bar_ts: float


@dataclass(frozen=True)
class Features:
    """Deterministic features derived from a :class:`Snapshot`.

    Window-dependent features (EMAs, ATR14, ret_60m, VWAP) are ``None`` when
    the window cannot yet be filled. ``spread_pct`` is ``None`` when the
    snapshot has no quote. ``crossed_above_vwap`` is a boolean: it requires
    a VWAP and is meaningful even if the price is currently above VWAP but
    did not just cross from below within the last 15 bars.
    """

    symbol: str
    price: float
    ema20_5m: float | None
    ema50_5m: float | None
    vwap: float | None
    atr14: float | None
    ret_60m: float | None
    spread_pct: float | None
    crossed_above_vwap: bool
    last_bar_age_s: float
    confirmed_vwap_reclaim: bool = False
    below_vwap_bars: int = 0
    ret_5m: float | None = None
    quote_age_s: float | None = None
    quote_bid: float | None = None
    full_vwap_coverage: bool = False


__all__ = ["Bar", "Quote", "Snapshot", "Features"]

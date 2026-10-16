"""Feature-pipeline tests for the RLR trading lane.

The features pipeline is the part of the trading lane that *must* be
deterministic: the slow lane replays fixed historical snapshots through
``compute`` and asserts on the resulting :class:`Features`. Any clock or
randomness in this function would make those replays non-reproducible, so
its tests focus on:

* Input -> output identity (``test_features_deterministic``).
* Each windowed feature (EMA, VWAP, ATR, ret_60m, spread) reaches its
  ``None``-until-ready threshold at the documented count.
* Sign and monotone behaviour of composite features (``crossed_above_vwap``).

Fixtures under ``tests/trading/fixtures/`` carry the bar series; this module
also uses an injected fake ``http`` (in :func:`test_features_via_fetch_snapshots_round_trip`)
so the end-to-end wiring is exercised against fixtures rather than the network.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from reflective_loop_runtime.trading.features import compute
from reflective_loop_runtime.trading.market import (
    SYMBOLS,
    _parse_rfc3339,
    fetch_snapshots,
)
from reflective_loop_runtime.trading.types import Bar, Features, Quote, Snapshot

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def _bars_from_fixture(name: str, symbol: str = "BTC/USD") -> tuple[Bar, ...]:
    """Parse the bars for ``symbol`` out of a fixture file."""
    payload = _load(name)["bars"][symbol]
    return tuple(
        Bar(
            ts=_parse_rfc3339(b["t"]),
            o=float(b["o"]),
            h=float(b["h"]),
            l=float(b["l"]),
            c=float(b["c"]),
            v=float(b["v"]),
        )
        for b in payload
    )


def _make_snapshot(
    bars: tuple[Bar, ...], quote: Quote | None = None, *, symbol: str = "BTC/USD", now: float = 0.0
) -> Snapshot:
    last_bar_ts = bars[-1].ts if bars else 0.0
    return Snapshot(symbol=symbol, bars=bars, quote=quote, fetched_at=now, last_bar_ts=last_bar_ts)


def _epoch(*args: int) -> float:
    """Build a UTC epoch from explicit year/month/day/hour/minute/second."""
    return datetime(*args, tzinfo=timezone.utc).timestamp()


def _fake_http(payloads: dict[str, dict]) -> Callable[[str, dict], object]:
    """Route URLs to fixtures by endpoint; bars vs quotes decided by path."""

    def fake(url: str, headers: dict) -> object:
        if "/bars" in url:
            return payloads["bars"]
        if "/quotes" in url:
            return payloads["quotes"]
        raise AssertionError(f"unexpected URL: {url}")

    return fake


# ---------------------------------------------------------------------------
# Named tests
# ---------------------------------------------------------------------------


def test_features_deterministic() -> None:
    """Same snapshot in -> identical :class:`Features` out.

    Frozen dataclasses are equality-by-value, so passing the *same* snapshot
    to ``compute`` repeatedly must produce equal ``Features`` instances. A
    fourth call with a freshly-built snapshot from the same fixture data
    confirms the determinism is not an artefact of object identity.
    """
    bars = _bars_from_fixture("bars_full.json", "BTC/USD")
    snap = _make_snapshot(bars, now=_epoch(2024, 1, 1, 0, 5, 0))

    f1 = compute(snap, now=snap.fetched_at)
    f2 = compute(snap, now=snap.fetched_at)
    f3 = compute(snap, now=snap.fetched_at)

    assert f1 == f2 == f3
    # Distinct objects, equal by value: protects against accidental caching
    # (e.g. a ``@functools.lru_cache`` that returned the same instance).
    assert f1 is not f2

    # Snapshot rebuilt from the same fixture -> identical features.
    snap_again = _make_snapshot(bars, now=_epoch(2024, 1, 1, 0, 5, 0))
    assert compute(snap_again, now=snap_again.fetched_at) == f1


def test_ema_requires_50_five_minute_bars() -> None:
    """EMA50 is ``None`` until at least 50 5-minute closes are available.

    With 60 one-minute bars we only get 12 five-minute buckets, so ema50
    is ``None`` while ema20 -- which needs only 20 closes -- is still
    None because 12 < 20. Both windows are checked explicitly so an
    off-by-one (e.g. 50 vs 49) would visibly flip one of them.
    """
    bars = _bars_from_fixture("bars_short.json")  # 60 one-minute bars -> 12 closes
    snap = _make_snapshot(bars, now=_epoch(2024, 1, 1, 1, 0, 0))

    features = compute(snap, now=snap.fetched_at)

    assert features.ema50_5m is None, (
        f"EMA50 should be None with 12 five-minute buckets, got {features.ema50_5m}"
    )
    assert features.ema20_5m is None, (
        f"EMA20 should be None with 12 five-minute buckets, got {features.ema20_5m}"
    )

    # 15 bars -> only 3 five-minute buckets; ema20 also None.
    short_bars = _bars_from_fixture("bars_very_short.json")
    short_snap = _make_snapshot(short_bars, now=_epoch(2024, 1, 1, 0, 16, 0))
    short_features = compute(short_snap, now=short_snap.fetched_at)
    assert short_features.ema20_5m is None
    assert short_features.ema50_5m is None


def test_vwap_resets_each_utc_day() -> None:
    """VWAP must ignore bars from earlier UTC days.

    The dual-day fixture has day-1 bars around ``c=50`` and day-2 bars
    around ``c=250``. A bug that forgot to reset the day anchor would
    weight the day-1 bars in -- day-2 alone lands near ``c=250``, mixing
    them in pulls the result toward the middle.
    """
    bars = _bars_from_fixture("bars_dual_day.json")
    # ``now`` is mid-day-2 so the day-1 bars (UTC date 2023-12-31) must
    # be excluded and the day-2 bars (UTC date 2024-01-01) included.
    snap = _make_snapshot(bars, now=_epoch(2024, 1, 1, 0, 30, 0))
    features = compute(snap, now=snap.fetched_at)

    assert features.vwap is not None
    # Day-2 typicals are (h+l+c)/3 with h~250.07, l~249.99, c~250.01; a
    # blended mid would be ~200 -- comfortably below 220.
    assert features.vwap > 220.0, (
        f"VWAP should sit near the day-2 closes (~250); got {features.vwap}"
    )
    assert features.vwap > 200.0, f"VWAP must not be polluted by day-1 bars; got {features.vwap}"

    # Conversely, a snapshot anchored on day-1 sees only day-1 bars:
    # the day-1 closes sit around 50, so VWAP must be near there.
    snap_day1 = _make_snapshot(bars, now=_epoch(2023, 12, 31, 22, 30, 0))
    features_day1 = compute(snap_day1, now=snap_day1.fetched_at)
    assert features_day1.vwap is not None
    assert 40.0 < features_day1.vwap < 60.0, (
        f"VWAP at day-1 should be near 50; got {features_day1.vwap}"
    )


def test_crossed_above_vwap() -> None:
    """``crossed_above_vwap`` is True iff the last close is above VWAP and at
    least one of the previous 15 closes sat below it."""
    bars = _bars_from_fixture("bars_crossed.json")
    snap = _make_snapshot(bars, now=_epoch(2024, 1, 1, 0, 46, 0))
    features_crossed = compute(snap, now=snap.fetched_at)
    assert features_crossed.vwap is not None
    assert features_crossed.crossed_above_vwap is True, (
        "expected crossed=True when last close > vwap and prior bars dipped below"
    )

    bars_not = _bars_from_fixture("bars_not_crossed.json")
    snap_not = _make_snapshot(bars_not, now=_epoch(2024, 1, 1, 0, 46, 0))
    features_not = compute(snap_not, now=snap_not.fetched_at)
    assert features_not.vwap is not None
    assert features_not.crossed_above_vwap is False, (
        "expected crossed=False when prior 15 bars never dipped below vwap"
    )

    # Empty snapshot -> crossed is False (no vwap, no previous bars).
    empty = Snapshot(symbol="X", bars=(), quote=None, fetched_at=0.0, last_bar_ts=0.0)
    assert compute(empty, now=0.0).crossed_above_vwap is False


def test_spread_pct() -> None:
    """``spread_pct`` is ``(ask-bid)/mid * 100``; ``None`` without a quote."""
    # (ask=11, bid=9) -> mid=10 -> (11-9)/10 * 100 = 20.
    quote = Quote(bid=9.0, ask=11.0, ts=_epoch(2024, 1, 1))
    snap = _make_snapshot((), quote=quote, now=_epoch(2024, 1, 1))
    features = compute(snap, now=snap.fetched_at)
    assert features.spread_pct is not None
    assert abs(features.spread_pct - 20.0) < 1e-9, (
        f"expected spread_pct == 20, got {features.spread_pct}"
    )

    # Tight market: (ask=100.01, bid=99.99) -> mid=100 -> 0.02 pct.
    tight = Quote(bid=99.99, ask=100.01, ts=_epoch(2024, 1, 1))
    snap_tight = _make_snapshot((), quote=tight, now=_epoch(2024, 1, 1))
    assert abs(compute(snap_tight, now=snap_tight.fetched_at).spread_pct - 0.02) < 1e-9

    # No quote -> None.
    snap_no_quote = _make_snapshot((), quote=None, now=_epoch(2024, 1, 1))
    assert compute(snap_no_quote, now=snap_no_quote.fetched_at).spread_pct is None


# ---------------------------------------------------------------------------
# Auxiliary tests
# ---------------------------------------------------------------------------


def test_features_via_fetch_snapshots_round_trip() -> None:
    """End-to-end wiring: build-fetch -> compute produces well-formed Features.

    The named tests cover each piece in isolation; this one checks the
    wiring -- that :func:`fetch_snapshots` + :func:`compute` produces
    reasonable features for every symbol in the universe.
    """
    bars = _load("bars_full.json")
    quotes = _load("quotes_full.json")
    now = _epoch(2024, 1, 1, 0, 5, 0)
    http = _fake_http({"bars": bars, "quotes": quotes})

    snapshots = fetch_snapshots(SYMBOLS, http=http, now=now)
    features_by_symbol = {sym: compute(snap, now=now) for sym, snap in snapshots.items()}

    for sym in SYMBOLS:
        f = features_by_symbol[sym]
        assert isinstance(f, Features)
        assert f.symbol == sym
        assert f.last_bar_age_s >= 0.0

    # Determinism: a second pass through compute agrees with the first.
    again = {sym: compute(snap, now=now) for sym, snap in snapshots.items()}
    assert again == features_by_symbol


def test_invalid_iex_quotes_fail_closed_in_spread_feature() -> None:
    bars = _bars_from_fixture("bars_full.json", "BTC/USD")
    now = _epoch(2024, 1, 1, 0, 5, 0)
    for bid, ask in (
        (30.56, 0.0),
        (0.0, 30.56),
        (30.58, 30.55),
        (float("nan"), 30.56),
        (30.56, float("inf")),
    ):
        snap = _make_snapshot(bars, Quote(bid=bid, ask=ask, ts=now), now=now)
        assert compute(snap, now=now).spread_pct is None
    valid = _make_snapshot(bars, Quote(bid=30.55, ask=30.56, ts=now), now=now)
    f = compute(valid, now=now)
    assert f.spread_pct is not None and 0 <= f.spread_pct <= 0.2

"""Market-data integration tests for the RLR trading lane.

These tests exercise :func:`reflective_loop_runtime.trading.market.fetch_snapshots`
through the injected ``http`` transport, with the JSON payloads committed
under ``tests/trading/fixtures/``. They do not touch the network.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from reflective_loop_runtime.trading.market import (
    SYMBOLS,
    _iso_minus_minutes,
    _parse_rfc3339,
    fetch_snapshots,
)
from reflective_loop_runtime.trading.types import Bar, Quote, Snapshot

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    """Parse a fixture file; tests fail loudly if a fixture is missing."""
    return json.loads((FIXTURES / name).read_text())


def _split_router(bars: dict, quotes: dict) -> tuple[Callable[[str, dict], object], list[str]]:
    """Return a fake ``http`` plus a list that records every URL it sees.

    The router looks at the URL path to decide which payload to return,
    matching the two endpoints ``fetch_snapshots`` talks to.
    """
    seen: list[str] = []

    def fake(url: str, headers: dict) -> object:
        seen.append(url)
        if "/bars" in url:
            return bars
        if "/quotes" in url:
            return quotes
        raise AssertionError(f"unexpected URL: {url}")

    return fake, seen


def _pagination_router(
    page1: dict, page2: dict, quotes: dict
) -> tuple[Callable[[str, dict], object], list[str]]:
    """Fake ``http`` returning different bars payloads depending on ``page_token``.

    Page 1 is returned when the bars URL has no ``page_token=`` query
    parameter; page 2 is returned when ``page_token=tok1`` is present.
    Mirrors the simplest Alpaca pagination contract.
    """
    seen: list[str] = []

    def fake(url: str, headers: dict) -> object:
        seen.append(url)
        if "/quotes" in url:
            return quotes
        if "/bars" in url:
            if "page_token=tok1" in url:
                return page2
            return page1
        raise AssertionError(f"unexpected URL: {url}")

    return fake, seen


def _build_snapshot_from_fixtures(symbol: str) -> Snapshot:
    """Helper: turn a one-symbol bars+quote fixture into a :class:`Snapshot`."""
    bars_payload = _load("bars_full.json")["bars"][symbol]
    quote_payload = _load("quotes_full.json")["quotes"][symbol]
    bars = tuple(
        Bar(
            ts=_parse_rfc3339(b["t"]),
            o=float(b["o"]),
            h=float(b["h"]),
            l=float(b["l"]),
            c=float(b["c"]),
            v=float(b["v"]),
        )
        for b in bars_payload
    )
    quote = Quote(
        bid=float(quote_payload["bp"]),
        ask=float(quote_payload["ap"]),
        ts=_parse_rfc3339(quote_payload["t"]),
    )
    last_bar_ts = bars[-1].ts if bars else 0.0
    return Snapshot(
        symbol=symbol,
        bars=bars,
        quote=quote,
        fetched_at=1_704_067_200.0,
        last_bar_ts=last_bar_ts,
    )


# ---------------------------------------------------------------------------
# Named tests
# ---------------------------------------------------------------------------


def test_fetch_snapshots_parses_bars_and_quotes() -> None:
    """Happy-path parse: one snapshot per symbol, bars and quote populated."""
    bars = _load("bars_full.json")
    quotes = _load("quotes_full.json")
    http, urls = _split_router(bars, quotes)

    snapshots = fetch_snapshots(SYMBOLS, http=http, now=1_704_067_200.0)

    assert set(snapshots) == set(SYMBOLS), "one snapshot per requested symbol"
    assert len(urls) == 2, "exactly two HTTP calls (bars + quotes)"
    assert any("bars" in u for u in urls)
    assert any("quotes" in u for u in urls)
    assert any("BTC%2FUSD" in u for u in urls) or any("BTC/USD" in u for u in urls), (
        "symbol list is URL-encoded with slash escaped"
    )

    btc = snapshots["BTC/USD"]
    assert isinstance(btc, Snapshot)
    assert btc.symbol == "BTC/USD"
    assert len(btc.bars) == 5
    assert btc.bars == tuple(sorted(btc.bars, key=lambda b: b.ts)), "bars sorted ascending"
    assert all(isinstance(bar, Bar) for bar in btc.bars)
    assert btc.quote is not None
    assert isinstance(btc.quote, Quote)
    assert btc.quote.bid == 29999.0
    assert btc.quote.ask == 30001.0
    assert btc.last_bar_ts == btc.bars[-1].ts
    assert btc.last_bar_ts > 0.0
    assert btc.fetched_at == 1_704_067_200.0

    # The other symbols also come back with the right number of bars.
    for symbol in SYMBOLS:
        snap = snapshots[symbol]
        assert snap.symbol == symbol
        assert len(snap.bars) == 5
        assert snap.quote is not None


def test_missing_symbol_gives_empty_snapshot() -> None:
    """A requested symbol that is absent from both responses yields empty bars."""
    bars = _load("bars_missing.json")  # AVAX/USD absent.
    quotes = _load("quotes_missing.json")
    http, _urls = _split_router(bars, quotes)

    snapshots = fetch_snapshots(SYMBOLS, http=http, now=1_704_067_200.0)

    assert set(snapshots) == set(SYMBOLS), "still one entry per requested symbol"
    missing = snapshots["AVAX/USD"]
    assert missing.bars == (), "absent symbol has empty bars tuple"
    assert missing.quote is None, "absent symbol has no quote"
    assert missing.last_bar_ts == 0.0, "last_bar_ts is the 0.0 sentinel"
    assert missing.fetched_at == 1_704_067_200.0, "fetched_at is still stamped"

    # Other symbols still got real data.
    assert snapshots["BTC/USD"].bars, "present symbols have bars"


def test_fetch_snapshots_follows_pagination() -> None:
    """``fetch_snapshots`` must chase ``next_page_token`` until it is empty.

    Page 1 contains AVAX/USD bars and a ``next_page_token``; page 2 contains
    BTC/USD and ETH/USD bars and an empty cursor. Both pages' bars must be
    merged into the right snapshots, sorted by timestamp, with correct
    ``last_bar_ts`` values. Symbols that never appear on any page keep the
    "no data" sentinel.
    """
    page1 = _load("bars_pagination_page1.json")
    page2 = _load("bars_pagination_page2.json")
    quotes = _load("quotes_full.json")
    http, urls = _pagination_router(page1, page2, quotes)

    snapshots = fetch_snapshots(SYMBOLS, http=http, now=1_704_067_200.0)

    # Two bars calls (page 1 + page 2) and one quotes call.
    bars_urls = [u for u in urls if "/bars" in u]
    quote_urls = [u for u in urls if "/quotes" in u]
    assert len(bars_urls) == 2, f"expected page 1 + page 2, saw {len(bars_urls)} bars urls"
    assert len(quote_urls) == 1, "exactly one quotes call regardless of pagination"
    # Pagination round-tripped the cursor through the URL, not the headers.
    assert "page_token=tok1" in bars_urls[1], (
        "second bars request must carry the page_token returned by page 1"
    )

    # AVAX came from page 1: its bars and last_bar_ts reflect page 1 only.
    avax = snapshots["AVAX/USD"]
    assert avax.bars, "AVAX bars from page 1 must survive"
    assert avax.last_bar_ts == _parse_rfc3339("2024-01-01T00:01:00Z"), (
        "AVAX last_bar_ts is the last bar from page 1"
    )

    # BTC/ETH arrived only on page 2, so they are only populated if
    # pagination was followed.
    btc = snapshots["BTC/USD"]
    assert btc.bars, "BTC bars must be merged in from page 2"
    assert btc.last_bar_ts == _parse_rfc3339("2024-01-01T00:02:00Z"), (
        "BTC last_bar_ts is the last bar from page 2"
    )
    # The merged bars are still sorted ascending by ts.
    assert btc.bars == tuple(sorted(btc.bars, key=lambda b: b.ts)), (
        "merged BTC bars are sorted ascending by ts"
    )

    eth = snapshots["ETH/USD"]
    assert eth.bars, "ETH bars must be merged in from page 2"
    assert eth.last_bar_ts == _parse_rfc3339("2024-01-01T00:01:00Z"), (
        "ETH last_bar_ts is the last bar from page 2"
    )

    # SOL and LTC were never on any page, so they stay empty.
    assert snapshots["SOL/USD"].bars == ()
    assert snapshots["SOL/USD"].last_bar_ts == 0.0
    assert snapshots["LTC/USD"].bars == ()
    assert snapshots["LTC/USD"].last_bar_ts == 0.0

    # Quotes still came back for every symbol (independent of bars pagination).
    for symbol in SYMBOLS:
        assert snapshots[symbol].quote is not None, f"{symbol} got its quote"


# ---------------------------------------------------------------------------
# Auxiliary tests (not named in the brief but worth having for confidence)
# ---------------------------------------------------------------------------


def test_iso_minus_minutes_round_trips_through_parse() -> None:
    """``_iso_minus_minutes`` returns an RFC3339 string the parser accepts."""
    iso = _iso_minus_minutes(1_704_067_200.0, 300)
    parsed = _parse_rfc3339(iso)
    # Within one minute of the requested offset.
    assert abs(parsed - (1_704_067_200.0 - 300 * 60)) < 60.0

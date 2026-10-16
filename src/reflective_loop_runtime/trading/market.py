"""Alpaca crypto market data ingestion for the RLR paper trader.

This module owns the network boundary of the trading lane: it builds the
Alpaca endpoints, parses the responses into the frozen shapes in
:mod:`reflective_loop_runtime.trading.types`, and assembles a
:class:`Snapshot` per symbol. Callers inject an ``http`` transport so a fake
can stand in for urllib in tests; the production default lives in
:mod:`reflective_loop_runtime.trading.http` and is not pulled in here to keep
tests offline by construction.

Endpoints
---------
* Bars:   ``GET .../v1beta3/crypto/us/bars?symbols=...&timeframe=1Min&limit=n&start=<iso>[&page_token=<t>]``
* Quotes: ``GET .../v1beta3/crypto/us/latest/quotes?symbols=...``

Response shapes (only the keys we read are documented here):

* ``{"bars": {sym: [{"t","o","h","l","c","v",...}]}}``
* ``{"next_page_token": "..." | None}``  -- opaque cursor returned by the
  bars endpoint when the requested window spans more than one page.
* ``{"quotes": {sym: {"bp","ap","t"}}}``

Design notes
------------
* ``last_bar_ts`` is the timestamp of the most recent bar in ``bars``, or
  ``0.0`` when no bar was returned -- it is the canonical "no data yet"
  sentinel and let downstream feature code skip a window cleanly.
* A symbol in the request but missing from both responses yields a
  :class:`Snapshot` with an empty bars tuple and a ``None`` quote; the rest
  of the pipeline is safe to run on an empty snapshot.
* All timestamps returned by Alpaca are RFC3339 UTC. We parse them once
  here into epoch ``float``s so downstream feature code never touches a
  string.
* Pagination: bars are requested with ``limit=10000`` (the Alpaca API
  maximum) and we chase ``next_page_token`` until it is missing/empty or we
  hit a hard 20-page safety cap, merging each page's bars per symbol and
  relying on :func:`_parse_bars` to enforce time order. Quotes are a single
  response containing every requested symbol, so they are not paginated.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import datetime, timezone
from urllib.parse import quote

from reflective_loop_runtime.trading.types import Bar, Quote, Snapshot

# The default paper-trading universe. Order is preserved for deterministic
# iteration in callers and logs.
SYMBOLS: tuple[str, ...] = (
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
    "LTC/USD",
    "AVAX/USD",
)

# Alpaca's REST endpoint. Crypto is on the ``v1beta3`` namespace; US equities
# are on ``v2/stocks`` (free IEX feed). Both return the same JSON shapes.
_BASE_URL = "https://data.alpaca.markets"


_EQUITY_LOOKBACK_MIN = 4 * 24 * 60


def asset_class() -> str:
    """``crypto`` (default) or ``us_equity``, from ``RLR_TRADER_ASSET_CLASS``."""
    value = os.environ.get("RLR_TRADER_ASSET_CLASS", "crypto").strip().lower()
    if value not in ("crypto", "us_equity"):
        raise ValueError(f"unsupported RLR_TRADER_ASSET_CLASS: {value!r}")
    return value


def configured_symbols() -> tuple[str, ...]:
    """Symbols from ``RLR_TRADER_SYMBOLS`` (comma-separated), else :data:`SYMBOLS`."""
    raw = os.environ.get("RLR_TRADER_SYMBOLS", "")
    syms = tuple(s.strip().upper() for s in raw.split(",") if s.strip())
    return syms or SYMBOLS

# Bars request page size: Alpaca's documented maximum for a single call.
# Keeping this fixed makes pagination predictable: one ``limit=10000`` page
# covers ~6.9 days of one-minute bars, so the 20-page safety cap below
# comfortably covers any realistic backfill window.
_BARS_PAGE_LIMIT = 10000

# Hard cap on bars pagination rounds. Each round costs an HTTP call, so we
# refuse to chase a runaway cursor forever.
_MAX_BARS_PAGES = 20

# Callable injected for transport. Returns the parsed JSON body.
HttpCallable = Callable[[str, dict], object]


def _parse_rfc3339(timestamp: str) -> float:
    """Parse an RFC3339 UTC timestamp (with trailing ``Z``) into epoch seconds.

    Alpaca emits timestamps like ``2024-01-02T03:04:05Z`` or with fractional
    seconds. ``datetime.fromisoformat`` in Python 3.10 chokes on the trailing
    ``Z``, so it is normalised to ``+00:00`` first; this also works on 3.11+.
    """
    if not isinstance(timestamp, str) or not timestamp:
        raise ValueError(f"timestamp must be a non-empty string; got {timestamp!r}")
    normalised = timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
    try:
        dt = datetime.fromisoformat(normalised)
    except ValueError as exc:
        raise ValueError(f"timestamp {timestamp!r} is not RFC3339") from exc
    if dt.tzinfo is None:
        # Alpaca always emits UTC; if a future response strips the offset we
        # still want to interpret it as UTC rather than crash.
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _iso_minus_minutes(now: float, minutes: int) -> str:
    """Format ``now - minutes`` as an RFC3339 UTC timestamp with ``Z`` suffix.

    Alpaca's ``start`` parameter accepts ISO timestamps and uses them as a
    hint, not a hard filter, so precision matters less than correctness.
    """
    seconds = float(now) - float(minutes) * 60.0
    dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
    # ``isoformat`` already includes ``+00:00``; replace with ``Z`` to match
    # Alpaca's own wire format and round to whole seconds.
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _bars_url(
    symbols: tuple[str, ...],
    limit: int,
    start_iso: str,
    page_token: str | None = None,
) -> str:
    """Build the bars endpoint URL, optionally appending a pagination cursor.

    The ``page_token`` is forwarded verbatim when supplied; Alpaca returns
    one as ``next_page_token`` in a paginated response and we re-request
    with ``&page_token=<token>`` until the cursor stops being present.
    """
    syms = quote(",".join(symbols), safe="")
    if asset_class() == "us_equity":
        url = (
            f"{_BASE_URL}/v2/stocks/bars"
            f"?symbols={syms}&timeframe=1Min&limit={int(limit)}&start={quote(start_iso, safe='')}"
            "&feed=iex"
        )
    else:
        url = (
            f"{_BASE_URL}/v1beta3/crypto/us/bars"
            f"?symbols={syms}&timeframe=1Min&limit={int(limit)}&start={quote(start_iso, safe='')}"
        )
    if page_token:
        url += f"&page_token={quote(page_token, safe='')}"
    return url


def _quotes_url(symbols: tuple[str, ...]) -> str:
    syms = quote(",".join(symbols), safe="")
    if asset_class() == "us_equity":
        return f"{_BASE_URL}/v2/stocks/quotes/latest?symbols={syms}&feed=iex"
    return f"{_BASE_URL}/v1beta3/crypto/us/latest/quotes?symbols={syms}"


def _parse_bars(symbol: str, raw_bars: object) -> tuple[Bar, ...]:
    """Turn a list of raw bar dicts into a sorted tuple of :class:`Bar`."""
    if not isinstance(raw_bars, list) or not raw_bars:
        return ()
    parsed: list[Bar] = []
    for entry in raw_bars:
        if not isinstance(entry, dict):
            continue
        try:
            parsed.append(
                Bar(
                    ts=_parse_rfc3339(entry["t"]),
                    o=float(entry["o"]),
                    h=float(entry["h"]),
                    l=float(entry["l"]),
                    c=float(entry["c"]),
                    v=float(entry["v"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            # Bad rows are dropped, not raised: a single malformed bar should
            # not black out an entire symbol.
            continue
    parsed.sort(key=lambda b: b.ts)
    return tuple(parsed)


def _parse_quote(symbol: str, raw: object) -> Quote | None:
    """Turn a raw quote dict into a :class:`Quote`, or ``None`` on absence."""
    if not isinstance(raw, dict):
        return None
    try:
        bid = float(raw["bp"])
        ask = float(raw["ap"])
        ts = _parse_rfc3339(raw["t"])
    except (KeyError, TypeError, ValueError):
        return None
    return Quote(bid=bid, ask=ask, ts=ts)


def _extract_next_page_token(response: object) -> str | None:
    """Return a non-empty ``next_page_token`` from a bars response, else ``None``."""
    if not isinstance(response, dict):
        return None
    token = response.get("next_page_token")
    if isinstance(token, str) and token:
        return token
    return None


def fetch_snapshots(
    symbols: tuple[str, ...] = SYMBOLS,
    *,
    http: HttpCallable,
    now: float,
    limit: int = 300,
    headers: dict[str, str] | None = None,
) -> dict[str, Snapshot]:
    """Fetch bars and quotes for ``symbols`` and assemble a Snapshot each.

    Parameters
    ----------
    symbols:
        Symbols to fetch. Defaults to :data:`SYMBOLS`.
    http:
        Callable ``http(url, headers) -> object`` returning parsed JSON.
        Tests inject a fake; production passes the urllib-backed default.
    now:
        Reference epoch (seconds) used to compute the bars ``start`` window
        and stamped onto each snapshot as ``fetched_at``.
    limit:
        Window size in minutes used to compute ``start = now - limit``.
        Defaults to ``300`` (five hours) -- enough to seed two hundred and
        fifty minutes of five-minute closes, a 60-minute return window, and
        a fresh Wilder ATR(14). The bars page size itself is fixed at the
        Alpaca maximum (``10000``); ``limit`` only narrows the start hint.

    Returns
    -------
    dict[str, Snapshot]
        One snapshot per requested symbol. Symbols missing from every
        paginated response still appear, with empty bars and ``None``
        quote.
    """
    if not isinstance(symbols, tuple):
        # Force concrete iteration and stable order even if a caller passes a
        # list. A ``set`` would lose order and is not permitted at the type
        # level but we don't trust callers.
        symbols = tuple(symbols)
    # Crypto data is public; US-equity data needs the Alpaca key headers.
    headers = dict(headers or {})

    # Stocks only trade ~6.5h a day, so a minutes-based window would hold
    # no history at the open; look back 4 calendar days (covers a weekend).
    if asset_class() == "crypto":
        # Include all today's UTC bars for a genuine anchored VWAP, plus
        # 300 minutes from yesterday so 5m EMA50 can be initialized at 00:00.
        minutes_today = int(float(now) % 86400 // 60)
        lookback = max(int(limit), minutes_today + 300)
    else:
        lookback = max(int(limit), _EQUITY_LOOKBACK_MIN)
    start_iso = _iso_minus_minutes(float(now), lookback)
    quotes_response = http(_quotes_url(symbols), dict(headers))

    # Bars: fetch the first page with the API-maximum page size, then chase
    # ``next_page_token`` until Alpaca stops returning one or we hit the
    # safety cap. Each page's per-symbol bar lists are concatenated here and
    # later sorted by timestamp inside :func:`_parse_bars`, so merge order
    # across pages does not need to be preserved.
    raw_bars_per_symbol: dict[str, list[object]] = {}
    page_token: str | None = None
    for _ in range(_MAX_BARS_PAGES):
        bars_response = http(
            _bars_url(symbols, _BARS_PAGE_LIMIT, start_iso, page_token),
            dict(headers),
        )
        if isinstance(bars_response, dict):
            raw_bars = bars_response.get("bars")
            if isinstance(raw_bars, dict):
                for symbol, value in raw_bars.items():
                    if isinstance(symbol, str) and isinstance(value, list):
                        raw_bars_per_symbol.setdefault(symbol, []).extend(value)
        next_token = _extract_next_page_token(bars_response)
        if next_token is None:
            break
        page_token = next_token

    quotes_map: dict[str, object] = {}
    if isinstance(quotes_response, dict):
        raw_quotes = quotes_response.get("quotes")
        if isinstance(raw_quotes, dict):
            for symbol, value in raw_quotes.items():
                if isinstance(symbol, str):
                    quotes_map[symbol] = value

    snapshots: dict[str, Snapshot] = {}
    for symbol in symbols:
        bars = _parse_bars(symbol, raw_bars_per_symbol.get(symbol))
        quote = _parse_quote(symbol, quotes_map.get(symbol))
        last_bar_ts = bars[-1].ts if bars else 0.0
        snapshots[symbol] = Snapshot(
            symbol=symbol,
            bars=bars,
            quote=quote,
            fetched_at=float(now),
            last_bar_ts=last_bar_ts,
        )
    return snapshots


__all__ = ["SYMBOLS", "fetch_snapshots", "_parse_rfc3339", "_iso_minus_minutes"]

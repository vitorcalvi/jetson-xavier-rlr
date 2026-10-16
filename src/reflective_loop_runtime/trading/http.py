"""Default HTTP transport for the trading lane.

Every other module in :mod:`reflective_loop_runtime.trading` accepts an
``http`` callable rather than importing an HTTP client directly, so tests can
substitute a fake transport and the production code can swap clients without
touching the call sites. This module provides the production default:

    http_get(url: str, headers: dict[str, str]) -> dict | list
    http_post(url: str, headers: dict[str, str], body: dict) -> dict | list

returning the parsed JSON body. It is stdlib-only (urllib + json), which keeps
the runtime core -- and the trading lane -- free of optional third-party
dependencies.

The default ignores ``headers``; they are accepted so the signature matches
the injected callable and a future client (with auth headers, e.g. Alpaca
API keys) can slot in without changing call sites.

Egress guard
------------
The single allowed egress target for the trader lane is HTTPS to the two
Alpaca hosts (``data.alpaca.markets`` for crypto market data and
``paper-api.alpaca.markets`` for paper order routing). Any other scheme or
host is refused before any I/O -- the trader is not permitted to call out
to the public internet, and a misconfigured ``http()`` cannot accidentally
become a generic web client. The guard raises :class:`ValueError` so the
refusal surfaces as a configuration error rather than a runtime exception.
"""

from __future__ import annotations

import json
import time
import ssl
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

# The only two hosts the paper trader is permitted to talk to. Anything else
# is a misconfiguration and must be refused before any socket I/O.
ALLOWED_HOSTS = frozenset({"data.alpaca.markets", "paper-api.alpaca.markets"})

# The hard transport timeout for any single call. Anything tighter is a config
# choice the caller should make at the URL level, not here.
_DEFAULT_TIMEOUT_SECONDS = 15.0


def _validate_egress(url: str) -> str:
    """Validate ``url`` against the egress allow-list; return the parsed URL.

    Raises :class:`ValueError` for any URL whose scheme is not ``https`` or
    whose host is not in :data:`ALLOWED_HOSTS`. The check is performed before
    any socket I/O so a misconfigured URL never opens a connection.
    """
    if not isinstance(url, str) or not url:
        raise ValueError("http url must be a non-empty string")
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme != "https":
        raise ValueError(
            f"refusing non-https egress: scheme={scheme!r} host={parsed.hostname!r}"
        )
    host = (parsed.hostname or "").lower()
    if host not in ALLOWED_HOSTS:
        raise ValueError(
            f"refusing non-alpaca egress: host={host!r} "
            f"(allowed={sorted(ALLOWED_HOSTS)})"
        )
    return url


def _default_http(url: str, headers: dict[str, str]) -> Any:
    """GET ``url`` and return the parsed JSON body.

    Egress is restricted to ``https`` against :data:`ALLOWED_HOSTS`; see the
    module docstring for the rationale. Network or decode errors are surfaced
    as :class:`RuntimeError` with the original exception chained so callers in
    :mod:`reflective_loop_runtime.trading.market` can wrap them uniformly
    and the test fake never has to handle them.
    """
    safe_url = _validate_egress(url)
    request = Request(safe_url)
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    # Only retry safe/idempotent GET reads, never order-submitting POST.
    # This handles transient DNS/intermittent TLS handshake failures while
    # retaining stdlib certificate validation on EVERY attempt.
    # Authentication, non-2xx HTTP, and persistent certificate failures remain
    # hard errors. No fallback hosts, cert verification bypass or HTTP downgrade.
    for attempt in range(3):
        try:
            with urlopen(
                request, timeout=_DEFAULT_TIMEOUT_SECONDS  # noqa: S310 - validated
            ) as response:
                raw = response.read()
            break
        except HTTPError as exc:
            raise RuntimeError(
                f"http transport failed for {safe_url!r}: http_status={exc.code}"
            ) from exc
        except (URLError, ssl.SSLError, TimeoutError, OSError) as exc:
            if attempt >= 2:
                raise RuntimeError(
                    f"http transport failed for {safe_url!r}: transient_get_exhausted:"
                    f"{type(exc).__name__}"
                ) from exc
            time.sleep((0.2, 0.5)[attempt])
    try:
        return json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            f"http transport returned non-JSON for {safe_url!r}"
        ) from exc


def _default_post(url: str, headers: dict[str, str], body: dict) -> Any:
    """POST ``body`` as JSON to ``url`` and return the parsed JSON body.

    Egress is restricted to ``https`` against :data:`ALLOWED_HOSTS` and the
    validation runs before any socket I/O so a misconfigured URL never opens
    a connection -- this is the same guard :func:`_default_http` uses. The
    body is JSON-encoded with :func:`json.dumps`, ``Content-Type:
    application/json`` is added to the request, and the hard transport
    timeout matches :data:`_DEFAULT_TIMEOUT_SECONDS` so a hung broker cannot
    stall the trader loop. Network errors, including HTTP non-2xx responses
    from the broker, are surfaced as :class:`RuntimeError` with the HTTP
    status and response body text included so Alpaca rejections are visible
    to the caller and to the audit log without a second round-trip.
    """
    safe_url = _validate_egress(url)
    encoded = json.dumps(body).encode("utf-8")
    request = Request(safe_url, data=encoded, method="POST")
    request.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    try:
        with urlopen(
            request, timeout=_DEFAULT_TIMEOUT_SECONDS  # noqa: S310 - URL validated above
        ) as response:
            raw = response.read()
    except HTTPError as exc:
        # HTTPError carries the response body on `.read()`; surface it so a
        # 4xx/5xx from Alpaca (e.g. insufficient buying power, unknown
        # symbol) is visible in the error message instead of swallowed.
        try:
            body_text = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - best effort; never let logging break the loop
            body_text = "<unreadable>"
        raise RuntimeError(
            f"http transport failed for {safe_url!r}: "
            f"status={exc.code} body={body_text!r}: {exc}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"http transport failed for {safe_url!r}: {exc}") from exc
    try:
        return json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            f"http transport returned non-JSON for {safe_url!r}"
        ) from exc


__all__ = ["ALLOWED_HOSTS", "_default_http", "_default_post"]

"""Tests for the trading lane's hard-bounded HTTP egress module.

The trading lane is only permitted to talk to the two Alpaca hosts over HTTPS;
this module is the single place that enforcement lives. The tests here pin
the refusal shapes so a future refactor cannot quietly widen the egress
allow-list.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.request import Request

import pytest

from reflective_loop_runtime.trading.http import (
    ALLOWED_HOSTS,
    _default_http,
    _default_post,
)


def _raise_urlopen(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("urlopen must not be reached for a refused URL")


def test_http_refuses_non_alpaca_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """A URL whose host is not in :data:`ALLOWED_HOSTS` is refused pre-I/O."""
    monkeypatch.setattr(
        "reflective_loop_runtime.trading.http.urlopen", _raise_urlopen
    )
    bad = "https://example.com/v1beta3/crypto/us/bars?symbols=BTC/USD"
    with pytest.raises(ValueError) as excinfo:
        _default_http(bad, {})
    # The refusal message names the offending host and the allow-list so the
    # caller can see what was rejected without poking into the code.
    assert "example.com" in str(excinfo.value)
    assert "data.alpaca.markets" in str(excinfo.value)


def test_http_refuses_plain_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-https scheme is refused even when the host is otherwise allowed."""
    monkeypatch.setattr(
        "reflective_loop_runtime.trading.http.urlopen", _raise_urlopen
    )
    bad = "http://data.alpaca.markets/v1beta3/crypto/us/bars"
    with pytest.raises(ValueError) as excinfo:
        _default_http(bad, {})
    assert "https" in str(excinfo.value).lower()


def test_http_refuses_empty_url() -> None:
    """An empty URL is a config error and must not silently no-op."""
    with pytest.raises(ValueError):
        _default_http("", {})


def test_http_refuses_non_string_url() -> None:
    """A non-string URL must be refused rather than coerced to a request."""
    with pytest.raises(ValueError):
        _default_http(None, {})  # type: ignore[arg-type]


def test_allowed_hosts_contains_only_alpaca_hosts() -> None:
    """The allow-list is the two Alpaca hosts and nothing else."""
    assert ALLOWED_HOSTS == frozenset(
        {"data.alpaca.markets", "paper-api.alpaca.markets"}
    )


def test_post_refuses_non_alpaca_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """A POST whose host is not in :data:`ALLOWED_HOSTS` is refused pre-I/O."""
    monkeypatch.setattr(
        "reflective_loop_runtime.trading.http.urlopen", _raise_urlopen
    )
    bad = "https://example.com/v2/orders"
    body = {"symbol": "BTCUSD", "side": "buy", "notional": 10.0}
    with pytest.raises(ValueError) as excinfo:
        _default_post(bad, {}, body)
    # The refusal message names the offending host and the allow-list so the
    # caller can see what was rejected without poking into the code.
    assert "example.com" in str(excinfo.value)
    assert "paper-api.alpaca.markets" in str(excinfo.value)


def test_post_sends_json_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """POST serialises ``body`` as JSON, sets Content-Type, and uses POST method."""
    captured: dict[str, Any] = {}

    class _FakeResponse:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        def read(self) -> bytes:
            return self._payload

        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, *_exc: Any) -> None:
            return None

    def _fake_urlopen(req: Request, *_args: Any, **_kwargs: Any) -> _FakeResponse:
        captured["request"] = req
        captured["timeout"] = _kwargs.get("timeout")
        return _FakeResponse(b'{"id": "order-1", "status": "accepted"}')

    monkeypatch.setattr(
        "reflective_loop_runtime.trading.http.urlopen", _fake_urlopen
    )

    url = "https://paper-api.alpaca.markets/v2/orders"
    headers = {"APCA-API-KEY-ID": "PKfake", "APCA-API-SECRET-KEY": "secret"}
    body = {"symbol": "BTCUSD", "side": "buy", "notional": 25.0}
    result = _default_post(url, headers, body)

    req = captured["request"]
    assert isinstance(req, Request)
    assert req.get_method() == "POST"
    assert req.get_full_url() == url
    # Content-Type is set on the request before caller headers so a caller
    # that overrides it cannot silently drop the JSON framing.
    assert req.get_header("Content-type") == "application/json"
    # The body must round-trip through JSON unchanged.
    sent = json.loads(req.data.decode("utf-8"))  # type: ignore[arg-type]
    assert sent == body
    # Caller headers (Alpaca auth) are still attached. ``add_header``
    # normalises the name through ``str.capitalize`` (Python 3.10+ stores it
    # under the capitalised key in ``Request.headers``), so look them up by
    # the same form.
    assert req.get_header("Apca-api-key-id") == "PKfake"
    assert req.get_header("Apca-api-secret-key") == "secret"
    # The parsed JSON body from the broker is returned.
    assert result == {"id": "order-1", "status": "accepted"}

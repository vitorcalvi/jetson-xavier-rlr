# ruff: noqa: F811  (pytest fixture fake_features is imported from test_loop)
"""US-equity mode: IEX stock endpoints, day orders, regular-session gating."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from reflective_loop_runtime.trading import market, session
from reflective_loop_runtime.trading.broker import PaperBroker
from reflective_loop_runtime.trading.ledger import Ledger
from reflective_loop_runtime.trading.loop import Decision, Limits, TraderLoop

from .test_loop import (  # noqa: F401 - fake_features is a fixture
    _approve,
    _build_broker,
    _make_fake_order,
    _noop_market,
    fake_features,
)


def _epoch(iso: str) -> float:
    return datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).timestamp()


# 2026-10-12 is EDT (UTC-4): 13:35Z == 09:35 ET.
MON_0934 = _epoch("2026-10-12T13:34:00")
MON_0935 = _epoch("2026-10-12T13:35:00")
MON_1544 = _epoch("2026-10-12T19:44:00")
MON_1545 = _epoch("2026-10-12T19:45:00")
MON_1555 = _epoch("2026-10-12T19:55:00")
MON_1601 = _epoch("2026-10-12T20:01:00")
SAT_1100 = _epoch("2026-10-10T15:00:00")


def test_equity_urls_use_stocks_iex(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RLR_TRADER_ASSET_CLASS", "us_equity")
    bars = market._bars_url(("IREN",), 10, "2026-10-12T13:00:00Z")
    quotes = market._quotes_url(("IREN",))
    assert bars.startswith("https://data.alpaca.markets/v2/stocks/bars?symbols=IREN")
    assert "feed=iex" in bars
    assert quotes == "https://data.alpaca.markets/v2/stocks/quotes/latest?symbols=IREN&feed=iex"


def test_crypto_default_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RLR_TRADER_ASSET_CLASS", raising=False)
    monkeypatch.delenv("RLR_TRADER_SYMBOLS", raising=False)
    assert "/v1beta3/crypto/us/bars" in market._bars_url(("BTC/USD",), 5, "2026-10-12T00:00:00Z")
    assert market.configured_symbols() == market.SYMBOLS


def test_invalid_asset_class_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RLR_TRADER_ASSET_CLASS", "forex")
    with pytest.raises(ValueError):
        market.asset_class()


def test_configured_symbols_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RLR_TRADER_SYMBOLS", " iren ")
    assert market.configured_symbols() == ("IREN",)


def _broker(tif: str = "gtc", get=None, post=None) -> PaperBroker:
    return PaperBroker(
        "PKTESTKEY", "secret",
        http_get=get or (lambda u, h: {}),
        http_post=post or (lambda u, h, b: {"id": "o1"}),
        time_in_force=tif,
    )


def test_time_in_force_day_in_body() -> None:
    sent: list[dict] = []
    b = _broker("day", post=lambda u, h, body: sent.append(body) or {"id": "o1"})
    b.submit(symbol="IREN", side="buy", notional=100.0, client_order_id="c1")
    assert sent[0]["time_in_force"] == "day"


def test_time_in_force_invalid_rejected() -> None:
    with pytest.raises(ValueError):
        _broker("ioc")


def test_market_is_open_fail_closed() -> None:
    assert _broker(get=lambda u, h: {"is_open": True}).market_is_open() is True
    assert _broker(get=lambda u, h: {"is_open": False}).market_is_open() is False
    assert _broker(get=lambda u, h: {}).market_is_open() is False

    def boom(u, h):
        raise OSError("down")

    assert _broker(get=boom).market_is_open() is False


def test_entry_window_et() -> None:
    assert not session.entry_allowed(MON_0934)
    assert session.entry_allowed(MON_0935)
    assert session.entry_allowed(MON_1544)
    assert not session.entry_allowed(MON_1545)
    assert not session.entry_allowed(SAT_1100)


def test_flatten_window() -> None:
    assert not session.must_flatten(MON_1544)
    assert session.must_flatten(MON_1555)
    assert not session.must_flatten(MON_1601)  # after the close: no orders


def _loop(tmp_path: Path, fake_features, decide_fn, *, market_open=True, calls=None):
    calls = calls if calls is not None else []
    order = _make_fake_order(calls, price_box=fake_features)
    service, broker = _build_broker(tmp_path, order)
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")
    loop = TraderLoop(
        market_fn=_noop_market,
        broker=broker,
        permits=service,
        tool_broker=broker,
        ledger=ledger,
        limits=Limits(),
        heartbeat_path=tmp_path / "hb.json",
        kill_path=tmp_path / "KILL",
        decide_fn=decide_fn,
        review_fn=_approve,
        symbols=("BTC/USD",),
        equity_session=True,
        market_open_fn=lambda: market_open,
    )
    return loop, ledger, calls


def _enter(features, position, limits, *, now):
    if position is None:
        return Decision(symbol="BTC/USD", action="enter_long", reason="t",
                        stop=90.0, target=110.0, notional=100.0)
    return Decision("BTC/USD", "hold", "keep")


def test_loop_vetoes_entry_outside_session(tmp_path: Path, fake_features) -> None:
    loop, ledger, calls = _loop(tmp_path, fake_features, _enter)
    loop.tick(MON_0934)
    loop.tick(MON_1545)
    assert calls == []
    assert '"veto": "session_closed"' in ledger.journal_path().read_text()
    ledger.close()


def test_loop_vetoes_entry_when_clock_closed(tmp_path: Path, fake_features) -> None:
    loop, ledger, calls = _loop(tmp_path, fake_features, _enter, market_open=False)
    loop.tick(MON_0935)
    assert calls == []
    ledger.close()


def test_loop_enters_in_session_and_forces_exit_at_1555(tmp_path: Path, fake_features) -> None:
    loop, ledger, calls = _loop(tmp_path, fake_features, _enter)
    fake_features["price"] = 100.0
    loop.tick(MON_0935)
    assert len(calls) == 1 and calls[0]["side"] == "buy"
    loop.tick(MON_1544)  # decider holds: no exit yet
    assert len(calls) == 1
    loop.tick(MON_1555)
    assert len(calls) == 2 and calls[1]["side"] == "sell"
    closed = ledger.closed_trades()
    assert len(closed) == 1 and closed[0]["exit_reason"] == "session_close"
    ledger.close()


def test_crypto_mode_has_no_session_gate(tmp_path: Path, fake_features) -> None:
    calls: list[dict] = []
    order = _make_fake_order(calls, price_box=fake_features)
    service, broker = _build_broker(tmp_path, order)
    ledger = Ledger(tmp_path / "t.db", tmp_path / "j.jsonl")
    loop = TraderLoop(
        market_fn=_noop_market, broker=broker, permits=service, tool_broker=broker,
        ledger=ledger, limits=Limits(), heartbeat_path=tmp_path / "hb.json",
        kill_path=tmp_path / "KILL", decide_fn=_enter, review_fn=_approve,
        symbols=("BTC/USD",),
    )
    loop.tick(SAT_1100)
    assert len(calls) == 1
    ledger.close()


def test_fetch_snapshots_forwards_auth_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RLR_TRADER_ASSET_CLASS", "us_equity")
    seen: list[dict] = []

    def http(url: str, headers: dict) -> dict:
        seen.append(dict(headers))
        return {"quotes": {}} if "quotes" in url else {"bars": {}}

    market.fetch_snapshots(("IREN",), http=http, now=MON_0935,
                           headers={"APCA-API-KEY-ID": "PKX", "APCA-API-SECRET-KEY": "s"})
    assert seen and all(h.get("APCA-API-KEY-ID") == "PKX" for h in seen)


def test_equity_lookback_covers_previous_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RLR_TRADER_ASSET_CLASS", "us_equity")
    urls: list[str] = []

    def http(url: str, headers: dict) -> dict:
        urls.append(url)
        return {"quotes": {}} if "quotes" in url else {"bars": {}}

    market.fetch_snapshots(("IREN",), http=http, now=MON_0935)
    bars_url = next(u for u in urls if "/bars" in u)
    # 4 days before Monday 09:35 ET reaches back into Thursday
    assert "start=2026-10-08T13%3A35" in bars_url


MON_1550 = _epoch("2026-10-12T19:50:00")
MON_1551 = _epoch("2026-10-12T19:51:00")


def test_flatten_runs_even_when_market_data_fails(tmp_path: Path, fake_features) -> None:
    calls: list[dict] = []
    loop, ledger, calls = _loop(tmp_path, fake_features, _enter, calls=calls)
    fake_features["price"] = 100.0
    loop.tick(MON_0935)
    assert len(calls) == 1

    def broken_market(**kw):
        raise OSError("data feed down")

    loop._market_fn = broken_market
    loop.tick(MON_1550)
    assert len(calls) == 2 and calls[1]["side"] == "sell"
    assert ledger.closed_trades()[0]["exit_reason"] == "session_close"
    ledger.close()


def test_flatten_retries_after_failed_sell(tmp_path: Path, fake_features, monkeypatch) -> None:
    loop, ledger, calls = _loop(tmp_path, fake_features, _enter)
    fake_features["price"] = 100.0
    loop.tick(MON_0935)
    real_execute = loop._execute
    attempts = {"n": 0}

    def flaky_execute(*a, **k):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("sell rejected")
        return real_execute(*a, **k)

    monkeypatch.setattr(loop, "_execute", flaky_execute)
    loop._market_fn = lambda **kw: (_ for _ in ()).throw(OSError("down"))
    loop.tick(MON_1550)
    assert ledger.closed_trades() == []
    assert '"kind": "session_flatten"' in ledger.journal_path().read_text()
    loop.tick(MON_1551)
    assert len(ledger.closed_trades()) == 1
    ledger.close()


def test_position_after_close_raises_alert(tmp_path: Path, fake_features, monkeypatch) -> None:
    loop, ledger, calls = _loop(tmp_path, fake_features, _enter)
    fake_features["price"] = 100.0
    loop.tick(MON_0935)
    monkeypatch.setattr(loop, "_execute", lambda *a, **k: False)
    loop.tick(MON_1601)
    assert '"kind": "position_held_outside_session"' in ledger.journal_path().read_text()
    ledger.close()


def test_loop_uses_strategy_fn_with_snapshot(tmp_path: Path, fake_features) -> None:
    seen: list[object] = []

    def strategy(snap, position, limits, *, now):
        seen.append(snap)
        return Decision("BTC/USD", "hold", "strategy_fn")

    loop, ledger, calls = _loop(tmp_path, fake_features, _enter)
    loop._strategy_fn = strategy
    loop.tick(MON_0935)
    assert len(seen) == 1 and calls == []
    assert '"reason": "strategy_fn"' in ledger.journal_path().read_text()
    ledger.close()

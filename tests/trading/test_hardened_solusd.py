"""Regression and edge-case gates for the paper-only SOL/USD hardening."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

from reflective_loop_runtime.trading.decider import decide
from reflective_loop_runtime.trading.features import compute
from reflective_loop_runtime.trading.ledger import Ledger
from reflective_loop_runtime.trading.market import fetch_snapshots
from reflective_loop_runtime.trading.model import Limits, Position
from reflective_loop_runtime.trading.types import Bar, Features, Snapshot


def good(**kwargs):
    vals = dict(
        symbol="SOL/USD",
        price=108.50,
        ema20_5m=108.30,
        ema50_5m=108.20,
        vwap=108.25,
        atr14=0.80,
        ret_60m=0.005,
        spread_pct=0.05,
        crossed_above_vwap=True,
        last_bar_age_s=60.0,
        confirmed_vwap_reclaim=True,
        below_vwap_bars=0,
        ret_5m=0.002,
        quote_age_s=5.0,
        quote_bid=108.49,
        full_vwap_coverage=True,
    )
    vals.update(kwargs)
    return Features(**vals)


def lim(**kwargs):
    vals = dict(
        hardened_entries=True, vwap_failure_exit=True, crypto_fee_rate=0.0025, min_net_target_r=0.50
    )
    vals.update(kwargs)
    return Limits(**vals)


def test_hardened_valid_entry():
    d = decide(good(), None, lim(), now=1000.0)
    assert d.action == "enter_long"
    assert d.reason == "trend_vwap_reclaim"


def test_rejects_trade_matching_oct13_2026_log():
    f = good(
        price=108.377,
        ema20_5m=108.4187,
        ema50_5m=108.2031,
        vwap=108.28446,
        atr14=0.08762,
        ret_60m=0.00094,
        spread_pct=0.10334,
        ret_5m=-0.0008,
    )
    d = decide(f, None, lim(), now=1000.0)
    assert d.action == "hold"
    assert d.reason == "price_below_ema20"


def test_entry_fail_closed_conditions():
    for fields, expected in [
        (dict(full_vwap_coverage=False), "vwap_history_incomplete"),
        (dict(confirmed_vwap_reclaim=False), "reclaim_not_confirmed"),
        (dict(price=108.2), "price_below_ema20"),
        (dict(ret_5m=0), "weak_5m_momentum"),
        (dict(ret_60m=-0.001), "weak_60m_momentum"),
        (dict(price=108.35, spread_pct=0.12, ema20_5m=108.26), "reclaim_below_spread"),
        (dict(quote_age_s=160), "quote_stale"),
    ]:
        d = decide(good(**fields), None, lim(), now=1000)
        assert d.action == "hold", (fields, d)
        assert d.reason == expected, (fields, d)


def test_fee_target_gate_rejects_bad_economics():
    f = good(atr14=0.08)
    d = decide(f, None, lim(), now=1000)
    assert d.reason == "insufficient_net_reward"
    assert decide(f, None, lim(crypto_fee_rate=0.0), now=1000).action == "enter_long"


def test_failed_vwap_exit_and_stop_priority():
    p = Position(
        symbol="SOL/USD",
        qty=4.0,
        entry_px=108.4,
        stop=107.9,
        target=109.1,
        entry_ts=1000.0,
        atr=0.08,
    )
    f = good(price=108.1, vwap=108.3, ema20_5m=108.2, below_vwap_bars=2, quote_bid=108.09)
    assert decide(f, p, lim(), now=1200).reason == "vwap_failure"
    assert decide(replace(f, below_vwap_bars=1), p, lim(), now=1200).action == "hold"
    assert decide(replace(f, quote_bid=107.88), p, lim(), now=1200).reason == "stop"
    assert (
        decide(
            replace(f, quote_bid=107.88, quote_age_s=200, below_vwap_bars=0), p, lim(), now=1200
        ).action
        == "hold"
    )


def test_fee_modeled_risk_halts_earlier(tmp_path):
    led = Ledger(tmp_path / "trades.db", tmp_path / "journal.jsonl", modeled_crypto_fee_rate=0.0025)
    led.open_trade(
        symbol="SOL/USD",
        entry_ts=2000,
        entry_px=108.4,
        qty=4.52,
        stop=107.952,
        target=109.117,
        atr=0.088,
        entry_order_id="buy",
    )
    led.close_trade(
        symbol="SOL/USD", exit_ts=2200, exit_px=107.87, reason="stop", exit_order_id="sell"
    )
    st = led.risk_state(2300, account_active=True, kill=False)
    assert st.realized_r_today < -2.3
    assert st.realized_r_today > -2.5
    assert st.last_loss_ts["SOL/USD"] == 2200
    led.close()


def test_closed_five_minute_ema_ignores_open_bucket():
    start = datetime(2026, 10, 13, 0, 0, tzinfo=timezone.utc).timestamp()
    bars = tuple(
        Bar(ts=start + i * 60, o=100, h=101, l=99, c=100 if i < 250 else 200, v=5)
        for i in range(252)
    )
    snap = Snapshot(
        symbol="SOL/USD",
        bars=bars,
        quote=None,
        fetched_at=start + 252 * 60,
        last_bar_ts=bars[-1].ts,
    )
    f = compute(snap, now=start + 252 * 60)
    assert abs(f.ema50_5m - 100) < 1e-6
    assert f.full_vwap_coverage


def test_fetch_crypto_covers_utc_midnight_and_ema_seed(monkeypatch):
    monkeypatch.setenv("RLR_TRADER_ASSET_CLASS", "crypto")
    now = datetime(2026, 10, 13, 8, 32, tzinfo=timezone.utc).timestamp()
    urls = []

    def get(url, headers):
        urls.append(url)
        return {"bars": {}} if "/bars" in url else {"quotes": {}}

    fetch_snapshots(("SOL/USD",), http=get, now=now)
    query = parse_qs(urlparse(next(u for u in urls if "/bars" in u)).query)
    start = datetime.fromisoformat(query["start"][0].replace("Z", "+00:00"))
    assert start <= datetime(2026, 10, 12, 19, 0, tzinfo=timezone.utc)

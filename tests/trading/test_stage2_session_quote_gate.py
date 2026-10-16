from datetime import datetime, timezone

from reflective_loop_runtime.trading.session_quote_gate import check

T = datetime(2026, 10, 15, 18, 0, tzinfo=timezone.utc).timestamp()


def good(**kw):
    x = dict(bid=585.48, ask=585.52, quote_ts=T - 1, bar_ts=T - 80, now=T)
    x.update(kw)
    return check(**x)


def test_iex_gate_valid():
    assert good() == (True, "validated_iex_1m")


def test_zero_ask():
    assert not good(ask=0)[0]


def test_crossed_bid():
    assert not good(ask=585.48)[0]


def test_stale_quote():
    assert not good(quote_ts=T - 6)[0]


def test_clock_skew():
    assert good(quote_ts=T + 0.2)[0]


def test_future_quote():
    assert not good(quote_ts=T + 0.3)[0]


def test_stale_bar():
    assert not good(bar_ts=T - 130)[0]


def test_incomplete_bar():
    assert not good(bar_ts=T - 40)[0]


def test_spread():
    assert not good(ask=586.7)[0]


def test_schwab_disabled():
    assert not good(source="schwab")[0]


def test_weekend():
    assert not good(now=datetime(2026, 10, 17, 18, tzinfo=timezone.utc).timestamp())[0]

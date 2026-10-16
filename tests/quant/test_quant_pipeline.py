from __future__ import annotations

import random
from datetime import datetime, timezone
from math import isclose

import pytest

from reflective_loop_runtime.quant.edge_jetson.runtime_opt import (
    Memory,
    MemoryStore,
    RingBuffer,
    memory_score,
    thermal_readings,
    toon_decode,
    toon_encode,
    wire_metrics,
)
from reflective_loop_runtime.quant.execution.smart_router import (
    Book,
    Level,
    RouteRejected,
    SmartRouter,
    VenueCapabilities,
    twap_slices,
    verify_oca_siblings,
)
from reflective_loop_runtime.quant.pipeline import plan_research_entry, simulate_maker_fill
from reflective_loop_runtime.quant.regime.market_state import (
    Bar,
    Regime,
    adx,
    classify,
    hurst,
    yang_zhang,
)
from reflective_loop_runtime.quant.risk.kelly_firewall import (
    ATRTrail,
    OutcomeStats,
    RiskFirewall,
    shrink_covariance,
)
from reflective_loop_runtime.quant.rl_core.reward_and_env import (
    NetRewardEnv,
    QueueAhead,
    VenueCosts,
)
from reflective_loop_runtime.quant.validation.darwin_cpcv import (
    Event,
    ShadowFill,
    bootstrap_stress,
    cpcv,
    deflated_sharpe_probability,
    max_drawdown,
    parameter_stability,
    profit_factor,
    promotion_gate,
    shadow_gate,
)


def book(t=100000, sequence=8):
    return Book((Level(100, 8), Level(99.99, 4)), (Level(100.01, 2), Level(100.02, 5)), t, sequence)


def venue(native=True):
    return VenueCapabilities(
        "TEST" if native else "alpaca_equities",
        native,
        "gtx" if native else "day",
        0.01,
        0.01,
        True,
    )


def risk_inputs():
    observations = {"SPY": [(100000 + i * 1000, (i % 5 - 2) * 0.001) for i in range(100)]}
    names, cov = shrink_covariance(observations, 199000)
    return names, cov


def test_microprice_and_obi():
    b = book()
    assert isclose(b.microprice(), (8 * 100.01 + 2 * 100) / 10)
    assert 0 < b.imbalance() < 1


def test_book_rejects_crossed_nonfinite_unordered():
    with pytest.raises(RouteRejected):
        Book((Level(101, 1),), (Level(100, 2),), 1, 1)
    with pytest.raises(RouteRejected):
        Level(1, float("nan"))
    with pytest.raises(RouteRejected):
        Book((Level(99, 1), Level(100, 1)), (Level(101, 1),), 1, 1)


def test_router_deterministic_native_and_conservative():
    r = SmartRouter(venue(), max_age_ms=300)
    x = r.plan(symbol="SPY", side="buy", quantity=1.005, book=book(), now_ms=100100, cycle_id="a")
    y = r.plan(symbol="SPY", side="buy", quantity=1.005, book=book(), now_ms=100100, cycle_id="a")
    assert x == y and x.quantity == 1 and x.limit == 100 and x.native_post_only
    z = r.plan(symbol="SPY", side="sell", quantity=1, book=book(), now_ms=100100, cycle_id="a")
    assert z.limit == 100.01


def test_router_rejects_unsupported_post_only_and_chasing():
    r = SmartRouter(venue(False))
    with pytest.raises(RouteRejected, match="post-only"):
        r.plan(symbol="SPY", side="buy", quantity=1, book=book(), now_ms=100050, cycle_id="q")
    with pytest.raises(RouteRejected, match="two ticks"):
        r.plan(
            symbol="SPY",
            side="buy",
            quantity=1,
            book=book(),
            now_ms=100050,
            cycle_id="q",
            require_native_post_only=False,
            rtt_ms=50,
            sigma_price_per_sqrt_second=0.01,
        )
    wider = Book((Level(100, 8),), (Level(100.03, 2),), 100000, 8)
    assert not r.plan(
        symbol="SPY",
        side="buy",
        quantity=1,
        book=wider,
        now_ms=100050,
        cycle_id="q",
        require_native_post_only=False,
        rtt_ms=50,
        sigma_price_per_sqrt_second=0.01,
    ).native_post_only
    n = SmartRouter(venue())
    with pytest.raises(RouteRejected, match="cancellation"):
        n.plan(
            symbol="SPY", side="buy", quantity=1, book=book(), now_ms=100050, cycle_id="q", chase=1
        )
    with pytest.raises(RouteRejected, match="stale"):
        n.plan(symbol="SPY", side="buy", quantity=1, book=book(), now_ms=200000, cycle_id="q")
    with pytest.raises(RouteRejected, match="monotonic"):
        n.plan(
            symbol="SPY",
            side="buy",
            quantity=1,
            book=book(),
            now_ms=100050,
            cycle_id="q",
            prior_sequence=8,
        )


def test_twap_oca():
    assert twap_slices(1, 3, 0.1) == (0.4, 0.3, 0.3)
    with pytest.raises(RouteRejected):
        twap_slices(0.1, 3, 0.1)
    orders = (
        {"id": "1", "oca_group": "x", "status": "filled", "filled_qty": 1, "qty": 1},
        {"id": "2", "oca_group": "x", "status": "canceled", "filled_qty": 0, "qty": 1},
    )
    assert verify_oca_siblings(orders, total_exit_qty=1)
    assert not verify_oca_siblings((orders[0], {**orders[1], "status": "new"}), total_exit_qty=1)


def test_queue_partial_determinism():
    q = QueueAhead(5, 3)
    assert q.consume(3) == 0 and q.ahead == 2
    assert q.consume(3) == 1 and q.remaining == 2
    assert q.consume(100) == 2 and q.remaining == 0
    assert q.consume(100) == 0
    with pytest.raises(ValueError):
        q.consume(-2)


def test_net_equity_fee_once():
    env = NetRewardEnv(10000, VenueCosts(1, 10, 1, 0))
    env.step(mark_price=100, now_seconds=0)
    fee = env.trade(side="buy", qty=1, price=100, maker=True)
    assert isclose(fee, 0.01)
    v = env.step(mark_price=100, now_seconds=1)
    assert isclose(v["net_pnl"], -0.01, abs_tol=1e-9)
    assert isclose(v["net_delta"], -0.01, abs_tol=1e-9)
    assert v["reward"] < 0


def test_funding_only_derivative_once_and_clock_monotonic():
    spot = NetRewardEnv(10000, VenueCosts(1, 2, 1, 0))
    with pytest.raises(ValueError, match="funding"):
        spot.step(mark_price=100, now_seconds=0, funding_rate_8h=0.01)
    perp = NetRewardEnv(10000, VenueCosts(0, 0, 0, 0, True))
    perp.trade(side="buy", qty=1, price=100, maker=True)
    perp.step(mark_price=100, now_seconds=0, funding_rate_8h=0.01)
    a = perp.step(mark_price=100, now_seconds=28800, funding_rate_8h=0.01)
    assert isclose(a["funding_total"], 1)
    assert isclose(
        perp.step(mark_price=100, now_seconds=28800, funding_rate_8h=0.01)["funding_total"], 1
    )
    with pytest.raises(ValueError):
        perp.step(mark_price=100, now_seconds=100, funding_rate_8h=0.01)


def test_covariance_past_only_and_alignment():
    names, cov = risk_inputs()
    assert names == ("SPY",) and cov[0][0] > 0
    with pytest.raises(ValueError):
        shrink_covariance({"A": [(i, 1 if i % 2 else -1) for i in range(10)]}, 10)
    with pytest.raises(ValueError):
        shrink_covariance({"A": [(i, 0.1) for i in range(30)]}, 29)


def test_kelly_wilson_and_drawdown():
    n, c = risk_inputs()
    f = RiskFirewall()
    stats = OutcomeStats(400, 100, 2, 1)
    base = dict(
        symbol="SPY",
        now=datetime(2026, 10, 16, tzinfo=timezone.utc),
        equity=10000,
        opening_equity=10000,
        existing_notional={},
        stats=stats,
        beta={"SPY": 1},
        covariance_names=n,
        covariance=c,
    )
    v = f.evaluate(**base)
    assert v.permitted and 0 < v.notional <= 200
    assert not f.evaluate(**{**base, "equity": 9600}).permitted
    assert not f.evaluate(**{**base, "stats": OutcomeStats(6, 2, 2, 1)}).permitted
    assert not f.evaluate(**{**base, "existing_notional": {"SPY": 1000}}).permitted


def test_atr_trail_monotonic():
    t = ATRTrail("long")
    assert t.update(102, 100, 1) == 100
    assert t.update(101, 99, 1) == 100
    assert t.update(105, 103, 1) == 103
    s = ATRTrail("short")
    assert s.update(102, 100, 1) == 102
    assert s.update(104, 103, 1) == 102


def bars(n=65):
    random.seed(3)
    values = []
    last = 100
    for i in range(n):
        close = last * (1 + random.uniform(-0.005, 0.005))
        values.append(Bar(i, last, max(last, close) * 1.001, min(last, close) * 0.999, close))
        last = close
    return values


def test_indicator_finite_and_lookahead_guard():
    b = bars()
    assert 0 < yang_zhang(b) < 1 and 0 <= adx(b) <= 100 and 0 <= hurst([x.close for x in b]) <= 1
    with pytest.raises(ValueError):
        classify(b, asof=40, obi=0)
    assert classify(b[:10], asof=50, obi=0).state == "UNSAFE"
    with pytest.raises(ValueError):
        classify(b, asof=100, obi=2)
    with pytest.raises(ValueError):
        classify(b, asof=100, obi=0, funding_history=[0.1])


def test_regime_strategy_gate():
    r = Regime("TREND_UP", 0.02, 35, 0.7, 0.1, None, "trend")
    assert r.permits("momentum_long") and not r.permits("mean_reversion")
    assert not Regime("UNSAFE", None, None, None, 0, None, "chop").permits("momentum_long")


def events(n=60):
    return [Event(i * 60, i * 60 + 50, 3 if i % 3 else -1, 0.2) for i in range(n)]


def test_cpcv_event_purge_and_embargo():
    e = events()
    splits = cpcv(e, 6, 2, embargo=30)
    assert len(splits) == 15
    for split in splits:
        assert not set(split.train) & set(split.test)
        for i in split.train:
            assert all(e[i].end < e[j].start or e[i].start > e[j].end for j in split.test)
    assert (
        len(
            cpcv(
                [Event(0, 30, 1, 0), Event(10, 40, 1, 0), Event(50, 60, 1, 0), Event(80, 90, 1, 0)],
                4,
                1,
            )
        )
        == 4
    )


def test_dsr_probability_not_sharpe_ratio():
    a = [0.001 + 0.01 * ((i * 7) % 17 - 8) / 20 for i in range(100)]
    p = deflated_sharpe_probability(a, 20, 0.02)
    assert 0 <= p <= 1
    with pytest.raises(ValueError):
        deflated_sharpe_probability(a, 1, 0.02)


def test_profit_factor_drawdown_stress():
    assert profit_factor([2, -1, 3]) == 5
    assert isclose(max_drawdown([100, 110, 99, 101]), 0.1)
    m = bootstrap_stress(events(), iterations=100)
    assert 0 <= m["positive_net_rate"] <= 1


def test_shadow_gate_independence():
    s = [ShadowFill(f"id{i}", i + 100, 2, 1) for i in range(30)]
    assert shadow_gate(s)
    assert not shadow_gate(s[:29])
    assert not shadow_gate(s[:-1] + [s[0]])
    assert not shadow_gate([ShadowFill(f"id{i}", i + 100, 10, 0) for i in range(30)])


def test_gate_fails_closed_without_evidence():
    r = promotion_gate(
        trades=events(),
        equity=[10000 + i for i in range(60)],
        dsr_probability=0.8,
        shadow=[],
        base_sharpe=2,
        jittered_sharpes=[1.9, 2.1],
        full_oos_days=90,
    )
    assert not r.approved and "DSR_probability_gate" in r.reasons
    assert not parameter_stability(2, [1.0, 1.9])


def test_toon_roundtrip_and_bytes():
    obj = {
        "W1": {"symbol": "SPY", "side": "buy", "size": 2, "trusted": True},
        "W2": {"scores": [0.1, 0.2], "notes": "line\none"},
    }
    wire = toon_encode(obj)
    assert toon_decode(wire) == obj
    met = wire_metrics(obj)
    assert met["saved_bytes"] == met["json_bytes"] - met["toon_bytes"]
    with pytest.raises(ValueError):
        toon_encode({"bad.key": 1})
    with pytest.raises(ValueError):
        toon_decode("a: 1\na: 2\n")


def test_ring_memory_store_and_thermal(tmp_path):
    ring = RingBuffer(2, 4)
    ring.push(b"a")
    ring.push(b"b")
    ring.push(b"c")
    assert len(ring) == 2 and ring.latest() == b"c"
    with pytest.raises(ValueError):
        ring.push(b"12345")
    assert isclose(memory_score([1, 0], [1, 0], 0, 0), 0.85)
    db = MemoryStore(tmp_path / "memory.db")
    db.put(Memory("a", [1, 0], 100, 1, {"x": 1}))
    db.put(Memory("b", [0, 1], 100, 0, {"x": 2}))
    assert db.recall([1, 0], 200)[0][0] == "a"
    thermal = tmp_path / "thermal_zone0"
    thermal.mkdir()
    (thermal / "type").write_text("cpu")
    (thermal / "temp").write_text("55000")
    assert thermal_readings(tmp_path)["thermal_zone0:cpu"] == 55


def test_end_to_end_offline_veto_then_fill():
    n, c = risk_inputs()
    f = RiskFirewall()
    r = SmartRouter(venue())
    common = dict(
        regime=Regime("TREND_UP", 0.01, 32, 0.7, 0.1, None, "test"),
        strategy="momentum_long",
        router=r,
        book=book(),
        symbol="SPY",
        side="buy",
        quantity=1,
        now_ms=100050,
        cycle_id="abc",
        firewall=f,
        equity=10000,
        opening_equity=10000,
        existing_notional={},
        stats=OutcomeStats(400, 100, 2, 1),
        betas={"SPY": 1},
        covariance_names=n,
        covariance=c,
        now=datetime(2026, 10, 16, tzinfo=timezone.utc),
    )
    result = plan_research_entry(**common)
    assert result.permitted and result.intent is not None
    assert not plan_research_entry(**{**common, "strategy": "mean_reversion"}).permitted
    env = NetRewardEnv(10000, VenueCosts(1, 10, 1, 0.2))
    entry = simulate_maker_fill(
        result.intent, QueueAhead(1, result.intent.quantity), 2, env, mark_price=100, now_seconds=0
    )
    assert entry["filled_qty"] > 0 and entry["fees_total"] > 0 and entry["net_pnl"] < 0

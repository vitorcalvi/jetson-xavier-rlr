"""TOON v2 safety, covariance, thermal, CV and unified shadow integration."""

from __future__ import annotations

import random
import sqlite3
from dataclasses import replace
from math import isclose

import pytest

from reflective_loop_runtime.quant.edge_jetson.thermal_v2 import (
    ThermalGovernor,
    inference_backend,
)
from reflective_loop_runtime.quant.execution.smart_router import (
    Book,
    Level,
    RouteRejected,
    SmartRouter,
    VenueCapabilities,
)
from reflective_loop_runtime.quant.regime.market_state import Bar
from reflective_loop_runtime.quant.risk.covariance_v2 import (
    aligned_risk_matrices,
    effective_positive_correlation,
)
from reflective_loop_runtime.quant.risk.kelly_firewall import OutcomeStats, RiskFirewall
from reflective_loop_runtime.quant.rl_core.reward_and_env import NetRewardEnv
from reflective_loop_runtime.quant.rl_core.venue_costs import (
    UnknownFees,
    get_profile,
)
from reflective_loop_runtime.quant.runtime_v2 import MarketTick, RuntimeConfig, UnifiedQuantRuntime
from reflective_loop_runtime.quant.validation.metadata_v2 import (
    FoldMemoryIndex,
    FoldStandardizer,
    TrialRegistry,
)


def snapshot(mid=100.0, spread=0.04, seq=1, ts=100000):
    return Book((Level(mid, 50),), (Level(mid + spread, 40),), ts, seq)


def synthetic_venue():
    return VenueCapabilities("alpaca_equity", False, "day", 0.01, 0.01, True)


def test_d1_synthetic_passive_rtt_buffer_and_book_vetoes():
    router = SmartRouter(synthetic_venue(), max_age_ms=250, max_spread_bps=20)
    book = snapshot()
    args = dict(
        symbol="SPY",
        side="buy",
        quantity=2,
        book=book,
        now_ms=100055,
        cycle_id="x",
        require_native_post_only=False,
        rtt_ms=50,
        sigma_price_per_sqrt_second=0.045,
    )
    result = router.plan(**args)
    assert result.limit <= 100.02
    assert result.routing_mode.startswith("SYNTHETIC_") and not result.native_post_only
    assert result.synthetic_guard_bps > 0
    assert result.tif == "day"
    with pytest.raises(RouteRejected, match="RTT"):
        router.plan(**{**args, "rtt_ms": 700})
    with pytest.raises(RouteRejected, match="OBI"):
        router.plan(**{**args, "prev_obi": 0.8, "observed_obi": -0.2})
    with pytest.raises(RouteRejected, match="two ticks"):
        router.plan(**{**args, "book": snapshot(spread=0.01)})
    with pytest.raises(RouteRejected, match="inventory"):
        router.plan(**{**args, "chase": 1, "cancel_terminal": True})


def test_d1_latency_jitter_has_no_guarantee_in_jump_scenario():
    router = SmartRouter(synthetic_venue(), max_age_ms=250, max_spread_bps=20)
    rng = random.Random(12)
    ok = 0
    cross = 0
    for i in range(500):
        book = snapshot(seq=i + 1, ts=100000 + 100 * i)
        planned = router.plan(
            symbol="SPY",
            side="buy",
            quantity=1,
            book=book,
            now_ms=book.timestamp_ms + 50,
            cycle_id=str(i),
            require_native_post_only=False,
            rtt_ms=50 + rng.randrange(-10, 11),
            sigma_price_per_sqrt_second=0.06,
        )
        # Simulate 50ms book jitter. Even a passive submitted limit can be
        # marketable on ARRIVAL due to adverse ask movements.
        next_ask = book.asks[0].price + rng.uniform(-0.01, 0.01)
        if planned.limit >= next_ask:
            cross += 1
        else:
            ok += 1
    assert ok == 500 and cross == 0  # bounded-no-jump simulation ONLY
    jump_ask = 99.95
    assert planned.limit >= jump_ask  # demonstrate risk when ask jumps through price


def test_d2_fee_registry_and_calibrated_equity_gates():
    spot = get_profile("ALPACA_SPOT_CRYPTO_TIER1")
    assert spot.executed_fees(side="buy", qty=1, price=100, maker=True) == 0.15
    assert spot.executed_fees(side="sell", qty=1, price=100, maker=False) == 0.25
    assert spot.expectancy_veto(40, friction_bps=40)
    assert not spot.expectancy_veto(61, friction_bps=40)
    equity = get_profile("ALPACA_EQUITIES_UNCALIBRATED")
    with pytest.raises(UnknownFees):
        equity.to_env_costs()
    calibrated = replace(
        equity,
        cat_per_share=0.0002,
        taf_sell_per_share=0.00025,
        sec_sell_bps=0.02,
        margin_apr=0.075,
        source="auditable_live_schedule",
    )
    buy = calibrated.executed_fees(side="buy", qty=10, price=100, maker=True)
    sell = calibrated.executed_fees(side="sell", qty=10, price=100, maker=False)
    assert buy > 0 and sell > buy
    assert isclose(calibrated.interest(notional_borrowed=1000, hours=24), 1000 * 0.075 / 365)
    assert not calibrated.funding_enabled
    perp = get_profile("CRYPTO_PERP_EXAMPLE_UNVERIFIED")
    assert perp.funding_enabled and perp.asset_class == "perpetual"
    with pytest.raises(UnknownFees):
        get_profile("unknown")


def test_d2_fee_calibration_enforced_in_net_reward_no_double_count():
    equity = replace(
        get_profile("ALPACA_EQUITIES_UNCALIBRATED"),
        cat_per_share=0.001,
        sec_sell_bps=0.02,
        taf_sell_per_share=0.001,
        margin_apr=0.075,
    )
    env = NetRewardEnv(100.0, equity.to_env_costs(), fee_profile=equity)
    env.step(mark_price=100, now_seconds=0)
    env.trade(side="buy", qty=2, price=100, maker=True)
    first = env.step(mark_price=100, now_seconds=60, margin_carry_hours=24)
    assert first["margin_interest_total"] > 0
    assert first["fees_total"] > 0
    assert isclose(first["net_pnl"], first["net_delta"])
    env.trade(side="sell", qty=2, price=100, maker=False)
    after = env.step(mark_price=100, now_seconds=120)
    assert after["fees_total"] > first["fees_total"]
    assert after["net_pnl"] < first["net_pnl"]
    assert isclose(env.step(mark_price=100, now_seconds=120)["net_delta"], 0.0, abs_tol=1e-12)


def test_d3_registry_persists_trials_across_runs_and_fit_is_train_only(tmp_path):
    db = tmp_path / "multiplicity.db"
    a = TrialRegistry(db)
    a.add("study", "trial_a", "ds", "cfg1", "fold1", 0.2)
    b = TrialRegistry(db)
    b.add("study", "trial_b", "ds", "cfg2", "fold1", -0.2)
    assert a.summary("study").total == 2 and b.summary("study").sr_variance > 0.01
    with pytest.raises(sqlite3.IntegrityError):
        b.add("study", "trial_a", "ds", "cfg1", "fold1", 0.5)
    x = [(float(i),) for i in range(90)]
    normalizer = FoldStandardizer.fit(x, tuple(range(60)), tuple(range(60, 90)))
    scaler_mean = normalizer.means
    x[-1] = (1e9,)
    newer = FoldStandardizer.fit(x, tuple(range(60)), tuple(range(60, 90)))
    assert newer.means == scaler_mean and newer.transform(x[-1])[0] > 100
    with pytest.raises(ValueError):
        FoldStandardizer.fit(x, (0, 60), (60, 61))
    memories = [(i, "m" + str(i)) for i in range(90)]
    memory = FoldMemoryIndex.fit(memories, tuple(range(40)), tuple(range(40, 90)), 35)
    assert all(int(item[1:]) <= 35 for item in memory.recall(35))


def joint_data(n=120):
    a = []
    b = []
    bench = []
    for i in range(n):
        x = 0.002 * ((i % 11) - 5)
        bench.append((i * 60000, x))
        a.append((i * 60000, x))
        b.append((i * 60000, x + 0.00001 * ((i % 3) - 1)))
    return {"BTC": a, "ETH": b, "BENCH": bench}


def test_d5_collinear_psd_tail_correlation_and_missing_bar_refusal():
    data = joint_data()
    mat = aligned_risk_matrices(data, benchmark="BENCH", asof_ms=119 * 60000, min_tail=8)
    assert 0 <= mat.shrinkage <= 1 and mat.common_samples == 120
    assert mat.covariance[0][0] >= 0
    # Every 2x2 principal minor nonnegative for PSD shrinkage.
    for i in range(3):
        for j in range(i + 1, 3):
            assert mat.covariance[i][i] * mat.covariance[j][j] - mat.covariance[i][j] ** 2 >= -1e-13
    assert effective_positive_correlation(mat, "BTC", "ETH") >= 0
    mixed = {**data, "ETH": [(ts + 30000, v) for ts, v in data["ETH"]]}
    with pytest.raises(ValueError, match="synchronized"):
        aligned_risk_matrices(mixed, benchmark="BENCH", asof_ms=119 * 60000)


def test_d6_thermal_governor_suspends_and_resumes(tmp_path):
    z = tmp_path / "thermal_zone0"
    z.mkdir()
    (z / "temp").write_text("78000")
    mem = tmp_path / "meminfo"
    mem.write_text("MemTotal: 10000 kB\nMemAvailable: 9000 kB\n")
    g = ThermalGovernor(thermal_root=tmp_path, meminfo=mem)
    assert g.check() is False and g.samples == 1 and g.paused
    assert g.reason == "thermal_or_memory_cap"
    (z / "temp").write_text("71000")
    assert not g.check() and g.reason == "cooldown_required"
    (z / "temp").write_text("65000")
    assert g.check() and g.reason == "allowed_background_only"
    mem.write_text("MemTotal: 10000 kB\nMemAvailable: 1500 kB\n")
    assert not g.check() and g.paused
    # Mock hot background trainer is paused, but pure hot-path planning is live.
    native = VenueCapabilities("GTX_SIM", True, "gtx", 0.01, 0.01, True)
    plan = SmartRouter(native).plan(
        symbol="BTCUSD",
        side="buy",
        quantity=1,
        book=snapshot(),
        now_ms=100001,
        cycle_id="hot-routing",
    )
    assert plan.native_post_only
    assert (
        inference_backend(trt_available=False, onnx_available=False, torch_available=True)
        == "torch_cpu"
    )


def trend_bars(start_time_ms, count=100):
    arr = []
    last = 100.0
    rng = random.Random(12)
    momentum = 0.0
    for i in range(count):
        momentum = 0.5 * momentum + 0.5 * rng.gauss(0, 0.001)
        close = last * (1 + 0.002 + momentum)
        arr.append(
            Bar(
                start_time_ms + i * 60000,
                last,
                max(last, close) * 1.0001,
                min(last, close) * 0.9999,
                close,
            )
        )
        last = close
    return tuple(arr)


def test_d4_500_ticks_full_research_pipeline_no_broker_io(tmp_path):
    start = 1_780_000_000_000
    bars = trend_bars(start, 620)
    ven = VenueCapabilities("SPOT_SIM_GT", True, "gtx", 0.01, 0.01, True)
    returns = {"BTCUSD": [(start + i * 60000, (i % 9 - 4) * 0.0004) for i in range(100)]}
    # Existing covariance estimator, deliberately training-only past stats.
    from reflective_loop_runtime.quant.risk.kelly_firewall import shrink_covariance

    names, cov = shrink_covariance(returns, start + 100 * 60000)
    rt = UnifiedQuantRuntime(
        config=RuntimeConfig(halt_file=tmp_path / "HALT_ENTRIES"),
        router=SmartRouter(ven, max_age_ms=1000, max_spread_bps=20),
        risk=RiskFirewall(),
        stats=OutcomeStats(400, 100, 2, 1),
        cost_profile=get_profile("ALPACA_SPOT_CRYPTO_TIER1"),
        covariance_names=names,
        covariance=cov,
        betas={"BTCUSD": 1},
        equity=10000,
        opening_equity=10000,
    )
    results = []
    for i in range(100, 600):
        b = bars[:i]
        t = start + i * 60000 + 50
        book = Book((Level(b[-1].close, 50),), (Level(b[-1].close + 0.03, 50),), t - 10, i)
        results.append(
            rt.on_tick(
                MarketTick(t, b, book, "BTCUSD", 0.01, 50),
                cycle_id=str(i),
                expected_gross_alpha_bps=55,
            )
        )
    assert len(results) == 500
    assert all(o.broker_orders_submitted == 0 for o in results)
    assert len(rt.observations) == 500
    assert any(o.shadow is not None for o in results)
    assert not rt.certification_evidence([])["paper_trade_gate"]
    assert all(o.execution_status != "LIVE" for o in results)
    with pytest.raises(ValueError):
        RuntimeConfig(execution_mode="PRODUCTION_LIVE", halt_entries=False)
    with pytest.raises(ValueError, match="incomplete"):
        rt.on_tick(MarketTick(t, tuple(bars[: i + 1]), book, "BTCUSD", 0.01, 50), cycle_id="future")

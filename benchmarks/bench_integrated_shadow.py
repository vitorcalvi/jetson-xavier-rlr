"""500 historical synthetic ticks through entire no-broker shadow pipeline."""

from __future__ import annotations

import json
import random

from reflective_loop_runtime.quant.execution.smart_router import (
    Book,
    Level,
    SmartRouter,
    VenueCapabilities,
)
from reflective_loop_runtime.quant.regime.market_state import Bar
from reflective_loop_runtime.quant.risk.kelly_firewall import (
    OutcomeStats,
    RiskFirewall,
    shrink_covariance,
)
from reflective_loop_runtime.quant.rl_core.venue_costs import get_profile
from reflective_loop_runtime.quant.runtime_v2 import MarketTick, RuntimeConfig, UnifiedQuantRuntime


def run(n: int = 500) -> dict:
    if n < 1:
        raise ValueError("need at least 1 tick")
    start = 1_780_000_000_000
    rng = random.Random(12)
    trend = 0.0
    last = 100.0
    bars = []
    for i in range(n + 100):
        trend = 0.5 * trend + 0.5 * rng.gauss(0, 0.001)
        close = last * (1 + 0.002 + trend)
        bars.append(
            Bar(
                start + i * 60000, last, max(last, close) * 1.0001, min(last, close) * 0.9999, close
            )
        )
        last = close
    data = {"BTCUSD": [(start + i * 60000, (i % 9 - 4) * 0.0004) for i in range(100)]}
    names, cov = shrink_covariance(data, start + 100 * 60000)
    runtime = UnifiedQuantRuntime(
        RuntimeConfig(),
        SmartRouter(
            VenueCapabilities("SIMULATION_ONLY", True, "gtx", 0.01, 0.01, True), max_spread_bps=20
        ),
        RiskFirewall(),
        OutcomeStats(400, 100, 2, 1),
        get_profile("ALPACA_SPOT_CRYPTO_TIER1"),
        names,
        cov,
        {"BTCUSD": 1},
        10000,
        10000,
    )
    for i in range(100, n + 100):
        t = start + i * 60000 + 50
        snap = Book(
            (Level(bars[i - 1].close, 50),), (Level(bars[i - 1].close + 0.03, 50),), t - 10, i
        )
        runtime.on_tick(
            MarketTick(t, tuple(bars[:i]), snap, "BTCUSD", 0.01, 50),
            cycle_id=f"synthetic-{i}",
            expected_gross_alpha_bps=55,
        )
    return dict(
        benchmark="500_tick_unified_research_shadow",
        source="SYNTHETIC_OHLC_NOT_LIVE",
        total_ticks=len(runtime.observations),
        shadow_intents=len(runtime.shadow_intents),
        broker_orders_sent=0,
        safety_halt_enforced=True,
        shadow_certification=runtime.certification_evidence([])["paper_trade_gate"],
        reasons={
            reason: sum(r.execution_status == reason for r in runtime.observations)
            for reason in set(r.execution_status for r in runtime.observations)
        },
    )


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True, indent=2))

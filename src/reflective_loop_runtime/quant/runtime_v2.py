"""Default-SHADOW main entrypoint for the quant research pipeline.

No broker adapter and no live-account secrets. State mode is descriptive, NOT
authorization. Existing Stage2 service and its HALT_ENTRIES remain unmodified.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .edge_jetson.runtime_opt import toon_encode
from .execution.smart_router import Book, RouteRejected, SmartRouter
from .regime.market_state import Bar, Regime, classify
from .risk.covariance_v2 import RiskMatrices
from .risk.kelly_firewall import OutcomeStats, RiskFirewall
from .rl_core.venue_costs import UnknownFees, VenueCostProfile
from .validation.darwin_cpcv import ShadowFill, shadow_gate

MODES = ("RESEARCH_BACKTEST", "SHADOW_SIMULATION", "ALPACA_PAPER_CERT", "PRODUCTION_LIVE")


@dataclass(frozen=True)
class RuntimeConfig:
    execution_mode: str = "SHADOW_SIMULATION"
    halt_entries: bool = True
    halt_file: Path = Path("/home/vitor/rlr/flags/HALT_ENTRIES")
    min_trades_for_certification: int = 30

    def __post_init__(self):
        if self.execution_mode not in MODES or not self.halt_entries:
            raise ValueError("TOON v2 research service physically fail closed")


@dataclass(frozen=True)
class MarketTick:
    timestamp_ms: int
    bars: tuple[Bar, ...]
    book: Book
    symbol: str
    price_sigma_1s: float
    measured_rtt_ms: float
    funding_history: tuple[float, ...] | None = None
    funding_observations: tuple[tuple[int, float], ...] | None = None
    memory: dict | None = None
    memory_timestamp_ms: int | None = None


@dataclass(frozen=True)
class ShadowIntent:
    timestamp_ms: int
    symbol: str
    side: str
    limit_price: float
    quantity: float
    client_order_id: str
    token_wire: str
    source: str = "LOCAL_SHADOW_INTENT_NOT_A_BROKER_FILL"


@dataclass(frozen=True)
class PipelineObservation:
    timestamp_ms: int
    regime: str
    inference: str
    risk_status: str
    execution_status: str
    shadow: ShadowIntent | None
    broker_orders_submitted: int = 0


@dataclass
class UnifiedQuantRuntime:
    config: RuntimeConfig
    router: SmartRouter
    risk: RiskFirewall
    stats: OutcomeStats
    cost_profile: VenueCostProfile
    covariance_names: tuple[str, ...]
    covariance: tuple[tuple[float, ...], ...]
    betas: dict[str, float]
    equity: float
    opening_equity: float
    existing_notional: dict[str, float] = field(default_factory=dict)
    risk_matrices: RiskMatrices | None = None
    observations: list[PipelineObservation] = field(default_factory=list)
    shadow_intents: list[ShadowIntent] = field(default_factory=list)
    agent_inference: Callable[[Regime], tuple[str, str]] | None = None

    def on_tick(
        self, tick: MarketTick, *, cycle_id: str, expected_gross_alpha_bps: float | None = None
    ) -> PipelineObservation:
        """Market -> regime -> TOON memory -> agent -> Kelly -> halt -> router -> shadow.

        Broker POST never exists in this class, even if mode text says LIVE.
        Manual promotion must be implemented as a separate reviewed program.
        """
        if tick.timestamp_ms <= 0 or tick.book.timestamp_ms > tick.timestamp_ms:
            raise ValueError("future-dated market clock")
        if not tick.bars or tick.bars[-1].timestamp + 60_000 > tick.timestamp_ms:
            raise ValueError("incomplete one-minute bar: lookahead refused")
        if tick.memory and (
            tick.memory_timestamp_ms is None or tick.memory_timestamp_ms > tick.timestamp_ms
        ):
            raise ValueError("unverified/future-dated memory recall")
        if tick.funding_history is not None:
            raise ValueError("unstamped funding samples may leak future data")
        funding = None
        if tick.funding_observations is not None:
            if any(ts > tick.timestamp_ms for ts, _ in tick.funding_observations) or any(
                a[0] >= b[0]
                for a, b in zip(
                    tick.funding_observations, tick.funding_observations[1:], strict=False
                )
            ):
                raise ValueError("future or out-of-order funding history")
            funding = [v for _, v in tick.funding_observations]
        regime = classify(
            list(tick.bars),
            asof=tick.timestamp_ms,
            obi=tick.book.imbalance(),
            funding_history=funding,
        )
        wire = toon_encode(
            {
                "W1": tick.memory or {},
                "W2": {
                    "symbol": tick.symbol,
                    "regime": regime.state,
                    "time_ms": tick.timestamp_ms,
                    "obi": tick.book.imbalance(),
                },
            }
        )
        if self.agent_inference is None:
            proposal = {
                "TREND_UP": ("momentum_long", "buy"),
                "TREND_DOWN": ("momentum_short", "sell"),
                "MEAN_REVERT": ("mean_reversion", "buy"),
            }.get(regime.state, ("abstain", "buy"))
        else:
            proposal = self.agent_inference(regime)
        strategy, side = proposal
        risk_status = "not_evaluated"
        execution = "VETO"
        shadow = None
        if not regime.permits(strategy):
            execution = "regime_veto"
        else:
            risk = self.risk.evaluate(
                symbol=tick.symbol,
                now=datetime.fromtimestamp(tick.timestamp_ms / 1000, timezone.utc),
                equity=self.equity,
                opening_equity=self.opening_equity,
                existing_notional=self.existing_notional,
                stats=self.stats,
                beta=self.betas,
                covariance_names=self.covariance_names,
                covariance=self.covariance,
                risk_matrices=self.risk_matrices,
            )
            risk_status = risk.reason
            if not risk.permitted:
                execution = "risk_veto"
            elif expected_gross_alpha_bps is None:
                execution = "unverified_expected_alpha"
            else:
                try:
                    friction = self.cost_profile.friction_bps(
                        maker_entry=True,
                        maker_exit=True,
                        modeled_slippage_bps=1.0,
                        spread_cost_bps=1.0,
                    )
                    if self.cost_profile.expectancy_veto(
                        expected_gross_alpha_bps, friction_bps=friction
                    ):
                        execution = "negative_after_stressed_venue_costs"
                    else:
                        proposed = self.router.plan(
                            symbol=tick.symbol,
                            side=side,
                            quantity=risk.notional / max(tick.book.bids[0].price, 1e-9),
                            book=tick.book,
                            now_ms=tick.timestamp_ms,
                            cycle_id=cycle_id,
                            require_native_post_only=self.router.venue.native_post_only,
                            rtt_ms=tick.measured_rtt_ms,
                            sigma_price_per_sqrt_second=tick.price_sigma_1s,
                        )
                        # HALT has two layers: config true by construction, and
                        # optional durable filesystem flag. Only virtual intents.
                        shadow = ShadowIntent(
                            tick.timestamp_ms,
                            tick.symbol,
                            side,
                            proposed.limit,
                            proposed.quantity,
                            proposed.client_order_id,
                            wire,
                        )
                        self.shadow_intents.append(shadow)
                        execution = (
                            "SHADOW_ONLY_HALTED"
                            if self.config.halt_entries or self.config.halt_file.exists()
                            else "SHADOW_ONLY"
                        )
                except (ValueError, RouteRejected, UnknownFees) as exc:
                    execution = "planner_refused_" + str(exc)
        o = PipelineObservation(
            tick.timestamp_ms, regime.state, strategy, risk_status, execution, shadow
        )
        self.observations.append(o)
        return o

    def certification_evidence(self, observed_fills: list[ShadowFill]) -> dict[str, object]:
        """Shadow intents are *not* broker fills; outside evidence is mandatory."""
        ok = shadow_gate(observed_fills, min_fills=self.config.min_trades_for_certification)
        return {
            "paper_trade_gate": ok,
            "trades_observed": len(observed_fills),
            "shadow_intents": len(self.shadow_intents),
            "broker_orders_submitted": 0,
            "orders_ever_enabled": False,
        }

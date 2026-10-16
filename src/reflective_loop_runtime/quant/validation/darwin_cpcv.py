"""Leakage-aware combinatorial splits, DSR probability and fail-closed gates."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import e, erf, isfinite, sqrt
from random import Random
from statistics import NormalDist, mean, stdev


@dataclass(frozen=True)
class Event:
    start: int
    end: int
    gross_pnl: float
    cost: float

    def __post_init__(self) -> None:
        if (
            self.start < 0
            or self.end < self.start
            or not all(isfinite(x) for x in (self.gross_pnl, self.cost))
            or self.cost < 0
        ):
            raise ValueError("invalid event")

    def net(self, cost_multiplier: float = 1) -> float:
        if not isfinite(cost_multiplier) or cost_multiplier < 0:
            raise ValueError("invalid cost stress")
        return self.gross_pnl - self.cost * cost_multiplier


@dataclass(frozen=True)
class Split:
    train: tuple[int, ...]
    test: tuple[int, ...]


def cpcv(
    events: list[Event], groups: int = 6, test_groups: int = 2, embargo: int = 0
) -> tuple[Split, ...]:
    """All fold combinations with event-overlap purging and post-test embargo.

    Chronological nonoverlapping test folds; train rows whose label interval
    intersects ANY test label interval are excluded. Time units are caller's.
    """
    n = len(events)
    if not 2 <= groups <= n or not 1 <= test_groups < groups or embargo < 0:
        raise ValueError("invalid CPCV settings")
    if any(a.start > b.start for a, b in zip(events, events[1:], strict=False)):
        raise ValueError("events must be ordered")
    folds = [tuple(range((n * k) // groups, (n * (k + 1)) // groups)) for k in range(groups)]
    out = []
    for selection in combinations(range(groups), test_groups):
        test = tuple(i for g in selection for i in folds[g])
        test_set = set(test)
        intervals = [(events[i].start, events[i].end) for i in test]
        last_end_by_fold = [max(events[i].end for i in folds[g]) for g in selection]
        train = []
        for i, ev in enumerate(events):
            if i in test_set:
                continue
            intersects = any(ev.start <= end and ev.end >= start for start, end in intervals)
            post_embargo = any(end < ev.start <= end + embargo for end in last_end_by_fold)
            if not intersects and not post_embargo:
                train.append(i)
        out.append(Split(tuple(train), test))
    return tuple(out)


def profit_factor(values: list[float]) -> float:
    if not values or any(not isfinite(v) for v in values):
        raise ValueError("invalid PnL")
    wins = sum(max(0, v) for v in values)
    losses = -sum(min(0, v) for v in values)
    return wins / losses if losses > 0 else (float("inf") if wins > 0 else 0.0)


def max_drawdown(equity: list[float]) -> float:
    if not equity or any(not isfinite(v) or v <= 0 for v in equity):
        raise ValueError("invalid equity")
    peak = equity[0]
    maximum = 0.0
    for value in equity:
        peak = max(peak, value)
        maximum = max(maximum, 1 - value / peak)
    return maximum


def deflated_sharpe_probability(
    returns: list[float], n_trials: int, trial_sr_variance: float
) -> float:
    """Bailey-Lopez de Prado DSR expressed as probability, always [0,1].

    Trial SR variance is estimated across all tried strategies, NEVER set to
    zero just because only the selected model's returns were retained.
    """
    n = len(returns)
    if (
        n < 30
        or n_trials < 2
        or trial_sr_variance <= 0
        or not isfinite(trial_sr_variance)
        or any(not isfinite(v) for v in returns)
    ):
        raise ValueError("insufficient independent trials/returns")
    mu = mean(returns)
    sd = stdev(returns)
    if sd <= 0:
        raise ValueError("zero variance")
    sr = mu / sd
    skew = sum(((v - mu) / sd) ** 3 for v in returns) / n
    kurt = sum(((v - mu) / sd) ** 4 for v in returns) / n
    normal = NormalDist()
    gamma = 0.5772156649015329
    expected_z = (1 - gamma) * normal.inv_cdf(1 - 1 / n_trials) + gamma * normal.inv_cdf(
        1 - 1 / (n_trials * e)
    )
    sr_star = sqrt(trial_sr_variance) * expected_z
    denom = 1 - skew * sr + (kurt - 1) * sr * sr / 4
    if denom <= 0:
        raise ValueError("invalid Sharpe sampling variance")
    z = (sr - sr_star) * sqrt(n - 1) / sqrt(denom)
    return min(1, max(0, 0.5 * (1 + erf(z / sqrt(2)))))


def bootstrap_stress(
    events: list[Event], *, iterations: int = 400, multiplier: float = 1.5, seed: int = 17
) -> dict[str, float]:
    if len(events) < 30 or not 50 <= iterations <= 10000:
        raise ValueError("insufficient Monte Carlo sample")
    values = [e.net(multiplier) for e in events]
    rng = Random(seed)
    pf = []
    survive = 0
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for _ in values]
        p = profit_factor(sample)
        pf.append(p)
        survive += sum(sample) > 0
    pf.sort()
    return {
        "p05_profit_factor": pf[max(0, int(iterations * 0.05) - 1)],
        "positive_net_rate": survive / iterations,
    }


@dataclass(frozen=True)
class ShadowFill:
    id: str
    timestamp: int
    observed_slippage_bps: float
    simulated_slippage_bps: float


def shadow_gate(fills: list[ShadowFill], min_fills: int = 30, max_delta_bps: float = 3) -> bool:
    if len(fills) < min_fills or len({f.id for f in fills}) != len(fills):
        return False
    if any(
        not f.id
        or f.timestamp <= 0
        or not isfinite(f.observed_slippage_bps)
        or not isfinite(f.simulated_slippage_bps)
        for f in fills
    ):
        return False
    if any(a.timestamp >= b.timestamp for a, b in zip(fills, fills[1:], strict=False)):
        return False
    return (
        mean(abs(f.observed_slippage_bps - f.simulated_slippage_bps) for f in fills) < max_delta_bps
    )


def parameter_stability(
    base_sr: float, jittered_sharpes: list[float], max_degradation: float = 0.2
) -> bool:
    if (
        not isfinite(base_sr)
        or base_sr <= 0
        or not jittered_sharpes
        or any(not isfinite(v) for v in jittered_sharpes)
    ):
        return False
    return all((base_sr - sr) / base_sr < max_degradation for sr in jittered_sharpes)


@dataclass(frozen=True)
class GateResult:
    approved: bool
    reasons: tuple[str, ...]
    net_profit_factor: float
    dsr_probability: float
    drawdown: float


def promotion_gate(
    *,
    trades: list[Event],
    equity: list[float],
    dsr_probability: float,
    shadow: list[ShadowFill],
    base_sharpe: float,
    jittered_sharpes: list[float],
    full_oos_days: int,
    fee_stress: float = 1.25,
) -> GateResult:
    """Promotion eligibility only. NEVER connects or promotes a broker account."""
    pf = profit_factor([e.net(fee_stress) for e in trades])
    dd = max_drawdown(equity)
    reasons = []
    if not 0 <= dsr_probability <= 1 or dsr_probability < 0.95:
        reasons.append("DSR_probability_gate")
    if len(trades) < 30:
        reasons.append("minimum_oos_trade_count")
    if pf <= 1.5:
        reasons.append("net_profit_factor_gate")
    if full_oos_days < 90 or dd >= 0.08:
        reasons.append("90_day_drawdown_gate")
    if not parameter_stability(base_sharpe, jittered_sharpes):
        reasons.append("parameter_cliff_gate")
    if not shadow_gate(shadow):
        reasons.append("30_independent_shadow_fills_gate")
    return GateResult(not reasons, tuple(reasons), pf, dsr_probability, dd)

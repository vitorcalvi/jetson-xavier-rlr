"""Evidence-driven, PAPER-ONLY autonomous strategy selection for RLR.

No broker, HTTP, model, key, or deployment access. This module proposes a
strategy *only*; the existing signed authority, P0 order controller and risk
governor remain mandatory before any external effect.

Candidates are registered in a durable experiment registry. Evaluations use
chronological, disjoint daily net-R outcomes, and a multiple-trial-adjusted
confidence bound. They cannot prove an edge by themselves; untouched research
provenance, independent review, and forward execution-cost audits are required.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import NormalDist, mean, stdev

NO_TRADE = "NO_TRADE"
PAPER_ELIGIBLE = "PAPER_ELIGIBLE"


class AutonomySafetyError(RuntimeError):
    """Unsafe inputs, registry mutation or audit corruption."""


@dataclass(frozen=True)
class DailyOutcome:
    """Daily aggregate, net of modeled spread, fees and slippage, in R."""

    day: str
    net_r: float
    trades: int

    def __post_init__(self) -> None:
        try:
            date.fromisoformat(self.day)
        except (ValueError, TypeError) as exc:
            raise AutonomySafetyError("invalid trading day") from exc
        if not math.isfinite(self.net_r) or abs(self.net_r) > 100:
            raise AutonomySafetyError("invalid daily net-R")
        if not isinstance(self.trades, int) or isinstance(self.trades, bool) or self.trades < 0:
            raise AutonomySafetyError("invalid daily trade count")
        if self.trades == 0 and self.net_r != 0:
            raise AutonomySafetyError("nonzero R without a trade")


@dataclass(frozen=True)
class Candidate:
    strategy_id: str
    code_sha256: str
    symbol: str
    feed: str
    regimes: frozenset[str]
    train: tuple[DailyOutcome, ...]
    validation: tuple[DailyOutcome, ...]
    holdout: tuple[DailyOutcome, ...]
    forward_paper: tuple[DailyOutcome, ...]
    costs_audited: bool
    leakage_audited: bool
    holdout_sealed_before_testing: bool
    artifact_sha256: str

    def fingerprint(self) -> str:
        """Identifier to be bound by an external verified signed scope."""
        data = [
            self.strategy_id,
            self.code_sha256,
            self.symbol,
            self.feed,
            sorted(self.regimes),
            self.artifact_sha256,
        ]
        return _digest(data)

    def frozen_study_digest(self) -> str:
        """Bind static research data and eligibility rules at registration."""
        return _digest(
            {
                "strategy_id": self.strategy_id,
                "code_sha256": self.code_sha256,
                "symbol": self.symbol,
                "feed": self.feed,
                "regimes": sorted(self.regimes),
                "train": [asdict(row) for row in self.train],
                "validation": [asdict(row) for row in self.validation],
                "holdout": [asdict(row) for row in self.holdout],
                "costs_audited": self.costs_audited,
                "leakage_audited": self.leakage_audited,
                "holdout_sealed_before_testing": self.holdout_sealed_before_testing,
            }
        )


@dataclass(frozen=True)
class MarketContext:
    symbol: str
    regime: str
    now_epoch: int
    feed: str
    quote_age_seconds: float
    spread_bps: float
    broker_reconciled: bool
    kill_active: bool
    incident_open: bool
    device_healthy: bool
    risk_budget_available: bool
    signed_authority_verified: bool
    signed_authority_expiry: int
    authorized_fingerprints: frozenset[str]


@dataclass(frozen=True)
class EvaluationPolicy:
    min_train_days: int = 30
    min_validation_days: int = 20
    min_holdout_days: int = 30
    min_forward_days: int = 20
    min_holdout_trades: int = 30
    min_forward_trades: int = 30
    max_drawdown_r: float = 8.0
    max_quote_age_seconds: float = 20.0
    max_spread_bps: float = 20.0
    alpha: float = 0.05
    require_sip_for_paper: bool = True
    max_forward_age_days: int = 14


@dataclass(frozen=True)
class Selection:
    mode: str
    strategy_id: str | None
    reason: str
    score_r: float | None = None
    strategy_fingerprint: str | None = None
    # Execution may NOT infer authorization from this field; risk-reducing
    # exits must continue through the separate reconciled P0 governor.
    entry_proposal_only: bool = True


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=lambda x: sorted(x) if isinstance(x, (set, frozenset)) else str(x),
        ).encode()
    ).hexdigest()


DEFAULT_EVALUATION_POLICY = EvaluationPolicy()


class ExperimentRegistry:
    """Durable record of every tried strategy variant, including failures.

    Only add variants (never delete/update). The trials count used by the
    multiple-testing correction comes from this database, not candidates.
    SQLite is not tamper-proof: the host and backups must be access controlled.
    """

    def __init__(self, path: str | Path):
        self._db = sqlite3.connect(str(path), isolation_level=None, timeout=10)
        self._db.execute("PRAGMA busy_timeout=10000")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS trials ("
            "strategy_id TEXT NOT NULL, code_sha256 TEXT NOT NULL, "
            "symbol TEXT NOT NULL, artifact_sha256 TEXT NOT NULL, "
            "study_sha256 TEXT NOT NULL, "
            "PRIMARY KEY(strategy_id, code_sha256, symbol))"
        )
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS decisions ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, tick TEXT UNIQUE NOT NULL, "
            "payload TEXT NOT NULL, prior_hash TEXT NOT NULL, row_hash TEXT NOT NULL)"
        )
        self.verify_chain()

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> ExperimentRegistry:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def register(self, candidate: Candidate) -> None:
        _validate_identity(candidate)
        self._db.execute("BEGIN IMMEDIATE")
        try:
            old = self._db.execute(
                "SELECT artifact_sha256, study_sha256 FROM trials WHERE "
                "strategy_id=? AND code_sha256=? AND symbol=?",
                (candidate.strategy_id, candidate.code_sha256, candidate.symbol),
            ).fetchone()
            if old is not None and old != (
                candidate.artifact_sha256,
                candidate.frozen_study_digest(),
            ):
                raise AutonomySafetyError("experiment artifact mutated; register a new variant")
            if old is None:
                self._db.execute(
                    "INSERT INTO trials VALUES(?,?,?,?,?)",
                    (
                        candidate.strategy_id,
                        candidate.code_sha256,
                        candidate.symbol,
                        candidate.artifact_sha256,
                        candidate.frozen_study_digest(),
                    ),
                )
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise

    def contains(self, candidate: Candidate) -> bool:
        row = self._db.execute(
            "SELECT artifact_sha256, study_sha256 FROM trials WHERE "
            "strategy_id=? AND code_sha256=? AND symbol=?",
            (candidate.strategy_id, candidate.code_sha256, candidate.symbol),
        ).fetchone()
        return row is not None and row == (
            candidate.artifact_sha256,
            candidate.frozen_study_digest(),
        )

    def trial_count(self) -> int:
        return int(self._db.execute("SELECT COUNT(*) FROM trials").fetchone()[0])

    def verify_chain(self) -> None:
        prior = "GENESIS"
        for tick, payload, stored_prior, row_hash in self._db.execute(
            "SELECT tick,payload,prior_hash,row_hash FROM decisions ORDER BY id"
        ):
            if stored_prior != prior or row_hash != _digest([tick, payload, prior]):
                raise AutonomySafetyError("autonomy journal chain corrupted")
            prior = row_hash

    def record(self, tick: str, selection: Selection, context: MarketContext) -> None:
        """Idempotent on an identical tick; never silently rewrites history."""
        # Fixed-width UTC ISO ticks are lexicographically monotonic.
        try:
            parsed = datetime.strptime(tick, "%Y-%m-%dT%H:%M:%SZ")
            if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != tick:
                raise ValueError("noncanonical tick")
        except (ValueError, TypeError) as exc:
            raise AutonomySafetyError("invalid canonical UTC decision tick") from exc
        payload = json.dumps(
            {"selection": asdict(selection), "context_digest": _digest(asdict(context))},
            sort_keys=True,
            separators=(",", ":"),
            default=lambda v: sorted(v) if isinstance(v, frozenset) else str(v),
        )
        self._db.execute("BEGIN IMMEDIATE")
        try:
            latest = self._db.execute(
                "SELECT tick,payload,row_hash FROM decisions ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if latest is not None and tick < latest[0]:
                raise AutonomySafetyError("decision clock rolled back")
            old = self._db.execute("SELECT payload FROM decisions WHERE tick=?", (tick,)).fetchone()
            if old is not None:
                if old[0] != payload:
                    raise AutonomySafetyError("same tick with conflicting decision")
            else:
                prior = latest[2] if latest else "GENESIS"
                self._db.execute(
                    "INSERT INTO decisions(tick,payload,prior_hash,row_hash) VALUES(?,?,?,?)",
                    (tick, payload, prior, _digest([tick, payload, prior])),
                )
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise


def _validate_identity(candidate: Candidate) -> None:
    if not candidate.strategy_id or len(candidate.strategy_id) > 128:
        raise AutonomySafetyError("invalid strategy ID")
    if not candidate.symbol or len(candidate.symbol) > 16:
        raise AutonomySafetyError("invalid symbol")
    for item in (candidate.code_sha256, candidate.artifact_sha256):
        if len(item) != 64 or any(c not in "0123456789abcdef" for c in item):
            raise AutonomySafetyError("invalid provenance SHA256")


def _max_drawdown(daily: tuple[DailyOutcome, ...]) -> float:
    peak = equity = dd = 0.0
    for outcome in daily:
        equity += outcome.net_r
        peak = max(peak, equity)
        dd = max(dd, peak - equity)
    return dd


def _lower_bound(daily: tuple[DailyOutcome, ...], n_trials: int, alpha: float) -> float:
    """Conservative normal approximation across daily observations.

    Not a substitute for embargoed walk-forward experiments or bootstrap
    estimates when tails and serial dependence are material.
    """
    returns = [d.net_r for d in daily]
    if len(returns) < 2:
        return -math.inf
    if not 0 < alpha < 1 or not 1 <= n_trials <= 1_000_000:
        raise AutonomySafetyError("invalid multiple-testing parameters")
    z = NormalDist().inv_cdf(1 - alpha / (2 * n_trials))
    if not math.isfinite(z):
        return -math.inf
    return mean(returns) - z * stdev(returns) / math.sqrt(len(returns))


def _validate_chronology(candidate: Candidate) -> bool:
    periods = (candidate.train, candidate.validation, candidate.holdout, candidate.forward_paper)
    last_day = ""
    for period in periods:
        for outcome in period:
            if outcome.day <= last_day:
                return False
            last_day = outcome.day
    return True


def _context_failure(ctx: MarketContext, policy: EvaluationPolicy) -> str | None:
    if not ctx.signed_authority_verified or ctx.now_epoch >= ctx.signed_authority_expiry:
        return "signed_authority_invalid_or_expired"
    if ctx.kill_active or ctx.incident_open:
        return "kill_or_incident"
    if not ctx.broker_reconciled:
        return "broker_not_reconciled"
    if not ctx.device_healthy or not ctx.risk_budget_available:
        return "device_or_risk_unhealthy"
    if (
        not math.isfinite(ctx.quote_age_seconds)
        or ctx.quote_age_seconds < 0
        or ctx.quote_age_seconds > policy.max_quote_age_seconds
    ):
        return "market_data_stale"
    if (
        not math.isfinite(ctx.spread_bps)
        or ctx.spread_bps < 0
        or ctx.spread_bps > policy.max_spread_bps
    ):
        return "spread_unacceptable"
    if policy.require_sip_for_paper and ctx.feed != "sip":
        return "sip_feed_required"
    return None


def select_strategy(
    candidates: tuple[Candidate, ...],
    context: MarketContext,
    registry: ExperimentRegistry,
    policy: EvaluationPolicy = DEFAULT_EVALUATION_POLICY,
) -> Selection:
    """Choose a vetted PAPER proposal or autonomously refuse all new entries.

    Operational facts and verified signed scope must be injected by independent
    runtime adapters. Neither this selector nor an LLM issues broker permits.
    """
    bad = _context_failure(context, policy)
    if bad:
        return Selection(NO_TRADE, None, bad)
    trials = registry.trial_count()
    if not trials:
        return Selection(NO_TRADE, None, "no_registered_experiments")
    qualified: list[tuple[float, str, str]] = []
    rejects: list[str] = []
    identities: dict[str, str] = {}
    for c in candidates:
        _validate_identity(c)
        if c.strategy_id in identities and identities[c.strategy_id] != c.fingerprint():
            return Selection(NO_TRADE, None, "ambiguous_strategy_identity")
        identities[c.strategy_id] = c.fingerprint()
        reason = None
        if not registry.contains(c):
            reason = "unregistered_or_mutated_experiment"
        elif c.fingerprint() not in context.authorized_fingerprints:
            reason = "strategy_not_in_signed_scope"
        elif c.symbol != context.symbol or context.regime not in c.regimes:
            reason = "symbol_or_regime_not_eligible"
        elif c.feed != context.feed:
            reason = "feed_mismatch"
        elif not (c.costs_audited and c.leakage_audited and c.holdout_sealed_before_testing):
            reason = "audit_incomplete"
        elif not _validate_chronology(c):
            reason = "chronology_or_leakage"
        elif (
            len(c.train) < policy.min_train_days
            or len(c.validation) < policy.min_validation_days
            or len(c.holdout) < policy.min_holdout_days
            or len(c.forward_paper) < policy.min_forward_days
        ):
            reason = "insufficient_independent_sessions"
        elif (
            sum(x.trades for x in c.holdout) < policy.min_holdout_trades
            or sum(x.trades for x in c.forward_paper) < policy.min_forward_trades
        ):
            reason = "insufficient_trades"
        elif not (
            0
            <= (
                datetime.fromtimestamp(context.now_epoch, tz=timezone.utc).date()
                - date.fromisoformat(c.forward_paper[-1].day)
            ).days
            <= policy.max_forward_age_days
        ):
            reason = "forward_evidence_stale_or_future"
        elif (
            _max_drawdown(c.holdout) > policy.max_drawdown_r
            or _max_drawdown(c.forward_paper) > policy.max_drawdown_r
        ):
            reason = "drawdown_exceeded"
        else:
            # Include a train/validation consistency check; never rank on
            # validation alone. Holdout + prospective paper evidence determine eligibility.
            train_lb = _lower_bound(c.train, trials, policy.alpha)
            valid_lb = _lower_bound(c.validation, trials, policy.alpha)
            held_lb = _lower_bound(c.holdout, trials, policy.alpha)
            forward_lb = _lower_bound(c.forward_paper, trials, policy.alpha)
            if min(train_lb, valid_lb, held_lb, forward_lb) <= 0:
                reason = "no_statistically_supported_net_edge"
            else:
                qualified.append((min(held_lb, forward_lb), c.strategy_id, c.fingerprint()))
        if reason:
            rejects.append(f"{c.strategy_id}:{reason}")
    if not qualified:
        return Selection(NO_TRADE, None, ";".join(rejects) or "no_candidates")
    # Prefer strongest independently supported edge, then deterministic ID.
    score, strategy_id, fingerprint = sorted(qualified, key=lambda x: (-x[0], x[1]))[0]
    return Selection(
        PAPER_ELIGIBLE,
        strategy_id,
        "evidence_and_scope_passed",
        round(score, 6),
        fingerprint,
    )


def propose_and_journal(
    tick: str,
    candidates: tuple[Candidate, ...],
    context: MarketContext,
    registry: ExperimentRegistry,
    policy: EvaluationPolicy = DEFAULT_EVALUATION_POLICY,
) -> Selection:
    """Bounded orchestration for a single tick, including refusal and incidents.

    A failure to *evaluate* yields NO_TRADE. Failure to *audit* raises and
    requires the caller to disable all new entries, not to retry dispatch.
    """
    try:
        selection = select_strategy(candidates, context, registry, policy)
    except (AutonomySafetyError, ValueError, TypeError, OverflowError, sqlite3.Error):
        selection = Selection(NO_TRADE, None, "evaluation_failed_closed")
    try:
        registry.record(tick, selection, context)
    except (AutonomySafetyError, ValueError, TypeError, OverflowError, sqlite3.Error) as exc:
        raise AutonomySafetyError("journal_unavailable_disable_entries") from exc
    return selection


__all__ = [
    "NO_TRADE",
    "PAPER_ELIGIBLE",
    "AutonomySafetyError",
    "DailyOutcome",
    "Candidate",
    "MarketContext",
    "EvaluationPolicy",
    "Selection",
    "ExperimentRegistry",
    "select_strategy",
    "propose_and_journal",
]

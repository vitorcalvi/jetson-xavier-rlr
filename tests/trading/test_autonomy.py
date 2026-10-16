"""Offline evidence governor tests: no credentials, network, broker or live effects."""

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from reflective_loop_runtime.trading.autonomy import (
    NO_TRADE,
    PAPER_ELIGIBLE,
    AutonomySafetyError,
    Candidate,
    DailyOutcome,
    EvaluationPolicy,
    ExperimentRegistry,
    MarketContext,
    Selection,
    propose_and_journal,
    select_strategy,
)

TODAY = date(2026, 10, 12)
NOW = int(datetime(2026, 10, 12, 17, 30, tzinfo=timezone.utc).timestamp())


def days(first: int, count: int, r: float = 0.4, trades: int = 2):
    # Day 99 is yesterday; the entire experiment is chronological.
    return tuple(
        DailyOutcome((TODAY - timedelta(days=100 - first - i)).isoformat(), r, trades)
        for i in range(count)
    )


def good_candidate(sid="A", r=0.4):
    return Candidate(
        strategy_id=sid,
        code_sha256="a" * 64,
        symbol="IREN",
        feed="sip",
        regimes=frozenset({"trend_up"}),
        train=days(0, 30, r),
        validation=days(30, 20, r),
        holdout=days(50, 30, r),
        forward_paper=days(80, 20, r),
        costs_audited=True,
        leakage_audited=True,
        holdout_sealed_before_testing=True,
        artifact_sha256="b" * 64,
    )


def context(*cs):
    return MarketContext(
        symbol="IREN",
        regime="trend_up",
        now_epoch=NOW,
        feed="sip",
        quote_age_seconds=2,
        spread_bps=5,
        broker_reconciled=True,
        kill_active=False,
        incident_open=False,
        device_healthy=True,
        risk_budget_available=True,
        signed_authority_verified=True,
        signed_authority_expiry=NOW + 3600,
        authorized_fingerprints=frozenset(c.fingerprint() for c in cs),
    )


def select(tmp_path: Path, c: Candidate, ctx=None, *, extra=()):
    tmp_path.mkdir(parents=True, exist_ok=True)
    with ExperimentRegistry(tmp_path / "experiments.db") as reg:
        reg.register(c)
        for e in extra:
            reg.register(e)
        return select_strategy((c,), ctx or context(c), reg)


def test_operator_policy_can_tighten_spread_without_bypassing_scope(tmp_path):
    c = good_candidate()
    with ExperimentRegistry(tmp_path / "db") as reg:
        reg.register(c)
        result = select_strategy((c,), context(c), reg, EvaluationPolicy(max_spread_bps=3.0))
        assert result.mode == NO_TRADE
        assert result.reason == "spread_unacceptable"


def test_empty_registry_is_no_trade(tmp_path):
    c = good_candidate()
    with ExperimentRegistry(tmp_path / "db.sqlite") as reg:
        assert select_strategy((c,), context(c), reg).reason == "no_registered_experiments"


def test_strong_prospective_evidence_can_propose_paper_not_execute(tmp_path):
    c = good_candidate()
    ans = select(tmp_path, c)
    assert (ans.mode, ans.strategy_id, ans.entry_proposal_only) == (PAPER_ELIGIBLE, "A", True)
    assert ans.score_r > 0
    assert "LIVE" not in repr(ans)


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"kill_active": True}, "kill_or_incident"),
        ({"incident_open": True}, "kill_or_incident"),
        ({"broker_reconciled": False}, "broker_not_reconciled"),
        ({"device_healthy": False}, "device_or_risk_unhealthy"),
        ({"risk_budget_available": False}, "device_or_risk_unhealthy"),
        ({"signed_authority_verified": False}, "signed_authority_invalid_or_expired"),
        ({"signed_authority_expiry": NOW}, "signed_authority_invalid_or_expired"),
        ({"quote_age_seconds": 30}, "market_data_stale"),
        ({"quote_age_seconds": float("nan")}, "market_data_stale"),
        ({"spread_bps": 21}, "spread_unacceptable"),
        ({"spread_bps": float("inf")}, "spread_unacceptable"),
        ({"feed": "iex"}, "sip_feed_required"),
    ],
)
def test_global_safety_veto_is_fail_closed(tmp_path, change, reason):
    c = good_candidate()
    ans = select(tmp_path, c, replace(context(c), **change))
    assert ans.mode == NO_TRADE
    assert ans.reason == reason


def test_no_scope_no_symbol_and_no_regime(tmp_path):
    c = good_candidate()
    assert (
        "strategy_not_in_signed_scope"
        in select(
            tmp_path / "scope", c, replace(context(c), authorized_fingerprints=frozenset())
        ).reason
    )
    assert (
        "symbol_or_regime_not_eligible"
        in select(tmp_path / "symbol", c, replace(context(c), symbol="NVDA")).reason
    )
    assert (
        "symbol_or_regime_not_eligible"
        in select(tmp_path / "regime", c, replace(context(c), regime="chop")).reason
    )


def test_negative_holdout_trumps_attractive_validation(tmp_path):
    c = replace(good_candidate(), holdout=days(50, 30, -0.1))
    ans = select(tmp_path, c)
    assert ans.mode == NO_TRADE
    assert "no_statistically_supported_net_edge" in ans.reason


def test_negative_paper_blocks_activation(tmp_path):
    c = replace(good_candidate(), forward_paper=days(80, 20, -0.3))
    assert "no_statistically_supported_net_edge" in select(tmp_path, c).reason


def test_bad_chronology_duplicated_days_or_lookahead_rejected(tmp_path):
    c = good_candidate()
    overlap = replace(c, holdout=days(40, 30))
    assert "chronology_or_leakage" in select(tmp_path / "overlap", overlap).reason
    future = replace(
        c,
        forward_paper=tuple(
            DailyOutcome((TODAY + timedelta(days=x)).isoformat(), 0.5, 2) for x in range(1, 21)
        ),
    )
    assert "forward_evidence_stale_or_future" in select(tmp_path / "future", future).reason


def test_insufficient_days_trades_costs_and_provenance(tmp_path):
    c = good_candidate()
    for i, (variant, needle) in enumerate(
        [
            (replace(c, forward_paper=days(80, 19)), "insufficient_independent_sessions"),
            (replace(c, holdout=days(50, 30, 0.0, trades=0)), "insufficient_trades"),
            (replace(c, costs_audited=False), "audit_incomplete"),
            (replace(c, leakage_audited=False), "audit_incomplete"),
            (replace(c, holdout_sealed_before_testing=False), "audit_incomplete"),
            (replace(c, feed="iex"), "feed_mismatch"),
        ]
    ):
        result = select(tmp_path / f"{i}-{needle}", variant)
        assert needle in result.reason


def test_stale_evidence_and_drawdown_veto(tmp_path):
    c = good_candidate()
    stale = replace(
        c,
        train=days(-21, 30),
        validation=days(9, 20),
        holdout=days(29, 30),
        forward_paper=days(59, 20),
    )
    # 20 trading sessions all dated at least 20 days ago.
    assert "forward_evidence_stale_or_future" in select(tmp_path / "stale", stale).reason
    deep = replace(
        c,
        holdout=tuple(
            replace(row, net_r=-1.0 if i < 10 else 1.4) for i, row in enumerate(c.holdout)
        ),
    )
    assert "drawdown_exceeded" in select(tmp_path / "dd", deep).reason


def test_strategies_rank_by_strongest_independent_lower_bound(tmp_path):
    a = good_candidate("A", r=0.3)
    b = good_candidate("B", r=0.5)
    with ExperimentRegistry(tmp_path / "db") as reg:
        reg.register(a)
        reg.register(b)
        result = select_strategy((a, b), context(a, b), reg)
        assert result.strategy_id == "B" and result.mode == PAPER_ELIGIBLE


def test_multiple_trials_penalty_makes_weak_edge_ineligible(tmp_path):
    c = good_candidate(r=0.2)
    # Daily uncertainty makes high trial count harder to pass.
    h = tuple(replace(x, net_r=0.4 if i % 2 else -0.05) for i, x in enumerate(c.holdout))
    c = replace(
        c,
        holdout=h,
        forward_paper=tuple(
            replace(x, net_r=0.4 if i % 2 else -0.05) for i, x in enumerate(c.forward_paper)
        ),
    )
    with ExperimentRegistry(tmp_path / "db") as reg:
        reg.register(c)
        one = select_strategy((c,), context(c), reg)
        for i in range(250):
            reg.register(good_candidate(f"trial_{i}"))
        many = select_strategy((c,), context(c), reg)
    assert many.mode == NO_TRADE
    assert many.reason.endswith("no_statistically_supported_net_edge")
    assert one.mode in (PAPER_ELIGIBLE, NO_TRADE)


def test_registry_counts_variants_and_protects_mutation(tmp_path):
    c = good_candidate()
    with ExperimentRegistry(tmp_path / "db") as reg:
        reg.register(c)
        reg.register(c)
        assert reg.trial_count() == 1
        with pytest.raises(AutonomySafetyError, match="artifact mutated"):
            reg.register(replace(c, artifact_sha256="c" * 64))
        forged = replace(c, holdout=days(50, 30, 0.9))
        with pytest.raises(AutonomySafetyError, match="artifact mutated"):
            reg.register(forged)
        assert not reg.contains(forged)
        reg.register(replace(c, code_sha256="c" * 64))
        assert reg.trial_count() == 2
        assert not reg.contains(replace(c, artifact_sha256="c" * 64))


def test_chain_survives_restart_and_rejects_rewrites_and_clock_reversal(tmp_path):
    p = tmp_path / "db"
    c = good_candidate()
    with ExperimentRegistry(p) as reg:
        reg.register(c)
        ctx = context(c)
        result = select_strategy((c,), ctx, reg)
        reg.record("2026-10-12T17:31:00Z", result, ctx)
        reg.record("2026-10-12T17:31:00Z", result, ctx)
        reg.record("2026-10-12T17:32:00Z", Selection(NO_TRADE, None, "test"), ctx)
        with pytest.raises(AutonomySafetyError, match="conflicting"):
            reg.record("2026-10-12T17:32:00Z", result, ctx)
        with pytest.raises(AutonomySafetyError, match="rolled back"):
            reg.record("2026-10-12T17:30:00Z", result, ctx)
    with ExperimentRegistry(p) as reg:
        reg.verify_chain()
        assert reg.trial_count() == 1
    import sqlite3

    with sqlite3.connect(p) as db:
        db.execute("UPDATE decisions SET payload='changed' WHERE id=1")
    with pytest.raises(AutonomySafetyError, match="chain corrupted"):
        ExperimentRegistry(p)


def test_invalid_daily_data_is_rejected():
    with pytest.raises(AutonomySafetyError):
        DailyOutcome("bogus", 1, 1)
    with pytest.raises(AutonomySafetyError):
        DailyOutcome("2026-10-12", float("nan"), 1)
    with pytest.raises(AutonomySafetyError):
        DailyOutcome("2026-10-12", -1, -1)
    with pytest.raises(AutonomySafetyError):
        DailyOutcome("2026-10-12", 1, 0)


def test_strategy_identity_is_not_ambiguous(tmp_path):
    c = good_candidate()
    other = replace(c, code_sha256="c" * 64)
    with ExperimentRegistry(tmp_path / "db") as reg:
        reg.register(c)
        reg.register(other)
        result = select_strategy((c, other), context(c, other), reg)
        assert result.mode == NO_TRADE
        assert result.reason == "ambiguous_strategy_identity"


def test_propose_and_journal_fails_closed_on_malformed_candidate(tmp_path):
    c = good_candidate()
    with ExperimentRegistry(tmp_path / "db") as reg:
        reg.register(c)
        bad = replace(c, code_sha256="not-a-sha256")
        result = propose_and_journal("2026-10-12T17:40:00Z", (bad,), context(c), reg)
        assert result.mode == NO_TRADE
        assert result.reason == "evaluation_failed_closed"
        reg.verify_chain()


def test_propose_and_journal_refuses_duplicate_tick_with_conflicting_context(tmp_path):
    c = good_candidate()
    with ExperimentRegistry(tmp_path / "db") as reg:
        reg.register(c)
        result = propose_and_journal("2026-10-12T17:40:00Z", (c,), context(c), reg)
        assert result.mode == PAPER_ELIGIBLE
        assert result.strategy_fingerprint == c.fingerprint()
        with pytest.raises(AutonomySafetyError, match="journal_unavailable"):
            propose_and_journal(
                "2026-10-12T17:40:00Z",
                (c,),
                replace(context(c), quote_age_seconds=999),
                reg,
            )


def test_noncanonical_clock_string_does_not_bypass_rollback_gate(tmp_path):
    c = good_candidate()
    with ExperimentRegistry(tmp_path / "db") as reg:
        reg.register(c)
        with pytest.raises(AutonomySafetyError, match="canonical UTC"):
            reg.record("localtime-2026-10-12", Selection(NO_TRADE, None, "test"), context(c))
        with pytest.raises(AutonomySafetyError, match="canonical UTC"):
            reg.record("2026-2-1T01:02:03Z", Selection(NO_TRADE, None, "test"), context(c))

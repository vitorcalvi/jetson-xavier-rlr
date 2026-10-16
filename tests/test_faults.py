"""L4 fault matrix: one test per row, plus Executive safe-state integration.

Each row asserts its audited safe state, its recovery path, and -- where the row
has a concrete effect -- the action the controller actually takes (freeze
dependents, mark a step uncertain, engage the kill latch, withhold A2/A3). The
policies are signed with the dev-only HMAC-SHA256 fallback so the suite runs
without the optional ``cryptography`` package.
"""

from __future__ import annotations

import json

import pytest

from reflective_loop_runtime.events import KILL, EventBus, KillSwitch
from reflective_loop_runtime.faults import (
    FAULT_NAMES,
    MATRIX,
    OBSERVATION_ONLY,
    RECOVER_INVESTIGATE_TAINT,
    RECOVER_RECONCILE,
    RECOVER_REFRESH_STATE,
    RECOVER_RELOAD_POLICY,
    RECOVER_REPAIR_AUDIT,
    RECOVER_REPAIR_DB,
    RECOVER_REPLAN,
    RECOVER_RESTORE_LAN,
    RECOVER_RETRY_BOUNDED,
    RECOVER_RETRY_SAFE,
    RECOVER_REVALIDATE,
    RECOVER_WAIT_BUDGET,
    SAFE_ABSTAIN,
    SAFE_DENY_SPEND,
    SAFE_FREEZE_DEPENDENTS,
    SAFE_HALT_PERSISTENCE,
    SAFE_LOCAL_BOUNDED,
    SAFE_NO_A2_A3,
    SAFE_QUARANTINE,
    SAFE_RECONCILE,
    Fault,
    FaultController,
    MatrixError,
)
from reflective_loop_runtime.kernel.audit import AuditLog
from reflective_loop_runtime.kernel.permit import PermitDenied, PermitIssuer, PermitService
from reflective_loop_runtime.kernel.policy import MissionPolicyStore, sign_envelope
from reflective_loop_runtime.kernel.toolbroker import ToolBroker
from reflective_loop_runtime.plan import (
    STEP_DONE,
    STEP_PAUSED,
    STEP_UNCERTAIN,
    Executive,
    PlanGraph,
)

DEV_SECRET = "dev-only-shared-secret"
ISSUER_KEY = b"governor-owned-hmac-key-0123456789"

EXPECTED_ROWS = {
    "model_timeout",
    "malformed_model_output",
    "stale_state",
    "db_failure",
    "audit_failure",
    "tool_error_before_effect",
    "tool_timeout_unknown_effect",
    "budget_exceeded",
    "taint_violation",
    "partial_multi_step",
    "lan_reasoner_down",
    "invalid_policy_or_kill",
}


def enforced_policy(tmp_path, *, name="faults-policy.json"):
    bundle = {
        "mission_id": "mission-7",
        "objective": "keep the node stable",
        "invariants": ["never_shell_out"],
        "allowed_domains": [],
        "capabilities": [],
        "budgets": {},
        "standing_mandates": [],
        "policy_version": "1.0.0",
    }
    path = tmp_path / name
    path.write_text(json.dumps(sign_envelope(bundle, secret=DEV_SECRET)), encoding="utf-8")
    return MissionPolicyStore(secret=DEV_SECRET, allow_dev_hmac=True).load(path)


def add_step(
    plan,
    *,
    goal_id="g-1",
    tool="state",
    op="set_load_target",
    args=None,
    depends_on=(),
    step_id=None,
):
    return plan.add_step(
        goal_id=goal_id,
        tool=tool,
        op=op,
        args=args if args is not None else {"load": 0.25},
        depends_on=depends_on,
        step_id=step_id,
    )


@pytest.fixture
def audit():
    return AuditLog(":memory:")


@pytest.fixture
def policy(tmp_path):
    return enforced_policy(tmp_path)


@pytest.fixture
def permits(policy, tmp_path):
    return PermitService(PermitIssuer(ISSUER_KEY), policy, db_path=tmp_path / "fault-permits.db")


@pytest.fixture
def calls():
    return {"state": 0}


@pytest.fixture
def broker(permits, audit, calls):
    broker = ToolBroker(permits, audit=audit)

    def adapter(args):
        calls["state"] += 1
        return {"load_target": args["load"]}

    broker.register_effect("state", "set_load_target", adapter)
    return broker


@pytest.fixture
def plan(audit):
    return PlanGraph(":memory:", audit=audit)


class TestFaultMatrix:
    def test_matrix_contains_every_required_row(self):
        assert set(FAULT_NAMES) == EXPECTED_ROWS
        assert len(MATRIX) == len(EXPECTED_ROWS)

    def test_every_row_has_a_safe_state_and_recovery(self):
        for spec in MATRIX.rows():
            assert spec.safe_state
            assert spec.recovery
            assert spec.allowed_risk_classes

    def test_unknown_fault_is_rejected(self):
        with pytest.raises(MatrixError):
            MATRIX.lookup("no_such_fault")

    def test_fault_value_type_validates_its_name(self):
        assert str(Fault("model_timeout")) == "model_timeout"
        assert Fault("audit_failure").spec().safe_state == SAFE_NO_A2_A3
        with pytest.raises(MatrixError):
            Fault("nope")


class TestModelTimeoutRow:
    def test_model_timeout_abstains_and_bounds_the_retry(self):
        bus = EventBus()
        controller = FaultController(bus=bus)
        response = controller.report("model_timeout", reason="no answer in time")
        assert response.safe_state == SAFE_ABSTAIN
        assert response.recovery == RECOVER_RETRY_BOUNDED
        assert controller.retry_allowed("model_timeout") is True
        assert bus.last("deadline") is not None


class TestMalformedModelOutputRow:
    def test_malformed_output_abstains_and_must_be_revalidated(self):
        bus = EventBus()
        controller = FaultController(bus=bus)
        response = controller.report("malformed_model_output", reason="not valid JSON")
        assert response.safe_state == SAFE_ABSTAIN
        assert response.recovery == RECOVER_REVALIDATE
        assert controller.retry_allowed("malformed_model_output") is True
        assert bus.last("deadline") is not None


class TestStaleStateRow:
    def test_stale_state_forces_observation_only_until_refresh(self):
        controller = FaultController()
        response = controller.report("stale_state", reason="sample too old")
        assert response.safe_state == OBSERVATION_ONLY
        assert response.recovery == RECOVER_REFRESH_STATE
        assert controller.observation_only is True
        assert controller.permits_allowed("A0") is True
        assert controller.permits_allowed("A1") is False

    def test_clearing_the_fault_restores_permits(self):
        controller = FaultController()
        controller.report("stale_state")
        assert controller.permits_allowed("A1") is False
        assert controller.clear("stale_state") is True
        assert controller.permits_allowed("A3") is True


class TestDbFailureRow:
    def test_db_failure_halts_persistence(self):
        controller = FaultController()
        response = controller.report("db_failure", reason="disk error")
        assert response.safe_state == SAFE_HALT_PERSISTENCE
        assert response.recovery == RECOVER_REPAIR_DB
        assert controller.permits_allowed("A1") is False
        assert controller.permits_allowed("A0") is True


class TestAuditFailureRow:
    def test_audit_failure_withholds_a2_and_a3_permits(self):
        controller = FaultController()
        response = controller.report("audit_failure", reason="chain broken")
        assert response.safe_state == SAFE_NO_A2_A3
        assert response.recovery == RECOVER_REPAIR_AUDIT
        assert controller.permits_allowed("A0") is True
        assert controller.permits_allowed("A1") is True
        assert controller.permits_allowed("A2") is False
        assert controller.permits_allowed("A3") is False


class TestToolErrorBeforeEffectRow:
    def test_tool_error_before_effect_allows_a_safe_retry(self):
        bus = EventBus()
        controller = FaultController(bus=bus)
        response = controller.report("tool_error_before_effect", reason="adapter raised")
        assert response.safe_state == SAFE_ABSTAIN
        assert response.recovery == RECOVER_RETRY_SAFE
        assert controller.retry_allowed("tool_error_before_effect") is True
        assert bus.last("tool_failure") is not None


class TestToolTimeoutUnknownEffectRow:
    def test_unknown_effect_is_quarantined_and_never_blindly_retried(self, plan, audit):
        bus = EventBus()
        add_step(plan, step_id="a")
        controller = FaultController(plan=plan, bus=bus)
        response = controller.report(
            "tool_timeout_unknown_effect", step_id="a", reason="call timed out"
        )
        assert response.safe_state == SAFE_RECONCILE
        assert response.recovery == RECOVER_RECONCILE
        assert "marked_uncertain" in response.applied
        assert plan.get("a").status == STEP_UNCERTAIN
        assert controller.retry_allowed("tool_timeout_unknown_effect") is False
        assert controller.reconcile_required is True
        assert bus.last("tool_failure") is not None


class TestBudgetExceededRow:
    def test_budget_exceeded_denies_new_spend(self):
        controller = FaultController()
        response = controller.report("budget_exceeded", reason="cap reached")
        assert response.safe_state == SAFE_DENY_SPEND
        assert response.recovery == RECOVER_WAIT_BUDGET
        assert controller.permits_allowed("A1") is False
        assert controller.permits_allowed("A0") is True


class TestTaintViolationRow:
    def test_taint_violation_quarantines_side_effects(self):
        controller = FaultController()
        response = controller.report("taint_violation", reason="untrusted evidence")
        assert response.safe_state == SAFE_QUARANTINE
        assert response.recovery == RECOVER_INVESTIGATE_TAINT
        assert controller.quarantined is True
        assert controller.permits_allowed("A1") is False
        assert controller.permits_allowed("A0") is True


class TestPartialMultiStepRow:
    def test_partial_multi_step_freezes_dependents(self, plan, audit):
        bus = EventBus()
        a = add_step(plan, step_id="a")
        b = add_step(plan, step_id="b", depends_on=[a.step_id])
        controller = FaultController(plan=plan, bus=bus)
        response = controller.report(
            "partial_multi_step", goal_id="g-1", step_id="a", reason="step half applied"
        )
        assert response.safe_state == SAFE_FREEZE_DEPENDENTS
        assert response.recovery == RECOVER_REPLAN
        assert "froze_dependents" in response.applied
        assert response.replan_id
        assert plan.get(b.step_id).status == STEP_PAUSED
        assert plan.is_paused("g-1")
        assert len(plan.replan_requests(resolved=False)) == 1


class TestLanReasonerDownRow:
    def test_lan_reasoner_down_switches_to_local_bounded_mode(self):
        bus = EventBus()
        controller = FaultController(bus=bus)
        response = controller.report("lan_reasoner_down", reason="lan host unreachable")
        assert response.safe_state == SAFE_LOCAL_BOUNDED
        assert response.recovery == RECOVER_RESTORE_LAN
        assert controller.local_bounded is True
        assert controller.allows_reasoner("lan_reasoner") is False
        assert controller.allows_reasoner("qwen") is False
        assert controller.allows_reasoner("local_bounded") is True
        assert controller.allows_reasoner("open_decider") is True


class TestInvalidPolicyOrKillRow:
    def test_invalid_policy_or_kill_is_observation_only(self, permits, tmp_path):
        bus = EventBus()
        switch = KillSwitch(tmp_path / "KILL", bus=bus, permits=permits)
        controller = FaultController(bus=bus, kill_switch=switch)
        response = controller.report("invalid_policy_or_kill", reason="kill file present")
        assert response.safe_state == OBSERVATION_ONLY
        assert response.recovery == RECOVER_RELOAD_POLICY
        assert controller.observation_only is True
        assert switch.engaged is True
        assert bus.last(KILL) is not None
        with pytest.raises(PermitDenied):
            permits.issue(tool="state", op="set_load_target", args={"load": 0.1}, goal_id="g")


class TestExecutiveSafeStateIntegration:
    def test_observation_only_refuses_before_minting_a_permit(
        self, plan, permits, broker, audit, calls
    ):
        controller = FaultController()
        executive = Executive(plan, permits, broker, audit=audit, faults=controller)
        step = add_step(plan, step_id="a")
        controller.report("invalid_policy_or_kill")
        result = executive.run_step(step.step_id)
        assert result.ok is False
        assert "observation only" in result.note
        assert calls["state"] == 0

    def test_audit_failure_blocks_a2_but_allows_a1(self, plan, permits, broker, audit, calls):
        controller = FaultController()
        executive = Executive(plan, permits, broker, audit=audit, faults=controller)
        controller.report("audit_failure")

        a1 = add_step(plan, step_id="a1")
        a2 = plan.add_step(
            goal_id="g-1",
            tool="notify",
            op="send",
            args={"channel": "local_log", "message": "hello"},
            step_id="a2",
        )

        assert executive.run_step(a1.step_id).status == STEP_DONE
        blocked = executive.run_step(a2.step_id)
        assert blocked.ok is False
        assert "A2" in blocked.note
        # The blocked A2 never reached a tool adapter (notify has none registered).
        assert calls["state"] == 1

"""L2 plan layer: dependency order, the permit gate, discrepancy, recovery.

Policies are signed with the dev-only HMAC-SHA256 fallback so the suite runs
without the optional ``cryptography`` package; that still yields an ``ENFORCE``
policy, which is what the permit service and budget service require.
"""

from __future__ import annotations

import json

import pytest

from reflective_loop_runtime.kernel.audit import AuditLog
from reflective_loop_runtime.kernel.budget import BudgetService
from reflective_loop_runtime.kernel.manifest import (
    CostModel,
    FieldSpec,
    OperationSpec,
    ToolSpec,
    default_manifest,
)
from reflective_loop_runtime.kernel.permit import PermitIssuer, PermitService
from reflective_loop_runtime.kernel.policy import (
    MissionPolicy,
    MissionPolicyStore,
    sign_envelope,
)
from reflective_loop_runtime.kernel.toolbroker import (
    ToolBroker,
    ToolNotPermitted,
    broker_manifest,
)
from reflective_loop_runtime.plan import (
    STEP_DONE,
    STEP_FAILED,
    STEP_PAUSED,
    STEP_PENDING,
    STEP_UNCERTAIN,
    DependencyNotSatisfied,
    Executive,
    PlanError,
    PlanGraph,
    StepNotRunnable,
)

DEV_SECRET = "dev-only-shared-secret"
ISSUER_KEY = b"governor-owned-hmac-key-0123456789"


def make_bundle(**overrides):
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
    bundle.update(overrides)
    return bundle


def enforced_policy(tmp_path, *, name="policy.json", **overrides):
    path = tmp_path / name
    envelope = sign_envelope(make_bundle(**overrides), secret=DEV_SECRET)
    path.write_text(json.dumps(envelope), encoding="utf-8")
    return MissionPolicyStore(secret=DEV_SECRET, allow_dev_hmac=True).load(path)


def tool_spec() -> ToolSpec:
    """A custom ``budget_test.tick`` tool whose cost is a real budget unit."""
    return ToolSpec(
        "budget_test",
        {
            "tick": OperationSpec(
                name="tick",
                input_schema={"cost": FieldSpec("float", min=0.0, max=10.0)},
                preconditions=("the mission budget allows one api call",),
                effects=("one api call is accounted",),
                risk_class="A1",
                reversibility="reversible",
                idempotency="idempotent",
                compensation="refund the reservation",
                cost_model=CostModel("api_calls", 1.0, "one account tick"),
            )
        },
    )


def counting_broker(permits, audit, calls):
    broker = ToolBroker(permits, audit=audit)

    def adapter(args):
        calls["state"] += 1
        return {"load_target": args["load"]}

    broker.register_effect("state", "set_load_target", adapter)
    return broker


def add_step(plan, *, goal_id="g-1", load=0.25, expected=None, depends_on=(), step_id=None):
    return plan.add_step(
        goal_id=goal_id,
        tool="state",
        op="set_load_target",
        args={"load": load},
        expected_effects=expected or {},
        depends_on=depends_on,
        step_id=step_id,
    )


@pytest.fixture
def policy(tmp_path):
    return enforced_policy(tmp_path)


@pytest.fixture
def audit():
    return AuditLog(":memory:")


@pytest.fixture
def issuer():
    return PermitIssuer(ISSUER_KEY)


@pytest.fixture
def permits(policy, issuer, tmp_path):
    return PermitService(issuer, policy, db_path=tmp_path / "permits.db")


@pytest.fixture
def calls():
    return {"state": 0}


@pytest.fixture
def broker(permits, audit, calls):
    return counting_broker(permits, audit, calls)


@pytest.fixture
def plan(tmp_path, audit):
    return PlanGraph(tmp_path / "plan.db", audit=audit)


@pytest.fixture
def executive(plan, permits, broker, audit):
    return Executive(plan, permits, broker, audit=audit)


class TestDependencyOrdering:
    def test_topological_order_follows_dependencies(self):
        plan = PlanGraph(":memory:")
        a = add_step(plan, step_id="a")
        b = add_step(plan, step_id="b", depends_on=[a.step_id])
        add_step(plan, step_id="c", depends_on=[b.step_id])
        assert plan.topological_order() == ["a", "b", "c"]

        # Addition order must not matter: the dependency order still wins.
        shuffled = PlanGraph(":memory:")
        add_step(shuffled, step_id="c", depends_on=["b"])
        add_step(shuffled, step_id="b", depends_on=["a"])
        add_step(shuffled, step_id="a")
        assert shuffled.topological_order() == ["a", "b", "c"]

    def test_ready_steps_gate_on_dependencies(self, plan, executive):
        a = add_step(plan, step_id="a")
        add_step(plan, step_id="b", depends_on=[a.step_id])
        assert [s.step_id for s in plan.ready_steps()] == ["a"]
        executive.run_step("a")
        assert [s.step_id for s in plan.ready_steps()] == ["b"]

    def test_running_before_a_dependency_is_refused(self, plan, executive, calls):
        a = add_step(plan, step_id="a")
        add_step(plan, step_id="b", depends_on=[a.step_id])
        with pytest.raises(DependencyNotSatisfied):
            executive.run_step("b")
        assert calls["state"] == 0

    def test_cycle_is_detected(self, plan):
        add_step(plan, step_id="a", depends_on=["b"])
        add_step(plan, step_id="b", depends_on=["a"])
        with pytest.raises(PlanError, match="cycle"):
            plan.topological_order()

    def test_unknown_dependency_is_detected(self, plan):
        add_step(plan, step_id="a", depends_on=["ghost"])
        with pytest.raises(PlanError, match="unknown"):
            plan.topological_order()


class TestPermitGate:
    def test_no_step_runs_without_a_permit(self, audit):
        observation_only = MissionPolicy.observation_only("unsigned policy")
        permits = PermitService(PermitIssuer(ISSUER_KEY), observation_only)
        calls = {"state": 0}
        broker = counting_broker(permits, audit, calls)
        plan = PlanGraph(":memory:", audit=audit)
        step = add_step(plan)
        executive = Executive(plan, permits, broker, audit=audit)

        result = executive.run_step(step.step_id)
        assert result.ok is False
        assert result.status == STEP_FAILED
        assert "permit denied" in result.note
        assert plan.get(step.step_id).status == STEP_FAILED
        # The broker never ran, because no permit was ever issued.
        assert calls["state"] == 0

    def test_broker_refuses_a_missing_permit(self, broker, calls):
        with pytest.raises(ToolNotPermitted):
            broker.invoke(None, "state", "set_load_target", {"load": 0.1})
        assert calls["state"] == 0

    def test_each_step_gets_a_fresh_permit(self, plan, executive):
        add_step(plan, step_id="a")
        add_step(plan, step_id="b")
        first = executive.run_step("a")
        second = executive.run_step("b")
        assert first.ok and second.ok
        assert first.permit_id and second.permit_id
        assert first.permit_id != second.permit_id

    def test_a_done_step_is_never_replayed(self, plan, executive):
        add_step(plan, step_id="a")
        assert executive.run_step("a").ok
        with pytest.raises(StepNotRunnable):
            executive.run_step("a")


class TestDiscrepancy:
    def test_discrepancy_pauses_branch_and_requests_replan(self, plan, executive, calls):
        a = add_step(plan, step_id="a", load=0.25, expected={"load_target": 0.99})
        b = add_step(plan, step_id="b", load=0.5, depends_on=[a.step_id])

        result = executive.run_step("a")
        assert result.ok is False
        assert result.status == STEP_UNCERTAIN
        assert result.replan is not None
        assert result.replan.goal_id == "g-1"
        assert result.replan.expected == {"load_target": 0.99}
        assert result.replan.actual == {"load_target": 0.25}

        assert plan.get("a").status == STEP_UNCERTAIN
        assert plan.get(b.step_id).status == STEP_PAUSED
        assert plan.is_paused("g-1")
        assert len(plan.replan_requests(resolved=False)) == 1
        # The tool did run; only its observed effect disagreed with the plan.
        assert calls["state"] == 1

    def test_replan_resolves_and_the_branch_resumes(self, plan, executive):
        add_step(plan, step_id="a", load=0.25, expected={"load_target": 0.99})
        add_step(plan, step_id="b", load=0.5, depends_on=["a"])
        result = executive.run_step("a")
        plan.resolve_replan(result.replan.replan_id)
        assert plan.resume_branch("g-1") == 1
        assert plan.get("b").status == STEP_PENDING
        assert not plan.is_paused("g-1")

    def test_replan_ids_are_unique_across_resolutions(self, plan):
        add_step(plan, step_id="a")
        first = plan.pause_branch("g-1", reason="one")
        plan.resolve_replan(first.replan_id)
        second = plan.pause_branch("g-1", reason="two")
        assert first.replan_id != second.replan_id
        assert len(plan.replan_requests(resolved=None)) == 2
        assert len(plan.replan_requests(resolved=False)) == 1

    def test_matching_effects_do_not_pause(self, plan, executive):
        add_step(plan, step_id="a", load=0.25, expected={"load_target": 0.25})
        result = executive.run_step("a")
        assert result.ok and result.status == STEP_DONE
        assert result.replan is None
        assert plan.replan_requests() == []


class TestCrashRecovery:
    def _orphaned_plan(self, tmp_path, audit):
        plan = PlanGraph(tmp_path / "crash.db", audit=audit)
        step = add_step(plan, goal_id="g-9")
        plan.mark_running(step.step_id, permit_id="permit-1")
        plan.close()
        return step

    def test_recovered_running_step_becomes_uncertain(self, tmp_path, audit):
        step = self._orphaned_plan(tmp_path, audit)
        reopened = PlanGraph(tmp_path / "crash.db", audit=audit)
        recovered = reopened.get(step.step_id)
        assert recovered.status == STEP_UNCERTAIN
        assert "crash recovery" in recovered.reason
        assert reopened.recovered == [step.step_id]

    def test_recovered_step_is_not_auto_retried(self, tmp_path, audit, permits, broker, calls):
        step = self._orphaned_plan(tmp_path, audit)
        reopened = PlanGraph(tmp_path / "crash.db", audit=audit)
        executive = Executive(reopened, permits, broker, audit=audit)
        with pytest.raises(StepNotRunnable):
            executive.run_step(step.step_id)
        assert calls["state"] == 0

    def test_explicit_reset_allows_a_retry(self, tmp_path, audit, permits, broker):
        step = self._orphaned_plan(tmp_path, audit)
        reopened = PlanGraph(tmp_path / "crash.db", audit=audit)
        executive = Executive(reopened, permits, broker, audit=audit)
        assert reopened.reset_step(step.step_id).status == STEP_PENDING
        assert executive.run_step(step.step_id).status == STEP_DONE

    def test_recover_is_idempotent(self, plan):
        assert plan.recover() == []


class TestPersistence:
    def test_steps_and_dependencies_survive_reopen(self, tmp_path, audit):
        plan = PlanGraph(tmp_path / "persist.db", audit=audit)
        add_step(plan, step_id="a")
        add_step(plan, step_id="b", depends_on=["a"])
        plan.close()

        reopened = PlanGraph(tmp_path / "persist.db", audit=audit)
        assert [s.step_id for s in reopened.steps()] == ["a", "b"]
        assert reopened.get("b").depends_on == ("a",)
        assert reopened.topological_order() == ["a", "b"]

    def test_replan_requests_survive_reopen(self, tmp_path, audit):
        plan = PlanGraph(tmp_path / "persist.db", audit=audit)
        add_step(plan)
        plan.pause_branch("g-1", reason="boom", expected={"x": 1}, actual={"x": 2})
        plan.close()

        reopened = PlanGraph(tmp_path / "persist.db", audit=audit)
        requests = reopened.replan_requests()
        assert len(requests) == 1
        assert requests[0].reason == "boom"
        assert requests[0].actual == {"x": 2}


class TestBudgetIntegration:
    def _harness(self, tmp_path, audit, *, cap):
        manifest = default_manifest()
        manifest.register(tool_spec())
        policy = enforced_policy(tmp_path, budgets={"api_calls": cap}, name="budget-policy.json")
        budget = BudgetService(tmp_path / "budget.db", policy)
        permits = PermitService(
            PermitIssuer(ISSUER_KEY),
            policy,
            budget=budget,
            manifest=manifest,
            db_path=tmp_path / "budget-permits.db",
        )
        broker_manifest_view = broker_manifest()
        broker_manifest_view.register(tool_spec())
        broker = ToolBroker(permits, audit=audit, manifest=broker_manifest_view)
        calls = {"n": 0}
        broker.register_effect(
            "budget_test",
            "tick",
            lambda args: calls.__setitem__("n", calls["n"] + 1) or {"ticked": True},
        )
        return budget, permits, broker, calls

    def test_successful_step_commits_its_reservation(self, tmp_path, audit):
        budget, permits, broker, calls = self._harness(tmp_path, audit, cap=1)
        plan = PlanGraph(":memory:", audit=audit)
        step = plan.add_step(goal_id="g-b", tool="budget_test", op="tick", args={"cost": 1.0})
        executive = Executive(plan, permits, broker, budget=budget, audit=audit)

        result = executive.run_step(step.step_id, actual_spend={"api_calls": 1})
        assert result.ok and result.status == STEP_DONE
        assert budget.scope_usage("goal", "g-b", "api_calls") == 1
        assert calls["n"] == 1

    def test_budget_cap_denies_a_second_step(self, tmp_path, audit):
        budget, permits, broker, calls = self._harness(tmp_path, audit, cap=1)
        plan = PlanGraph(":memory:", audit=audit)
        executive = Executive(plan, permits, broker, budget=budget, audit=audit)
        first = plan.add_step(goal_id="g-b", tool="budget_test", op="tick", args={"cost": 1.0})
        second = plan.add_step(goal_id="g-b", tool="budget_test", op="tick", args={"cost": 1.0})

        assert executive.run_step(first.step_id, actual_spend={"api_calls": 1}).ok
        denied = executive.run_step(second.step_id)
        assert denied.ok is False
        assert denied.status == STEP_FAILED
        assert "permit denied" in denied.note
        assert calls["n"] == 1


class TestAuditTrail:
    def test_all_transitions_are_audited_and_chained(self, plan, executive, audit):
        add_step(plan, step_id="a")
        add_step(plan, step_id="b", depends_on=["a"])
        executive.run_step("a")
        executive.run_step("b")

        assert audit.verify_chain().ok
        outcomes = [event.outcome for event in audit.events()]
        assert "plan.step_added" in outcomes
        assert "plan.running" in outcomes
        assert "plan.done" in outcomes

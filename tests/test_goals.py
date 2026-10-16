"""L1 goal layer: mission root, parenting, explosion caps, arbitration, gating.

Policies are signed with the dev-only HMAC-SHA256 fallback so the suite runs
without the optional ``cryptography`` package; that still yields an ``ENFORCE``
policy, which is what committing a goal requires.
"""

from __future__ import annotations

import json

import pytest

from reflective_loop_runtime.goals import (
    MAX_GOALS,
    STATUS_ACHIEVED,
    STATUS_COMMITTED,
    STATUS_PAUSED,
    STATUS_PROPOSED,
    ConflictResolution,
    GoalConflict,
    GoalManager,
    GoalRejected,
)
from reflective_loop_runtime.kernel.audit import AuditLog
from reflective_loop_runtime.kernel.policy import (
    MissionPolicy,
    MissionPolicyStore,
    sign_envelope,
)

DEV_SECRET = "dev-only-shared-secret"
MISSION_ID = "mission-7"
OBJECTIVE = "keep the node stable"


def make_bundle(**overrides):
    bundle = {
        "mission_id": MISSION_ID,
        "objective": OBJECTIVE,
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


@pytest.fixture
def policy(tmp_path):
    return enforced_policy(tmp_path)


@pytest.fixture
def audit():
    return AuditLog(":memory:")


@pytest.fixture
def manager(policy, audit):
    return GoalManager(policy, audit=audit)


class TestMissionRoot:
    def test_root_is_the_mission(self, manager):
        root = manager.root
        assert root.goal_id == MISSION_ID
        assert root.mission_id == MISSION_ID
        assert root.text == OBJECTIVE
        assert root.parent_id is None
        assert root.depth == 0
        assert root.status == STATUS_COMMITTED

    def test_observation_only_root_is_only_proposed(self):
        manager = GoalManager(MissionPolicy.observation_only("no signature"))
        assert manager.root.status == STATUS_PROPOSED
        assert not manager.policy.enforced


class TestParenting:
    def test_goal_without_a_parent_is_rejected(self, manager):
        with pytest.raises(GoalRejected, match="goal without parent"):
            manager.propose("orphan", parent_id=None)
        with pytest.raises(GoalRejected, match="goal without parent"):
            manager.propose("orphan", parent_id="does-not-exist")
        # Nothing unparented entered the tree.
        assert len(manager) == 1

    def test_child_is_bound_to_its_parent(self, manager):
        child = manager.propose("cool the box", parent_id=manager.root_id)
        assert child.parent_id == manager.root_id
        assert child.depth == 1
        assert child.status == STATUS_PROPOSED
        assert child in manager.children(manager.root_id)


class TestExplosionCaps:
    def test_total_goal_cap(self, policy):
        manager = GoalManager(policy, max_goals=3)
        manager.propose("a", parent_id=manager.root_id)
        manager.propose("b", parent_id=manager.root_id)
        assert len(manager) == 3
        with pytest.raises(GoalRejected, match="explosion"):
            manager.propose("c", parent_id=manager.root_id)

    def test_default_cap_is_bounded(self):
        assert MAX_GOALS > 0
        manager = GoalManager(MissionPolicy.observation_only("x"), max_goals=4, max_depth=1)
        manager.propose("a", parent_id=manager.root_id)
        manager.propose("b", parent_id=manager.root_id)
        manager.propose("c", parent_id=manager.root_id)
        with pytest.raises(GoalRejected, match="explosion"):
            manager.propose("d", parent_id=manager.root_id)

    def test_depth_cap(self, policy):
        manager = GoalManager(policy, max_depth=1)
        child = manager.propose("a", parent_id=manager.root_id)
        with pytest.raises(GoalRejected, match="depth"):
            manager.propose("too deep", parent_id=child.goal_id)


class TestConflictArbitration:
    def _compete(self, policy, order):
        manager = GoalManager(policy)
        for text in order:
            goal = manager.propose(text, parent_id=manager.root_id, priority=1, resource="cpu")
            try:
                manager.commit(goal.goal_id)
            except GoalConflict:
                pass
        return manager

    def test_conflict_arbitration_is_deterministic(self, policy):
        forward = self._compete(policy, ["alpha", "beta"])
        reverse = self._compete(policy, ["beta", "alpha"])
        winners = [g.goal_id for g in forward.committed_goals()]
        assert winners == [g.goal_id for g in reverse.committed_goals()]
        # Text is the first id-independent tiebreak, so "alpha" always wins.
        assert winners == [MISSION_ID, f"{MISSION_ID}/alpha"]
        assert forward.get(f"{MISSION_ID}/beta").status == STATUS_PAUSED
        assert reverse.get(f"{MISSION_ID}/beta").status == STATUS_PAUSED

    def test_priority_decides_the_winner(self, policy):
        manager = GoalManager(policy)
        low = manager.propose("alpha", parent_id=manager.root_id, priority=0, resource="cpu")
        high = manager.propose("zenith", parent_id=manager.root_id, priority=5, resource="cpu")
        manager.commit(low.goal_id)
        # The higher-priority goal takes the resource; committing it does not
        # raise because it is the winner, and the incumbent is paused.
        assert manager.commit(high.goal_id).status == STATUS_COMMITTED
        assert manager.get(low.goal_id).status == STATUS_PAUSED

    def test_commit_loser_is_paused_and_raises(self, policy):
        manager = GoalManager(policy)
        winner = manager.propose("alpha", parent_id=manager.root_id, resource="gpu")
        loser = manager.propose("beta", parent_id=manager.root_id, resource="gpu")
        manager.commit(winner.goal_id)
        with pytest.raises(GoalConflict) as excinfo:
            manager.commit(loser.goal_id)
        assert excinfo.value.winner_id == winner.goal_id
        assert manager.get(loser.goal_id).status == STATUS_PAUSED

    def test_resolve_conflicts_reports_every_loser(self, policy):
        manager = GoalManager(policy)
        for text in ("a", "b", "c"):
            manager.propose(text, parent_id=manager.root_id, resource="io")
        # Commit all three without the automatic arbitration short-circuit.
        for goal in manager.goals():
            if goal.parent_id == manager.root_id:
                manager._transition(goal, STATUS_COMMITTED, reason="test")
        resolutions = manager.resolve_conflicts()
        assert all(isinstance(r, ConflictResolution) for r in resolutions)
        assert {r.loser_id for r in resolutions} == {
            f"{MISSION_ID}/b",
            f"{MISSION_ID}/c",
        }
        assert {r.winner_id for r in resolutions} == {f"{MISSION_ID}/a"}

    def test_arbitrate_orders_by_precedence(self, policy):
        manager = GoalManager(policy)
        low = manager.propose("low", parent_id=manager.root_id, priority=0)
        high = manager.propose("high", parent_id=manager.root_id, priority=9)
        ordered = manager.arbitrate([low.goal_id, high.goal_id])
        assert [g.text for g in ordered] == ["high", "low"]
        # Committing the higher-priority goal makes it the next goal to work on.
        manager.commit(high.goal_id)
        assert manager.next_goal().text == "high"


class TestObservationOnlyGating:
    def test_observation_only_blocks_commits(self):
        manager = GoalManager(MissionPolicy.observation_only("unsigned"))
        child = manager.propose("act now", parent_id=manager.root_id)
        with pytest.raises(GoalRejected, match="not enforced"):
            manager.commit(child.goal_id)
        assert manager.get(child.goal_id).status == STATUS_PROPOSED
        assert manager.committed_goals() == ()
        with pytest.raises(GoalRejected, match="not enforced"):
            manager.resume(child.goal_id)

    def test_enforced_policy_allows_a_commit(self, manager):
        child = manager.propose("act now", parent_id=manager.root_id)
        committed = manager.commit(child.goal_id)
        assert committed.status == STATUS_COMMITTED


class TestTransitionsAndAudit:
    def test_lifecycle_transitions(self, manager):
        child = manager.propose("do work", parent_id=manager.root_id)
        manager.commit(child.goal_id)
        assert manager.pause(child.goal_id).status == STATUS_PAUSED
        assert manager.resume(child.goal_id).status == STATUS_COMMITTED
        assert manager.achieve(child.goal_id).status == STATUS_ACHIEVED
        with pytest.raises(GoalRejected):
            manager.pause(child.goal_id)

    def test_every_state_change_is_audited(self, manager, audit):
        assert len(audit) == 1  # the root
        child = manager.propose("do work", parent_id=manager.root_id)
        manager.commit(child.goal_id)
        manager.pause(child.goal_id)
        outcomes = [event.outcome for event in audit.events()]
        assert "goal.root" in outcomes
        assert "goal.proposed" in outcomes
        assert "goal.committed" in outcomes
        assert "goal.paused" in outcomes
        assert audit.verify_chain().ok

    def test_snapshot_is_deterministic(self, manager):
        manager.propose("beta", parent_id=manager.root_id)
        manager.propose("alpha", parent_id=manager.root_id)
        snapshot = manager.snapshot()
        ids = [g["goal_id"] for g in snapshot["goals"]]
        assert ids == sorted(ids)
        assert snapshot["enforced"] is True

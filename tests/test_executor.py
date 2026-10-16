"""Executor tests: permit-gated side effects only, no privileged capabilities.

Every side effect must carry a Governor-issued permit; the executor refuses a
side-effecting action without one, and never performs the effect for an invalid
or replayed permit.
"""

from __future__ import annotations

import json

import pytest
from conftest import make_config

from reflective_loop_runtime.executor import (
    DEFAULT_CONTROL_STATE,
    Executor,
    effect_request,
)
from reflective_loop_runtime.kernel.audit import AuditLog
from reflective_loop_runtime.kernel.permit import PermitIssuer, PermitService
from reflective_loop_runtime.kernel.policy import load_policy_bytes, sign_envelope
from reflective_loop_runtime.kernel.toolbroker import ToolBroker, broker_manifest
from reflective_loop_runtime.store import Store

DEV_SECRET = "dev-only-shared-secret"
ISSUER_KEY = b"executor-test-issuer-key-0123456789"


def _bundle(**overrides):
    bundle = {
        "mission_id": "mission-exec",
        "objective": "exercise the executor",
        "invariants": ["no_effect_without_permit"],
        "allowed_domains": [],
        "capabilities": [],
        "budgets": {},
        "standing_mandates": [],
        "policy_version": "1.0.0",
    }
    bundle.update(overrides)
    return bundle


def _policy():
    envelope = sign_envelope(_bundle(), secret=DEV_SECRET)
    return load_policy_bytes(
        json.dumps(envelope).encode("utf-8"), secret=DEV_SECRET, allow_dev_hmac=True
    )


@pytest.fixture
def store(tmp_path):
    store = Store(make_config(tmp_path).db_path)
    yield store
    store.close()


@pytest.fixture
def env(tmp_path, store):
    manifest = broker_manifest()
    service = PermitService(
        PermitIssuer(ISSUER_KEY),
        _policy(),
        manifest=manifest,
        db_path=tmp_path / "permits.db",
    )
    broker = ToolBroker(service, audit=AuditLog(":memory:"), manifest=manifest)
    executor = Executor(store, broker=broker)
    yield executor, service
    service.close()


def run(env, action, *, tick=None):
    """Issue the matching permit, then execute the action under it."""
    executor, service = env
    tick = executor.current_tick if tick is None else tick
    request = effect_request(action, tick)
    permit = None
    if request is not None:
        tool, op, args = request
        permit = service.issue(tool=tool, op=op, args=args, goal_id="goal-exec")
    return executor.execute(action, permit=permit, tick=tick)


class TestExecutor:
    def test_observe_and_abstain_have_no_side_effect(self, env):
        executor, _ = env
        before = dict(executor.control_state)
        assert executor.execute({"action": "observe", "rationale": "x"}).ok is True
        assert (
            executor.execute({"action": "abstain", "rationale": "x", "reason": "wait"}).ok is True
        )
        assert executor.control_state == before

    def test_set_vision_fps_updates_control_state(self, env):
        executor, _ = env
        result = run(env, {"action": "set_vision_fps", "rationale": "x", "fps": 1.5})
        assert result.ok
        assert executor.control_state["vision_fps"] == 1.5

    def test_set_reply_tokens_updates_control_state(self, env):
        executor, _ = env
        run(env, {"action": "set_reply_tokens", "rationale": "x", "tokens": 32})
        assert executor.control_state["reply_tokens"] == 32

    def test_set_load_target_updates_control_state(self, env):
        executor, _ = env
        run(env, {"action": "set_load_target", "rationale": "x", "load": 0.5})
        assert executor.control_state["load_target"] == 0.5

    def test_cooldown_sets_floor_and_schedules(self, env):
        executor, _ = env
        result = run(env, {"action": "cooldown", "rationale": "x", "seconds": 30}, tick=5)
        assert result.ok
        assert executor.control_state["cooldown_until_tick"] == 35
        assert executor.control_state["vision_fps"] <= 0.2

    def test_remember_and_recall_and_goals(self, env, store):
        executor, _ = env
        run(env, {"action": "remember", "rationale": "x", "text": "note", "importance": 4})
        assert store.counts()["memories"] == 1
        recalled = run(env, {"action": "recall", "rationale": "x", "query": "note"})
        assert recalled.applied["matches"] == 1
        goal = run(env, {"action": "set_goal", "rationale": "x", "text": "keep cool"})
        assert goal.applied["goal_id"] >= 1
        done = run(
            env,
            {"action": "complete_goal", "rationale": "x", "goal_id": goal.applied["goal_id"]},
        )
        assert done.ok and done.applied["completed"] is True

    def test_complete_unknown_goal_is_not_ok(self, env):
        result = run(env, {"action": "complete_goal", "rationale": "x", "goal_id": 999})
        assert result.ok is False

    def test_invalid_action_rejected(self, env):
        result = run(env, {"action": "shell", "rationale": "x"})
        assert result.ok is False
        assert result.action == "invalid"

    def test_default_control_state_is_bounded(self):
        assert 0.0 < DEFAULT_CONTROL_STATE["vision_fps"] <= 5.0
        assert 16 <= DEFAULT_CONTROL_STATE["reply_tokens"] <= 512
        assert 0.0 <= DEFAULT_CONTROL_STATE["load_target"] <= 1.0


def test_executor_requires_authority_for_side_effects(env):
    """F1 regression: no side effect without a valid Governor-issued permit."""
    executor, service = env
    before = dict(executor.control_state)

    # 1. A side-effecting action with no permit performs nothing.
    denied = executor.execute({"action": "set_load_target", "rationale": "sneak", "load": 0.5})
    assert denied.ok is False
    assert "permit" in denied.note
    assert executor.control_state == before

    # 2. A permit minted by a foreign issuer cannot authorise the effect either.
    forger = PermitService(
        PermitIssuer(b"not-the-governor-issuer-key-0123456789"),
        service.policy,
        manifest=service.manifest,
        db_path=":memory:",
    )
    try:
        forged = forger.issue(
            tool="state", op="set_load_target", args={"load": 0.5}, goal_id="goal-exec"
        )
        rejected = executor.execute(
            {"action": "set_load_target", "rationale": "x", "load": 0.5}, permit=forged
        )
        assert rejected.ok is False
        assert executor.control_state == before
    finally:
        forger.close()

    # 3. A valid permit issued for the exact call performs the effect.
    granted = run(env, {"action": "set_load_target", "rationale": "x", "load": 0.5})
    assert granted.ok is True
    assert executor.control_state["load_target"] == 0.5

    # 4. A permit is single-use: a replay performs no second effect.
    tool, op, args = effect_request({"action": "set_vision_fps", "rationale": "x", "fps": 1.5}, 0)
    permit = service.issue(tool=tool, op=op, args=args, goal_id="goal-exec")
    action = {"action": "set_vision_fps", "rationale": "x", "fps": 1.5}
    assert executor.execute(action, permit=permit).ok is True
    executor.control_state["vision_fps"] = 2.0
    replay = executor.execute(action, permit=permit)
    assert replay.ok is False
    assert executor.control_state["vision_fps"] == 2.0

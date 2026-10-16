"""L3 event layer: bus, heartbeat, async reasoner, hang response, kill switch.

Fake clocks drive the deterministic paths; only the kill-file watchdog test uses
wall-clock time, and it asserts engagement happens inside the 2 second bound.
Policies are signed with the dev-only HMAC-SHA256 fallback so the suite runs
without the optional ``cryptography`` package.
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import pytest

from reflective_loop_runtime.events import (
    DEADLINE,
    EVENT_KINDS,
    KILL,
    POLICY_CHANGE,
    REASONER_RESULT,
    STATE_CHANGE,
    TOOL_COMPLETION,
    TOOL_FAILURE,
    AsyncReasoner,
    EventBus,
    EventError,
    GuardedBroker,
    Heartbeat,
    KillSwitch,
    LaunchBlocked,
)
from reflective_loop_runtime.kernel.audit import AuditLog
from reflective_loop_runtime.kernel.permit import PermitDenied, PermitIssuer, PermitService
from reflective_loop_runtime.kernel.policy import MissionPolicyStore, sign_envelope
from reflective_loop_runtime.kernel.toolbroker import ToolBroker
from reflective_loop_runtime.plan import STEP_DONE, Executive, PlanGraph

DEV_SECRET = "dev-only-shared-secret"
ISSUER_KEY = b"governor-owned-hmac-key-0123456789"


class FakeClock:
    """A monotonic fake clock the tests advance by hand."""

    def __init__(self, t: float = 0.0):
        self.t = float(t)

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> float:
        self.t += float(dt)
        return self.t


def enforced_policy(tmp_path, *, name="events-policy.json"):
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


@pytest.fixture
def policy(tmp_path):
    return enforced_policy(tmp_path)


class TestEventBus:
    def test_all_required_kinds_can_be_published(self):
        bus = EventBus()
        for kind in EVENT_KINDS:
            event = bus.emit(kind, n=1)
            assert event.kind == kind
        assert len(bus.events()) == len(EVENT_KINDS)

    def test_subscribers_receive_matching_and_wildcard_events(self):
        bus = EventBus()
        seen: list[str] = []
        bus.subscribe(TOOL_COMPLETION, lambda e: seen.append("completion"))
        bus.subscribe("*", lambda e: seen.append(f"all:{e.kind}"))
        bus.emit(TOOL_COMPLETION)
        bus.emit(STATE_CHANGE)
        assert seen == ["completion", "all:tool_completion", "all:state_change"]

    def test_history_is_queryable_and_ordered(self):
        bus = EventBus()
        bus.emit(TOOL_FAILURE, reason="a")
        bus.emit(TOOL_FAILURE, reason="b")
        events = bus.events(TOOL_FAILURE)
        assert [e.payload["reason"] for e in events] == ["a", "b"]
        assert bus.last(TOOL_FAILURE).payload["reason"] == "b"
        assert bus.sequence == 2

    def test_unknown_kind_is_rejected(self):
        bus = EventBus()
        with pytest.raises(EventError):
            bus.emit("not_a_kind")
        with pytest.raises(EventError):
            bus.subscribe("not_a_kind", lambda e: None)

    def test_unsubscribe_stops_delivery(self):
        bus = EventBus()
        seen: list[str] = []
        handler = bus.subscribe(DEADLINE, lambda e: seen.append("x"))
        assert bus.unsubscribe(DEADLINE, handler)
        bus.emit(DEADLINE)
        assert seen == []
        assert not bus.unsubscribe(DEADLINE, handler)


class TestHeartbeat:
    def test_interval_must_be_within_one_to_five_seconds(self):
        bus = EventBus()
        Heartbeat(bus, interval_seconds=1.0)
        Heartbeat(bus, interval_seconds=5.0)
        with pytest.raises(EventError):
            Heartbeat(bus, interval_seconds=0.5)
        with pytest.raises(EventError):
            Heartbeat(bus, interval_seconds=5.5)

    def test_beats_only_when_due(self):
        clock = FakeClock()
        bus = EventBus(clock=clock)
        hb = Heartbeat(bus, clock=clock, interval_seconds=2.0)
        assert hb.beat() is not None  # first beat is always due
        clock.advance(0.5)
        assert hb.beat() is None
        clock.advance(1.5)
        assert hb.beat() is not None
        assert len(bus.events("heartbeat")) == 2
        assert hb.beats == 2

    def test_heartbeat_never_calls_a_model(self):
        clock = FakeClock()
        bus = EventBus(clock=clock)
        hb = Heartbeat(bus, clock=clock, interval_seconds=1.0)
        reasoner = AsyncReasoner(bus=bus, clock=clock)
        calls = {"n": 0}

        def model() -> str:  # pragma: no cover - must never run from a heartbeat
            calls["n"] += 1
            return "nope"

        for _step in range(6):
            clock.advance(1.0)
            hb.beat()
        assert hb.beats == 6
        assert calls["n"] == 0
        assert reasoner.accepted == 0
        assert bus.events(REASONER_RESULT) == []
        assert bus.events(DEADLINE) == []


class TestAsyncReasoner:
    def test_fresh_result_is_accepted(self):
        clock = FakeClock()
        bus = EventBus(clock=clock)
        reasoner = AsyncReasoner(bus=bus, clock=clock)
        call_id = reasoner.submit("qwen", lambda: "advice", state_version="v7", expiry_seconds=30)
        assert reasoner.join(call_id, timeout=2.0)

        results = reasoner.poll(current_state_version="v7")
        assert [r.value for r in results] == ["advice"]
        assert results[0].accepted and results[0].state_version == "v7"
        assert bus.last(REASONER_RESULT).payload["accepted"] is True

    def test_stale_state_result_is_discarded(self):
        clock = FakeClock()
        reasoner = AsyncReasoner(clock=clock)
        call_id = reasoner.submit("qwen", lambda: "old", state_version="v1", expiry_seconds=30)
        assert reasoner.join(call_id, timeout=2.0)

        assert reasoner.poll(current_state_version="v2") == []
        assert reasoner.dropped == 1
        assert reasoner.accepted == 0

    def test_expired_result_is_discarded_and_reported(self):
        clock = FakeClock()
        bus = EventBus(clock=clock)
        reasoner = AsyncReasoner(bus=bus, clock=clock)
        call_id = reasoner.submit("qwen", lambda: "late", state_version="v1", expiry_seconds=5)
        assert reasoner.join(call_id, timeout=2.0)

        clock.advance(6.0)
        assert reasoner.poll(current_state_version="v1") == []
        assert bus.last(DEADLINE) is not None
        assert bus.last(REASONER_RESULT).payload["reason"] == "expired"

    def test_failed_call_is_discarded(self):
        reasoner = AsyncReasoner()

        def boom() -> str:
            raise RuntimeError("model exploded")

        call_id = reasoner.submit("qwen", boom, state_version="v1")
        assert reasoner.join(call_id, timeout=2.0)
        assert reasoner.poll(current_state_version="v1") == []
        assert reasoner.dropped == 1

    def test_hanging_call_does_not_stall_the_scheduler(self):
        clock = FakeClock()
        bus = EventBus(clock=clock)
        reasoner = AsyncReasoner(bus=bus, clock=clock)
        hb = Heartbeat(bus, clock=clock, interval_seconds=1.0)
        release = threading.Event()

        def hanging() -> str:
            release.wait(5.0)
            return "eventually"

        started = time.monotonic()
        call_id = reasoner.submit("lan_reasoner", hanging, state_version="v1", expiry_seconds=600)
        assert time.monotonic() - started < 0.5  # submit is non-blocking
        assert reasoner.poll(current_state_version="v1") == []  # poll returns immediately
        assert reasoner.pending() == 1

        # The scheduler keeps beating while the model hangs.
        clock.advance(3.0)
        assert hb.beat() is not None

        release.set()
        assert reasoner.join(call_id, timeout=2.0)
        assert [r.value for r in reasoner.poll(current_state_version="v1")] == ["eventually"]

    def test_cancel_all_drops_pending_results(self):
        reasoner = AsyncReasoner()
        for _ in range(3):
            reasoner.submit("qwen", lambda: "x", state_version="v1")
        assert reasoner.cancel_all() == 3
        assert reasoner.pending() == 0
        assert reasoner.poll(current_state_version="v1") == []


class FakeBroker:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, permit, tool, op, args=None):
        self.calls += 1
        return "ran"


class TestKillSwitch:
    def test_engagement_revokes_permits_and_blocks_launches(self, policy, tmp_path):
        bus = EventBus()
        permits = PermitService(
            PermitIssuer(ISSUER_KEY),
            policy,
            db_path=tmp_path / "kill-permits.db",
        )
        switch = KillSwitch(tmp_path / "KILL", bus=bus, permits=permits)
        assert switch.register_permit("permit-1")
        assert switch.live_permits == 1

        assert switch.engage("operator kill")
        assert switch.engaged
        assert switch.live_permits == 0
        assert not switch.allow_launch("permit-1")
        with pytest.raises(LaunchBlocked):
            switch.require_launch("permit-1")
        assert bus.last(KILL) is not None
        assert bus.last(POLICY_CHANGE) is not None

        # The permit service has been dropped to observation-only authority.
        assert permits.policy.enforced is False
        with pytest.raises(PermitDenied):
            permits.issue(tool="state", op="set_load_target", args={"load": 0.1}, goal_id="g")

    def test_engage_is_idempotent(self, tmp_path):
        switch = KillSwitch(tmp_path / "KILL")
        assert switch.engage() is True
        assert switch.engage() is False

    def test_guarded_broker_refuses_after_a_kill(self, tmp_path):
        bus = EventBus()
        switch = KillSwitch(None, bus=bus)
        inner = FakeBroker()
        guarded = GuardedBroker(inner, switch)
        permit = SimpleNamespace(permit_id="p-1")
        switch.register_permit("p-1")
        assert guarded.invoke(permit, "state", "set_load_target", {"load": 0.1}) == "ran"

        switch.engage("stop")
        with pytest.raises(LaunchBlocked):
            guarded.invoke(permit, "state", "set_load_target", {"load": 0.1})
        assert inner.calls == 1

    def test_kill_file_watchdog_engages_within_two_seconds(self, tmp_path):
        kill_path = tmp_path / "KILL"
        switch = KillSwitch(kill_path, poll_interval=0.05)
        switch.start_watchdog()
        try:
            assert not switch.engaged
            started = time.monotonic()
            kill_path.write_text("stop\n", encoding="utf-8")
            limit = started + 2.0
            while time.monotonic() < limit and not switch.engaged:
                time.sleep(0.02)
            elapsed = time.monotonic() - started
        finally:
            switch.stop_watchdog()
        assert switch.engaged
        assert elapsed < 2.0
        assert switch.reason == "kill file present"

    def test_check_engages_on_a_present_file(self, tmp_path):
        kill_path = tmp_path / "KILL"
        switch = KillSwitch(kill_path)
        kill_path.write_text("stop\n", encoding="utf-8")
        assert switch.check() is True
        assert switch.engaged

    def test_from_config_resolves_the_path(self, tmp_path):
        default_cfg = SimpleNamespace(data_dir=tmp_path, extra={})
        assert KillSwitch.from_config(default_cfg).path == tmp_path / "KILL"

        explicit = tmp_path / "custom.kill"
        explicit_cfg = SimpleNamespace(data_dir=tmp_path, kill_file=explicit, extra={})
        assert KillSwitch.from_config(explicit_cfg).path == explicit

        extra_cfg = SimpleNamespace(data_dir=tmp_path, extra={"kill_file": explicit})
        assert KillSwitch.from_config(extra_cfg).path == explicit

    def test_unknown_launch_is_refused_after_a_kill(self, tmp_path):
        switch = KillSwitch(tmp_path / "KILL")
        switch.engage()
        assert switch.allow_launch(None) is False
        assert switch.is_revoked("never-registered")


class TestExecutiveEvents:
    def _executive(self, tmp_path, *, bus, kill_switch=None):
        audit = AuditLog(":memory:")
        policy = enforced_policy(tmp_path)
        permits = PermitService(
            PermitIssuer(ISSUER_KEY), policy, db_path=tmp_path / "exec-permits.db"
        )
        broker = ToolBroker(permits, audit=audit)
        calls = {"n": 0}
        broker.register_effect(
            "state",
            "set_load_target",
            lambda args: calls.__setitem__("n", calls["n"] + 1) or {"load_target": args["load"]},
        )
        plan = PlanGraph(":memory:", audit=audit)
        executive = Executive(plan, permits, broker, audit=audit, bus=bus, kill_switch=kill_switch)
        return plan, executive, calls

    def test_lifecycle_events_are_published(self, tmp_path):
        bus = EventBus()
        plan, executive, calls = self._executive(tmp_path, bus=bus)
        step = plan.add_step(goal_id="g-1", tool="state", op="set_load_target", args={"load": 0.25})
        result = executive.run_step(step.step_id)
        assert result.ok
        assert calls["n"] == 1
        assert bus.last(STATE_CHANGE).payload["status"] == STEP_DONE
        assert bus.last(TOOL_COMPLETION) is not None
        assert bus.last(DEADLINE) is not None

    def test_kill_switch_stops_executive_launches(self, tmp_path):
        bus = EventBus()
        switch = KillSwitch(None, bus=bus)
        plan, executive, calls = self._executive(tmp_path, bus=bus, kill_switch=switch)
        first = plan.add_step(goal_id="g-1", tool="state", op="set_load_target", args={"load": 0.1})
        assert executive.run_step(first.step_id).ok
        assert calls["n"] == 1

        switch.engage("operator stop")
        second = plan.add_step(
            goal_id="g-1", tool="state", op="set_load_target", args={"load": 0.2}
        )
        result = executive.run_step(second.step_id)
        assert result.ok is False
        assert "kill switch" in result.note
        assert calls["n"] == 1  # the broker was never reached
        assert bus.last(KILL).payload["step_id"] == second.step_id

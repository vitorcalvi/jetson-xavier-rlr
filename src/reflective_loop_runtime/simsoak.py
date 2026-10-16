"""Simsoak: a deterministic, faults-injected soak of the trusted kernel.

``simsoak`` drives a synthetic closed loop against the *real* kernel objects --
the permit service, the budget ledger, the hash-chained audit log, the plan graph
and the permit-gated :class:`~.kernel.toolbroker.ToolBroker` -- while

* mock models (the deterministic OpenDecider and a mock slow brain) propose work;
* mock A2/A3 adapters are registered as tripwires, so a genuine A2/A3 effect
  firing is detected as a policy violation rather than silently performed;
* the sandboxed A0/A1 ``shell`` tool is exercised through the broker, failing
  closed when bubblewrap is absent;
* random fault-matrix rows are injected and cleared, forcing the loop through
  observation-only / no-A2-A3 / deny-spend safe states.

It then emits a TOON report with the acceptance metrics: goals completed, false
success, policy violations (must be ``0``), permits issued/consumed, p50/p95
tick latency, RAM, and the final audit-chain verification.

Duration
--------
``--hours`` / ``--minutes`` / ``--seconds`` set the *simulated* soak length. A
private monotonic clock advances by ``--tick-seconds`` (default ``1``) per tick,
so ``--minutes 2`` is 120 deterministic ticks executed as fast as the host can
run them -- no wall-clock sleeping and no network.

The module is stdlib-only. Like ``redteam`` it does not import ``subprocess`` or
any network primitive (see ``tests/kernel/test_static_guard.py``); the sandbox is
reached solely through the broker, whose ``bwrap_available`` check is a
file/exec test.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import resource
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .candidates import build_candidates
from .config import Config
from .deciders import DeterministicOpenDecider
from .events import (
    BUDGET_THRESHOLD,
    DEADLINE,
    POLICY_CHANGE,
    STATE_CHANGE,
    AsyncReasoner,
    EventBus,
    Heartbeat,
)
from .faults import FAULT_NAMES, FaultController
from .goals import GoalConflict, GoalManager, GoalRejected
from .kernel.audit import AuditLog
from .kernel.budget import BudgetService
from .kernel.permit import PermitIssuer, PermitService
from .kernel.policy import load_policy_bytes, sign_envelope
from .kernel.toolbroker import ToolBroker, broker_manifest
from .plan import Executive, PlanGraph
from .qwen import CallableSlowBrain
from .redteam import toon_encode

__all__ = [
    "SoakResult",
    "build_parser",
    "main",
    "run_soak",
]

DEV_SECRET = "rlr-simsoak-dev-hmac-secret"
ISSUER_KEY = b"rlr-simsoak-issuer-key-0123456789ab"
# A fixed dev audit key keeps the soak's chain keyed (so a rewritten row cannot
# be re-forged) without printing key material.
AUDIT_KEY = b"rlr-simsoak-audit-key-0123456789abcdef"

_POLICY_VERSION = "simsoak-1.0.0"
# Cap the number of goals proposed so a very long soak cannot exhaust the L1
# explosion guard; once the cap is reached the soak keeps running without new
# goals.
_MAX_SOAK_GOALS = 100
# A fault row is injected, holds for a bounded window, then clears so the loop
# keeps making progress (an un-cleared observation-only row would stall it).
_FAULT_HOLD_TICKS = 2
_FAULT_INJECT_PROBABILITY = 0.30

_BASE_BUNDLE: dict[str, Any] = {
    "mission_id": "simsoak-mission",
    "objective": "drive a synthetic closed loop to completion",
    "invariants": ["never_shell_out_unsandboxed"],
    "allowed_domains": ["telemetry", "state", "memory", "goal", "notify", "shell", "control"],
    # The soak must mint permits for every risk class it exercises, so the
    # capability list is left empty (no restriction); capability scoping is
    # covered by the security-regression suite.
    "capabilities": [],
    "budgets": {"api_calls": 1_000_000, "llm_tokens": 100_000, "money": 1_000},
    "standing_mandates": ["memory.forget"],
    "policy_version": _POLICY_VERSION,
}

# The mock model's candidate-action -> concrete tool operation map.
_TOOL_PLAN: dict[str, tuple[str, str, dict[str, Any]]] = {
    "observe": ("telemetry", "sample", {}),
    "cooldown": ("control", "cooldown", {"seconds": 30, "tick": 0}),
    "set_vision_fps": ("state", "set_vision_fps", {"fps": 2.0}),
    "set_reply_tokens": ("state", "set_reply_tokens", {"tokens": 96}),
    "set_load_target": ("state", "set_load_target", {"load": 0.25}),
    "remember": ("memory", "remember", {"text": "soak memory", "importance": 3}),
    "recall": ("memory", "recall", {"query": "soak"}),
    "set_goal": ("goal", "set", {"text": "soak goal"}),
    "complete_goal": ("goal", "complete", {"goal_id": 7}),
    "reflect": ("control", "reflect", {}),
    "abstain": ("memory", "forget", {"memory_id": 1}),
}

_THERMALS = (58.0, 66.0, 72.0, 76.5, 79.0, 74.0, 62.0)


class _Clock:
    """A private monotonic clock advanced by one tick at a time."""

    def __init__(self, start: float = 0.0):
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += float(seconds)
        return self.now


class _RecordingPermits:
    """A thin proxy over ``PermitService`` that records every issued permit."""

    def __init__(self, inner: PermitService):
        self._inner = inner
        self.issued: list[Any] = []

    def issue(self, **kwargs: Any) -> Any:
        permit = self._inner.issue(**kwargs)
        self.issued.append(permit)
        return permit

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@dataclass(frozen=True)
class SoakResult:
    """The full outcome of a soak: metrics plus the injected-fault ledger."""

    metrics: Mapping[str, Any]
    faults: list[dict[str, Any]] = field(default_factory=list)
    ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "soak": {
                "duration_seconds": self.metrics.get("duration_seconds"),
                "ticks": self.metrics.get("ticks"),
                "tick_seconds": self.metrics.get("tick_seconds"),
                "seed": self.metrics.get("seed"),
            },
            "metrics": dict(self.metrics),
            "faults": [dict(row) for row in self.faults],
            "result": {
                "ok": self.ok,
                "policy_violations": self.metrics.get("policy_violations"),
                "false_success": self.metrics.get("false_success"),
                "audit_chain_verified": self.metrics.get("audit_chain_verified"),
            },
        }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = fraction * (len(ordered) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def _ram_mib() -> float:
    """Peak resident set size in MiB (``ru_maxrss`` is KiB on Linux, bytes on macOS)."""
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def _thermal_level(temperature: float, config: Config) -> str:
    if temperature >= config.cooldown_c:
        return "cooldown"
    if temperature >= config.degraded_c:
        return "degraded"
    if temperature >= config.conserve_c:
        return "conserve"
    return "normal"


def _mock_reasoner_call() -> Callable[[], Any]:
    def call() -> dict[str, Any]:
        return {"model": "mock-reasoner", "tokens": 16}

    return call


def _mock_slow_brain() -> CallableSlowBrain:
    def respond(payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("kind") == "reflect":
            return {"summary": "synthetic reflection"}
        return {"summary": "synthetic advice", "focus": ["continuity", "thermal margin"]}

    return CallableSlowBrain(respond)


def _tripwire(tool: str, op: str, violations: list[str]) -> Callable[[dict[str, Any]], Any]:
    def effect(args: dict[str, Any]) -> dict[str, Any]:
        violations.append(f"{tool}.{op}")
        return {"unexpected": True}

    return effect


def _build_policy() -> Any:
    envelope = sign_envelope(dict(_BASE_BUNDLE), secret=DEV_SECRET)
    return load_policy_bytes(
        json.dumps(envelope).encode("utf-8"), secret=DEV_SECRET, allow_dev_hmac=True
    )


def run_soak(
    duration_seconds: float,
    *,
    data_dir: str | Path | None = None,
    seed: int = 17,
    tick_seconds: float = 1.0,
    max_ticks: int | None = None,
) -> SoakResult:
    """Run one deterministic soak of ``duration_seconds`` simulated time.

    Returns a :class:`SoakResult` whose ``ok`` is true only when there were zero
    policy violations and the audit chain verified.
    """
    duration_seconds = float(duration_seconds)
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be a positive, finite number")
    tick_seconds = float(tick_seconds)
    if not math.isfinite(tick_seconds) or tick_seconds <= 0:
        raise ValueError("tick_seconds must be a positive, finite number")
    ticks = max(1, int(round(duration_seconds / tick_seconds)))
    if max_ticks is not None:
        ticks = min(ticks, max(1, int(max_ticks)))

    rng = random.Random(int(seed))
    scratch: tempfile.TemporaryDirectory[str] | None = None
    if data_dir is None:
        scratch = tempfile.TemporaryDirectory(prefix="rlr-simsoak-")
        root = Path(scratch.name)
    else:
        root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)

    config = Config(data_dir=root, tick_seconds=tick_seconds)
    policy = _build_policy()
    clock = _Clock()
    bus = EventBus()
    audit = AuditLog(str(root / "audit.db"), key=AUDIT_KEY)
    plan = PlanGraph(str(root / "plan.db"), audit=audit)
    faults = FaultController(plan=plan, bus=bus, clock=clock)
    reasoner = AsyncReasoner(bus=bus, clock=clock, default_expiry_seconds=5.0)
    budget = BudgetService(root / "budget.db", policy)
    service = PermitService(
        PermitIssuer(ISSUER_KEY),
        policy,
        manifest=broker_manifest(),
        budget=budget,
        db_path=root / "permits.db",
    )
    recording = _RecordingPermits(service)
    goals = GoalManager(policy, audit=audit)
    broker = ToolBroker(
        service, audit=audit, workspace=root / "workspace", manifest=broker_manifest()
    )
    executive = Executive(
        plan, recording, broker, goals=goals, budget=budget, audit=audit, bus=bus, faults=faults
    )
    heartbeat = Heartbeat(bus, clock=clock, interval_seconds=2.0)
    decider = DeterministicOpenDecider()
    slow_brain = _mock_slow_brain()

    # Mock A0/A1 adapters execute; A2/A3 adapters are tripwires that must never
    # fire (the broker simulates A2/A3 instead of performing them).
    violations: list[str] = []
    broker.register_effect("goal", "complete", _tripwire("goal", "complete", violations))
    broker.register_effect("notify", "send", _tripwire("notify", "send", violations))
    broker.register_effect("memory", "forget", _tripwire("memory", "forget", violations))
    broker.register_effect("telemetry", "sample", lambda args: {"sampled": True})
    broker.register_effect("state", "set_load_target", lambda args: {"load_target": args["load"]})
    broker.register_effect(
        "state", "set_reply_tokens", lambda args: {"reply_tokens": args["tokens"]}
    )
    broker.register_effect("state", "set_vision_fps", lambda args: {"vision_fps": args["fps"]})
    broker.register_effect("memory", "remember", lambda args: {"memory_id": 1})
    broker.register_effect("memory", "recall", lambda args: {"matches": 0})
    broker.register_effect("goal", "set", lambda args: {"goal_id": 1})
    broker.register_effect(
        "control",
        "cooldown",
        lambda args: {"cooldown_until_tick": int(args["tick"]) + int(args["seconds"])},
    )
    broker.register_effect("control", "reflect", lambda args: {})

    goals_completed = 0
    false_success = 0
    simulated_a2_a3 = 0
    executed_a0_a1 = 0
    shell_attempts = 0
    shell_refused = 0
    used_advice = 0
    fault_rows: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    active_fault: str | None = None
    active_fault_at = 0
    pending_goal: str | None = None
    goal_seq = 0

    try:
        for tick in range(1, ticks + 1):
            started = time.perf_counter()
            clock.advance(tick_seconds)
            heartbeat.beat()

            # -- random fault injection / recovery -------------------------
            if active_fault is None and rng.random() < _FAULT_INJECT_PROBABILITY:
                name = rng.choice(FAULT_NAMES)
                response = faults.report(name, reason="simulated fault injection")
                active_fault = name
                active_fault_at = tick
                fault_rows.append({"tick": tick, "fault": name, "safe_state": response.safe_state})
                bus.emit(POLICY_CHANGE, mode="constrained", reason=name)
            elif active_fault is not None and tick - active_fault_at >= _FAULT_HOLD_TICKS:
                faults.clear(active_fault)
                active_fault = None

            # -- synthetic state + async mock model ------------------------
            temperature = _THERMALS[tick % len(_THERMALS)]
            level = _thermal_level(temperature, config)
            state = {
                "thermal_level": level,
                "temperature_c": temperature,
                "ram_used_mib": 2048.0 + (tick % 10) * 32,
                "ram_total_mib": 8192.0,
                "capabilities": ["vision", "generation", "load_control"],
                "active_goals": [{"id": 1, "text": "soak"}],
                "episode_count": tick,
                "memory_count": tick,
                "cooldown_active": False,
            }
            bus.emit(STATE_CHANGE, tick=tick, thermal_level=level)

            state_version = f"state-{tick}"
            call_id = reasoner.submit(
                "mock-decider", _mock_reasoner_call(), state_version=state_version
            )
            reasoner.join(call_id, timeout=1.0)
            reasoner.poll(current_state_version=state_version)
            if tick % 3 == 0:
                stale_id = reasoner.submit(
                    "mock-decider", _mock_reasoner_call(), state_version="superseded"
                )
                reasoner.join(stale_id, timeout=1.0)
                reasoner.poll(current_state_version=state_version)
            bus.emit(DEADLINE, tick=tick, state_version=state_version)

            candidates = build_candidates(state, config=config)
            scored = decider.score(candidates, state)
            if scored.confidence < config.confidence_threshold:
                slow_brain.advise(state=state, candidates=candidates, previous=scored)
                used_advice += 1

            # -- choose a tool op and drive it through the kernel ----------
            # A fixed rotation guarantees every risk class is exercised even
            # when the mock cortex would keep selecting a safe observe/abstain.
            if tick % 13 == 0:
                tool, op, args = "memory", "forget", {"memory_id": 1}
            elif tick % 11 == 0:
                tool, op, args = "shell", "run", {"command": "true"}
            elif tick % 9 == 0:
                tool, op, args = "notify", "send", {"channel": "local_log", "message": "soak"}
            elif tick % 7 == 0:
                tool, op, args = "shell", "read", {"command": "echo soak"}
            elif tick % 5 == 0:
                tool, op, args = "goal", "complete", {"goal_id": 7}
            else:
                action = scored.choice.split(":")[0]
                tool, op, args = _TOOL_PLAN.get(action, ("telemetry", "sample", {}))
                args = dict(args)
                if op == "cooldown":
                    args["tick"] = tick
            try:
                risk = broker.manifest.risk_class(tool, op)
            except Exception:  # noqa: BLE001 - an unknown op is a mock-planning bug
                risk = "?"

            step = plan.add_step(
                goal_id=goals.root_id or "root",
                tool=tool,
                op=op,
                args=args,
                step_id=f"soak-{tick}",
            )
            execution = executive.run_step(step.step_id)
            result = dict(execution.result or {})

            if execution.status == "DONE":
                if result.get("ok") is False:
                    false_success += 1
                if risk in ("A2", "A3"):
                    if result.get("executed") is True:
                        false_success += 1
                    else:
                        simulated_a2_a3 += 1
                elif risk in ("A0", "A1"):
                    if result.get("executed") is True:
                        executed_a0_a1 += 1
                    else:
                        false_success += 1
            elif risk in ("A2", "A3") and result.get("executed") is True:
                false_success += 1

            if tool == "shell":
                shell_attempts += 1
                if execution.status != "DONE":
                    shell_refused += 1

            if tick % 25 == 0:
                bus.emit(BUDGET_THRESHOLD, tick=tick, reason="simulated threshold")

            # -- goal lifecycle -------------------------------------------
            if pending_goal is None and len(goals) < _MAX_SOAK_GOALS:
                try:
                    goal = goals.propose(f"soak goal {goal_seq}", parent_id=goals.root_id)
                    goals.commit(goal.goal_id)
                    pending_goal = goal.goal_id
                    goal_seq += 1
                except (GoalRejected, GoalConflict):
                    pending_goal = None
            if pending_goal is not None and (tick % 5 == 0 or tick == ticks):
                goals.achieve(pending_goal, reason="soak progress")
                goals_completed += 1
                pending_goal = None

            latencies_ms.append((time.perf_counter() - started) * 1000.0)
    finally:
        chain = audit.verify_chain()
        issued = len(recording.issued)
        consumed = sum(1 for permit in recording.issued if service.consumed(permit.nonce))
        plan.close()
        service.close()
        audit.close()
        if scratch is not None:
            scratch.cleanup()

    policy_violations = len(violations)
    metrics: dict[str, Any] = {
        "duration_seconds": round(duration_seconds, 3),
        "ticks": ticks,
        "tick_seconds": tick_seconds,
        "seed": int(seed),
        "goals_completed": goals_completed,
        "false_success": false_success,
        "policy_violations": policy_violations,
        "permits_issued": issued,
        "permits_consumed": consumed,
        "simulated_a2_a3": simulated_a2_a3,
        "executed_a0_a1": executed_a0_a1,
        "shell_attempts": shell_attempts,
        "shell_refused": shell_refused,
        "shell_sandboxed": broker.bwrap_available(),
        "latency_ms_p50": round(_percentile(latencies_ms, 0.50), 4),
        "latency_ms_p95": round(_percentile(latencies_ms, 0.95), 4),
        "ram_mib": round(_ram_mib(), 3),
        "reasoner_accepted": reasoner.accepted,
        "reasoner_dropped": reasoner.dropped,
        "events_emitted": bus.sequence,
        "faults_injected": len(fault_rows),
        "used_advice": used_advice,
        "audit_chain_verified": chain.ok,
        "audit_chain_reason": chain.reason,
    }
    ok = policy_violations == 0 and chain.ok and false_success == 0
    return SoakResult(metrics=metrics, faults=fault_rows, ok=ok)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlr-simsoak", description="Faults-injected soak of the RLR kernel"
    )
    parser.add_argument("--hours", type=float, default=0.0, help="simulated soak hours")
    parser.add_argument("--minutes", type=float, default=0.0, help="simulated soak minutes")
    parser.add_argument("--seconds", type=float, default=0.0, help="simulated soak seconds")
    parser.add_argument(
        "--tick-seconds", type=float, default=1.0, help="simulated seconds per tick"
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-ticks", type=int, default=0)
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of TOON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    duration = float(args.hours) * 3600.0 + float(args.minutes) * 60.0 + float(args.seconds)
    if duration <= 0:
        parser.error("one of --hours, --minutes or --seconds must be positive")
    result = run_soak(
        duration,
        data_dir=args.data_dir or None,
        seed=args.seed,
        tick_seconds=args.tick_seconds,
        max_ticks=(args.max_ticks or None),
    )
    document = result.to_dict()
    if args.json:
        print(json.dumps(document, indent=2, sort_keys=True))
    else:
        print(toon_encode(document))
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())

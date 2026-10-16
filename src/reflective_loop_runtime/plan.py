"""Persistent plan graph and the permit-gated Executive (L2).

The plan layer turns committed goals into an explicit, durable graph of tool
steps and then executes them through the kernel's single tool path.

PlanGraph
---------
* Steps are stored in SQLite, so a process crash leaves the exact graph behind.
* Ordering is by dependency: :meth:`PlanGraph.ready_steps` returns only the
  pending steps whose prerequisites have completed, and
  :meth:`PlanGraph.topological_order` is a deterministic total order.
* A crash can leave a step ``RUNNING`` with no completion record. Recovery maps
  every such step to ``UNCERTAIN`` -- never back to ``PENDING`` -- so a step
  whose effect may or may not have happened is never silently retried.
* When a step's observed effect disagrees with its declared expectation,
  :meth:`PlanGraph.pause_branch` suspends the rest of that branch and records a
  :class:`ReplanRequest`.

Executive
---------
The Executive owns the only path from a plan step to a tool effect. For each
step it obtains a *fresh* :class:`~.kernel.permit.ExecutionPermit` (so the budget
is reserved and the call is bound to the tool, arguments, goal, plan and step),
then invokes the tool **only** through
:class:`~.kernel.toolbroker.ToolBroker`. A step that cannot obtain a permit
never reaches the broker. Every transition is appended to the kernel's
hash-chained :class:`~.kernel.audit.AuditLog`.

The module is stdlib-only and deterministic; it opens no sockets and spawns no
processes of its own.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .events import (
    BUDGET_THRESHOLD,
    DEADLINE,
    KILL,
    STATE_CHANGE,
    TOOL_COMPLETION,
    TOOL_FAILURE,
    EventBus,
    KillSwitch,
    LaunchBlocked,
)
from .kernel.audit import AuditLog, utc_now
from .kernel.budget import BudgetError, BudgetService
from .kernel.permit import PermitDenied, PermitService
from .kernel.policy import canonical_json
from .kernel.toolbroker import ToolBroker, ToolExecutionError, ToolNotPermitted

__all__ = [
    "PLAN_ACTIVE",
    "PLAN_DONE",
    "PLAN_PAUSED",
    "STEP_DONE",
    "STEP_FAILED",
    "STEP_PAUSED",
    "STEP_PENDING",
    "STEP_READY",
    "STEP_RUNNING",
    "STEP_SKIPPED",
    "STEP_UNCERTAIN",
    "EffectDiscrepancy",
    "Executive",
    "ExecutiveError",
    "DependencyNotSatisfied",
    "PlanError",
    "PlanGraph",
    "PlanStep",
    "ReplanRequest",
    "StepExecution",
    "StepNotRunnable",
]

# Step lifecycle.
STEP_PENDING = "PENDING"
STEP_READY = "READY"
STEP_RUNNING = "RUNNING"
STEP_DONE = "DONE"
STEP_FAILED = "FAILED"
STEP_UNCERTAIN = "UNCERTAIN"
STEP_PAUSED = "PAUSED"
STEP_SKIPPED = "SKIPPED"

# Derived plan lifecycle.
PLAN_ACTIVE = "ACTIVE"
PLAN_PAUSED = "PAUSED"
PLAN_DONE = "DONE"

_TERMINAL_STEPS = frozenset({STEP_DONE, STEP_FAILED, STEP_SKIPPED, STEP_UNCERTAIN})
_PAUSABLE_STEPS = frozenset({STEP_PENDING, STEP_READY, STEP_RUNNING})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS plan_steps (
    step_id          TEXT PRIMARY KEY,
    plan_id          TEXT NOT NULL,
    goal_id          TEXT NOT NULL,
    seq              INTEGER NOT NULL,
    tool             TEXT NOT NULL,
    op               TEXT NOT NULL,
    args             TEXT NOT NULL,
    depends_on       TEXT NOT NULL,
    expected_effects TEXT NOT NULL,
    status           TEXT NOT NULL,
    attempt          INTEGER NOT NULL DEFAULT 0,
    permit_id        TEXT NOT NULL DEFAULT '',
    result           TEXT NOT NULL DEFAULT '{}',
    reason           TEXT NOT NULL DEFAULT '',
    version          INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_plan_steps_plan ON plan_steps (plan_id, seq);
CREATE TABLE IF NOT EXISTS replan_requests (
    replan_id  TEXT PRIMARY KEY,
    plan_id    TEXT NOT NULL,
    goal_id    TEXT NOT NULL,
    step_id    TEXT NOT NULL,
    reason     TEXT NOT NULL,
    expected   TEXT NOT NULL,
    actual     TEXT NOT NULL,
    resolved   INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
"""


class PlanError(ValueError):
    """Base class for every plan-layer failure."""


class DependencyNotSatisfied(PlanError):
    """Raised when a step is run before its prerequisites are complete."""


class StepNotRunnable(PlanError):
    """Raised when a step cannot run in its current state (e.g. UNCERTAIN)."""


class ExecutiveError(RuntimeError):
    """Base class for Executive failures."""


def _json(value: Any, name: str) -> str:
    try:
        return canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise PlanError(f"{name} is not canonically-JSON encodable: {exc}") from exc


def _load(text: str, default: Any) -> Any:
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class PlanStep:
    """One durable, dependency-ordered tool step."""

    step_id: str
    plan_id: str
    goal_id: str
    seq: int
    tool: str
    op: str
    args: Mapping[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    expected_effects: Mapping[str, Any] = field(default_factory=dict)
    status: str = STEP_PENDING
    attempt: int = 0
    permit_id: str = ""
    result: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""
    version: int = 0
    created_at: str = ""
    updated_at: str = ""

    @property
    def terminal(self) -> bool:
        return self.status in _TERMINAL_STEPS

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "plan_id": self.plan_id,
            "goal_id": self.goal_id,
            "seq": self.seq,
            "tool": self.tool,
            "op": self.op,
            "args": dict(self.args),
            "depends_on": list(self.depends_on),
            "expected_effects": dict(self.expected_effects),
            "status": self.status,
            "attempt": self.attempt,
            "permit_id": self.permit_id,
            "result": dict(self.result),
            "reason": self.reason,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class ReplanRequest:
    """A branch-level request to re-plan, raised when reality diverges."""

    replan_id: str
    plan_id: str
    goal_id: str
    step_id: str
    reason: str
    expected: Mapping[str, Any] = field(default_factory=dict)
    actual: Mapping[str, Any] = field(default_factory=dict)
    resolved: bool = False
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "replan_id": self.replan_id,
            "plan_id": self.plan_id,
            "goal_id": self.goal_id,
            "step_id": self.step_id,
            "reason": self.reason,
            "expected": dict(self.expected),
            "actual": dict(self.actual),
            "resolved": self.resolved,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class EffectDiscrepancy:
    """The concrete divergence between an expected and an observed effect."""

    expected: Mapping[str, Any]
    actual: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"expected": dict(self.expected), "actual": dict(self.actual)}


class PlanGraph:
    """A durable, dependency-ordered graph of tool steps.

    ``path`` may be a file (persisted across runs) or ``":memory:"`` for an
    ephemeral graph. On construction every ``RUNNING`` step left by a crash is
    recovered to ``UNCERTAIN``.
    """

    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        audit: AuditLog | None = None,
        clock: Any = utc_now,
    ):
        target = str(path)
        if target != ":memory:":
            Path(target).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = target
        self._audit = (
            audit if audit is not None else AuditLog(":memory:", key=secrets.token_bytes(32))
        )
        self._clock = clock
        self._db = sqlite3.connect(target, isolation_level=None, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)
        self.recovered = self.recover()

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> PlanGraph:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def audit(self) -> AuditLog:
        return self._audit

    # -- writes ------------------------------------------------------------
    def add_step(
        self,
        *,
        goal_id: str,
        tool: str,
        op: str,
        args: Mapping[str, Any] | None = None,
        depends_on: Any = (),
        expected_effects: Mapping[str, Any] | None = None,
        step_id: str | None = None,
        plan_id: str | None = None,
    ) -> PlanStep:
        """Append one step to a plan and return it in ``PENDING`` state."""
        if not isinstance(goal_id, str) or not goal_id:
            raise PlanError("goal_id is required")
        if not isinstance(tool, str) or not tool:
            raise PlanError("tool is required")
        if not isinstance(op, str) or not op:
            raise PlanError("op is required")
        if args is None:
            args = {}
        if not isinstance(args, Mapping):
            raise PlanError("args must be a mapping")
        if expected_effects is None:
            expected_effects = {}
        if not isinstance(expected_effects, Mapping):
            raise PlanError("expected_effects must be a mapping")

        deps = self._normalise_deps(depends_on)
        resolved_plan = plan_id or f"plan:{goal_id}"
        seq = self._next_seq(resolved_plan)
        resolved_id = step_id or f"{resolved_plan}:s{seq}"
        if resolved_id in deps:
            raise PlanError(f"step {resolved_id!r} cannot depend on itself")
        if self.get(resolved_id) is not None:
            raise PlanError(f"step id {resolved_id!r} already exists")

        now = self._clock()
        self._db.execute(
            "INSERT INTO plan_steps (step_id, plan_id, goal_id, seq, tool, op, args,"
            " depends_on, expected_effects, status, attempt, permit_id, result,"
            " reason, version, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, '', '{}', '', 0, ?, ?)",
            (
                resolved_id,
                resolved_plan,
                goal_id,
                seq,
                tool,
                op,
                _json(dict(args), "args"),
                _json(list(deps), "depends_on"),
                _json(dict(expected_effects), "expected_effects"),
                STEP_PENDING,
                now,
                now,
            ),
        )
        self._emit("plan.step_added", self._require(resolved_id))
        return self._require(resolved_id)

    def mark_running(self, step_id: str, *, permit_id: str = "") -> PlanStep:
        """Move a pending/ready step to ``RUNNING``. Never retries a finished one."""
        step = self._require(step_id)
        if step.status not in (STEP_PENDING, STEP_READY):
            raise StepNotRunnable(
                f"step {step_id!r} is {step.status} and cannot start; "
                "UNCERTAIN/FAILED steps require an explicit reset"
            )
        return self._update(
            step,
            status=STEP_RUNNING,
            permit_id=str(permit_id),
            attempt=step.attempt + 1,
            reason="",
        )

    def mark_done(self, step_id: str, *, result: Mapping[str, Any] | None = None) -> PlanStep:
        step = self._require(step_id)
        return self._update(step, status=STEP_DONE, result=dict(result or {}), reason="")

    def mark_failed(
        self, step_id: str, *, reason: str = "", result: Mapping[str, Any] | None = None
    ) -> PlanStep:
        step = self._require(step_id)
        return self._update(step, status=STEP_FAILED, result=dict(result or {}), reason=str(reason))

    def mark_uncertain(
        self, step_id: str, *, reason: str = "", result: Mapping[str, Any] | None = None
    ) -> PlanStep:
        step = self._require(step_id)
        return self._update(
            step, status=STEP_UNCERTAIN, result=dict(result or {}), reason=str(reason)
        )

    def mark_skipped(self, step_id: str, *, reason: str = "") -> PlanStep:
        step = self._require(step_id)
        return self._update(step, status=STEP_SKIPPED, reason=str(reason))

    def reset_step(self, step_id: str, *, reason: str = "explicit reset") -> PlanStep:
        """Explicitly return a non-terminal, failed or uncertain step to ``PENDING``.

        This is the *only* way a ``UNCERTAIN`` or ``FAILED`` step may run again,
        so a crash-recovered step is never retried implicitly.
        """
        step = self._require(step_id)
        if step.status == STEP_DONE:
            raise StepNotRunnable("a DONE step is already complete")
        return self._update(step, status=STEP_PENDING, reason=str(reason))

    # -- branch control ----------------------------------------------------
    def pause_branch(
        self,
        goal_id: str,
        *,
        reason: str,
        expected: Mapping[str, Any] | None = None,
        actual: Mapping[str, Any] | None = None,
        step_id: str = "",
        plan_id: str | None = None,
    ) -> ReplanRequest:
        """Pause every schedulable step of a branch and request a re-plan."""
        if not isinstance(goal_id, str) or not goal_id:
            raise PlanError("goal_id is required")
        branch = [s for s in self.steps(goal_id=goal_id)]
        if plan_id is None:
            plan_id = branch[0].plan_id if branch else f"plan:{goal_id}"
        for step in branch:
            if step.status in _PAUSABLE_STEPS:
                self._update(step, status=STEP_PAUSED, reason=str(reason))
        request = ReplanRequest(
            replan_id=f"replan:{plan_id}:{len(self.replan_requests(resolved=None)) + 1}",
            plan_id=plan_id,
            goal_id=goal_id,
            step_id=step_id,
            reason=str(reason),
            expected=dict(expected or {}),
            actual=dict(actual or {}),
            resolved=False,
            created_at=self._clock(),
        )
        self._db.execute(
            "INSERT INTO replan_requests (replan_id, plan_id, goal_id, step_id, reason,"
            " expected, actual, resolved, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)",
            (
                request.replan_id,
                request.plan_id,
                request.goal_id,
                request.step_id,
                request.reason,
                _json(dict(request.expected), "expected"),
                _json(dict(request.actual), "actual"),
                request.created_at,
            ),
        )
        self._audit.append(
            goal_version=goal_id,
            plan_version=plan_id,
            governor_result={
                "layer": "L2",
                "outcome": "plan.replan_requested",
                "goal_id": goal_id,
                "step_id": step_id,
                "replan_id": request.replan_id,
                "reason": request.reason,
                "expected": dict(request.expected),
                "actual": dict(request.actual),
            },
            outcome="plan.replan_requested",
        )
        return request

    def resume_branch(self, goal_id: str, *, reason: str = "resumed") -> int:
        """Return every paused step of a branch to ``PENDING``; return the count."""
        count = 0
        for step in self.steps(goal_id=goal_id):
            if step.status == STEP_PAUSED:
                self._update(step, status=STEP_PENDING, reason=str(reason))
                count += 1
        return count

    def is_paused(self, goal_id: str) -> bool:
        """Whether a branch is halted: it has a paused step or an open re-plan.

        A discrepancy on the *last* schedulable step pauses nothing concrete,
        so the unresolved :class:`ReplanRequest` is what marks the branch as
        halted until a caller re-plans it.
        """
        if any(s.status == STEP_PAUSED for s in self.steps(goal_id=goal_id)):
            return True
        return any(
            request.goal_id == str(goal_id) and not request.resolved
            for request in self.replan_requests(resolved=False)
        )

    def paused_goals(self) -> list[str]:
        """Every branch currently halted by pause or by an open re-plan."""
        goals = {s.goal_id for s in self.steps() if s.status == STEP_PAUSED}
        goals.update(r.goal_id for r in self.replan_requests(resolved=False))
        return sorted(goals)

    # -- recovery ----------------------------------------------------------
    def recover(self) -> list[str]:
        """Map every crash-orphaned ``RUNNING`` step to ``UNCERTAIN``.

        Called automatically on construction. A step whose effect may have
        happened is quarantined rather than re-run; only an explicit
        :meth:`reset_step` can make it schedulable again.
        """
        rows = self._db.execute(
            "SELECT step_id FROM plan_steps WHERE status=? ORDER BY plan_id, seq",
            (STEP_RUNNING,),
        ).fetchall()
        recovered: list[str] = []
        for row in rows:
            step = self._require(str(row["step_id"]))
            self._update(
                step,
                status=STEP_UNCERTAIN,
                reason="crash recovery: step was RUNNING with no completion record",
            )
            recovered.append(step.step_id)
        return recovered

    # -- reads -------------------------------------------------------------
    def get(self, step_id: str) -> PlanStep | None:
        row = self._db.execute(
            "SELECT * FROM plan_steps WHERE step_id=?", (str(step_id),)
        ).fetchone()
        return _row_to_step(row) if row is not None else None

    def steps(self, *, plan_id: str | None = None, goal_id: str | None = None) -> list[PlanStep]:
        clauses: list[str] = []
        params: list[Any] = []
        if plan_id is not None:
            clauses.append("plan_id=?")
            params.append(str(plan_id))
        if goal_id is not None:
            clauses.append("goal_id=?")
            params.append(str(goal_id))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._db.execute(
            f"SELECT * FROM plan_steps{where} ORDER BY plan_id, seq", params
        ).fetchall()
        return [_row_to_step(row) for row in rows]

    def plan_ids(self) -> list[str]:
        rows = self._db.execute(
            "SELECT DISTINCT plan_id FROM plan_steps ORDER BY plan_id"
        ).fetchall()
        return [str(row["plan_id"]) for row in rows]

    def topological_order(self, *, plan_id: str | None = None) -> list[str]:
        """A deterministic dependency order for a plan.

        Raises :class:`PlanError` when a dependency is unknown or the graph
        contains a cycle.
        """
        steps = self.steps(plan_id=plan_id)
        by_id = {step.step_id: step for step in steps}
        for step in steps:
            for dep in step.depends_on:
                if dep not in by_id:
                    raise PlanError(f"step {step.step_id!r} depends on unknown step {dep!r}")
        ordered_ids = [step.step_id for step in sorted(steps, key=lambda s: (s.seq, s.step_id))]
        resolved: set[str] = set()
        remaining = set(ordered_ids)
        order: list[str] = []
        while remaining:
            ready = [
                sid
                for sid in ordered_ids
                if sid in remaining and set(by_id[sid].depends_on) <= resolved
            ]
            if not ready:
                raise PlanError("plan graph contains a dependency cycle")
            for sid in ready:
                remaining.discard(sid)
                resolved.add(sid)
                order.append(sid)
        return order

    def ready_steps(
        self, *, plan_id: str | None = None, goal_id: str | None = None
    ) -> list[PlanStep]:
        """Pending steps whose dependencies have all completed, in plan order."""
        steps = self.steps(plan_id=plan_id, goal_id=goal_id)
        by_id = {step.step_id: step for step in self.steps(plan_id=plan_id)}
        ready: list[PlanStep] = []
        for step in steps:
            if step.status != STEP_PENDING:
                continue
            if all(
                dep in by_id and by_id[dep].status in (STEP_DONE, STEP_SKIPPED)
                for dep in step.depends_on
            ):
                ready.append(step)
        return ready

    def plan_status(self, plan_id: str) -> str:
        steps = self.steps(plan_id=plan_id)
        if not steps:
            return PLAN_DONE
        if any(s.status == STEP_PAUSED for s in steps):
            return PLAN_PAUSED
        schedulable = [s for s in steps if s.status in (STEP_PENDING, STEP_READY, STEP_RUNNING)]
        if schedulable:
            return PLAN_ACTIVE
        return PLAN_DONE

    # -- replan requests ---------------------------------------------------
    def replan_requests(self, *, resolved: bool | None = False) -> list[ReplanRequest]:
        clause = ""
        params: list[Any] = []
        if resolved is not None:
            clause = " WHERE resolved=?"
            params.append(1 if resolved else 0)
        rows = self._db.execute(
            f"SELECT * FROM replan_requests{clause} ORDER BY created_at, replan_id", params
        ).fetchall()
        return [_row_to_replan(row) for row in rows]

    def resolve_replan(self, replan_id: str) -> ReplanRequest:
        row = self._db.execute(
            "SELECT * FROM replan_requests WHERE replan_id=?", (str(replan_id),)
        ).fetchone()
        if row is None:
            raise PlanError(f"unknown replan request {replan_id!r}")
        self._db.execute(
            "UPDATE replan_requests SET resolved=1 WHERE replan_id=?", (str(replan_id),)
        )
        self._emit_replan(
            "plan.replan_resolved", str(row["goal_id"]), str(row["plan_id"]), str(replan_id)
        )
        return _row_to_replan(
            self._db.execute(
                "SELECT * FROM replan_requests WHERE replan_id=?", (str(replan_id),)
            ).fetchone()
        )

    # -- snapshot ----------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        return {
            "plans": {
                plan_id: [step.to_dict() for step in self.steps(plan_id=plan_id)]
                for plan_id in self.plan_ids()
            },
            "replan_requests": [r.to_dict() for r in self.replan_requests(resolved=None)],
        }

    # -- internals ---------------------------------------------------------
    def _normalise_deps(self, depends_on: Any) -> tuple[str, ...]:
        if depends_on is None:
            return ()
        if isinstance(depends_on, str):
            raise PlanError("depends_on must be a sequence of step ids, not a string")
        try:
            items = [str(dep) for dep in depends_on]
        except TypeError as exc:
            raise PlanError("depends_on must be an iterable of step ids") from exc
        seen: list[str] = []
        for dep in items:
            if not dep:
                raise PlanError("dependency ids must be non-empty")
            if dep in seen:
                raise PlanError(f"duplicate dependency {dep!r}")
            seen.append(dep)
        return tuple(seen)

    def _next_seq(self, plan_id: str) -> int:
        row = self._db.execute(
            "SELECT COALESCE(MAX(seq), -1) + 1 AS next FROM plan_steps WHERE plan_id=?",
            (plan_id,),
        ).fetchone()
        return int(row["next"])

    def _require(self, step_id: str) -> PlanStep:
        step = self.get(step_id)
        if step is None:
            raise PlanError(f"unknown step {step_id!r}")
        return step

    def _update(self, step: PlanStep, *, status: str, **changes: Any) -> PlanStep:
        fields = {
            "args": dict(step.args),
            "depends_on": list(step.depends_on),
            "expected_effects": dict(step.expected_effects),
            "status": status,
            "attempt": step.attempt,
            "permit_id": step.permit_id,
            "result": dict(step.result),
            "reason": step.reason,
            "version": step.version + 1,
            "updated_at": self._clock(),
        }
        fields.update(changes)
        self._db.execute(
            "UPDATE plan_steps SET status=?, attempt=?, permit_id=?, result=?, reason=?,"
            " version=?, updated_at=? WHERE step_id=?",
            (
                fields["status"],
                fields["attempt"],
                fields["permit_id"],
                _json(dict(fields["result"]), "result"),
                fields["reason"],
                fields["version"],
                fields["updated_at"],
                step.step_id,
            ),
        )
        updated = self._require(step.step_id)
        self._emit(f"plan.{status.lower()}", updated)
        return updated

    def _emit(self, outcome: str, step: PlanStep) -> None:
        self._audit.append(
            goal_version=step.goal_id,
            plan_version=step.plan_id,
            permit_id=step.permit_id,
            governor_result={
                "layer": "L2",
                "outcome": outcome,
                "step_id": step.step_id,
                "status": step.status,
                "tool": step.tool,
                "op": step.op,
                "attempt": step.attempt,
                "reason": step.reason,
            },
            outcome=outcome,
        )

    def _emit_replan(self, outcome: str, goal_id: str, plan_id: str, replan_id: str) -> None:
        self._audit.append(
            goal_version=goal_id,
            plan_version=plan_id,
            governor_result={"layer": "L2", "outcome": outcome, "replan_id": replan_id},
            outcome=outcome,
        )


def _row_to_step(row: sqlite3.Row) -> PlanStep:
    return PlanStep(
        step_id=str(row["step_id"]),
        plan_id=str(row["plan_id"]),
        goal_id=str(row["goal_id"]),
        seq=int(row["seq"]),
        tool=str(row["tool"]),
        op=str(row["op"]),
        args=_load(str(row["args"]), {}),
        depends_on=tuple(_load(str(row["depends_on"]), [])),
        expected_effects=_load(str(row["expected_effects"]), {}),
        status=str(row["status"]),
        attempt=int(row["attempt"]),
        permit_id=str(row["permit_id"]),
        result=_load(str(row["result"]), {}),
        reason=str(row["reason"]),
        version=int(row["version"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _row_to_replan(row: sqlite3.Row) -> ReplanRequest:
    return ReplanRequest(
        replan_id=str(row["replan_id"]),
        plan_id=str(row["plan_id"]),
        goal_id=str(row["goal_id"]),
        step_id=str(row["step_id"]),
        reason=str(row["reason"]),
        expected=_load(str(row["expected"]), {}),
        actual=_load(str(row["actual"]), {}),
        resolved=bool(row["resolved"]),
        created_at=str(row["created_at"]),
    )


@dataclass(frozen=True)
class StepExecution:
    """The Executive's report of one attempted step."""

    step_id: str
    status: str
    ok: bool
    permit_id: str = ""
    result: Mapping[str, Any] = field(default_factory=dict)
    replan: ReplanRequest | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "status": self.status,
            "ok": self.ok,
            "permit_id": self.permit_id,
            "result": dict(self.result),
            "replan": self.replan.to_dict() if self.replan is not None else None,
            "note": self.note,
        }


class Executive:
    """Runs plan steps through a fresh permit and the single ToolBroker path.

    Parameters
    ----------
    plan:
        The durable :class:`PlanGraph` the Executive acts on.
    permits:
        The kernel :class:`~.kernel.permit.PermitService`. One permit is minted
        per step; there is no cached or reused authority.
    broker:
        The :class:`~.kernel.toolbroker.ToolBroker`. Steps are *only* ever
        dispatched through :meth:`ToolBroker.invoke`.
    budget:
        Optional :class:`~.kernel.budget.BudgetService` used to finalise the
        reservation the permit took (commit on success, release otherwise).
    bus:
        Optional :class:`~.events.EventBus` for L3 lifecycle events. When
        supplied, the Executive publishes ``state_change``, ``tool_completion``,
        ``tool_failure``, ``deadline``, ``budget_threshold`` and ``kill`` events.
    kill_switch:
        Optional :class:`~.events.KillSwitch`. When present the kill file is
        checked before any permit is minted, every permit is registered, and a
        launch is refused if the latch is set.
    faults:
        Optional :class:`~.faults.FaultController`. When present the Executive
        consults the active safe state before minting a permit and again for the
        permit's risk class.
    """

    def __init__(
        self,
        plan: PlanGraph,
        permits: PermitService,
        broker: ToolBroker,
        *,
        goals: Any = None,
        budget: BudgetService | None = None,
        audit: AuditLog | None = None,
        clock: Any = utc_now,
        bus: EventBus | None = None,
        kill_switch: KillSwitch | None = None,
        faults: Any = None,
    ):
        if not isinstance(plan, PlanGraph):
            raise ExecutiveError("plan must be a PlanGraph")
        self.plan = plan
        self.permits = permits
        self.broker = broker
        self.goals = goals
        self.budget = budget
        self._audit = audit if audit is not None else plan.audit
        self._clock = clock
        self.bus = bus
        self.kill_switch = kill_switch
        self.faults = faults
        if kill_switch is not None and getattr(kill_switch, "_permits", None) is None:
            kill_switch.bind_permits(permits)

    # -- recovery ----------------------------------------------------------
    def recover(self) -> list[str]:
        """Delegate crash recovery: orphaned ``RUNNING`` steps become ``UNCERTAIN``."""
        return self.plan.recover()

    # -- execution ---------------------------------------------------------
    def run_step(
        self, step_id: str, *, actual_spend: Mapping[str, Any] | None = None
    ) -> StepExecution:
        """Run one step: fresh permit -> broker -> effect check -> audit.

        A step that is already ``UNCERTAIN`` (typically crash-recovered) or
        otherwise not schedulable is refused, so nothing is retried implicitly.
        """
        step = self.plan.get(step_id)
        if step is None:
            raise ExecutiveError(f"unknown step {step_id!r}")
        if step.status not in (STEP_PENDING, STEP_READY):
            raise StepNotRunnable(f"step {step_id!r} is {step.status} and will not run implicitly")
        # 0. The kill latch and the active safe state gate every launch.
        if self.kill_switch is not None:
            self.kill_switch.check()
            if self.kill_switch.engaged:
                self._emit(KILL, step_id=step_id, reason=self.kill_switch.reason)
                return self._refuse(step_id, note="kill switch engaged")
        if self.faults is not None and self.faults.observation_only:
            return self._refuse(step_id, note="fault guard: observation only")

        self._require_dependencies(step)

        # 1. A fresh permit is mandatory. No permit -> no broker call, ever.
        try:
            permit = self.permits.issue(
                tool=step.tool,
                op=step.op,
                args=dict(step.args),
                goal_id=step.goal_id,
                plan_id=step.plan_id,
                step_id=step.step_id,
                state_version=str(step.version),
            )
        except (PermitDenied, BudgetError) as exc:
            if isinstance(exc, BudgetError):
                self._emit(BUDGET_THRESHOLD, step_id=step_id, reason=str(exc))
            denied = self.plan.mark_failed(step_id, reason=f"permit denied: {exc}")
            self._audit.append(
                goal_version=step.goal_id,
                plan_version=step.plan_id,
                governor_result={
                    "layer": "L2",
                    "outcome": "step.permit_denied",
                    "step_id": step_id,
                    "reason": str(exc),
                },
                outcome="step.permit_denied",
            )
            self._emit(STATE_CHANGE, step_id=step_id, status=denied.status)
            return StepExecution(
                step_id=step_id,
                status=denied.status,
                ok=False,
                note=f"permit denied: {exc}",
            )
        if permit is None:
            self.plan.mark_failed(step_id, reason="no permit was issued")
            self._emit(STATE_CHANGE, step_id=step_id, status=STEP_FAILED)
            return StepExecution(
                step_id=step_id,
                status=STEP_FAILED,
                ok=False,
                note="no permit issued; refusing to run the step",
            )

        # The kill latch tracks every permit it may need to revoke; the active
        # fault matrix decides whether this risk class may still run.
        if self.kill_switch is not None:
            self.kill_switch.register_permit(permit.permit_id)
        if self.faults is not None and not self.faults.permits_allowed(permit.risk_class):
            self.plan.mark_failed(
                step_id, reason=f"fault guard: {permit.risk_class} permits withheld"
            )
            self._finalise_budget(permit, success=False, actual_spend=None)
            self._emit(TOOL_FAILURE, step_id=step_id, reason="fault guard withheld permit")
            self._emit(STATE_CHANGE, step_id=step_id, status=STEP_FAILED)
            return StepExecution(
                step_id=step_id,
                status=STEP_FAILED,
                ok=False,
                permit_id=permit.permit_id,
                note=f"fault guard withheld a {permit.risk_class} permit",
            )
        self._emit(
            DEADLINE,
            step_id=step_id,
            permit_id=permit.permit_id,
            expires_at=getattr(permit, "expires_at", None),
        )

        # 2. Mark running, then dispatch through the broker only.
        self.plan.mark_running(step_id, permit_id=permit.permit_id)
        self._emit(STATE_CHANGE, step_id=step_id, status=STEP_RUNNING)
        try:
            if self.kill_switch is not None:
                self.kill_switch.require_launch(permit.permit_id)
            result = self.broker.invoke(permit, step.tool, step.op, dict(step.args))
        except LaunchBlocked as exc:
            self.plan.mark_failed(step_id, reason=f"kill switch: {exc}")
            self._finalise_budget(permit, success=False, actual_spend=None)
            self._emit(KILL, step_id=step_id, reason=str(exc))
            self._emit(STATE_CHANGE, step_id=step_id, status=STEP_FAILED)
            return StepExecution(
                step_id=step_id,
                status=STEP_FAILED,
                ok=False,
                permit_id=permit.permit_id,
                note=f"kill switch: {exc}",
            )
        except (ToolNotPermitted, ToolExecutionError) as exc:
            self.plan.mark_failed(step_id, reason=f"tool refused: {exc}")
            self._finalise_budget(permit, success=False, actual_spend=None)
            self._audit.append(
                goal_version=step.goal_id,
                plan_version=step.plan_id,
                permit_id=permit.permit_id,
                governor_result={
                    "layer": "L2",
                    "outcome": "step.tool_refused",
                    "step_id": step_id,
                    "reason": str(exc),
                },
                outcome="step.tool_refused",
            )
            self._emit(TOOL_FAILURE, step_id=step_id, reason=str(exc))
            self._emit(STATE_CHANGE, step_id=step_id, status=STEP_FAILED)
            return StepExecution(
                step_id=step_id,
                status=STEP_FAILED,
                ok=False,
                permit_id=permit.permit_id,
                note=f"tool refused: {exc}",
            )

        result_dict = result.as_dict()

        # 3. A failed result is a failure; a wrong-but-successful result is a
        #    discrepancy that pauses the branch and asks for a re-plan.
        if not result.ok:
            self.plan.mark_failed(
                step_id, reason=result.note or "tool reported failure", result=result_dict
            )
            self._finalise_budget(permit, success=False, actual_spend=None)
            self._record_result(step, permit, result_dict, outcome="step.failed")
            self._emit(
                TOOL_FAILURE,
                step_id=step_id,
                permit_id=permit.permit_id,
                reason=result.note or "tool reported failure",
            )
            self._emit(STATE_CHANGE, step_id=step_id, status=STEP_FAILED)
            return StepExecution(
                step_id=step_id,
                status=STEP_FAILED,
                ok=False,
                permit_id=permit.permit_id,
                result=result_dict,
                note=result.note or "tool reported failure",
            )

        discrepancy = self._effect_discrepancy(step, result_dict)
        if discrepancy is not None:
            self.plan.mark_uncertain(
                step_id,
                reason="observed effect did not match expectation",
                result=result_dict,
            )
            request = self.plan.pause_branch(
                step.goal_id,
                reason=f"effect discrepancy at step {step_id}",
                expected=dict(discrepancy.expected),
                actual=dict(discrepancy.actual),
                step_id=step_id,
                plan_id=step.plan_id,
            )
            self._finalise_budget(permit, success=False, actual_spend=None)
            self._record_result(step, permit, result_dict, outcome="step.discrepancy")
            self._emit(
                TOOL_FAILURE,
                step_id=step_id,
                permit_id=permit.permit_id,
                reason="effect discrepancy",
            )
            self._emit(STATE_CHANGE, step_id=step_id, status=STEP_UNCERTAIN)
            return StepExecution(
                step_id=step_id,
                status=STEP_UNCERTAIN,
                ok=False,
                permit_id=permit.permit_id,
                result=result_dict,
                replan=request,
                note="effect discrepancy; branch paused and re-plan requested",
            )

        self.plan.mark_done(step_id, result=result_dict)
        self._finalise_budget(permit, success=True, actual_spend=actual_spend)
        self._record_result(step, permit, result_dict, outcome="step.done")
        self._emit(TOOL_COMPLETION, step_id=step_id, permit_id=permit.permit_id)
        self._emit(STATE_CHANGE, step_id=step_id, status=STEP_DONE)
        return StepExecution(
            step_id=step_id,
            status=STEP_DONE,
            ok=True,
            permit_id=permit.permit_id,
            result=result_dict,
            note="executed",
        )

    def run_ready(
        self, *, plan_id: str | None = None, goal_id: str | None = None, limit: int | None = None
    ) -> list[StepExecution]:
        """Run the currently-ready steps of a plan/branch, in order."""
        executions: list[StepExecution] = []
        for step in self.plan.ready_steps(plan_id=plan_id, goal_id=goal_id):
            executions.append(self.run_step(step.step_id))
            if limit is not None and len(executions) >= limit:
                break
        return executions

    # -- internals ---------------------------------------------------------
    def _emit(self, kind: str, **payload: Any) -> None:
        """Publish an L3 event when a bus is attached; otherwise a no-op."""
        if self.bus is not None:
            self.bus.emit(kind, **payload)

    def _refuse(self, step_id: str, *, note: str) -> StepExecution:
        """Fail a step closed without minting or launching anything."""
        self.plan.mark_failed(step_id, reason=note)
        self._emit(STATE_CHANGE, step_id=step_id, status=STEP_FAILED)
        return StepExecution(step_id=step_id, status=STEP_FAILED, ok=False, note=note)

    def _require_dependencies(self, step: PlanStep) -> None:
        pending: list[str] = []
        for dep in step.depends_on:
            other = self.plan.get(dep)
            if other is None:
                raise DependencyNotSatisfied(
                    f"step {step.step_id!r} depends on unknown step {dep!r}"
                )
            if other.status not in (STEP_DONE, STEP_SKIPPED):
                pending.append(f"{dep} ({other.status})")
        if pending:
            raise DependencyNotSatisfied(
                f"step {step.step_id!r} is not ready; unfinished dependencies: {', '.join(pending)}"
            )

    @staticmethod
    def _effect_discrepancy(step: PlanStep, result: Mapping[str, Any]) -> EffectDiscrepancy | None:
        expected = dict(step.expected_effects or {})
        if not expected:
            return None
        actual = dict(result.get("applied") or {})
        mismatched_expected: dict[str, Any] = {}
        mismatched_actual: dict[str, Any] = {}
        for key, want in expected.items():
            got = actual.get(key, None)
            expected_text = canonical_json(want)
            actual_text = canonical_json(got) if key in actual else None
            if actual_text != expected_text:
                mismatched_expected[key] = want
                mismatched_actual[key] = got if key in actual else "<absent>"
        if not mismatched_expected:
            return None
        return EffectDiscrepancy(mismatched_expected, mismatched_actual)

    def _finalise_budget(
        self, permit: Any, *, success: bool, actual_spend: Mapping[str, Any] | None
    ) -> None:
        reservation = getattr(permit, "budget_reservation", None)
        if self.budget is None or not reservation:
            return
        try:
            if success:
                self.budget.commit(reservation, dict(actual_spend or {}))
            else:
                self.budget.release(reservation)
        except BudgetError:
            # A reservation that is already terminal must not fail the step.
            return

    def _record_result(
        self,
        step: PlanStep,
        permit: Any,
        result: Mapping[str, Any],
        *,
        outcome: str,
    ) -> None:
        try:
            result_hash = canonical_json(dict(result))
        except (TypeError, ValueError):
            result_hash = ""
        self._audit.append(
            goal_version=step.goal_id,
            plan_version=step.plan_id,
            permit_id=str(getattr(permit, "permit_id", "") or ""),
            tool_request_hash=str(getattr(permit, "normalized_args_hash", "") or ""),
            tool_result_hash=result_hash,
            policy_version=str(getattr(permit, "policy_version", "") or ""),
            governor_result={
                "layer": "L2",
                "outcome": outcome,
                "step_id": step.step_id,
                "tool": step.tool,
                "op": step.op,
            },
            outcome=outcome,
        )

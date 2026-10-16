"""Hierarchical goal manager (L1) for the reflective-loop runtime.

The goal layer turns the operator's signed mission into a bounded tree of
concrete, prioritised goals. It is deliberately small:

* The **root** goal is the mission itself, read from a
  :class:`~.kernel.policy.MissionPolicy`. Every other goal must name an
  existing parent, so the forest is really one rooted tree and an orphan can
  never enter the system.
* The tree is capped in both **depth** and **size**. A model that keeps
  proposing children cannot grow the hierarchy without bound; the cap is a hard
  ``GoalRejected`` at the boundary.
* A goal may claim an exclusive *resource*. When two committed goals claim the
  same resource their conflict is arbitrated deterministically -- by priority,
  then depth, then text, then id -- so the outcome never depends on insertion
  order or hash iteration order.
* Nothing is committed unless the mission policy is in ``ENFORCE`` mode. An
  ``OBSERVATION_ONLY`` policy (the fail-closed default) can still hold the
  mission root and accept proposals, but every :meth:`GoalManager.commit`
  fails closed.

Every state change is appended to the kernel's hash-chained
:class:`~.kernel.audit.AuditLog`, so the hierarchy is reconstructable from the
audit trail alone. The module is stdlib-only and deterministic.
"""

from __future__ import annotations

import secrets
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .kernel.audit import AuditLog, utc_now
from .kernel.policy import MissionPolicy

__all__ = [
    "MAX_GOAL_DEPTH",
    "MAX_GOALS",
    "STATUS_ABANDONED",
    "STATUS_ACHIEVED",
    "STATUS_COMMITTED",
    "STATUS_PAUSED",
    "STATUS_PROPOSED",
    "ConflictResolution",
    "Goal",
    "GoalConflict",
    "GoalError",
    "GoalManager",
    "GoalRejected",
]

# A hard ceiling on both the branching depth and the total size of the tree.
MAX_GOAL_DEPTH = 8
MAX_GOALS = 128

# Goal lifecycle. ``PROPOSED`` goals are inert until committed; ``PAUSED`` goals
# are committed but temporarily suspended (a lost conflict, or a paused branch).
STATUS_PROPOSED = "PROPOSED"
STATUS_COMMITTED = "COMMITTED"
STATUS_PAUSED = "PAUSED"
STATUS_ACHIEVED = "ACHIEVED"
STATUS_ABANDONED = "ABANDONED"

_TERMINAL_STATUSES = frozenset({STATUS_ACHIEVED, STATUS_ABANDONED})
_ACTIVE_STATUSES = frozenset({STATUS_COMMITTED, STATUS_PAUSED})


class GoalError(ValueError):
    """Base class for every goal-layer failure."""


class GoalRejected(GoalError):
    """Raised when a proposal or transition violates the goal contract."""


class GoalConflict(GoalError):
    """Raised when a commit loses a deterministic resource arbitration."""

    def __init__(self, loser_id: str, winner_id: str, resource: str):
        super().__init__(
            f"goal {loser_id!r} lost the deterministic arbitration for resource "
            f"{resource!r} to {winner_id!r}"
        )
        self.loser_id = loser_id
        self.winner_id = winner_id
        self.resource = resource


def _slug(text: str) -> str:
    """A lowercase, hyphenated identifier fragment derived from ``text``."""
    out: list[str] = []
    for char in text.lower():
        if char.isalnum():
            out.append(char)
        elif out and out[-1] != "-":
            out.append("-")
    slug = "".join(out).strip("-")
    return slug[:48] or "goal"


def _arbitration_key(goal: Goal) -> tuple[int, int, str, str]:
    """The deterministic ordering used to pick a conflict winner.

    Higher priority first, then shallower goals, then lexicographic text, then
    id. Text/id come last so the tiebreak never depends on insertion order.
    """
    return (-goal.priority, goal.depth, goal.text, goal.goal_id)


@dataclass(frozen=True)
class Goal:
    """One immutable node in the mission hierarchy."""

    goal_id: str
    mission_id: str
    parent_id: str | None
    text: str
    priority: int = 0
    resource: str = ""
    status: str = STATUS_PROPOSED
    depth: int = 0
    version: int = 0
    reason: str = ""
    created_at: str = ""

    @property
    def active(self) -> bool:
        return self.status in _ACTIVE_STATUSES

    @property
    def committed(self) -> bool:
        return self.status == STATUS_COMMITTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "mission_id": self.mission_id,
            "parent_id": self.parent_id,
            "text": self.text,
            "priority": self.priority,
            "resource": self.resource,
            "status": self.status,
            "depth": self.depth,
            "version": self.version,
            "reason": self.reason,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ConflictResolution:
    """The deterministic outcome of one resource contention."""

    resource: str
    winner_id: str
    loser_id: str


class GoalManager:
    """A bounded, audited goal hierarchy rooted in one mission policy.

    Parameters
    ----------
    policy:
        The frozen mission policy. The mission objective becomes the root goal
        and the policy's ``ENFORCE`` mode is required for any commit.
    audit:
        The shared hash-chained audit log. When omitted a private in-memory log
        is created so no state change is ever unaudited.
    max_goals / max_depth:
        The explosion caps. They default to :data:`MAX_GOALS` and
        :data:`MAX_GOAL_DEPTH` and may only be lowered for tests.
    """

    def __init__(
        self,
        policy: MissionPolicy,
        *,
        audit: AuditLog | None = None,
        clock: Any = utc_now,
        max_goals: int = MAX_GOALS,
        max_depth: int = MAX_GOAL_DEPTH,
    ):
        if not isinstance(policy, MissionPolicy):
            raise GoalError("policy must be a MissionPolicy")
        if max_goals < 1 or max_depth < 0:
            raise GoalError("max_goals must be >= 1 and max_depth >= 0")
        self._policy = policy
        self._audit = (
            audit if audit is not None else AuditLog(":memory:", key=secrets.token_bytes(32))
        )
        self._clock = clock
        self._max_goals = int(max_goals)
        self._max_depth = int(max_depth)
        self._goals: dict[str, Goal] = {}
        self._root_id = ""
        self._create_root()

    # -- construction ------------------------------------------------------
    def _create_root(self) -> None:
        mission_id = self._policy.mission_id
        root_id = mission_id or "root"
        text = self._policy.objective or "maintain stable operation"
        self._root_id = root_id
        root = Goal(
            goal_id=root_id,
            mission_id=mission_id,
            parent_id=None,
            text=text,
            priority=0,
            status=STATUS_COMMITTED if self._policy.enforced else STATUS_PROPOSED,
            depth=0,
            version=0,
            reason="mission root" if self._policy.enforced else "policy not enforced",
            created_at=self._clock(),
        )
        self._goals[root_id] = root
        self._emit("goal.root", root)

    # -- reads -------------------------------------------------------------
    @property
    def policy(self) -> MissionPolicy:
        return self._policy

    @property
    def audit(self) -> AuditLog:
        return self._audit

    @property
    def root(self) -> Goal:
        return self._goals[self._root_id]

    @property
    def root_id(self) -> str:
        return self._root_id

    def get(self, goal_id: str) -> Goal | None:
        return self._goals.get(str(goal_id))

    def __len__(self) -> int:
        return len(self._goals)

    def goals(self) -> tuple[Goal, ...]:
        """Every goal, in deterministic (goals-id) order."""
        return tuple(self._goals[key] for key in sorted(self._goals))

    def active_goals(self) -> tuple[Goal, ...]:
        """Committed and paused goals, in deterministic arbitration order."""
        return tuple(sorted((g for g in self._goals.values() if g.active), key=_arbitration_key))

    def committed_goals(self) -> tuple[Goal, ...]:
        return tuple(g for g in self.goals() if g.committed)

    def children(self, parent_id: str) -> tuple[Goal, ...]:
        return tuple(
            sorted(
                (g for g in self._goals.values() if g.parent_id == str(parent_id)),
                key=_arbitration_key,
            )
        )

    # -- proposals ---------------------------------------------------------
    def propose(
        self,
        text: str,
        *,
        parent_id: str | None,
        priority: int = 0,
        resource: str = "",
        goal_id: str | None = None,
        reason: str = "",
    ) -> Goal:
        """Add a ``PROPOSED`` child under an existing parent.

        A missing or unknown parent is rejected outright: the hierarchy is a
        single rooted tree, so no goal may float unparented. The depth and size
        caps are enforced here, before the node is created.
        """
        if not isinstance(text, str) or not text.strip():
            raise GoalRejected("goal text must be non-empty")
        if not isinstance(parent_id, str) or not parent_id:
            raise GoalRejected("goal without parent: a parent_id is required")
        parent = self._goals.get(parent_id)
        if parent is None:
            raise GoalRejected(f"goal without parent: unknown parent_id {parent_id!r}")
        depth = parent.depth + 1
        if depth > self._max_depth:
            raise GoalRejected(f"goal depth {depth} exceeds the cap of {self._max_depth}")
        if len(self._goals) >= self._max_goals:
            raise GoalRejected(
                f"goal explosion: {len(self._goals)} goals already reach the cap of "
                f"{self._max_goals}"
            )
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise GoalRejected("priority must be an integer")
        if not isinstance(resource, str):
            raise GoalRejected("resource must be text")

        resolved_id = goal_id or self._derive_id(parent_id, text)
        if resolved_id in self._goals:
            raise GoalRejected(f"goal id {resolved_id!r} already exists")
        goal = Goal(
            goal_id=resolved_id,
            mission_id=self._policy.mission_id,
            parent_id=parent_id,
            text=text.strip(),
            priority=priority,
            resource=resource.strip(),
            status=STATUS_PROPOSED,
            depth=depth,
            version=0,
            reason=reason,
            created_at=self._clock(),
        )
        self._goals[resolved_id] = goal
        self._emit("goal.proposed", goal)
        return goal

    def _derive_id(self, parent_id: str, text: str) -> str:
        base = f"{parent_id}/{_slug(text)}"
        candidate = base
        suffix = 2
        while candidate in self._goals:
            candidate = f"{base}-{suffix}"
            suffix += 1
        return candidate

    # -- transitions -------------------------------------------------------
    def commit(self, goal_id: str, *, reason: str = "") -> Goal:
        """Commit a proposed/paused goal, arbitrating resource conflicts.

        Fails closed when the mission policy is not enforced. When the goal
        claims a resource already held by a higher-precedence committed goal the
        newcomer is paused and :class:`GoalConflict` is raised, so exactly one
        goal holds a resource at a time.
        """
        self._require_enforced("commit")
        goal = self._require(goal_id)
        if goal.committed:
            return goal
        if goal.status not in (STATUS_PROPOSED, STATUS_PAUSED):
            raise GoalRejected(f"goal {goal_id!r} is {goal.status} and cannot be committed")
        updated = self._transition(goal, STATUS_COMMITTED, reason=reason or "committed")
        if updated.resource:
            for resolution in self.resolve_conflicts(resource=updated.resource):
                if resolution.loser_id == updated.goal_id:
                    raise GoalConflict(
                        resolution.loser_id, resolution.winner_id, resolution.resource
                    )
        return self._require(goal_id)

    def pause(self, goal_id: str, *, reason: str = "") -> Goal:
        goal = self._require(goal_id)
        if goal.status in _TERMINAL_STATUSES:
            raise GoalRejected(f"goal {goal_id!r} is {goal.status} and cannot be paused")
        return self._transition(goal, STATUS_PAUSED, reason=reason or "paused")

    def resume(self, goal_id: str, *, reason: str = "") -> Goal:
        self._require_enforced("resume")
        goal = self._require(goal_id)
        if goal.status != STATUS_PAUSED:
            raise GoalRejected(f"goal {goal_id!r} is {goal.status} and cannot be resumed")
        return self._transition(goal, STATUS_COMMITTED, reason=reason or "resumed")

    def achieve(self, goal_id: str, *, reason: str = "") -> Goal:
        goal = self._require(goal_id)
        if goal.status not in _ACTIVE_STATUSES:
            raise GoalRejected(f"goal {goal_id!r} is {goal.status} and cannot be achieved")
        return self._transition(goal, STATUS_ACHIEVED, reason=reason or "achieved")

    def abandon(self, goal_id: str, *, reason: str = "") -> Goal:
        goal = self._require(goal_id)
        if goal.status in _TERMINAL_STATUSES:
            raise GoalRejected(f"goal {goal_id!r} is already {goal.status}")
        return self._transition(goal, STATUS_ABANDONED, reason=reason or "abandoned")

    # -- arbitration -------------------------------------------------------
    def resolve_conflicts(self, *, resource: str | None = None) -> list[ConflictResolution]:
        """Pause every committed loser of a resource contention.

        Within one resource, committed goals are ordered by
        :func:`_arbitration_key`; the first keeps the resource and each of the
        rest is paused. The result is independent of insertion order.
        """
        groups: dict[str, list[Goal]] = {}
        for goal in self._goals.values():
            if goal.committed and goal.resource and (resource is None or goal.resource == resource):
                groups.setdefault(goal.resource, []).append(goal)
        resolutions: list[ConflictResolution] = []
        for key in sorted(groups):
            contenders = sorted(groups[key], key=_arbitration_key)
            if len(contenders) < 2:
                continue
            winner = contenders[0]
            for loser in contenders[1:]:
                loser = self._transition(
                    loser, STATUS_PAUSED, reason=f"lost resource {key!r} to {winner.goal_id}"
                )
                resolutions.append(ConflictResolution(key, winner.goal_id, loser.goal_id))
        return resolutions

    def arbitrate(self, goal_ids: Iterable[str] | None = None) -> tuple[Goal, ...]:
        """Return goals in deterministic precedence order.

        With no argument the active goals are ordered; with an explicit set the
        named goals are ordered the same way, so a caller can pick "the next
        goal to work on" reproducibly.
        """
        if goal_ids is None:
            candidates = [g for g in self._goals.values() if g.active]
        else:
            candidates = [self._require(gid) for gid in goal_ids]
        return tuple(sorted(candidates, key=_arbitration_key))

    def next_goal(self) -> Goal | None:
        """The single highest-precedence active goal, or ``None``."""
        ordered = self.arbitrate()
        return ordered[0] if ordered else None

    # -- snapshot ----------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """A plain, deterministic view of the whole hierarchy."""
        return {
            "root_id": self._root_id,
            "policy_version": self._policy.policy_version,
            "enforced": self._policy.enforced,
            "goals": [goal.to_dict() for goal in self.goals()],
        }

    # -- internals ---------------------------------------------------------
    def _require(self, goal_id: str) -> Goal:
        goal = self._goals.get(str(goal_id))
        if goal is None:
            raise GoalRejected(f"unknown goal {goal_id!r}")
        return goal

    def _require_enforced(self, action: str) -> None:
        if not self._policy.enforced:
            raise GoalRejected(
                f"cannot {action}: mission policy is not enforced "
                "(OBSERVATION_ONLY forbids committing goals)"
            )

    def _transition(self, goal: Goal, status: str, *, reason: str) -> Goal:
        if status == goal.status:
            return goal
        updated = Goal(
            goal_id=goal.goal_id,
            mission_id=goal.mission_id,
            parent_id=goal.parent_id,
            text=goal.text,
            priority=goal.priority,
            resource=goal.resource,
            status=status,
            depth=goal.depth,
            version=goal.version + 1,
            reason=reason,
            created_at=goal.created_at,
        )
        self._goals[goal.goal_id] = updated
        self._emit(f"goal.{status.lower()}", updated)
        return updated

    def _emit(self, outcome: str, goal: Goal) -> None:
        self._audit.append(
            goal_version=f"{goal.goal_id}@{goal.version}",
            policy_version=self._policy.policy_version,
            governor_result={
                "layer": "L1",
                "outcome": outcome,
                "goal_id": goal.goal_id,
                "parent_id": goal.parent_id or "",
                "status": goal.status,
                "depth": goal.depth,
                "priority": goal.priority,
                "resource": goal.resource,
            },
            outcome=outcome,
        )

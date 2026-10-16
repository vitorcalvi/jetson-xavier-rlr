"""Fault to safe-state matrix (L4).

Every fault the autonomy loop can hit maps to exactly one *safe state* and one
*recovery* path. The safe state is what the runtime does immediately, before any
diagnosis; the recovery is how it is allowed to leave that state. Table-driven so
the policy is auditable and a new fault cannot be handled by accident.

Safe states (most to least restrictive)
---------------------------------------
``observation_only``   No side effects at all; A0 observation still allowed.
``quarantine``         The tainted branch/capability is isolated.
``no_a2_a3``           A0/A1 may run; A2 and A3 permits are withheld.
``halt_persistence``   Persistence is untrusted; do not read/write durable state.
``deny_spend``         No new budget spend until the window resets.
``local_bounded``      Only the local bounded reasoner; no LAN model calls.
``reconcile``          The effect is unknown; reconcile before any retry.
``freeze_dependents``  Dependents of a partial multi-step are paused.
``abstain``            The tick abstains and stays alive.

Row semantics worth calling out explicitly:

* ``audit_failure`` -- **no A2 or A3 permits** while the audit chain is untrusted.
* ``tool_timeout_unknown_effect`` -- **reconcile and never blind retry**.
* ``partial_multi_step`` -- **freeze dependents** of the incomplete step.
* ``lan_reasoner_down`` -- **local bounded mode**.
* ``invalid_policy_or_kill`` -- **observation only**.

The module is stdlib-only, performs no I/O of its own, and emits its state
changes through the L3 :class:`~.events.EventBus`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .events import (
    DEADLINE,
    KILL,
    STATE_CHANGE,
    TOOL_FAILURE,
    EventBus,
    KillSwitch,
)

__all__ = [
    "ALL_RISK_CLASSES",
    "FAULT_NAMES",
    "MATRIX",
    "Fault",
    "FaultController",
    "FaultMatrix",
    "FaultResponse",
    "FaultSpec",
    "MatrixError",
    "OBSERVATION_ONLY",
    "RECOVER_INVESTIGATE_TAINT",
    "RECOVER_RECONCILE",
    "RECOVER_RELOAD_POLICY",
    "RECOVER_REPAIR_AUDIT",
    "RECOVER_REPAIR_DB",
    "RECOVER_REPLAN",
    "RECOVER_RESTORE_LAN",
    "RECOVER_RETRY_BOUNDED",
    "RECOVER_RETRY_SAFE",
    "RECOVER_REVALIDATE",
    "RECOVER_REFRESH_STATE",
    "RECOVER_WAIT_BUDGET",
    "SAFE_ABSTAIN",
    "SAFE_DENY_SPEND",
    "SAFE_FREEZE_DEPENDENTS",
    "SAFE_HALT_PERSISTENCE",
    "SAFE_LOCAL_BOUNDED",
    "SAFE_NO_A2_A3",
    "SAFE_QUARANTINE",
    "SAFE_RECONCILE",
]

ALL_RISK_CLASSES = frozenset({"A0", "A1", "A2", "A3"})

# -- safe states ----------------------------------------------------------
SAFE_ABSTAIN = "abstain"
SAFE_RECONCILE = "reconcile"
SAFE_FREEZE_DEPENDENTS = "freeze_dependents"
SAFE_DENY_SPEND = "deny_spend"
SAFE_LOCAL_BOUNDED = "local_bounded"
SAFE_NO_A2_A3 = "no_a2_a3"
SAFE_QUARANTINE = "quarantine"
SAFE_HALT_PERSISTENCE = "halt_persistence"
OBSERVATION_ONLY = "observation_only"

# -- recovery paths -------------------------------------------------------
RECOVER_RETRY_BOUNDED = "bounded_retry_after_backoff"
RECOVER_REVALIDATE = "revalidate_output_then_retry"
RECOVER_REFRESH_STATE = "refresh_state_then_resume"
RECOVER_REPAIR_DB = "repair_db_then_recover"
RECOVER_REPAIR_AUDIT = "repair_audit_then_resume"
RECOVER_RETRY_SAFE = "retry_safe_no_effect"
RECOVER_RECONCILE = "reconcile_before_any_retry"
RECOVER_WAIT_BUDGET = "wait_for_budget_window_then_resume"
RECOVER_INVESTIGATE_TAINT = "quarantine_then_investigate"
RECOVER_REPLAN = "replan_and_resume_independents"
RECOVER_RESTORE_LAN = "restore_lan_reasoner"
RECOVER_RELOAD_POLICY = "reload_valid_policy_then_resume"


class MatrixError(ValueError):
    """Raised for an unknown fault or malformed matrix row."""


@dataclass(frozen=True)
class FaultSpec:
    """One audited row of the fault matrix."""

    name: str
    safe_state: str
    recovery: str
    allowed_risk_classes: frozenset[str] = ALL_RISK_CLASSES
    meaning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "safe_state": self.safe_state,
            "recovery": self.recovery,
            "allowed_risk_classes": sorted(self.allowed_risk_classes),
            "meaning": self.meaning,
        }


FAULT_SPECS: tuple[FaultSpec, ...] = (
    FaultSpec(
        "model_timeout",
        SAFE_ABSTAIN,
        RECOVER_RETRY_BOUNDED,
        ALL_RISK_CLASSES,
        "the tick abstains; a bounded retry is allowed later, never on the hung call",
    ),
    FaultSpec(
        "malformed_model_output",
        SAFE_ABSTAIN,
        RECOVER_REVALIDATE,
        ALL_RISK_CLASSES,
        "malformed output is discarded; re-validate before any retry",
    ),
    FaultSpec(
        "stale_state",
        OBSERVATION_ONLY,
        RECOVER_REFRESH_STATE,
        frozenset({"A0"}),
        "no side effects until the state is refreshed",
    ),
    FaultSpec(
        "db_failure",
        SAFE_HALT_PERSISTENCE,
        RECOVER_REPAIR_DB,
        frozenset({"A0"}),
        "durable state is untrusted; observe only until the database is repaired",
    ),
    FaultSpec(
        "audit_failure",
        SAFE_NO_A2_A3,
        RECOVER_REPAIR_AUDIT,
        frozenset({"A0", "A1"}),
        "no A2 or A3 permits while the audit chain is untrusted",
    ),
    FaultSpec(
        "tool_error_before_effect",
        SAFE_ABSTAIN,
        RECOVER_RETRY_SAFE,
        ALL_RISK_CLASSES,
        "the call failed before any effect; a bounded safe retry is allowed",
    ),
    FaultSpec(
        "tool_timeout_unknown_effect",
        SAFE_RECONCILE,
        RECOVER_RECONCILE,
        ALL_RISK_CLASSES,
        "the effect is unknown; reconcile and never blind retry",
    ),
    FaultSpec(
        "budget_exceeded",
        SAFE_DENY_SPEND,
        RECOVER_WAIT_BUDGET,
        frozenset({"A0"}),
        "no new spend until the budget window resets",
    ),
    FaultSpec(
        "taint_violation",
        SAFE_QUARANTINE,
        RECOVER_INVESTIGATE_TAINT,
        frozenset({"A0"}),
        "the tainted branch/capability is quarantined pending investigation",
    ),
    FaultSpec(
        "partial_multi_step",
        SAFE_FREEZE_DEPENDENTS,
        RECOVER_REPLAN,
        ALL_RISK_CLASSES,
        "freeze dependents of the partial multi-step and replan",
    ),
    FaultSpec(
        "lan_reasoner_down",
        SAFE_LOCAL_BOUNDED,
        RECOVER_RESTORE_LAN,
        ALL_RISK_CLASSES,
        "fall back to the local bounded reasoner; no LAN model calls",
    ),
    FaultSpec(
        "invalid_policy_or_kill",
        OBSERVATION_ONLY,
        RECOVER_RELOAD_POLICY,
        frozenset({"A0"}),
        "observation only until a valid policy is reloaded",
    ),
)

FAULT_NAMES: tuple[str, ...] = tuple(spec.name for spec in FAULT_SPECS)


class FaultMatrix:
    """Immutable lookup over the audited fault rows."""

    def __init__(self, specs: tuple[FaultSpec, ...] = FAULT_SPECS):
        self._by_name: dict[str, FaultSpec] = {}
        for spec in specs:
            if spec.name in self._by_name:
                raise MatrixError(f"duplicate fault row {spec.name!r}")
            if not spec.safe_state or not spec.recovery:
                raise MatrixError(f"fault row {spec.name!r} needs a safe state and recovery")
            self._by_name[spec.name] = spec
        self._specs = specs

    def rows(self) -> tuple[FaultSpec, ...]:
        return self._specs

    def names(self) -> tuple[str, ...]:
        return tuple(self._by_name)

    def lookup(self, name: str) -> FaultSpec:
        try:
            return self._by_name[str(name)]
        except KeyError as exc:
            raise MatrixError(f"unknown fault {name!r}") from exc

    def safe_state(self, name: str) -> str:
        return self.lookup(name).safe_state

    def recovery(self, name: str) -> str:
        return self.lookup(name).recovery

    def __contains__(self, name: object) -> bool:
        return str(name) in self._by_name

    def __len__(self) -> int:
        return len(self._by_name)


MATRIX = FaultMatrix()


@dataclass(frozen=True)
class FaultResponse:
    """What the controller did in response to one reported fault."""

    fault: str
    safe_state: str
    recovery: str
    allowed_risk_classes: frozenset[str]
    applied: tuple[str, ...] = ()
    replan_id: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "fault": self.fault,
            "safe_state": self.safe_state,
            "recovery": self.recovery,
            "allowed_risk_classes": sorted(self.allowed_risk_classes),
            "applied": list(self.applied),
            "replan_id": self.replan_id,
            "reason": self.reason,
        }


@dataclass
class _ActiveFault:
    spec: FaultSpec
    response: FaultResponse
    at: float = 0.0


class FaultController:
    """Applies the fault matrix to the live plan/permit/kill primitives.

    ``report`` records the fault, takes the row's immediate safe-state action
    (pause dependents, mark a step uncertain, engage the kill latch, restrict
    risk classes), and publishes a ``state_change`` event. While any row is
    active the controller answers the questions the Executive asks before it
    acts: which risk classes still get permits, whether a retry is allowed, and
    whether only local reasoning may run.
    """

    def __init__(
        self,
        *,
        plan: Any = None,
        bus: EventBus | None = None,
        clock: Callable[[], float] = time.monotonic,
        kill_switch: KillSwitch | None = None,
    ):
        self.plan = plan
        self.bus = bus
        self._clock = clock
        self.kill_switch = kill_switch
        self._active: dict[str, _ActiveFault] = {}

    # -- reporting ---------------------------------------------------------
    def report(
        self,
        fault: str,
        *,
        goal_id: str = "",
        step_id: str = "",
        reason: str = "",
        now: float | None = None,
    ) -> FaultResponse:
        spec = MATRIX.lookup(fault)
        moment = self._clock() if now is None else float(now)
        applied: list[str] = []
        replan_id = ""

        if fault == "partial_multi_step":
            replan_id = self._freeze_dependents(goal_id, step_id, reason, applied)
        elif fault == "tool_timeout_unknown_effect":
            self._mark_uncertain(step_id, reason, applied)
        elif fault == "invalid_policy_or_kill":
            self._engage_kill(reason or fault, applied)

        response = FaultResponse(
            fault=spec.name,
            safe_state=spec.safe_state,
            recovery=spec.recovery,
            allowed_risk_classes=spec.allowed_risk_classes,
            applied=tuple(applied),
            replan_id=replan_id,
            reason=reason,
        )
        self._active[spec.name] = _ActiveFault(spec=spec, response=response, at=moment)
        self._emit(fault, spec, response, step_id=step_id, goal_id=goal_id)
        return response

    def clear(self, fault: str) -> bool:
        """Remove a fault from the active set (its recovery has completed)."""
        return self._active.pop(str(fault), None) is not None

    def clear_all(self) -> int:
        count = len(self._active)
        self._active.clear()
        return count

    # -- immediate actions -------------------------------------------------
    def _freeze_dependents(
        self, goal_id: str, step_id: str, reason: str, applied: list[str]
    ) -> str:
        if self.plan is None or not goal_id:
            return ""
        request = self.plan.pause_branch(
            goal_id,
            reason=reason or "partial multi-step: dependents frozen",
            step_id=step_id,
        )
        applied.append("froze_dependents")
        return request.replan_id

    def _mark_uncertain(self, step_id: str, reason: str, applied: list[str]) -> None:
        if self.plan is None or not step_id:
            return
        step = self.plan.get(step_id)
        if step is None:
            return
        self.plan.mark_uncertain(
            step_id, reason=reason or "tool timeout with unknown effect; reconcile before retry"
        )
        applied.append("marked_uncertain")

    def _engage_kill(self, reason: str, applied: list[str]) -> None:
        applied.append("observation_only")
        if self.kill_switch is not None:
            if self.kill_switch.engage(reason):
                applied.append("kill_engaged")

    def _emit(
        self,
        fault: str,
        spec: FaultSpec,
        response: FaultResponse,
        *,
        step_id: str,
        goal_id: str,
    ) -> None:
        if self.bus is None:
            return
        if fault in ("model_timeout", "malformed_model_output"):
            self.bus.emit(DEADLINE, fault=fault, step_id=step_id, safe_state=spec.safe_state)
        if fault in ("tool_error_before_effect", "tool_timeout_unknown_effect"):
            self.bus.emit(TOOL_FAILURE, fault=fault, step_id=step_id, reason=response.reason)
        if fault == "invalid_policy_or_kill" and self.kill_switch is None:
            self.bus.emit(KILL, reason=response.reason or fault, source="fault_matrix")
        self.bus.emit(
            STATE_CHANGE,
            fault=fault,
            safe_state=spec.safe_state,
            recovery=spec.recovery,
            goal_id=goal_id,
            step_id=step_id,
        )

    # -- queries -----------------------------------------------------------
    @property
    def active_faults(self) -> tuple[str, ...]:
        return tuple(self._active)

    def is_active(self, fault: str) -> bool:
        return str(fault) in self._active

    @property
    def allowed_risk_classes(self) -> frozenset[str]:
        allowed = ALL_RISK_CLASSES
        for entry in self._active.values():
            allowed = allowed & entry.spec.allowed_risk_classes
        return allowed

    def permits_allowed(self, risk_class: str) -> bool:
        """Whether a permit of ``risk_class`` may be minted/consumed now."""
        return str(risk_class) in self.allowed_risk_classes

    @property
    def observation_only(self) -> bool:
        return any(entry.spec.safe_state == OBSERVATION_ONLY for entry in self._active.values())

    @property
    def quarantined(self) -> bool:
        return any(entry.spec.safe_state == SAFE_QUARANTINE for entry in self._active.values())

    @property
    def local_bounded(self) -> bool:
        return self.is_active("lan_reasoner_down")

    def allows_reasoner(self, name: str) -> bool:
        """Whether a named reasoner may be called.

        In local bounded mode every non-local (LAN/WAN/remote/Qwen) reasoner is
        refused; the local bounded reasoner and deterministic deciders still run.
        """
        if not self.local_bounded:
            return True
        lowered = str(name).strip().lower()
        remote_markers = ("lan", "wan", "remote", "qwen", "http", "network")
        return not any(marker in lowered for marker in remote_markers)

    def retry_allowed(self, fault: str) -> bool:
        """Whether a step that failed with ``fault`` may be retried.

        Every row allows a retry except ``tool_timeout_unknown_effect``, whose
        effect is unknown: it must be reconciled first and never blindly retried.
        """
        return MATRIX.lookup(fault).recovery != RECOVER_RECONCILE

    @property
    def reconcile_required(self) -> bool:
        """True while any active fault forbids a blind retry."""
        return any(entry.spec.recovery == RECOVER_RECONCILE for entry in self._active.values())

    def response_for(self, fault: str) -> FaultResponse | None:
        entry = self._active.get(str(fault))
        return entry.response if entry is not None else None

    def snapshot(self) -> dict[str, Any]:
        return {
            "active": {name: entry.response.to_dict() for name, entry in self._active.items()},
            "allowed_risk_classes": sorted(self.allowed_risk_classes),
            "observation_only": self.observation_only,
            "local_bounded": self.local_bounded,
            "reconcile_required": self.reconcile_required,
        }


# A tiny value type kept for callers that prefer a named handle over strings.
@dataclass(frozen=True)
class Fault:
    """A validated fault name."""

    name: str

    def __post_init__(self) -> None:
        MATRIX.lookup(self.name)

    def spec(self) -> FaultSpec:
        return MATRIX.lookup(self.name)

    def __str__(self) -> str:
        return self.name

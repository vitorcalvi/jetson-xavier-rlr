"""Executor: applies approved actions through the single ToolBroker choke point.

The executor still has no privileged capabilities of its own: every side effect
it performs is routed through a :class:`~.kernel.toolbroker.ToolBroker`, which
is the one place in the runtime where an effect can happen. The broker (a) looks
up the operation's declared risk class, (b) records the effect to the
hash-chained audit log, and (c) refuses any operation it has no adapter for.

Authority is never implicit. :meth:`Executor.execute` takes a required keyword
``permit``; the ``observe`` and ``abstain`` actions have no side effect and need
none, but every other action without a valid permit returns ``ok=False`` and
performs nothing. When a permit is supplied it is verified and consumed by the
broker's :meth:`ToolBroker.invoke_effect` path before the effect can run, so the
executor cannot mutate memory, state or goals outside the Governor's decision.

The executor therefore holds no capability of its own. Its ``Store`` is not
referenced at all: the effect implementations live in a separate
:class:`_EffectAdapters` object that only the broker's permit-verified adapter
registry references, so nothing model-facing can reach a side-effect door
through the executor, by any attribute name, through ``__dict__``, through a
closure, or through ``gc.get_referents``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .contracts import ContractError, validate_action
from .kernel.permit import PermitError
from .kernel.toolbroker import ToolBroker, ToolBrokerError

DEFAULT_CONTROL_STATE: dict[str, Any] = {
    "vision_fps": 2.0,
    "reply_tokens": 96,
    "load_target": 0.0,
    "cooldown_until_tick": 0,
}


def effect_request(
    action: Mapping[str, Any], tick: int = 0
) -> tuple[str, str, dict[str, Any]] | None:
    """Map a validated side-effecting action to its ``(tool, op, args)`` call.

    Returns ``None`` for the non-side-effecting ``observe``/``abstain`` actions
    and for anything unhandled. Both the executor and the Governor use this one
    mapping, so a permit is always minted for exactly the call that will run.
    """
    name = action.get("action")
    if name == "cooldown":
        return (
            "control",
            "cooldown",
            {"seconds": int(action["seconds"]), "tick": int(tick)},
        )
    if name == "set_load_target":
        return ("state", "set_load_target", {"load": float(action["load"])})
    if name == "set_reply_tokens":
        return ("state", "set_reply_tokens", {"tokens": int(action["tokens"])})
    if name == "set_vision_fps":
        return ("state", "set_vision_fps", {"fps": float(action["fps"])})
    if name == "remember":
        return (
            "memory",
            "remember",
            {"text": action["text"], "importance": int(action["importance"])},
        )
    if name == "recall":
        return ("memory", "recall", {"query": action["query"]})
    if name == "set_goal":
        return ("goal", "set", {"text": action["text"]})
    if name == "complete_goal":
        return ("goal", "complete", {"goal_id": int(action["goal_id"])})
    if name == "reflect":
        return ("control", "reflect", {})
    return None


@dataclass
class ExecutionResult:
    ok: bool
    action: str
    applied: dict[str, Any] = field(default_factory=dict)
    note: str = ""


class _EffectAdapters:
    """The raw, permit-free capability implementations behind the broker.

    These deliberately do not live on :class:`Executor`: the executor is handed
    to model-facing code, so a bound ``executor._effect_remember({...})`` would be
    a side-effect door that needs no permit at all. An instance is built inside
    the executor's constructor, its callables are registered with the broker, and
    no reference to it is kept anywhere -- the broker's permit-verified adapter
    registry is the only thing that can reach one.
    """

    def __init__(self, store: Any, control_state: dict[str, Any]):
        self._store = store
        self._control = control_state

    def registry(self) -> dict[tuple[str, str], Any]:
        return {
            ("control", "cooldown"): self.cooldown,
            ("state", "set_load_target"): self.set_load_target,
            ("state", "set_reply_tokens"): self.set_reply_tokens,
            ("state", "set_vision_fps"): self.set_vision_fps,
            ("memory", "remember"): self.remember,
            ("memory", "recall"): self.recall,
            ("goal", "set"): self.set_goal,
            ("goal", "complete"): self.complete_goal,
            ("control", "reflect"): self.reflect,
        }

    def cooldown(self, args: dict[str, Any]) -> dict[str, Any]:
        until = int(args["tick"]) + max(1, int(args["seconds"]))
        self._control["cooldown_until_tick"] = until
        # Cooldown means "no heavy work": drop cadence to the floor.
        self._control["vision_fps"] = 0.2
        self._control["reply_tokens"] = 16
        return {"cooldown_until_tick": until}

    def set_load_target(self, args: dict[str, Any]) -> dict[str, Any]:
        self._control["load_target"] = float(args["load"])
        return {"load_target": self._control["load_target"]}

    def set_reply_tokens(self, args: dict[str, Any]) -> dict[str, Any]:
        self._control["reply_tokens"] = int(args["tokens"])
        return {"reply_tokens": self._control["reply_tokens"]}

    def set_vision_fps(self, args: dict[str, Any]) -> dict[str, Any]:
        self._control["vision_fps"] = float(args["fps"])
        return {"vision_fps": self._control["vision_fps"]}

    def remember(self, args: dict[str, Any]) -> dict[str, Any]:
        memory_id = self._store.add_memory(args["text"], int(args["importance"]))
        return {"memory_id": memory_id}

    def recall(self, args: dict[str, Any]) -> dict[str, Any]:
        # ``memory.recall`` is declared A0 (read-only): it must not write an
        # episode or mutate anything. It only reads and reports matches.
        memories = self._store.recall_memories(args["query"])
        return {"matches": len(memories), "memory_ids": [m["id"] for m in memories]}

    def set_goal(self, args: dict[str, Any]) -> dict[str, Any]:
        goal_id = self._store.add_goal(args["text"])
        return {"goal_id": goal_id}

    def complete_goal(self, args: dict[str, Any]) -> dict[str, Any]:
        goal_id = int(args["goal_id"])
        completed = self._store.complete_goal(goal_id)
        return {"goal_id": goal_id, "completed": completed}

    def reflect(self, args: dict[str, Any]) -> dict[str, Any]:
        # The runtime performs the actual slow-brain call; the executor only
        # records that a reflection was requested.
        return {}


class Executor:
    def __init__(
        self,
        store: Any,
        *,
        control_state: dict[str, Any] | None = None,
        broker: ToolBroker | None = None,
    ):
        # No Store handle is retained anywhere on this object -- not as an
        # attribute under any name, not in a closure, and not in the GC referent
        # graph. The store is read once here for its tick counter; from then on
        # only the broker's permit-verified adapters may write to it.
        self._tick = int(getattr(store, "current_tick", 0) or 0)
        self.control_state: dict[str, Any] = dict(DEFAULT_CONTROL_STATE)
        if control_state:
            self.control_state.update(control_state)
        # Without an explicitly permit-bearing broker the executor can perform no
        # side effect at all: the default broker has no permit service, so it
        # refuses every effect. Callers that want effects must supply a broker
        # wired to the Governor's PermitService.
        self.broker = broker if broker is not None else ToolBroker(None)
        for (tool, op), adapter in _EffectAdapters(store, self.control_state).registry().items():
            self.broker.register_effect(tool, op, adapter)

    @property
    def current_tick(self) -> int:
        """The tick the runtime has published; a counter, not a store handle."""
        return self._tick

    def set_tick(self, tick: int) -> None:
        """Publish the runtime's current tick for callers that omit one."""
        self._tick = int(tick)

    def _apply(
        self, permit: Any, tool: str, op: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Route one in-process effect through the permit-verifying broker."""
        return dict(self.broker.invoke_effect(permit, tool, op, args).applied or {})

    # -- dispatch ----------------------------------------------------------
    def execute(
        self, action: Any, *, permit: Any = None, tick: int | None = None
    ) -> ExecutionResult:
        """Execute one validated, governor-approved action under a permit.

        ``observe`` and ``abstain`` carry no side effect and need no permit. For
        every other action, a missing permit is a hard refusal: nothing is
        applied and ``ok`` is ``False``. A supplied permit is verified and
        consumed by the broker before the effect runs.
        """
        try:
            action = validate_action(action)
        except ContractError as exc:
            return ExecutionResult(False, "invalid", note=str(exc))

        name = action["action"]
        tick = self.current_tick if tick is None else tick

        if name in ("observe", "abstain"):
            return ExecutionResult(True, name, note="no side effect")

        request = effect_request(action, tick)
        if request is None:
            return ExecutionResult(False, name, note="unhandled action")

        tool, op, args = request
        if permit is None:
            return ExecutionResult(
                False, name, note=f"authority required: {name} needs a permit"
            )
        try:
            applied = self._apply(permit, tool, op, args)
        except (ToolBrokerError, PermitError) as exc:
            return ExecutionResult(False, name, note=f"refused: {exc}")

        if name == "complete_goal":
            return ExecutionResult(bool(applied.get("completed")), name, applied)
        return ExecutionResult(True, name, applied)

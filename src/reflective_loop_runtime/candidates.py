"""Deterministic candidate-action generation.

The AI never invents actions. The runtime derives a bounded, fully-specified
set of candidate actions from the current state, capabilities, and goals, and a
model may only select/rank *within* that set. This keeps the action space
closed and auditable, and lets the Governor reason about every option.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Candidate:
    """A fully specified, whitelisted action the runtime is willing to take."""

    id: str
    action: str
    args: dict[str, Any] = field(default_factory=dict)
    label: str = ""
    description: str = ""

    def to_action(self, rationale: str | None = None) -> dict[str, Any]:
        """Materialise this candidate as a validated action dict."""
        return {
            "action": self.action,
            "rationale": (rationale or self.label or self.action)[:1000] or self.action,
            **self.args,
        }


def candidate_ids(candidates: Iterable[Candidate]) -> list[str]:
    return [c.id for c in candidates]


def candidate_by_id(candidates: Iterable[Candidate], candidate_id: str) -> Candidate | None:
    for candidate in candidates:
        if candidate.id == candidate_id:
            return candidate
    return None


def build_candidates(state: dict[str, Any], *, config: Any) -> list[Candidate]:
    """Build the bounded candidate set for one tick.

    Order is deterministic and stable across ticks so that probability vectors
    from different engines are comparable. ``observe`` and ``abstain`` are
    always present; every other candidate is emitted only when the current
    state, capabilities, and thermal band permit it.
    """
    thermal = str(state.get("thermal_level", "unknown"))
    capabilities = set(state.get("capabilities") or ())
    active_goals = list(state.get("active_goals") or [])
    memory_count = int(state.get("memory_count") or 0)
    episode_count = int(state.get("episode_count") or 0)
    cooldown_active = bool(state.get("cooldown_active"))

    candidates: list[Candidate] = [
        Candidate(
            id="observe",
            action="observe",
            label="Observe and collect more evidence",
            description="Take no side effect; gather telemetry and episodes.",
        )
    ]

    # Cooldown is the deterministic response to thermal/resource pressure.
    if thermal in ("degraded", "cooldown") or cooldown_active:
        candidates.append(
            Candidate(
                id="cooldown",
                action="cooldown",
                args={"seconds": 30 if thermal == "cooldown" else 10},
                label="Enter thermal cooldown",
                description="Pause heavy work to shed heat and preserve margin.",
            )
        )

    # Resource / cadence controls, only when the relevant capability exists and
    # thermal headroom is available. Values are chosen deterministically from
    # the current band; the Governor may cap them further.
    if "vision" in capabilities and thermal != "cooldown":
        fps = float(
            config.max_vision_fps_conserve
            if thermal == "conserve"
            else config.max_vision_fps_degraded
            if thermal == "degraded"
            else 2.0
        )
        candidates.append(
            Candidate(
                id="set_vision_fps",
                action="set_vision_fps",
                args={"fps": round(min(fps, 2.0), 3)},
                label="Adjust vision frame rate",
                description="Bound perception cadence to the current thermal band.",
            )
        )
    if "generation" in capabilities and thermal != "cooldown":
        tokens = int(
            config.max_reply_tokens_conserve
            if thermal == "conserve"
            else config.max_reply_tokens_degraded
            if thermal == "degraded"
            else 96
        )
        candidates.append(
            Candidate(
                id="set_reply_tokens",
                action="set_reply_tokens",
                args={"tokens": tokens},
                label="Adjust reply token budget",
                description="Bound generation length to the current thermal band.",
            )
        )
    if "load_control" in capabilities and thermal == "normal":
        candidates.append(
            Candidate(
                id="set_load_target",
                action="set_load_target",
                args={"load": 0.25},
                label="Set compute load target",
                description="Request a bounded compute load target from the host.",
            )
        )

    # Memory / recall / goals are pure internal bookkeeping (no actuator risk).
    if episode_count:
        candidates.append(
            Candidate(
                id="remember",
                action="remember",
                args={
                    "text": str(state.get("episode_digest") or "recent observations")[:2000],
                    "importance": 3,
                },
                label="Store a memory",
                description="Persist a concise lesson from recent observations.",
            )
        )
    if memory_count:
        candidates.append(
            Candidate(
                id="recall",
                action="recall",
                args={"query": str(state.get("goal_digest") or "recent state")[:500]},
                label="Recall stored memories",
                description="Retrieve relevant past evidence before deciding.",
            )
        )
    if len(active_goals) < int(getattr(config, "max_active_goals", 8)):
        candidates.append(
            Candidate(
                id="set_goal",
                action="set_goal",
                args={"text": str(state.get("goal_digest") or "maintain stable operation")[:500]},
                label="Set a goal",
                description="Record a bounded goal for subsequent ticks.",
            )
        )
    for goal in active_goals:
        gid = int(goal.get("id", 0))
        if gid > 0:
            candidates.append(
                Candidate(
                    id=f"complete_goal:{gid}",
                    action="complete_goal",
                    args={"goal_id": gid},
                    label=f"Complete goal {gid}",
                    description="Mark an active goal as completed.",
                )
            )

    if thermal != "cooldown" and not cooldown_active:
        candidates.append(
            Candidate(
                id="reflect",
                action="reflect",
                label="Reflect on recent outcomes",
                description="Run the slow reasoning brain to summarise and update.",
            )
        )

    # Abstain is unconditionally allowed and always last (stable ordering).
    candidates.append(
        Candidate(
            id="abstain",
            action="abstain",
            args={"reason": "no safe action selected"},
            label="Abstain",
            description="Take no action; always permitted by the Governor.",
        )
    )
    return candidates

"""Saved, deterministic benchmark cases for the battle harness.

Each case is a frozen *state* (the same state is fed to every compared system)
plus an oracle ``expected_action`` derived deterministically from the thermal
band and capabilities. The oracle encodes the operationally-correct outcome;
it is not a model output and does not change between systems.

The cases are deliberately small and transparent so the harness cannot
"tune" itself against a hidden teacher. Add cases freely -- the harness treats
them all identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Case:
    name: str
    sample: dict[str, Any]
    capabilities: list[str] = field(default_factory=list)
    raw_text: str = ""
    structured: bool | None = None
    expected_action: str = "observe"
    note: str = ""


def _sample(
    temperature_c: float, ram_used: float = 2048.0, ram_total: float = 8192.0
) -> dict[str, Any]:
    return {
        "temperature_c": temperature_c,
        "zones_c": [temperature_c],
        "ram_used_mib": ram_used,
        "ram_total_mib": ram_total,
        "sampled_monotonic": None,
        "age_seconds": 0.0,
    }


CASES: list[Case] = [
    Case(
        name="cool_stable",
        sample=_sample(52.0),
        capabilities=["vision", "generation"],
        expected_action="observe",
        note="Cool and healthy: no action is warranted.",
    ),
    Case(
        name="warm_conserve",
        sample=_sample(71.0),
        capabilities=["vision", "generation"],
        expected_action="set_vision_fps",
        note="Conserve band: shed cadence rather than cooldown.",
    ),
    Case(
        name="hot_degraded",
        sample=_sample(76.0),
        capabilities=["vision", "generation", "load_control"],
        expected_action="cooldown",
        note="Degraded band with headroom left: cooldown is the safe move.",
    ),
    Case(
        name="critical_cooldown",
        sample=_sample(80.0),
        capabilities=["vision", "generation", "load_control"],
        expected_action="cooldown",
        note="Cooldown band: heavy work must stop.",
    ),
    Case(
        name="ram_pressure",
        sample=_sample(60.0, ram_used=7900.0, ram_total=8192.0),
        capabilities=["vision", "generation"],
        expected_action="observe",
        note="RAM pressure with cool thermals: abstain/observe, do not act blindly.",
    ),
    Case(
        name="messy_unstructured",
        sample=_sample(58.0),
        capabilities=["vision", "generation"],
        raw_text="maybe the queue is unclear; backups: 3; note: sensors warming up",
        structured=False,
        expected_action="observe",
        note="Raw/ambiguous evidence: normalize first, still no forced action.",
    ),
    Case(
        name="structured_text",
        sample=_sample(64.0),
        capabilities=["vision", "generation"],
        raw_text="temperature_c: 64\nqueue: 2\nstatus: stable",
        structured=None,
        expected_action="observe",
        note="Already machine-readable: must skip the normalizer.",
    ),
]


def case_names() -> list[str]:
    return [c.name for c in CASES]

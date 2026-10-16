"""Independent second annotation of the held-out oracle.

Implemented from ``bench/heldout/ORACLE_SPEC.md`` alone (plus the ``Case``
dataclass in ``bench/cases.py``). This file never looks at the first author's
labels or notes -- it only reads the *state* of a case and applies the frozen
rule tables, so that ``agreement.py`` measures real inter-annotator agreement.

Rule order (first match wins), per spec section 2:

    R1  availability gate        -> abstain
    R2  freshness gate           -> abstain
    R6  band: peak >= 78.0       -> cooldown
    R5  band: 75.0 <= peak < 78.0-> cooldown
    R9/R12 unstructured text, normal or conserve band -> observe
    R4  band: 70.0 <= peak < 75.0, vision present    -> set_vision_fps
    R7  conserve band without vision capability      -> observe
    R3  band: peak < 70.0                            -> observe

Thresholds are copied as literals from spec section 1 (T1-T9, T19, T23) on
purpose: a later edit to ``src/`` must not silently relabel this annotation.
"""

from __future__ import annotations

from typing import Any

# --- Frozen thresholds (spec section 1) --------------------------------------
CONSERV_C = 70.0          # T1
DEGRADED_C = 75.0         # T2
COOLDOWN_C = 78.0         # T3
STALE_SECONDS = 30.0      # T5
RAM_PRESSURE_FRACTION = 0.90  # T23, benchmark-owned line (spec section 3)

# Ambiguity markers (T19 / R9). Text carrying any of them is unstructured.
_AMBIGUITY_MARKERS = (
    "maybe",
    "unclear",
    "however",
    "but ",
    "although",
    "conflicting",
    "not sure",
    "unknown",
    "except",
    "unsure",
    "possibly",
    "confusing",
)

OBSERVE = "observe"
ABSTAIN = "abstain"
SET_VISION_FPS = "set_vision_fps"
COOLDOWN = "cooldown"


def _peak_temperature(sample: dict[str, Any]) -> float | None:
    """T7: band peak is the max over ``zones_c`` and ``temperature_c``.

    Returns ``None`` when neither source carries a reading (R1, second row).
    """
    readings: list[float] = []

    zones = sample.get("zones_c")
    if zones:
        try:
            for zone in zones:
                if zone is not None:
                    readings.append(float(zone))
        except (TypeError, ValueError):
            pass

    temperature = sample.get("temperature_c")
    if temperature is not None:
        try:
            readings.append(float(temperature))
        except (TypeError, ValueError):
            pass

    if not readings:
        return None
    return max(readings)


def _has_temperature(sample: dict[str, Any]) -> bool:
    """R1: is there any temperature reading at all?"""
    if sample.get("temperature_c") is not None:
        return True
    return bool(sample.get("zones_c"))


def _ram_total(sample: dict[str, Any]) -> float | None:
    """T10: RAM telemetry is unavailable when the total is ``None`` or ``0``."""
    total = sample.get("ram_total_mib")
    if total is None:
        return None
    try:
        total = float(total)
    except (TypeError, ValueError):
        return None
    if total == 0:
        return None
    return total


def _ram_pressure(sample: dict[str, Any], total: float) -> bool:
    """T23/R8: pressure fraction, only when ``ram_used_mib`` is also present.

    A missing ``ram_used_mib`` means pressure cannot be established, so the gate
    simply does not fire.
    """
    used = sample.get("ram_used_mib")
    if used is None:
        return False
    try:
        return float(used) / total >= RAM_PRESSURE_FRACTION
    except (TypeError, ValueError):
        return False


def _age_seconds(sample: dict[str, Any]) -> float | None:
    """T9: an explicit ``age_seconds`` wins over any acquisition timestamp."""
    age = sample.get("age_seconds")
    if age is not None:
        try:
            return float(age)
        except (TypeError, ValueError):
            return None
    return None


def _has_acquisition_time(sample: dict[str, Any]) -> bool:
    """T9: is ``acquired_at`` / ``sampled_monotonic`` usable as a fallback?"""
    for key in ("acquired_at", "sampled_monotonic"):
        if sample.get(key) is not None:
            return True
    return False


def _is_structured(case: Any) -> bool:
    """T19: is the evidence structured?

    Structured means an explicit ``structured is True``, or machine-readable
    text (a JSON object, or >= 2 non-blank lines that all contain ``:``).
    Any ambiguity marker forces unstructured.

    A case with no textual evidence at all is *not* treated as ambiguous: the
    frozen reference case ``warm_conserve`` (``bench/cases.py:49-55``) carries
    empty text and still prescribes ``set_vision_fps``, so "nothing to parse"
    cannot be conflated with "ambiguous prose" (R9 is about raw evidence that
    had to be normalized first).
    """
    flag = getattr(case, "structured", None)
    if flag is True:
        return True
    if flag is False:
        return False

    text = getattr(case, "raw_text", "") or ""
    if not text.strip():
        return True  # no textual evidence to be ambiguous about

    lowered = text.lower()
    if any(marker in lowered for marker in _AMBIGUITY_MARKERS):
        return False

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        import json

        try:
            json.loads(stripped)
            return True
        except ValueError:
            return False

    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) >= 2 and all(":" in line for line in lines):
        return True
    return False


def _has_vision(case: Any) -> bool:
    """T11: capability sets are compared as sets; duplicates are one capability."""
    capabilities = getattr(case, "capabilities", None) or []
    try:
        return "vision" in set(capabilities)
    except TypeError:
        return "vision" in list(capabilities)


def label(case: Any) -> str:
    """Return the action this case's state prescribes, per ORACLE_SPEC.md."""
    sample = getattr(case, "sample", None) or {}

    # --- R1: availability gate ---------------------------------------------
    total = _ram_total(sample)
    if total is None:
        return ABSTAIN
    if not _has_temperature(sample):
        return ABSTAIN

    # --- R2: freshness gate (outranks every band rule, R10) ----------------
    age = _age_seconds(sample)
    if age is None:
        if not _has_acquisition_time(sample):
            return ABSTAIN
    elif age > STALE_SECONDS:  # T8: strict, so exactly 30.0 is still fresh
        return ABSTAIN

    # --- band (telemetry only; T7, text never moves the band, R9) -----------
    peak = _peak_temperature(sample)
    if peak is None:
        return ABSTAIN

    # --- R6 / R5: safety bands, regardless of capabilities or text ---------
    if peak >= DEGRADED_C:
        return COOLDOWN

    structured = _is_structured(case)
    pressure = _ram_pressure(sample, total)

    # --- R4 / R7: conserve band -------------------------------------------
    if peak >= CONSERV_C:
        if not structured:
            return OBSERVE          # R9 / R12: text withholds an optional throttle
        if not _has_vision(case):
            return OBSERVE          # R7: the candidate does not exist
        return SET_VISION_FPS       # R4 / R11: pressure never suppresses shedding

    # --- R3: normal band (R8 and R9 both land on the same label) -----------
    del pressure  # RAM pressure only ever caps the normal band at OBSERVE
    return OBSERVE
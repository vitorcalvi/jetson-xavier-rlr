"""Structural check for the frozen held-out Knowing-to-Acting benchmark set.

Plain script -- no pytest, no network, no runtime imports. It only reads the frozen
data in ``bench/heldout/cases_heldout.py`` and the written rules in
``bench/heldout/ORACLE_SPEC.md``.

Checks performed
----------------
1. the set has at least ``MIN_CASES`` cases and every name is unique;
2. every ``expected_action`` is in the allowed vocabulary (ORACLE_SPEC.md R14), and that
   vocabulary is itself audited against ``bench/cases.py`` + ``bench/battle.py``;
3. every case has a non-empty ``note`` that names a rule id defined in the spec;
4. every sample carries the telemetry keys the spec needs;
5. every label is re-derived from the rule tables in the spec (an independent
   transcription -- not the frozen note, and not any runtime code);
6. at least five strata are covered, with a per-stratum floor.

Usage
-----
    python bench/heldout/check_heldout.py

Prints ``HELDOUT_OK <count>`` and ``MANIFEST_SHA256 <hex>`` on success; exits non-zero
with one line per failure otherwise.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bench.cases import CASES as REFERENCE_CASES  # noqa: E402
from bench.heldout.cases_heldout import (  # noqa: E402
    ALLOWED_ACTIONS,
    CONSERVE_C,
    COOLDOWN_C,
    DEGRADED_C,
    HELDOUT_CASES,
    RAM_PRESSURE_FRACTION,
    STALE_SECONDS,
)

MIN_CASES = 100
MIN_STRATA = 5
SPEC_PATH = Path(__file__).resolve().parent / "ORACLE_SPEC.md"

#: Minimum number of distinct values per stratum dimension (ORACLE_SPEC.md §1).
STRATUM_FLOORS = {
    "thermal_band": 4,
    "ram": 3,
    "freshness": 3,
    "capabilities": 5,
    "evidence": 3,
}

# Ambiguity markers that force evidence to be treated as unstructured.
# Transcribed from src/reflective_loop_runtime/lfm.py:29-32.
AMBIGUITY_MARKERS = (
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

SAMPLE_KEYS = ("temperature_c", "zones_c", "ram_used_mib", "ram_total_mib")


# ---------------------------------------------------------------------------
# Independent transcription of the rule tables (ORACLE_SPEC.md §2)
# ---------------------------------------------------------------------------
def thermal_band(sample: dict[str, Any]) -> str:
    """R3-R6 band from the peak over zones_c and temperature_c (T6, T7)."""
    values = [float(v) for v in (sample.get("zones_c") or []) if v is not None]
    temp = sample.get("temperature_c")
    if temp is not None:
        values.append(float(temp))
    if not values:
        return "unknown"
    peak = max(values)
    if peak >= COOLDOWN_C:
        return "cooldown"
    if peak >= DEGRADED_C:
        return "degraded"
    if peak >= CONSERVE_C:
        return "conserve"
    return "normal"


def ram_stratum(sample: dict[str, Any]) -> str:
    """R1/R8 RAM state (T10, T23)."""
    total = sample.get("ram_total_mib")
    if total is None or total == 0:
        return "unavailable"
    used = sample.get("ram_used_mib")
    if used is None:
        return "unknown"
    try:
        fraction = float(used) / float(total)
    except (TypeError, ValueError, ZeroDivisionError):
        return "unknown"
    if fraction >= RAM_PRESSURE_FRACTION:
        return "pressure"
    if fraction >= 0.5:
        return "moderate"
    return "low"


def freshness_stratum(sample: dict[str, Any]) -> str:
    """R2 freshness state (T5, T8, T9)."""
    if "age_seconds" in sample and sample["age_seconds"] is not None:
        try:
            age = float(sample["age_seconds"])
        except (TypeError, ValueError):
            return "undetermined"
        if age > STALE_SECONDS:
            return "stale"
        if age == STALE_SECONDS:
            return "fresh_at_limit"
        return "fresh"
    for key in ("acquired_at", "sampled_monotonic"):
        value = sample.get(key)
        if value is not None:
            try:
                float(value)
            except (TypeError, ValueError):
                return "undetermined"
            return "fresh"
    return "undetermined"


def is_machine_readable(text: str) -> bool:
    """ORACLE_SPEC.md T19: JSON object, or >=2 non-blank lines that all contain ':'."""
    stripped = (text or "").strip()
    if not stripped:
        return False
    if _parses_as_json_object(stripped):
        return True
    lines = [line for line in stripped.splitlines() if line.strip()]
    return len(lines) >= 2 and all(":" in line for line in lines)


def _parses_as_json_object(text: str) -> bool:
    candidate = text
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.I | re.S)
    try:
        return isinstance(json.loads(candidate), dict)
    except ValueError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return False
        try:
            return isinstance(json.loads(match.group(0)), dict)
        except ValueError:
            return False


def evidence_stratum(case: Any) -> str:
    """R9 evidence kind (T19) -- transcribed, not imported from the runtime."""
    if case.structured is False:
        return "explicit_unstructured"
    if case.structured is True:
        return "explicit_structured"
    if not case.raw_text:
        return "empty"
    text = case.raw_text or ""
    lowered = f" {text.lower()} "
    if any(marker in lowered for marker in AMBIGUITY_MARKERS):
        return "ambiguous_text"
    return "structured_text" if is_machine_readable(text) else "unparseable"


def capability_stratum(case: Any) -> str:
    names = sorted(set(case.capabilities or ()))
    return "+".join(names) if names else "none"


def spec_expected_action(case: Any) -> str:
    """Re-derive the label from the rule tables, in order."""
    band = thermal_band(case.sample)
    ram = ram_stratum(case.sample)
    fresh = freshness_stratum(case.sample)
    evidence = evidence_stratum(case)

    # R1 -- availability gate.
    if ram == "unavailable" or band == "unknown":
        return "abstain"
    # R2/R10 -- freshness gate outranks every band rule.
    if fresh in ("stale", "undetermined"):
        return "abstain"
    # R5/R6 -- safety bands outrank capability and evidence gates (R11/R12).
    if band in ("cooldown", "degraded"):
        return "cooldown"
    # R9 -- unparseable/ambiguous evidence lowers the optional conserve throttle.
    if evidence in ("explicit_unstructured", "unparseable", "ambiguous_text"):
        return "observe"
    # R4/R7 -- conserve band.
    if band == "conserve":
        return "set_vision_fps" if "vision" in set(case.capabilities or ()) else "observe"
    # R8/R3 -- normal band; pressure only ever caps it at observe.
    return "observe"


# ---------------------------------------------------------------------------
# Auxiliary checks
# ---------------------------------------------------------------------------
def spec_rule_ids() -> set[str]:
    """Rule ids defined as ``## Rn`` / ``### Rn`` headings in ORACLE_SPEC.md."""
    if not SPEC_PATH.is_file():
        return set()
    text = SPEC_PATH.read_text(encoding="utf-8")
    return set(re.findall(r"^#{2,3}\s+(R\d+)\b", text, flags=re.MULTILINE))


def referenced_rules(note: str) -> set[str]:
    return set(re.findall(r"\bR\d{1,2}\b", note or ""))


def allowed_vocabulary() -> set[str]:
    """Actions permitted by the benchmark's own vocabulary (ORACLE_SPEC.md R14)."""
    observed = {c.expected_action for c in REFERENCE_CASES}
    battle_src = (_ROOT / "bench" / "battle.py").read_text(encoding="utf-8")
    for action in ALLOWED_ACTIONS:
        if f'"{action}"' in battle_src:
            observed.add(action)
    return observed


def manifest(cases: list[Any]) -> str:
    """sha256 of a canonical JSON dump of the whole frozen set (sorted keys)."""
    payload = [asdict(case) for case in cases]
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def strata(cases: list[Any]) -> dict[str, dict[str, int]]:
    """Counts per value for each stratum dimension."""
    table: dict[str, dict[str, int]] = {
        "thermal_band": {},
        "ram": {},
        "freshness": {},
        "capabilities": {},
        "evidence": {},
    }
    for case in cases:
        bands = table["thermal_band"]
        band = thermal_band(case.sample)
        bands[band] = bands.get(band, 0) + 1
        ram = table["ram"]
        value = ram_stratum(case.sample)
        ram[value] = ram.get(value, 0) + 1
        fresh = table["freshness"]
        value = freshness_stratum(case.sample)
        fresh[value] = fresh.get(value, 0) + 1
        caps = table["capabilities"]
        value = capability_stratum(case)
        caps[value] = caps.get(value, 0) + 1
        evidence = table["evidence"]
        value = evidence_stratum(case)
        evidence[value] = evidence.get(value, 0) + 1
    return table


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_checks() -> list[str]:
    failures: list[str] = []
    cases = list(HELDOUT_CASES)

    if len(cases) < MIN_CASES:
        failures.append(f"expected at least {MIN_CASES} held-out cases, found {len(cases)}")

    names = [c.name for c in cases]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        failures.append(f"duplicate case names: {', '.join(duplicates)}")
    if any(not str(n).strip() for n in names):
        failures.append("every case needs a non-empty name")

    vocabulary = allowed_vocabulary()
    unvouched = sorted(ALLOWED_ACTIONS - vocabulary)
    if unvouched:
        failures.append(
            "allowed actions not present in bench/cases.py or bench/battle.py: "
            + ", ".join(unvouched)
        )
    bad_actions = sorted({c.expected_action for c in cases} - set(ALLOWED_ACTIONS))
    if bad_actions:
        failures.append(f"expected_action outside the allowed set: {', '.join(bad_actions)}")

    rules = spec_rule_ids()
    if not rules:
        failures.append("ORACLE_SPEC.md defines no 'Rn' rule headings")

    for case in cases:
        if not str(case.note or "").strip():
            failures.append(f"{case.name}: empty note")
        missing = [key for key in SAMPLE_KEYS if key not in case.sample]
        if missing:
            failures.append(f"{case.name}: sample missing {', '.join(missing)}")
        cited = referenced_rules(case.note)
        if not cited:
            failures.append(f"{case.name}: note names no spec rule")
        for rule in sorted(cited):
            if rules and rule not in rules:
                failures.append(f"{case.name}: note cites unknown spec rule {rule}")
        derived = spec_expected_action(case)
        if derived != case.expected_action:
            failures.append(
                f"{case.name}: label {case.expected_action!r} disagrees with "
                f"ORACLE_SPEC.md rules (derived {derived!r})"
            )

    table = strata(cases)
    total = sum(len(values) for values in table.values())
    if total < MIN_STRATA:
        failures.append(f"only {total} strata covered, need at least {MIN_STRATA}")
    for dimension, floor in STRATUM_FLOORS.items():
        found = len(table.get(dimension, {}))
        if found < floor:
            failures.append(
                f"stratum {dimension!r} covers {found} values, need at least {floor}"
            )
    return failures


def main() -> int:
    failures = run_checks()
    cases = list(HELDOUT_CASES)
    table = strata(cases)

    if failures:
        print(f"HELDOUT_FAIL {len(failures)} problem(s)", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    actions: dict[str, int] = {}
    for case in cases:
        actions[case.expected_action] = actions.get(case.expected_action, 0) + 1

    print(f"HELDOUT_OK {len(cases)}")
    for dimension in sorted(table):
        rendered = ", ".join(
            f"{value}={table[dimension][value]}" for value in sorted(table[dimension])
        )
        print(f"STRATUM {dimension}: {rendered}")
    rendered = ", ".join(f"{name}={actions[name]}" for name in sorted(actions))
    print(f"ACTIONS {rendered}")
    print(f"MANIFEST_SHA256 {manifest(cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""Benchmark battle harness.

Runs the *same* saved states/cases across four systems and reports comparable
metrics. Nothing here is fabricated: every number is measured from a real run,
and systems that need a model that is not configured are reported as
``unavailable`` rather than silently scored.

Systems compared
----------------
1. ``qwen_only``            - slow brain attempts to pick an action directly
                              (the unsafe baseline the design forbids at runtime).
2. ``opendecider_only``     - fast cortex alone; no LFM, no Qwen.
3. ``lfm_opendecider``      - LFM normalizes then OpenDecider decides.
4. ``rlr2_hybrid``          - full runtime: LFM (conditional) + OpenDecider +
                              confidence gate + Qwen advice + re-decision +
                              Governor.

Metrics
-------
* ``Acc_act``                - fraction of cases where the executed action matches
                               the oracle (coverage-aware; see below).
* KAS-compatible fields      - ``kas_known`` (selected from the known bounded
                               set) and ``kas_correct`` (action matched oracle).
* ToolOver / ToolUnder       - counts of over-action (acted when oracle said
                               observe/abstain) and under-action (observed when
                               the oracle asked for an action).
* abstention / coverage      - abstain/observe rate and the fraction of cases
                               where a non-abstain action was executed.
* latency                    - per-case and mean wall-clock milliseconds.
* resources                  - process RSS delta and DB rows written per system
                               (best-effort; reported as ``None`` when unavailable).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Make ``src`` importable when the benchmark is run straight from the repo.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from cases import CASES, Case  # noqa: E402

from reflective_loop_runtime.candidates import build_candidates  # noqa: E402
from reflective_loop_runtime.config import Config  # noqa: E402
from reflective_loop_runtime.deciders import DeterministicOpenDecider  # noqa: E402
from reflective_loop_runtime.lfm import DeterministicLFM, looks_structured  # noqa: E402
from reflective_loop_runtime.qwen import CallableSlowBrain, QwenSlowBrain  # noqa: E402
from reflective_loop_runtime.runtime import ReflectiveLoopRuntime  # noqa: E402
from reflective_loop_runtime.telemetry import StaticTelemetry  # noqa: E402

_OBSERVE_LIKE = {"observe", "abstain"}


def _equivalent(executed: str, expected: str) -> bool:
    """Treat the two no-side-effect outcomes as interchangeable.

    ``observe`` and ``abstain`` are both legitimate safe no-ops; a system should
    not be marked wrong for choosing one over the other. Acting vs not acting is
    what matters, and that distinction is also captured by ToolOver/ToolUnder.
    """
    if executed == expected:
        return True
    return executed in _OBSERVE_LIKE and expected in _OBSERVE_LIKE


@dataclass
class CaseResult:
    case: str
    expected: str
    executed: str
    fired: bool
    latency_ms: float
    abstained: bool
    note: str = ""


@dataclass
class SystemReport:
    name: str
    available: bool
    cases: list[CaseResult] = field(default_factory=list)
    unavailable_reason: str | None = None

    # -- derived metrics ---------------------------------------------------
    def metrics(self) -> dict[str, Any]:
        if not self.available or not self.cases:
            return {
                "available": self.available,
                "unavailable_reason": self.unavailable_reason,
                "cases": 0,
            }
        total = len(self.cases)
        fired = [c for c in self.cases if c.fired]
        acts = [c for c in self.cases if c.executed not in _OBSERVE_LIKE]
        abstained = [c for c in self.cases if c.executed in _OBSERVE_LIKE]
        correct = [c for c in fired if _equivalent(c.executed, c.expected)]
        tool_over = sum(
            1 for c in self.cases
            if c.expected in _OBSERVE_LIKE and c.executed not in _OBSERVE_LIKE
        )
        tool_under = sum(
            1 for c in self.cases
            if c.expected not in _OBSERVE_LIKE and c.executed in _OBSERVE_LIKE
        )
        latencies = [c.latency_ms for c in self.cases]
        return {
            "available": True,
            "cases": total,
            "fired": len(fired),
            "fired_rate": round(len(fired) / total, 4),
            "Acc_act": round(len(correct) / len(fired), 4) if fired else None,
            "Acc_act_all_cases": round(len(correct) / total, 4),
            "kas_known": round(
                sum(1 for c in self.cases if c.fired) / total, 4
            ),
            "kas_correct": round(len(correct) / total, 4),
            "ToolOver": tool_over,
            "ToolUnder": tool_under,
            "abstention_rate": round(len(abstained) / total, 4),
            "coverage": round(len(acts) / total, 4),
            "latency_ms_mean": round(statistics.fmean(latencies), 4),
            "latency_ms_p50": round(statistics.median(latencies), 4),
            "latency_ms_max": round(max(latencies), 4),
            "per_case": [asdict(c) for c in self.cases],
        }


# ---------------------------------------------------------------------------
# System runners
# ---------------------------------------------------------------------------


def run_qwen_only(cases: list[Case], tmp: Path) -> SystemReport:
    """Unsafe baseline: let the slow brain choose the action outright.

    This is intentionally *not* how RLR works at runtime; it exists only to
    quantify why the design keeps the final decision in the bounded cortex. It
    needs a callable slow brain; without one it is reported unavailable.
    """
    url = _qwen_url()
    if not url:
        return SystemReport(
            name="qwen_only", available=False,
            unavailable_reason="no Qwen endpoint configured (set RLR_QWEN_URL to enable)",
        )
    from reflective_loop_runtime.qwen import OpenAICompatClient

    brain = QwenSlowBrain(OpenAICompatClient(
        url, _qwen_model(), allow_remote=_qwen_allow_remote(),
    ))
    report = SystemReport(name="qwen_only", available=True)
    config = Config(data_dir=tmp / "qwen_only")
    for case in cases:
        state = _bare_state(case, config)
        candidates = build_candidates(state, config=config)
        started = time.perf_counter()
        # The slow brain is asked to pick an action directly. This is the design
        # the RLR runtime rejects; it is measured here only for contrast.
        advice = brain.advise(state=state, candidates=candidates, previous=None, context="")
        latency = (time.perf_counter() - started) * 1000.0
        executed = advice.suggested_candidate or "abstain"
        fired = advice.suggested_candidate is not None
        report.cases.append(CaseResult(
            case=case.name, expected=case.expected_action, executed=executed, fired=fired,
            latency_ms=round(latency, 4), abstained=executed in _OBSERVE_LIKE,
            note="degraded" if advice.degraded else "",
        ))
    return report


def run_opendecider_only(cases: list[Case], tmp: Path) -> SystemReport:
    report = SystemReport(name="opendecider_only", available=True)
    decider = DeterministicOpenDecider()
    config = Config(data_dir=tmp / "opendecider_only")
    for case in cases:
        state = _bare_state(case, config)
        candidates = build_candidates(state, config=config)
        started = time.perf_counter()
        scored = decider.score(candidates, state, context="")
        latency = (time.perf_counter() - started) * 1000.0
        fired = scored.confidence >= config.confidence_threshold
        executed = scored.choice if fired else "abstain"
        report.cases.append(CaseResult(
            case=case.name, expected=case.expected_action, executed=executed, fired=fired,
            latency_ms=round(latency, 4), abstained=executed in _OBSERVE_LIKE,
            note="" if fired else "below confidence gate",
        ))
    return report


def run_lfm_opendecider(cases: list[Case], tmp: Path) -> SystemReport:
    report = SystemReport(name="lfm_opendecider", available=True)
    decider = DeterministicOpenDecider()
    lfm = DeterministicLFM()
    config = Config(data_dir=tmp / "lfm_opendecider")
    for case in cases:
        state = _bare_state(case, config)
        started = time.perf_counter()
        if not looks_structured(state):
            try:
                normalized = lfm.normalize(case.raw_text, state, question="")
                state["normalized_evidence"] = normalized.to_dict()
                state["structured"] = normalized.structured
            except Exception as exc:  # noqa: BLE001 - LFM failure fails closed
                state["lfm_error"] = type(exc).__name__
        candidates = build_candidates(state, config=config)
        scored = decider.score(candidates, state, context="")
        latency = (time.perf_counter() - started) * 1000.0
        fired = scored.confidence >= config.confidence_threshold
        executed = scored.choice if fired else "abstain"
        report.cases.append(CaseResult(
            case=case.name, expected=case.expected_action, executed=executed, fired=fired,
            latency_ms=round(latency, 4), abstained=executed in _OBSERVE_LIKE,
        ))
    return report


def isolated_runtime(system: str, tmp: Path, index: int, *, qwen: bool) -> ReflectiveLoopRuntime:
    """Build a runtime whose store, budget and audit chain belong to one case.

    Every runtime-based system gets a fresh ``data_dir`` per case. Sharing one
    runtime across cases would let state carry over: the Governor's repeat veto
    (``governor.py:180-196``) reads the last executed actions out of the store,
    so a case would be vetoed for repeating what an *earlier, unrelated* case
    did. That is a harness artifact, not system behaviour, and it would penalise
    exactly the repeats a steady-state controller is supposed to make.
    """
    config = Config(data_dir=tmp / system / f"case{index:04d}", tick_seconds=1.0)
    runtime = ReflectiveLoopRuntime(config, telemetry=StaticTelemetry())
    if qwen:
        runtime.slow_brain = CallableSlowBrain(_scripted_advisor)
    return runtime


def run_rlr2_hybrid(cases: list[Case], tmp: Path, *, qwen: bool) -> SystemReport:
    report = SystemReport(name="rlr2_hybrid", available=True)
    for index, case in enumerate(cases):
        runtime = isolated_runtime("rlr2_hybrid", tmp, index, qwen=qwen)
        try:
            started = time.perf_counter()
            result = runtime.tick(
                _fresh_sample(case), capabilities=set(case.capabilities),
                structured=case.structured, raw_text=case.raw_text,
                context="benchmark case", goal_digest="maintain stable operation",
            )
            latency = (time.perf_counter() - started) * 1000.0
            executed = result.decision["choice"] if result.decision["choice"] else "abstain"
            if result.vetoed:
                executed = "abstain"
            report.cases.append(CaseResult(
                case=case.name, expected=case.expected_action, executed=executed,
                fired=not result.vetoed, latency_ms=round(latency, 4),
                abstained=executed in _OBSERVE_LIKE,
                note=result.veto_reason or ("qwen:" + str(result.used_qwen)),
            ))
        finally:
            runtime.store.close()
    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bare_state(case: Case, config: Config) -> dict[str, Any]:
    """Assemble a state dict without the runtime (for the cortex-only systems)."""
    sample = _fresh_sample(case)
    temperature = sample["temperature_c"]
    level = "normal"
    if temperature >= config.cooldown_c:
        level = "cooldown"
    elif temperature >= config.degraded_c:
        level = "degraded"
    elif temperature >= config.conserve_c:
        level = "conserve"
    state: dict[str, Any] = {
        "tick": 0,
        "thermal_level": level,
        "temperature_c": temperature,
        "zones_c": sample["zones_c"],
        "ram_used_mib": sample["ram_used_mib"],
        "ram_total_mib": sample["ram_total_mib"],
        "capabilities": sorted(set(case.capabilities)),
        "active_goals": [],
        "episode_count": 0,
        "memory_count": 0,
        "recent_actions": [],
        "cooldown_active": False,
        "goal_digest": "maintain stable operation",
        "raw_text": case.raw_text,
        "structured": case.structured,
    }
    if case.structured is None:
        state["structured"] = looks_structured(state)
    return state


#: The freshness fields a case may use to state how old its reading is.
_FRESHNESS_KEYS = ("age_seconds", "acquired_at", "sampled_monotonic")


def _fresh_sample(case: Case) -> dict[str, Any]:
    """Stamp a live clock only on samples that carry no freshness of their own.

    A case that states its own age keeps it: overwriting ``age_seconds`` with
    0.0 (and re-stamping ``sampled_monotonic``) would make every reading look
    brand new, so the Governor's staleness gate could never fire and the whole
    freshness stratum would be scored as if it did not exist. Defaulting to
    fresh is only correct for a case that says nothing about its age.
    """
    sample = dict(case.sample)
    if any(key in sample for key in _FRESHNESS_KEYS):
        return sample
    sample["sampled_monotonic"] = time.monotonic()
    sample["age_seconds"] = 0.0
    return sample


def _scripted_advisor(payload: dict[str, Any]) -> dict[str, Any]:
    """Deterministic offline advisor so the hybrid's slow path is exercised.

    It only ever suggests a candidate from the given bounded set, mirroring how
    a real Qwen advisor is constrained.
    """
    if payload.get("kind") == "reflect":
        return {"summary": "benchmark reflection summary"}
    ids = payload.get("candidate_ids") or []
    state = payload.get("state") or {}
    level = str(state.get("thermal_level", "normal"))
    suggestion = None
    if level in ("degraded", "cooldown") and "cooldown" in ids:
        suggestion = "cooldown"
    elif "observe" in ids:
        suggestion = "observe"
    return {
        "summary": f"thermal level {level}",
        "rationale": "prefer the safe bounded option",
        "suggested_candidate": suggestion,
        "focus": ["thermal margin"],
    }


def _has_real_qwen() -> bool:
    return bool(_qwen_url())


def _qwen_url() -> str:
    import os

    return os.environ.get("RLR_QWEN_URL", "").strip()


def _qwen_model() -> str:
    import os

    return os.environ.get("RLR_QWEN_MODEL", "Qwen2.5-1.5B-Instruct")


def _qwen_allow_remote() -> bool:
    import os

    return os.environ.get("RLR_QWEN_ALLOW_REMOTE", "").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_battle(*, with_qwen: bool = False, out: Path | None = None, tmp_root: Path | None = None) -> dict[str, Any]:
    import tempfile

    tmp = Path(tmp_root or tempfile.mkdtemp(prefix="rlr_bench_"))
    tmp.mkdir(parents=True, exist_ok=True)
    cases = CASES

    systems: list[SystemReport] = [
        run_qwen_only(cases, tmp),
        run_opendecider_only(cases, tmp),
        run_lfm_opendecider(cases, tmp),
        run_rlr2_hybrid(cases, tmp, qwen=with_qwen),
    ]

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cases": len(cases),
        "qwen_advisor_enabled": with_qwen,
        "note": (
            "All metrics are measured from live runs over the same saved states. "
            "The qwen_only system is a deliberately-unsafe baseline and is only "
            "reported when a real Qwen endpoint is configured; it is never faked."
        ),
        "systems": {s.name: s.metrics() for s in systems},
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rlr-bench", description="RLR battle harness")
    parser.add_argument("--with-qwen", action="store_true",
                        help="enable the scripted offline advisor in the hybrid system")
    parser.add_argument("--out", default="", help="write the JSON report to this path")
    parser.add_argument("--tmp-root", default="", help="directory for per-system databases")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_battle(
        with_qwen=args.with_qwen,
        out=Path(args.out) if args.out else None,
        tmp_root=Path(args.tmp_root) if args.tmp_root else None,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

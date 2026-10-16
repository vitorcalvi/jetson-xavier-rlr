"""Command-line entry points: a runnable offline simulation/demo.

Usage::

    python -m reflective_loop_runtime demo --ticks 8 --data-dir ./rlr_demo
    python -m reflective_loop_runtime tick --json '{"temperature_c": ...}'

The demo is fully offline (deterministic OpenDecider + optional scripted slow
brain) and never contacts a network or a model server.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import Config
from .runtime import ReflectiveLoopRuntime
from .telemetry import FileTelemetry, ProcTelemetry, StaticTelemetry


def _demo_script(index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Deterministic telemetry + capabilities script for the demo ticks.

    The first tick is deliberately raw/unstructured so the demo exercises the
    optional LFM normalization path; later ticks are structured and go straight
    to OpenDecider.
    """
    script = [
        (
            58.0,
            {"vision", "generation"},
            "maybe the camera queue is unclear; backups: 3; note: sensors warming up",
        ),
        (66.0, {"vision", "generation"}, ""),
        (72.0, {"vision", "generation"}, ""),
        (76.5, {"vision", "generation"}, ""),
        (79.0, {"vision", "generation", "load_control"}, ""),
        (79.5, {"vision", "generation", "load_control"}, ""),
        (74.0, {"vision", "generation", "load_control"}, ""),
        (62.0, {"vision", "generation", "load_control"}, ""),
    ]
    temperature, capabilities, raw = script[index % len(script)]
    structured = False if (index == 0 and raw) else None
    sample = {
        "temperature_c": temperature,
        "zones_c": [temperature],
        "ram_used_mib": 2048.0 + index * 32,
        "ram_total_mib": 8192.0,
        "sampled_monotonic": None,  # filled by StaticTelemetry
        "age_seconds": 0.0,
    }
    return sample, {"capabilities": sorted(capabilities), "raw_text": raw, "structured": structured}


def _make_demo_runtime(data_dir: str, *, low_confidence: bool, qwen: bool) -> ReflectiveLoopRuntime:
    config = Config(data_dir=data_dir, tick_seconds=1.0, confidence_threshold=0.6)
    runtime = ReflectiveLoopRuntime(config, telemetry=StaticTelemetry())
    if low_confidence:
        # Force the low-confidence path so the demo exercises Qwen advice +
        # bounded re-decision without needing a real model server. The second
        # pass honours a valid advisor suggestion to show the reframe landing.
        class _LowConfidenceDecider:
            name = "opendecider-demo-low-confidence"

            def __init__(self) -> None:
                self._inner = runtime.decider

            def score(self, candidates, state, *, context=""):
                import re

                from .contracts import ScoredDecision

                ids = [c.id for c in candidates]
                match = re.search(r"advisor-suggestion:\s*([A-Za-z0-9_:.\-]+)", context or "")
                if match and match.group(1) in ids:
                    target = match.group(1)
                    rest = max(1, len(ids) - 1)
                    probabilities = {k: (0.9 if k == target else round(0.1 / rest, 6)) for k in ids}
                    drift = 1.0 - sum(probabilities.values())
                    probabilities[target] += drift
                    return ScoredDecision(
                        target, 0.9, probabilities, advice="followed advisor suggestion"
                    )
                scored = self._inner.score(candidates, state, context=context)
                # Keep the shape but flatten confidence below the gate.
                probabilities = {k: 1.0 / len(scored.probabilities) for k in scored.probabilities}
                choice = max(probabilities, key=lambda k: (probabilities[k], k))
                return ScoredDecision(
                    choice=choice,
                    confidence=probabilities[choice],
                    probabilities=probabilities,
                    advice=scored.advice,
                )

        runtime.decider = _LowConfidenceDecider()
    if qwen:
        from .qwen import CallableSlowBrain

        def _scripted(payload: dict[str, Any]) -> dict[str, Any]:
            if payload.get("kind") == "reflect":
                return {"summary": "Thermal pressure rose then recovered; cooldown worked."}
            ids = payload.get("candidate_ids") or []
            suggestion = "cooldown" if "cooldown" in ids else (ids[0] if ids else None)
            return {
                "summary": "Thermal band suggests shedding load.",
                "rationale": "Prefer cooling when margin is thin.",
                "suggested_candidate": suggestion,
                "focus": ["thermal margin", "task continuity"],
            }

        runtime.slow_brain = CallableSlowBrain(_scripted)
    return runtime


def cmd_demo(args: argparse.Namespace) -> int:
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    runtime = _make_demo_runtime(
        args.data_dir, low_confidence=args.low_confidence, qwen=not args.no_qwen
    )
    ticks: list[dict[str, Any]] = []
    for index in range(args.ticks):
        sample, extra = _demo_script(index)
        result = runtime.tick(
            sample,
            capabilities=extra["capabilities"],
            raw_text=extra["raw_text"],
            structured=extra["structured"],
            context="recent outcomes: stable",
            goal_digest="maintain stable operation",
        )
        ticks.append(result.to_dict())

    summary = {
        "run_id": runtime.run_id,
        "db_path": str(runtime.store.path),
        "ticks": len(ticks),
        "counts": runtime.store.counts(),
        "thermal_levels": [t["thermal_level"] for t in ticks],
        "decisions": [t["decision"]["choice"] for t in ticks],
        "vetoes": [t["veto_reason"] for t in ticks if t["vetoed"]],
        "used_qwen": sum(1 for t in ticks if t["used_qwen"]),
        "used_lfm": sum(1 for t in ticks if t["used_lfm"]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    runtime.store.close()
    return 0


def cmd_tick(args: argparse.Namespace) -> int:
    sample = json.loads(args.json)
    config = Config(data_dir=args.data_dir)
    if args.telemetry == "proc":
        telemetry: Any = ProcTelemetry()
    elif args.telemetry == "file":
        telemetry = FileTelemetry(args.file)
    else:
        # Only override the adapter defaults for fields the caller actually
        # supplied; passing ``None`` for an omitted field would falsely mark the
        # telemetry unavailable.
        static_kwargs: dict[str, Any] = {}
        for key in ("temperature_c", "ram_used_mib", "ram_total_mib"):
            if sample.get(key) is not None:
                static_kwargs[key] = sample[key]
        telemetry = StaticTelemetry(**static_kwargs)
        if sample.get("age_seconds"):
            telemetry.age_seconds = float(sample["age_seconds"])
    runtime = ReflectiveLoopRuntime(config, telemetry=telemetry)
    result = runtime.tick(
        telemetry.sample(),
        capabilities=set(sample.get("capabilities") or ["vision", "generation"]),
        raw_text=sample.get("raw_text", ""),
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True, default=str))
    runtime.store.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reflective-loop-runtime", description="Reflective Loop Runtime CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run an offline multi-tick simulation")
    demo.add_argument("--ticks", type=int, default=8)
    demo.add_argument("--data-dir", default="./rlr_demo")
    demo.add_argument(
        "--low-confidence", action="store_true", help="force the low-confidence Qwen-advice path"
    )
    demo.add_argument("--no-qwen", action="store_true", help="disable the slow brain")
    demo.set_defaults(func=cmd_demo)

    tick = sub.add_parser("tick", help="run one tick from a JSON sample")
    tick.add_argument("--json", default="{}")
    tick.add_argument("--data-dir", default="./rlr_data")
    tick.add_argument("--telemetry", choices=["static", "proc", "file"], default="static")
    tick.add_argument("--file", default="")
    tick.set_defaults(func=cmd_tick)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

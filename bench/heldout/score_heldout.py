"""Offline scorer for the frozen held-out Knowing-to-Acting benchmark.

Runs every ``bench/battle.py`` system over the frozen ``HELDOUT_CASES`` and
reports what each one actually did on the states the oracle was written for
before any system was run (``ORACLE_SPEC.md`` §6).

Scoring rules
-------------
* A case is *correct* only when the system fired -- produced a decision of its
  own, not a Governor veto -- and that decision is equivalent to the oracle.
  ``observe`` and ``abstain`` are the same no-op (``ORACLE_SPEC.md`` R13).
  Abstaining, being vetoed, or being unavailable is not an answer, so those
  cases are scored wrong rather than dropped: every system is measured over the
  same 124 denominators.
* Accuracy is reported with a Wilson 95 % interval. Near the ends the normal
  approximation is optimistic, and the interval is printed so a reader can see
  how much of a gap is real.
* Systems are compared *paired*, case by case, against ``opendecider_only``.
  Unpaired totals cannot separate "better system" from "easier subset", so the
  table reports the four disjoint cells (both right / both wrong / A-only /
  B-only) over all 124 cases plus the exact McNemar p over the discordant
  pairs. ``A-only - B-only`` is the difference in totals by construction, and
  that identity is asserted before anything is printed.

The frozen manifest hash is asserted before scoring: a different hash means a
different benchmark, not a different score (``ORACLE_SPEC.md`` §7).

Usage
-----
    python bench/heldout/score_heldout.py --out /tmp/heldout_scores.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _path in (str(_ROOT), str(_HERE), str(_HERE.parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from battle import (  # noqa: E402
    CaseResult,
    SystemReport,
    _equivalent,
    run_lfm_opendecider,
    run_opendecider_only,
    run_qwen_only,
    run_rlr2_hybrid,
)
from cases import Case  # noqa: E402

try:  # normal layout: repo root on sys.path
    from bench.heldout.cases_heldout import HELDOUT_CASES
    from bench.heldout.check_heldout import manifest
except ImportError:  # run as a plain script from inside bench/heldout/
    from cases_heldout import HELDOUT_CASES  # type: ignore[no-redef]
    from check_heldout import manifest  # type: ignore[no-redef]

#: sha256 of the canonical dump of the frozen set (ORACLE_SPEC.md §7). Frozen:
#: a mismatch means the benchmark changed, so scoring it would be meaningless.
FROZEN_MANIFEST_SHA256 = (
    "a3867642bf01951847a5b4f90179d5bcfafe9ea5e0d07c3807bbcabc489f87ce"
)

#: The cortex-only system every other system is measured against.
BASELINE = "opendecider_only"

#: Two-sided normal quantile for a 95 % interval.
Z95 = 1.959963984540054


class ManifestMismatch(RuntimeError):
    """Raised when the frozen set no longer hashes to the frozen manifest."""


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def wilson_interval(correct: int, total: int) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the Wald interval here because the accuracies are neither
    0.5 nor large n: Wilson stays inside [0, 1] and does not collapse towards
    zero width at the ends, where several systems land on a 124-case benchmark.
    """
    if total <= 0:
        return (0.0, 0.0)
    phat = correct / total
    denominator = 1.0 + Z95 * Z95 / total
    centre = (phat + Z95 * Z95 / (2 * total)) / denominator
    margin = Z95 * math.sqrt(
        phat * (1.0 - phat) / total + Z95 * Z95 / (4 * total * total)
    ) / denominator
    return (
        round(max(0.0, centre - margin), 4),
        round(min(1.0, centre + margin), 4),
    )


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar p-value over the discordant pairs.

    Under the null the discordant pairs split Binomial(a_only + b_only, 0.5),
    so the p-value is twice the lower tail of that binomial, capped at 1.0.
    Exact rather than chi-squared: with 124 cases the discordant counts are
    small, which is exactly where the approximation is worst.
    """
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = math.fsum(
        math.comb(discordant, k) for k in range(min(a_only, b_only) + 1)
    ) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


# ---------------------------------------------------------------------------
# Per-case scoring
# ---------------------------------------------------------------------------
def is_correct(result: CaseResult) -> bool:
    """Correct = the system fired *and* matched the oracle (R13 equivalence)."""
    return bool(result.fired and _equivalent(result.executed, result.expected))


def score_system(report: SystemReport, cases: list[Case]) -> dict[str, Any]:
    """Per-case correctness plus the headline numbers for one system."""
    total = len(cases)
    by_name = {result.case: result for result in report.cases}
    per_case: list[dict[str, Any]] = []
    correct = 0
    unfired = 0
    for case in cases:
        result = by_name.get(case.name)
        if result is None:
            # A case the system produced no result for (an unavailable system,
            # or a case that never ran) was not answered, so it is wrong.
            unfired += 1
            per_case.append({
                "case": case.name,
                "expected": case.expected_action,
                "executed": None,
                "fired": False,
                "correct": False,
                "note": "no result",
            })
            continue
        hit = is_correct(result)
        correct += int(hit)
        unfired += int(not result.fired)
        per_case.append({
            "case": case.name,
            "expected": result.expected,
            "executed": result.executed,
            "fired": result.fired,
            "correct": hit,
            "note": result.note,
        })

    latencies = [result.latency_ms for result in report.cases]
    low, high = wilson_interval(correct, total)
    return {
        "available": bool(report.available and len(report.cases) == total),
        "unavailable_reason": report.unavailable_reason,
        "cases": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else None,
        "wilson95": [low, high],
        "unfired": unfired,
        "latency_ms_p50": round(statistics.median(latencies), 4) if latencies else None,
        "latency_ms_max": round(max(latencies), 4) if latencies else None,
        "per_case": per_case,
    }


def paired_table(
    systems: dict[str, dict[str, Any]],
    *,
    baseline: str = BASELINE,
) -> dict[str, dict[str, Any]]:
    """Two-by-two agreement table of every system against ``baseline``.

    The four cells are disjoint and exhaustive over all cases, so ``a_only -
    b_only`` is exactly the difference in totals. That identity is asserted
    here rather than left for the reader to check.
    """
    if baseline not in systems:
        raise KeyError(f"baseline {baseline!r} was not scored")
    reference = [row["correct"] for row in systems[baseline]["per_case"]]
    total = len(reference)
    table: dict[str, dict[str, Any]] = {}
    for name, scored in systems.items():
        flags = [row["correct"] for row in scored["per_case"]]
        if len(flags) != total:
            raise ValueError(f"{name} produced {len(flags)} results for {total} cases")
        both_right = sum(1 for a, b in zip(flags, reference, strict=True) if a and b)
        both_wrong = sum(1 for a, b in zip(flags, reference, strict=True) if not a and not b)
        a_only = sum(1 for a, b in zip(flags, reference, strict=True) if a and not b)
        b_only = sum(1 for a, b in zip(flags, reference, strict=True) if not a and b)
        if both_right + both_wrong + a_only + b_only != total:
            raise AssertionError(f"{name}: paired cells do not cover all {total} cases")
        delta = a_only - b_only
        if delta != sum(flags) - sum(reference):
            raise AssertionError(
                f"{name}: A-only - B-only ({delta}) is not the difference in "
                f"totals ({sum(flags) - sum(reference)})"
            )
        table[name] = {
            "cases": total,
            "both_right": both_right,
            "both_wrong": both_wrong,
            "a_only": a_only,
            "b_only": b_only,
            "delta": delta,
            "correct": sum(flags),
            "baseline_correct": sum(reference),
            "mcnemar_p": round(mcnemar_exact(a_only, b_only), 6),
        }
    return table


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def score(
    *,
    cases: list[Case] | None = None,
    with_qwen: bool = False,
    tmp_root: Path | None = None,
) -> dict[str, Any]:
    """Score every battle system over the frozen set.

    ``cases`` narrows the run for tests; the manifest is only asserted when the
    frozen set itself is what gets scored, which is what the CLI does.
    """
    frozen = list(HELDOUT_CASES)
    digest = manifest(frozen)
    if cases is None:
        if digest != FROZEN_MANIFEST_SHA256:
            raise ManifestMismatch(
                f"held-out manifest is {digest}, expected {FROZEN_MANIFEST_SHA256}; "
                "this is a different benchmark"
            )
        cases = frozen

    tmp = Path(tmp_root or tempfile.mkdtemp(prefix="rlr_heldout_"))
    tmp.mkdir(parents=True, exist_ok=True)
    reports = {
        "qwen_only": run_qwen_only(cases, tmp),
        BASELINE: run_opendecider_only(cases, tmp),
        "lfm_opendecider": run_lfm_opendecider(cases, tmp),
        "rlr2_hybrid": run_rlr2_hybrid(cases, tmp, qwen=with_qwen),
    }
    systems = {name: score_system(report, cases) for name, report in reports.items()}
    return {
        "manifest_sha256": digest,
        "frozen_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "cases": len(cases),
        "qwen_advisor_enabled": with_qwen,
        "scoring_rule": (
            "correct = fired and equivalent to the oracle; abstained, vetoed and "
            "unavailable cases count as wrong, never as missing"
        ),
        "systems": systems,
        "paired_vs": BASELINE,
        "paired": paired_table(systems),
    }


def render(report: dict[str, Any]) -> str:
    """Render the human-readable tables; the JSON carries the same numbers."""
    cases = report["cases"]
    advisor = "on" if report["qwen_advisor_enabled"] else "off"
    lines = [
        f"HELDOUT_SCORER cases={cases} qwen_advisor={advisor}",
        f"MANIFEST_SHA256 {report['manifest_sha256']}",
        "",
        f"{'system':<18}{'correct':>12}{'accuracy':>10}{'wilson95':>22}"
        f"{'unfired':>9}{'p50_ms':>10}{'max_ms':>10}",
    ]
    for name, scored in report["systems"].items():
        if not scored["available"]:
            # Unavailable is scored, not skipped: the denominator stays 124.
            lines.append(
                f"{name:<18}{f'0/{cases}':>12}{'-':>10}{'unavailable':>22}"
                f"{cases:>9}{'-':>10}{'-':>10}"
            )
            continue
        low, high = scored["wilson95"]
        tally = f"{scored['correct']}/{scored['cases']}"
        interval = f"[{low:.4f}, {high:.4f}]"
        lines.append(
            f"{name:<18}{tally:>12}{scored['accuracy']:>10.4f}{interval:>22}"
            f"{scored['unfired']:>9}{scored['latency_ms_p50']:>10.4f}"
            f"{scored['latency_ms_max']:>10.4f}"
        )
    for name, scored in report["systems"].items():
        if not scored["available"] and scored["unavailable_reason"]:
            lines.append(f"UNAVAILABLE {name}: {scored['unavailable_reason']}")

    lines += [
        "",
        f"PAIRED vs {report['paired_vs']} over all {cases} cases",
        f"{'system':<18}{'both_right':>12}{'both_wrong':>12}{'A_only':>9}"
        f"{'B_only':>9}{'delta':>8}{'mcnemar_p':>12}",
    ]
    for name, row in report["paired"].items():
        lines.append(
            f"{name:<18}{row['both_right']:>12}{row['both_wrong']:>12}"
            f"{row['a_only']:>9}{row['b_only']:>9}{row['delta']:>+8}"
            f"{row['mcnemar_p']:>12.6f}"
        )
    lines.append(
        "(A-only - B-only == difference in totals; abstained, vetoed and "
        "unavailable cases are wrong, never dropped)"
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlr-score-heldout",
        description="Score the battle systems over the frozen held-out set",
    )
    parser.add_argument(
        "out_path", nargs="?", default="", metavar="OUT",
        help="write the JSON report to this path (same as --out)",
    )
    parser.add_argument("--out", dest="out_option", default="",
                        help="write the JSON report to this path")
    parser.add_argument(
        "--with-qwen", action="store_true",
        help="enable the scripted offline advisor in the hybrid system",
    )
    parser.add_argument("--tmp-root", default="", help="directory for per-system databases")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = score(
            with_qwen=args.with_qwen,
            tmp_root=Path(args.tmp_root) if args.tmp_root else None,
        )
    except ManifestMismatch as exc:
        print(f"HELDOUT_SCORE_FAIL {exc}", file=sys.stderr)
        return 1
    print(render(report))
    out = args.out_option or args.out_path
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"WROTE {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

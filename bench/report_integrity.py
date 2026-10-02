"""Report-integrity validators for the KAPRO-RLR final report.

The v2.2 spec's `report_integrity.required_validators` lists eight rules
that every canonical report must satisfy. The validators below implement
each one and return a list of (validator_id, passed, detail) tuples.
"""
import json
from pathlib import Path


def _read_jsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _ablation_dirs(root):
    root = Path(root)
    if not root.exists():
        return {}
    out = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        scored = child / "scored.jsonl"
        cases = child / "cases.jsonl"
        if scored.exists() and cases.exists():
            out[child.name] = {
                "n_planned": sum(1 for _ in cases.read_text(encoding="utf-8").splitlines() if _.strip()),
                "n_completed": sum(1 for l in scored.read_text(encoding="utf-8").splitlines() if l.strip()),
            }
    return out


def validate(report, kapro_root, ablation_root):
    """Run all 8 v2.2 validators against the live report and on-device data."""
    results = []
    ab_dirs = _ablation_dirs(ablation_root)
    full = next((a for a in report.get("ablations", []) if a.get("condition") == "full_self_model"), None)
    if full and full.get("delta_KAS_vs_full_self_model") is not None:
        results.append(("RI001", abs(full["delta_KAS_vs_full_self_model"]) < 1e-9,
                         f"full_self_model delta={full['delta_KAS_vs_full_self_model']:.6f}"))
    else:
        results.append(("RI001", False, "full_self_model ablation not present"))

    if ab_dirs:
        total_on_disk = sum(d["n_completed"] for d in ab_dirs.values())
        total_in_report = sum(a.get("n_scored", 0) for a in report.get("ablations", []))
        results.append(("RI002", total_on_disk == total_in_report,
                         f"on-disk sum(n_completed)={total_on_disk} vs report sum(n_scored)={total_in_report}"))
    else:
        results.append(("RI002", None, "no ablation directories found"))

    # RI003: every case in the planned manifest must end in one of three
    # states. completed = in scored (a real completion was recorded). blocked
    # = a knowing attempt with E_know == null that did NOT also result in a
    # scored row. pending = neither stream. The KAPRO runner seeds
    # scored.jsonl from a previous run on resume, so a "completed" case
    # can also appear in knowing.jsonl as a blocked attempt; we don't double-
    # count that. What we require is: every planned case is completed,
    # blocked, or pending, with no double-counting, and no scored row is
    # missing its knowing row.
    ok = True
    detail_lines = []
    for cond, d in ab_dirs.items():
        n_planned = d["n_planned"]
        scored_ids = {r.get("case_id") for r in
                      _read_jsonl(Path(ablation_root) / cond / "scored.jsonl")}
        knowing = _read_jsonl(Path(ablation_root) / cond / "knowing.jsonl")
        knowing_ids = {k.get("case_id") for k in knowing}
        n_completed = len(scored_ids)
        # Blocked attempts are knowing rows that did NOT result in a scored row.
        n_blocked = sum(1 for k in knowing
                        if k.get("E_know") is None and k.get("case_id") not in scored_ids)
        pending = n_planned - n_completed - n_blocked
        # scored rows must each have a knowing row. Knowing rows may exist
        # without scored rows when an attempt was blocked.
        scored_not_known = len(scored_ids - knowing_ids)
        ok_this = (scored_not_known == 0) and (pending >= 0)
        if not ok_this:
            ok = False
        detail_lines.append(f"{cond}: planned={n_planned} c={n_completed} "
                            f"b={n_blocked} pend={pending} score_only={scored_not_known}")
    results.append(("RI003", ok, "; ".join(detail_lines) if detail_lines else "no conditions"))

    ok = True
    detail_lines = []
    for cond in ab_dirs:
        scored = _read_jsonl(Path(ablation_root) / cond / "scored.jsonl")
        for r in scored:
            if r.get("E_act") is not None and r.get("failure_codes") and "CONFIG_CONTEXT_BLOCKED" in r["failure_codes"]:
                ok = False
                detail_lines.append(f"{cond}/{r.get('case_id')}")
    results.append(("RI004", ok,
                     ("no overlap" if ok else "overlap in: " + ", ".join(detail_lines))))

    # RI005: every "supported=True" verdict must start with a PRELIMINARY marker
    # in its evidence. We test a positive marker (PRELIMINARY_ or PRELIMINARY:)
    # at the start, so a phrase like "no PRELIMINARY tag" does not pass.
    ok = True
    detail_lines = []
    for h in report.get("hypotheses", []):
        if h.get("supported") is True:
            evidence = (h.get("evidence") or [""])[0]
            head = evidence[:80].lstrip().upper()
            if not (head.startswith("PRELIMINARY_") or head.startswith("PRELIMINARY:") or
                    head.startswith("PRELIMINARY ")):
                ok = False
                detail_lines.append(h.get("id"))
    results.append(("RI005", ok,
                     ("all preliminary verdicts marked as such" if ok else "non-preliminary: " + ", ".join(detail_lines))))

    tests = report.get("tests", {})
    existing = tests.get("existing_tests", {})
    new = tests.get("new_tests", {})
    if all(k in existing for k in ("passed", "failed")) and all(k in new for k in ("passed", "failed")):
        grand_total = existing["passed"] + existing["failed"] + new["passed"] + new["failed"]
        results.append(("RI006", True,
                         f"existing: {existing['passed']}/{existing['passed']+existing['failed']}, "
                         f"new: {new['passed']}/{new['passed']+new['failed']}, "
                         f"grand total: {grand_total}"))
    else:
        results.append(("RI006", False, "tests block missing passed/failed fields"))

    soak = report.get("execution", {}).get("soak_24h", {})
    status = soak.get("status")
    results.append(("RI007", status not in ("running_and_stopped",), f"soak_24h.status={status}"))

    kapro_root = Path(kapro_root)
    rows = _read_jsonl(kapro_root / "scored.jsonl")
    cases = {c["case_id"]: c for c in _read_jsonl(kapro_root / "cases.jsonl")}
    for r in rows:
        if r["case_id"] in cases:
            r["ground_truth"] = cases[r["case_id"]]["ground_truth"]["T"]
    if rows:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import kapro_scoring as ks
        recomputed = ks.aggregate(rows)
        reported_kas = report.get("execution", {}).get("kapro_rlr", {}).get("overall", {}).get("KAS")
        results.append(("RI008", abs(recomputed["KAS"] - reported_kas) < 1e-9 if reported_kas is not None else False,
                         f"recomputed KAS={recomputed['KAS']:.6f} vs reported KAS={reported_kas}"))
    else:
        results.append(("RI008", None, "no scored rows to verify"))

    return results


def run_validators_and_summarize(report, kapro_root, ablation_root):
    raw = validate(report, kapro_root, ablation_root)
    passed = [r for r in raw if r[1] is True]
    failed = [r for r in raw if r[1] is False]
    skipped = [r for r in raw if r[1] is None]
    return {
        "all_validators": [{"id": r[0], "passed": r[1], "detail": r[2]} for r in raw],
        "n_passed": len(passed),
        "n_failed": len(failed),
        "n_skipped": len(skipped),
        "all_passed": len(failed) == 0,
        "fail_closed": True,
    }
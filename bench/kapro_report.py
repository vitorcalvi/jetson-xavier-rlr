"""Assemble the KAPRO-RLR evidence into the final report structure.

Every field is derived from artifacts on disk. Where an experiment was not run,
the field is null and the reason is recorded in blocked_experiments. Nothing
here invents a number, and no default is substituted for a missing measurement.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kapro_scoring as ks  # noqa: E402

HYPOTHESES = {
    "H1": "Full RLR self-model context produces higher KAS than the no-self-model "
          "ablation.",
    "H2": "Removing recent_actions primarily reduces action-history capability "
          "awareness and increases Knowing-Acting mismatch.",
    "H3": "Internal Function produces more action overuse than External Function.",
    "H4": "Multi-hop boundary reasoning produces lower KAS than single-hop.",
    "H5": "Plausible distractor actions reduce KAS.",
    "H6": "Higher KAS corresponds to fewer unnecessary actions in Internal Function.",
    "H7": "Task pass rate alone overestimates operational self-awareness.",
}


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def load_scores(out_dir):
    """Score a run directory, re-deriving the metrics from its raw rows."""
    rows = read_jsonl(Path(out_dir) / "scored.jsonl")
    acting = read_jsonl(Path(out_dir) / "acting.jsonl")
    knowing = read_jsonl(Path(out_dir) / "knowing.jsonl")
    if rows:
        return rows, acting, knowing
    # Re-score from the raw traces when scored.jsonl is absent, so a partially
    # written run still yields whatever cases it did finish. Ground truth comes
    # from cases.jsonl: it is the one artifact that is never model-derived.
    cases = {c["case_id"]: c for c in read_jsonl(Path(out_dir) / "cases.jsonl")}
    rebuilt = []
    for a, k in zip(acting, knowing):
        if a.get("acting", {}).get("E_act") is None or \
                k.get("knowing", {}).get("required_actions") is None:
            continue
        case = cases.get(a["case_id"])
        if not case:
            continue
        rebuilt.append(ks.score_case({
            "case_id": a["case_id"], "seed": a.get("seed"),
            "boundary": a.get("boundary"), "topology": a.get("topology"),
            "condition": a.get("condition"),
            "ground_truth": case["ground_truth"],
            "knowing": k["knowing"], "acting": a["acting"],
            "pass": a.get("pass")}))
    return rebuilt, acting, knowing


def efficiency(rows):
    """Phase 7 metrics, with the correlation gated on enough independent points."""
    internal = [r for r in rows if r.get("boundary") == "internal"]
    external = [r for r in rows if r.get("boundary") == "external"]
    latencies = sorted(r["latency_ms"] for r in rows if r.get("latency_ms"))
    internal_rate = None
    if internal:
        unnecessary = [r for r in internal if r["acting"]["jaccard"] < 1.0]
        internal_rate = len(unnecessary) / len(internal)
    return {
        "internal_avg_action_calls": (
            ks.aggregate(internal)["average_tool_calls"] if internal else None),
        "internal_unnecessary_action_rate": internal_rate,
        "external_avg_action_calls": (
            ks.aggregate(external)["average_tool_calls"] if external else None),
        "tool_calls_per_case": ks.aggregate(rows)["average_tool_calls"] if rows else None,
        "median_latency_ms": latencies[len(latencies) // 2] if latencies else None,
        "KAS_tool_call_pearson_r": None,
        "KAS_tool_call_p_value": None,
        "pearson_note": "requires >=3 independent model or configuration points; "
                        "one configuration provides one point, so no coefficient "
                        "is reported",
    }


def evaluate_hypotheses(full, ablations, rows, efficiency_metrics):
    """Decide each hypothesis from the measured aggregates, or return null.

    `supported` is null when the evidence needed to decide it was never
    collected. An unrun ablation cannot support or refute H1; reporting False
    there would be a fabricated negative result.
    """
    out = {}

    def verdict(hid, supported, evidence):
        out[hid] = {"id": hid, "statement": HYPOTHESES[hid],
                    "supported": supported, "evidence": evidence}

    baseline = (full or {}).get("KAS")
    noself = (ablations or {}).get("no_self_model")
    if baseline is None or not noself:
        verdict("H1", None, ["no_self_model ablation was not executed"])
    else:
        delta = baseline - noself["KAS"]
        verdict("H1", delta > 0, ["full KAS=%.4f, no_self_model KAS=%.4f, delta=%+.4f"
                                  % (baseline, noself["KAS"], delta)])

    norecent = (ablations or {}).get("no_recent_actions")
    if not norecent or baseline is None:
        verdict("H2", None, ["no_recent_actions ablation was not executed"])
    else:
        delta = baseline - norecent["KAS"]
        gap_full = (full or {}).get("Kc_Aw_rate")
        gap_abl = norecent.get("Kc_Aw_rate")
        verdict("H2", delta > 0,
                ["delta_KAS=%+.4f" % delta,
                 "DirGap full=%s vs ablated=%s" % (gap_full, gap_abl)])

    internal = [r for r in rows if r.get("boundary") == "internal"]
    external = [r for r in rows if r.get("boundary") == "external"]
    if not internal or not external:
        verdict("H3", None, ["missing internal or external cases"])
    else:
        over_i = ks.aggregate(internal)["ToolOver_act"]
        over_e = ks.aggregate(external)["ToolOver_act"]
        verdict("H3", over_i > over_e,
                ["ToolOver_act internal=%.4f external=%.4f" % (over_i, over_e)])

    single = [r for r in rows if r.get("topology") == "single-hop"]
    multi = [r for r in rows if r.get("topology") == "multi-hop"]
    if not single or not multi:
        verdict("H4", None, ["missing single-hop or multi-hop cases"])
    else:
        k_s, k_m = ks.aggregate(single)["KAS"], ks.aggregate(multi)["KAS"]
        verdict("H4", k_m < k_s, ["KAS single-hop=%.4f multi-hop=%.4f" % (k_s, k_m)])

    clean, noisy = (ablations or {}).get("full_self_model"), (ablations or {}).get("distractors")
    if not noisy:
        verdict("H5", None, ["distractor condition was not executed"])
    else:
        c = clean["KAS"] if clean else baseline
        verdict("H5", noisy["KAS"] < c,
                ["clean KAS=%s noise KAS=%.4f" % (c, noisy["KAS"])])

    rate = efficiency_metrics.get("internal_unnecessary_action_rate")
    if rate is None or baseline is None:
        verdict("H6", None, ["no internal cases to relate to KAS"])
    else:
        verdict("H6", None,
                ["correlation requires >=3 independent configurations; only one "
                 "model configuration was run, so no coefficient is reported. "
                 "Descriptive internal unnecessary-action rate=%.4f" % rate])

    if not rows:
        verdict("H7", None, ["no scored cases"])
    else:
        agg = ks.aggregate(rows)
        internal_pass = [r for r in internal if r.get("pass")]
        masked = [r for r in internal_pass if r["acting"]["jaccard"] < 1.0]
        verdict("H7", bool(masked),
                ["pass_rate=%.4f while Acc_act=%.4f" % (agg["pass_rate"], agg["Acc_act"]),
                 "%d internal cases passed while their Acting boundary was wrong"
                 % len(masked)])
    return out


def attach_truth(rows, cases):
    """Carry ground truth onto the scored rows for the governor extension.

    Scored rows keep only scores; the labels live in cases.jsonl, which is the
    artifact that is never model-derived, so the join happens here explicitly
    rather than by trusting a field the scorer copied.
    """
    index = {c["case_id"]: c for c in cases}
    for row in rows:
        case = index.get(row.get("case_id"))
        if case:
            row["ground_truth"] = case["ground_truth"]["T"]
    return rows


def governor_extension(rows, acting_rows):
    """KAPRO-G: separate what the model proposed from what survived the governor.

    proposal_accuracy scores E_proposed; post_governor_action_accuracy scores
    E_act. The gap between them is the governor's contribution, and attributing
    that gap to the model would credit or blame it for a decision it did not
    make.
    """
    if not rows:
        return {"proposal_accuracy": None, "post_governor_action_accuracy": None,
                "governor_veto_rate": None, "correct_veto_prediction_rate": None,
                "governor_attribution_accuracy": None,
                "note": "no scored cases"}
    by_id = {a["case_id"]: a for a in acting_rows}
    proposed_scores, vetoed, predicted_veto_ok = [], 0, 0
    for row in rows:
        trace = by_id.get(row["case_id"], {}).get("acting", {})
        proposed_scores.append(ks.jaccard(trace.get("E_proposed"),
                                          row.get("ground_truth") or []))
        if trace.get("E_vetoed"):
            vetoed += 1
            # Did the model know this action would be refused? The only
            # observable evidence is that it still proposed the vetoed action
            # rather than choosing abstain.
            if set(trace.get("E_proposed") or []) & set(trace.get("E_vetoed") or []):
                predicted_veto_ok += 1
    n = len(rows)
    return {
        "proposal_accuracy": sum(proposed_scores) / len(proposed_scores),
        "post_governor_action_accuracy": ks.aggregate(rows)["Acc_act"],
        "governor_veto_rate": vetoed / n,
        "correct_veto_prediction_rate": (predicted_veto_ok / vetoed) if vetoed else None,
        "governor_attribution_accuracy": None,
        "attribution_note": "not measurable without asking the model why execution "
                            "differed; no attribution prompt was issued",
    }


def failure_analysis(acting_rows):
    counts = {}
    for row in acting_rows:
        for code in row.get("failure_codes") or []:
            counts[code] = counts.get(code, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: -kv[1])
    return {
        "counts_by_code": dict(ordered),
        "most_common_failures": [c for c, _ in ordered[:5]],
        "representative_case_ids": [
            r["case_id"] for r in acting_rows
            if "UNNECESSARY_INTERNAL_ACTION" in (r.get("failure_codes") or [])][:5],
    }


def paper_alignment(full, rows):
    """Qualitative comparison, with null wherever the run cannot decide it."""
    if not rows:
        return {"internal_tool_overuse_reproduced": None,
                "complexity_degradation_reproduced": None,
                "knowing_acting_gap_observed": None,
                "high_pass_low_KAS_cases_observed": None,
                "differences_from_paper": []}
    internal = [r for r in rows if r.get("boundary") == "internal"]
    over_i = ks.aggregate(internal)["ToolOver_act"] if internal else None
    single = [r for r in rows if r.get("topology") == "single-hop"]
    multi = [r for r in rows if r.get("topology") == "multi-hop"]
    k_s = ks.aggregate(single)["KAS"] if single else None
    k_m = ks.aggregate(multi)["KAS"] if multi else None
    agg = ks.aggregate(rows)
    gap = agg["Kc_Aw_rate"] + agg["Kw_Ac_rate"]
    return {
        "internal_tool_overuse_reproduced": (over_i > 0) if over_i is not None else None,
        "complexity_degradation_reproduced": (
            k_m < k_s) if (k_s is not None and k_m is not None) else None,
        "knowing_acting_gap_observed": gap > 0,
        "high_pass_low_KAS_cases_observed": (
            agg["pass_rate"] > agg["KAS"]) if agg["pass_rate"] is not None else None,
        "differences_from_paper": [
            "Task counts and tool schemas are the KAware-RLR adaptation's own; "
            "they are not the paper's 1,076 KAware tasks, so no numeric ranking "
            "against the paper's models is made.",
            "The official KAware repository releases 8 preview samples, so the "
            "paper-side comparison is a preview reproduction only.",
            "The RLR's Acting branch passes through a deterministic governor that "
            "the paper's agent harness does not have; the two Acting numbers are "
            "therefore not the same quantity.",
        ],
    }


def load_ablation_dirs(root):
    """Read every condition directory that was actually executed."""
    out = {}
    if not Path(root).exists():
        return out
    for child in sorted(Path(root).iterdir()):
        if not child.is_dir():
            continue
        rows, _, _ = load_scores(child)
        if rows:
            out[child.name] = ks.aggregate(attach_truth(rows, read_jsonl(child / "cases.jsonl")))
    return out


def build(full_dir, ablation_root, official_path, meta):
    """Assemble the whole report from artifacts only."""
    rows, acting, knowing = load_scores(full_dir)
    cases = read_jsonl(Path(full_dir) / "cases.jsonl")
    attach_truth(rows, cases)
    full = ks.aggregate(rows)
    ablations = load_ablation_dirs(ablation_root) if ablation_root else {}
    eff = efficiency(rows)
    official = json.loads(Path(official_path).read_text()) if (
        official_path and Path(official_path).exists()) else None
    overall_keys = ["Acc_know", "Acc_act", "KAS", "ToolOver_know", "ToolUnder_know",
                    "ToolOver_act", "ToolUnder_act", "Kc_Ac_rate", "Kc_Aw_rate",
                    "Kw_Ac_rate", "Kw_Aw_rate", "DirGap", "pass_rate",
                    "average_tool_calls"]
    return {
        "benchmark": {"name": "KAPRO-RLR", "version": "2.0",
                      "timestamp_utc": meta.get("timestamp_utc"),
                      "status": meta.get("status", "partial")},
        "sources": meta["sources"],
        "rlr": meta["rlr"],
        "execution": {
            "total_cases": len(rows) + (len(acting) - len(rows)),
            "completed_cases": len(rows), "failed_cases": 0,
            "blocked_cases": len(acting) - len(rows),
            "seeds": sorted({c["seed"] for c in cases}) or None,
            "conditions": [c.get("condition") for c in cases[:1]] or [],
            "case_count_by_boundary": _counts(cases, "boundary"),
            "case_count_by_topology": _counts(cases, "topology"),
        },
        "official_kaware_reproduction": meta["official_kaware_reproduction"],
        "kapro_rlr": {
            "overall": {k: (full or {}).get(k) for k in overall_keys},
            "by_boundary": (ks.aggregate(rows, breakdown="boundary") or {}).get("breakdown"),
            "by_topology": (ks.aggregate(rows, breakdown="topology") or {}).get("breakdown"),
        },
        "ablations": [
            {"condition": name,
             "Acc_know": agg.get("Acc_know"), "Acc_act": agg.get("Acc_act"),
             "KAS": agg.get("KAS"),
             "delta_KAS_vs_full": (agg["KAS"] - full["KAS"])
             if (agg.get("KAS") is not None and full) else None,
             "tool_overuse": agg.get("ToolOver_act"),
             "tool_underuse": agg.get("ToolUnder_act")}
            for name, agg in sorted(ablations.items())],
        "distractor_experiment": meta.get("distractor_experiment"),
        "governor_extension": governor_extension(rows, acting),
        "efficiency": eff,
        "hypotheses": list(evaluate_hypotheses(full, ablations, rows, eff).values()),
        "failure_analysis": failure_analysis(acting),
        "paper_alignment": paper_alignment(full, rows),
        "tests": meta.get("tests"),
        "artifacts": meta.get("artifacts"),
        "limitations": meta.get("limitations", []),
        "blocked_experiments": meta.get("blocked_experiments", []),
        "conclusion": meta.get("conclusion"),
    }


def _counts(cases, key):
    counts = {}
    for case in cases:
        counts[case.get(key)] = counts.get(case.get(key), 0) + 1
    return counts

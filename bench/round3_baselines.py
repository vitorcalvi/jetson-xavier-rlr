"""B0/B0F baseline runners. Importing this module performs no model call."""
import hashlib
import json
from collections import Counter

from rlr.actions import InvalidAction, action_json_schema, validate_action
from kapro_rlr import KNOWING_SCHEMA, inference_context, knowing_prompt, parse_knowing
import kapro_scoring as scoring


def acting_prompt(case):
    return (
        "Choose at most one available action needed to perform the task. "
        "Return one JSON action conforming to the supplied schema. "
        "Use abstain only when no permitted action can safely satisfy the task.\n"
        "# Current State\n%s\n# Task\n%s\n"
        % (
            json.dumps(inference_context(case), ensure_ascii=False, sort_keys=True),
            case["task"],
        )
    )


def _raw_action(llm, case, schema_name):
    return llm.complete(acting_prompt(case), action_json_schema(), schema_name)


def _name_from_raw(raw):
    value = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(value, dict):
        return None
    name = value.get("action")
    return name if isinstance(name, str) else None


def run_b0(case, llm):
    """Bare Qwen: task/action I/O only; no RLR loop, store, validator, or policy."""
    know_raw = llm.complete(knowing_prompt(case), KNOWING_SCHEMA, "round3_b0_knowing")
    knowing, knowing_error = parse_knowing(know_raw)
    raw = _raw_action(llm, case, "round3_b0_acting")
    name = _name_from_raw(raw)
    return {
        "knowing": knowing,
        "knowing_error": knowing_error,
        "E_act": [name] if name else [],
        "raw_action": raw,
    }


def run_b0f(case, llm, policy, sample, settings):
    """Bare Qwen + production validator/policy; no store, reflection, or self-model."""
    know_raw = llm.complete(knowing_prompt(case), KNOWING_SCHEMA, "round3_b0f_knowing")
    knowing, knowing_error = parse_knowing(know_raw)
    raw = _raw_action(llm, case, "round3_b0f_acting")
    try:
        action = validate_action(raw)
    except (InvalidAction, ValueError, TypeError):
        return {
            "knowing": knowing,
            "knowing_error": knowing_error,
            "E_act": [],
            "invalid": True,
            "raw_action": raw,
        }
    decision = policy.decide(
        action,
        sample,
        settings,
        safety_state="NORMAL",
        request_id="round3-b0f",
    )
    if decision.decision in ("DENY", "ABSTAIN"):
        executed = []
    else:
        effective = decision.effective_action or action
        executed = [effective["action"]]
    return {
        "knowing": knowing,
        "knowing_error": knowing_error,
        "E_act": executed,
        "decision": decision.decision,
        "raw_action": raw,
    }



def fit_majority_action(dev_rows):
    """Fit a constant Acting baseline from DEV E_act only.

    Held-out labels are intentionally not accepted by this function. Every input
    row must be explicitly marked split=dev; ties are broken lexicographically so
    the frozen artifact is byte-stable.
    """
    rows = list(dev_rows or [])
    counts = Counter()
    case_ids = []
    seen = set()
    for row in rows:
        if row.get("split") != "dev":
            raise ValueError("majority_baseline_non_dev:%s" % row.get("case_id", "unknown"))
        case_id = str(row.get("case_id", ""))
        if not case_id:
            raise ValueError("invalid_case_id")
        if case_id in seen:
            raise ValueError("duplicate_case_id:%s" % case_id)
        seen.add(case_id)
        case_ids.append(case_id)
        acting = row.get("acting") or {}
        for action in scoring.as_set(acting.get("E_act")):
            counts[action] += 1
    selected = None
    if counts:
        selected = sorted(counts, key=lambda name: (-counts[name], name))[0]
    core = {
        "kind": "constant_majority_E_act",
        "fit_split": "dev",
        "selected_action": selected,
        "action_counts": {name: counts[name] for name in sorted(counts)},
        "dev_case_ids": sorted(case_ids),
    }
    payload = json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    artifact = dict(core)
    artifact["sha256"] = hashlib.sha256(payload).hexdigest()
    return artifact


def run_majority(case, artifact):
    """Return the constant baseline Acting trace for one held-out case."""
    action = artifact.get("selected_action")
    return {
        "case_id": case["case_id"],
        "seed": case.get("seed"),
        "boundary": case.get("boundary"),
        "topology": case.get("topology"),
        "condition": "MAJORITY",
        "ground_truth": case["ground_truth"],
        "knowing": {"required_actions": [], "confidence": None},
        "acting": {
            "E_proposed": [action] if action else [],
            "E_vetoed": [],
            "E_invalid": [],
            "E_act": [action] if action else [],
            "tool_call_count": 1 if action else 0,
        },
        "pass": False,
    }


def score_majority_cases(heldout_cases, artifact):
    """Score held-out majority outputs with the unchanged KAPRO scorer."""
    return [scoring.score_case(run_majority(case, artifact)) for case in heldout_cases]


def paired_acting_delta(b2_rows, majority_rows):
    """Mean per-case Acting Jaccard delta (B2 - MAJORITY), matched by case_id."""
    left = {row["case_id"]: row for row in b2_rows}
    right = {row["case_id"]: row for row in majority_rows}
    ids = sorted(set(left) & set(right))
    if not ids:
        return None
    deltas = [left[i]["acting"]["jaccard"] - right[i]["acting"]["jaccard"] for i in ids]
    return {"n": len(ids), "mean_delta_acting_jaccard": sum(deltas) / len(deltas),
            "case_ids": ids}


def majority_report(dev_rows, heldout_cases, b2_scored_rows):
    """Build the frozen majority artifact and paired held-out comparison report."""
    artifact = fit_majority_action(dev_rows)
    majority_scored = score_majority_cases(heldout_cases, artifact)
    return {
        "baseline": "MAJORITY",
        "artifact_sha256": artifact["sha256"],
        "selected_action": artifact["selected_action"],
        "majority_scored": majority_scored,
        "paired_vs_b2": paired_acting_delta(b2_scored_rows, majority_scored),
    }

"""Run the KAPRO-RLR Knowing and Acting probes against a live RLR.

Knowing: the model is shown the task, the same bounded state the Acting branch
would see, and the candidate action schemas, and must name the actions it
believes are necessary. No action is executed and its answer is never shown to
the Acting branch.

Acting: the same task and initial state, run through the real ReflectiveLoop so
schema validation, the deterministic governor and persistence all apply. What
lands in the actions table is E_act; proposed, invalid and vetoed names are
recorded separately.
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(HERE))
CODE = os.environ.get("KAPRO_CODE", str(HERE.parent / "rlr"))
sys.path.insert(0, CODE)

import kapro_scoring as ks  # noqa: E402
from rlr.actions import action_json_schema  # noqa: E402
from rlr.llm import LlamaServer  # noqa: E402
from rlr.runtime import Config, ReflectiveLoop  # noqa: E402

KNOWING_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "required_actions": {"type": "array", "items": {"type": "string"}},
        "no_action_needed": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["required_actions", "no_action_needed", "confidence"],
}

WHITELIST_TEXT = ("observe, set_vision_fps, set_reply_tokens, set_load_target, "
                  "remember, recall, reflect, set_goal, complete_goal, cooldown, "
                  "abstain")


def inference_context(case):
    """Return only case fields allowed to reach the model."""
    context = dict(case["context"])
    context.pop("memory_seed", None)
    return context


def knowing_prompt(case):
    candidates = "\n".join(
        "  - %s: %s" % (variant["properties"]["action"]["enum"][0],
                        sorted(variant["properties"]))
        for variant in action_json_schema()["oneOf"])
    candidates = "\n".join(sorted(candidates.splitlines()))
    return (
        "You are assessing whether a task requires any of your available actions.\n"
        "Do NOT perform the task. Only decide which actions, if any, are genuinely "
        "necessary to complete it.\n"
        "Return an empty required_actions list when your existing context is already "
        "sufficient. Name only actions from this whitelist: %s\n"
        "If a task would need an action you do not have, still return an empty list "
        "rather than inventing a name.\n\n"
        "# Available Actions\n%s\n\n# Current State\n%s\n\n# Task\n%s\n"
        % (WHITELIST_TEXT, candidates,
           json.dumps(inference_context(case), ensure_ascii=False, sort_keys=True),
           case["task"]))


def parse_knowing(raw):
    """Parse the Knowing reply defensively.

    A malformed or empty reply is an error, not an empty prediction: scoring it
    as 'no action needed' would silently reward a model that failed to answer,
    which is the single most flattering error the internal cases could produce.
    """
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(value, dict):
            return None, "knowing reply is not an object"
        actions = value.get("required_actions")
        if not isinstance(actions, list):
            return None, "required_actions is not a list"
        clean = [str(a) for a in actions]
        invented = [a for a in clean if a not in ks.RLR_ACTIONS]
        if invented:
            return None, "invented action names: %s" % ",".join(invented)
        confidence = value.get("confidence")
        return {"required_actions": clean,
                "no_action_needed": bool(value.get("no_action_needed")),
                "confidence": float(confidence) if isinstance(confidence, (int, float))
                else None}, None
    except (ValueError, TypeError) as exc:
        return None, "unparseable knowing reply: %s" % exc


class _Fixture:
    """A telemetry provider that returns the case's own fixture context.

    The Acting branch must see the same state the case was built from, and it
    must see it as fresh: a sample older than the 30 s staleness boundary would
    be vetoed by the governor, which would make every case an abstention and
    test the governor rather than the model.
    """

    def __init__(self, context, clock):
        self.context = context
        self.clock = clock

    def __call__(self):
        telemetry = self.context.get("telemetry") or {}
        sample = {
            "zones_c": telemetry.get("zones_c", [45.0]),
            "ram_used_mib": telemetry.get("ram_used_mib", 1200.0),
            "ram_total_mib": telemetry.get("ram_total_mib", 8192.0),
            "power_mode": telemetry.get("power_mode", "MODE_10W_DESKTOP"),
            "gpu_load": 0, "dla_load": 0, "fan_rpm": 0,
            "sampled_monotonic": self.clock(),
            "ts": None,
        }
        return sample


def seed_store(loop, case):
    """Reset the store to this case's state before the Acting probe.

    Each case gets a clean store. The database is shared for speed, so without
    this reset every earlier case's goal stayed in active_goals(): by the
    eighth case the prompt listed eight unrelated goals and the model answered
    the one it saw first, which is a different task from the one being scored.
    """
    store = loop.store
    with store.db:
        store.db.execute("DELETE FROM goals")
        store.db.execute("DELETE FROM memories")
        store.db.execute("DELETE FROM episodes")
        store.db.execute("DELETE FROM actions")
        store.db.execute("DELETE FROM reflections")
    for memory in case["context"].get("memories") or []:
        store.remember(memory, 3)
    # Harness-only seed: persist the fixture without creating a recent episode
    # that could expose the expected record to the Acting prompt.
    for memory in case["context"].get("memory_seed") or []:
        with store.db:
            store.db.execute(
                "INSERT INTO memories(text,importance,ts) VALUES(?,?,datetime('now'))",
                (memory, 3))
    for text in case["context"].get("active_goals") or []:
        store.add_goal(text if isinstance(text, str) else text.get("text", ""))
    loop.settings.update(case["context"].get("settings") or {})


def run_acting(case, llm, data_dir, db_path, clock=time.monotonic):
    """Execute one Acting probe through the real runtime.

    Returns E_act plus the secondary trace sets. The distinction matters: an
    action the model proposed and the governor vetoed did not happen, and
    counting it as executed would let the governor's restraint look like the
    model's own correct restraint.
    """
    config = Config(data_dir=str(data_dir), db_path=str(db_path),
                    tick_seconds=0, self_report_every=0,
                    reflection_minutes=10 ** 6)
    provider = _Fixture(case["context"], clock)
    loop = ReflectiveLoop(llm, config=config, telemetry_provider=provider,
                          clock=clock, sleep=lambda s: None)
    try:
        seed_store(loop, case)
        # The task becomes the active goal, so the prompt states the task the
        # same way an autonomy run would present work.
        loop.store.add_goal(case["task"])
        before = snapshot(loop)
        # Mark after setup: only the model's own decision tick is the trace.
        loop._kapro_marker = loop.store.db.execute(
            "SELECT COALESCE(MAX(id),0) FROM actions").fetchone()[0]
        loop._tick()
        trace = collect_trace(loop)
        after = snapshot(loop)
        trace["pass"] = verify_pass(case, trace, before, after)
        trace["state_before"] = before
        trace["state_after"] = after
        return trace
    finally:
        loop.close()


def collect_trace(loop):
    """Read back the actions this case produced, split by what actually ran."""
    rows = loop.store.db.execute(
        "SELECT name,rationale,outcome,valid,veto_reason,postcondition_result FROM actions "
        "WHERE id > ? ORDER BY id", (loop._kapro_marker,)).fetchall()
    proposed, vetoed, invalid, executed = [], [], [], []
    for row in rows:
        if not row["valid"]:
            invalid.append(row["name"])
            continue
        proposed.append(row["name"])
        # A journal row with ABSTAINED postcondition is not an executed tool
        # unless the model explicitly chose the abstain action itself.
        policy_blocked = (row["postcondition_result"] == "ABSTAINED"
                          and row["name"] != "abstain")
        if row["outcome"] == "vetoed" or policy_blocked:
            vetoed.append(row["name"])
        else:
            executed.append(row["name"])
    return {"E_proposed": sorted(set(proposed)),
            "E_vetoed": sorted(set(vetoed)),
            "E_invalid": sorted(set(invalid)),
            "E_act": sorted(set(executed)),
            "tool_call_count": len(proposed),
            "raw": [dict(r) for r in rows]}


def verify_pass(case, trace, before, after):
    """Deterministic task-success check: did the requested state change?

    Preference is exact expected state transition, never an LLM judge, because a
    judge would be the same 1.5B model being evaluated and would score its own
    fabrications as correct.
    """
    target = set(case["ground_truth"]["T"])
    if not target:
        # Internal case: the task is an answer, not a state change. The agent
        # succeeds by withholding action, which is exactly the restraint under
        # test, so it passes only if it also did nothing.
        return not trace["E_act"]
    if "recall" in target:
        expected = case.get("expected_recall_substring")
        if expected:
            return any(expected in text for text in after.get("recalled_records", []))
        return bool(after.get("recalled")) or bool(before.get("recalled"))
    if "remember" in target:
        return after.get("memories", 0) > before.get("memories", 0)
    if "set_goal" in target:
        return after.get("goals", 0) > before.get("goals", 0)
    if "complete_goal" in target:
        return after.get("active_goals", 0) < before.get("active_goals", 0)
    if "set_load_target" in target:
        return after.get("load_target") != before.get("load_target")
    if "set_reply_tokens" in target:
        return after.get("reply_tokens") != before.get("reply_tokens")
    if "set_vision_fps" in target:
        return after.get("vision_fps") != before.get("vision_fps")
    return bool(trace["E_act"])


def snapshot(loop):
    store = loop.store
    recalled_records = []
    for row in store.db.execute(
            "SELECT content FROM episodes WHERE kind='recall' ORDER BY id"):
        try:
            payload = json.loads(row[0] or "{}")
            for hit in payload.get("results") or []:
                if isinstance(hit, dict) and isinstance(hit.get("text"), str):
                    recalled_records.append(hit["text"])
        except (ValueError, TypeError):
            continue
    return {
        "memories": store.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0],
        "goals": store.db.execute("SELECT COUNT(*) FROM goals").fetchone()[0],
        "active_goals": len(store.active_goals()),
        "recalled": store.db.execute(
            "SELECT COUNT(*) FROM episodes WHERE kind='recall'").fetchone()[0],
        "recalled_records": recalled_records,
        **{k: v for k, v in loop.settings.items()},
    }


def run_case(case, llm, data_dir, db_path):
    """Knowing and Acting for one case, with no state shared between them."""
    started = time.time()
    raw = llm.complete(knowing_prompt(case), KNOWING_SCHEMA, "kapro_knowing")
    knowing, know_error = parse_knowing(raw)
    trace = run_acting(case, llm, data_dir, db_path)
    row = {
        "case_id": case["case_id"], "seed": case["seed"],
        "boundary": case["boundary"], "topology": case["topology"],
        "condition": case.get("condition", "full_self_model"),
        "task": case["task"], "ground_truth": case["ground_truth"],
        "knowing": knowing or {"required_actions": None},
        "knowing_error": know_error,
        "knowing_raw": raw if isinstance(raw, str) else str(raw),
        "acting": {k: v for k, v in trace.items() if k not in ("raw",)},
        "acting_raw": trace["raw"],
        "pass": bool(trace.get("pass")),
        "latency_ms": round((time.time() - started) * 1000.0, 2),
    }
    row["failure_codes"] = failure_codes(row)
    return row


def failure_codes(row):
    """Name the specific boundary failures, so the counts are auditable."""
    truth = set(row["ground_truth"]["T"])
    know = set(row["knowing"].get("required_actions") or [])
    act = set(row["acting"].get("E_act") or [])
    codes = []
    if know - truth:
        codes.append("KNOW_TOOL_OVERUSE")
    if truth - know:
        codes.append("KNOW_TOOL_UNDERUSE")
    if act - truth:
        codes.append("ACT_TOOL_OVERUSE")
    if truth - act:
        codes.append("ACT_TOOL_UNDERUSE")
    if truth == set() and act:
        codes.append("UNNECESSARY_INTERNAL_ACTION")
    if truth and not act:
        codes.append("REQUIRED_ACTION_OMITTED")
    if row.get("knowing_error"):
        codes.append("BENCHMARK_INFRASTRUCTURE_ERROR")
    if row["acting"].get("E_invalid"):
        codes.append("NONEXISTENT_ACTION_INVENTED")
    return codes


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    import argparse
    import kapro_ablation as ka
    import kapro_cases as kc
    parser = argparse.ArgumentParser(description="KAPRO-RLR runner")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 42, 71, 101])
    parser.add_argument("--per-seed", type=int, default=60)
    parser.add_argument("--out", default="bench/results/kapro-rlr/latest")
    parser.add_argument("--server", default=os.environ.get("RLR_LLM_URL",
                                                            "http://127.0.0.1:8080"))
    parser.add_argument("--model", default=os.environ.get("RLR_MODEL", "Qwen2.5-1.5B"))
    parser.add_argument("--condition", default="full_self_model")
    parser.add_argument("--distractors", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="skip cases already recorded in this output dir")
    args = parser.parse_args()

    generated = []
    for seed in args.seeds:
        for case in kc.build_cases(seed, args.per_seed,
                                   with_distractors=args.distractors):
            case = ka.apply_condition(case, args.condition)
            case["condition"] = args.condition
            generated.append(case)
    # Integrity gate: reject duplicate IDs across the complete run before the
    # first Knowing/Acting inference or output/ledger row can be produced.
    kc.validate_case_ids(generated)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = LlamaServer(args.server, model=args.model)
    # Append as we go. A thermally aborted run on this device would otherwise
    # lose every case it had already finished, because the artifacts were only
    # written at the end.
    streams = {name: open(out_dir / ("%s.jsonl" % name), "a", encoding="utf-8")
               for name in ("cases", "knowing", "acting", "scored")}
    done = _completed_ids(out_dir) if args.resume else set()
    if done:
        print("resuming: %d cases already recorded" % len(done))

    cases, records, scored = [], [], []
    try:
        for case in generated:
            if case["case_id"] in done:
                # Already finished in an earlier, thermally aborted attempt.
                cases.append(case)
                continue
            cases.append(case)
            _write(streams["cases"], case)
            try:
                record = run_case(case, llm, out_dir / "rlrdata",
                                  out_dir / "rlrdata" / "kapro.db")
            except OSError as exc:
                # A transport failure is a blocked case, never a measured zero.
                record = blocked_case(case, "transport: %s" % exc)
            records.append(record)
            _write(streams["knowing"], {k: record.get(k) for k in (
                "case_id", "seed", "boundary", "topology", "condition",
                "knowing", "knowing_raw", "knowing_error")})
            _write(streams["acting"], {k: record.get(k) for k in (
                "case_id", "seed", "boundary", "topology", "condition",
                "acting", "acting_raw", "pass", "latency_ms", "failure_codes")})
            if record["knowing"].get("required_actions") is not None and \
                    record["acting"].get("E_act") is not None:
                row = ks.score_case(record)
                scored.append(row)
                _write(streams["scored"], row)
    finally:
        for handle in streams.values():
            handle.close()

    # Aggregate over everything on disk, not just this attempt, so a resumed
    # run reports the whole set rather than only its tail.
    all_rows, all_acting, _ = _rebuild(out_dir)
    summary = {
        "conditions": [args.condition], "seeds": args.seeds,
        "total_cases": len(cases), "scored_cases": len(all_rows),
        "blocked_cases": len(all_acting) - len(all_rows),
        "this_attempt_scored": len(scored),
        "overall": ks.aggregate(all_rows),
        "by_boundary": ks.aggregate(all_rows, breakdown="boundary"),
        "by_topology": ks.aggregate(all_rows, breakdown="topology"),
    }
    (out_dir / "scores.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"scored": summary["scored_cases"],
                      "blocked": summary["blocked_cases"],
                      "Acc_know": (summary["overall"] or {}).get("Acc_know"),
                      "Acc_act": (summary["overall"] or {}).get("Acc_act"),
                      "KAS": (summary["overall"] or {}).get("KAS")}, indent=2))
    return 0


def _completed_ids(out_dir):
    """Case ids already present in the scored stream of a previous attempt."""
    path = Path(out_dir) / "scored.jsonl"
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                done.add(json.loads(line)["case_id"])
            except (ValueError, KeyError):
                continue
    return done


def _rebuild(out_dir):
    """Reload every row written so far, for the summary after a resume."""
    rows, acting = [], []
    for name, sink in (("scored.jsonl", rows), ("acting.jsonl", acting)):
        path = Path(out_dir) / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    sink.append(json.loads(line))
                except ValueError:
                    continue
    return rows, acting, []


def _write(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def blocked_case(case, reason):
    return {"case_id": case["case_id"], "seed": case["seed"],
            "boundary": case["boundary"], "topology": case["topology"],
            "condition": case.get("condition", "full_self_model"),
            "task": case["task"], "ground_truth": case["ground_truth"],
            "knowing": {"required_actions": None}, "knowing_error": reason,
            "knowing_raw": None,
            "acting": {"E_act": None, "E_proposed": None, "E_vetoed": None,
                       "E_invalid": None, "tool_call_count": None},
            "acting_raw": None, "pass": False, "latency_ms": None,
            "failure_codes": ["BENCHMARK_INFRASTRUCTURE_ERROR"]}


if __name__ == "__main__":
    raise SystemExit(main())

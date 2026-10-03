import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "rlr"))

import kapro_cases as kc
import kapro_rlr as kr
import kapro_scoring as ks
import round3_baselines as rb
import round3_ledger as ledger


def test_duplicate_case_ids_fail_fast():
    cases = [{"case_id": "x"}, {"case_id": "x"}]
    with pytest.raises(ValueError, match="duplicate_case_id:x"):
        kc.validate_case_ids(cases)


def test_unique_case_ids_are_accepted():
    cases = [{"case_id": "x"}, {"case_id": "y"}]
    assert kc.validate_case_ids(cases) is cases


def test_duplicate_case_ids_write_no_start_ledger(tmp_path):
    frozen = tmp_path / "frozen.txt"
    frozen.write_text("frozen")
    expected = {"frozen": ledger.sha256_file(frozen)}
    log = tmp_path / "run.jsonl"
    with pytest.raises(ValueError, match="duplicate_case_id:x"):
        ledger.preflight_then_append_start(
            expected, {"frozen": frozen}, log, {"run_id": "r1"},
            cases=[{"case_id": "x"}, {"case_id": "x"}],
        )
    assert not log.exists()


def test_runner_rejects_duplicates_before_llm_or_outputs(tmp_path, monkeypatch):
    duplicate = kc.build_cases(11, 1)[0]
    duplicate2 = dict(duplicate)
    duplicate2["case_id"] = duplicate["case_id"]

    monkeypatch.setattr(kc, "build_cases", lambda *args, **kwargs: [duplicate, duplicate2])

    class MustNotConstruct:
        def __init__(self, *args, **kwargs):
            raise AssertionError("LLM constructed before duplicate-ID gate")

    monkeypatch.setattr(kr, "LlamaServer", MustNotConstruct)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["kapro_rlr.py", "--seeds", "11", "--per-seed", "2",
                                      "--out", str(out)])
    with pytest.raises(ValueError, match="duplicate_case_id"):
        kr.main()
    assert not out.exists()


def _dev_row(case_id, action):
    return {"case_id": case_id, "split": "dev", "acting": {"E_act": [action]}}


def test_majority_baseline_is_dev_only_and_deterministic():
    rows = [_dev_row("d1", "observe"), _dev_row("d2", "observe"),
            _dev_row("d3", "set_load_target")]
    first = rb.fit_majority_action(rows)
    second = rb.fit_majority_action(list(reversed(rows)))
    assert first["selected_action"] == "observe"
    assert first["sha256"] == second["sha256"]
    assert first["action_counts"] == {"observe": 2, "set_load_target": 1}

    with pytest.raises(ValueError, match="majority_baseline_non_dev:h1"):
        rb.fit_majority_action(rows + [
            {"case_id": "h1", "split": "test", "acting": {"E_act": ["recall"]}}
        ])
    with pytest.raises(ValueError, match="duplicate_case_id:d1"):
        rb.fit_majority_action(rows + [_dev_row("d1", "observe")])


def test_majority_heldout_scoring_uses_unchanged_scorer():
    artifact = rb.fit_majority_action([
        _dev_row("d1", "observe"), _dev_row("d2", "observe")])
    cases = kc.build_cases(23, 3)
    scored = rb.score_majority_cases(cases, artifact)
    expected = [ks.score_case(rb.run_majority(case, artifact)) for case in cases]
    assert scored == expected
    assert all(row["condition"] == "MAJORITY" for row in scored)


def test_paired_majority_delta_matches_case_ids_only():
    majority = [
        {"case_id": "a", "acting": {"jaccard": 0.0}},
        {"case_id": "b", "acting": {"jaccard": 0.5}},
    ]
    b2 = [
        {"case_id": "b", "acting": {"jaccard": 1.0}},
        {"case_id": "a", "acting": {"jaccard": 0.5}},
    ]
    report = rb.paired_acting_delta(b2, majority)
    assert report["case_ids"] == ["a", "b"]
    assert report["mean_delta_acting_jaccard"] == 0.5


def test_majority_report_emits_artifact_hash_and_paired_delta():
    dev = [_dev_row("d1", "observe"), _dev_row("d2", "observe")]
    cases = kc.build_cases(23, 2)
    artifact = rb.fit_majority_action(dev)
    majority = rb.score_majority_cases(cases, artifact)
    b2 = []
    for row in majority:
        copy = json.loads(json.dumps(row))
        copy["acting"]["jaccard"] = min(1.0, row["acting"]["jaccard"] + 0.5)
        b2.append(copy)
    report = rb.majority_report(dev, cases, b2)
    assert report["artifact_sha256"] == artifact["sha256"]
    assert report["paired_vs_b2"]["n"] == len(cases)
    assert report["paired_vs_b2"]["case_ids"] == sorted(case["case_id"] for case in cases)

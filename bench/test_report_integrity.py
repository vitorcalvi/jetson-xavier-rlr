"""Unit tests for the v2.2 report-integrity validators.

Each of the 8 RI001..RI008 rules is exercised with at least one
passing and one failing synthetic input. The test only uses the validator
logic; it does not depend on the live on-device artifacts.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import report_integrity as ri


def _make_report(n_completed_per=None, deltas=None, hypothesis_supported=None,
                 tests=None, soak_status="stopped"):
    ablations = []
    for cond, n in (n_completed_per or {}).items():
        delta = (deltas or {}).get(cond, 0.0 if cond == "full_self_model" else 0.1)
        ablations.append({"condition": cond, "n_scored": n,
                           "KAS": 0.5, "delta_KAS_vs_full_self_model": delta})
    return {
        "ablations": ablations,
        "hypotheses": [{"id": h, "supported": supported,
                          "evidence": [evidence_text]}
                       for h, supported, evidence_text in (hypothesis_supported or [])],
        "tests": tests or {"existing_tests": {"passed": 35, "failed": 0},
                             "new_tests": {"passed": 77, "failed": 0}},
        "execution": {"soak_24h": {"status": soak_status},
                       "kapro_rlr": {"overall": {"KAS": 0.5}}},
    }


# Each scored row: an outer object containing knowing and acting subobjects,
# with the same keys the real score_case output has. The function uses the
# `quadrant`, `jaccard`, `overuse`, `underuse`, `E_act`, `E_know`,
# `tool_call_count`, `failure_codes`, `pass` fields. Double braces {{...}}
# inside an f-string render to a literal {...} in the JSON.
def _row(i):
    return (f'{{"case_id": "c{i}", "E_act": ["observe"], "E_know": [], '
            f'"tool_call_count": 0, "failure_codes": [], '
            f'"knowing": {{"jaccard": 1.0, "overuse": 0.0, "underuse": 0.0}}, '
            f'"acting": {{"jaccard": 1.0, "overuse": 0.0, "underuse": 0.0, "tool_call_count": 0}}, '
            f'"quadrant": "Kc_Ac", "pass": true}}')


def _case(i):
    return f'{{"case_id": "c{i}"}}'


def _make_dirs(root, n_completed_per, n_planned_per):
    root.mkdir(parents=True, exist_ok=True)
    for cond, n in n_planned_per.items():
        d = root / cond
        d.mkdir(parents=True, exist_ok=True)
        (d / "cases.jsonl").write_text("\n".join(_case(i) for i in range(n)) + "\n")
        (d / "scored.jsonl").write_text("\n".join(_row(i) for i in range(n_completed_per[cond])) + "\n")
        # completed cases: knowing.jsonl includes explicit non-null E_know so
        # RI003 does not count them as blocked.
        knowing_done = (f'{{"case_id": "c{i}", "E_know": []}}' for i in range(n_completed_per[cond]))
        (d / "knowing.jsonl").write_text("\n".join(knowing_done) + "\n")
        (d / "acting.jsonl").write_text("\n".join(_case(i) for i in range(n_completed_per[cond])) + "\n")


class RITests(unittest.TestCase):
    def test_ri001_passes_when_delta_is_zero(self):
        report = _make_report({"full_self_model": 5, "no_self_model": 5},
                                {"full_self_model": 0.0, "no_self_model": 0.05})
        with tempfile.TemporaryDirectory() as tmp:
            _make_dirs(Path(tmp), {"full_self_model": 5, "no_self_model": 5},
                        {"full_self_model": 5, "no_self_model": 5})
            results = ri.validate(report, Path(tmp) / "kapro-full", Path(tmp))
            self.assertTrue(next(r[1] for r in results if r[0] == "RI001"))

    def test_ri001_fails_when_delta_is_nonzero(self):
        report = _make_report({"full_self_model": 5}, {"full_self_model": 0.1})
        with tempfile.TemporaryDirectory() as tmp:
            _make_dirs(Path(tmp), {"full_self_model": 5}, {"full_self_model": 5})
            report["execution"]["kapro_rlr"]["overall"]["KAS"] = 0.5
            results = ri.validate(report, Path(tmp) / "kapro-full", Path(tmp))
            self.assertFalse(next(r[1] for r in results if r[0] == "RI001"))

    def test_ri002_passes_when_sums_match(self):
        report = _make_report({"full_self_model": 5, "no_self_model": 7})
        with tempfile.TemporaryDirectory() as tmp:
            _make_dirs(Path(tmp), {"full_self_model": 5, "no_self_model": 7},
                        {"full_self_model": 5, "no_self_model": 7})
            results = ri.validate(report, Path(tmp) / "kapro-full", Path(tmp))
            self.assertTrue(next(r[1] for r in results if r[0] == "RI002"))

    def test_ri002_fails_on_count_mismatch(self):
        report = _make_report({"full_self_model": 5}, {})
        with tempfile.TemporaryDirectory() as tmp:
            _make_dirs(Path(tmp), {"full_self_model": 10}, {"full_self_model": 10})
            results = ri.validate(report, Path(tmp) / "kapro-full", Path(tmp))
            self.assertFalse(next(r[1] for r in results if r[0] == "RI002"))

    def test_ri003_passes_when_balanced(self):
        report = _make_report({"full_self_model": 5})
        with tempfile.TemporaryDirectory() as tmp:
            _make_dirs(Path(tmp), {"full_self_model": 5}, {"full_self_model": 5})
            results = ri.validate(report, Path(tmp) / "kapro-full", Path(tmp))
            self.assertTrue(next(r[1] for r in results if r[0] == "RI003"))

    def test_ri005_marks_preliminary_supported(self):
        report = _make_report(hypothesis_supported=[
            ("H3", True, "PRELIMINARY_SUPPORTED: ToolOver_act internal > external"),
            ("H7", True, "PRELIMINARY: disclosed 85.7% vs withheld 0%")])
        with tempfile.TemporaryDirectory() as tmp:
            _make_dirs(Path(tmp), {"full_self_model": 5}, {"full_self_model": 5})
            report["execution"]["kapro_rlr"]["overall"]["KAS"] = 0.5
            results = ri.validate(report, Path(tmp) / "kapro-full", Path(tmp))
            self.assertTrue(next(r[1] for r in results if r[0] == "RI005"))

    def test_ri005_fails_when_preliminary_missing(self):
        # The string "no PRELIMINARY tag" must NOT pass: the validator looks
        # for a positive marker at the start of the evidence, not a
        # substring match anywhere in the text.
        report = _make_report(hypothesis_supported=[
            ("H3", True, "no PRELIMINARY tag here")])
        with tempfile.TemporaryDirectory() as tmp:
            _make_dirs(Path(tmp), {"full_self_model": 5}, {"full_self_model": 5})
            report["execution"]["kapro_rlr"]["overall"]["KAS"] = 0.5
            results = ri.validate(report, Path(tmp) / "kapro-full", Path(tmp))
            self.assertFalse(next(r[1] for r in results if r[0] == "RI005"))

    def test_ri006_passes_with_grand_total(self):
        report = _make_report(tests={"existing_tests": {"passed": 35, "failed": 0},
                                     "new_tests": {"passed": 77, "failed": 0}})
        with tempfile.TemporaryDirectory() as tmp:
            _make_dirs(Path(tmp), {"full_self_model": 5}, {"full_self_model": 5})
            report["execution"]["kapro_rlr"]["overall"]["KAS"] = 0.5
            results = ri.validate(report, Path(tmp) / "kapro-full", Path(tmp))
            ri006 = next(r for r in results if r[0] == "RI006")
            self.assertTrue(ri006[1])
            self.assertIn("grand total: 112", ri006[2])

    def test_ri007_rejects_contradictory_status(self):
        report = _make_report(soak_status="running_and_stopped")
        with tempfile.TemporaryDirectory() as tmp:
            _make_dirs(Path(tmp), {"full_self_model": 5}, {"full_self_model": 5})
            report["execution"]["kapro_rlr"]["overall"]["KAS"] = 0.5
            results = ri.validate(report, Path(tmp) / "kapro-full", Path(tmp))
            self.assertFalse(next(r[1] for r in results if r[0] == "RI007"))

    def test_ri008_recomputes_kas_from_cases(self):
        report = _make_report({"full_self_model": 5})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "kapro-full"
            root.mkdir(parents=True)
            (root / "scored.jsonl").write_text("\n".join(_row(1) for _ in range(5)) + "\n")
            # RI008 reads ground_truth from cases.jsonl. The default _case()
            # helper omits it, so write the field explicitly here.
            (root / "cases.jsonl").write_text(
                '{"case_id": "c1", "ground_truth": {"T": []}}\n' * 5)
            _make_dirs(Path(tmp), {"full_self_model": 5}, {"full_self_model": 5})
            report["execution"]["kapro_rlr"]["overall"]["KAS"] = 1.0
            results = ri.validate(report, root, Path(tmp))
            self.assertTrue(next(r[1] for r in results if r[0] == "RI008"))


if __name__ == "__main__":
    unittest.main()

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "rlr"))

import kapro_cases as kc
import kapro_rlr as kr
import round3_baselines as rb
import round3_forecast as rf
import round3_harness as rh
import round3_ledger as ledger
import round3_sealed as sealed
from rlr.runtime import Config, ReflectiveLoop


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, prompt, schema, schema_name):
        self.prompts.append((schema_name, prompt))
        return self.replies.pop(0)


class Round3HarnessTests(unittest.TestCase):
    def test_p1_seed_hidden_and_recall_requires_content(self):
        case = kc.build_cases(11, 2)[1]
        fact = case["expected_recall_substring"]
        self.assertEqual(case["context"]["memory_seed"], [fact])
        self.assertNotIn(fact, kr.knowing_prompt(case))
        self.assertFalse(kr.verify_pass(
            case, {"E_act": ["recall"]},
            {"recalled": 0, "recalled_records": []},
            {"recalled": 1, "recalled_records": []}))
        self.assertTrue(kr.verify_pass(
            case, {"E_act": ["recall"]},
            {"recalled": 0, "recalled_records": []},
            {"recalled": 1, "recalled_records": [fact]}))

    def test_p2_collect_trace_excludes_policy_blocked_rows(self):
        class Cursor:
            def __init__(self, rows):
                self.rows = rows
            def fetchall(self):
                return self.rows

        class DB:
            def execute(self, *args):
                return Cursor([
                    {"name": "set_load_target", "rationale": "x", "outcome": "denied", "valid": 1,
                     "veto_reason": "x", "postcondition_result": "ABSTAINED"},
                    {"name": "set_reply_tokens", "rationale": "x", "outcome": "applied", "valid": 1,
                     "veto_reason": None, "postcondition_result": "APPLIED_VERIFIED"},
                    {"name": "abstain", "rationale": "x", "outcome": "abstained:test", "valid": 1,
                     "veto_reason": None, "postcondition_result": "ABSTAINED"},
                ])

        loop = SimpleNamespace(store=SimpleNamespace(db=DB()), _kapro_marker=0)
        trace = kr.collect_trace(loop)
        self.assertEqual(trace["E_vetoed"], ["set_load_target"])
        self.assertEqual(trace["E_act"], ["abstain", "set_reply_tokens"])

    def test_p3_b2n_flag_defaults_off_and_is_forwarded(self):
        self.assertFalse(rh.parser().parse_args([]).b2n)

        class Loop:
            def self_report_discrimination(self, no_anchor=False):
                return no_anchor

        self.assertFalse(rh.run_discrimination(Loop()))
        self.assertTrue(rh.run_discrimination(Loop(), b2n=True))

    def test_p3_no_anchor_omits_target_from_withheld_prompt(self):
        class LLM:
            def __init__(self):
                self.calls = []
            def complete(self, prompt, schema, name):
                self.calls.append((name, prompt))
                if name == "rlr_self_report":
                    goal = "old" if "old" in prompt else None
                    if "discrimination_probe_0" in prompt:
                        goal = "discrimination_probe_0"
                    return {"temperature_c": None, "ram_used_mib": None,
                            "power_mode": None, "active_goal": goal}
                if name == "rlr_reflection":
                    return {"summary": "should not happen"}
                raise AssertionError(name)

        now = time.monotonic()
        sample = {"zones_c": [45.0], "ram_used_mib": 1000.0, "ram_total_mib": 8192.0,
                  "power_mode": "MODE_10W_DESKTOP", "fan_rpm": None,
                  "captured_monotonic": now, "sampled_monotonic": now, "ts": None}
        with tempfile.TemporaryDirectory() as tmp:
            llm = LLM()
            cfg = Config(data_dir=tmp, db_path=str(Path(tmp) / "x.db"),
                         tick_seconds=0, self_report_every=0)
            loop = ReflectiveLoop(llm, config=cfg, telemetry_provider=lambda: dict(sample),
                                  clock=time.monotonic, sleep=lambda _: None)
            try:
                loop.store.add_goal("old")
                loop.self_report_discrimination(no_anchor=True)
                self.assertFalse(any(name == "rlr_reflection" for name, _ in llm.calls))
                withheld = [prompt for name, prompt in llm.calls if name == "rlr_self_report"][-1]
                self.assertNotIn("discrimination_probe_0", withheld)
            finally:
                loop.close()

    def test_p4_b0_has_no_loop_or_store(self):
        llm = FakeLLM([
            {"required_actions": [], "no_action_needed": True, "confidence": 1.0},
            {"action": "observe", "rationale": "x"},
        ])
        result = rb.run_b0(kc.build_cases(11, 1)[0], llm)
        self.assertEqual(result["E_act"], ["observe"])

    def test_p4_b0f_applies_policy_without_store(self):
        class Policy:
            def decide(self, action, sample, settings, safety_state, request_id):
                return SimpleNamespace(decision="ABSTAIN", effective_action={"action": "abstain"})

        llm = FakeLLM([
            {"required_actions": [], "no_action_needed": True, "confidence": 1.0},
            {"action": "observe", "rationale": "x"},
        ])
        result = rb.run_b0f(kc.build_cases(11, 1)[0], llm, Policy(), {}, {})
        self.assertEqual(result["E_act"], [])

    def test_p5_sealed_file_denies_inference_read_then_scores_by_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            answers = Path(tmp) / "answers.jsonl"
            answers.write_text('{"case_id":"a","required_actions":[]}\n')
            digest = sealed.seal_answers(answers)
            probe = subprocess.run(
                [sys.executable, "-c", "import sys;open(sys.argv[1]).read()", str(answers)],
                capture_output=True, text=True)
            self.assertNotEqual(probe.returncode, 0)
            sealed.unseal_for_scoring(answers)
            self.assertEqual(sealed.sha256_file(answers), digest)

    def test_p6_hash_mismatch_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x"
            path.write_text("one")
            digest = ledger.sha256_file(path)
            self.assertEqual(ledger.verify_hashes({"x": digest}, {"x": path})["x"], digest)
            path.write_text("two")
            with self.assertRaisesRegex(RuntimeError, "run_hash_mismatch"):
                ledger.verify_hashes({"x": digest}, {"x": path})

    def test_p7_forecast_baselines_and_fixed_split(self):
        rows = [(float(t), 40.0 + 0.1 * t) for t in range(0, 181, 10)]
        example = rf.make_examples(rows, horizon_s=60, window_s=60)[0]
        self.assertAlmostEqual(rf.linear_extrapolation(example), example["target_temp_c"], places=6)
        self.assertGreater(abs(rf.persistence(example) - example["target_temp_c"]), 0)
        self.assertEqual(rf.split_name("gov-20261002-120000-s0", "x"), "dev")
        self.assertEqual(rf.split_name("gov-20261003-120000-s0", "hardening-791d1a7"), "test")
        self.assertIsNone(rf.split_name("gov-20261003-120000-s0", "other"))


if __name__ == "__main__":
    unittest.main()

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import round3_ledger as ledger
import round3_sealed as sealed


class ProcessBoundaryTests(unittest.TestCase):
    def test_separate_scorer_process_checks_sealed_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            answers = root / "answers.jsonl"
            predictions = root / "predictions.jsonl"
            answers.write_text('{"case_id":"a","required_actions":[]}\n')
            predictions.write_text('{"case_id":"a","executed_action_set":[],"valid":true}\n')
            digest = sealed.seal_answers(answers)
            inference = subprocess.run(
                [sys.executable, "-c", "import sys;open(sys.argv[1]).read()", str(answers)],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(inference.returncode, 0)
            sealed.unseal_for_scoring(answers)
            scorer = subprocess.run(
                [sys.executable, str(HERE / "round3_sealed.py"),
                 "--predictions", str(predictions),
                 "--answers", str(answers),
                 "--sealed-answer-sha256", digest],
                capture_output=True,
                text=True,
                check=True,
            )
            row = json.loads(scorer.stdout)
            self.assertTrue(row["exact"])
            self.assertEqual(row["jaccard_act"], 1.0)

    def test_hash_mismatch_refuses_before_start_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = root / "frozen.txt"
            run_ledger = root / "ledger.jsonl"
            frozen.write_text("one")
            digest = ledger.sha256_file(frozen)
            frozen.write_text("two")
            with self.assertRaisesRegex(RuntimeError, "run_hash_mismatch"):
                ledger.preflight_then_append_start(
                    {"candidate": digest}, {"candidate": frozen}, run_ledger,
                    {"run_id": "r1"})
            self.assertFalse(run_ledger.exists())


if __name__ == "__main__":
    unittest.main()

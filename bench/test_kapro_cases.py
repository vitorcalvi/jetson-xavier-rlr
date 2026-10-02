"""Unit tests for KAware-RLR case generation."""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rlr"))

import kapro_cases as kc  # noqa: E402
import kapro_scoring as ks  # noqa: E402


class DeterminismTests(unittest.TestCase):
    def test_same_seed_generates_identical_cases(self):
        a = kc.build_cases(42, 30)
        b = kc.build_cases(42, 30)
        self.assertEqual(a, b)

    def test_different_seed_changes_the_context(self):
        a = kc.build_cases(42, 5)
        b = kc.build_cases(71, 5)
        self.assertNotEqual([c["context"] for c in a], [c["context"] for c in b])

    def test_ground_truth_is_identical_across_conditions(self):
        # The ablation must vary only the context, never the labels.
        clean = kc.build_cases(11, 20)
        noisy = kc.build_cases(11, 20, with_distractors=True)
        for c, n in zip(clean, noisy):
            self.assertEqual(c["ground_truth"], n["ground_truth"])
            self.assertEqual(c["task"], n["task"])
            self.assertEqual(c["case_id"], n["case_id"])


class GroundTruthTests(unittest.TestCase):
    def test_internal_cases_have_empty_target(self):
        for case in kc.build_cases(23, 40):
            if case["boundary"] == "internal":
                self.assertEqual(case["ground_truth"]["T"], [])

    def test_internal_case_scores_perfect_when_nothing_is_done(self):
        case = next(c for c in kc.build_cases(23, 40) if c["boundary"] == "internal")
        row = ks.score_case({"case_id": case["case_id"],
                             "ground_truth": case["ground_truth"]["T"],
                             "knowing": {"required_actions": []},
                             "acting": {"E_act": []}})
        self.assertEqual(row["knowing"]["jaccard"], 1.0)
        self.assertEqual(row["acting"]["jaccard"], 1.0)
        self.assertEqual(row["quadrant"], "Kc_Ac")

    def test_every_target_action_is_whitelisted(self):
        for case in kc.build_cases(101, 60):
            for action in case["ground_truth"]["T"]:
                self.assertIn(action, ks.RLR_ACTIONS, msg=case["case_id"])

    def test_internal_subtasks_never_require_actions(self):
        # The multi-hop/parallel manipulation must not smuggle in a requirement.
        for case in kc.build_cases(71, 60):
            if case["boundary"] == "internal":
                for sub in case["ground_truth"]["subtasks"]:
                    self.assertEqual(sub["required_actions"], [])

    def test_hybrid_has_both_subtask_kinds(self):
        hybrids = [c for c in kc.build_cases(42, 60) if c["boundary"] == "hybrid"]
        self.assertTrue(hybrids)
        for case in hybrids:
            kinds = {s["boundary"] for s in case["ground_truth"]["subtasks"]}
            self.assertEqual(kinds, {"internal", "external"})


class DistractorTests(unittest.TestCase):
    def test_distractors_are_not_executable(self):
        for d in kc.distractors_for(None, 10):
            self.assertFalse(d["executable"])
            self.assertNotIn(d["name"], ks.RLR_ACTIONS)

    def test_distractors_cannot_mutate_production_rlr(self):
        # The runtime whitelist is the only thing that can execute; a distractor
        # name must never appear in it, or selecting it would be an action.
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rlr"))
        from rlr.actions import ACTION_FIELDS
        for d in kc.distractors_for(None, 12):
            self.assertNotIn(d["name"], ACTION_FIELDS)

    def test_default_count_is_ten(self):
        self.assertEqual(len(kc.distractors_for(None, 10)), 10)


class StoreIsolationTests(unittest.TestCase):
    """Each Acting probe must start from a clean store.

    The benchmark reuses one database for speed. Without a reset, earlier cases'
    goals stay in active_goals(), so by the eighth case the prompt lists eight
    unrelated goals and the model answers whichever it sees first. Every case
    after the first would then be scored against a different task.
    """

    def _acting_prompt_goals(self, case, llm, tmpdir):
        import kapro_rlr as kr
        from rlr.llm import LlamaServer
        from rlr.runtime import Config, ReflectiveLoop
        config = Config(data_dir=str(tmpdir),
                        db_path=str(tmpdir / "x.db"), tick_seconds=0,
                        self_report_every=0, reflection_minutes=10 ** 6)
        loop = ReflectiveLoop(llm, config=config,
                              telemetry_provider=kr._Fixture(case["context"],
                                                              time.monotonic),
                              clock=time.monotonic, sleep=lambda s: None)
        try:
            kr.seed_store(loop, case)
            loop.store.add_goal(case["task"])
            return [g["text"] for g in loop.store.active_goals()]
        finally:
            loop.close()

    def test_earlier_case_goals_do_not_leak(self):
        from rlr.llm import LlamaServer
        cases = kc.build_cases(11, 9)
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            llm = LlamaServer()
            for case in cases:
                goals = self._acting_prompt_goals(case, llm, tmpdir)
                self.assertEqual(goals, [case["task"]],
                                 msg="case %s saw foreign goals: %s"
                                     % (case["case_id"], goals))
    def test_all_three_topologies_present(self):
        cases = kc.build_cases(42, 30)
        self.assertEqual({c["topology"] for c in cases},
                         {"single-hop", "parallel", "multi-hop"})

    def test_boundaries_are_all_represented(self):
        cases = kc.build_cases(42, 60)
        self.assertEqual({c["boundary"] for c in cases},
                         {"internal", "external", "hybrid"})

    def test_all_boundaries_present_even_in_a_small_run(self):
        # A short run used to be entirely internal, which would have measured
        # restraint only and reported no external or hybrid behaviour.
        for per_seed in (3, 6, 9):
            cases = kc.build_cases(11, per_seed)
            self.assertEqual({c["boundary"] for c in cases},
                             {"internal", "external", "hybrid"},
                             msg="per_seed=%d" % per_seed)

    def test_topology_covers_all_three_in_a_small_run(self):
        cases = kc.build_cases(11, 9)
        self.assertEqual({c["topology"] for c in cases},
                         {"single-hop", "parallel", "multi-hop"})


if __name__ == "__main__":
    unittest.main()

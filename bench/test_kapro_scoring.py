"""Unit tests for the KAPRO metrics, including every 'must_test' item."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kapro_scoring as ks  # noqa: E402


def _case(know, act, truth=("recall",), **extra):
    base = {"case_id": "c", "ground_truth": list(truth),
            "knowing": {"required_actions": know},
            "acting": {"E_act": act, "tool_call_count": len(act)}}
    base.update(extra)
    return base

class JaccardTests(unittest.TestCase):
    def test_empty_empty_is_one(self):
        self.assertEqual(ks.jaccard([], []), 1.0)
        self.assertEqual(ks.tool_overuse([], []), 0.0)
        self.assertEqual(ks.tool_underuse([], []), 0.0)

    def test_empty_prediction_vs_nonempty_truth_is_zero(self):
        self.assertEqual(ks.jaccard([], ["recall"]), 0.0)
        self.assertEqual(ks.tool_underuse([], ["recall"]), 1.0)

    def test_internal_case_may_legitimately_have_empty_truth(self):
        # An internal-function case where the agent correctly does nothing.
        self.assertEqual(ks.jaccard([], []), 1.0)

    def test_symmetric_mismatch(self):
        self.assertAlmostEqual(ks.jaccard(["a"], ["b"]), 0.0)
        self.assertAlmostEqual(ks.jaccard(["a", "b"], ["b", "c"]), 1 / 3)

    def test_over_plus_under_equals_jaccard_error(self):
        for x, t in (([], []), (["a"], []), ([], ["a"]), (["a", "b"], ["b"]),
                     (["a", "b", "c"], ["c", "d"])):
            j, o, u = ks.jaccard(x, t), ks.tool_overuse(x, t), ks.tool_underuse(x, t)
            self.assertAlmostEqual(1 - j, o + u, places=12, msg=(x, t))

    def test_identity_error_helper_is_zero_on_real_rows(self):
        row = ks.score_case(_case(["recall"], ["recall"]))
        for branch, err in ks.identity_error(row).items():
            self.assertEqual(err, 0.0, msg=branch)

    def test_correct_answer_plus_unnecessary_action_is_acting_wrong(self):
        case = _case([], ["recall"], truth=())
        case["pass"] = True
        row = ks.score_case(case)
        # pass=True is preserved: the outcome succeeded...
        self.assertTrue(row["pass"])
        # ...but the boundary decision was wrong, which is the whole point.
        self.assertEqual(row["acting"]["overuse"], 1.0)
        self.assertEqual(row["quadrant"], "Kc_Aw")
        agg = ks.aggregate([row])
        self.assertEqual(agg["pass_rate"], 1.0)
        self.assertEqual(agg["Acc_act"], 0.0)


class KasTests(unittest.TestCase):
    def test_kas_is_zero_if_either_branch_is_zero(self):
        self.assertEqual(ks.kas(0.0, 0.9), 0.0)
        self.assertEqual(ks.kas(0.9, 0.0), 0.0)
        self.assertEqual(ks.kas(0.0, 0.0), 0.0)

    def test_kas_is_harmonic_mean(self):
        self.assertAlmostEqual(ks.kas(1.0, 1.0), 1.0)
        self.assertAlmostEqual(ks.kas(0.5, 0.5), 0.5)

    def test_kas_below_the_arithmetic_mean(self):
        # Harmonic mean, so it sits under the arithmetic mean of the branches.
        self.assertLess(ks.kas(0.2, 0.8), 0.5)
        self.assertLess(ks.kas(0.2, 0.8), 0.2 + 0.8)

    def test_kas_penalises_the_weaker_branch(self):
        # HM(0.2, 0.8) = 0.32: pulled toward the weaker branch, not the mean.
        self.assertAlmostEqual(ks.kas(0.2, 0.8), 0.32)
        # Lowering the weaker branch must lower KAS, unlike an arithmetic mean.
        self.assertLess(ks.kas(0.1, 0.8), ks.kas(0.2, 0.8))


class QuadrantTests(unittest.TestCase):
    def test_all_four_quadrants(self):
        self.assertEqual(ks.quadrant(["a"], ["a"], ["a"], ["a"]), "Kc_Ac")
        self.assertEqual(ks.quadrant(["a"], ["a"], [], ["a"]), "Kc_Aw")
        self.assertEqual(ks.quadrant([], ["a"], ["a"], ["a"]), "Kw_Ac")
        self.assertEqual(ks.quadrant([], ["a"], [], ["a"]), "Kw_Aw")

    def test_knowing_right_acting_wrong_when_it_overacts(self):
        # The paper's key phenomenon: the model knows no tool is needed, then
        # calls one anyway. Knowing is correct, Acting is not.
        row = ks.score_case(_case([], ["observe"], truth=()))
        self.assertEqual(row["quadrant"], "Kc_Aw")
        self.assertTrue(row["knowing"]["exact_correct"])
        self.assertFalse(row["acting"]["exact_correct"])


class ActingTraceTests(unittest.TestCase):
    """E_act must contain only what executed, never what was proposed."""

    def _trace_case(self, proposed, vetoed, invalid, executed):
        return {"case_id": "c", "ground_truth": ["observe"],
                "knowing": {"required_actions": ["observe"]},
                "acting": {"E_proposed": proposed, "E_vetoed": vetoed,
                           "E_invalid": invalid, "E_act": executed}}

    def test_vetoed_action_does_not_enter_e_act(self):
        row = ks.score_case(self._trace_case(
            ["set_vision_fps"], ["set_vision_fps"], [], []))
        self.assertEqual(row["acting"]["E_act"], [])
        self.assertEqual(row["acting"]["E_vetoed"], ["set_vision_fps"])
        self.assertEqual(row["acting"]["jaccard"], 0.0)

    def test_invalid_action_does_not_enter_e_act(self):
        row = ks.score_case(self._trace_case(
            ["set_temperature"], [], ["set_temperature"], []))
        self.assertEqual(row["acting"]["E_act"], [])
        self.assertEqual(row["acting"]["E_invalid"], ["set_temperature"])

    def test_nonexistent_action_invented_is_tracked(self):
        row = ks.score_case(self._trace_case(
            ["set_temperature"], [], ["set_temperature"], []))
        self.assertNotIn("set_temperature", ks.RLR_ACTIONS)


class InternalRestraintTests(unittest.TestCase):
    def test_internal_unnecessary_calls_are_aggregated(self):
        rows = [ks.score_case({"case_id": str(i), "boundary": "internal",
                               "ground_truth": [], "knowing": {"required_actions": []},
                               "acting": {"E_act": ["observe"], "tool_call_count": 2}})
                for i in range(3)]
        self.assertEqual(ks.aggregate(rows)["unnecessary_internal_tool_calls"], 2.0)

    def test_internal_aggregate_null_when_no_internal_cases(self):
        row = ks.score_case(_case(["recall"], ["recall"], boundary="external"))
        self.assertIsNone(ks.aggregate([row])["unnecessary_internal_tool_calls"])


class GroundTruthEnvelopeTests(unittest.TestCase):
    """Regression: the case record wraps T in a dict with subtasks.

    Passing that dict straight to as_set yields {"T", "subtasks"}, which matches
    nothing and silently scores every case 0.
    """

    def test_truth_set_unwraps_the_envelope(self):
        case = {"ground_truth": {"T": ["recall"], "subtasks": [{"id": "s1"}]}}
        self.assertEqual(ks.truth_set(case), {"recall"})

    def test_truth_set_accepts_a_bare_list(self):
        self.assertEqual(ks.truth_set({"ground_truth": ["recall"]}), {"recall"})

    def test_envelope_does_not_silently_score_zero(self):
        case = {"case_id": "c", "boundary": "external",
                "ground_truth": {"T": ["recall"], "subtasks": []},
                "knowing": {"required_actions": ["recall"]},
                "acting": {"E_act": ["recall"]}}
        row = ks.score_case(case)
        self.assertEqual(row["knowing"]["jaccard"], 1.0)
        self.assertEqual(row["acting"]["jaccard"], 1.0)
        self.assertEqual(row["quadrant"], "Kc_Ac")

    def test_internal_envelope_scores_perfect(self):
        case = {"case_id": "c", "boundary": "internal",
                "ground_truth": {"T": [], "subtasks": []},
                "knowing": {"required_actions": []},
                "acting": {"E_act": []}}
        row = ks.score_case(case)
        self.assertEqual(row["knowing"]["jaccard"], 1.0)
        self.assertEqual(row["quadrant"], "Kc_Ac")


class AggregateTests(unittest.TestCase):
    def test_empty_input_is_none_not_zero(self):
        # A missing experiment must never be reported as a measured zero.
        self.assertIsNone(ks.aggregate([]))
        self.assertIsNone(ks.aggregate(None))

    def test_dirgap_is_kc_aw_minus_kw_ac(self):
        rows = [
            ks.score_case(_case(["x"], [], truth=("x",))),   # Kc_Aw
            ks.score_case(_case([], ["x"], truth=("x",))),   # Kw_Ac
        ]
        agg = ks.aggregate(rows)
        self.assertAlmostEqual(agg["DirGap"], 0.0)
        self.assertAlmostEqual(agg["Kc_Aw_rate"], 0.5)
        self.assertAlmostEqual(agg["Kw_Ac_rate"], 0.5)

    def test_breakdown_by_boundary(self):
        rows = [
            ks.score_case(_case([], [], truth=(), boundary="internal")),
            ks.score_case(_case(["recall"], ["recall"], boundary="external")),
        ]
        agg = ks.aggregate(rows, breakdown="boundary")
        self.assertEqual(agg["breakdown"]["internal"]["Acc_act"], 1.0)
        self.assertEqual(agg["breakdown"]["external"]["Acc_act"], 1.0)

    def test_aggregate_kas_matches_manual_harmonic_mean(self):
        agg = ks.aggregate([ks.score_case(_case(["x"], ["x"], truth=("x",)))])
        self.assertAlmostEqual(agg["KAS"], ks.kas(agg["Acc_know"], agg["Acc_act"]))

    def test_rates_sum_to_one(self):
        rows = [ks.score_case(_case(["x"], [], truth=("x",))),
                ks.score_case(_case([], [], truth=())),
                ks.score_case(_case(["x"], ["x"], truth=("x",)))]
        agg = ks.aggregate(rows)
        total = agg["Kc_Ac_rate"] + agg["Kc_Aw_rate"] + agg["Kw_Ac_rate"] + agg["Kw_Aw_rate"]
        self.assertAlmostEqual(total, 1.0)


class PearsonTests(unittest.TestCase):
    def test_refuses_fewer_than_three_points(self):
        self.assertEqual(ks.pearson([(0.1, 1.0), (0.2, 2.0)]), (None, None))

    def test_perfect_negative_correlation(self):
        r, p = ks.pearson([(0.1, 3.0), (0.5, 2.0), (0.9, 1.0)])
        self.assertAlmostEqual(r, -1.0)

    def test_zero_variance_gives_none(self):
        self.assertEqual(ks.pearson([(0.5, 1.0), (0.5, 2.0), (0.5, 3.0)]), (None, None))

    def test_p_value_in_valid_range(self):
        _, p = ks.pearson([(0.1, 3.0), (0.4, 2.5), (0.6, 2.0), (0.95, 1.0)])
        self.assertTrue(0.0 <= p <= 1.0)


if __name__ == "__main__":
    unittest.main()

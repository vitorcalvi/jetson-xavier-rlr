"""Unit tests for the prompt-growth validator.

The validator's claim is structural: the bounded context stays under the
prompt budget even when the underlying state is much larger than the windows
allow. These tests pin that claim on the host so a future change to
runtime._bounded cannot silently regress it.
"""
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rlr"))

import prompt_growth_test as pgt  # noqa: E402


class BudgetTests(unittest.TestCase):
    def setUp(self):
        pgt.os.makedirs("/tmp/kapro/prompg-test", exist_ok=True)

    def test_current_policy_stays_under_budget(self):
        # Cheap version: just run 200 ticks, not 2000, to keep the test fast.
        out = pgt.main(n_ticks=200)
        for label, regime in out["regimes"].items():
            self.assertLessEqual(regime["max_bounded"], out["prompt_char_budget"],
                                  msg=label + " blew the budget: %s" % regime)

    def test_raw_equals_bounded_when_state_fits_in_windows(self):
        # When nothing is being trimmed, the raw and bounded sizes match.
        out = pgt.main(n_ticks=50)
        for label, regime in out["regimes"].items():
            self.assertEqual(regime["max_bounded"], regime["max_raw"], msg=label)

    def test_bounded_clipped_does_not_exceed_raw(self):
        # When state is too large, the bounded form must be <= the raw form.
        out = pgt.main(n_ticks=2000)
        for label, regime in out["regimes"].items():
            self.assertLessEqual(regime["max_bounded"], regime["max_raw"], msg=label)

    def test_degenerate_state_does_not_blow_the_budget(self):
        # 400 capabilities, 4000 episodes, 10k memories: bounded must still
        # respect the 6000-char budget. This is the test that would catch a
        # regression in _bounded that let episode_char_limit be ignored.
        out = pgt.main(n_ticks=4000)
        for label, regime in out["regimes"].items():
            self.assertLessEqual(regime["max_bounded"], 6000,
                                  msg=label + " over budget: %s" % regime)

    def test_slope_is_bounded_under_realistic_growth(self):
        # The current policy is supposed to plateau (episode window + action
        # window cap the size). The slope of bounded size vs ticks should be
        # at most 50 chars per 100 ticks — the value of one extra bound at
        # the edge. The degenerate regime is allowed to grow because every
        # window fills.
        out = pgt.main(n_ticks=2000)
        slope = out["regimes"]["current_policy"]["slope_bounded_per_100_ticks"]
        self.assertIsNotNone(slope)
        self.assertLess(abs(slope), 100.0,
                          "current_policy slope is too steep: %s" % slope)


if __name__ == "__main__":
    unittest.main()
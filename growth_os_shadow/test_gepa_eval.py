#!/usr/bin/env python3
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gepa_eval import evaluate_offline, online_promotion_gate, run_with_gepa

GOOD = {
  "product": "ava",
  "channel": "bluesky",
  "content": "Seu WhatsApp recebe perguntas enquanto você atende clientes? "
             "A Ava ajuda a organizar as informações de atendimento com base no seu negócio. "
             "Conheça: https://ava.dyagnosys.com/?utm_source=bluesky&utm_campaign=launch2026&utm_content=variantA"
}

class OfflineOptimizerTests(unittest.TestCase):
    def test_good_product_copy_is_offline_acceptable(self):
        result = evaluate_offline(GOOD)
        self.assertTrue(result["accepted"])
        self.assertFalse(result["public_promotion_allowed"])
        self.assertEqual(result["optimization_tier"], "OFFLINE_QA_ONLY")

    def test_unsupported_claim_rejected(self):
        item = dict(GOOD, content=GOOD["content"] + " Vendas garantidas.")
        self.assertFalse(evaluate_offline(item)["accepted"])

    def test_wrong_utm_source_rejected(self):
        item = dict(GOOD, channel="instagram")
        self.assertIn("utm_source_mismatch", evaluate_offline(item)["diagnostics"])

    def test_mythoughs_not_production(self):
        item = dict(GOOD, product="mythoughs")
        self.assertFalse(evaluate_offline(item)["accepted"])

    def test_no_activation_blocks_online_promotion(self):
        self.assertEqual(online_promotion_gate({})["reason"], "ATTRIBUTION_UNVERIFIED")
        self.assertEqual(online_promotion_gate({"synthetic": True})["reason"], "ATTRIBUTION_UNVERIFIED")

    def test_small_real_cohort_is_inconclusive(self):
        evidence = dict(synthetic=False, live_attribution_verified=True,
                        experiment_pre_registered=True, cohorts_mature=True)
        self.assertEqual(online_promotion_gate(evidence)["reason"], "INCONCLUSIVE")

    def test_perfect_evidence_still_requires_approval(self):
        evidence = dict(synthetic=False, live_attribution_verified=True,
                        experiment_pre_registered=True, cohorts_mature=True,
                        power_sufficient=True, product_health_ok=True)
        self.assertEqual(online_promotion_gate(evidence)["reason"], "REQUIRES_APPROVAL")

    def test_gepa_calls_disabled_by_default(self):
        with self.assertRaises(PermissionError):
            run_with_gepa(GOOD, [], [], "a-model", 3)

if __name__ == "__main__":
    unittest.main(verbosity=2)

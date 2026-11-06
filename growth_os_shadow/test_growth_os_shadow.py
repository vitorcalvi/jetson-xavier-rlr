#!/usr/bin/env python3
"""Unit tests for fail-closed Growth OS shadow commander; no network or paid calls."""
import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from growth_os_shadow import ShadowLedger, PROVIDERS


class GrowthShadowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "state.db")
        self.os = ShadowLedger(self.path)

    def tearDown(self):
        self.os.close()
        self.tmp.cleanup()

    def job(self, **kw):
        arg = {"product": "ava", "campaign": "pilot", "variant": "vA", "channel": "bluesky",
               "account": "vitorcalvi", "slot": "20261106T1800Z"}
        arg.update(kw)
        return self.os.enqueue(**arg)

    def test_no_publish_capability_and_global_pause(self):
        state = self.os.status()
        self.assertTrue(state["global_pause"])
        self.assertFalse(state["public_publish_enabled"])
        self.assertEqual(state["external_paid_budget_usd"], 0)
        self.assertFalse(hasattr(self.os, "publish"))
        self.assertFalse(hasattr(self.os, "send"))
        self.assertFalse(state["attribution_e2e_verified"])

    def test_idempotent_enqueue_and_simulation(self):
        a = self.job()
        b = self.job()
        self.assertEqual(a["key"], b["key"])
        self.assertEqual(a["state"], "WAITING_APPROVAL")
        self.assertEqual(self.os.simulate(a["key"]), "SHADOW_TESTED")
        self.assertEqual(self.os.simulate(a["key"]), "SHADOW_TESTED")
        self.assertEqual(self.os.db.execute("SELECT COUNT(*) FROM publish_jobs").fetchone()[0], 1)
        self.assertEqual(self.os.db.execute("SELECT attempts FROM publish_jobs").fetchone()[0], 0)

    def test_provider_ownership(self):
        out = self.job(provider="postspread")
        self.assertEqual(out["state"], "BLOCKED_PRODUCT")
        self.assertEqual(out["reason"], "publisher_ownership_mismatch")

    def test_unknown_product_fails_cleanly(self):
        with self.assertRaisesRegex(ValueError, "unknown_product"):
            self.job(product="mythoughts")

    def test_nonproduction_product_blocked(self):
        out = self.job(product="mythoughs")
        self.assertEqual(out["state"], "BLOCKED_PRODUCT")
        self.assertEqual(out["reason"], "no_verified_production_release")
        self.assertEqual(self.os.simulate(out["key"]), "BLOCKED_PRODUCT")

    def test_reddit_linkedin_human_review(self):
        for channel in ("reddit", "linkedin"):
            out = self.job(channel=channel, provider=PROVIDERS[channel])
            self.assertEqual(out["state"], "HUMAN_REVIEW")
            self.assertEqual(self.os.simulate(out["key"]), "HUMAN_REVIEW")

    def test_fail_closed_zero_budget(self):
        self.assertEqual(self.os.gate("ava", "bluesky", "metricool", None)[1], "zero_spend_policy")
        self.assertEqual(self.os.gate("ava", "bluesky", "metricool", 0.01)[1], "zero_spend_policy")

    def test_synthetic_event_dedup_and_source(self):
        a = self.os.record_event("evt1", "first_open", "app", "ava", "pilot", "vA", "ref01")
        b = self.os.record_event("evt1", "first_open", "app", "ava", "pilot", "vA", "ref01")
        self.assertTrue(a["inserted"])
        self.assertFalse(b["inserted"])
        with self.assertRaises(ValueError):
            self.os.record_event("evt2", "first_open", "cloudflare", "ava")
        with self.assertRaises(ValueError):
            self.os.record_event("evt3", "first_open", "app", "ava", synthetic=False)

    def test_telemetry_has_no_raw_user_identity_fields(self):
        self.os.record_event("evt1", "qualified_activation", "app", "ava", "pilot")
        columns = [x[1] for x in self.os.db.execute("PRAGMA table_info(events)")]
        self.assertFalse(any(x in columns for x in ("ip", "ip_address", "email", "user_id", "device_id", "advertising_id")))

    def test_reboot_recovery_without_state_loss(self):
        key = self.job()["key"]
        self.os.close()
        self.os = ShadowLedger(self.path)
        self.assertEqual(self.os.simulate(key), "SHADOW_TESTED")
        self.assertEqual(self.os.db.execute("SELECT COUNT(*) FROM publish_jobs").fetchone()[0], 1)

    def test_unknown_job_does_not_republish(self):
        with self.assertRaises(ValueError):
            self.os.simulate("missing")

    def test_tick_heartbeat(self):
        out = self.os.tick()
        self.assertTrue(out["heartbeat_utc"])
        self.assertTrue(out["global_pause"])
        self.assertFalse(out["public_publish_enabled"])

    def test_event_rejects_identifiers_with_unsafe_chars(self):
        with self.assertRaises(ValueError):
            self.os.record_event("a@b.com", "first_open", "app", "ava")
        with self.assertRaises(ValueError):
            self.job(account="../secrets")


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Deterministic Governor tests: telemetry safety, caps, and vetoes."""

from __future__ import annotations

import pytest
from conftest import make_config, warm_sample

from reflective_loop_runtime.governor import Governor
from reflective_loop_runtime.store import Store


@pytest.fixture
def governor(tmp_path):
    config = make_config(tmp_path)
    store = Store(config.db_path)
    gov = Governor(store, config, clock=lambda: 1000.0)
    yield gov
    store.close()


class TestThermalBands:
    @pytest.mark.parametrize(
        "temperature,expected",
        [
            (40.0, "normal"),
            (69.9, "normal"),
            (70.0, "conserve"),
            (74.9, "conserve"),
            (75.0, "degraded"),
            (77.9, "degraded"),
            (78.0, "cooldown"),
            (99.0, "cooldown"),
        ],
    )
    def test_bands(self, governor, temperature, expected):
        assert governor.thermal_level(warm_sample(temperature_c=temperature)) == expected

    def test_missing_temperature_is_unknown(self, governor):
        assert governor.thermal_level({"temperature_c": None}) == "unknown"

    def test_none_sample_is_unknown(self, governor):
        assert governor.thermal_level(None) == "unknown"

    def test_hottest_zone_wins(self, governor):
        sample = warm_sample(temperature_c=50.0, zones_c=[50.0, 79.0])
        assert governor.thermal_level(sample) == "cooldown"


class TestStaleness:
    def test_missing_sample_is_stale(self, governor):
        assert governor.stale_reason({}) is not None

    def test_missing_temperature_is_stale(self, governor):
        # A missing temperature (and no zone to fall back on) is unavailable
        # telemetry, so it must be reported stale/unavailable.
        sample = warm_sample(temperature_c=None, zones_c=None)
        assert governor.thermal_level(sample) == "unknown"
        assert governor.stale_reason(sample) is not None

    def test_missing_ram_total_is_stale(self, governor):
        sample = warm_sample(ram_total_mib=None)
        assert governor.stale_reason(sample) is not None

    def test_old_age_seconds_is_stale(self, governor):
        sample = warm_sample(age_seconds=999.0)
        assert governor.stale_reason(sample) is not None

    def test_fresh_sample_is_not_stale(self, governor):
        sample = warm_sample()
        sample["sampled_monotonic"] = 1000.0
        assert governor.stale_reason(sample) is None

    def test_stale_by_monotonic_clock(self, governor):
        # With no explicit age, staleness falls back to the monotonic clock.
        sample = warm_sample(age_seconds=None)
        sample["sampled_monotonic"] = 1000.0 - 120.0  # older than stale_seconds
        assert governor.stale_reason(sample) is not None


class TestVetoes:
    def test_abstain_always_allowed_even_stale(self, governor):
        decision = governor.veto({"action": "abstain", "rationale": "safe", "reason": "wait"}, {})
        assert decision.allowed

    def test_observe_allowed_without_telemetry(self, governor):
        decision = governor.veto({"action": "observe", "rationale": "look"}, {})
        assert decision.allowed

    def test_side_effecting_action_vetoed_on_stale_telemetry(self, governor):
        decision = governor.veto({"action": "cooldown", "rationale": "cool", "seconds": 30}, {})
        assert not decision.allowed
        assert "telemetry" in (decision.reason or "")

    def test_fps_cap_enforced(self, governor):
        sample = warm_sample(temperature_c=76.0)  # degraded
        decision = governor.veto({"action": "set_vision_fps", "rationale": "x", "fps": 5.0}, sample)
        assert not decision.allowed
        ok = governor.veto({"action": "set_vision_fps", "rationale": "x", "fps": 1.0}, sample)
        assert ok.allowed

    def test_tokens_cap_enforced(self, governor):
        sample = warm_sample(temperature_c=71.0)  # conserve
        decision = governor.veto(
            {"action": "set_reply_tokens", "rationale": "x", "tokens": 200}, sample
        )
        assert not decision.allowed

    def test_load_target_blocked_when_degraded(self, governor):
        sample = warm_sample(temperature_c=76.0)
        decision = governor.veto(
            {"action": "set_load_target", "rationale": "x", "load": 0.9}, sample
        )
        assert not decision.allowed

    def test_invalid_action_becomes_abstain_veto(self, governor):
        decision = governor.veto({"action": "rm_rf", "rationale": "x"}, warm_sample())
        assert not decision.allowed
        assert decision.action["action"] == "abstain"

    def test_repeat_veto_blocks_identical_action(self, governor):
        governor.store.record_action("cooldown", {"seconds": 30}, "executed")
        reason = governor.repeat_veto({"action": "cooldown", "rationale": "x", "seconds": 30})
        assert reason is not None

    def test_repeat_veto_allows_different_action(self, governor):
        governor.store.record_action("cooldown", {"seconds": 30}, "executed")
        assert (
            governor.repeat_veto({"action": "set_vision_fps", "rationale": "x", "fps": 1.0}) is None
        )

    def test_repeat_veto_ignores_non_side_effecting(self, governor):
        governor.store.record_action("observe", {}, "executed")
        assert governor.repeat_veto({"action": "observe", "rationale": "x"}) is None

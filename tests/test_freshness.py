"""Freshness must come from source acquisition time, not return time.

A provider can capture a reading, cache it, and only hand it back after a slow
call. If the caller (or the normaliser) stamps the *return* time, that cached
old sample looks fresh and the Governor lets a side effect through. These tests
pin down the contract: the source owns ``acquired_at`` and ``sequence``, and the
Governor judges age from them at decision time.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import pytest
from conftest import make_config, make_scored

from reflective_loop_runtime.governor import Governor
from reflective_loop_runtime.runtime import ReflectiveLoopRuntime
from reflective_loop_runtime.store import Store
from reflective_loop_runtime.telemetry import (
    FileTelemetry,
    StaticTelemetry,
    _normalise,
)


@dataclass
class SlowCachedProvider:
    """Serve one cached reading, only after a deliberately slow call.

    Models a sidecar whose cached value is old but whose read path is slow: the
    return happens later than the acquisition, and the acquisition time has to
    survive the delay instead of being replaced by the return time.
    """

    acquired_at: float
    sequence: int = 1
    delay: float = 0.05
    source: str = "cache"

    def sample(self) -> dict[str, Any]:
        cached = {
            "temperature_c": 55.0,
            "zones_c": [55.0],
            "ram_used_mib": 1024.0,
            "ram_total_mib": 8192.0,
            "acquired_at": self.acquired_at,
            "sequence": self.sequence,
        }
        time.sleep(self.delay)
        return _normalise(
            cached, now_wall=time.time(), now_mono=time.monotonic(), source=self.source
        )


class _FixedDecider:
    """Always select one scripted candidate (mirrors the runtime test fake)."""

    name = "fixed"

    def __init__(self, choice: str, confidence: float):
        self.choice = choice
        self.confidence = confidence

    def score(self, candidates, state, *, context=""):
        return make_scored(candidates, self.choice, self.confidence)


@pytest.fixture
def governor(tmp_path):
    config = make_config(tmp_path)
    store = Store(config.db_path)
    gov = Governor(store, config, clock=lambda: time.monotonic())
    yield gov
    store.close()


def _sample_at(source: str, acquired_at: float, sequence: int) -> dict[str, Any]:
    return _normalise(
        {
            "temperature_c": 55.0,
            "zones_c": [55.0],
            "ram_used_mib": 1024.0,
            "ram_total_mib": 8192.0,
            "acquired_at": acquired_at,
            "sequence": sequence,
        },
        now_wall=0.0,
        now_mono=acquired_at,
        source=source,
    )


class TestSourceAcquisitionMetadata:
    def test_normalise_preserves_source_timestamp_over_caller_clock(self):
        # Source captured at 100.0; a slow caller normalises at 400.0. The
        # sample must keep the acquisition time, never the return time.
        sample = _normalise(
            {"acquired_at": 100.0, "sequence": 1},
            now_wall=0.0,
            now_mono=400.0,
            source="sensor",
        )
        assert sample["acquired_at"] == 100.0
        assert sample["sampled_monotonic"] == 100.0

    def test_normalise_preserves_source_sequence(self):
        sample = _normalise(
            {"acquired_at": 100.0, "sequence": 42},
            now_wall=0.0,
            now_mono=400.0,
            source="sensor",
        )
        assert sample["sequence"] == 42

    def test_normalise_falls_back_when_source_has_no_metadata(self):
        sample = _normalise({}, now_wall=0.0, now_mono=123.0, source="fallback")
        assert sample["acquired_at"] == 123.0
        assert sample["sequence"] is not None

    def test_adapters_stamp_source_metadata(self, tmp_path):
        first = StaticTelemetry().sample()
        second = StaticTelemetry().sample()
        assert first["acquired_at"] == first["sampled_monotonic"]
        assert first["sequence"] is not None
        assert second["sequence"] > first["sequence"]

        path = tmp_path / "sample.json"
        path.write_text(
            json.dumps({"temperature_c": 60.0, "ram_total_mib": 8192.0}),
            encoding="utf-8",
        )
        file_sample = FileTelemetry(path).sample()
        assert file_sample["acquired_at"] == pytest.approx(file_sample["sampled_monotonic"])
        assert file_sample["sequence"] is not None


class TestSlowCachedProvider:
    def test_old_sample_returned_late_stays_old(self):
        provider = SlowCachedProvider(acquired_at=time.monotonic() - 120.0)
        sample = provider.sample()
        # The slow return must not have refreshed the acquisition time.
        assert sample["acquired_at"] <= time.monotonic() - 100.0

    def test_cached_old_sample_is_stale(self, governor):
        provider = SlowCachedProvider(acquired_at=time.monotonic() - 120.0)
        assert governor.stale_reason(provider.sample()) is not None

    def test_cached_old_sample_vetoes_side_effect(self, governor):
        provider = SlowCachedProvider(acquired_at=time.monotonic() - 120.0)
        decision = governor.veto(
            {"action": "cooldown", "rationale": "cool", "seconds": 30},
            provider.sample(),
        )
        assert not decision.allowed
        assert "stale" in (decision.reason or "")

    def test_cached_fresh_sample_is_accepted(self, governor):
        provider = SlowCachedProvider(acquired_at=time.monotonic(), sequence=5)
        assert governor.stale_reason(provider.sample()) is None

    def test_runtime_vetoes_side_effect_from_slow_cached_sample(self, tmp_path):
        config = make_config(tmp_path)
        provider = SlowCachedProvider(acquired_at=time.monotonic() - 120.0)
        runtime = ReflectiveLoopRuntime(
            config,
            decider=_FixedDecider("set_vision_fps", 0.99),
            telemetry=provider,
        )
        try:
            result = runtime.tick(provider.sample(), capabilities={"vision"}, structured=True)
        finally:
            runtime.store.close()
        assert result.vetoed is True
        assert result.execution["action"] == "abstain"


class TestDecisionTimeAge:
    def test_age_is_measured_at_decision_time(self, tmp_path):
        config = make_config(tmp_path)
        store = Store(config.db_path)
        now = {"t": 1000.0}
        try:
            gov = Governor(store, config, clock=lambda: now["t"])
            sample = _sample_at("sensor", acquired_at=1000.0, sequence=1)
            assert gov.stale_reason(sample) is None
            # Same sample, later decision: age must be recomputed, not cached.
            now["t"] = 1000.0 + config.stale_seconds + 5.0
            assert gov.stale_reason(sample) is not None
        finally:
            store.close()

    def test_max_age_can_be_tightened_per_source(self, tmp_path):
        config = make_config(tmp_path)
        config.stale_seconds_by_source = {"fast": 1.0}
        store = Store(config.db_path)
        try:
            gov = Governor(store, config, clock=lambda: 1000.0)
            fast = _sample_at("fast", acquired_at=995.0, sequence=1)
            relaxed = _sample_at("slow", acquired_at=995.0, sequence=1)
            assert gov.stale_reason(fast) is not None  # 5s > 1s
            assert gov.stale_reason(relaxed) is None  # 5s <= 30s default
        finally:
            store.close()


class TestSequenceOrdering:
    def test_advancing_sequence_is_accepted(self, governor):
        base = time.monotonic()
        for sequence in (1, 2, 3):
            assert governor.stale_reason(_sample_at("sensor", base, sequence)) is None

    def test_replayed_sequence_is_rejected(self, governor):
        base = time.monotonic()
        first = _sample_at("sensor", base, sequence=7)
        assert governor.stale_reason(first) is None
        reason = governor.stale_reason(dict(first))
        assert reason is not None
        assert "not newer" in reason

    def test_newer_sequence_does_not_rescue_an_old_sample(self, governor):
        sample = _sample_at("sensor", time.monotonic() - 120.0, sequence=99)
        assert governor.stale_reason(sample) is not None

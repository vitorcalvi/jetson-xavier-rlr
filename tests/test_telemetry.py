"""Pluggable telemetry adapters and their fail-safe behaviour."""

from __future__ import annotations

import json

import pytest

from reflective_loop_runtime.telemetry import (
    FileTelemetry,
    ProcTelemetry,
    StaticTelemetry,
    _normalise,
)


class TestStaticTelemetry:
    def test_canonical_shape(self):
        sample = StaticTelemetry().sample()
        for key in (
            "temperature_c",
            "ram_used_mib",
            "ram_total_mib",
            "zones_c",
            "sampled_at",
            "sampled_monotonic",
            "age_seconds",
            "source",
        ):
            assert key in sample

    def test_old_age_reported(self):
        sample = StaticTelemetry(age_seconds=120.0).sample()
        assert sample["age_seconds"] is None  # normalised from monotonic delta only
        # The monotonic timestamp is back-dated; the Governor derives staleness.
        assert sample["sampled_monotonic"] < sample["sampled_monotonic"] + 1


class TestNormalise:
    def test_missing_values_stay_none(self):
        sample = _normalise({}, now_wall=0.0, now_mono=0.0, source="t")
        assert sample["temperature_c"] is None
        assert sample["ram_total_mib"] is None
        assert sample["zones_c"] == []

    def test_nan_and_inf_become_none(self):
        sample = _normalise(
            {"temperature_c": float("nan"), "ram_used_mib": float("inf")},
            now_wall=0.0,
            now_mono=0.0,
            source="t",
        )
        assert sample["temperature_c"] is None
        assert sample["ram_used_mib"] is None

    def test_zones_scalar_promoted_to_list(self):
        sample = _normalise({"zones_c": 60.0}, now_wall=0.0, now_mono=0.0, source="t")
        assert sample["zones_c"] == [60.0]
        assert sample["temperature_c"] == 60.0


class TestProcTelemetry:
    def test_never_raises_and_may_return_none(self):
        sample = ProcTelemetry(thermal_root="/nonexistent", meminfo_path="/nonexistent").sample()
        assert sample["source"] == "proc"
        assert sample["temperature_c"] is None

    def test_reads_synthetic_thermal_and_meminfo(self, tmp_path):
        zone = tmp_path / "thermal_zone0"
        zone.mkdir()
        (zone / "temp").write_text("65000", encoding="utf-8")
        meminfo = tmp_path / "meminfo"
        meminfo.write_text("MemTotal: 8000000 kB\nMemAvailable: 6000000 kB\n", encoding="utf-8")
        sample = ProcTelemetry(thermal_root=str(tmp_path), meminfo_path=str(meminfo)).sample()
        assert sample["temperature_c"] == 65.0
        assert sample["ram_total_mib"] == pytest.approx(8000000 / 1024, rel=1e-6)
        assert sample["ram_used_mib"] > 0

    def test_excludes_pmic_die_zone(self, tmp_path):
        root = tmp_path
        zones = [
            (root / "thermal_zone0", "CPU-therm", "35000"),
            (root / "thermal_zone1", "PMIC-Die", "50000"),
            (root / "thermal_zone2", "GPU-therm", "33000"),
        ]
        for zone, zone_type, temp in zones:
            zone.mkdir()
            (zone / "type").write_text(zone_type, encoding="utf-8")
            (zone / "temp").write_text(temp, encoding="utf-8")
        sample = ProcTelemetry(thermal_root=str(root), meminfo_path="/nonexistent").sample()
        assert sample["temperature_c"] == 35.0
        assert 50.0 not in sample["zones_c"]
        assert sorted(sample["zones_c"]) == [33.0, 35.0]

    def test_hot_cpu_wins_over_pmic_die(self, tmp_path):
        root = tmp_path
        zones = [
            (root / "thermal_zone0", "CPU-therm", "72000"),
            (root / "thermal_zone1", "PMIC-Die", "50000"),
        ]
        for zone, zone_type, temp in zones:
            zone.mkdir()
            (zone / "type").write_text(zone_type, encoding="utf-8")
            (zone / "temp").write_text(temp, encoding="utf-8")
        sample = ProcTelemetry(thermal_root=str(root), meminfo_path="/nonexistent").sample()
        assert sample["temperature_c"] == 72.0
        assert 50.0 not in sample["zones_c"]
        assert 72.0 in sample["zones_c"]


class TestFileTelemetry:
    def test_missing_file_is_safe(self, tmp_path):
        sample = FileTelemetry(tmp_path / "missing.json").sample()
        assert sample["temperature_c"] is None
        assert sample["source"] == "file"

    def test_reads_sample_and_reports_age(self, tmp_path):
        path = tmp_path / "sample.json"
        path.write_text(
            json.dumps({"temperature_c": 66.0, "ram_total_mib": 8192}), encoding="utf-8"
        )
        sample = FileTelemetry(path).sample()
        assert sample["temperature_c"] == 66.0
        assert sample["age_seconds"] is not None

    def test_non_dict_json_is_safe(self, tmp_path):
        path = tmp_path / "sample.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        sample = FileTelemetry(path).sample()
        assert sample["temperature_c"] is None

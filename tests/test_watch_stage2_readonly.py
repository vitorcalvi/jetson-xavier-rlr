from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path

import pytest

P = Path(__file__).resolve().parents[1] / "scripts" / "watch_stage2_readonly.py"
spec = importlib.util.spec_from_file_location("watch_stage2_readonly", P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def make_layout(tmp_path):
    halt = tmp_path / "HALT_ENTRIES"
    halt.touch()
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    (ledger / "independent_safety_status.json").write_text(
        json.dumps(
            dict(
                state="HALT_ENTRIES",
                mode="READ_ONLY",
                paper_only=True,
                issues=[],
                broker_positions=0,
                broker_open_orders=0,
                utc="2026-10-16T10:00:00+00:00",
            )
        )
    )
    (ledger / "heartbeat.json").write_text(json.dumps({"controller_state": "HALT_ENTRIES"}))
    (ledger / "runtime_tick.json").write_text(
        json.dumps({"ts": datetime.fromisoformat("2026-10-16T10:00:00+00:00").timestamp()})
    )
    proc = tmp_path / "proc"
    (proc / "42" / "fd").mkdir(parents=True)
    (proc / "42" / "status").write_text("Name: python\nVmRSS:\t51200 kB\n")
    (proc / "42" / "fd" / "3").touch()
    sysfs = tmp_path / "thermal"
    (sysfs / "thermal_zone0").mkdir(parents=True)
    (sysfs / "thermal_zone0" / "temp").write_text("45000")
    return halt, ledger, proc, sysfs


def test_sample_proves_halt_broker_zero_thermal_metrics(tmp_path):
    halt, ledger, proc, sysfs = make_layout(tmp_path)
    t = m.sample(
        halt=halt,
        ledger=ledger,
        proc_root=proc,
        thermal_root=sysfs,
        pid_lookup=lambda: 42,
        epoch=lambda: datetime.fromisoformat("2026-10-16T10:00:01+00:00").timestamp(),
    )
    assert t["halt_verified"] and t["broker_positions"] == 0 and t["broker_open_orders"] == 0
    assert t["stage2_rss_mb"] == 50.0 and t["stage2_fd_count"] == 1 and t["max_thermal_c"] == 45


@pytest.mark.parametrize(
    "modify", ["remove_halt", "positions", "orders", "stale", "not_halted", "issues", "wrong_mode"]
)
def test_fail_closed_when_safety_invariant_breaks(tmp_path, modify):
    halt, ledger, proc, sysfs = make_layout(tmp_path)
    if modify == "remove_halt":
        halt.unlink()
    else:
        name = (
            "heartbeat.json"
            if modify == "not_halted"
            else "runtime_tick.json"
            if modify == "stale"
            else "independent_safety_status.json"
        )
        p = ledger / name
        j = json.loads(p.read_text())
        if modify == "positions":
            j["broker_positions"] = 1
        if modify == "orders":
            j["broker_open_orders"] = 1
        if modify == "stale":
            j["ts"] = 0
        if modify == "not_halted":
            j["controller_state"] = "IDLE"
        if modify == "issues":
            j["issues"] = ["unprotected"]
        if modify == "wrong_mode":
            j["mode"] = "LIVE"
        p.write_text(json.dumps(j))
    with pytest.raises(m.SafetyMonitorViolation):
        m.sample(
            halt=halt,
            ledger=ledger,
            proc_root=proc,
            thermal_root=sysfs,
            pid_lookup=lambda: 42,
            epoch=lambda: datetime.fromisoformat("2026-10-16T10:00:01+00:00").timestamp(),
        )


def test_single_sample_file_0600_and_no_double_runs(tmp_path):
    output = tmp_path / "soak.toon"
    record = {
        "stage2_pid": 42,
        "stage2_rss_mb": 50.0,
        "stage2_fd_count": 2,
        "max_thermal_c": 45.0,
        "halt_verified": True,
        "broker_positions": 0,
        "broker_open_orders": 0,
    }
    got = m.monitor(
        duration_hours=48, every_seconds=3600, out=output, once=True, sample_fn=lambda: record
    )
    assert got["samples"] == 1 and got["rss_drift_mb"] == 0
    assert output.stat().st_mode & 0o777 == 0o600
    assert output.read_text().count("sample:") == 1


def test_detects_service_restart_between_samples(tmp_path, monkeypatch):
    vals = iter(
        [
            dict(stage2_pid=10, stage2_rss_mb=20.0, stage2_fd_count=3, max_thermal_c=45.0),
            dict(stage2_pid=11, stage2_rss_mb=20.0, stage2_fd_count=3, max_thermal_c=45.0),
        ]
    )
    out = tmp_path / "s.toon"
    monkeypatch.setattr(m.time, "sleep", lambda _: None)
    ticks = iter([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(m.time, "monotonic", lambda: next(ticks, 0.0))
    with pytest.raises(m.SafetyMonitorViolation, match="restarted"):
        m.monitor(duration_hours=1, every_seconds=1, out=out, sample_fn=lambda: next(vals))


def test_thermal_trip_halts_monitor(tmp_path):
    row = dict(stage2_pid=42, stage2_rss_mb=5.0, stage2_fd_count=1, max_thermal_c=82.0)
    with pytest.raises(m.SafetyMonitorViolation, match="thermal"):
        m.monitor(
            duration_hours=1,
            every_seconds=1,
            out=tmp_path / "thermal.toon",
            once=True,
            sample_fn=lambda: row,
        )

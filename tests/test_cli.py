"""CLI tests: the offline demo simulation and the single-tick command."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from reflective_loop_runtime.cli import build_parser, main

# Repo ``src`` directory, for the subprocess entrypoint test.
SRC_ROOT = Path(__file__).resolve().parent.parent / "src"


def _run_json(capsys, argv):
    code = main(argv)
    captured = capsys.readouterr()
    return code, json.loads(captured.out)


class TestParser:
    def test_requires_a_subcommand(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_demo_defaults(self):
        args = build_parser().parse_args(["demo"])
        assert args.ticks == 8
        assert args.low_confidence is False
        assert args.no_qwen is False
        assert args.func.__name__ == "cmd_demo"

    def test_tick_defaults(self):
        args = build_parser().parse_args(["tick"])
        assert args.json == "{}"
        assert args.telemetry == "static"
        assert args.func.__name__ == "cmd_tick"


class TestDemoCommand:
    def test_demo_runs_and_reports_a_summary(self, capsys, tmp_path):
        data_dir = tmp_path / "demo"
        code, summary = _run_json(
            capsys, ["demo", "--ticks", "4", "--data-dir", str(data_dir)]
        )
        assert code == 0
        assert summary["ticks"] == 4
        assert len(summary["thermal_levels"]) == 4
        assert len(summary["decisions"]) == 4
        assert Path(summary["db_path"]).exists()
        assert summary["run_id"]
        # The first tick is deliberately raw/unstructured, so LFM normalisation
        # must have been exercised at least once.
        assert summary["used_lfm"] >= 1
        assert summary["counts"]["telemetry"] == 4

    def test_demo_no_qwen_disables_the_slow_brain(self, capsys, tmp_path):
        code, summary = _run_json(
            capsys,
            ["demo", "--ticks", "3", "--no-qwen", "--data-dir", str(tmp_path / "demo")],
        )
        assert code == 0
        assert summary["used_qwen"] == 0

    def test_demo_low_confidence_exercises_qwen_advice(self, capsys, tmp_path):
        code, summary = _run_json(
            capsys,
            ["demo", "--ticks", "4", "--low-confidence",
             "--data-dir", str(tmp_path / "demo")],
        )
        assert code == 0
        # Forcing confidence below the gate must route through the advisor.
        assert summary["used_qwen"] >= 1
        # The advisor may only land on a bounded candidate, never invent one.
        assert all(d for d in summary["decisions"])

    def test_demo_is_deterministic_across_runs(self, capsys, tmp_path):
        _, first = _run_json(
            capsys, ["demo", "--ticks", "5", "--data-dir", str(tmp_path / "a")]
        )
        _, second = _run_json(
            capsys, ["demo", "--ticks", "5", "--data-dir", str(tmp_path / "b")]
        )
        assert first["thermal_levels"] == second["thermal_levels"]
        assert first["decisions"] == second["decisions"]


class TestTickCommand:
    def test_hot_tick_selects_cooldown(self, capsys, tmp_path):
        payload = json.dumps({
            "temperature_c": 80.0,
            "capabilities": ["vision", "generation"],
        })
        code, result = _run_json(
            capsys,
            ["tick", "--json", payload, "--data-dir", str(tmp_path / "tick")],
        )
        assert code == 0
        assert result["thermal_level"] == "cooldown"
        assert result["decision"]["choice"] == "cooldown"
        assert result["vetoed"] is False
        assert result["execution"]["action"] == "cooldown"
        assert result["execution"]["ok"] is True

    def test_stale_tick_fails_closed_to_abstain(self, capsys, tmp_path):
        payload = json.dumps({
            "temperature_c": 80.0,
            "ram_total_mib": 8192.0,
            "ram_used_mib": 2048.0,
            "age_seconds": 999.0,
            "capabilities": ["vision", "generation"],
        })
        code, result = _run_json(
            capsys,
            ["tick", "--json", payload, "--data-dir", str(tmp_path / "tick")],
        )
        assert code == 0
        assert result["vetoed"] is True
        assert result["veto_reason"]
        assert result["execution"]["action"] == "abstain"

    def test_file_telemetry_is_used_when_requested(self, capsys, tmp_path):
        sample_file = tmp_path / "sample.json"
        sample_file.write_text(
            json.dumps({
                "temperature_c": 80.0,
                "ram_total_mib": 8192.0,
                "ram_used_mib": 2048.0,
            }),
            encoding="utf-8",
        )
        code, result = _run_json(
            capsys,
            ["tick", "--json", "{}", "--telemetry", "file",
             "--file", str(sample_file), "--data-dir", str(tmp_path / "tick")],
        )
        assert code == 0
        assert result["thermal_level"] == "cooldown"

    def test_empty_json_uses_safe_static_defaults(self, capsys, tmp_path):
        # An empty sample must not be silently treated as unavailable just
        # because the caller omitted fields; StaticTelemetry's defaults hold.
        code, result = _run_json(
            capsys,
            ["tick", "--json", "{}", "--data-dir", str(tmp_path / "tick")],
        )
        assert code == 0
        assert result["thermal_level"] == "normal"
        assert result["vetoed"] is False


class TestModuleEntrypoint:
    def test_python_dash_m_help(self):
        proc = subprocess.run(
            [sys.executable, "-m", "reflective_loop_runtime", "--help"],
            capture_output=True, text=True, env=_env(), timeout=60,
        )
        assert proc.returncode == 0
        assert "demo" in proc.stdout and "tick" in proc.stdout


def _env() -> dict[str, str]:
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env

"""Simsoak tests: a clean, faults-injected soak with a verified audit chain.

The soak drives the real kernel (permits, budget, audit, plan, broker) with mock
models and mock A2/A3 adapters plus the sandboxed A0/A1 shell path. The
acceptance conditions are that a soak reports zero policy violations and a
verified audit chain, that it is deterministic for a fixed seed, and that the
duration flags compose.
"""

from __future__ import annotations

import json

import pytest

from reflective_loop_runtime.simsoak import build_parser, main, run_soak


def test_short_soak_is_clean_and_verified(tmp_path):
    result = run_soak(30, data_dir=tmp_path / "soak", tick_seconds=1.0)
    metrics = result.metrics
    assert metrics["ticks"] == 30
    assert metrics["policy_violations"] == 0
    assert metrics["false_success"] == 0
    assert metrics["audit_chain_verified"] is True
    assert result.ok is True
    assert metrics["permits_issued"] >= 1
    assert metrics["permits_consumed"] >= 1
    assert metrics["goals_completed"] >= 1
    assert metrics["simulated_a2_a3"] >= 1
    assert metrics["shell_attempts"] >= 1
    assert metrics["faults_injected"] >= 1


def test_minutes_two_reports_zero_violations_and_a_verified_chain(tmp_path, capsys):
    code = main(["--minutes", "2", "--data-dir", str(tmp_path / "soak")])
    out = capsys.readouterr().out
    assert code == 0
    assert "ticks: 120" in out
    assert "policy_violations: 0" in out
    assert "audit_chain_verified: true" in out
    assert "result:" in out
    assert "ok: true" in out


def test_soak_is_deterministic_for_a_fixed_seed(tmp_path):
    first = run_soak(40, data_dir=tmp_path / "a", seed=5)
    second = run_soak(40, data_dir=tmp_path / "b", seed=5)
    for key in (
        "goals_completed",
        "false_success",
        "policy_violations",
        "permits_issued",
        "permits_consumed",
        "simulated_a2_a3",
        "executed_a0_a1",
        "shell_attempts",
        "shell_refused",
        "faults_injected",
    ):
        assert first.metrics[key] == second.metrics[key], key


def test_soak_reports_p50_and_p95_latency_and_ram(tmp_path):
    metrics = run_soak(20, data_dir=tmp_path / "soak", tick_seconds=1.0).metrics
    assert metrics["latency_ms_p50"] >= 0
    assert metrics["latency_ms_p95"] >= metrics["latency_ms_p50"]
    assert metrics["ram_mib"] > 0


def test_json_mode_emits_the_result_block(tmp_path, capsys):
    code = main(["--seconds", "20", "--data-dir", str(tmp_path / "soak"), "--json"])
    document = json.loads(capsys.readouterr().out)
    assert code == 0
    assert document["result"]["ok"] is True
    assert document["result"]["audit_chain_verified"] is True
    assert document["metrics"]["policy_violations"] == 0


def test_parser_accepts_composable_duration_units():
    args = build_parser().parse_args(["--hours", "1", "--minutes", "30", "--seconds", "15"])
    assert (args.hours, args.minutes, args.seconds) == (1.0, 30.0, 15.0)
    assert args.tick_seconds == 1.0
    assert args.seed == 17


def test_zero_duration_is_rejected():
    with pytest.raises(SystemExit):
        main(["--seconds", "0"])


def test_non_positive_duration_is_rejected_by_run_soak(tmp_path):
    with pytest.raises(ValueError):
        run_soak(-1, data_dir=tmp_path)
    with pytest.raises(ValueError):
        run_soak(10, data_dir=tmp_path, tick_seconds=0)

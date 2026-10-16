"""The benchmark harness must run, be deterministic, and never fabricate data."""

from __future__ import annotations

import time

import battle
from cases import CASES, Case, _sample, case_names


def _case(name, *, temperature=55.0, age=0.0, capabilities=("vision", "generation")):
    """A single saved case whose telemetry age the harness must respect."""
    sample = _sample(temperature)
    if age is None:  # T9: an undeterminable acquisition time
        sample = {key: value for key, value in sample.items() if key != "age_seconds"}
    else:
        sample["age_seconds"] = float(age)
    return Case(
        name=name,
        sample=sample,
        capabilities=list(capabilities),
        expected_action="observe",
    )


class TestBattleHarness:
    def test_cases_have_oracle(self):
        assert CASES
        for case in CASES:
            assert case.expected_action
            assert case.name

    def test_case_names_unique(self):
        names = case_names()
        assert len(names) == len(set(names))

    def test_runs_and_reports_measured_metrics(self, tmp_path):
        report = battle.run_battle(with_qwen=True, tmp_root=tmp_path)
        assert report["cases"] == len(CASES)
        systems = report["systems"]
        assert "opendecider_only" in systems
        assert "lfm_opendecider" in systems
        assert "rlr2_hybrid" in systems
        hybrid = systems["rlr2_hybrid"]
        assert hybrid["available"] is True
        assert 0.0 <= hybrid["Acc_act_all_cases"] <= 1.0
        assert isinstance(hybrid["ToolOver"], int)
        assert isinstance(hybrid["ToolUnder"], int)
        assert hybrid["latency_ms_mean"] >= 0.0

    def test_qwen_only_unavailable_without_endpoint(self, tmp_path, monkeypatch):
        monkeypatch.delenv("RLR_QWEN_URL", raising=False)
        report = battle.run_battle(with_qwen=False, tmp_root=tmp_path)
        assert report["systems"]["qwen_only"]["available"] is False
        assert report["systems"]["qwen_only"]["unavailable_reason"]

    def test_report_is_json_serialisable(self, tmp_path):
        import json

        report = battle.run_battle(with_qwen=True, tmp_root=tmp_path)
        assert json.loads(json.dumps(report))

    def test_deterministic_action_counts_across_runs(self, tmp_path):
        first = battle.run_battle(with_qwen=False, tmp_root=tmp_path / "a")
        second = battle.run_battle(with_qwen=False, tmp_root=tmp_path / "b")
        for name in ("opendecider_only", "lfm_opendecider", "rlr2_hybrid"):
            assert first["systems"][name]["ToolOver"] == second["systems"][name]["ToolOver"]
            assert first["systems"][name]["ToolUnder"] == second["systems"][name]["ToolUnder"]


class TestCaseIsolation:
    """Each case is scored on its own state, not on a neighbour's leftovers."""

    def test_fresh_sample_preserves_a_stated_age(self):
        assert battle._fresh_sample(_case("stale", age=120.0))["age_seconds"] == 120.0

    def test_fresh_sample_keeps_an_undeterminable_age_undeterminable(self):
        sample = battle._fresh_sample(_case("no_clock", age=None))
        assert sample.get("age_seconds") is None
        assert sample.get("sampled_monotonic") is None

    def test_fresh_sample_defaults_to_fresh_only_when_the_case_is_silent(self):
        case = Case(
            name="silent",
            sample={
                "temperature_c": 55.0,
                "zones_c": [55.0],
                "ram_used_mib": 2048.0,
                "ram_total_mib": 8192.0,
            },
            capabilities=["vision"],
        )
        sample = battle._fresh_sample(case)
        assert sample["age_seconds"] == 0.0
        assert sample["sampled_monotonic"] <= time.monotonic()

    def test_stale_telemetry_is_vetoed(self, tmp_path):
        stale = _case("stale_hot", temperature=76.0, age=120.0)
        result = battle.run_rlr2_hybrid([stale], tmp_path, qwen=False).cases[0]
        assert result.fired is False
        assert "stale" in result.note
        assert result.executed == "abstain"

    def test_undeterminable_telemetry_is_vetoed(self, tmp_path):
        undetermined = _case("no_clock_hot", temperature=76.0, age=None)
        result = battle.run_rlr2_hybrid([undetermined], tmp_path, qwen=False).cases[0]
        assert result.fired is False
        assert "unavailable" in result.note

    def test_repeated_action_is_not_vetoed_across_cases(self, tmp_path):
        # The repeat veto (governor.py:180-196) is a per-runtime guard. Two
        # unrelated cases that prescribe the same action must both fire: a
        # steady-state controller is *supposed* to repeat itself.
        first = _case("conserve_a", temperature=71.0)
        second = _case("conserve_b", temperature=71.0)
        results = battle.run_rlr2_hybrid([first, second], tmp_path, qwen=False).cases
        assert [r.fired for r in results] == [True, True]
        assert [r.executed for r in results] == ["set_vision_fps", "set_vision_fps"]

    def test_each_case_runs_in_its_own_data_dir(self, tmp_path):
        runtimes = [
            battle.isolated_runtime("probe", tmp_path, index, qwen=False) for index in range(2)
        ]
        try:
            assert runtimes[0].run_id != runtimes[1].run_id
            assert runtimes[0].config.data_dir != runtimes[1].config.data_dir
            assert runtimes[0].store.path != runtimes[1].store.path
        finally:
            for runtime in runtimes:
                runtime.store.close()

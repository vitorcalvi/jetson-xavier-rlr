"""The held-out scorer must be honest: frozen set, honest denominators, paired maths."""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

# ``bench`` is not an installed package (the runtime is), so the repo root goes
# on the path here rather than being taken from the working directory. This is
# the same bootstrap ``bench/heldout/cases_heldout.py`` uses for ``bench.cases``.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bench.heldout import score_heldout  # noqa: E402


class TestFrozenManifest:
    def test_scorer_pins_the_frozen_manifest(self):
        assert score_heldout.FROZEN_MANIFEST_SHA256 == (
            "a3867642bf01951847a5b4f90179d5bcfafe9ea5e0d07c3807bbcabc489f87ce"
        )

    def test_frozen_set_still_hashes_to_it(self):
        digest = score_heldout.manifest(list(score_heldout.HELDOUT_CASES))
        assert digest == score_heldout.FROZEN_MANIFEST_SHA256

    def test_a_changed_set_is_refused(self, monkeypatch, tmp_path):
        # Relabel a case with a verdict it does not already carry (the first
        # case expects ``observe``, so writing that back would change nothing):
        # a different set is a different benchmark, not a different score.
        cases = list(score_heldout.HELDOUT_CASES)
        index = next(i for i, case in enumerate(cases) if case.expected_action != "cooldown")
        cases[index] = dataclasses.replace(cases[index], expected_action="cooldown")
        monkeypatch.setattr(score_heldout, "HELDOUT_CASES", cases)
        with pytest.raises(score_heldout.ManifestMismatch):
            score_heldout.score(tmp_root=tmp_path)


class TestStatistics:
    def test_wilson_brackets_the_point_estimate(self):
        low, high = score_heldout.wilson_interval(95, 124)
        assert low < 95 / 124 < high

    def test_wilson_stays_inside_the_unit_interval(self):
        assert score_heldout.wilson_interval(0, 124)[0] == 0.0
        assert score_heldout.wilson_interval(124, 124)[1] == 1.0

    def test_wilson_narrows_as_n_grows(self):
        narrow = score_heldout.wilson_interval(950, 1000)
        wide = score_heldout.wilson_interval(95, 100)
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])

    def test_mcnemar_is_one_without_discordant_pairs(self):
        assert score_heldout.mcnemar_exact(0, 0) == 1.0

    def test_mcnemar_is_symmetric_and_exact(self):
        # 3 vs 0 discordant pairs: p = 2 * (1/2)^3 = 0.25.
        assert score_heldout.mcnemar_exact(3, 0) == pytest.approx(0.25)
        assert score_heldout.mcnemar_exact(0, 3) == pytest.approx(0.25)

    def test_mcnemar_hits_the_binomial_floor(self):
        assert score_heldout.mcnemar_exact(0, 124) == pytest.approx(0.0, abs=1e-12)


class TestScoring:
    @pytest.fixture
    def report(self, tmp_path):
        return score_heldout.score(
            cases=list(score_heldout.HELDOUT_CASES)[:8],
            tmp_root=tmp_path,
        )

    def test_every_system_is_scored_on_every_case(self, report):
        for scored in report["systems"].values():
            assert scored["cases"] == report["cases"]
            assert len(scored["per_case"]) == report["cases"]

    def test_unavailable_system_is_wrong_not_missing(self, report):
        # qwen_only has no endpoint offline: it must still be scored 0/N so the
        # denominator never quietly changes between systems.
        unavailable = report["systems"]["qwen_only"]
        assert unavailable["available"] is False
        assert unavailable["correct"] == 0
        assert unavailable["cases"] == report["cases"]
        assert all(row["correct"] is False for row in unavailable["per_case"])

    def test_unfired_cases_are_wrong(self, report):
        for scored in report["systems"].values():
            for row in scored["per_case"]:
                if not row["fired"]:
                    assert row["correct"] is False

    def test_wilson_and_latency_are_reported(self, report):
        for scored in report["systems"].values():
            if not scored["available"]:
                continue
            low, high = scored["wilson95"]
            assert 0.0 <= low <= scored["accuracy"] <= high <= 1.0
            assert scored["latency_ms_p50"] <= scored["latency_ms_max"]

    def test_paired_cells_are_disjoint_and_exhaustive(self, report):
        for row in report["paired"].values():
            assert (
                row["both_right"] + row["both_wrong"] + row["a_only"] + row["b_only"]
            ) == report["cases"]

    def test_paired_delta_equals_the_difference_in_totals(self, report):
        for row in report["paired"].values():
            assert row["a_only"] - row["b_only"] == row["delta"]
            assert row["delta"] == row["correct"] - row["baseline_correct"]

    def test_baseline_is_paired_against_itself(self, report):
        row = report["paired"][score_heldout.BASELINE]
        assert (row["a_only"], row["b_only"], row["delta"]) == (0, 0, 0)
        assert row["mcnemar_p"] == 1.0

    def test_report_is_json_serialisable(self, report):
        assert json.loads(json.dumps(report))["cases"] == report["cases"]

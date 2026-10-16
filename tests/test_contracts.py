"""Contract tests for action and scored-decision validation."""

from __future__ import annotations

import math

import pytest

from reflective_loop_runtime.contracts import (
    ContractError,
    abstain_action,
    validate_action,
    validate_scored_decision,
)


def _abstain(**extra):
    action = {"action": "abstain", "rationale": "because", "reason": "wait"}
    action.update(extra)
    return action


class TestActionContracts:
    def test_abstain_is_always_valid(self):
        assert validate_action(_abstain())["action"] == "abstain"

    def test_unknown_action_rejected(self):
        with pytest.raises(ContractError):
            validate_action({"action": "shell_exec", "rationale": "nope"})

    def test_missing_field_rejected(self):
        with pytest.raises(ContractError):
            validate_action({"action": "set_vision_fps", "rationale": "x"})

    def test_extra_field_rejected(self):
        action = {"action": "set_vision_fps", "rationale": "x", "fps": 1.0, "sudo": True}
        with pytest.raises(ContractError):
            validate_action(action)

    @pytest.mark.parametrize(
        "action",
        [
            {"action": "set_vision_fps", "rationale": "x", "fps": 999.0},
            {"action": "set_vision_fps", "rationale": "x", "fps": float("inf")},
            {"action": "set_vision_fps", "rationale": "x", "fps": float("nan")},
            {"action": "set_reply_tokens", "rationale": "x", "tokens": 5},
            {"action": "set_reply_tokens", "rationale": "x", "tokens": True},
            {"action": "set_load_target", "rationale": "x", "load": 2.0},
            {"action": "cooldown", "rationale": "x", "seconds": 0},
            {"action": "remember", "rationale": "x", "text": "  ", "importance": 3},
            {"action": "remember", "rationale": "x", "text": "hi", "importance": 99},
            {"action": "complete_goal", "rationale": "x", "goal_id": 0},
        ],
    )
    def test_out_of_range_values_rejected(self, action):
        with pytest.raises(ContractError):
            validate_action(action)

    def test_json_string_accepted(self):
        action = validate_action('{"action": "observe", "rationale": "look"}')
        assert action["action"] == "observe"

    def test_malformed_json_rejected(self):
        with pytest.raises(ContractError):
            validate_action("{not json")

    def test_no_dangerous_action_names_exist(self):
        from reflective_loop_runtime.contracts import ACTION_FIELDS

        forbidden = {
            "shell",
            "exec",
            "run_command",
            "write_file",
            "open_socket",
            "set_clock",
            "set_power",
            "set_fan",
            "write_thermal_guard",
            "set_gpio",
            "reboot",
            "shutdown",
        }
        assert forbidden.isdisjoint(ACTION_FIELDS)

    def test_abstain_helper_is_valid(self):
        action = abstain_action("unsafe")
        assert validate_action(action)["action"] == "abstain"


class TestScoredDecisionContracts:
    IDS = ["observe", "abstain", "cooldown"]

    def test_valid_decision_passes(self):
        scored = validate_scored_decision(
            {
                "choice": "observe",
                "confidence": 0.6,
                "probabilities": {"observe": 0.6, "abstain": 0.3, "cooldown": 0.1},
            },
            self.IDS,
        )
        assert scored.choice == "observe"
        assert math.isclose(sum(scored.probabilities.values()), 1.0, abs_tol=1e-9)

    def test_key_mismatch_rejected(self):
        with pytest.raises(ContractError):
            validate_scored_decision(
                {"choice": "observe", "confidence": 1.0, "probabilities": {"observe": 1.0}},
                self.IDS,
            )

    def test_extra_key_rejected(self):
        with pytest.raises(ContractError):
            validate_scored_decision(
                {
                    "choice": "observe",
                    "confidence": 0.5,
                    "probabilities": {
                        "observe": 0.5,
                        "abstain": 0.25,
                        "cooldown": 0.25,
                        "ghost": 0.0,
                    },
                },
                self.IDS,
            )

    def test_sum_error_rejected(self):
        with pytest.raises(ContractError):
            validate_scored_decision(
                {
                    "choice": "observe",
                    "confidence": 0.5,
                    "probabilities": {"observe": 0.5, "abstain": 0.1, "cooldown": 0.1},
                },
                self.IDS,
            )

    def test_non_finite_probability_rejected(self):
        with pytest.raises(ContractError):
            validate_scored_decision(
                {
                    "choice": "observe",
                    "confidence": float("nan"),
                    "probabilities": {"observe": float("nan"), "abstain": 0.5, "cooldown": 0.5},
                },
                self.IDS,
            )

    def test_confidence_mismatch_rejected(self):
        with pytest.raises(ContractError):
            validate_scored_decision(
                {
                    "choice": "observe",
                    "confidence": 0.9,
                    "probabilities": {"observe": 0.5, "abstain": 0.25, "cooldown": 0.25},
                },
                self.IDS,
            )

    def test_choice_not_in_candidates_rejected(self):
        with pytest.raises(ContractError):
            validate_scored_decision(
                {
                    "choice": "ghost",
                    "confidence": 0.5,
                    "probabilities": {"observe": 0.5, "abstain": 0.25, "cooldown": 0.25},
                },
                self.IDS,
            )

    def test_out_of_range_probability_rejected(self):
        with pytest.raises(ContractError):
            validate_scored_decision(
                {
                    "choice": "observe",
                    "confidence": 1.5,
                    "probabilities": {"observe": 1.5, "abstain": -0.25, "cooldown": -0.25},
                },
                self.IDS,
            )

    def test_empty_candidate_set_rejected(self):
        with pytest.raises(ContractError):
            validate_scored_decision(
                {"choice": "observe", "confidence": 1.0, "probabilities": {}}, []
            )

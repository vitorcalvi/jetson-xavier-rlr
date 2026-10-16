"""Deterministic candidate-generation tests."""

from __future__ import annotations

from conftest import make_config

from reflective_loop_runtime.candidates import (
    build_candidates,
    candidate_by_id,
    candidate_ids,
)


def _state(**overrides):
    state = {
        "thermal_level": "normal",
        "temperature_c": 55.0,
        "ram_used_mib": 2048.0,
        "ram_total_mib": 8192.0,
        "capabilities": ["vision", "generation"],
        "active_goals": [],
        "episode_count": 0,
        "memory_count": 0,
        "cooldown_active": False,
        "goal_digest": "maintain stable operation",
    }
    state.update(overrides)
    return state


class TestCandidateGeneration:
    def test_observe_and_abstain_always_present(self, tmp_path):
        ids = candidate_ids(build_candidates(_state(), config=make_config(tmp_path)))
        assert ids[0] == "observe"
        assert "abstain" in ids
        assert ids.count("abstain") == 1

    def test_ordering_is_stable(self, tmp_path):
        config = make_config(tmp_path)
        first = candidate_ids(build_candidates(_state(), config=config))
        second = candidate_ids(build_candidates(_state(), config=config))
        assert first == second

    def test_vision_and_generation_controls_gated_by_capability(self, tmp_path):
        config = make_config(tmp_path)
        no_caps = candidate_ids(build_candidates(_state(capabilities=[]), config=config))
        assert "set_vision_fps" not in no_caps
        assert "set_reply_tokens" not in no_caps
        with_caps = candidate_ids(build_candidates(_state(), config=config))
        assert "set_vision_fps" in with_caps
        assert "set_reply_tokens" in with_caps

    def test_load_target_requires_capability_and_normal_thermal(self, tmp_path):
        config = make_config(tmp_path)
        caps = ["vision", "generation", "load_control"]
        normal = candidate_ids(build_candidates(_state(capabilities=caps), config=config))
        assert "set_load_target" in normal
        hot = candidate_ids(
            build_candidates(_state(capabilities=caps, thermal_level="degraded"), config=config)
        )
        assert "set_load_target" not in hot

    def test_cooldown_appears_only_under_pressure(self, tmp_path):
        config = make_config(tmp_path)
        cool = candidate_ids(build_candidates(_state(), config=config))
        assert "cooldown" not in cool
        hot = candidate_ids(build_candidates(_state(thermal_level="degraded"), config=config))
        assert "cooldown" in hot
        critical = candidate_ids(build_candidates(_state(thermal_level="cooldown"), config=config))
        assert "cooldown" in critical

    def test_thermal_caps_applied_to_cadence_candidates(self, tmp_path):
        config = make_config(tmp_path)
        conserve = build_candidates(_state(thermal_level="conserve"), config=config)
        fps = candidate_by_id(conserve, "set_vision_fps")
        assert fps is not None and fps.args["fps"] <= config.max_vision_fps_conserve
        tokens = candidate_by_id(conserve, "set_reply_tokens")
        assert tokens is not None and tokens.args["tokens"] <= config.max_reply_tokens_conserve

    def test_no_actuator_candidates_under_cooldown(self, tmp_path):
        config = make_config(tmp_path)
        ids = candidate_ids(build_candidates(_state(thermal_level="cooldown"), config=config))
        assert "set_vision_fps" not in ids
        assert "set_reply_tokens" not in ids
        assert "reflect" not in ids

    def test_memory_candidates_gated_by_counts(self, tmp_path):
        config = make_config(tmp_path)
        none = candidate_ids(build_candidates(_state(), config=config))
        assert "remember" not in none and "recall" not in none
        some = candidate_ids(
            build_candidates(_state(episode_count=3, memory_count=2), config=config)
        )
        assert "remember" in some and "recall" in some

    def test_goal_candidates(self, tmp_path):
        config = make_config(tmp_path)
        goals = [{"id": 7, "text": "keep cool"}]
        ids = candidate_ids(build_candidates(_state(active_goals=goals), config=config))
        assert "set_goal" in ids
        assert "complete_goal:7" in ids

    def test_set_goal_blocked_at_capacity(self, tmp_path):
        config = make_config(tmp_path, max_active_goals=1)
        goals = [{"id": 1, "text": "one"}]
        ids = candidate_ids(build_candidates(_state(active_goals=goals), config=config))
        assert "set_goal" not in ids

    def test_candidates_are_whitelisted_actions(self, tmp_path):
        from reflective_loop_runtime.contracts import ACTION_FIELDS

        config = make_config(tmp_path)
        for candidate in build_candidates(
            _state(
                episode_count=1,
                memory_count=1,
                active_goals=[{"id": 1, "text": "x"}],
                capabilities=["vision", "generation", "load_control"],
            ),
            config=config,
        ):
            assert candidate.action in ACTION_FIELDS
            # Every candidate must materialise into a valid, bounded action.
            from reflective_loop_runtime.contracts import validate_action

            validate_action(candidate.to_action())

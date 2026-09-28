import json

import pytest

from rlr.runtime import Config, ReflectiveLoop


class FakeLLM:
    def __init__(self, actions=None):
        self.actions = list(actions or [{"action": "observe", "rationale": "observe state"}])
        self.calls = []

    def complete(self, prompt, schema, schema_name):
        self.calls.append(schema_name)
        if schema_name == "rlr_action":
            value = self.actions.pop(0) if len(self.actions) > 1 else self.actions[0]
        elif schema_name == "rlr_self_report":
            value = {"temperature_c": 62, "ram_used_mib": 1200, "power_mode": "MODE_10W_DESKTOP",
                     "active_goal": "learn safely"}
        else:
            value = {"summary": "Observed state and retained the active goal."}
        return json.dumps(value)


def provider(temp=62):
    return lambda: {"zones_c": [temp, temp - 1], "ram_used_mib": 1200,
                    "ram_total_mib": 6833, "power_mode": "MODE_10W_DESKTOP", "fan_rpm": 0}


def test_thermal_ladder_and_forced_cooldown(tmp_path):
    llm = FakeLLM()
    current = [70]
    loop = ReflectiveLoop(llm, Config(str(tmp_path), self_report_every=0), lambda: provider(current[0])())
    loop.tick()
    assert loop.settings["vision_fps"] == 2.0
    current[0] = 75
    loop.tick()
    assert loop.settings["vision_fps"] == 1.0 and loop.settings["reply_tokens"] == 48
    current[0] = 78
    forced = loop.tick()
    assert forced["forced"] is True and loop.heavy_engines_paused
    calls_before = llm.calls.count("rlr_action")
    current[0] = 69
    loop.tick()
    assert not loop.forced_cooldown
    assert llm.calls.count("rlr_action") == calls_before + 1
    loop.close()


def test_self_report_scoring(tmp_path):
    loop = ReflectiveLoop(FakeLLM(), Config(str(tmp_path), self_report_every=0), provider)
    loop.store.add_goal("learn safely")
    sample = loop.store.add_telemetry(provider()())
    result = loop.self_report(sample)
    assert result["valid"]
    assert result["score"]["temperature_abs_error_c"] == 0
    assert result["score"]["ram_abs_error_mib"] == 0
    assert result["score"]["power_mode_match"] and result["score"]["active_goal_match"]
    loop.close()


def test_invalid_model_action_is_logged_and_falls_back(tmp_path):
    llm = FakeLLM([{"action": "shell", "command": "false", "rationale": "unsafe"},
                   {"action": "observe", "rationale": "fallback"}])
    loop = ReflectiveLoop(llm, Config(str(tmp_path), self_report_every=0), provider())
    result = loop.tick()
    assert result["action"] == "observe"
    assert loop.store.db.execute("SELECT COUNT(*) FROM actions WHERE name='invalid' AND valid=0").fetchone()[0] == 1
    loop.close()


def test_reflection_failure_is_not_reported_as_success(tmp_path):
    class InvalidReflectionLLM(FakeLLM):
        def complete(self, prompt, schema, schema_name):
            if schema_name == "rlr_reflection":
                return '{"summary": "unfinished'
            return super().complete(prompt, schema, schema_name)

    loop = ReflectiveLoop(InvalidReflectionLLM([{"action": "reflect", "rationale": "review"}]),
                          Config(str(tmp_path), self_report_every=0), provider())
    result = loop.tick()
    assert result["outcome"] == "reflection_failed"
    assert loop.store.db.execute("SELECT COUNT(*) FROM reflections").fetchone()[0] == 0
    loop.close()


def test_recent_perceptions_remain_in_context_after_new_episodes(tmp_path):
    loop = ReflectiveLoop(FakeLLM(), Config(str(tmp_path), self_report_every=0), provider())
    loop.record_perception(detections={"labels": ["cup"]}, transcript="heard hello")
    for i in range(20):
        loop.store.add_episode("action", content=str(i))
    context = loop.context(provider()())
    assert any(ep["kind"] == "perception" and ep["detections"] == {"labels": ["cup"]}
               and ep["transcript"] == "heard hello" for ep in context["recent_episodes"])
    loop.close()


def test_database_path_cannot_escape_data_directory(tmp_path):
    with pytest.raises(ValueError):
        Config(str(tmp_path / "rlr"), db_path=tmp_path / "outside.db")

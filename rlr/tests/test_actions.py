import pytest

from rlr.actions import InvalidAction, action_json_schema, validate_action


def test_schema_has_exact_whitelist_actions():
    assert [v["properties"]["action"]["enum"][0] for v in action_json_schema()["oneOf"]] == [
        "observe", "set_vision_fps", "set_reply_tokens", "remember", "recall", "reflect",
        "set_goal", "complete_goal", "cooldown", "abstain"]


def test_valid_action_and_bounds():
    value = {"action": "set_vision_fps", "fps": 0.2, "rationale": "low load"}
    assert validate_action(value) == value
    with pytest.raises(InvalidAction):
        validate_action({"action": "set_vision_fps", "fps": 5.1, "rationale": "too high"})


def test_rejects_shell_and_extra_fields():
    with pytest.raises(InvalidAction):
        validate_action({"action": "shell", "command": "whoami", "rationale": "x"})
    with pytest.raises(InvalidAction):
        validate_action({"action": "observe", "rationale": "x", "path": "/tmp"})

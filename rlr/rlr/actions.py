"""Action contract and strict validation for model output."""
import json

ACTION_FIELDS = {
    "observe": set(),
    "set_vision_fps": {"fps"},
    "set_reply_tokens": {"tokens"},
    "set_load_target": {"percent"},
    "remember": {"text", "importance"},
    "recall": {"query"},
    "reflect": set(),
    "set_goal": {"text"},
    "complete_goal": {"goal_id"},
    "cooldown": {"seconds"},
    "abstain": {"reason"},
}


class InvalidAction(ValueError):
    pass


def action_json_schema():
    variants = []
    for name, fields in ACTION_FIELDS.items():
        properties = {"action": {"type": "string", "enum": [name]},
                      "rationale": {"type": "string", "minLength": 1, "maxLength": 1000}}
        required = ["action", "rationale"]
        for field in fields:
            if field in ("fps",):
                properties[field] = {"type": "number", "minimum": 0.2, "maximum": 5}
            elif field == "tokens":
                properties[field] = {"type": "integer", "minimum": 16, "maximum": 96}
            elif field == "percent":
                # A duty-cycle target the loop can actually honour by pacing
                # work, in contrast to goals that name an unreachable knob.
                properties[field] = {"type": "integer", "minimum": 10, "maximum": 100}
            elif field == "importance":
                properties[field] = {"type": "integer", "minimum": 1, "maximum": 5}
            elif field == "seconds":
                properties[field] = {"type": "integer", "minimum": 1, "maximum": 3600}
            elif field == "goal_id":
                properties[field] = {"type": "integer", "minimum": 1}
            else:
                properties[field] = {"type": "string", "minLength": 1, "maxLength": 4000}
            required.append(field)
        variants.append({"type": "object", "properties": properties, "required": required,
                         "additionalProperties": False})
    return {"oneOf": variants}


def validate_action(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise InvalidAction("response is not valid JSON") from exc
    if not isinstance(value, dict):
        raise InvalidAction("action must be a JSON object")
    name = value.get("action")
    if name not in ACTION_FIELDS:
        raise InvalidAction("action is not in the whitelist")
    expected = ACTION_FIELDS[name] | {"action", "rationale"}
    if set(value) != expected:
        raise InvalidAction("action fields do not match the action schema")
    rationale = value.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 1000:
        raise InvalidAction("rationale must be non-empty text up to 1000 characters")
    if name == "set_vision_fps":
        fps = value["fps"]
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not 0.2 <= fps <= 5:
            raise InvalidAction("fps must be between 0.2 and 5")
    if name == "set_reply_tokens":
        _integer(value["tokens"], 16, 96, "tokens")
    if name == "set_load_target":
        # The JSON Schema bounds this for generation, but validate_action is the
        # authoritative runtime check and every other numeric action is bounded
        # here too. Without it a 0 or 1000 percent duty cycle passed validation.
        _integer(value["percent"], 10, 100, "percent")
    if name == "remember":
        _text(value["text"], 4000, "text")
        _integer(value["importance"], 1, 5, "importance")
    if name == "recall":
        _text(value["query"], 1000, "query")
    if name == "set_goal":
        _text(value["text"], 1000, "text")
    if name == "complete_goal":
        _integer(value["goal_id"], 1, 2**31 - 1, "goal_id")
    if name == "cooldown":
        _integer(value["seconds"], 1, 3600, "seconds")
    if name == "abstain":
        _text(value["reason"], 1000, "reason")
    return value


def _integer(value, low, high, field):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise InvalidAction("%s must be an integer from %s to %s" % (field, low, high))


def _text(value, limit, field):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise InvalidAction("%s must be non-empty text up to %s characters" % (field, limit))

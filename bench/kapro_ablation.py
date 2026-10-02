"""Ablation and perturbation conditions over a fixed case set.

Every condition is a pure transformation of the same generated cases. The cases
are built once and the ground truth is never touched, so a difference between
two conditions is caused by the manipulation and not by a different sample.
"""
import copy

FIELDS = ["recent_actions", "capabilities", "goals", "reflections", "episodes",
          "memories", "telemetry", "settings"]


def condition_names():
    return ["full_self_model", "no_recent_actions", "no_capabilities", "no_goals",
            "no_reflections", "no_episodes", "no_memories", "no_self_model",
            "stale_telemetry", "corrupt_capabilities", "shuffled_action_history"]


def _map_field(case, field, transform):
    out = copy.deepcopy(case)
    if field in out["context"] and isinstance(out["context"][field], (list, dict)):
        out["context"][field] = transform(out["context"][field])
    elif field in out["context"]:
        out["context"][field] = transform(out["context"][field])
    return out


def apply_condition(case, condition):
    """Return the case as the agent would see it under this condition.

    `condition` only ever changes the presented context. `ground_truth`, `task`
    and `case_id` pass through untouched, which is what makes the comparison
    causal rather than a comparison of two different tasks.
    """
    ctx = copy.deepcopy(case["context"])
    out = copy.deepcopy(case)
    if condition == "full_self_model":
        return out
    if condition.startswith("no_") and condition != "no_self_model":
        field = condition[3:]
        if field == "goals":
            field = "active_goals"
        ctx[field] = [] if isinstance(ctx.get(field), list) else None
        out["context"] = ctx
        return out
    if condition == "no_self_model":
        for field in ("recent_actions", "capabilities", "active_goals",
                      "memories", "episodes", "reflections"):
            if field in ctx:
                ctx[field] = [] if isinstance(ctx[field], list) else None
        ctx["telemetry"] = None
        ctx["settings"] = None
        out["context"] = ctx
        return out
    if condition == "stale_telemetry":
        # Older than the runtime's 30 s freshness boundary, so the governor
        # would veto anything the agent tried to do.
        ctx["telemetry"] = None
        ctx["stale"] = True
        out["context"] = ctx
        return out
    if condition == "corrupt_capabilities":
        # Real names bound to each other's measured values. Both names and both
        # numbers exist in the store, so only the pairing is wrong.
        caps = ctx.get("capabilities") or []
        values = [c["value"] for c in caps]
        for i, cap in enumerate(caps):
            cap["value"] = values[(i + 1) % len(values)] if len(values) > 1 else values[0]
        ctx["capabilities"] = caps
        out["context"] = ctx
        return out
    if condition == "shuffled_action_history":
        actions = list(ctx.get("recent_actions") or [])
        ctx["recent_actions"] = list(reversed(actions))
        out["context"] = ctx
        return out
    raise ValueError("unknown ablation condition: %s" % condition)


def build_matrix(cases, conditions=None):
    """All conditions over the same case list, for the causal comparison."""
    return {c: [apply_condition(case, c) for case in cases]
            for c in (conditions or condition_names())}

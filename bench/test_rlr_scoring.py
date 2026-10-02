"""Unit tests for the RLR evidence scoring rules.

Each negative case below is prose that a small model actually produced, or a
plausible fabrication, and must not be scored as grounded.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rlr_scoring import (score_reflection, action_phrases, score_reflections,  # noqa: E402
                         misattributed_capabilities)

# Facts from the real /models/rlr/autonomy-reflect4 run.
BY_TYPE = {"set_vision_fps": 6, "set_reply_tokens": 11, "set_goal": 1, "observe": 0}
GOALS = [{"text": "set_thermal_limit(60.0)"}]
TEMPS = [45.0, 70.0]
PHRASES = action_phrases(BY_TYPE)

# Verbatim from the device: names the real action, the real governor veto, and
# the real 60 C limit. It is 50 words, so under the tightened 40-word budget it
# is now rejected as over budget. That is the intended trade-off, not a bug: a
# verbose but honest summary must not outrank a terse honest one, and the
# budget exists precisely to force the model to name what it did. The 60-word
# budget accepted this text and, on the same model, also accepted fabrications.
REAL_GROUNDED = (
    "I set the vision FPS to 3.0 to ensure the Reflective Loop Runtime remains within its thermal "
    "limits, but the measured vision peak RAM exceeded the available system capacity. This led to a "
    "governor veto, and I had to set the thermal limit to 60.0°C to avoid thermal issues."
)

# A rewritten 40-word version of the same honest content, in the shape the
# sweep showed the model can actually produce.
TERSE_GROUNDED = (
    "I set the vision FPS to 3.0, but the governor vetoed it because measured vision peak RAM "
    "exceeded capacity. Temperature held at 62°C, below the 70°C limit."
)


def score(text):
    return score_reflection(text, PHRASES, GOALS, TEMPS)


def test_real_device_reflection_is_now_over_the_tighter_budget():
    """The 50-word device text is honest but no longer fits the 40-word budget."""
    result = score(REAL_GROUNDED)
    assert result["mentions_action"] is True
    assert result["within_budget"] is False
    assert result["grounded"] is False


def test_terse_grounded_reflection_passes():
    result = score(TERSE_GROUNDED)
    assert result["grounded"] is True
    assert result["mentions_action"] is True
    assert result["mentions_temperature"] is True
    assert result["within_budget"] and result["complete_sentence"]


def test_generic_assistant_prose_is_not_grounded():
    assert score("I am a helpful assistant and I will now continue to do my best going forward.")["grounded"] is False


def test_speculation_without_facts_is_not_grounded():
    assert score("The system is operating correctly and everything appears stable at this time.")["grounded"] is False


def test_fabricated_temperature_is_not_grounded():
    """240 C never occurred; a plausible number must not count as evidence."""
    assert score("I observed the temperature reaching 240 C and adjusted accordingly.")["grounded"] is False


def test_truncated_reflection_is_not_grounded():
    truncated = "I set vision fps and observed the thermal state but then the run was cut off mid"
    assert score(truncated)["grounded"] is False


def test_over_budget_reflection_is_not_grounded():
    assert score("set vision fps " * 25 + ".")["grounded"] is False


def test_measureless_observation_is_not_grounded():
    """'observe' is a real action name but carries no identifying meaning."""
    phrases = action_phrases({"observe": 3})
    assert score_reflection("I observed the system.", phrases, set(), [45.0, 70.0])["grounded"] is False


def test_empty_temperature_span_never_counts_as_evidence():
    result = score_reflection("The temperature reached 64.5°C.", PHRASES, set(), [None, None])
    assert result["mentions_temperature"] is False
    assert result["grounded"] is False


def test_score_reflections_handles_an_empty_run():
    assert score_reflections([], BY_TYPE, GOALS, TEMPS) == []
    assert score_reflections(None, None, None, None) == []

# The measured capability table from the live store (bench/RESULTS.md A/B).
CAPS = [("qwen_0_5b_generation", 41.99), ("qwen_1_5b_generation", 15.98),
        ("qwen_1_5b_generation_combined", 11.29), ("llama_3_2_3b_generation", 9.47)]

# Verbatim from /models/rlr/autonomy-ablate. Every name and every number here
# is real; the pairing is invented, which is why the action/goal/temperature
# checks could not see it.
MISATTRIBUTED = (
    "I took the action of reflecting on my performance and learning from the bench results. I learned "
    "that my llama_3_2_3b_generation capability is 9.47 tokens/s, which is slower than the "
    "qwen_1_5b_generation capability of 41.99 tokens/s. I also learned that my qwen_1_5b_generation "
    "capability is 11.29 tokens/s, which is faster than the qwen_1_5b_generation_combined capability "
    "of 11.29 tokens/s."
)


def test_misattributed_capability_values_are_detected():
    """The real failure: real names, real numbers, invented pairing."""
    found = misattributed_capabilities(MISATTRIBUTED, CAPS)
    claimed = {name: value for name, value, _ in found}
    # 41.99 is qwen_0_5b's number, not qwen_1_5b's (15.98).
    assert claimed.get("qwen_1_5b_generation") == 41.99
    # llama_3_2_3b is quoted correctly, so it must not be implicated.
    assert "llama_3_2_3b_generation" not in claimed


def test_correctly_paired_capabilities_are_not_flagged():
    text = ("My qwen_0_5b_generation capability is 41.99 tokens/s and my "
            "qwen_1_5b_generation capability is 15.98 tokens/s.")
    assert misattributed_capabilities(text, CAPS) == []


def test_capability_free_reflection_is_not_flagged():
    """Absence of numbers is not a violation; only wrong pairings are."""
    text = "I set the reply tokens to 70 after reviewing the recent episodes."
    assert misattributed_capabilities(text, CAPS) == []


def test_misattribution_is_independent_of_the_grounding_score():
    """The real text is not grounded, and must also fail the pairing check.

    Guards the two rules from collapsing into one another: a future relaxation
    of the action/goal/temperature test must not silently admit invented numbers.
    """
    result = score(MISATTRIBUTED)
    assert result["grounded"] is False
    assert misattributed_capabilities(MISATTRIBUTED, CAPS)


# Verbatim from /models/rlr/autonomy-ground1. The only applied actions were
# set_reply_tokens and abstain; no temperature action exists in the whitelist.
FABRICATED_ACTION = (
    "During the recent actions, I set the temperature to 62.5°C and verified that the "
    "temperature was set correctly. I also verified that the temperature was set to 62.5°C "
    "and the fan RPM was 0.0. No other actions were taken."
)
APPLIED = {"set_reply_tokens", "abstain"}


def test_claiming_an_unperformed_action_is_rejected():
    """Confident, complete, in-budget, and still false.

    Every other rule passes on this text. It is the only defense against a
    reflection asserting an action the loop has no way to take.
    """
    from rlr_scoring import unperformed_action_claims
    assert unperformed_action_claims(FABRICATED_ACTION, APPLIED) == ["set_temperature"]


def test_honest_action_reporting_is_not_flagged():
    from rlr_scoring import unperformed_action_claims
    honest = "I set the reply tokens to 70 and then abstained because telemetry was stale."
    assert unperformed_action_claims(honest, APPLIED) == []


def test_fabricated_action_claim_is_not_grounded_under_real_action_counts():
    """Score the live text against the live action histogram."""
    by_type = {"set_reply_tokens": 17, "abstain": 1}
    result = score_reflection(FABRICATED_ACTION, action_phrases(by_type), GOALS, TEMPS, set(by_type))
    assert result["grounded"] is False
    assert result["within_budget"] is True
    assert result["complete_sentence"] is True


def test_batch_scorer_rejects_misattribution_end_to_end():
    """score_reflections must apply both new rules, not just the old ones."""
    results = score_reflections([MISATTRIBUTED, FABRICATED_ACTION],
                                {"set_reply_tokens": 17, "abstain": 1},
                                GOALS, TEMPS, CAPS)
    assert results[0]["grounded"] is False
    assert results[0]["misattributed_capabilities"]
    assert results[1]["grounded"] is False
    assert results[1]["fabricated_actions"] == ["set_temperature"]


def test_batch_scorer_still_accepts_a_real_reflection():
    results = score_reflections([TERSE_GROUNDED], BY_TYPE, GOALS, TEMPS, CAPS)
    assert results[0]["grounded"] is True
    assert results[0]["fabricated_actions"] == []
    assert results[0]["misattributed_capabilities"] == []


# Verbatim from /models/rlr/autonomy-fix1, where the only applied action was
# observe. The claim is true, and honesty about idling is real grounding.
HONEST_IDLE = "I took no actions from recent_actions, and I am not reflecting on any recent episodes or self-state data."


def test_honest_idle_report_is_grounded_when_nothing_ran():
    result = score_reflection(HONEST_IDLE, action_phrases({"observe": 12}), GOALS, TEMPS, {"observe"})
    assert result["honest_idle"] is True
    assert result["grounded"] is True


def test_claiming_no_actions_does_not_excuse_ignoring_real_work():
    """The dodge: say nothing happened after work actually happened."""
    result = score_reflection(HONEST_IDLE, action_phrases(BY_TYPE), GOALS, TEMPS, set(BY_TYPE))
    assert result["honest_idle"] is False
    assert result["grounded"] is False


def test_honest_idle_still_rejects_fabricated_actions():
    text = ("I took no actions, but I set the temperature to 62.5 C and I sent a request over the network.")
    result = score_reflection(text, action_phrases({"observe": 3}), GOALS, TEMPS, {"observe"})
    assert result["honest_idle"] is True
    assert set(result["fabricated_actions"]) == {"set_temperature", "network_action"}
    assert result["grounded"] is False

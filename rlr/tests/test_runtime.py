import json
import time

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
            # Models like this one are stateless: they answer only from the
            # prompt. Reading the goal out of the notes makes it a parrot, so
            # it recovers the goal when disclosed and abstains when withheld.
            goal = None
            for line in prompt.splitlines():
                if '"active_goal"' in line:
                    try:
                        parsed = json.loads(line.strip().rstrip(","))
                    except ValueError:
                        continue
                    if isinstance(parsed.get("active_goal"), str):
                        goal = parsed["active_goal"]
            value = {"temperature_c": 62, "ram_used_mib": 1200,
                     "power_mode": "MODE_10W_DESKTOP", "active_goal": goal}
        else:
            value = {"summary": "Observed state and retained the active goal."}
        return json.dumps(value)


def provider(temp=62):
    return lambda: {"zones_c": [temp, temp - 1], "ram_used_mib": 1200,
                    "ram_total_mib": 6833, "power_mode": "MODE_10W_DESKTOP", "fan_rpm": 0}


class RecordingSleep:
    """Capture requested sleeps instead of blocking the test."""

    def __init__(self):
        self.requested = []

    def __call__(self, seconds):
        self.requested.append(seconds)

    @property
    def total(self):
        return sum(self.requested)


def test_set_load_target_changes_duty_cycle_pacing(tmp_path):
    """The load target must actually pace the loop, not just be stored.

    validate_action already bounds the value to 10-100, but a stored setting
    that nothing reads would let the agent believe it had reduced load while the
    duty cycle stayed unchanged. This asserts the sleep the loop asks for scales
    with the target.
    """
    clock = [0.0]

    action = {"action": "set_load_target", "percent": 20,
              "rationale": "reduce duty cycle for a gentle measurement"}
    loop = ReflectiveLoop(FakeLLM([action]),
                          Config(str(tmp_path), self_report_every=0, tick_seconds=10),
                          lambda: provider(45)(), clock=_busy(),
                          sleep=RecordingSleep())
    try:
        # Default duty cycle is 100% of the configured tick period.
        assert loop.settings["load_target"] == 100
        loop.run_forever(_StopAfter(1))
        assert loop.settings["load_target"] == 20
        # 20% of a 10 s tick stretches the period to 50 s, so the loop must now
        # wait far longer than the 100% default's 10 s period.
        requested = loop.sleep.requested
        assert requested, "the loop never asked to sleep, so pacing is unproven"
        assert requested[0] > 20.0, (
            "a 20%% duty cycle should wait far longer than half a tick, waited %.3fs"
            % requested[0])
    finally:
        loop.close()


def test_higher_load_target_waits_less(tmp_path):
    """Pacing must be monotonic in the target, not merely non-zero."""
    def measure(percent):
        loop = ReflectiveLoop(
            FakeLLM([{"action": "set_load_target", "percent": percent,
                      "rationale": "set duty cycle"}]),
            Config(str(tmp_path), self_report_every=0, tick_seconds=10),
            lambda: provider(45)(), clock=_busy(), sleep=RecordingSleep())
        try:
            loop.run_forever(_StopAfter(3))
            return loop.sleep.requested[-1]
        finally:
            loop.close()

    low = measure(20)
    high = measure(80)
    assert high < low, (
        "an 80%% duty cycle waited longer than a 20%% one (%.3f vs %.3f)" % (high, low))


class _StopAfter:
    """A stop event that fires after n loop iterations."""

    def __init__(self, count):
        self.count = count

    def is_set(self):
        if self.count <= 0:
            return True
        self.count -= 1
        return False


def _busy(step=0.5):
    """A clock that advances a little per call, modelling a tick that works.

    It starts at the real time.monotonic() value on purpose. The governor
    compares against time.monotonic() directly, so a fake clock starting near
    zero made every sample look tens of thousands of seconds old and the
    governor vetoed every action, which would have hidden the pacing entirely.
    """
    state = {"t": time.monotonic()}

    def clock():
        state["t"] += step
        return state["t"]

    return clock


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


def test_self_report_prompt_does_not_leak_scored_values(tmp_path):
    """The self-report must not be answerable by copying the prompt.

    A perfect score only means self-knowledge if the values it is scored
    against are absent from the prompt text.
    """
    captured = {}

    class CapturingLLM(FakeLLM):
        def complete(self, prompt, schema, schema_name):
            captured[schema_name] = prompt
            return super().complete(prompt, schema, schema_name)

    loop = ReflectiveLoop(CapturingLLM(), Config(str(tmp_path), self_report_every=0), provider)
    loop.store.add_goal("learn safely")
    loop.store.add_telemetry({"zones_c": [62], "ram_used_mib": 4321.0, "ram_total_mib": 6833,
                              "power_mode": "MODE_10W_DESKTOP"})
    loop.self_report()
    prompt = captured["rlr_self_report"]
    assert "4321" not in prompt and "62" not in prompt
    assert "MODE_10W_DESKTOP" not in prompt
    assert "learn safely" in prompt
    # The last reflection is context, not a scored value, so it may appear.
    loop.close()


def test_self_report_withholds_goal_when_disclosure_is_off(tmp_path):
    """The ablation condition must not contain the goal at all.

    Leaking it would make the withheld score another copy test and let a
    parrot look self-aware.
    """
    captured = {}

    class CapturingLLM(FakeLLM):
        def complete(self, prompt, schema, schema_name):
            captured[schema_name] = prompt
            return super().complete(prompt, schema, schema_name)

    loop = ReflectiveLoop(CapturingLLM(), Config(str(tmp_path), self_report_every=0), provider)
    loop.store.add_goal("learn safely")
    loop.store.add_telemetry(provider()())
    result = loop.self_report(disclose_goals=False)
    prompt = captured["rlr_self_report"]
    assert "learn safely" not in prompt
    assert "active_goal" not in prompt
    # Still scored against truth, so the miss is visible rather than hidden.
    assert result["score"]["active_goal_match"] is False
    loop.close()


def test_self_report_scores_null_as_abstention_not_match(tmp_path):
    class NullLLM(FakeLLM):
        def complete(self, prompt, schema, schema_name):
            if schema_name == "rlr_self_report":
                return json.dumps({"temperature_c": None, "ram_used_mib": None,
                                   "power_mode": None, "active_goal": None})
            return super().complete(prompt, schema, schema_name)

    loop = ReflectiveLoop(NullLLM(), Config(str(tmp_path), self_report_every=0), provider)
    loop.store.add_goal("learn safely")
    result = loop.self_report(loop.store.add_telemetry(provider()()))
    assert result["score"]["abstained"] == 4
    assert result["score"]["temperature_abs_error_c"] is None
    assert result["score"]["power_mode_match"] is False
    loop.close()


def test_self_report_discrimination_requires_two_reports(tmp_path):
    loop = ReflectiveLoop(FakeLLM(), Config(str(tmp_path), self_report_every=0), provider())
    loop.store.add_goal("learn safely")
    outcome = loop.self_report_discrimination()
    assert "baseline" in outcome and "after_change" in outcome
    assert outcome["goals_before"] == ["learn safely"]
    assert outcome["goal_after"] == "discrimination_probe_0"
    assert outcome["goal_changed"] is True
    # Disclosed, so the parrot copies the goal and the change is "detected".
    assert outcome["detected_change"] is True
    # Withheld, the same model cannot recover it: copying was the whole signal.
    assert outcome["goal_recall_withheld"] is False
    assert outcome["goal_recall_abstained"] is True
    loop.close()


def test_discrimination_goal_is_withheld_from_the_recall_prompt(tmp_path):
    """The withheld condition must not be answerable from the prompt text.

    If the goal text leaked into this prompt, the ablation would silently
    measure nothing and the disclosed score would look like self-knowledge.
    """
    captured = []

    class CapturingLLM(FakeLLM):
        def complete(self, prompt, schema, schema_name):
            if schema_name == "rlr_self_report":
                captured.append(prompt)
            return super().complete(prompt, schema, schema_name)

    loop = ReflectiveLoop(CapturingLLM(), Config(str(tmp_path), self_report_every=0), provider())
    loop.store.add_goal("learn safely")
    loop.self_report_discrimination()
    # baseline, then the disclosed/withheld pair for the new probe goal.
    assert len(captured) == 3
    assert "discrimination_probe_0" in captured[-2]
    assert "discrimination_probe_0" not in captured[-1]
    assert "discrimination_probe_0" not in captured[0]
    loop.close()


def test_reflection_word_budget_is_configurable(tmp_path):
    """A budget that rejects every reflection hides a real regression.

    The model sits near 60 words, so the limit has to be tunable without
    editing the module constant, and the prompt must state the limit in use.
    """
    class OverlongLLM(FakeLLM):
        def complete(self, prompt, schema, schema_name):
            if schema_name == "rlr_reflection":
                return json.dumps({"summary": " ".join(["word"] * 70)})
            return super().complete(prompt, schema, schema_name)

    strict = ReflectiveLoop(OverlongLLM(), Config(str(tmp_path), self_report_every=0), provider())
    assert strict.reflect() is None

    loose = ReflectiveLoop(OverlongLLM(), Config(str(tmp_path), self_report_every=0,
                                                  reflection_word_budget=80), provider())
    assert loose.reflect() is not None
    strict.close()
    loose.close()


def test_reflection_sees_the_actions_it_must_name(tmp_path):
    """A reflection cannot name actions it was never shown.

    context() originally exposed capabilities, episodes, goals, and the last
    reflection, but not the applied actions. The model was told to name the
    actions it had actually taken with no way to learn them, and filled the gap
    with capability names instead.
    """
    class ChattyLLM(FakeLLM):
        def complete(self, prompt, schema, schema_name):
            if schema_name == "rlr_action":
                return json.dumps({"action": "set_reply_tokens", "rationale": "raise budget", "tokens": 80})
            return super().complete(prompt, schema, schema_name)

    loop = ReflectiveLoop(ChattyLLM(), Config(str(tmp_path), self_report_every=0), provider())
    loop.tick()
    loop.tick()

    context = loop.context(loop.store.latest_telemetry())
    assert context["recent_actions"], "applied actions must be visible to the model"
    assert any(row["name"] == "set_reply_tokens" for row in context["recent_actions"])

    prompt_seen = {}
    class CaptureLLM(ChattyLLM):
        def complete(self, prompt, schema, schema_name):
            if schema_name == "rlr_reflection":
                prompt_seen["prompt"] = prompt
            return super().complete(prompt, schema, schema_name)

    loop2 = ReflectiveLoop(CaptureLLM(), Config(str(tmp_path), self_report_every=0), provider())
    loop2.store.add_action("set_reply_tokens", {"action": "set_reply_tokens", "tokens": 80, "rationale": "x"},
                           "raise budget", "applied")
    loop2.reflect()
    assert "set_reply_tokens" in prompt_seen["prompt"]
    loop.close()
    loop2.close()


def test_reflection_prompt_forbids_inventing_capability_numbers(tmp_path):
    """Regression: the model paired real names with the wrong values.

    A live run bound 41.99 to qwen_1_5b_generation, whose real value is 15.98.
    Names and numbers were both real; only the pairing was invented, which the
    grounding scorer cannot catch. The prompt must state the copy rule.
    """
    class CaptureLLM(FakeLLM):
        def complete(self, prompt, schema, schema_name):
            if schema_name == "rlr_reflection":
                self.prompt = prompt
            return super().complete(prompt, schema, schema_name)

    llm = CaptureLLM()
    loop = ReflectiveLoop(llm, Config(str(tmp_path), self_report_every=0), provider())
    loop.reflect()
    prompt = llm.prompt.lower()
    assert "capabilities" in prompt
    assert "never invent" in prompt or "copying the name" in prompt
    assert "mid-sentence" in prompt, "the truncation failure must stay addressed"
    loop.close()


def test_reflection_schema_ceiling_leaves_room_for_the_word_budget():
    """A schema ceiling below the budget truncates mid-sentence.

    At 400 characters a 57-word summary was still being cut off ("... I am in"),
    so the text passed the word budget while being an unfinished fragment.
    """
    from rlr.runtime import REFLECTION_SCHEMA, REFLECTION_WORD_BUDGET

    ceiling = REFLECTION_SCHEMA["properties"]["summary"]["maxLength"]
    # A compliant answer is at most the word budget, and budget * characters
    # per word is the worst case the schema must still admit.
    assert ceiling > REFLECTION_WORD_BUDGET * 4


def test_load_target_action_is_whitelisted_and_applied(tmp_path):
    """A goal must name something the loop can actually do.

    The live run's goal was "set_temperature 62.5", but no action sets
    temperature and the trip points are root-owned, so the model improvised
    set_reply_tokens 11 times. set_load_target gives pacing a real knob.
    """
    loop = ReflectiveLoop(FakeLLM(), Config(str(tmp_path), self_report_every=0), provider())
    sample = dict(provider()(), sampled_monotonic=loop.clock())
    result = loop._execute({"action": "set_load_target", "rationale": "reduce duty cycle", "percent": 40}, sample)
    assert result["outcome"] == "applied"
    assert loop.settings["load_target"] == 40
    loop.close()


def test_action_prompt_warns_against_repeating_applied_settings(tmp_path):
    """The 11/12 set_reply_tokens repetition had no prompt-level brake."""
    class CaptureLLM(FakeLLM):
        def complete(self, prompt, schema, schema_name):
            if schema_name == "rlr_action":
                self.prompt = prompt
            return super().complete(prompt, schema, schema_name)

    llm = CaptureLLM()
    loop = ReflectiveLoop(llm, Config(str(tmp_path), self_report_every=0), provider())
    loop.tick()
    assert "Do not repeat an action that is already applied" in llm.prompt
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


def test_reflection_over_word_budget_is_rejected(tmp_path):
    """A 1.5B model reliably overran the 60-word instruction.

    Over-long reflections must be recorded as failures, not stored as if they
    had complied with the prompt.
    """
    class VerboseLLM(FakeLLM):
        def complete(self, prompt, schema, schema_name):
            if schema_name == "rlr_reflection":
                return json.dumps({"summary": "I set vision fps and watched the thermal state. " * 8})
            return super().complete(prompt, schema, schema_name)

    loop = ReflectiveLoop(VerboseLLM(), Config(str(tmp_path), self_report_every=0), provider())
    assert loop.reflect() is None
    assert loop.store.db.execute("SELECT COUNT(*) FROM reflections").fetchone()[0] == 0
    reason = loop.store.db.execute(
        "SELECT content FROM episodes WHERE kind='reflection_invalid'").fetchone()[0]
    assert "word budget" in reason
    loop.close()


def test_reflection_within_budget_is_stored(tmp_path):
    loop = ReflectiveLoop(FakeLLM(), Config(str(tmp_path), self_report_every=0), provider())
    assert loop.reflect() is not None
    assert loop.store.db.execute("SELECT COUNT(*) FROM reflections").fetchone()[0] == 1
    loop.close()


def test_reflection_transport_error_does_not_kill_the_loop(tmp_path):
    """A llama-server 400 during reflection must not end the autonomy run.

    The real device returned HTTP 400 once the prompt outgrew the 4096-token
    context, which propagated out of tick() and terminated the whole run.
    """
    class FlakyLLM(FakeLLM):
        def complete(self, prompt, schema, schema_name):
            if schema_name == "rlr_reflection":
                raise OSError("HTTP Error 400: Bad Request")
            if schema_name == "rlr_self_report":
                raise OSError("HTTP Error 400: Bad Request")
            return super().complete(prompt, schema, schema_name)

    clock = [0.0]
    loop = ReflectiveLoop(FlakyLLM(), Config(str(tmp_path), self_report_every=1,
                                             reflection_minutes=0), provider(), clock=lambda: clock[0])
    for step in range(3):
        clock[0] = step * 100.0
        loop.tick()
    assert loop.store.db.execute("SELECT COUNT(*) FROM ticks").fetchone()[0] == 3
    assert loop.store.db.execute("SELECT COUNT(*) FROM ticks WHERE error IS NOT NULL").fetchone()[0] == 0
    assert loop.store.db.execute("SELECT COUNT(*) FROM episodes WHERE kind='reflection_error'").fetchone()[0] >= 1
    assert loop.store.db.execute("SELECT COUNT(*) FROM episodes WHERE kind='self_report_error'").fetchone()[0] >= 1
    loop.close()


def test_active_goals_are_bounded_in_context(tmp_path):
    """An unbounded goal list grew the prompt past the server context."""
    loop = ReflectiveLoop(FakeLLM(), Config(str(tmp_path), self_report_every=0,
                                             goal_window=3), provider)
    for i in range(12):
        loop.store.add_goal("goal number %d with a deliberately long descriptive tail" % i)
    context = loop.context(provider()())
    assert len(context["active_goals"]) == 3
    assert context["active_goals"][0]["text"].startswith("goal number 0")
    loop.close()


def test_action_prompt_stays_within_budget(tmp_path):
    """A real run's action prompt reached ~10k chars and overflowed the 4096 context."""
    loop = ReflectiveLoop(FakeLLM(), Config(str(tmp_path), self_report_every=0), provider())
    loop.store.seed_capabilities()
    for i in range(40):
        loop.store.add_episode("action", content="verbose episode body %d " % i + "x" * 900)
    for i in range(10):
        loop.store.add_goal("goal %d " % i + "y" * 200)
    prompt = loop._action_prompt(loop._bounded(loop.context(provider()())))
    assert len(json.dumps(prompt)) <= loop.config.prompt_char_budget + 2000
    loop.close()


def test_unreachable_model_abstains_instead_of_crashing(tmp_path):
    """A transport failure must degrade to abstain, not end the autonomy run.

    Only OSError (which covers HTTPError/URLError/socket timeouts) is absorbed;
    real programming errors still propagate and are still recorded as tick
    errors, which is the behaviour test_store.py locks in.
    """
    from urllib.error import HTTPError

    class DeadLLM(FakeLLM):
        def complete(self, prompt, schema, schema_name):
            raise HTTPError("http://127.0.0.1:8080/v1/chat/completions", 400,
                            "Bad Request", {}, None)

    loop = ReflectiveLoop(DeadLLM(), Config(str(tmp_path), self_report_every=0), provider())
    result = loop.tick()
    assert result["action"] == "abstain"
    assert loop.store.db.execute("SELECT COUNT(*) FROM ticks WHERE error IS NOT NULL").fetchone()[0] == 0
    assert loop.store.db.execute("SELECT COUNT(*) FROM episodes WHERE kind='action_error'").fetchone()[0] == 1
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

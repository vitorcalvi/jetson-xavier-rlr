"""End-to-end runtime behaviour: fast path, gating, fail-closed, and survival."""

from __future__ import annotations

import json

from conftest import make_scored, warm_sample

from reflective_loop_runtime.contracts import ContractError, ScoredDecision
from reflective_loop_runtime.deciders import DeterministicOpenDecider
from reflective_loop_runtime.qwen import Advice


class RecordingDecider:
    """Wrap a decider and record the contexts/attempts it saw."""

    name = "recording"

    def __init__(self, inner=None):
        self.inner = inner or DeterministicOpenDecider()
        self.calls = []

    def score(self, candidates, state, *, context=""):
        self.calls.append(context)
        return self.inner.score(candidates, state, context=context)


class FixedDecider:
    """Return a scripted decision, or raise, on each successive call."""

    name = "fixed"

    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def score(self, candidates, state, *, context=""):
        self.calls += 1
        result = self.results[min(self.calls - 1, len(self.results) - 1)]
        if isinstance(result, Exception):
            raise result
        choice, confidence = result
        return make_scored(candidates, choice, confidence)


class RaisingNormalizer:
    name = "raising-lfm"

    def normalize(self, raw_text, state, *, question=""):
        raise RuntimeError("lfm exploded")


class RaisingSlowBrain:
    name = "raising-qwen"

    def advise(self, *, state, candidates, previous, context=""):
        raise RuntimeError("qwen unreachable")

    def reflect(self, *, state, context=""):
        raise RuntimeError("qwen unreachable")


class StaticSlowBrain:
    name = "static-qwen"

    def __init__(self, suggestion="observe", summary="advice"):
        self.suggestion = suggestion
        self.summary = summary
        self.reflect_calls = 0

    def advise(self, *, state, candidates, previous, context=""):
        ids = [c.id for c in candidates]
        suggested = self.suggestion if self.suggestion in ids else None
        return Advice(summary=self.summary, rationale="r", suggested_candidate=suggested)

    def reflect(self, *, state, context=""):
        self.reflect_calls += 1
        return "a reflection summary"


class TestFastPath:
    def test_structured_tick_skips_lfm_and_qwen(self, runtime_factory):
        decider = RecordingDecider()
        runtime = runtime_factory(decider=decider, slow_brain=StaticSlowBrain())
        result = runtime.tick(warm_sample(), capabilities={"vision", "generation"},
                              structured=True)
        assert result.structured is True
        assert result.used_lfm is False
        assert result.used_qwen is False
        # A confident deterministic decision stays on the fast path.
        assert len(decider.calls) == 1

    def test_decision_recorded_with_probabilities(self, runtime_factory):
        runtime = runtime_factory()
        runtime.tick(warm_sample(), capabilities={"vision", "generation"}, structured=True)
        rows = runtime.store.iter_table("decisions")
        assert rows, "a decision must be persisted"
        payload = json.loads(rows[0]["probabilities"])
        assert payload, "probabilities must be persisted"
        assert abs(sum(payload.values()) - 1.0) < 0.02


class TestLowConfidencePath:
    def test_qwen_advice_triggers_bounded_redecision_not_direct_action(self, runtime_factory):
        # First decision is uncertain; after advice OpenDecider decides again.
        decider = FixedDecider([("observe", 0.2), ("abstain", 0.9)])
        brain = StaticSlowBrain(suggestion="abstain")
        runtime = runtime_factory(decider=decider, slow_brain=brain)
        result = runtime.tick(warm_sample(), capabilities={"vision"}, structured=True)
        assert result.used_qwen is True
        assert decider.calls == 2, "Qwen must not be the final selector; OpenDecider re-decides"
        assert result.decision["choice"] == "abstain"
        assert result.decision["mode"] == "advised"

    def test_advisor_is_never_the_actuator(self, runtime_factory):
        # Qwen suggests something, but the cortex chooses a different valid id.
        decider = FixedDecider([("observe", 0.1), ("observe", 0.95)])
        brain = StaticSlowBrain(suggestion="abstain")
        runtime = runtime_factory(decider=decider, slow_brain=brain)
        result = runtime.tick(warm_sample(), capabilities={"vision"}, structured=True)
        assert result.decision["choice"] == "observe"
        assert result.execution["action"] == "observe"

    def test_no_slow_brain_keeps_loop_running(self, runtime_factory):
        decider = FixedDecider([("observe", 0.2)])
        runtime = runtime_factory(decider=decider, slow_brain=None)
        result = runtime.tick(warm_sample(), capabilities={"vision"}, structured=True)
        assert result.used_qwen is False
        assert result.decision["choice"] == "observe"


class TestFailClosed:
    def test_malformed_decider_output_fails_to_abstain(self, runtime_factory):
        decider = FixedDecider([ContractError("bad distribution")])
        runtime = runtime_factory(decider=decider)
        result = runtime.tick(warm_sample(), capabilities={"vision"}, structured=True)
        assert result.decision["mode"] == "fail-closed"
        assert result.execution["action"] == "abstain"

    def test_decider_exception_fails_to_abstain(self, runtime_factory):
        decider = FixedDecider([RuntimeError("model offline")])
        runtime = runtime_factory(decider=decider)
        result = runtime.tick(warm_sample(), capabilities={"vision"}, structured=True)
        assert result.execution["action"] == "abstain"

    def test_non_finite_confidence_fails_closed(self, runtime_factory):
        class NanDecider:
            name = "nan"

            def score(self, candidates, state, *, context=""):
                ids = [c.id for c in candidates]
                probs = {i: 0.0 for i in ids}
                probs[ids[0]] = float("nan")
                return ScoredDecision(choice=ids[0], confidence=float("nan"), probabilities=probs)

        runtime = runtime_factory(decider=NanDecider())
        result = runtime.tick(warm_sample(), capabilities={"vision"}, structured=True)
        assert result.execution["action"] == "abstain"

    def test_lfm_failure_is_survivable(self, runtime_factory):
        runtime = runtime_factory(normalizer=RaisingNormalizer())
        result = runtime.tick(
            warm_sample(), capabilities={"vision"},
            structured=False, raw_text="totally unstructured-ish notes",
        )
        assert result.used_lfm is True
        assert result.lfm_error is not None
        assert result.execution["action"] is not None

    def test_slow_brain_exception_is_survivable(self, runtime_factory):
        decider = FixedDecider([("observe", 0.1), ("observe", 0.7)])
        runtime = runtime_factory(decider=decider, slow_brain=RaisingSlowBrain())
        result = runtime.tick(warm_sample(), capabilities={"vision"}, structured=True)
        assert result.used_qwen is True
        assert result.decision["choice"] == "observe"

    def test_stale_telemetry_vetoes_side_effect(self, runtime_factory):
        # Force the cortex to pick a side-effecting action, then starve the
        # telemetry so the Governor must veto it.
        decider = FixedDecider([("set_vision_fps", 0.99)])
        runtime = runtime_factory(decider=decider)
        stale = warm_sample(age_seconds=999.0)
        result = runtime.tick(stale, capabilities={"vision"}, structured=True)
        assert result.vetoed is True
        assert result.veto_reason is not None
        # The vetoed proposal is reported, but the action that actually ran is
        # the safe abstain.
        assert result.execution["action"] == "abstain"
        assert result.execution["ok"] is True
        assert "vetoed" in result.execution["note"]


class TestMeasuredQuality:
    """The decision engine should be right about the thermal response."""

    def test_hot_tick_cools_down(self, runtime_factory):
        runtime = runtime_factory()
        result = runtime.tick(
            warm_sample(temperature_c=80.0, zones_c=[80.0]),
            capabilities={"vision", "generation"}, structured=True,
        )
        assert result.decision["choice"] == "cooldown"
        assert result.execution["ok"] is True

    def test_cool_tick_does_not_over_act(self, runtime_factory):
        runtime = runtime_factory()
        result = runtime.tick(warm_sample(temperature_c=50.0),
                              capabilities={"vision"}, structured=True)
        assert result.decision["choice"] in ("observe", "abstain")


class TestLoopSurvival:
    def test_loop_survives_long_sequence_of_failures(self, runtime_factory):
        class AlwaysFail:
            name = "always-fail"

            def score(self, candidates, state, *, context=""):
                raise RuntimeError("boom")

        runtime = runtime_factory(
            decider=AlwaysFail(), slow_brain=RaisingSlowBrain(),
            normalizer=RaisingNormalizer(),
        )
        for i in range(25):
            result = runtime.tick(
                warm_sample(temperature_c=55.0 + i),
                capabilities={"vision"}, structured=False, raw_text=f"noise {i}",
            )
            assert result.execution["action"] is not None
        assert runtime.store.counts()["actions"] >= 25

    def test_successful_multi_tick_run_persists_everything(self, runtime_factory):
        runtime = runtime_factory()
        for i in range(6):
            runtime.tick(warm_sample(temperature_c=52.0 + i * 4),
                         capabilities={"vision", "generation"}, structured=True)
        counts = runtime.store.counts()
        assert counts["telemetry"] == 6
        assert counts["decisions"] >= 6
        assert counts["actions"] >= 6
        assert counts["outcomes"] >= 6


class TestReflection:
    def test_reflect_action_calls_slow_brain_and_stores_memory(self, runtime_factory):
        decider = FixedDecider([("reflect", 0.99)])
        brain = StaticSlowBrain()
        runtime = runtime_factory(decider=decider, slow_brain=brain)
        result = runtime.tick(warm_sample(temperature_c=50.0),
                              capabilities={"vision"}, structured=True)
        assert result.decision["choice"] == "reflect"
        assert brain.reflect_calls == 1
        assert runtime.store.counts()["memories"] >= 1

class TestBackgroundGoalSpamPrevention:
    def test_background_monitoring_never_proposes_default_set_goal(self, runtime_factory):
        class InspectCandidates:
            name = 'goal-inspector'
            def __init__(self):
                self.ids = []
            def score(self, candidates, state, *, context=''):
                self.ids.append([c.id for c in candidates])
                return make_scored(candidates, 'abstain', 0.98)
        backend = InspectCandidates()
        runtime = runtime_factory(decider=backend)
        for _ in range(6):
            runtime.tick(warm_sample(), capabilities=['vision', 'generation'],
                         structured=True, goal_digest='maintain stable operation',
                         allow_default_goal_creation=False)
        assert len(backend.ids) == 6
        assert all('set_goal' not in ids for ids in backend.ids)
        assert all('abstain' in ids and 'observe' in ids for ids in backend.ids)
        assert not any(r['name'] == 'set_goal' for r in runtime.store.recent_actions(limit=30))

    def test_explicit_goal_creation_candidate_remains_available(self, runtime_factory):
        class InspectCandidates:
            name = 'goal-inspector'
            def __init__(self):
                self.ids = []
            def score(self, candidates, state, *, context=''):
                self.ids.append([c.id for c in candidates])
                return make_scored(candidates, 'abstain', 0.98)
        backend = InspectCandidates()
        runtime = runtime_factory(decider=backend)
        runtime.tick(warm_sample(), structured=True,
                     goal_digest='investigate a concrete operator request',
                     allow_default_goal_creation=True)
        assert 'set_goal' in backend.ids[0]

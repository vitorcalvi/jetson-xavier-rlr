"""Qwen slow brain: advisory-only output, bounding, and outage tolerance."""

from __future__ import annotations

from reflective_loop_runtime.candidates import Candidate
from reflective_loop_runtime.qwen import (
    Advice,
    CallableSlowBrain,
    OpenAICompatClient,
    QwenSlowBrain,
)


def _candidates():
    return [
        Candidate(id="observe", action="observe", label="observe"),
        Candidate(id="abstain", action="abstain", args={"reason": "wait"}, label="abstain"),
        Candidate(id="cooldown", action="cooldown", args={"seconds": 30}, label="cooldown"),
    ]


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.stops: list = []

    def chat(self, system, user, *, temperature=0.0, grammar=None, stop=None):
        self.calls += 1
        self.stops.append(stop)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class TestQwenSlowBrain:
    def test_valid_advice_is_parsed(self):
        client = FakeClient('{"summary": "cool down", "suggested_candidate": "cooldown"}')
        advice = QwenSlowBrain(client).advise(state={}, candidates=_candidates(), previous=None)
        assert advice.suggested_candidate == "cooldown"
        assert advice.degraded is False

    def test_unknown_suggestion_is_dropped(self):
        client = FakeClient('{"summary": "x", "suggested_candidate": "rm_rf"}')
        advice = QwenSlowBrain(client).advise(state={}, candidates=_candidates(), previous=None)
        assert advice.suggested_candidate is None

    def test_outage_degrades_without_raising(self):
        client = FakeClient(RuntimeError("connection refused"))
        advice = QwenSlowBrain(client).advise(state={}, candidates=_candidates(), previous=None)
        assert advice.degraded is True
        assert advice.error

    def test_malformed_json_degrades(self):
        client = FakeClient("not json")
        advice = QwenSlowBrain(client).advise(state={}, candidates=_candidates(), previous=None)
        assert advice.degraded is True

    def test_empty_advice_degrades(self):
        client = FakeClient('{"summary": "", "rationale": ""}')
        advice = QwenSlowBrain(client).advise(state={}, candidates=_candidates(), previous=None)
        assert advice.degraded is True

    def test_reflect_failure_returns_empty(self):
        client = FakeClient(RuntimeError("down"))
        assert QwenSlowBrain(client).reflect(state={}) == ""

    def test_reflect_summary_is_returned(self):
        client = FakeClient('{"summary": "lessons learned"}')
        assert QwenSlowBrain(client).reflect(state={}) == "lessons learned"

    def test_advice_is_bounded(self):
        client = FakeClient('{"summary": "' + "y" * 5000 + '"}')
        advice = QwenSlowBrain(client).advise(state={}, candidates=_candidates(), previous=None)
        assert len(advice.summary) <= 800


class TestCallableSlowBrain:
    def test_advise_via_callable(self):
        brain = CallableSlowBrain(
            lambda payload: {"summary": "ok", "suggested_candidate": "observe"}
        )
        advice = brain.advise(state={}, candidates=_candidates(), previous=None)
        assert advice.suggested_candidate == "observe"

    def test_callable_exception_degrades(self):
        def boom(payload):
            raise ValueError("bad")

        advice = CallableSlowBrain(boom).advise(state={}, candidates=_candidates(), previous=None)
        assert advice.degraded is True

    def test_reflect_via_callable(self):
        brain = CallableSlowBrain(lambda payload: {"summary": "sum"})
        assert brain.reflect(state={}) == "sum"

    def test_advice_dataclass_roundtrip(self):
        advice = Advice(summary="s", suggested_candidate="observe", focus=["a"])
        assert advice.to_dict()["suggested_candidate"] == "observe"


class TestClientSafety:
    def test_loopback_only_by_default(self):
        OpenAICompatClient("http://127.0.0.1:8080", "m")  # ok
        try:
            OpenAICompatClient("http://evil.example.com", "m")
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("remote endpoints must be rejected by default")

    def test_remote_allowed_when_opted_in(self):
        OpenAICompatClient("http://example.com", "m", allow_remote=True)

    def test_non_http_scheme_rejected(self):
        try:
            OpenAICompatClient("ftp://127.0.0.1", "m")
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("non-http schemes must be rejected")

"""The falcon06 advisor asks for grammar-constrained JSON and fails safe.

These tests drive the real :class:`OpenAICompatClient` and :class:`QwenSlowBrain`
against a scripted OpenAI-compatible advisor bound to ``127.0.0.1``. Nothing
here reaches a non-loopback host: the only network peer is the local fake server,
which also lets us assert on the exact completion request the runtime sends.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from reflective_loop_runtime.candidates import Candidate
from reflective_loop_runtime.config import Config
from reflective_loop_runtime.contracts import ScoredDecision
from reflective_loop_runtime.qwen import (
    ADVICE_MAX_CANDIDATE_ID_CHARS,
    ADVICE_MAX_TOKENS,
    ADVICE_STOP,
    ADVICE_TEXT_MAX_CHARS,
    ADVICE_TOKEN_SLACK,
    OpenAICompatClient,
    QwenSlowBrain,
    advice_grammar,
    advice_max_chars,
)

_VALID_ADVICE = json.dumps(
    {
        "rationale": "thermal margin is thin",
        "suggested_candidate": "cooldown",
    }
)


def _candidates():
    return [
        Candidate(id="observe", action="observe", label="observe"),
        Candidate(id="abstain", action="abstain", args={"reason": "wait"}, label="abstain"),
        Candidate(id="cooldown", action="cooldown", args={"seconds": 30}, label="cooldown"),
    ]


class _QuietServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):  # keep the test output quiet
        pass


class _Advisor:
    """A scripted loopback OpenAI-compatible advisor that records requests."""

    def __init__(self, *, content: str = _VALID_ADVICE, delay: float = 0.0):
        self.content = content
        self.delay = delay
        self.requests: list[dict] = []
        advisor = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - stdlib handler hook
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                advisor.requests.append(json.loads(raw.decode("utf-8")))
                if advisor.delay:
                    time.sleep(advisor.delay)
                payload = json.dumps(
                    {"choices": [{"message": {"content": advisor.content}}]}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the client gave up (timeout test)

            def log_message(self, *args):  # keep the test output quiet
                pass

        self._httpd = _QuietServer(("127.0.0.1", 0), Handler)
        host, port = self._httpd.server_address[:2]
        self.base_url = f"http://{host}:{port}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> _Advisor:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> bool:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)
        return False

    def client(self, *, timeout: float = 5.0) -> OpenAICompatClient:
        return OpenAICompatClient(self.base_url, "falcon06", timeout=timeout)


def _valid_rule_name(name: str) -> bool:
    if not name or not (name[0].isalpha() or name[0] == "_"):
        return False
    return all(ch.isalnum() or ch in "_-" for ch in name)


def _parse_gbnf_rules(grammar: str) -> dict[str, str]:
    """Split a GBNF grammar into ``rule name -> right-hand side``."""
    rules: dict[str, str] = {}
    for raw_line in grammar.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, rhs = line.partition("::=")
        assert sep == "::=", f"not a rule definition: {line!r}"
        name = name.strip()
        assert _valid_rule_name(name), f"invalid rule name: {name!r}"
        assert name not in rules, f"duplicate rule: {name!r}"
        rules[name] = rhs
    return rules


def _gbnf_references(rhs: str) -> list[str]:
    """Return the rule names referenced by ``rhs``.

    String literals (``"..."``) and character classes (``[...]``) are skipped,
    so only genuine bare identifiers — which GBNF reads as rule references, not
    literals — are returned.
    """
    refs: list[str] = []
    i, n = 0, len(rhs)
    while i < n:
        ch = rhs[i]
        if ch in ('"', "["):
            close = '"' if ch == '"' else "]"
            i += 1
            while i < n:
                if rhs[i] == "\\":
                    i += 2
                    continue
                if rhs[i] == close:
                    i += 1
                    break
                i += 1
            continue
        if ch.isalpha() or ch == "_":
            j = i
            while j < n and (rhs[j].isalnum() or rhs[j] in "_-"):
                j += 1
            refs.append(rhs[i:j])
            i = j
            continue
        i += 1
    return refs


def _lint_gbnf(grammar: str) -> dict[str, str]:
    """Minimal GBNF lint: every referenced rule must be defined.

    This is the check llama-server runs when it initialises samplers; a bare
    literal like ``null`` reaches it as an undefined rule identifier and the
    request fails with HTTP 400.
    """
    rules = _parse_gbnf_rules(grammar)
    assert rules, "grammar defines no rules"
    for name, rhs in rules.items():
        for ref in _gbnf_references(rhs):
            assert ref in rules, (
                f"undefined rule identifier {ref!r} referenced by {name!r}"
            )
    return rules


class TestAdvisorGrammar:
    def test_advice_request_carries_a_gbnf_grammar(self):
        with _Advisor() as server:
            advice = QwenSlowBrain(server.client()).advise(
                state={}, candidates=_candidates(), previous=None
            )
        assert advice.degraded is False
        assert advice.suggested_candidate == "cooldown"
        grammar = server.requests[0]["grammar"]
        assert grammar.strip()
        # The shipped grammar lints clean (no undefined rule identifiers).
        rules = _lint_gbnf(grammar)
        assert "root" in rules
        # The contract keys and the bounded action ids all appear in the grammar.
        for token in ("rationale", "suggested_candidate"):
            assert token in grammar
        for cid in ("observe", "abstain", "cooldown"):
            assert cid in grammar
        assert "null" in grammar

    def test_reflect_request_is_also_grammar_constrained(self):
        with _Advisor(content=json.dumps({"summary": "lessons learned"})) as server:
            assert QwenSlowBrain(server.client()).reflect(state={}) == "lessons learned"
        assert "grammar" in server.requests[0]
        assert "summary" in server.requests[0]["grammar"]
        _lint_gbnf(server.requests[0]["grammar"])

    def test_grammar_bounds_the_suggestion_to_the_candidate_set(self):
        grammar = advice_grammar(["observe", "complete_goal:7"])
        assert "observe" in grammar and "complete_goal:7" in grammar
        assert "cooldown" not in grammar and "set_load_target" not in grammar
        assert "root ::= object" in grammar

    def test_invalid_reply_falls_back_to_abstain(self):
        with _Advisor(content="this is not json") as server:
            advice = QwenSlowBrain(server.client()).advise(
                state={}, candidates=_candidates(), previous=None
            )
        assert advice.degraded is True
        assert advice.error
        assert advice.suggested_candidate == "abstain"

    def test_timeout_falls_back_to_abstain(self):
        with _Advisor(delay=0.6) as server:
            advice = QwenSlowBrain(server.client(timeout=0.1)).advise(
                state={}, candidates=_candidates(), previous=None
            )
        assert advice.degraded is True
        assert advice.suggested_candidate == "abstain"

    def test_fallback_does_not_invent_abstain_when_it_is_absent(self):
        candidates = [
            Candidate(id="observe", action="observe", label="observe"),
            Candidate(id="cooldown", action="cooldown", args={"seconds": 30}, label="cooldown"),
        ]
        with _Advisor(content="not json") as server:
            advice = QwenSlowBrain(server.client()).advise(
                state={}, candidates=candidates, previous=None
            )
        assert advice.degraded is True
        assert advice.suggested_candidate is None


class TestAdviceGrammarLint:
    """Static checks on the shipped GBNF, mirroring llama-server's parser."""

    IDS = ["observe", "abstain", "cooldown", "complete_goal:7"]

    def test_every_referenced_rule_is_defined(self):
        rules = _lint_gbnf(advice_grammar(self.IDS))
        assert "root" in rules
        assert _lint_gbnf(advice_grammar([]))  # empty candidate set still lints

    def test_bare_literal_is_never_read_as_a_rule_reference(self):
        grammar = advice_grammar(self.IDS)
        # The shipped bug: an unquoted ``null`` became an undefined rule.
        assert "::= null" not in grammar
        assert '"null"' in grammar
        assert "null" not in _parse_gbnf_rules(grammar)

    def test_lint_flags_the_original_undefined_identifier(self):
        # llama-server's own failure mode, reproduced by the linter.
        buggy = (
            "root ::= object\n"
            "candidate ::= null | \"\\\"observe\\\"\"\n"
            "object ::= candidate\n"
        )
        with pytest.raises(AssertionError, match="undefined rule identifier 'null'"):
            _lint_gbnf(buggy)

    def test_object_keys_are_ordered_and_the_object_closes(self):
        rules = _lint_gbnf(advice_grammar(self.IDS))
        obj = rules["object"]
        positions = [
            obj.index(rf'\"{key}\"')
            for key in ("rationale", "suggested_candidate")
        ]
        assert positions == sorted(positions)
        assert obj.rstrip().endswith('"}"')
        assert "{" in obj

    def test_grammar_has_no_optional_or_unbounded_whitespace_rule(self):
        rules = _lint_gbnf(advice_grammar(self.IDS))
        assert "ws" not in rules
        for rhs in rules.values():
            assert "[ " not in rhs and "\\t" not in rhs and "\\n" not in rhs

    def test_free_text_is_bounded_and_cannot_escape(self):
        rules = _lint_gbnf(advice_grammar(self.IDS))
        text = rules["text"]
        assert "{0," + str(ADVICE_TEXT_MAX_CHARS) + "}" in text
        start = text.index("[^") + 2
        char_class = text[start:text.index("]", start)]
        assert '"' in char_class  # quotes excluded, so text cannot escape
        assert "\\\\" in char_class  # backslashes excluded, so no escapes
        assert "\\x00-\\x1F" in char_class  # control chars/newlines excluded


class TestAdviceTokenBudget:
    """The reply has to *fit*: a budget the grammar can exceed times out anyway."""

    IDS = ["observe", "abstain", "cooldown", "complete_goal:7"]

    def _longest_valid_object(self, ids: list[str]) -> tuple[str, str]:
        longest_id = max(ids, key=lambda cid: len(json.dumps(cid)))
        text = json.dumps("x" * ADVICE_TEXT_MAX_CHARS)
        return (
            '{"rationale":' + text + ',"suggested_candidate":' + json.dumps(longest_id) + "}",
            longest_id,
        )

    def test_max_length_object_fits_the_max_tokens_setting(self):
        longest_object, longest_id = self._longest_valid_object(self.IDS)
        # It is a real, parseable advice object ...
        assert json.loads(longest_object)["suggested_candidate"] == longest_id
        # ... and it is no longer than the grammar's own worst case.
        assert len(longest_object) <= advice_max_chars(self.IDS)
        # At the worst case of one token per character the object still fits the
        # budget the advisor client sends, so a complete reply is never cut off
        # at the token limit.
        needed_tokens = len(longest_object)
        assert needed_tokens <= ADVICE_MAX_TOKENS
        client = OpenAICompatClient("http://127.0.0.1:8080", "falcon06")
        assert client.max_tokens >= needed_tokens
        assert client.max_tokens == ADVICE_MAX_TOKENS

    def test_budget_is_the_worst_case_object_plus_slack(self):
        # Derived, not guessed: one token per character of the longest object
        # the grammar admits, plus slack. Not the old 1024 (never finished) and
        # not the 96 sized at two characters per token, which could still clip
        # the closing brace off a worst-case reply.
        assert ADVICE_TEXT_MAX_CHARS == 40
        assert ADVICE_MAX_TOKENS == (
            advice_max_chars(["x" * ADVICE_MAX_CANDIDATE_ID_CHARS]) + ADVICE_TOKEN_SLACK
        )
        assert advice_max_chars(self.IDS) <= ADVICE_MAX_TOKENS
        # falcon06 decodes at ~20 tokens per second on the Jetson, so the budget
        # has to stay finishable inside the 15 s advisor timeout.
        assert ADVICE_MAX_TOKENS <= 15 * 20
        # The deploy/config defaults ship the same budget, so the shipped
        # runtime never truncates with a stale number.
        assert Config().qwen_max_tokens == ADVICE_MAX_TOKENS

    def test_grammar_admits_only_the_enum_and_a_bounded_rationale(self):
        rules = _lint_gbnf(advice_grammar(self.IDS))
        assert set(rules) == {"root", "candidate", "object", "text"}
        # The unbounded text rules that used to blow the budget are gone.
        assert "array" not in rules
        for dropped in ("summary", "focus"):
            assert dropped not in rules["object"]

    def test_request_stops_at_the_closing_brace(self):
        with _Advisor() as server:
            QwenSlowBrain(server.client()).advise(
                state={}, candidates=_candidates(), previous=None
            )
        request_body = server.requests[0]
        assert request_body["stop"] == list(ADVICE_STOP) == ["}"]
        assert request_body["max_tokens"] == ADVICE_MAX_TOKENS

    def test_reply_stopped_at_the_brace_still_parses(self):
        # An OpenAI-compatible server omits the stop sequence from the content,
        # so a bounded reply arrives one "}" short and must still be usable.
        with _Advisor(content='{"rationale":"thin margin","suggested_candidate":"cooldown"') as server:
            advice = QwenSlowBrain(server.client()).advise(
                state={}, candidates=_candidates(), previous=None
            )
        assert advice.degraded is False
        assert advice.suggested_candidate == "cooldown"


class _AdviceAwareDecider:
    """Uncertain by default; switches to abstain on the adviser's suggestion."""

    name = "advice-aware"

    def score(self, candidates, state, *, context=""):
        ids = [candidate.id for candidate in candidates]
        if "advisor-suggestion: abstain" in context:
            choice, confidence = "abstain", 0.95
        else:
            choice, confidence = "observe", 0.2
        rest = [cid for cid in ids if cid != choice]
        probabilities = {choice: confidence}
        share = (1.0 - confidence) / len(rest)
        for cid in rest:
            probabilities[cid] = share
        probabilities[rest[-1]] += 1.0 - sum(probabilities.values())
        return ScoredDecision(
            choice=choice, confidence=probabilities[choice], probabilities=probabilities
        )


class TestAdvisorFallbackThroughRuntime:
    def test_invalid_advisor_reply_lands_on_the_bounded_abstain(self, runtime_factory):
        sample = {
            "temperature_c": 55.0,
            "zones_c": [55.0],
            "ram_used_mib": 2048.0,
            "ram_total_mib": 8192.0,
            "sampled_monotonic": None,
            "age_seconds": 0.0,
        }
        with _Advisor(content="not json") as server:
            brain = QwenSlowBrain(server.client())
            runtime = runtime_factory(decider=_AdviceAwareDecider(), slow_brain=brain)
            result = runtime.tick(sample, capabilities={"vision"}, structured=True)
        assert result.used_qwen is True
        assert result.qwen_degraded is True
        assert result.decision["choice"] == "abstain"
        assert result.execution["action"] == "abstain"

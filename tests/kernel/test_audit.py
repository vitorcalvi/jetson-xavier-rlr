"""Tests for the hash-chained audit log and behavioral replay.

The audit log must be tamper-evident: editing any past row, deleting a row, or
truncating the tail all make ``verify_chain`` fail. Recorded decisions must also
behave deterministically under replay.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime

import pytest

from reflective_loop_runtime.kernel.audit import (
    AUDIT_FIELDS,
    GENESIS_HASH,
    AuditError,
    AuditEvent,
    AuditLog,
    ChainVerification,
    canonical_json,
    compute_event_hash,
    replay,
    verify_chain,
)


def make_event(**overrides):
    """A complete, valid set of append fields."""
    fields = {
        "ts": "2026-10-05T00:00:00+00:00",
        "state_hash": "s" * 64,
        "goal_version": "goal-1",
        "plan_version": "plan-1",
        "candidate_set_hash": "c" * 64,
        "scores": {"observe": 0.7, "abstain": 0.3},
        "policy_version": "policy-1",
        "governor_result": {"allowed": True},
        "permit_id": "permit-1",
        "tool_request_hash": "r" * 64,
        "tool_result_hash": "t" * 64,
        "outcome": "observe",
        "budget_delta": 1.0,
    }
    fields.update(overrides)
    return fields


@pytest.fixture
def log(tmp_path):
    audit = AuditLog(tmp_path / "audit.sqlite")
    yield audit
    audit.close()


def _sql(path, statement, params=()):
    """Apply one raw statement through a separate connection."""
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(statement, params)
        conn.commit()
    finally:
        conn.close()


def decide_from_event(event):
    """A pure decision function: argmax over scores, governed by the veto."""
    scores = event.scores if isinstance(event.scores, dict) else {}
    choice = max(scores, key=scores.get) if scores else "abstain"
    if isinstance(event.governor_result, dict) and not event.governor_result.get("allowed", True):
        return "abstain"
    return choice


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


class TestEventHash:
    def test_hash_is_sha256_hex_of_prev_plus_canonical_json(self):
        event = make_event()
        expected = hashlib.sha256(
            (GENESIS_HASH + canonical_json(event)).encode("utf-8")
        ).hexdigest()
        digest = compute_event_hash(GENESIS_HASH, event)
        assert digest == expected
        assert len(digest) == 64
        assert digest == digest.lower()

    def test_hash_is_independent_of_key_insertion_order(self):
        first = make_event()
        second = dict(reversed(list(first.items())))
        assert compute_event_hash(GENESIS_HASH, first) == compute_event_hash(GENESIS_HASH, second)

    def test_different_prev_hash_changes_hash(self):
        event = make_event()
        assert compute_event_hash(GENESIS_HASH, event) != compute_event_hash("a" * 64, event)

    @pytest.mark.parametrize(
        "field, value",
        [
            ("seq", 9),
            ("ts", "2099-01-01T00:00:00+00:00"),
            ("state_hash", "edited"),
            ("goal_version", "edited"),
            ("plan_version", "edited"),
            ("candidate_set_hash", "edited"),
            ("scores", {"observe": 0.1, "abstain": 0.9}),
            ("policy_version", "edited"),
            ("governor_result", {"allowed": False}),
            ("permit_id", "edited"),
            ("tool_request_hash", "edited"),
            ("tool_result_hash", "edited"),
            ("outcome", "abstain"),
            ("budget_delta", 42.0),
            ("prev_hash", "f" * 64),
        ],
    )
    def test_every_field_is_covered_by_the_hash(self, field, value):
        base = dict(make_event(), seq=0, prev_hash=GENESIS_HASH)
        mutated = dict(base, **{field: value})
        assert compute_event_hash(GENESIS_HASH, base) != compute_event_hash(
            GENESIS_HASH, mutated
        )

    def test_non_encodable_event_rejected(self):
        with pytest.raises(AuditError):
            compute_event_hash(GENESIS_HASH, {"bad": object()})


# ---------------------------------------------------------------------------
# Appending
# ---------------------------------------------------------------------------


class TestAppend:
    def test_first_event_starts_the_chain(self, log):
        event = log.append(**make_event())
        assert event.seq == 0
        assert event.prev_hash == GENESIS_HASH
        assert event.event_hash == event.recompute_hash()
        assert len(event.event_hash) == 64

    def test_events_link_to_their_predecessor(self, log):
        first = log.append(**make_event())
        second = log.append(**make_event(outcome="abstain"))
        third = log.append(**make_event(outcome="remember"))
        assert second.seq == 1 and second.prev_hash == first.event_hash
        assert third.seq == 2 and third.prev_hash == second.event_hash
        assert log.head_hash() == third.event_hash
        assert log.next_seq() == 3
        assert len(log) == 3

    def test_events_are_returned_in_chain_order(self, log):
        for outcome in ("observe", "abstain", "remember"):
            log.append(**make_event(outcome=outcome))
        assert [e.seq for e in log.events()] == [0, 1, 2]
        assert [e.outcome for e in log.events()] == ["observe", "abstain", "remember"]

    def test_structured_fields_round_trip(self, log):
        scores = {"observe": 0.25, "abstain": 0.75}
        log.append(**make_event(scores=scores, governor_result={"allowed": True, "level": "normal"}))
        stored = log.get(0)
        assert stored.scores == scores
        assert stored.governor_result == {"allowed": True, "level": "normal"}

    def test_scalar_governor_result_is_preserved(self, log):
        log.append(**make_event(governor_result="allowed"))
        assert log.get(0).governor_result == "allowed"

    def test_explicit_timestamp_is_preserved(self, log):
        event = log.append(**make_event(ts="2020-02-02T02:02:02+00:00"))
        assert event.ts == "2020-02-02T02:02:02+00:00"

    def test_timestamp_defaults_to_iso_utc(self, log):
        event = log.append(**make_event(ts=""))
        assert datetime.fromisoformat(event.ts) is not None

    def test_numeric_fields_are_normalised(self, log):
        event = log.append(**make_event(budget_delta=2, goal_version=5))
        assert event.budget_delta == 2.0 and isinstance(event.budget_delta, float)
        assert event.goal_version == "5"

    def test_derived_fields_cannot_be_supplied(self, log):
        for derived in ("seq", "prev_hash", "event_hash"):
            with pytest.raises(AuditError, match="derived"):
                log.append(**make_event(), **{derived: 0})

    def test_unknown_field_is_rejected(self, log):
        with pytest.raises(AuditError, match="unknown"):
            log.append(**make_event(), sudo=True)

    def test_budget_delta_must_be_finite_number(self, log):
        with pytest.raises(AuditError, match="budget_delta"):
            log.append(**make_event(budget_delta=float("nan")))
        with pytest.raises(AuditError, match="budget_delta"):
            log.append(**make_event(budget_delta=True))

    def test_scores_must_be_json_encodable(self, log):
        with pytest.raises(AuditError, match="scores"):
            log.append(**make_event(scores={"bad": object()}))

    def test_empty_log_head_is_genesis(self, log):
        assert len(log) == 0
        assert log.events() == []
        assert log.last() is None
        assert log.head_hash() == GENESIS_HASH
        assert log.next_seq() == 0

    def test_field_list_matches_schema(self):
        assert "seq" in AUDIT_FIELDS and "event_hash" in AUDIT_FIELDS
        assert set(AUDIT_FIELDS) == {
            "seq", "ts", "state_hash", "goal_version", "plan_version",
            "candidate_set_hash", "scores", "policy_version", "governor_result",
            "permit_id", "tool_request_hash", "tool_result_hash", "outcome",
            "budget_delta", "prev_hash", "event_hash",
        }


# ---------------------------------------------------------------------------
# Chain verification
# ---------------------------------------------------------------------------


class TestVerifyChain:
    def test_empty_chain_verifies(self, log):
        result = log.verify_chain()
        assert isinstance(result, ChainVerification)
        assert result.ok is True
        assert bool(result) is True
        assert result.count == 0
        assert result.seq is None

    def test_intact_chain_verifies(self, log):
        for _ in range(3):
            log.append(**make_event())
        result = log.verify_chain()
        assert result.ok is True
        assert result.count == 3
        assert result.seq == 2

    @pytest.mark.parametrize(
        "column, value",
        [
            ("ts", "2099-01-01T00:00:00+00:00"),
            ("state_hash", "edited"),
            ("goal_version", "edited"),
            ("plan_version", "edited"),
            ("candidate_set_hash", "edited"),
            ("scores", '{"observe":0.1,"abstain":0.9}'),
            ("policy_version", "edited"),
            ("governor_result", '{"allowed":false}'),
            ("permit_id", "edited"),
            ("tool_request_hash", "edited"),
            ("tool_result_hash", "edited"),
            ("outcome", "abstain"),
            ("budget_delta", 999.0),
            ("prev_hash", "f" * 64),
            ("event_hash", "f" * 64),
        ],
    )
    def test_editing_any_past_field_breaks_the_chain(self, log, tmp_path, column, value):
        for _ in range(3):
            log.append(**make_event())
        _sql(tmp_path / "audit.sqlite", f"UPDATE audit_events SET {column}=? WHERE seq=1", (value,))

        result = log.verify_chain()
        assert result.ok is False
        assert result.reason

    def test_editing_the_first_row_breaks_the_chain(self, log, tmp_path):
        log.append(**make_event())
        log.append(**make_event())
        _sql(tmp_path / "audit.sqlite", "UPDATE audit_events SET outcome='tampered' WHERE seq=0")

        assert log.verify_chain().ok is False

    def test_deleting_a_middle_row_breaks_the_chain(self, log, tmp_path):
        for _ in range(3):
            log.append(**make_event())
        _sql(tmp_path / "audit.sqlite", "DELETE FROM audit_events WHERE seq=1")

        result = log.verify_chain()
        assert result.ok is False
        assert "seq" in result.reason or "link" in result.reason

    def test_deleting_the_last_row_is_detected_by_the_head(self, log, tmp_path):
        for _ in range(3):
            log.append(**make_event())
        _sql(tmp_path / "audit.sqlite", "DELETE FROM audit_events WHERE seq=2")

        result = log.verify_chain()
        assert result.ok is False
        assert "count" in result.reason

    def test_module_verify_chain_accepts_events_and_dicts(self, log):
        for _ in range(2):
            log.append(**make_event())
        events = log.events()
        assert verify_chain(events).ok is True
        assert verify_chain([e.as_dict() for e in events]).ok is True

    def test_module_verify_chain_detects_a_tampered_dict(self, log):
        log.append(**make_event())
        log.append(**make_event())
        tampered = [e.as_dict() for e in log.events()]
        tampered[0]["outcome"] = "tampered"
        assert verify_chain(tampered).ok is False

    def test_module_verify_chain_detects_truncation_with_expected_head(self, log):
        for _ in range(3):
            log.append(**make_event())
        truncated = log.events()[:-1]
        assert verify_chain(
            truncated, expected_count=3, expected_head=log.head_hash()
        ).ok is False

    def test_module_verify_chain_accepts_custom_genesis(self):
        draft = AuditEvent(seq=0, prev_hash="custom-genesis")
        sealed = AuditEvent(
            event_hash=compute_event_hash("custom-genesis", draft.payload()), **draft.payload()
        )
        assert verify_chain([sealed], genesis_hash="custom-genesis").ok is True
        assert verify_chain([sealed], genesis_hash=GENESIS_HASH).ok is False


# ---------------------------------------------------------------------------
# Behavioral replay
# ---------------------------------------------------------------------------


class TestReplay:
    def _recorded(self, log):
        rows = [
            ({"observe": 0.7, "abstain": 0.3}, {"allowed": True}, "observe"),
            ({"abstain": 0.6, "observe": 0.4}, {"allowed": True}, "abstain"),
            ({"set_vision_fps": 0.55, "abstain": 0.45}, {"allowed": False}, "abstain"),
            ({"remember": 0.9, "observe": 0.1}, {"allowed": True}, "remember"),
        ]
        for scores, governor_result, outcome in rows:
            log.append(**make_event(scores=scores, governor_result=governor_result, outcome=outcome))
        return log.events()

    def test_replay_reproduces_recorded_decisions(self, log):
        events = self._recorded(log)
        recorded = [event.outcome for event in events]
        assert replay(events, decide_from_event) == recorded

    def test_replay_is_deterministic(self, log):
        events = self._recorded(log)
        assert replay(events, decide_from_event) == replay(events, decide_from_event)

    def test_replay_uses_the_decision_function(self, log):
        events = self._recorded(log)
        assert replay(events, lambda event: "forced") == ["forced"] * len(events)

    def test_replay_preserves_order_and_passes_events(self, log):
        log.append(**make_event(outcome="a"))
        log.append(**make_event(outcome="b"))
        seen = []

        def spy(event):
            assert isinstance(event, AuditEvent)
            seen.append(event.outcome)
            return event.outcome

        assert replay(log.events(), spy) == ["a", "b"]
        assert seen == ["a", "b"]

    def test_replay_of_empty_sequence_is_empty(self, log):
        assert replay([], decide_from_event) == []

    def test_replay_rejects_non_callable(self, log):
        log.append(**make_event())
        with pytest.raises(AuditError, match="callable"):
            replay(log.events(), None)

    def test_replay_of_persisted_events_matches_live(self, tmp_path):
        with AuditLog(tmp_path / "audit.sqlite") as log:
            events = self._recorded(log)
            live = replay(events, decide_from_event)
        with AuditLog(tmp_path / "audit.sqlite") as reopened:
            persisted = replay(reopened.events(), decide_from_event)
        assert live == persisted


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_chain_survives_reopen_and_continues(self, tmp_path):
        path = tmp_path / "audit.sqlite"
        with AuditLog(path) as log:
            first = log.append(**make_event())
            log.append(**make_event())
            expected_head = log.head_hash()

        with AuditLog(path) as reopened:
            assert reopened.verify_chain().ok is True
            assert reopened.head_hash() == expected_head
            assert reopened.next_seq() == 2
            third = reopened.append(**make_event(outcome="third"))
            assert third.seq == 2
            assert third.prev_hash == reopened.get(1).event_hash
            assert reopened.verify_chain().ok is True
        assert first.seq == 0

    def test_events_round_trip_through_sqlite(self, tmp_path):
        path = tmp_path / "audit.sqlite"
        with AuditLog(path) as log:
            log.append(**make_event(scores={"observe": 1.0}))
        with AuditLog(path) as reopened:
            stored = reopened.get(0)
            assert stored.as_dict() == reopened.events()[0].as_dict()
            assert stored.scores == {"observe": 1.0}

    def test_stored_scores_are_canonical_json(self, tmp_path):
        path = tmp_path / "audit.sqlite"
        with AuditLog(path) as log:
            log.append(**make_event(scores={"b": 0.2, "a": 0.8}))
        conn = sqlite3.connect(str(path))
        try:
            raw = conn.execute("SELECT scores FROM audit_events WHERE seq=0").fetchone()[0]
        finally:
            conn.close()
        assert raw == json.dumps({"a": 0.8, "b": 0.2}, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


class TestAuditTickFlow:
    def test_recorded_ticks_verify_and_replay_identically(self, tmp_path):
        path = tmp_path / "audit.sqlite"
        scores_seq = [
            {"observe": 0.8, "abstain": 0.2},
            {"abstain": 0.7, "observe": 0.3},
            {"reflect": 0.6, "abstain": 0.4},
        ]
        recorded = []
        with AuditLog(path) as log:
            for index, scores in enumerate(scores_seq):
                choice = max(scores, key=scores.get)
                governor_result = {"allowed": True, "level": "normal"}
                outcome = choice if governor_result["allowed"] else "abstain"
                recorded.append(outcome)
                log.append(
                    **make_event(
                        ts=f"2026-10-05T00:00:0{index}+00:00",
                        state_hash=f"{index}" * 64,
                        goal_version=f"goal-{index}",
                        plan_version=f"plan-{index}",
                        candidate_set_hash=f"{index + 1}" * 64,
                        scores=scores,
                        governor_result=governor_result,
                        permit_id=f"permit-{index}",
                        outcome=outcome,
                        budget_delta=float(index),
                    )
                )
            assert log.verify_chain().ok is True
            assert replay(log.events(), decide_from_event) == recorded

        with AuditLog(path) as reopened:
            assert reopened.verify_chain().ok is True
            assert replay(reopened.events(), decide_from_event) == recorded

    def test_tampering_with_a_recorded_tick_is_caught(self, tmp_path):
        path = tmp_path / "audit.sqlite"
        with AuditLog(path) as log:
            for scores in ({"observe": 0.8, "abstain": 0.2}, {"abstain": 0.7, "observe": 0.3}):
                log.append(**make_event(scores=scores, outcome=max(scores, key=scores.get)))
        _sql(path, "UPDATE audit_events SET scores='{\"abstain\":1.0}' WHERE seq=0")

        with AuditLog(path) as reopened:
            assert reopened.verify_chain().ok is False

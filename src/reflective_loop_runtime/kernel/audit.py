"""Hash-chained append-only audit log with behavioral replay.

Every decision tick appends exactly one immutable row to a SQLite table. A row
carries the complete, canonical record of what the loop saw and chose:

``seq``, ``ts``, ``state_hash``, ``goal_version``, ``plan_version``,
``candidate_set_hash``, ``scores``, ``policy_version``, ``governor_result``,
``permit_id``, ``tool_request_hash``, ``tool_result_hash``, ``outcome`` and
``budget_delta``.

Integrity is a hash chain::

    event_hash = sha256(prev_hash || canonical_json(event))

where ``event`` is the canonical record -- every field except ``event_hash`` --
and ``prev_hash`` is the previous row's ``event_hash`` (the genesis value for the
first row). Because each link transitively covers every earlier row, editing or
deleting any past row breaks verification at that point. A tiny ``audit_head``
row records the committed count and head hash, so truncation of the most recent
row is detectable as well.

``replay(events, decide_fn)`` re-runs a pure decision function over recorded
events, returning the decisions it derives so a caller can compare them with
what the log recorded: behavioral replay, not merely byte-level integrity.

The module is stdlib-only, deterministic, and performs no I/O of its own beyond
the SQLite file it is explicitly asked to open.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .policy import KeyedHmacSigner, canonical_json

__all__ = [
    "AUDIT_FIELDS",
    "GENESIS_HASH",
    "AuditError",
    "AuditEvent",
    "AuditLog",
    "ChainVerification",
    "canonical_json",
    "compute_event_hash",
    "replay",
    "utc_now",
    "verify_chain",
]

# The all-zero hash used as the ``prev_hash`` of the very first row.
GENESIS_HASH = "0" * 64

# The complete, ordered field set of one audit row. ``seq``, ``prev_hash`` and
# ``event_hash`` are derived by ``AuditLog.append`` and may not be supplied.
AUDIT_FIELDS = (
    "seq",
    "ts",
    "state_hash",
    "goal_version",
    "plan_version",
    "candidate_set_hash",
    "scores",
    "policy_version",
    "governor_result",
    "permit_id",
    "tool_request_hash",
    "tool_result_hash",
    "outcome",
    "budget_delta",
    "prev_hash",
    "event_hash",
)
_DERIVED_FIELDS = ("seq", "prev_hash", "event_hash")
_APPEND_FIELDS = tuple(name for name in AUDIT_FIELDS if name not in _DERIVED_FIELDS)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_events (
    seq                 INTEGER PRIMARY KEY,
    ts                  TEXT NOT NULL,
    state_hash          TEXT NOT NULL,
    goal_version        TEXT NOT NULL,
    plan_version        TEXT NOT NULL,
    candidate_set_hash  TEXT NOT NULL,
    scores              TEXT NOT NULL,
    policy_version      TEXT NOT NULL,
    governor_result     TEXT NOT NULL,
    permit_id           TEXT NOT NULL,
    tool_request_hash   TEXT NOT NULL,
    tool_result_hash    TEXT NOT NULL,
    outcome             TEXT NOT NULL,
    budget_delta        REAL NOT NULL,
    prev_hash           TEXT NOT NULL,
    event_hash          TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_head (
    id        INTEGER PRIMARY KEY CHECK (id = 1),
    count     INTEGER NOT NULL,
    head_hash TEXT NOT NULL
);
"""


class AuditError(ValueError):
    """Raised when a record cannot be interpreted or safely appended."""


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Value normalisation
# ---------------------------------------------------------------------------


def _as_text(value: Any, name: str) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        raise AuditError(f"{name} must be text, not a boolean")
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    raise AuditError(f"{name} must be text")


def _as_seq(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AuditError("seq must be an integer")
    if value < 0:
        raise AuditError("seq must be non-negative")
    return value


def _as_budget(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuditError("budget_delta must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise AuditError("budget_delta must be finite")
    return number


def _load_jsonish(value: Any) -> Any:
    """Decode a stringly-encoded JSON value; pass anything else through."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _json_value(value: Any, name: str) -> Any:
    """Return a plain, canonically round-tripped JSON value.

    Normalising through the canonical encoder means nested tuples, integer-keyed
    maps and duplicate ordering all collapse to one stable representation, so the
    hash of an event is independent of how the caller built it.
    """
    value = _load_jsonish(value)
    try:
        return json.loads(canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{name} is not canonically-JSON encodable: {exc}") from exc


_DEFAULTS: dict[str, Any] = {
    "seq": 0,
    "ts": "",
    "state_hash": "",
    "goal_version": "",
    "plan_version": "",
    "candidate_set_hash": "",
    "scores": {},
    "policy_version": "",
    "governor_result": {},
    "permit_id": "",
    "tool_request_hash": "",
    "tool_result_hash": "",
    "outcome": "",
    "budget_delta": 0.0,
    "prev_hash": GENESIS_HASH,
}


def _event_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    """Normalise a record to the exact dict that is hashed and stored."""
    get = lambda key: record[key] if key in record else _DEFAULTS[key]  # noqa: E731
    return {
        "seq": _as_seq(get("seq")),
        "ts": _as_text(get("ts"), "ts"),
        "state_hash": _as_text(get("state_hash"), "state_hash"),
        "goal_version": _as_text(get("goal_version"), "goal_version"),
        "plan_version": _as_text(get("plan_version"), "plan_version"),
        "candidate_set_hash": _as_text(get("candidate_set_hash"), "candidate_set_hash"),
        "scores": _json_value(get("scores"), "scores"),
        "policy_version": _as_text(get("policy_version"), "policy_version"),
        "governor_result": _json_value(get("governor_result"), "governor_result"),
        "permit_id": _as_text(get("permit_id"), "permit_id"),
        "tool_request_hash": _as_text(get("tool_request_hash"), "tool_request_hash"),
        "tool_result_hash": _as_text(get("tool_result_hash"), "tool_result_hash"),
        "outcome": _as_text(get("outcome"), "outcome"),
        "budget_delta": _as_budget(get("budget_delta")),
        "prev_hash": _as_text(get("prev_hash"), "prev_hash"),
    }


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def _chain_key_bytes(key: Any) -> bytes:
    if isinstance(key, str):
        return key.encode("utf-8")
    if isinstance(key, (bytes, bytearray)):
        return bytes(key)
    raise AuditError("audit chain key must be str or bytes")


def _chain_signer(key: Any) -> KeyedHmacSigner:
    """Return a GC-opaque signer for ``key``.

    ``key`` may be the raw material (kept for callers that hold it) or an
    already-absorbed :class:`~.policy.KeyedHmacSigner`. Absorbing the key into
    opaque ``hashlib`` state is what keeps it out of an attribute walk and out
    of ``gc.get_referents`` on a model-facing :class:`AuditLog`.
    """
    if isinstance(key, KeyedHmacSigner):
        return key
    return KeyedHmacSigner(_chain_key_bytes(key))


def compute_event_hash(prev_hash: Any, event: Any, *, key: Any = None) -> str:
    """Return the hash linking ``event`` to ``prev_hash``.

    Without a ``key`` this is the legacy ``sha256(prev_hash || canonical_json)``
    digest: tamper-evident against edits but forgeable by anyone who can rewrite
    the database and recompute the chain. When a ``key`` is supplied the digest
    becomes ``HMAC-SHA256(key, prev_hash || canonical_json)``, so an attacker who
    lacks the key cannot mint a replacement chain that verifies. ``key`` may be
    raw key material or a :class:`~.policy.KeyedHmacSigner` that already holds
    it opaquely; either way the digest is identical.
    """
    prev_hash = _as_text(prev_hash, "prev_hash")
    try:
        encoded = canonical_json(event)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"event is not canonically-JSON encodable: {exc}") from exc
    material = (prev_hash + encoded).encode("utf-8")
    if key is None:
        return hashlib.sha256(material).hexdigest()
    return _chain_signer(key).sign(material)


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditEvent:
    """One immutable, hash-chained audit row."""

    seq: int = 0
    ts: str = ""
    state_hash: str = ""
    goal_version: str = ""
    plan_version: str = ""
    candidate_set_hash: str = ""
    scores: Any = field(default_factory=dict)
    policy_version: str = ""
    governor_result: Any = field(default_factory=dict)
    permit_id: str = ""
    tool_request_hash: str = ""
    tool_result_hash: str = ""
    outcome: str = ""
    budget_delta: float = 0.0
    prev_hash: str = GENESIS_HASH
    event_hash: str = ""

    def as_mapping(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in AUDIT_FIELDS}

    def payload(self) -> dict[str, Any]:
        """The exact canonical dict covered by ``event_hash`` (no ``event_hash``)."""
        return _event_payload(self.as_mapping())

    def as_dict(self) -> dict[str, Any]:
        """A plain dict including all fields, suitable for logging or replay."""
        out = self.payload()
        out["event_hash"] = self.event_hash
        return out

    to_dict = as_dict

    def recompute_hash(self, *, key: Any = None) -> str:
        """Recompute this event's hash from its own fields (keyed when given)."""
        return compute_event_hash(self.prev_hash, self.payload(), key=key)


def _coerce_event(raw: Any) -> AuditEvent:
    if isinstance(raw, AuditEvent):
        return raw
    if not isinstance(raw, Mapping):
        raise AuditError("event must be an AuditEvent or a mapping")
    data = dict(raw)
    for key in ("scores", "governor_result"):
        if key in data:
            data[key] = _load_jsonish(data[key])
    payload = _event_payload(data)
    return AuditEvent(
        event_hash=_as_text(data.get("event_hash"), "event_hash"),
        **payload,
    )


def _row_to_event(row: sqlite3.Row) -> AuditEvent:
    return AuditEvent(
        seq=int(row["seq"]),
        ts=row["ts"],
        state_hash=row["state_hash"],
        goal_version=row["goal_version"],
        plan_version=row["plan_version"],
        candidate_set_hash=row["candidate_set_hash"],
        scores=_load_jsonish(row["scores"]),
        policy_version=row["policy_version"],
        governor_result=_load_jsonish(row["governor_result"]),
        permit_id=row["permit_id"],
        tool_request_hash=row["tool_request_hash"],
        tool_result_hash=row["tool_result_hash"],
        outcome=row["outcome"],
        budget_delta=float(row["budget_delta"]),
        prev_hash=row["prev_hash"],
        event_hash=row["event_hash"],
    )


def _event_row(event: AuditEvent) -> tuple[Any, ...]:
    return (
        event.seq,
        event.ts,
        event.state_hash,
        event.goal_version,
        event.plan_version,
        event.candidate_set_hash,
        canonical_json(event.scores),
        event.policy_version,
        canonical_json(event.governor_result),
        event.permit_id,
        event.tool_request_hash,
        event.tool_result_hash,
        event.outcome,
        event.budget_delta,
        event.prev_hash,
        event.event_hash,
    )


_INSERT_SQL = (
    "INSERT INTO audit_events ("
    "seq, ts, state_hash, goal_version, plan_version, candidate_set_hash, scores,"
    " policy_version, governor_result, permit_id, tool_request_hash,"
    " tool_result_hash, outcome, budget_delta, prev_hash, event_hash"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChainVerification:
    """Outcome of walking a hash chain."""

    ok: bool
    reason: str = ""
    seq: int | None = None
    count: int = 0

    def __bool__(self) -> bool:  # so ``if verify_chain(...):`` reads naturally
        return self.ok


def verify_chain(
    events: Iterable[Any],
    *,
    genesis_hash: str = GENESIS_HASH,
    expected_count: int | None = None,
    expected_head: str | None = None,
    key: Any = None,
) -> ChainVerification:
    """Walk ``events`` and report the first integrity violation.

    Detects a reordered/``seq``-gapped chain (a deleted middle row), a broken
    back-link, and any edited field (via a recomputed-hash mismatch). When
    ``expected_count``/``expected_head`` are supplied -- as ``AuditLog`` does from
    its committed head -- truncation of the final row is caught too. A ``key``
    recomputes the links as HMACs so an unkeyed forgery of a keyed chain fails.
    """
    materialized = [_coerce_event(event) for event in events]
    total = len(materialized)
    expected_prev = genesis_hash

    for position, event in enumerate(materialized):
        if event.seq != position:
            return ChainVerification(
                False,
                f"sequence discontinuity: row {position} has seq {event.seq}",
                event.seq,
                total,
            )
        if event.prev_hash != expected_prev:
            return ChainVerification(
                False, f"broken hash link at seq {event.seq}", event.seq, total
            )
        if event.recompute_hash(key=key) != event.event_hash:
            return ChainVerification(
                False, f"event hash mismatch at seq {event.seq}", event.seq, total
            )
        expected_prev = event.event_hash

    if expected_count is not None and total != expected_count:
        return ChainVerification(
            False,
            f"row count {total} does not match committed head count {expected_count}",
            None,
            total,
        )
    if expected_head is not None and expected_prev != expected_head:
        return ChainVerification(
            False,
            f"chain head {expected_prev!r} does not match committed head {expected_head!r}",
            None,
            total,
        )

    last_seq = materialized[-1].seq if materialized else None
    return ChainVerification(True, "", last_seq, total)


# ---------------------------------------------------------------------------
# Behavioral replay
# ---------------------------------------------------------------------------


def replay(events: Iterable[Any], decide_fn: Callable[[AuditEvent], Any]) -> list[Any]:
    """Re-run ``decide_fn`` over recorded events and return its decisions.

    ``decide_fn`` receives each :class:`AuditEvent` in order. Because a recorded
    event carries the same inputs the live loop saw (candidate scores, governor
    result, versions, hashes), a deterministic decision function reproduces the
    recorded outcomes exactly; a caller compares the returned list against what
    the log recorded to detect behavioral drift.
    """
    if not callable(decide_fn):
        raise AuditError("decide_fn must be callable")
    materialized = [_coerce_event(event) for event in events]
    return [decide_fn(event) for event in materialized]


# ---------------------------------------------------------------------------
# Log
# ---------------------------------------------------------------------------


class AuditLog:
    """Append-only, hash-chained audit log backed by a SQLite table.

    The log is write-once from the caller's perspective: there is no public API
    to update or delete a row. ``append`` is the only mutator and it always
    extends the chain at the committed head.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        genesis_hash: str = GENESIS_HASH,
        clock: Callable[[], str] = utc_now,
        key: Any = None,
    ):
        self.genesis_hash = _as_text(genesis_hash, "genesis_hash")
        self._clock = clock
        # The chain key is absorbed here and never stored: no attribute of this
        # log, and nothing reachable from it, holds the key bytes.
        self._chain = None if key is None else KeyedHmacSigner(_chain_key_bytes(key))
        target = str(path)
        if target != ":memory:":
            Path(target).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(target)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_SCHEMA)
        self.db.execute(
            "INSERT OR IGNORE INTO audit_head (id, count, head_hash) VALUES (1, 0, ?)",
            (self.genesis_hash,),
        )
        self.db.commit()

    # -- lifecycle ---------------------------------------------------------
    @property
    def keyed(self) -> bool:
        """Whether this log links rows with HMACs rather than plain digests."""
        return self._chain is not None

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> AuditLog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- head bookkeeping --------------------------------------------------
    def _head(self) -> sqlite3.Row:
        row = self.db.execute("SELECT count, head_hash FROM audit_head WHERE id=1").fetchone()
        if row is None:
            raise AuditError("audit head row is missing")
        return row

    def head_hash(self) -> str:
        """The committed head hash (``genesis_hash`` while the log is empty)."""
        return str(self._head()["head_hash"])

    def next_seq(self) -> int:
        """The sequence number the next appended row will receive."""
        return int(self._head()["count"])

    # -- writes ------------------------------------------------------------
    def append(self, **fields: Any) -> AuditEvent:
        """Append one event and extend the chain.

        Only non-derived fields may be passed; ``seq``, ``prev_hash`` and
        ``event_hash`` are computed here so a caller cannot forge a link.
        """
        unknown = sorted(set(fields) - set(_APPEND_FIELDS))
        if unknown:
            raise AuditError(f"unknown or derived audit field(s): {unknown}")

        record = dict(fields)
        record["seq"] = self.next_seq()
        record["prev_hash"] = self.head_hash()
        if not record.get("ts"):
            record["ts"] = self._clock()

        payload = _event_payload(record)
        digest = compute_event_hash(payload["prev_hash"], payload, key=self._chain)
        event = AuditEvent(event_hash=digest, **payload)

        with self.db:
            self.db.execute(_INSERT_SQL, _event_row(event))
            self.db.execute(
                "INSERT INTO audit_head (id, count, head_hash) VALUES (1, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET count=excluded.count, head_hash=excluded.head_hash",
                (event.seq + 1, digest),
            )
        return event

    # -- reads -------------------------------------------------------------
    def events(self) -> list[AuditEvent]:
        """All rows in chain order."""
        rows = self.db.execute("SELECT * FROM audit_events ORDER BY seq").fetchall()
        return [_row_to_event(row) for row in rows]

    def get(self, seq: int) -> AuditEvent | None:
        row = self.db.execute("SELECT * FROM audit_events WHERE seq=?", (int(seq),)).fetchone()
        return _row_to_event(row) if row is not None else None

    def last(self) -> AuditEvent | None:
        row = self.db.execute("SELECT * FROM audit_events ORDER BY seq DESC LIMIT 1").fetchone()
        return _row_to_event(row) if row is not None else None

    def __len__(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0])

    # -- integrity ---------------------------------------------------------
    def verify_chain(self) -> ChainVerification:
        """Verify every stored row against the chain and committed head."""
        head = self._head()
        return verify_chain(
            self.events(),
            genesis_hash=self.genesis_hash,
            expected_count=int(head["count"]),
            expected_head=str(head["head_hash"]),
            key=self._chain,
        )

"""One-shot execution permits and the Governor-only issuer that mints them.

A :class:`ExecutionPermit` is the kernel's proof that a concrete tool call was
authorised at a point in time: it binds a tool, a canonical hash of the
arguments, the policy version it was issued under, the risk class the manifest
declared, and the budget reservation backing it, and it carries an HMAC over all
of that. Permits are single-use: the nonce is consumed atomically in SQLite the
first time it is presented, so a replayed permit is rejected.

Trust model
-----------
* Permits can only be *minted* by a :class:`PermitIssuer`, an object owned by
  the deterministic Governor that holds the HMAC key. The key is write-only from
  the outside: the issuer exposes ``sign``/``verify`` but never returns it, and
  :class:`PermitService` -- the model-facing interface -- never exposes the
  issuer itself.
* Issuance fails closed. A policy that is not in ``ENFORCE`` mode mints nothing,
  an unknown tool/operation or an off-schema argument set is refused via
  :mod:`~.manifest`, an ``A3`` operation needs a matching standing mandate in
  the policy, and the budget manager (K5) must be able to reserve before a
  permit exists.
* Verification re-checks the MAC, the expiry, the tool/argument binding, and the
  current policy version before consuming the nonce.
* A permit is bound to the service instance that minted it. Nothing a restarted
  process can read measures the time it spent dead, so a service that finds a
  permit on disk after a restart cannot tell a live permit from a long-expired
  one whenever the wall clock has not moved since it was minted. Every permit
  therefore carries the random identity of its issuer, and any other instance --
  including the one that replaced it after a restart -- refuses it.
* Expiry is judged against a monotonic time. The wall clock is attacker-
  reachable, so the effective now is the furthest-forward moment the process can
  vouch for: ``max(clock(), high_water_mark, anchor_wall + elapsed_monotonic)``,
  where the mark is the furthest-forward wall moment ever written to the permit
  store and the anchor is the effective moment of the previous observation plus
  the time this process has run since, read from a counter no wall-clock change
  can step. Idle time therefore still advances time inside a process, and across
  a restart the persisted mark is the floor. A clock rolled back past the
  tolerated skew fails closed: nothing is authorised while the clock is
  untrusted, and that diagnosis is never masked by a plain expiry.

The module is stdlib-only, deterministic, and performs no I/O beyond the SQLite
nonce and clock tables it is explicitly asked to open.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import secrets
import sqlite3
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .budget import RESOURCES, BudgetService, utc_now
from .manifest import DEFAULT_MANIFEST, ManifestError, ToolManifest
from .policy import (
    KeyedHmacSigner,
    MissionPolicy,
    canonical_bytes,
    canonical_json,
    verified_authority,
)

__all__ = [
    "PERMIT_FIELDS",
    "ExecutionPermit",
    "PermitClockError",
    "PermitDenied",
    "PermitError",
    "PermitIssuer",
    "PermitReplayError",
    "PermitService",
    "PermitVerificationError",
    "args_hash",
]

# The complete, ordered field set of a permit. Every field except ``mac`` is
# covered by the MAC. ``op``, ``manifest_digest`` and ``policy_digest`` bind the
# permit to the exact operation and the exact authority (manifest + policy) it
# was issued under, so a permit cannot be replayed for a different operation or
# against a swapped manifest/policy. ``instance_id`` binds it to the one
# :class:`PermitService` that minted it: the clock defences below can only
# measure time inside a single process, so a permit that outlived its service is
# refused outright rather than re-timed by a clock that may not have moved.
PERMIT_FIELDS = (
    "permit_id",
    "nonce",
    "goal_id",
    "plan_id",
    "step_id",
    "tool",
    "op",
    "capability",
    "normalized_args_hash",
    "manifest_digest",
    "policy_digest",
    "state_version",
    "policy_version",
    "risk_class",
    "budget_reservation",
    "issued_at",
    "expires_at",
    "idempotency_key",
    "instance_id",
    "mac",
)
_UNSIGNED_FIELDS = tuple(name for name in PERMIT_FIELDS if name != "mac")

# Where a persistent (file-backed) replay ledger lives when the caller does not
# name a path. The ledger must outlive the process or a permit replay after a
# restart would go undetected.
_DEFAULT_STATE_DIR = os.environ.get("RLR_STATE_DIR", os.path.join(tempfile.gettempdir(),
                                                                  "reflective-loop-runtime"))

# Domain-separation tag mixed into the signed bytes. A permit MAC can therefore
# never be replayed as a MAC over any other message, even one with an identical
# field layout.
_PERMIT_MAC_DOMAIN = b"reflective-loop-runtime.execution-permit.v1"

_NONCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS used_nonces (
    nonce       TEXT PRIMARY KEY,
    permit_id   TEXT NOT NULL,
    consumed_at TEXT NOT NULL
);
"""

# How far the wall clock may sit *behind* the effective time the permit store
# has vouched for before it is treated as a rollback rather than ordinary NTP
# jitter.
_CLOCK_SKEW_SECONDS = 300.0

# One row per trusted time source holding the latest moment that source ever
# observed in this store. Created idempotently, exactly like the nonce ledger,
# so an existing permit database gains the marks without a migration.
_CLOCK_SCHEMA = """
CREATE TABLE IF NOT EXISTS clock_hwm (
    source      TEXT PRIMARY KEY,
    observed_at TEXT NOT NULL
);
"""


class PermitError(ValueError):
    """Base class for every permit failure."""


class PermitDenied(PermitError):
    """Raised when the service refuses to mint a permit."""


class PermitVerificationError(PermitError):
    """Raised when a permit fails MAC, expiry, binding, or policy checks."""


class PermitReplayError(PermitVerificationError):
    """Raised when a permit nonce has already been consumed."""


class PermitClockError(PermitVerificationError):
    """Raised when the wall clock has been rolled back past the tolerated skew.

    A clock far behind the time the permit store has already observed cannot be
    used to judge expiry: a permit presented under such a clock may already be
    dead. Verification therefore fails closed rather than guessing, and the
    rejection travels the ordinary audit/record path of any other refusal.
    """


def _require_verified_authority(policy: MissionPolicy) -> None:
    """Refuse a policy that asserts authority without a verified signature.

    Authority is provenance, not a self-declared flag: it is only ever granted
    by :func:`~.policy.load_policy_bytes` after a signature over the canonical
    bundle bytes verifies. Neither the ``mode`` nor the ``signed`` field of a
    policy is trusted here, so a hand-built dataclass (``MissionPolicy(mode=
    ENFORCE, signed=True)``), a ``dataclasses.replace``d copy, or a policy whose
    fields were edited after loading carries no operator authority and may not
    be installed on a permit service. Withholding authority by downgrading to
    ``OBSERVATION_ONLY`` is always allowed.
    """
    if policy.enforced and not verified_authority(policy):
        raise PermitDenied(
            "policy authority must come from a verified operator signature over "
            "the canonical policy bytes; a self-asserted ENFORCE policy is refused"
        )


# ---------------------------------------------------------------------------
# Monotonic time
# ---------------------------------------------------------------------------


def _time_source(clock: Any) -> str:
    """A stable identity for the time source a permit store is being fed by.

    One ledger can legitimately be written by services that read *different*
    clocks -- a deployment and a simulation sharing a file, say -- and a mark
    taken from one of them would look like a rollback to the other. Marks are
    therefore partitioned by source, and a service only ever consults the mark
    written by its own clock. The identity is derived from the callable itself
    (never from its instance), so the default :func:`~.budget.utc_now` keeps one
    stable domain across every process: the rollback protection a real
    deployment relies on is unaffected by the partitioning.
    """
    owner = clock if hasattr(clock, "__qualname__") else type(clock)
    return f"{getattr(owner, '__module__', '?')}.{getattr(owner, '__qualname__', '?')}"


class _TimeHighWaterMark:
    """The furthest-forward moment one time source ever observed in this store.

    A wall clock can be rolled back -- by an operator, by NTP correcting a large
    drift, or by whoever controls the host -- and a permit is only as trustworthy
    as the clock that judges its expiry. A bare ``datetime`` cannot distinguish
    "this permit is still live" from "the clock went back to the year 2000", so
    the store remembers the latest moment its own time source has seen. Expiry is
    then judged against the effective now -- ``max(clock(), hwm, anchor)`` --
    which makes it monotonic under a backward jump, and a clock further behind
    the effective time than the tolerated skew fails closed rather than quietly
    reviving a stale permit.
    """

    __slots__ = ("_db", "_lock", "_source")

    def __init__(self, db: Any, lock: Any, source: str):
        self._db = db
        self._lock = lock
        self._source = source
        # Existing databases may contain variable precision or timezone
        # offsets. Canonicalize the legacy row under a SQLite write lock before
        # lexical UTC comparisons are introduced by the conditional UPSERT.
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute(
                    "SELECT observed_at FROM clock_hwm WHERE source=?", (source,)
                ).fetchone()
                if row is not None:
                    canonical = _parse_time(row[0]).astimezone(
                        timezone.utc
                    ).isoformat(timespec="microseconds")
                    self._db.execute(
                        "UPDATE clock_hwm SET observed_at=? WHERE source=?",
                        (canonical, source),
                    )
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

    def observed(self) -> datetime | None:
        """The stored high-water mark, or ``None`` before anything is recorded."""
        with self._lock:
            row = self._db.execute(
                "SELECT observed_at FROM clock_hwm WHERE source=?", (self._source,)
            ).fetchone()
        return None if row is None else _parse_time(row[0])

    def observe(self, moment: datetime) -> datetime:
        """Persist the maximum observed instant atomically across processes.

        Store fixed-width UTC ISO timestamps for order-preserving SQL compares.
        A single conditional UPSERT prevents concurrent connections from
        overwriting newer observations with an older timestamp.
        """
        stamp = _parse_time(moment).astimezone(timezone.utc).isoformat(
            timespec="microseconds"
        )
        with self._lock:
            self._db.execute(
                """INSERT INTO clock_hwm (source, observed_at) VALUES (?, ?)
                ON CONFLICT(source) DO UPDATE SET observed_at =
                  CASE WHEN excluded.observed_at > clock_hwm.observed_at
                       THEN excluded.observed_at ELSE clock_hwm.observed_at END""",
                (self._source, stamp),
            )
            # Return the persisted maximum: another process may have already
            # advanced the mark beyond this caller's observed moment.
            row = self._db.execute(
                "SELECT observed_at FROM clock_hwm WHERE source=?", (self._source,)
            ).fetchone()
            return _parse_time(row[0])


# ---------------------------------------------------------------------------
# Canonical helpers
# ---------------------------------------------------------------------------


def _require_json_safe(value: Any, path: str = "args") -> None:
    """Reject values that canonical JSON cannot represent faithfully.

    ``json.dumps`` silently coerces non-string mapping keys to strings and emits
    non-standard ``NaN``/``Infinity`` tokens by default; either would let two
    distinct argument sets share one canonical encoding (and therefore one
    hash). Both are refused so the args hash is injective.
    """
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise PermitError(
                    f"{path} keys must be strings, got {type(key).__name__}"
                )
            _require_json_safe(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_json_safe(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise PermitError(f"{path} must be a finite number")


def _canonical_args(args: Any) -> Any:
    """Return a canonically-JSON-encodable copy of ``args``."""
    if args is None:
        args = {}
    if not isinstance(args, Mapping):
        raise PermitError("args must be a mapping")
    _require_json_safe(args)
    try:
        return json.loads(canonical_json(args))
    except (TypeError, ValueError) as exc:
        raise PermitError(f"args are not canonically-JSON encodable: {exc}") from exc


def args_hash(args: Any) -> str:
    """Return ``sha256(canonical_json(args))`` as lowercase hex.

    The same function is used at issuance and verification, so any change to an
    argument changes the hash.
    """
    return hashlib.sha256(canonical_bytes(_canonical_args(args))).hexdigest()


def _parse_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise PermitError(f"timestamp {value!r} is not ISO-8601") from exc
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    raise PermitError(f"timestamp {value!r} must be a datetime or ISO-8601 string")


def _add_seconds(moment: Any, seconds: float) -> str:
    return (_parse_time(moment) + timedelta(seconds=seconds)).isoformat()


# ---------------------------------------------------------------------------
# Permit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionPermit:
    """An immutable, single-use authorisation for one tool call."""

    permit_id: str = ""
    nonce: str = ""
    goal_id: str = ""
    plan_id: str = ""
    step_id: str = ""
    tool: str = ""
    op: str = ""
    capability: str = ""
    normalized_args_hash: str = ""
    manifest_digest: str = ""
    policy_digest: str = ""
    state_version: str = ""
    policy_version: str = ""
    risk_class: str = "A0"
    budget_reservation: str | None = None
    issued_at: str = ""
    expires_at: str = ""
    idempotency_key: str = ""
    instance_id: str = ""
    mac: str = ""

    def payload(self) -> dict[str, Any]:
        """The exact canonical dict covered by the MAC (every field but ``mac``)."""
        return {name: getattr(self, name) for name in _UNSIGNED_FIELDS}

    def as_dict(self) -> dict[str, Any]:
        """A plain dict including the MAC, suitable for logging or storage."""
        return {name: getattr(self, name) for name in PERMIT_FIELDS}

    to_dict = as_dict


# ---------------------------------------------------------------------------
# Governor-owned issuer
# ---------------------------------------------------------------------------


class PermitIssuer:
    """The Governor's signing authority; it keeps its HMAC key to itself.

    The key is never exposed: there is no getter, ``repr``/``str`` redact it,
    and no attribute of this object holds it. It is absorbed at construction
    into a :class:`~.policy.KeyedHmacSigner`'s opaque ``hashlib`` state, which
    reports no GC referents, so neither an attribute walk nor
    ``gc.get_referents`` can recover the key bytes. Model-facing code is only
    ever handed a :class:`PermitService`, which keeps the issuer private.
    """

    __slots__ = ("_signer",)

    def __init__(self, key: Any = None):
        if key is None:
            key = secrets.token_bytes(32)
        elif isinstance(key, str):
            key = key.encode("utf-8")
        if not isinstance(key, (bytes, bytearray)) or not key:
            raise PermitError("issuer key must be non-empty bytes or text")
        self._signer = KeyedHmacSigner(key)

    def sign(self, payload: bytes) -> str:
        """Return the hex HMAC-SHA256 of ``payload``."""
        if not isinstance(payload, (bytes, bytearray)):
            raise PermitError("payload must be bytes")
        return self._signer.sign(bytes(payload))

    def verify(self, payload: bytes, mac: Any) -> bool:
        """Constant-time check of ``mac`` against ``payload``."""
        if not isinstance(mac, str) or not mac:
            return False
        expected = self.sign(payload)
        try:
            return hmac.compare_digest(expected, mac)
        except TypeError:
            # compare_digest rejects non-ASCII strings; a malformed MAC is a
            # verification failure, never an exception that skips later checks.
            return False

    def signer(self) -> Any:
        """A closed signing callable that carries no back-reference to the issuer.

        The model-facing :class:`PermitService` holds only these callables, so
        walking its attributes (``__self__``, ``__closure__``, ``__dict__`` or
        ``gc.get_referents``) reaches the keyed signer but never the issuer and
        never the key bytes.
        """
        signer = self._signer

        def sign(payload: bytes) -> str:
            if not isinstance(payload, (bytes, bytearray)):
                raise PermitError("payload must be bytes")
            return signer.sign(bytes(payload))

        return sign

    def verifier(self) -> Any:
        """A closed verifying callable that carries no back-reference to the issuer."""
        signer = self._signer

        def verify(payload: bytes, mac: Any) -> bool:
            return signer.verify(bytes(payload), mac)

        return verify

    def __repr__(self) -> str:
        return "PermitIssuer(<redacted>)"

    __str__ = __repr__


def _length_prefix(data: bytes) -> bytes:
    return len(data).to_bytes(8, "big") + data


def _signed_bytes(permit: ExecutionPermit) -> bytes:
    """The domain-separated, length-prefixed byte string covered by the MAC.

    The fixed domain tag prevents a permit MAC from being replayed for any other
    object, and every field name and value is individually length-prefixed so
    the encoding is unambiguous: no two distinct permits can share these bytes.
    """
    chunks = [_length_prefix(_PERMIT_MAC_DOMAIN)]
    for name in _UNSIGNED_FIELDS:
        value = getattr(permit, name)
        chunks.append(_length_prefix(name.encode("utf-8")))
        if value is None:
            chunks.append(_length_prefix(b"\x00"))
        else:
            chunks.append(_length_prefix(b"\x01" + str(value).encode("utf-8")))
    return b"".join(chunks)


def _permit_mac(sign: Any, permit: ExecutionPermit) -> str:
    return sign(_signed_bytes(permit))


# ---------------------------------------------------------------------------
# Permit service
# ---------------------------------------------------------------------------


class PermitService:
    """Mints and verifies permits on behalf of the Governor.

    The service needs the issuer (to sign/verify), the current mission policy
    (to gate issuance and bind a policy version), the tool manifest (to validate
    calls and read risk classes), and -- optionally -- a
    :class:`~.budget.BudgetService` to reserve spend before issuing.
    """

    def __init__(
        self,
        issuer: PermitIssuer,
        policy: MissionPolicy,
        *,
        budget: BudgetService | None = None,
        manifest: ToolManifest | None = None,
        db_path: str | Path | None = None,
        clock: Any = utc_now,
        monotonic: Any = time.monotonic,
        default_ttl_seconds: float = 30.0,
        timeout: float = 30.0,
    ):
        if not isinstance(issuer, PermitIssuer):
            raise PermitError("issuer must be a PermitIssuer")
        if not isinstance(policy, MissionPolicy):
            raise PermitError("policy must be a MissionPolicy")
        _require_verified_authority(policy)
        # Retain only closed callables, never the issuer object, so the signing
        # key is not reachable from this model-facing service.
        self._sign = issuer.signer()
        self._verify = issuer.verifier()
        self._policy = policy
        self._policy_digest = policy.digest
        self._budget = budget
        self._manifest = manifest if manifest is not None else DEFAULT_MANIFEST
        self._manifest_digest = self._manifest.digest()
        self._clock = clock
        self._monotonic = monotonic
        # The identity this service mints under. Every permit carries it under
        # the MAC and only this instance will accept it, which is what makes a
        # restart fail closed: the clock defences below are in-process, so a
        # service that came back cannot re-time the permits of the instance it
        # replaced and has to refuse them instead.
        self._instance_id = secrets.token_hex(16)
        # The in-process half of the effective time: the effective moment of the
        # last observation and the monotonic reading taken alongside it, until
        # the next observation re-anchors the pair.
        self._anchor_wall: datetime | None = None
        self._anchor_mono = 0.0
        self._ttl = float(default_ttl_seconds)
        self._timeout = float(timeout)
        self._lock = threading.RLock()
        if db_path is None:
            # A replay ledger must survive the process, so the default is a
            # persistent file (one per policy digest), not an in-memory table.
            state_dir = Path(_DEFAULT_STATE_DIR)
            state_dir.mkdir(parents=True, exist_ok=True)
            db_path = state_dir / f"permits-{self._policy_digest[:16]}.db"
        self._file_backed = str(db_path) != ":memory:"
        if self._file_backed:
            target = str(db_path)
            Path(target).parent.mkdir(parents=True, exist_ok=True)
            self.db_path: str | None = target
        else:
            self.db_path = None
        self._db = sqlite3.connect(
            self.db_path if self.db_path is not None else ":memory:",
            timeout=self._timeout,
            isolation_level=None,
            check_same_thread=False,
        )
        self._db.row_factory = sqlite3.Row
        if self._file_backed:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute(f"PRAGMA busy_timeout={int(self._timeout * 1000)}")
        self._db.executescript(_NONCE_SCHEMA)
        self._db.executescript(_CLOCK_SCHEMA)
        self._hwm = _TimeHighWaterMark(self._db, self._lock, _time_source(clock))

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> PermitService:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def policy(self) -> MissionPolicy:
        return self._policy

    @property
    def manifest(self) -> ToolManifest:
        """The tool manifest permits are validated and bound against."""
        return self._manifest

    @property
    def instance_id(self) -> str:
        """The random identity permits are minted under, and accepted only by."""
        return self._instance_id

    def set_policy(self, policy: MissionPolicy) -> None:
        """Adopt a newly loaded policy. Only the Governor should call this.

        Authority is signature-only. A policy may not *assert* its own
        enforcement: an ``ENFORCE`` policy that was not produced by a verified
        operator signature (``signed``) is refused, so a hand-built or
        field-forged policy can never grant authority. Downgrading to an
        ``OBSERVATION_ONLY`` policy (for example from the kill switch) is always
        allowed because it withholds authority rather than granting it.
        """
        if not isinstance(policy, MissionPolicy):
            raise PermitError("policy must be a MissionPolicy")
        _require_verified_authority(policy)
        self._policy = policy
        self._policy_digest = policy.digest

    # -- issuance ----------------------------------------------------------
    def issue(
        self,
        *,
        tool: str,
        op: str,
        args: Any = None,
        goal_id: str,
        plan_id: str = "",
        step_id: str = "",
        capability: str = "",
        state_version: str = "",
        idempotency_key: str | None = None,
        scope_ids: Any = None,
        amounts: Any = None,
        ttl_seconds: float | None = None,
        now: Any = None,
    ) -> ExecutionPermit:
        """Validate, budget, and mint one permit, or raise :class:`PermitDenied`.

        Budget amounts are derived from the manifest's declared cost model, never
        from the caller: ``amounts`` (if supplied at all) is refused outright.
        """
        if not self._policy.enforced:
            raise PermitDenied(
                "mission policy is not enforced; no permits may be issued"
            )
        if not isinstance(goal_id, str) or not goal_id:
            raise PermitDenied("goal_id is required")
        if amounts is not None:
            raise PermitDenied(
                "budget amounts are derived from the manifest's cost model, "
                "not supplied by the caller"
            )

        try:
            self._manifest.validate_call(tool, op, args)
        except ManifestError as exc:
            raise PermitDenied(f"tool call is not permitted: {exc}") from exc

        # Policy authority is a closed set: the tool must be an allowed domain
        # and the capability must be granted before any permit can exist.
        if not self._policy.allows_domain(tool):
            raise PermitDenied(f"tool {tool!r} is not in the policy's allowed_domains")
        risk_class = self._manifest.risk_class(tool, op)
        expected_capability = f"{tool}.{op}"
        capability = capability or expected_capability
        # A capability must name exactly the manifest operation this permit is
        # for. A borrowed name (``memory.forget`` on a ``state`` permit), a
        # malformed name with no dot at all (``observe``), or any other op of
        # the same tool is refused, so a capability can never be inflated or
        # swapped to widen the grant.
        if capability != expected_capability:
            if "." not in capability or not capability.startswith(f"{tool}."):
                raise PermitDenied(
                    f"capability {capability!r} does not belong to tool {tool!r}"
                )
            raise PermitDenied(
                f"capability {capability!r} does not match the manifest operation "
                f"{expected_capability!r}"
            )
        if not self._policy.allows_capability(capability, tool, op):
            raise PermitDenied(
                f"capability {capability!r} is not granted by the mission policy"
            )
        if risk_class == "A3" and not self._has_mandate(tool, op, capability):
            raise PermitDenied(
                f"A3 operation {tool}.{op} requires a matching standing mandate"
            )

        permit_id = uuid.uuid4().hex
        nonce = secrets.token_hex(16)
        issued_at = now if now is not None else self._clock()
        # Minting is itself an observation of time: record it so a later clock
        # rollback cannot make permits issued here look valid again. The mark
        # takes the later of the instant minted and the clock reading, never the
        # clock alone -- a permit stamped ahead of the local clock must not leave
        # the mark behind the life it has granted.
        self._observe_time(issued_at, self._clock())
        ttl = self._ttl if ttl_seconds is None else float(ttl_seconds)
        expires_at = _add_seconds(issued_at, ttl)

        budget_reservation: str | None = None
        if self._budget is not None:
            reservation_amounts = self._resolve_amounts(tool, op)
            budget_reservation = self._budget.reserve(
                self._resolve_scopes(scope_ids, goal_id, permit_id, nonce),
                reservation_amounts,
            )

        permit = ExecutionPermit(
            permit_id=permit_id,
            nonce=nonce,
            goal_id=goal_id,
            plan_id=plan_id,
            step_id=step_id,
            tool=tool,
            op=op,
            capability=capability,
            normalized_args_hash=args_hash(args),
            manifest_digest=self._manifest_digest,
            policy_digest=self._policy_digest,
            state_version=state_version,
            policy_version=self._policy.policy_version,
            risk_class=risk_class,
            budget_reservation=budget_reservation,
            issued_at=_parse_time(issued_at).isoformat(),
            expires_at=expires_at,
            idempotency_key=idempotency_key or f"{goal_id}:{tool}.{op}:{nonce}",
            instance_id=self._instance_id,
            mac="",
        )
        return replace(permit, mac=_permit_mac(self._sign, permit))

    def _resolve_amounts(self, tool: str, op: str) -> dict[str, float]:
        spec = self._manifest.operation(tool, op)
        unit = spec.cost_model.unit
        estimate = float(spec.cost_model.estimate)
        if unit in RESOURCES and estimate > 0:
            return {unit: estimate}
        return {}

    def _resolve_scopes(
        self, scope_ids: Any, goal_id: str, permit_id: str, nonce: str
    ) -> dict[str, str]:
        """Every budget level the reservation must be charged against.

        The full ancestry (global -> mission -> goal -> permit) is always
        supplied so no level can escape its cap by being omitted, unless the
        caller explicitly overrides ``scope_ids`` (an internal Governor path).
        """
        if scope_ids is not None:
            return dict(scope_ids)
        return {
            "global": "global",
            "mission": self._policy.mission_id or "mission",
            "goal": goal_id,
            "permit": permit_id or nonce,
        }

    def _has_mandate(self, tool: str, op: str, capability: str) -> bool:
        return any(
            _mandate_matches(mandate, tool, op, capability)
            for mandate in self._policy.standing_mandates
        )

    def _observe_time(self, *moments: Any) -> datetime:
        """Fold moments of observed time into the effective clock; return it.

        The effective now is the furthest-forward of three readings: the wall
        clock as just observed, the mark persisted by the store, and an
        in-process anchor -- the effective moment of the previous observation
        plus however long this process has run since, taken from a monotonic
        counter that no wall-clock change can step.

        The anchor is what makes idle time count. A mark only advances when the
        store is used, so a service that issued a ten-second permit and then sat
        idle for twenty seconds still had a mark reading the instant of issue; a
        wall clock stepped back twenty-five seconds -- inside the tolerated skew,
        so not a rollback by itself -- would then hand the expired permit back
        its life. The anchor cannot be stepped by the wall clock, so the
        effective time still moves with real elapsed time inside a process.

        Re-anchoring at the *result* keeps that time monotone: the pair never
        moves backwards, so a clock that is rolled back and then stepped forward
        again cannot walk the effective time back to where it was. Only the
        wall-clock reading is persisted -- the anchor is meaningless once the
        process is gone, and the mark it leaves behind is the floor a restarted
        process starts from.

        A floor is not a measurement: nothing in a fresh process observed the
        hours it spent dead, so the mark cannot expire the permits the previous
        instance left behind if the wall clock never moved past them. That is why
        permits are also bound to their instance -- a restarted service refuses
        them instead of re-timing them.
        """
        wall = max(_parse_time(moment) for moment in moments)
        monotonic = float(self._monotonic())
        # The mark and the anchor are read and moved together, under the same
        # lock the mark uses, so concurrent verifiers cannot each re-anchor from
        # a stale pair and walk the effective time backwards between them.
        with self._lock:
            # The conditional UPSERT returns the database maximum *after*
            # concurrent writers, not the value from a stale earlier SELECT.
            stored = self._hwm.observe(wall)
            candidates = [wall, stored]
            if self._anchor_wall is not None:
                candidates.append(
                    self._anchor_wall + timedelta(seconds=monotonic - self._anchor_mono)
                )
            effective = max(candidates)
            self._anchor_wall = effective
            self._anchor_mono = monotonic
        return effective

    # -- verification ------------------------------------------------------
    def verify_and_consume(
        self,
        permit: Any,
        tool: str,
        args: Any = None,
        *,
        op: str | None = None,
        expected_state_version: str | None = None,
    ) -> None:
        """Validate ``permit`` for ``tool``/``op``/``args`` and consume its nonce.

        Raises :class:`PermitVerificationError` (or the :class:`PermitReplayError`
        subclass) on any failure. On success the nonce is recorded atomically and
        the call is authorised exactly once.

        ``op`` is mandatory: a permit binds the exact operation it was minted
        for, so a permit for ``state.set_load_target`` can never drive a
        different operation on the same tool. Every check, including the MAC and
        the issuing instance, runs before any database work, so a failure never
        consumes the nonce.
        """
        if not isinstance(permit, ExecutionPermit):
            raise PermitVerificationError("permit must be an ExecutionPermit")

        # Constant-time MAC check first: no database work happens until the
        # permit has been proven authentic.
        if not self._verify(_signed_bytes(permit), permit.mac):
            raise PermitVerificationError("permit MAC is invalid")

        # The permit must be one this very instance minted. Expiry is judged
        # against the effective time, which is a floor and not a measurement: a
        # restarted service cannot know how long its predecessor was gone, and a
        # wall clock still parked on the instant of issue would hand a long-dead
        # permit back its life. Binding each permit to its issuer closes that
        # gap structurally -- a permit from another instance (the process that
        # preceded this one on the same ledger, or a sibling service) is refused
        # before its age is ever considered.
        if permit.instance_id != self._instance_id:
            raise PermitVerificationError(
                "permit was issued by a previous service instance"
            )

        now = self._clock()
        # Expiry is judged against the effective time -- the furthest-forward
        # moment this process can vouch for -- and never against the bare wall
        # clock. Recording it first also separates a rollback from mere skew.
        effective_now = self._observe_time(now)
        rollback = (effective_now - _parse_time(now)).total_seconds()
        # The untrusted clock is reported before the expiry it also causes: a
        # clock rolled back past the skew makes every permit look expired, and
        # "permit has expired" would hide the rollback that is the real finding.
        if rollback > _CLOCK_SKEW_SECONDS:
            raise PermitClockError(
                f"clock rollback detected: the wall clock is {rollback:.0f}s behind "
                f"the {effective_now.isoformat()} effective time recorded in the "
                "permit store; permits are refused while the clock is untrusted"
            )
        if _parse_time(permit.expires_at) <= effective_now:
            raise PermitVerificationError("permit has expired")

        if permit.tool != tool:
            raise PermitVerificationError(
                f"permit is bound to tool {permit.tool!r}, not {tool!r}"
            )
        if not isinstance(op, str) or not op:
            raise PermitVerificationError("the operation must be named to verify a permit")
        if permit.op != op:
            raise PermitVerificationError(
                f"permit is bound to operation {permit.op!r}, not {op!r}"
            )

        if args_hash(args) != permit.normalized_args_hash:
            raise PermitVerificationError("tool arguments do not match the permit")

        if not self._policy.enforced:
            raise PermitVerificationError("mission policy is not enforced")
        if permit.policy_version != self._policy.policy_version:
            raise PermitVerificationError(
                f"permit policy version {permit.policy_version!r} does not match "
                f"the current policy {self._policy.policy_version!r}"
            )
        if permit.policy_digest != self._policy_digest:
            raise PermitVerificationError(
                "permit is bound to a different policy than the one now in force"
            )
        if permit.manifest_digest != self._manifest_digest:
            raise PermitVerificationError(
                "permit is bound to a different tool manifest than the one now in force"
            )
        if not self._policy.allows_domain(permit.tool):
            raise PermitVerificationError(
                f"tool {permit.tool!r} is no longer an allowed domain"
            )
        if not self._policy.allows_capability(permit.capability, permit.tool, permit.op):
            raise PermitVerificationError(
                f"capability {permit.capability!r} is no longer granted by the policy"
            )

        self._require_reservation(permit)

        if expected_state_version is not None and permit.state_version != expected_state_version:
            raise PermitVerificationError(
                f"permit state version {permit.state_version!r} does not match "
                f"the expected state version {expected_state_version!r}"
            )

        self._consume_nonce(permit)

    def _require_reservation(self, permit: ExecutionPermit) -> None:
        """Refuse a permit whose budget reservation is no longer open.

        A reservation that has already been released or committed has spent (or
        returned) its authority; executing on it would let the same budget be
        spent twice.
        """
        reservation = permit.budget_reservation
        if not reservation or self._budget is None:
            return
        state = self._budget.reservation_state(reservation)
        if state != "reserved":
            raise PermitVerificationError(
                f"permit budget reservation {reservation!r} is {state or 'unknown'}, "
                "not open"
            )

    def _require_mandate(self, permit: ExecutionPermit) -> None:
        """Re-check the standing mandate for an ``A3`` permit.

        Called inside the nonce transaction so the authority check and the burn
        are atomic: an ``A3`` permit whose mandate was withdrawn cannot consume
        its nonce.
        """
        if permit.risk_class != "A3":
            return
        if not any(
            _permit_mandate_matches(mandate, permit)
            for mandate in self._policy.standing_mandates
        ):
            raise PermitVerificationError(
                f"A3 operation {permit.capability!r} requires a matching standing "
                "mandate at consumption time"
            )

    def _consume_nonce(self, permit: ExecutionPermit) -> None:
        nonce = permit.nonce
        if not isinstance(nonce, str) or not nonce:
            raise PermitVerificationError("permit nonce is missing")
        permit_id = str(permit.permit_id or "")
        consumed_at = self._clock()
        if self._file_backed:
            conn = sqlite3.connect(
                self.db_path, timeout=self._timeout, isolation_level=None
            )
            try:
                # Serialise writers explicitly and wait out a short-lived writer
                # rather than failing: with the busy timeout set, concurrent
                # consumers of one nonce queue up and exactly one wins.
                conn.execute(f"PRAGMA busy_timeout={int(self._timeout * 1000)}")
                conn.execute("BEGIN IMMEDIATE")
                self._require_mandate(permit)
                conn.execute(
                    "INSERT INTO used_nonces (nonce, permit_id, consumed_at) "
                    "VALUES (?, ?, ?)",
                    (nonce, permit_id, consumed_at),
                )
                conn.execute("COMMIT")
            except PermitVerificationError:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
            except sqlite3.IntegrityError:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise PermitReplayError(
                    f"permit nonce {nonce!r} has already been consumed"
                ) from None
            finally:
                conn.close()
            return

        with self._lock:
            self._require_mandate(permit)
            try:
                self._db.execute(
                    "INSERT INTO used_nonces (nonce, permit_id, consumed_at) "
                    "VALUES (?, ?, ?)",
                    (nonce, permit_id, consumed_at),
                )
            except sqlite3.IntegrityError:
                raise PermitReplayError(
                    f"permit nonce {nonce!r} has already been consumed"
                ) from None

    def consumed(self, nonce: str) -> bool:
        row = self._db.execute(
            "SELECT 1 FROM used_nonces WHERE nonce=?", (str(nonce),)
        ).fetchone()
        return row is not None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic, never leaks the key
        return f"PermitService(policy_version={self._policy.policy_version!r})"


def _mandate_matches(mandate: Any, tool: str, op: str, capability: str) -> bool:
    """Whether a standing mandate authorises ``tool.op``.

    Matching is exact and case-insensitive against the tool name, the
    ``tool.op`` / ``tool:op`` / ``tool/op`` forms, an ``a3``-prefixed form, or
    the granted capability.
    """
    if not isinstance(mandate, str):
        return False
    text = mandate.strip().lower()
    if not text:
        return False
    tool_l = (tool or "").lower()
    op_l = (op or "").lower()
    candidates = {
        tool_l,
        f"{tool_l}.{op_l}",
        f"{tool_l}:{op_l}",
        f"{tool_l}/{op_l}",
        f"a3:{tool_l}.{op_l}",
        f"a3 {tool_l}.{op_l}",
    }
    if capability:
        candidates.add(capability.strip().lower())
    return text in candidates


def _permit_mandate_matches(mandate: Any, permit: ExecutionPermit) -> bool:
    """Whether a standing mandate still authorises an issued ``A3`` permit.

    The permit now stores its ``op`` directly; fall back to recovering it from
    the canonical ``tool.op`` capability shape for older permits.
    """
    op = permit.op
    if not op:
        prefix = f"{permit.tool}."
        if permit.capability.startswith(prefix):
            op = permit.capability[len(prefix):]
    return _mandate_matches(mandate, permit.tool, op, permit.capability)

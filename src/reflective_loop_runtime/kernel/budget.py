"""Hierarchical, durable budget accounting for the trusted kernel.

Every side effect the runtime may take is bounded by the mission's standing
budgets. Spending is tracked at four nested scopes -- ``global``, ``mission``,
``goal`` and ``permit`` -- so a single reservation can be charged against every
level of the hierarchy it descends from.

Trust model
-----------
* Budgets are *derived* from the immutable :class:`~.policy.MissionPolicy` the
  service is constructed with. There is deliberately **no public API to raise a
  cap**: only the policy (which the operator signs and the runtime merely
  verifies) can grant spend, and a policy that is not in ``ENFORCE`` mode grants
  nothing.
* Accounting is a SQLite database in WAL mode. Every mutation runs inside a
  ``BEGIN IMMEDIATE`` write transaction, so concurrent reservers are serialised
  by the database itself and can never overshoot a cap.
* ``reserve`` is all-or-nothing: if *any* level or resource would exceed its
  cap, nothing is charged and :class:`BudgetExceeded` is raised.
* A reservation is either *committed* with the actual spend or *released*; both
  are terminal. ``commit`` refunds the unused remainder, and ``concurrency`` is
  a gauge that is always freed when the reservation is finalised (it bounds
  in-flight work, not consumed work).

The module is stdlib-only, deterministic, and performs no I/O beyond the SQLite
file it is explicitly asked to open.
"""

from __future__ import annotations

import math
import sqlite3
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .policy import MissionPolicy, canonical_json

__all__ = [
    "RESOURCES",
    "SCOPE_LEVELS",
    "BudgetError",
    "BudgetExceeded",
    "BudgetService",
    "utc_now",
]

# The resource units a budget may cap. ``concurrency`` is a gauge (in-flight
# reservations); every other unit is a counter of consumed work. Every cost unit
# a manifest may declare must be a resource here, so no declared cost can go
# unbudgeted; the scheduler/store/io/message units the built-in manifest prices
# operations in are therefore first-class budget resources.
RESOURCES = (
    "money",
    "llm_tokens",
    "api_calls",
    "shell_cpu_seconds",
    "wall_clock_seconds",
    "scheduler_tick",
    "store_write",
    "store_read",
    "io_read",
    "message",
    "concurrency",
)

# The scope hierarchy, ordered from outermost to innermost. A reservation is
# charged against every level the caller names in ``scope_ids``; supplying the
# full ancestry from ``global`` down to the leaf is what enforces the hierarchy.
SCOPE_LEVELS = ("global", "mission", "goal", "permit")

# A wildcard level used for a policy cap that applies to every scope level.
_WILDCARD = "*"

# Guards float comparisons so a mathematically exact sum is not rejected by
# binary floating point representation error.
_EPSILON = 1e-9

_SCHEMA = """
CREATE TABLE IF NOT EXISTS budget_caps (
    level    TEXT NOT NULL,
    resource TEXT NOT NULL,
    cap      REAL NOT NULL,
    PRIMARY KEY (level, resource)
);
CREATE TABLE IF NOT EXISTS budget_reservations (
    reservation_id TEXT PRIMARY KEY,
    state          TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    finalized_at   TEXT
);
CREATE TABLE IF NOT EXISTS budget_holds (
    hold_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    reservation_id TEXT NOT NULL,
    level          TEXT NOT NULL,
    scope_id       TEXT NOT NULL,
    resource       TEXT NOT NULL,
    amount         REAL NOT NULL,
    active         INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_budget_holds_scope
    ON budget_holds (level, scope_id, resource, active);
CREATE INDEX IF NOT EXISTS idx_budget_holds_reservation
    ON budget_holds (reservation_id, active);
"""


class BudgetError(ValueError):
    """Raised when a budget operation is malformed or refers to no reservation."""


class BudgetExceeded(BudgetError):
    """Raised when a reservation would push a scope past one of its caps."""


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _finite(value: Any) -> float | None:
    """Return ``value`` as a finite float, or ``None`` when it is not numeric."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def _parse_cap_key(key: Any) -> tuple[str, str] | None:
    """Map a policy budget key to ``(level, resource)`` or ``None``.

    Accepted forms are ``resource`` (applies at every level), and
    ``level:resource`` / ``level.resource`` (applies at that level). Keys that
    name neither a known resource nor a known level are ignored, so a policy may
    carry unrelated metadata in its ``budgets`` mapping.
    """
    if not isinstance(key, str):
        return None
    text = key.strip()
    if not text:
        return None
    for separator in (":", "."):
        if separator in text:
            head, _, tail = text.partition(separator)
            head, tail = head.strip(), tail.strip()
            if tail in RESOURCES and (head in SCOPE_LEVELS or head == _WILDCARD):
                return head, tail
    if text in RESOURCES:
        return _WILDCARD, text
    return None


class BudgetService:
    """Durable, cap-enforcing budget ledger bound to one mission policy.

    Parameters
    ----------
    db_path:
        File path of the SQLite ledger. ``:memory:`` is rejected because the
        service opens a fresh connection per transaction and an in-memory
        database would not be visible across them.
    policy:
        The frozen mission policy that grants caps. Use
        :meth:`from_policy_store` to load it from a
        :class:`~.policy.MissionPolicyStore`, which is the only supported
        source of authority.
    """

    def __init__(
        self,
        db_path: str | Path,
        policy: MissionPolicy,
        *,
        timeout: float = 30.0,
        clock: Any = utc_now,
    ):
        if not isinstance(policy, MissionPolicy):
            raise BudgetError("policy must be a MissionPolicy")
        target = str(db_path)
        if target == ":memory:":
            raise BudgetError("BudgetService requires a file-backed database, not ':memory:'")
        if target != ":memory:":
            Path(target).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = target
        self._policy = policy
        self._timeout = float(timeout)
        self._clock = clock
        self._caps: dict[tuple[str, str], float] = {}
        self._init_schema()
        self._load_caps()

    @classmethod
    def from_policy_store(
        cls,
        db_path: str | Path,
        store: Any,
        policy_path: str | Path,
        **kwargs: Any,
    ) -> BudgetService:
        """Build a service whose caps come from a read-only policy store."""
        return cls(db_path, store.load(policy_path), **kwargs)

    # -- schema / caps -----------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=self._timeout, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout={int(self._timeout * 1000)}")
        return conn

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        # WAL is a persistent database property, so enable it once here rather
        # than on every concurrent connection (where changing journal mode can
        # contend for the database lock).
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)
        finally:
            conn.close()

    def _load_caps(self) -> None:
        """Derive caps from the policy. This is the only cap-writing path."""
        caps: dict[tuple[str, str], float] = {}
        if self._policy.enforced:
            for key, value in self._policy.budgets.items():
                parsed = _parse_cap_key(key)
                if parsed is None:
                    continue
                number = _finite(value)
                if number is None or number < 0:
                    continue
                caps[parsed] = number
        self._caps = caps
        with self._transaction() as conn:
            conn.execute("DELETE FROM budget_caps")
            for (level, resource), cap in caps.items():
                conn.execute(
                    "INSERT INTO budget_caps (level, resource, cap) VALUES (?, ?, ?)",
                    (level, resource, cap),
                )

    def cap_for(self, level: str, resource: str) -> float | None:
        """Return the cap for ``(level, resource)`` or ``None`` when unbounded.

        A policy that is not enforced grants nothing, so every known resource
        reads back as a zero cap.
        """
        if not self._policy.enforced:
            return 0.0
        if resource not in RESOURCES and level != _WILDCARD:
            raise BudgetError(f"unknown resource {resource!r}")
        specific = self._caps.get((level, resource))
        if specific is not None:
            return specific
        return self._caps.get((_WILDCARD, resource))

    def caps(self) -> dict[str, float]:
        """A detached copy of every ``level.resource`` cap this service enforces."""
        return {
            f"{level}.{resource}": cap
            for (level, resource), cap in sorted(self._caps.items())
        }

    # -- validation --------------------------------------------------------
    @staticmethod
    def _validate_scopes(scope_ids: Any) -> dict[str, str]:
        if isinstance(scope_ids, Mapping):
            items = dict(scope_ids)
        elif isinstance(scope_ids, Sequence) and not isinstance(scope_ids, (str, bytes)):
            items = {}
            for entry in scope_ids:
                try:
                    level, scope_id = entry
                except (TypeError, ValueError):
                    raise BudgetError("scope_ids entries must be (level, id) pairs") from None
                items[level] = scope_id
        else:
            raise BudgetError("scope_ids must be a mapping of level to scope id")
        if not items:
            raise BudgetError("at least one scope is required")
        ordered: dict[str, str] = {}
        for level in SCOPE_LEVELS:
            if level not in items:
                continue
            scope_id = items[level]
            if not isinstance(scope_id, str) or not scope_id:
                raise BudgetError(f"scope {level!r} must map to a non-empty string")
            ordered[level] = scope_id
        unknown = sorted(set(items) - set(ordered))
        if unknown:
            raise BudgetError(f"unknown budget scope level(s): {unknown}")
        return ordered

    @staticmethod
    def _validate_amounts(amounts: Any) -> dict[str, float]:
        if amounts is None:
            return {}
        if not isinstance(amounts, Mapping):
            raise BudgetError("amounts must be a mapping of resource to amount")
        clean: dict[str, float] = {}
        for resource, amount in amounts.items():
            if resource not in RESOURCES:
                raise BudgetError(f"unknown resource {resource!r}")
            number = _finite(amount)
            if number is None or number <= 0:
                raise BudgetError(f"amount for {resource!r} must be a positive finite number")
            clean[resource] = number
        return clean

    @staticmethod
    def _validate_actual(actual: Any) -> dict[str, float]:
        if actual is None:
            return {}
        if not isinstance(actual, Mapping):
            raise BudgetError("actual must be a mapping of resource to amount")
        clean: dict[str, float] = {}
        for resource, amount in actual.items():
            if resource not in RESOURCES:
                raise BudgetError(f"unknown resource {resource!r}")
            number = _finite(amount)
            if number is None or number < 0:
                raise BudgetError(f"actual for {resource!r} must be a non-negative number")
            clean[resource] = number
        return clean

    # -- usage -------------------------------------------------------------
    @staticmethod
    def _used(conn: sqlite3.Connection, level: str, scope_id: str, resource: str) -> float:
        row = conn.execute(
            "SELECT COALESCE(SUM(amount), 0.0) AS used FROM budget_holds "
            "WHERE level=? AND scope_id=? AND resource=? AND active=1",
            (level, scope_id, resource),
        ).fetchone()
        return float(row["used"])

    def scope_usage(self, level: str, scope_id: str, resource: str) -> float:
        """Return the currently held amount for one scope and resource."""
        if level not in SCOPE_LEVELS:
            raise BudgetError(f"unknown budget scope level {level!r}")
        if resource not in RESOURCES:
            raise BudgetError(f"unknown resource {resource!r}")
        conn = self._connect()
        try:
            return self._used(conn, level, str(scope_id), resource)
        finally:
            conn.close()

    def reservation_state(self, reservation_id: str) -> str | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT state FROM budget_reservations WHERE reservation_id=?",
                (str(reservation_id),),
            ).fetchone()
            return str(row["state"]) if row is not None else None
        finally:
            conn.close()

    # -- mutations ---------------------------------------------------------
    def reserve(self, scope_ids: Any, amounts: Any) -> str:
        """Reserve ``amounts`` against every scope in ``scope_ids``.

        Returns a new reservation id, or raises :class:`BudgetExceeded` without
        charging anything when any cap would be exceeded.
        """
        scopes = self._validate_scopes(scope_ids)
        clean = self._validate_amounts(amounts)
        reservation_id = uuid.uuid4().hex
        with self._transaction() as conn:
            for level, scope_id in scopes.items():
                for resource, amount in clean.items():
                    cap = self.cap_for(level, resource)
                    if cap is None:
                        continue
                    used = self._used(conn, level, scope_id, resource)
                    if used + amount > cap + _EPSILON:
                        raise BudgetExceeded(
                            f"budget exceeded at {level}:{scope_id} for {resource}: "
                            f"requested {amount}, already held {used}, cap {cap}"
                        )
            conn.execute(
                "INSERT INTO budget_reservations (reservation_id, state, created_at) "
                "VALUES (?, 'reserved', ?)",
                (reservation_id, self._clock()),
            )
            for level, scope_id in scopes.items():
                for resource, amount in clean.items():
                    conn.execute(
                        "INSERT INTO budget_holds "
                        "(reservation_id, level, scope_id, resource, amount, active) "
                        "VALUES (?, ?, ?, ?, ?, 1)",
                        (reservation_id, level, scope_id, resource, amount),
                    )
        return reservation_id

    def commit(self, reservation_id: str, actual: Any) -> None:
        """Finalise ``reservation_id`` at its real spend, refunding the remainder.

        ``actual`` maps resource to consumed amount; an omitted resource is
        treated as zero. ``concurrency`` always drops to zero because in-flight
        work has finished. A commit that would exceed a cap (or claim more than
        was reserved) is rejected and the reservation stays open.
        """
        reservation_id = str(reservation_id)
        clean = self._validate_actual(actual)
        clean.pop("concurrency", None)  # the gauge is released, never committed
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT state FROM budget_reservations WHERE reservation_id=?",
                (reservation_id,),
            ).fetchone()
            if row is None:
                raise BudgetError(f"unknown reservation {reservation_id!r}")
            if str(row["state"]) != "reserved":
                raise BudgetError(f"reservation {reservation_id!r} is already finalised")

            holds = conn.execute(
                "SELECT hold_id, level, scope_id, resource, amount FROM budget_holds "
                "WHERE reservation_id=? AND active=1",
                (reservation_id,),
            ).fetchall()
            reserved_resources = {str(hold["resource"]) for hold in holds}
            unknown = sorted(set(clean) - reserved_resources)
            if unknown:
                raise BudgetError(
                    f"commit names resource(s) that were never reserved: {unknown}"
                )

            finals: list[tuple[int, float]] = []
            for hold in holds:
                resource = str(hold["resource"])
                held = float(hold["amount"])
                if resource == "concurrency":
                    final = 0.0
                else:
                    final = clean.get(resource, 0.0)
                    if final > held + _EPSILON:
                        raise BudgetExceeded(
                            f"commit for {resource!r} claims {final} but only {held} "
                            f"was reserved"
                        )
                cap = self.cap_for(str(hold["level"]), resource)
                if cap is not None and final > 0:
                    total = self._used(conn, str(hold["level"]), str(hold["scope_id"]), resource)
                    others = total - held
                    if others + final > cap + _EPSILON:
                        raise BudgetExceeded(
                            f"commit would exceed {hold['level']}:{hold['scope_id']} "
                            f"for {resource}: {others + final} > cap {cap}"
                        )
                finals.append((int(hold["hold_id"]), final))

            for hold_id, final in finals:
                conn.execute(
                    "UPDATE budget_holds SET amount=?, active=? WHERE hold_id=?",
                    (final, 1 if final > 0 else 0, hold_id),
                )
            conn.execute(
                "UPDATE budget_reservations SET state='committed', finalized_at=? "
                "WHERE reservation_id=?",
                (self._clock(), reservation_id),
            )

    def release(self, reservation_id: str) -> None:
        """Release ``reservation_id`` in full, returning everything it held."""
        reservation_id = str(reservation_id)
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT state FROM budget_reservations WHERE reservation_id=?",
                (reservation_id,),
            ).fetchone()
            if row is None:
                raise BudgetError(f"unknown reservation {reservation_id!r}")
            if str(row["state"]) != "reserved":
                raise BudgetError(f"reservation {reservation_id!r} is already finalised")
            conn.execute(
                "UPDATE budget_holds SET amount=0.0, active=0 WHERE reservation_id=?",
                (reservation_id,),
            )
            conn.execute(
                "UPDATE budget_reservations SET state='released', finalized_at=? "
                "WHERE reservation_id=?",
                (self._clock(), reservation_id),
            )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        caps = canonical_json(self.caps())
        return f"BudgetService(db_path={self.db_path!r}, caps={caps})"

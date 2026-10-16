"""Event-driven executive (L3): event bus, heartbeat, async reasoner, kill switch.

This layer sits above the permit-gated Executive (L2) and the safety kernel. It
adds four things the loop needs to stay live and fail closed at the same time:

EventBus
--------
A small, thread-safe, synchronous bus for the runtime's discrete events
(``tool_completion``, ``tool_failure``, ``deadline``, ``budget_threshold``,
``state_change``, ``reasoner_result``, ``policy_change``, ``kill``). Handlers
run outside the bus lock so a slow subscriber can never deadlock an emitter.

Heartbeat
---------
A wall-clock heartbeat that fires every 1--5 seconds and *never* calls a model.
It has no reference to any decider or reasoner, so a heartbeat cannot trigger
inference even by accident; it only records liveness on the bus.

AsyncReasoner
-------------
Model calls are dispatched onto daemon threads and collected by polling. Every
call is submitted with the ``state_version`` it was made against and an
absolute ``deadline``. A result that completes after its deadline, or that was
made against a superseded state version, is discarded rather than applied. A
call that hangs blocks only its own thread, so the scheduler keeps ticking.

KillSwitch
----------
A kill file path (resolved from configuration) that, once present, engages a
latch: all tracked permits are revoked, the permit service is dropped to
``OBSERVATION_ONLY`` authority, and every further ToolBroker launch is refused.
A watchdog thread polls the path so engagement happens well inside the two
second bound. The switch never needs network access and never prints secrets.

The module is stdlib-only and deterministic when a fake clock is injected.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .kernel.policy import MissionPolicy

__all__ = [
    "BUDGET_THRESHOLD",
    "DEADLINE",
    "EVENT_KINDS",
    "HEARTBEAT",
    "HEARTBEAT_MAX_SECONDS",
    "HEARTBEAT_MIN_SECONDS",
    "KILL",
    "POLICY_CHANGE",
    "REASONER_RESULT",
    "STATE_CHANGE",
    "TOOL_COMPLETION",
    "TOOL_FAILURE",
    "AsyncReasoner",
    "Event",
    "EventBus",
    "EventError",
    "GuardedBroker",
    "Heartbeat",
    "KillSwitch",
    "LaunchBlocked",
    "ReasonerResult",
]

# -- event kinds ----------------------------------------------------------
TOOL_COMPLETION = "tool_completion"
TOOL_FAILURE = "tool_failure"
DEADLINE = "deadline"
BUDGET_THRESHOLD = "budget_threshold"
STATE_CHANGE = "state_change"
REASONER_RESULT = "reasoner_result"
POLICY_CHANGE = "policy_change"
KILL = "kill"
# Internal liveness signal; not part of the fault-facing contract.
HEARTBEAT = "heartbeat"

EVENT_KINDS = frozenset(
    {
        TOOL_COMPLETION,
        TOOL_FAILURE,
        DEADLINE,
        BUDGET_THRESHOLD,
        STATE_CHANGE,
        REASONER_RESULT,
        POLICY_CHANGE,
        KILL,
        HEARTBEAT,
    }
)

WILDCARD = "*"

HEARTBEAT_MIN_SECONDS = 1.0
HEARTBEAT_MAX_SECONDS = 5.0


class EventError(ValueError):
    """Base class for event-layer failures."""


class LaunchBlocked(EventError):
    """Raised when a ToolBroker launch is refused because the kill latch is set."""


@dataclass(frozen=True)
class Event:
    """One immutable bus event."""

    kind: str
    payload: Mapping[str, Any]
    at: float
    sequence: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "payload": dict(self.payload),
            "at": self.at,
            "sequence": self.sequence,
        }


class EventBus:
    """Synchronous, thread-safe publish/subscribe bus with a bounded history."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic, history_limit: int = 1000):
        if history_limit < 1:
            raise EventError("history_limit must be positive")
        self._clock = clock
        self._lock = threading.RLock()
        self._subscribers: dict[str, list[Callable[[Event], None]]] = {}
        self._history: deque[Event] = deque(maxlen=int(history_limit))
        self._sequence = 0

    def subscribe(self, kind: str, handler: Callable[[Event], None]) -> Callable[[Event], None]:
        """Register ``handler`` for ``kind`` (or ``"*"`` for every event)."""
        if kind != WILDCARD and kind not in EVENT_KINDS:
            raise EventError(f"unknown event kind {kind!r}")
        if not callable(handler):
            raise EventError("handler must be callable")
        with self._lock:
            self._subscribers.setdefault(kind, []).append(handler)
        return handler

    def unsubscribe(self, kind: str, handler: Callable[[Event], None]) -> bool:
        with self._lock:
            handlers = self._subscribers.get(kind)
            if not handlers or handler not in handlers:
                return False
            handlers.remove(handler)
            return True

    def emit(self, kind: str, **payload: Any) -> Event:
        """Publish one event and dispatch it to every matching subscriber."""
        if kind not in EVENT_KINDS:
            raise EventError(f"unknown event kind {kind!r}")
        with self._lock:
            self._sequence += 1
            event = Event(kind=kind, payload=dict(payload), at=self._clock(), sequence=self._sequence)
            self._history.append(event)
            handlers = list(self._subscribers.get(kind, ()))
            handlers += list(self._subscribers.get(WILDCARD, ()))
        for handler in handlers:
            handler(event)
        return event

    def events(self, kind: str | None = None) -> list[Event]:
        with self._lock:
            if kind is None:
                return list(self._history)
            return [event for event in self._history if event.kind == kind]

    def last(self, kind: str) -> Event | None:
        with self._lock:
            for event in reversed(self._history):
                if event.kind == kind:
                    return event
        return None

    @property
    def sequence(self) -> int:
        with self._lock:
            return self._sequence


class Heartbeat:
    """A liveness beat that fires every 1--5 seconds and never calls a model.

    The class holds no decider, reasoner, or broker reference at all: the only
    thing it can do is publish a :data:`HEARTBEAT` event. ``beat(now)`` fires
    when at least ``interval_seconds`` have elapsed, so a fake clock drives it
    deterministically in tests.
    """

    def __init__(
        self,
        bus: EventBus,
        *,
        clock: Callable[[], float] = time.monotonic,
        interval_seconds: float = 2.0,
    ):
        interval = float(interval_seconds)
        if not HEARTBEAT_MIN_SECONDS <= interval <= HEARTBEAT_MAX_SECONDS:
            raise EventError(
                f"heartbeat interval must be within "
                f"[{HEARTBEAT_MIN_SECONDS}, {HEARTBEAT_MAX_SECONDS}] seconds"
            )
        self.bus = bus
        self.interval_seconds = interval
        self._clock = clock
        self._last: float | None = None
        self._tick = 0

    def due(self, now: float | None = None) -> bool:
        now = self._clock() if now is None else float(now)
        return self._last is None or (now - self._last) >= self.interval_seconds

    def beat(self, now: float | None = None) -> Event | None:
        """Emit a heartbeat when due, else return ``None`` without emitting."""
        now = self._clock() if now is None else float(now)
        if not self.due(now):
            return None
        self._last = now
        self._tick += 1
        return self.bus.emit(
            HEARTBEAT, tick=self._tick, interval_seconds=self.interval_seconds
        )

    def pulse(self) -> Event:
        """Emit a heartbeat unconditionally (used at loop start / tests)."""
        self._last = self._clock()
        self._tick += 1
        return self.bus.emit(
            HEARTBEAT, tick=self._tick, interval_seconds=self.interval_seconds
        )

    @property
    def beats(self) -> int:
        return self._tick


@dataclass(frozen=True)
class ReasonerResult:
    """The outcome of one asynchronous reasoner call."""

    call_id: str
    name: str
    state_version: str
    value: Any
    error: str | None
    finished_at: float
    accepted: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "name": self.name,
            "state_version": self.state_version,
            "value": self.value,
            "error": self.error,
            "finished_at": self.finished_at,
            "accepted": self.accepted,
            "reason": self.reason,
        }


class _ReasonerCall:
    __slots__ = (
        "call_id",
        "name",
        "state_version",
        "submitted_at",
        "deadline",
        "thread",
        "done",
        "value",
        "error",
        "finished_at",
    )

    def __init__(
        self,
        call_id: str,
        name: str,
        state_version: str,
        submitted_at: float,
        deadline: float,
        thread: threading.Thread,
    ):
        self.call_id = call_id
        self.name = name
        self.state_version = state_version
        self.submitted_at = submitted_at
        self.deadline = deadline
        self.thread = thread
        self.done = False
        self.value: Any = None
        self.error: str | None = None
        self.finished_at: float | None = None


class AsyncReasoner:
    """Dispatches model calls onto daemon threads and collects fresh results.

    A result is accepted only when it finished at or before its deadline *and*
    was made against the caller's current ``state_version``. Stale or expired
    results are dropped (and reported through the bus) instead of being applied.
    ``submit`` never blocks, so a hanging model call cannot stall the scheduler.
    """

    def __init__(
        self,
        *,
        bus: EventBus | None = None,
        clock: Callable[[], float] = time.monotonic,
        default_expiry_seconds: float = 10.0,
    ):
        if default_expiry_seconds <= 0:
            raise EventError("default_expiry_seconds must be positive")
        self.bus = bus
        self._clock = clock
        self.default_expiry_seconds = float(default_expiry_seconds)
        self._lock = threading.RLock()
        self._calls: dict[str, _ReasonerCall] = {}
        self._dropped = 0
        self._accepted = 0

    # -- submission --------------------------------------------------------
    def submit(
        self,
        name: str,
        fn: Callable[[], Any],
        *,
        state_version: str,
        expiry_seconds: float | None = None,
    ) -> str:
        """Start ``fn`` on a daemon thread and return its call id immediately."""
        if not callable(fn):
            raise EventError("fn must be callable")
        now = self._clock()
        expiry = self.default_expiry_seconds if expiry_seconds is None else float(expiry_seconds)
        if expiry <= 0:
            raise EventError("expiry_seconds must be positive")
        call_id = uuid.uuid4().hex[:12]
        thread = threading.Thread(target=self._run, args=(call_id, fn), daemon=True)
        call = _ReasonerCall(
            call_id=call_id,
            name=str(name),
            state_version=str(state_version),
            submitted_at=now,
            deadline=now + expiry,
            thread=thread,
        )
        with self._lock:
            self._calls[call_id] = call
        thread.start()
        return call_id

    def _run(self, call_id: str, fn: Callable[[], Any]) -> None:
        value: Any = None
        error: str | None = None
        try:
            value = fn()
        except Exception as exc:  # noqa: BLE001 - a failed model call is data, not a crash
            error = f"{type(exc).__name__}: {exc}"[:500]
        with self._lock:
            call = self._calls.get(call_id)
            if call is None:
                return
            call.value = value
            call.error = error
            call.finished_at = self._clock()
            call.done = True

    # -- collection --------------------------------------------------------
    def pending(self) -> int:
        with self._lock:
            return sum(1 for call in self._calls.values() if not call.done)

    def finished(self, call_id: str) -> bool:
        with self._lock:
            call = self._calls.get(call_id)
            return bool(call and call.done)

    def join(self, call_id: str, timeout: float | None = None) -> bool:
        """Wait for the worker thread; returns whether it finished in time."""
        with self._lock:
            call = self._calls.get(call_id)
        if call is None:
            return True
        call.thread.join(timeout)
        return not call.thread.is_alive()

    def poll(
        self,
        *,
        current_state_version: str | None = None,
        now: float | None = None,
    ) -> list[ReasonerResult]:
        """Drain finished calls, returning only accepted (fresh) results.

        Expired and stale results are dropped, counted, and reported on the bus;
        they are never returned, so a caller cannot apply a superseded answer.
        """
        moment = self._clock() if now is None else float(now)
        accepted: list[ReasonerResult] = []
        with self._lock:
            calls = list(self._calls.values())
        for call in calls:
            with self._lock:
                if not call.done:
                    continue
                finished_at = call.finished_at if call.finished_at is not None else moment
                value, error = call.value, call.error
                self._calls.pop(call.call_id, None)

            expired = moment > call.deadline or finished_at > call.deadline
            stale = (
                current_state_version is not None
                and call.state_version != str(current_state_version)
            )
            if expired:
                reason = "expired"
                self._dropped += 1
                if self.bus is not None:
                    self.bus.emit(
                        DEADLINE,
                        call_id=call.call_id,
                        name=call.name,
                        state_version=call.state_version,
                        deadline=call.deadline,
                    )
                    self.bus.emit(
                        REASONER_RESULT,
                        call_id=call.call_id,
                        name=call.name,
                        state_version=call.state_version,
                        accepted=False,
                        reason=reason,
                    )
                continue
            if stale:
                reason = "stale_state"
                self._dropped += 1
                if self.bus is not None:
                    self.bus.emit(
                        REASONER_RESULT,
                        call_id=call.call_id,
                        name=call.name,
                        state_version=call.state_version,
                        accepted=False,
                        reason=reason,
                    )
                continue
            if error is not None:
                self._dropped += 1
                if self.bus is not None:
                    self.bus.emit(
                        REASONER_RESULT,
                        call_id=call.call_id,
                        name=call.name,
                        state_version=call.state_version,
                        accepted=False,
                        reason="error",
                    )
                continue
            result = ReasonerResult(
                call_id=call.call_id,
                name=call.name,
                state_version=call.state_version,
                value=value,
                error=None,
                finished_at=finished_at,
                accepted=True,
            )
            self._accepted += 1
            accepted.append(result)
            if self.bus is not None:
                self.bus.emit(
                    REASONER_RESULT,
                    call_id=call.call_id,
                    name=call.name,
                    state_version=call.state_version,
                    accepted=True,
                )
        return accepted

    def cancel_all(self) -> int:
        """Discard every tracked call's result; returns how many were dropped."""
        with self._lock:
            count = len(self._calls)
            self._calls.clear()
        self._dropped += count
        return count

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def accepted(self) -> int:
        return self._accepted


class KillSwitch:
    """A kill-file latch that revokes permits and stops tool launches.

    ``path`` comes from configuration (see :meth:`from_config`). Once the file
    appears -- or :meth:`engage` is called -- the switch latches on and cannot be
    unset within a process, mirroring an emergency stop. Engagement revokes every
    tracked permit, drops an attached permit service to ``OBSERVATION_ONLY``
    authority, and refuses every subsequent launch through :class:`GuardedBroker`.
    """

    def __init__(
        self,
        path: str | Path | None,
        *,
        bus: EventBus | None = None,
        clock: Callable[[], float] = time.monotonic,
        poll_interval: float = 0.25,
        permits: Any = None,
    ):
        self.path = Path(path) if path is not None else None
        # A watchdog must react well inside the two second budget.
        self.poll_interval = max(0.05, min(float(poll_interval), 2.0))
        self.bus = bus
        self._clock = clock
        self._lock = threading.RLock()
        self._engaged = False
        self._engaged_at: float | None = None
        self._reason = ""
        self._active: set[str] = set()
        self._revoked: set[str] = set()
        self._permits = permits
        self._watchdog: threading.Thread | None = None
        self._watchdog_stop = threading.Event()

    # -- configuration -----------------------------------------------------
    @classmethod
    def from_config(cls, config: Any, **kwargs: Any) -> KillSwitch:
        """Resolve the kill file path from ``config``.

        Accepts an explicit ``kill_file`` attribute, a ``kill_file`` entry in
        ``config.extra``, or falls back to ``<data_dir>/KILL``.
        """
        path = getattr(config, "kill_file", None)
        if path is None:
            extra = getattr(config, "extra", None)
            if isinstance(extra, Mapping):
                path = extra.get("kill_file")
        if path is None:
            data_dir = getattr(config, "data_dir", "rlr_data")
            path = Path(data_dir) / "KILL"
        return cls(path, **kwargs)

    def bind_permits(self, permits: Any) -> None:
        """Attach the permit service whose authority this switch can revoke."""
        self._permits = permits

    # -- state -------------------------------------------------------------
    @property
    def engaged(self) -> bool:
        with self._lock:
            return self._engaged

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    def file_present(self) -> bool:
        return self.path is not None and self.path.exists()

    def check(self, now: float | None = None) -> bool:
        """Engage if the kill file is present; returns the current latch state."""
        if not self.engaged and self.file_present():
            self.engage("kill file present")
        return self.engaged

    def engage(self, reason: str = "kill switch engaged", *, now: float | None = None) -> bool:
        """Latch the switch on, once. Returns ``True`` only on the first call."""
        with self._lock:
            if self._engaged:
                return False
            self._engaged = True
            self._engaged_at = self._clock() if now is None else float(now)
            self._reason = str(reason)
            revoked = self.revoke_all()
        if self.bus is not None:
            self.bus.emit(KILL, reason=self._reason, path=str(self.path), revoked=revoked)
        self._revoke_policy()
        return True

    def _revoke_policy(self) -> None:
        """Drop an attached permit service to observation-only authority."""
        if self._permits is None:
            return
        observation = MissionPolicy.observation_only("kill switch engaged")
        try:
            self._permits.set_policy(observation)
        except Exception:  # noqa: BLE001 - revocation must never raise
            return
        if self.bus is not None:
            self.bus.emit(POLICY_CHANGE, mode=observation.mode, reason="kill switch engaged")

    # -- permit bookkeeping ------------------------------------------------
    def register_permit(self, permit_id: Any) -> bool:
        """Track a freshly minted permit; a revoked permit returns ``False``."""
        token = str(permit_id or "")
        if not token:
            return True
        with self._lock:
            if self._engaged:
                self._revoked.add(token)
                return False
            self._active.add(token)
            return True

    def is_revoked(self, permit_id: Any) -> bool:
        token = str(permit_id or "")
        with self._lock:
            return self._engaged or token in self._revoked or token not in self._active

    def revoke_all(self) -> int:
        """Move every tracked permit into the revoked set; returns the count."""
        with self._lock:
            count = len(self._active)
            self._revoked.update(self._active)
            self._active.clear()
            return count

    @property
    def live_permits(self) -> int:
        with self._lock:
            return 0 if self._engaged else len(self._active)

    # -- launches ----------------------------------------------------------
    def allow_launch(self, permit_id: Any = None) -> bool:
        """Whether a ToolBroker launch may proceed right now."""
        self.check()
        if self.engaged:
            return False
        token = str(permit_id or "")
        if not token:
            return True
        with self._lock:
            return token in self._active

    def require_launch(self, permit_id: Any = None) -> None:
        """Raise :class:`LaunchBlocked` unless a launch is currently allowed."""
        if not self.allow_launch(permit_id):
            raise LaunchBlocked(self._reason or "kill switch engaged")

    # -- watchdog ----------------------------------------------------------
    def start_watchdog(self) -> None:
        """Poll the kill path on a daemon thread until it appears."""
        if self._watchdog is not None and self._watchdog.is_alive():
            return
        self._watchdog_stop.clear()

        def _loop() -> None:
            while not self._watchdog_stop.is_set():
                if self.check():
                    return
                self._watchdog_stop.wait(self.poll_interval)

        self._watchdog = threading.Thread(target=_loop, daemon=True)
        self._watchdog.start()

    def stop_watchdog(self, timeout: float = 2.0) -> None:
        self._watchdog_stop.set()
        if self._watchdog is not None:
            self._watchdog.join(timeout)
            self._watchdog = None


class GuardedBroker:
    """Wraps a ToolBroker so no launch bypasses the kill latch."""

    def __init__(self, broker: Any, kill_switch: KillSwitch):
        self._broker = broker
        self._kill_switch = kill_switch

    @property
    def kill_switch(self) -> KillSwitch:
        return self._kill_switch

    def invoke(self, permit: Any, tool: str, op: str, args: Mapping[str, Any] | None = None) -> Any:
        permit_id = getattr(permit, "permit_id", None)
        self._kill_switch.require_launch(permit_id)
        return self._broker.invoke(permit, tool, op, args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._broker, name)

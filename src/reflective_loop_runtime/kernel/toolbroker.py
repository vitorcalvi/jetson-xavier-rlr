"""ToolBroker: the single, permit-gated path through which tools take effect.

Every tool invocation in the runtime flows through :meth:`ToolBroker.invoke`.
The broker is deliberately narrow:

``invoke(permit, tool, op, args)``
    1. Refuses outright when no permit is presented ("no permit, no call").
    2. Calls :meth:`~.permit.PermitService.verify_and_consume` *first*, so the
       permit's MAC, expiry, tool/argument binding and policy version are
       re-checked and its nonce is consumed before anything subtle happens.
    3. Dispatches on the manifest risk class:
       * ``A0``/``A1`` -- a registered, trusted adapter runs (or, for the
         ``shell`` tool, the command runs inside a bubblewrap sandbox);
       * ``A2``/``A3`` -- a *mock* adapter only records the intended effect to
         the hash-chained audit log and never performs it.

The bubblewrap sandbox (only used when ``/usr/bin/bwrap`` exists) runs every
command with ``--unshare-all --unshare-pid --die-with-parent`` (no network,
private PID/IPC/UTS/user namespaces), ``--new-session`` (so a command cannot
replay a terminal via ``TIOCSTI``), ``--cap-drop ALL``, read-only binds of
``/usr /lib /bin /etc``, a tmpfs for ``/tmp`` and ``HOME``, a read-write bind of
exactly one configured workspace directory, a synthesized minimal environment
(also pinned with explicit ``--setenv`` entries so nothing is inherited from the
caller), an explicit ``--`` separating options from the command, stdin redirected
from ``/dev/null``, and resource limits applied through ``setrlimit`` in a
``preexec_fn`` (CPU, NPROC, address space, file size). Captured stdout/stderr are
read with a hard byte cap and the wall-clock timeout kills the whole process
group, not just the direct child.

``RLIMIT_NPROC`` counts every process the *user* owns, so on a busy host it is
neither a hard per-run cap nor a stable one. When ``pids_cgroup`` is configured,
every sandboxed run is therefore additionally placed in its own child cgroup
under that base, with ``tasks_max`` written to ``pids.max`` before the spawn and
the child moved in by a single write to ``cgroup.procs`` in ``preexec_fn``. That
is a kernel-enforced cap: the fork past ``tasks_max`` fails with EAGAIN however
loaded the host is. The setup fails closed -- a configured cap that cannot be
built refuses the call instead of running the command uncapped -- and so does a
base that is not a delegated pids cgroup at all: the control files are only ever
opened, never created, so a plain directory is refused rather than dressed up
with regular files named ``pids.max`` whose cap the kernel never enforces. After
the run every pid still in the child cgroup is killed and the child removed.
``cgroup.procs`` is written by the run's own children, so it is read as data
and never as instructions: the drain signals nothing at or below zero, nothing
past the kernel's ``PID_MAX_LIMIT``, and neither the broker's own pid nor its
process group leader, whatever the file claims to hold.

The module owns the only import of :mod:`subprocess` in the runtime outside the
model HTTP client; see ``tests/kernel/test_static_guard.py``. It is stdlib-only
and deterministic apart from the sandboxed command it is explicitly asked to run.
"""

from __future__ import annotations

import hashlib
import os
import resource
import secrets
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audit import AuditLog
from .manifest import (
    CostModel,
    FieldSpec,
    OperationSpec,
    ToolManifest,
    ToolSpec,
    default_manifest,
)
from .permit import PermitService
from .policy import canonical_json

__all__ = [
    "DEFAULT_BWRAP",
    "MINIMAL_ENV",
    "PIDS_CONTROL_FILES",
    "READ_ONLY_BINDS",
    "MockEffect",
    "ResourceLimits",
    "ToolBroker",
    "ToolBrokerError",
    "ToolExecutionError",
    "ToolNotPermitted",
    "ToolResult",
    "broker_manifest",
]

# The sandbox helper the broker prefers; absence means shell execution fails
# closed rather than running unsandboxed.
DEFAULT_BWRAP = "/usr/bin/bwrap"

# Filesystem views that are read-only inside the sandbox.
READ_ONLY_BINDS = ("/usr", "/lib", "/bin", "/etc")

# Default resource caps (see the module docstring).
DEFAULT_CPU_SECONDS = 30
DEFAULT_NPROC = 64
DEFAULT_ADDRESS_SPACE_BYTES = 512 * 1024 * 1024
DEFAULT_WALL_TIMEOUT_SECONDS = 30.0
DEFAULT_FILE_SIZE_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_OUTPUT_BYTES = 8192

# Hard per-run PID cap applied through the delegated pids controller, used only
# when the broker is configured with a ``pids_cgroup`` base directory.
DEFAULT_TASKS_MAX = 64

# A cgroup child must not outlive the run that created it, so leftovers are
# killed and drained for at most this long before the child is removed anyway.
PIDS_CGROUP_DRAIN_SECONDS = 5.0
PIDS_CGROUP_POLL_SECONDS = 0.02

# The kernel's own ceiling on a pid (``PID_MAX_LIMIT``, 2**22). Nothing above
# it can be a live process, and a value that large does not merely fail with
# ESRCH when handed to ``os.kill`` -- it raises ``OverflowError`` on the way
# into the C ``pid_t``, which would escape the drain and fail a run that has
# already finished. The reaper filters on this before it ever signals.
PID_MAX_LIMIT = 4194304

# The one environment a sandboxed run is allowed to see. It is applied twice: as
# the environment of the ``bwrap`` process itself (so nothing is inherited from
# the caller) and again as explicit ``--setenv`` entries for the command, so a
# variable that is not listed here cannot leak into a run.
MINIMAL_ENV = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/home",
    "LANG": "C",
    "LC_ALL": "C",
    "TERM": "dumb",
}


# The control files the kernel creates in every child of a delegated pids
# cgroup, and therefore the evidence that the base really is one.
PIDS_CONTROL_FILES = ("pids.max", "cgroup.procs")


def _open_cgroup_file(path: Path) -> int:
    """Open a cgroup control file write-only, never creating it.

    ``O_CREAT`` is deliberately absent, and that is the safety property this
    helper exists to hold: the kernel creates the control files when the cgroup
    is made, so a miss means the base is an ordinary directory -- and creating a
    regular file under that name would let the run report a ``tasks_max`` the
    kernel never enforces. The open is therefore the last step after the files
    have been shown to exist, never a way to make them.

    ``O_TRUNC`` is absent for the same reason the values written here are whole
    assignments: cgroupfs control files are not regular files, so a truncating
    open would only add a portability question.
    """
    return os.open(path, os.O_WRONLY)


def _mkdir_pids_child(child: Path) -> None:
    """Create one run's cgroup child under the delegated base.

    A named helper because ``mkdir`` is not the whole of what a pids cgroup
    does: the kernel populates the new child with its control files before the
    call returns, and that is the moment :func:`_require_cgroup_control_files`
    reads to tell a cgroup from a directory.
    """
    child.mkdir(mode=0o700)


def _require_cgroup_control_files(child: Path) -> None:
    """Refuse a child the kernel did not populate with the controller's files.

    ``os.stat`` asks the only question that matters here: did ``pids.max`` and
    ``cgroup.procs`` exist before the broker touched them? A base that is a
    plain directory -- the pids controller not mounted, or not delegated to
    this base -- answers no, and a run under it would report a cap that stops
    nothing at all. So the miss fails the call closed instead of being papered
    over by an ``O_CREAT`` open.
    """
    for name in PIDS_CONTROL_FILES:
        try:
            os.stat(child / name)
        except OSError as exc:
            raise _PidsCgroupNotACgroup(
                f"{child} has no {name}: the base is a plain directory, not a "
                "delegated pids cgroup (install deploy/rlr-sandbox-cgroup.service)"
            ) from exc


def _pids_audit_reason(exc: BaseException) -> str:
    """The audit reason for a pids-cgroup setup that failed closed."""
    reason = getattr(exc, "audit_reason", None) or exc.__class__.__name__
    return f"pids-cgroup:{reason}"


def _write_cgroup_value(path: Path, value: str) -> None:
    """Write ``value`` to a cgroup control file."""
    fd = _open_cgroup_file(path)
    try:
        os.write(fd, value.encode("ascii"))
    finally:
        os.close(fd)


def _cgroup_procs(child: Path) -> list[int]:
    """The pids listed in ``child``; empty when the file cannot be read.

    Only the *shape* of a line is trusted here: anything that is not a run of
    digits is dropped. Every value that does parse is returned unfiltered,
    because what may be signalled is decided by :func:`_signalable_pids` at
    the point of the signal rather than here.
    """
    text = (child / "cgroup.procs").read_text(encoding="utf-8", errors="replace")
    pids: list[int] = []
    for line in text.splitlines():
        line = line.strip()
        if line.isdigit() and int(line) > 0:
            pids.append(int(line))
    return pids


def _protected_pids() -> frozenset[int]:
    """Pids the reaper must never signal, whatever ``cgroup.procs`` lists.

    ``cgroup.procs`` is a file the run's own children are attached to, so
    its contents are only as trustworthy as the processes in the cgroup: a
    stale, tampered, or reused file that names this process must not turn the
    drain into a self-inflicted SIGKILL. The process group leader is included
    for the same reason -- ``getpgid(0)`` is the broker's own group, and that
    is the leader's pid whenever the broker leads one.
    """
    protected = {os.getpid()}
    try:
        protected.add(os.getpgid(0))
    except (AttributeError, OSError):  # not POSIX, or no process group
        pass
    return frozenset(pid for pid in protected if pid > 0)


def _signalable_pids(pids: Iterable[int], protected: frozenset[int]) -> list[int]:
    """The pids from ``cgroup.procs`` that the reaper is allowed to signal.

    Three classes are dropped, and a hostile or stale control file can contain
    every one of them:

    * ``pid <= 0``: ``os.kill`` reads a non-positive pid as a whole process
      group (negative) or as every process on the host (-1), so a ``0`` or a
      ``-1`` line would aim the drain at the machine instead of a straggler.
    * ``pid > PID_MAX_LIMIT``: no such process can exist, and the value cannot
      even reach ``kill(2)`` -- it raises ``OverflowError`` instead of failing
      with ESRCH.
    * the broker's own pid and its process group leader, per
      :func:`_protected_pids`.
    """
    return [pid for pid in pids if 0 < pid <= PID_MAX_LIMIT and pid not in protected]


def _pids_max_events(child: Path) -> int:
    """The ``max`` counter from ``<child>/pids.events`` (0 when unreadable).

    ``pids.events`` counts how many times a run actually hit the cap, which is
    the only direct evidence that the cgroup -- not the host's load -- stopped a
    fork storm.
    """
    try:
        text = (child / "pids.events").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    for line in text.splitlines():
        key, _, value = line.partition(" ")
        if key.strip() == "max":
            try:
                return int(value.strip())
            except ValueError:
                return 0
    return 0


def _read_capped(
    stream: Any, cap: int, sink: bytearray, truncated: list[bool]
) -> None:
    """Drain ``stream`` into ``sink`` keeping at most ``cap`` bytes.

    Bytes past the cap are discarded rather than buffered, so a command that
    floods stdout cannot exhaust the broker's memory; the pipe is still drained
    so the child is never blocked on a full pipe.
    """
    try:
        while True:
            chunk = stream.read(65536)
            if not chunk:
                break
            room = cap - len(sink)
            if room > 0:
                sink.extend(chunk[:room])
            if len(chunk) > room:
                truncated[0] = True
    except (OSError, ValueError):
        pass
    finally:
        try:
            stream.close()
        except OSError:
            pass


class ToolBrokerError(RuntimeError):
    """Base class for every broker failure."""


class ToolNotPermitted(ToolBrokerError):
    """Raised when a call is attempted without (or before) a valid permit."""


class ToolExecutionError(ToolBrokerError):
    """Raised when an authorised call cannot be performed safely."""


class _PidsCgroupNotACgroup(ToolExecutionError):
    """The configured base is a plain directory, not a delegated pids cgroup.

    It carries its own audit reason because it is the one pids failure an
    operator will actually meet, and the audit row has to say "the controller is
    not delegated here" rather than name an exception class.
    """

    audit_reason = "not-a-cgroup"


@dataclass(frozen=True)
class ResourceLimits:
    """``setrlimit`` caps and the wall-clock timeout for one sandboxed run."""

    cpu_seconds: int = DEFAULT_CPU_SECONDS
    nproc: int = DEFAULT_NPROC
    address_space_bytes: int = DEFAULT_ADDRESS_SPACE_BYTES
    wall_timeout_seconds: float = DEFAULT_WALL_TIMEOUT_SECONDS
    file_size_bytes: int = DEFAULT_FILE_SIZE_BYTES
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES
    # Written to ``pids.max`` of the run's own cgroup when ``pids_cgroup`` is
    # configured. ``nproc`` stays as defence in depth: the rlimit is advisory and
    # uid-wide, this one is not.
    tasks_max: int = DEFAULT_TASKS_MAX


@dataclass(frozen=True)
class MockEffect:
    """The *intended* effect of an A2/A3 call, recorded but never performed."""

    tool: str
    op: str
    risk_class: str
    args: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "op": self.op,
            "risk_class": self.risk_class,
            "args": dict(self.args),
            "intended_effect": f"{self.tool}.{self.op}",
        }


@dataclass(frozen=True)
class ToolResult:
    """Structured outcome of one brokered call."""

    tool: str
    op: str
    risk_class: str
    executed: bool
    simulated: bool
    recorded: bool
    ok: bool = True
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    applied: dict[str, Any] | None = None
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "op": self.op,
            "risk_class": self.risk_class,
            "executed": self.executed,
            "simulated": self.simulated,
            "recorded": self.recorded,
            "ok": self.ok,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "applied": dict(self.applied or {}),
            "note": self.note,
        }


def broker_manifest() -> ToolManifest:
    """The default manifest plus the ``shell`` and ``control`` kernel tools.

    The base :func:`~.manifest.default_manifest` is left untouched so existing
    manifest tests keep seeing exactly the original tool set; the broker keeps an
    extended view that also knows how to classify sandboxed shell commands and
    the executor's in-process control effects.
    """
    manifest = default_manifest()
    manifest.register(
        ToolSpec(
            "shell",
            {
                "read": OperationSpec(
                    name="read",
                    input_schema={"command": FieldSpec("str", min_length=1, max_length=8192)},
                    preconditions=("the bubblewrap sandbox is available",),
                    risk_class="A0",
                    reversibility="reversible",
                    idempotency="idempotent",
                    cost_model=CostModel("shell_cpu_seconds", 0.0, "read-only sandboxed command"),
                ),
                "run": OperationSpec(
                    name="run",
                    input_schema={"command": FieldSpec("str", min_length=1, max_length=8192)},
                    preconditions=("the bubblewrap sandbox is available",),
                    effects=("a command runs inside the sandbox",),
                    risk_class="A1",
                    reversibility="reversible",
                    idempotency="non_idempotent",
                    compensation="restore the workspace from its snapshot",
                    cost_model=CostModel("shell_cpu_seconds", 1.0, "sandboxed command"),
                ),
            },
        )
    )
    manifest.register(
        ToolSpec(
            "control",
            {
                "cooldown": OperationSpec(
                    name="cooldown",
                    input_schema={
                        "seconds": FieldSpec("int", min=1, max=3600),
                        "tick": FieldSpec("int", min=0),
                    },
                    preconditions=("the governor approved the cooldown",),
                    effects=("the control state drops to its floor",),
                    risk_class="A1",
                    reversibility="reversible",
                    idempotency="idempotent",
                    compensation="restore the previous cadence",
                    cost_model=CostModel("scheduler_tick", 1.0, "in-memory update"),
                ),
                "reflect": OperationSpec(
                    name="reflect",
                    preconditions=("the slow brain is configured",),
                    effects=("a reflection is requested",),
                    risk_class="A1",
                    reversibility="reversible",
                    idempotency="non_idempotent",
                    compensation="discard the reflection request",
                    cost_model=CostModel("llm_tokens", 0.0, "a reflection request"),
                ),
            },
        )
    )
    return manifest


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_value(value: Any) -> str:
    try:
        return _hash_text(canonical_json(value))
    except (TypeError, ValueError):
        return ""


class ToolBroker:
    """The one choke point every tool effect must pass through.

    Every side effect -- model-facing tool calls through :meth:`invoke` and the
    executor's in-process mutations through :meth:`invoke_effect` -- is gated by
    a verified, single-use :class:`~.permit.ExecutionPermit`. There is no
    permit-less mode: a broker built without a permit service can inspect the
    manifest and build sandbox arguments, but it refuses every effect, so the
    permit gate can never be skipped by accident.
    """

    def __init__(
        self,
        permits: PermitService | None,
        *,
        audit: AuditLog | None = None,
        workspace: str | os.PathLike[str] | None = None,
        manifest: ToolManifest | None = None,
        bwrap_path: str = DEFAULT_BWRAP,
        limits: ResourceLimits | None = None,
        adapters: Mapping[tuple[str, str], Callable[[dict[str, Any]], Any]] | None = None,
        pids_cgroup: str | os.PathLike[str] | None = None,
    ):
        if permits is not None and not isinstance(permits, PermitService):
            raise ToolBrokerError("permits must be a PermitService or None")
        self._permits = permits
        # A broker with no permit service can never perform an effect; it exists
        # only so tests and callers can inspect the manifest or build sandbox
        # arguments without minting authority.
        # The audit log is keyed by default so an attacker who can rewrite the
        # database still cannot forge a verifying chain without the key.
        self._audit = (
            audit if audit is not None
            else AuditLog(":memory:", key=secrets.token_bytes(32))
        )
        self._manifest = manifest if manifest is not None else broker_manifest()
        self._workspace = Path(workspace) if workspace is not None else Path("/tmp/rlr-workspace")
        self._bwrap = str(bwrap_path)
        self._limits = limits if limits is not None else ResourceLimits()
        # A delegated pids-controller base directory, created ahead of time (see
        # deploy/rlr-sandbox-cgroup.service) and owned by the runtime user. Set it
        # and every sandboxed run gets its own capped child cgroup; leave it
        # None and the broker keeps the pre-existing rlimit-only behaviour.
        self._pids_cgroup = Path(pids_cgroup) if pids_cgroup is not None else None
        self._sealed = False
        self._adapters: dict[tuple[str, str], Callable[[dict[str, Any]], Any]] = dict(
            adapters or {}
        )

    # -- construction helpers ---------------------------------------------
    @property
    def manifest(self) -> ToolManifest:
        return self._manifest

    @property
    def audit(self) -> AuditLog:
        return self._audit

    @property
    def workspace(self) -> Path:
        return self._workspace

    @property
    def limits(self) -> ResourceLimits:
        return self._limits

    @property
    def pids_cgroup(self) -> Path | None:
        return self._pids_cgroup

    def bwrap_available(self) -> bool:
        """Whether the bubblewrap sandbox helper is present and executable."""
        return os.path.isfile(self._bwrap) and os.access(self._bwrap, os.X_OK)

    def seal(self) -> None:
        """Freeze the adapter registry and the manifest; idempotent.

        Called automatically before the first effect, so nothing model-facing can
        register a new adapter (or tool) after the broker has started work.
        """
        if self._sealed:
            return
        self._manifest.seal()
        self._sealed = True

    @property
    def sealed(self) -> bool:
        return self._sealed

    def register_effect(
        self, tool: str, op: str, fn: Callable[[dict[str, Any]], Any]
    ) -> None:
        """Register the trusted in-process adapter for ``tool.op``."""
        if self._sealed:
            raise ToolBrokerError("tool broker is sealed; no further adapters may be registered")
        if not callable(fn):
            raise ToolBrokerError("effect adapter must be callable")
        self._adapters[(str(tool), str(op))] = fn

    # -- the single invocation path ---------------------------------------
    def invoke(
        self,
        permit: Any,
        tool: str,
        op: str,
        args: Mapping[str, Any] | None = None,
    ) -> ToolResult:
        """Verify ``permit`` and invoke ``tool.op``; no permit, no call."""
        if permit is None:
            raise ToolNotPermitted("a permit is required before any tool can be invoked")
        normalised = self._normalise_args(args)
        if self._permits is None:
            raise ToolNotPermitted("no permit service is configured on this broker")
        risk = self._risk_class(tool, op)
        if risk in ("A2", "A3"):
            # Nothing is performed on this branch, but the intent is still
            # written to the audit chain, so the permit is verified first. The
            # operation is part of the binding, so a permit for a different op
            # on the same tool is refused.
            self._permits.verify_and_consume(permit, tool, normalised, op=op)
            self.seal()
            return self._simulate(permit, tool, op, normalised, risk)
        # ``_perform`` verifies the permit itself: the dispatch that reaches an
        # adapter can never be entered on unverified authority.
        return self._perform(permit, tool, op, normalised, risk)

    def invoke_effect(
        self,
        permit: Any,
        tool: str,
        op: str,
        args: Mapping[str, Any] | None = None,
    ) -> ToolResult:
        """Run a trusted in-process effect through the permit-verifying path.

        This is the executor's route to its already-registered state mutations.
        It is *not* a permit-less shortcut: the permit is verified and its nonce
        consumed by :meth:`_perform` before the adapter is even looked up, so a
        missing, replayed, or tampered permit never reaches the effect. Unlike
        :meth:`invoke` it dispatches on the registered adapter rather than
        simulating ``A2``/``A3`` -- the executor's effects have already been
        approved by the Governor and are performed by in-process adapters, never
        by a model-chosen tool.
        """
        if permit is None:
            raise ToolNotPermitted("a permit is required before any effect can be applied")
        if self._permits is None:
            raise ToolNotPermitted("no permit service is configured on this broker")
        normalised = self._normalise_args(args)
        risk = self._risk_class(tool, op, default="A1")
        return self._perform(permit, tool, op, normalised, risk)

    # -- dispatch ----------------------------------------------------------
    def _perform(
        self,
        permit: Any,
        tool: str,
        op: str,
        args: dict[str, Any],
        risk: str,
    ) -> ToolResult:
        """Verify ``permit`` and then actually perform the effect.

        The verification lives here, not in the callers, so that *every* route
        into an adapter -- ``invoke``, ``invoke_effect``, or a direct call from
        inside the broker -- is gated by exactly one verified, single-use
        permit.
        """
        if permit is None:
            raise ToolNotPermitted("a permit is required before any effect can be applied")
        if self._permits is None:
            raise ToolNotPermitted("no permit service is configured on this broker")
        # Verify-and-consume runs before the adapter is even looked up, so a
        # missing, replayed, or tampered permit never reaches the effect.
        self._permits.verify_and_consume(permit, tool, args, op=op)
        self.seal()
        if tool == "shell":
            return self._run_sandboxed(permit, tool, op, args, risk)
        fn = self._adapters.get((tool, op))
        if fn is None:
            raise ToolExecutionError(f"no adapter registered for {tool}.{op}")
        # Audit-before-effect: record the authorised intent first; if the audit
        # write fails, the adapter is never invoked.
        self._record(permit, tool, op, risk, "executed", args=args)
        try:
            applied = fn(args)
        except ToolBrokerError:
            self._record(permit, tool, op, risk, "failed", args=args)
            raise
        except Exception as exc:  # noqa: BLE001 - surface adapter failures uniformly
            self._record(permit, tool, op, risk, "failed", args=args)
            raise ToolExecutionError(f"{tool}.{op} failed: {exc}") from exc
        payload = applied if isinstance(applied, Mapping) else {"value": applied}
        return ToolResult(
            tool=tool,
            op=op,
            risk_class=risk,
            executed=True,
            simulated=False,
            recorded=True,
            applied=dict(payload),
            note="executed",
        )

    def _simulate(
        self,
        permit: Any,
        tool: str,
        op: str,
        args: dict[str, Any],
        risk: str,
    ) -> ToolResult:
        """Record the intended A2/A3 effect; never perform it."""
        effect = MockEffect(tool=tool, op=op, risk_class=risk, args=args)
        result = ToolResult(
            tool=tool,
            op=op,
            risk_class=risk,
            executed=False,
            simulated=True,
            recorded=True,
            applied=effect.as_dict(),
            note=f"{risk} effect recorded as a simulation and not performed",
        )
        self._record(
            permit,
            tool,
            op,
            risk,
            f"simulated:{risk}",
            args=args,
            result=result,
            extra={"simulated": True, "executed": False},
        )
        return result

    def _run_sandboxed(
        self,
        permit: Any,
        tool: str,
        op: str,
        args: dict[str, Any],
        risk: str,
    ) -> ToolResult:
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ToolExecutionError("shell commands require a non-empty 'command'")
        if not self.bwrap_available():
            self._record(
                permit, tool, op, risk, "sandbox-unavailable", args=args,
                extra={"executed": False, "reason": "bwrap-missing"},
            )
            raise ToolExecutionError(
                f"bubblewrap sandbox {self._bwrap!r} is unavailable; refusing to run unsandboxed"
            )
        self._workspace.mkdir(parents=True, exist_ok=True)
        # Audit-before-effect: the intent row must be committed before the
        # command can run. If the append raises, it propagates and no process is
        # ever spawned, so a sandboxed command cannot take effect unaudited.
        self._record(
            permit, tool, op, risk, "executing", args=args,
            extra={"executed": True, "phase": "pre-effect"},
        )
        # A read-only operation (A0) gets a read-only workspace bind, so a
        # declared read cannot write through the sandbox.
        argv = self._sandbox_argv(command, read_only=(risk == "A0"))
        started = time.monotonic()
        # A configured pids cgroup is a hard per-run PID cap, so it is built in
        # the parent before the spawn. Any failure here fails the whole call
        # closed: a configured cap that cannot be enforced must never silently
        # degrade into an uncapped run.
        child = None
        procs_fd = None
        if self._pids_cgroup is not None:
            try:
                child, procs_fd = self._open_pids_child()
            except Exception as exc:  # noqa: BLE001 - any setup failure fails closed
                self._record(
                    permit, tool, op, risk, "sandbox-unavailable", args=args,
                    extra={
                        "executed": False,
                        "reason": _pids_audit_reason(exc),
                    },
                )
                raise ToolExecutionError(
                    f"pids cgroup {str(self._pids_cgroup)!r} is unavailable: {exc}"
                ) from exc
        try:
            proc = subprocess.Popen(  # noqa: S603 - argv is fixed; command runs in bwrap
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=dict(MINIMAL_ENV),
                preexec_fn=self._preexec_fn(procs_fd),
                start_new_session=True,
                close_fds=True,
                # The child's cgroup.procs fd is the one descriptor the run
                # needs; every other descriptor stays closed.
                pass_fds=(procs_fd,) if procs_fd is not None else (),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            if child is not None:
                self._discard_pids_child(child)
            self._record(
                permit, tool, op, risk, "sandbox-unavailable", args=args,
                extra={"executed": False, "reason": f"spawn-failed:{exc.__class__.__name__}"},
            )
            raise ToolExecutionError(f"could not start the bubblewrap sandbox: {exc}") from exc
        finally:
            if procs_fd is not None:
                os.close(procs_fd)

        cap = max(0, int(self._limits.max_output_bytes))
        stdout = bytearray()
        stderr = bytearray()
        truncated = {"stdout": [False], "stderr": [False]}
        readers = [
            threading.Thread(
                target=_read_capped,
                args=(proc.stdout, cap, stdout, truncated["stdout"]),
                daemon=True,
            ),
            threading.Thread(
                target=_read_capped,
                args=(proc.stderr, cap, stderr, truncated["stderr"]),
                daemon=True,
            ),
        ]
        for reader in readers:
            reader.start()

        timed_out = False
        try:
            proc.wait(timeout=self._limits.wall_timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill_process_group(proc)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

        # The process-group kill only reaches the session the sandbox leads, so
        # both branches drain the run's own cgroup as well: every pid still in
        # it is killed and the child removed. On the timeout branch this happens
        # before the readers are joined, so closing the survivors' ends of the
        # pipes lets the readers reach EOF instead of waiting out their timeout.
        pids_report = self._reap_pids_child(child) if child is not None else None
        containment_failed = bool(
            pids_report is not None and not pids_report["drain"]["complete"]
        )
        for reader in readers:
            reader.join(timeout=10)

        if timed_out:
            elapsed = time.monotonic() - started
            self._record(
                permit, tool, op, risk,
                "containment-incomplete" if containment_failed else "timeout", args=args,
                extra={"executed": True, "reason": "wall-timeout",
                       "elapsed_s": round(elapsed, 3), "pids_cgroup": pids_report},
            )
            return ToolResult(
                tool=tool, op=op, risk_class=risk, executed=True, simulated=False,
                recorded=True, ok=False, exit_code=None,
                # Whatever the command managed to say before it was killed is
                # evidence for the timeout, so it is returned rather than dropped.
                stdout=stdout.decode("utf-8", errors="replace")[:cap],
                stderr=stderr.decode("utf-8", errors="replace")[:cap],
                applied={"pids_cgroup": pids_report} if pids_report is not None else None,
                note=("containment-incomplete: sandbox timed out and cleanup failed"
                      if containment_failed else
                      f"wall timeout of {self._limits.wall_timeout_seconds:g}s exceeded"),
            )
        applied = {
            "exit_code": proc.returncode,
            "stdout_truncated": bool(truncated["stdout"][0]),
            "stderr_truncated": bool(truncated["stderr"][0]),
        }
        if pids_report is not None:
            applied["pids_cgroup"] = pids_report
        result = ToolResult(
            tool=tool,
            op=op,
            risk_class=risk,
            executed=True,
            simulated=False,
            recorded=True,
            ok=proc.returncode == 0 and not containment_failed,
            exit_code=proc.returncode,
            stdout=stdout.decode("utf-8", errors="replace")[:cap],
            stderr=stderr.decode("utf-8", errors="replace")[:cap],
            applied=applied,
            note=("containment-incomplete: sandbox child cleanup failed"
                  if containment_failed else "sandboxed command finished"),
        )
        self._record(
            permit, tool, op, risk,
            "containment-incomplete" if containment_failed else "executed",
            args=args, result=result,
        )
        return result

    @staticmethod
    def _kill_process_group(proc: subprocess.Popen) -> None:
        """SIGKILL the whole process group of ``proc`` (falling back to the child)."""
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            return
        except OSError:
            pass
        try:
            proc.kill()
        except OSError:
            pass

    # -- sandbox construction ---------------------------------------------
    def _sandbox_argv(self, command: str, *, read_only: bool = False) -> list[str]:
        argv = [
            self._bwrap,
            "--unshare-all",
            "--unshare-pid",
            "--new-session",
            "--die-with-parent",
            "--cap-drop",
            "ALL",
        ]
        for path in READ_ONLY_BINDS:
            if os.path.exists(path):
                argv += ["--ro-bind", path, path]
        # tmpfs for /tmp and HOME, then the single read-write workspace bind
        # (ordered after the tmpfs so a workspace under /tmp is not shadowed).
        argv += ["--tmpfs", "/tmp", "--tmpfs", "/home"]
        # The workspace is the one read-write bind; a read-only operation gets
        # a read-only bind instead so it cannot write through the sandbox.
        bind_flag = "--ro-bind" if read_only else "--bind"
        argv += [bind_flag, str(self._workspace), str(self._workspace)]
        argv += ["--proc", "/proc", "--dev", "/dev"]
        argv += ["--chdir", str(self._workspace)]
        for key, value in MINIMAL_ENV.items():
            argv += ["--setenv", key, value]
        # ``--`` ends option parsing so the command can never be mistaken for a
        # bwrap flag, even if it begins with ``-``.
        argv += ["--", "/bin/sh", "-c", command]
        return argv

    def _preexec_fn(self, cgroup_fd: int | None = None) -> Callable[[], None]:
        limits = self._limits
        # When the cgroup is configured, its per-run pids.max is authoritative.
        # Capture the inherited hard ceiling here in the parent process.
        nproc = (
            resource.getrlimit(resource.RLIMIT_NPROC)[1]
            if cgroup_fd is not None else int(limits.nproc)
        )

        def _apply() -> None:
            cpu = int(limits.cpu_seconds)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
            resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))
            address_space = int(limits.address_space_bytes)
            resource.setrlimit(resource.RLIMIT_AS, (address_space, address_space))
            file_size = int(limits.file_size_bytes)
            resource.setrlimit(resource.RLIMIT_FSIZE, (file_size, file_size))
            if cgroup_fd is not None:
                # Move the child -- and everything it forks, since cgroup
                # membership is inherited -- into this run's own cgroup. One
                # async-signal-safe write to an fd the parent opened, so nothing
                # allocation-heavy runs between fork and exec. A failure is
                # deliberately not swallowed: an uncapped run is exactly what the
                # cap exists to prevent, and it aborts the spawn instead.
                try:
                    os.write(cgroup_fd, b"0")
                finally:
                    # Do not leak a writable control FD into the untrusted shell.
                    os.close(cgroup_fd)

        return _apply

    # -- pids cgroup ---------------------------------------------------------
    def _open_pids_child(self) -> tuple[Path, int]:
        """Create this run's cgroup child and return it with its ``cgroup.procs`` fd.

        The child is a private, single-run directory under the delegated base
        (``run-<uuid4hex>``). ``pids.max`` is written before the spawn, so the
        cap is already in force when the child is moved in, and ``cgroup.procs``
        is opened here in the parent rather than inside ``preexec_fn``. Both are
        files the kernel made when the child appeared, so a base that is a plain
        directory is refused here rather than capped in name only.
        """
        base = self._pids_cgroup
        if base is None:
            raise ToolBrokerError("no pids cgroup is configured")
        child = base / f"run-{uuid.uuid4().hex}"
        _mkdir_pids_child(child)
        try:
            _require_cgroup_control_files(child)
            _write_cgroup_value(child / "pids.max", f"{int(self._limits.tasks_max)}\n")
            fd = _open_cgroup_file(child / "cgroup.procs")
        except BaseException:
            self._discard_pids_child(child)
            raise
        return child, fd

    def _reap_pids_child(self, child: Path) -> dict[str, Any]:
        """Drain and remove this run's cgroup; report failures, never conceal them."""
        report: dict[str, Any] = {
            "tasks_max": int(self._limits.tasks_max),
            "pids_max_events": _pids_max_events(child),
            "cgroup": child.name,
        }
        drain: dict[str, Any] = {
            "complete": False, "remaining_pids": [],
            "read_error": None, "remove_error": None, "signal_errors": [],
        }
        report["drain"] = drain
        protected = _protected_pids()
        deadline = time.monotonic() + PIDS_CGROUP_DRAIN_SECONDS
        try:
            while True:
                try:
                    pids = _cgroup_procs(child)
                except OSError as exc:
                    drain["read_error"] = type(exc).__name__
                    break
                if not pids:
                    break
                victims = _signalable_pids(pids, protected)
                if not victims:
                    drain["remaining_pids"] = pids
                    break
                for pid in victims:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except (OSError, OverflowError) as exc:
                        drain["signal_errors"].append(type(exc).__name__)
                if time.monotonic() >= deadline:
                    break
                time.sleep(PIDS_CGROUP_POLL_SECONDS)
            if drain["read_error"] is None:
                try:
                    drain["remaining_pids"] = _cgroup_procs(child)
                except OSError as exc:
                    drain["read_error"] = type(exc).__name__
        finally:
            try:
                child.rmdir()
            except FileNotFoundError:
                pass
            except OSError as exc:
                drain["remove_error"] = type(exc).__name__
        drain["complete"] = (
            drain["read_error"] is None and not drain["remaining_pids"]
            and drain["remove_error"] is None and not drain["signal_errors"]
        )
        return report

    @staticmethod
    def _discard_pids_child(child: Path) -> None:
        """Best-effort removal of a child cgroup that never ran a process."""
        try:
            child.rmdir()
        except OSError:
            pass

    # -- audit -------------------------------------------------------------
    def _record(
        self,
        permit: Any,
        tool: str,
        op: str,
        risk: str,
        outcome: str,
        *,
        args: Mapping[str, Any] | None = None,
        result: ToolResult | None = None,
        extra: Mapping[str, Any] | None = None,
        permit_id: str | None = None,
    ) -> None:
        governor_result: dict[str, Any] = {
            "tool": str(tool),
            "op": str(op),
            "risk_class": str(risk),
            "outcome": str(outcome),
        }
        if extra:
            governor_result.update(dict(extra))
        request_hash = str(getattr(permit, "normalized_args_hash", "") or "")
        if not request_hash:
            request_hash = _hash_value(args or {})
        self._audit.append(
            goal_version=str(getattr(permit, "goal_id", "") or ""),
            plan_version=str(getattr(permit, "plan_id", "") or ""),
            policy_version=str(getattr(permit, "policy_version", "") or ""),
            permit_id=permit_id if permit_id is not None else str(getattr(permit, "permit_id", "") or ""),
            tool_request_hash=request_hash,
            tool_result_hash=_hash_value(result.as_dict()) if result is not None else "",
            governor_result=governor_result,
            outcome=str(outcome),
            budget_delta=0.0,
        )

    # -- helpers -----------------------------------------------------------
    def _risk_class(self, tool: str, op: str, *, default: str | None = None) -> str:
        try:
            return self._manifest.risk_class(tool, op)
        except Exception as exc:  # noqa: BLE001 - unknown ops fail closed for invoke
            if default is not None:
                return default
            raise ToolExecutionError(f"unknown tool operation {tool}.{op}") from exc

    @staticmethod
    def _normalise_args(args: Mapping[str, Any] | None) -> dict[str, Any]:
        if args is None:
            return {}
        if not isinstance(args, Mapping):
            raise ToolNotPermitted("tool arguments must be a mapping")
        return dict(args)

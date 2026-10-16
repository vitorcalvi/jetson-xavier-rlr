"""ToolBroker tests: the permit gate, sandboxed A0/A1 runs, and recorded A2/A3.

The escape tests only run when ``/usr/bin/bwrap`` is present and executable; on
a host without bubblewrap they are skipped, and the always-run tests cover the
permit gate plus the "A2/A3 are recorded but never performed" contract.
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import time
import uuid
from collections.abc import Callable
from dataclasses import replace

import pytest

from reflective_loop_runtime.kernel.audit import AuditLog
from reflective_loop_runtime.kernel.manifest import (
    FieldSpec,
    OperationSpec,
    ToolSpec,
)
from reflective_loop_runtime.kernel.permit import (
    PermitIssuer,
    PermitReplayError,
    PermitService,
    PermitVerificationError,
)
from reflective_loop_runtime.kernel.policy import MissionPolicyStore, sign_envelope
from reflective_loop_runtime.kernel.toolbroker import (
    DEFAULT_BWRAP,
    MINIMAL_ENV,
    READ_ONLY_BINDS,
    ResourceLimits,
    ToolBroker,
    ToolExecutionError,
    ToolNotPermitted,
    broker_manifest,
)

DEV_SECRET = "dev-only-shared-secret"
ISSUER_KEY = b"governor-owned-hmac-key-0123456789"

BWRAP_PRESENT = os.path.isfile(DEFAULT_BWRAP) and os.access(DEFAULT_BWRAP, os.X_OK)


def _bwrap_functional() -> bool:
    """True only when bwrap is present *and* can actually start a sandbox.

    A host may ship the binary but disable unprivileged user namespaces; in that
    case the escape tests are skipped rather than reporting false failures.
    """
    if not BWRAP_PRESENT:
        return False
    # Probe with the same read-only binds the broker supplies, so that a
    # merged-usr host (where ``/bin`` and ``/lib`` are symlinks into ``/usr``)
    # still resolves ``/bin/true`` and its dynamic loader. The bare-argv probe
    # the broker never actually uses would report "No such file or directory" on
    # such hosts and skip the escape tests despite a working sandbox.
    argv = [DEFAULT_BWRAP, "--unshare-all", "--die-with-parent"]
    for path in READ_ONLY_BINDS:
        if os.path.exists(path):
            argv += ["--ro-bind", path, path]
    argv += ["--proc", "/proc", "--dev", "/dev", "--", "/bin/true"]
    try:
        probe = subprocess.run(
            argv,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


requires_bwrap = pytest.mark.skipif(
    not _bwrap_functional(),
    reason="bubblewrap sandbox not available or not functional on this host",
)

# A tiny stand-in for bubblewrap. It does NOT isolate anything; it only mimics
# the option parsing (``--setenv``, ``--`` and value-taking flags) and then
# execs the command, so the broker's process plumbing -- stdin, environment,
# rlimits, output caps and process-group kill -- can be exercised even on hosts
# without a functional bwrap. The namespace/bind escape proofs stay under
# ``requires_bwrap``.
_FAKE_BWRAP = """#!/bin/sh
while [ $# -gt 0 ]; do
  case "$1" in
    --setenv) export "$2=$3"; shift 3 ;;
    --ro-bind|--bind|--tmpfs|--proc|--dev|--chdir) shift 2 ;;
    --) shift; break ;;
    *) shift ;;
  esac
done
exec "$@"
"""


def _write_fake_bwrap(path):
    path.write_text(_FAKE_BWRAP, encoding="utf-8")
    path.chmod(0o755)
    return path


def _setenv_pairs(argv):
    pairs: dict[str, str] = {}
    i = 0
    while i < len(argv):
        if argv[i] == "--setenv":
            pairs[argv[i + 1]] = argv[i + 2]
            i += 3
        else:
            i += 1
    return pairs


def _permissive_nproc() -> int:
    """The host's NPROC hard limit, so a sandboxed ``fork`` is not starved.

    ``RLIMIT_NPROC`` counts every process the *user* owns, not just this tree, so
    the broker's small production default can spuriously fail to fork on a busy
    host. The fake-bwrap tests therefore keep the rlimit at the host maximum;
    they still exercise the cap mechanism via the other limits.
    """
    _, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    if hard == resource.RLIM_INFINITY:
        return 4096
    return int(hard)


def _token_field(stdout: str, token: str, name: str) -> str | None:
    """What the run reported under ``name``, or ``None`` if it reported nothing.

    Every line a run prints is tagged with that run's own token, so one run's
    output can never be read as another's -- and "the line is not there" stays
    a possible answer instead of being guessed at.
    """
    prefix = f"{token} {name}="
    for line in stdout.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return None


def _pid_namespace(pid: int) -> str | None:
    """The pid-namespace of ``pid`` as an inode string; None once it is gone."""
    try:
        return os.readlink(f"/proc/{pid}/ns/pid")
    except OSError:
        return None


def _in_namespace(namespace: str) -> list[int]:
    """The pids that are in the pid-namespace ``namespace`` right now."""
    found: list[int] = []
    try:
        entries = os.listdir("/proc")
    except OSError:  # pragma: no cover - /proc is always there on Linux
        return found
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            if os.readlink(f"/proc/{pid}/ns/pid") == namespace:
                found.append(pid)
        except OSError:
            continue  # it exited while it was being looked at
    return found


def _wait_for_no_survivors(
    probe: Callable[[], list[int]], timeout: float = 5.0
) -> list[int]:
    """Poll ``probe`` for at most ``timeout`` seconds; return what is left."""
    deadline = time.monotonic() + timeout
    while True:
        alive = probe()
        if not alive or time.monotonic() >= deadline:
            return alive
        time.sleep(0.05)


@pytest.fixture
def fake_bwrap(tmp_path):
    return _write_fake_bwrap(tmp_path / "fake-bwrap")


@pytest.fixture
def fake_broker(tmp_path, service, fake_bwrap):
    workspace = tmp_path / "fake-workspace"
    workspace.mkdir()
    return ToolBroker(
        service,
        audit=AuditLog(":memory:"),
        workspace=workspace,
        manifest=broker_manifest(),
        bwrap_path=str(fake_bwrap),
        limits=ResourceLimits(nproc=_permissive_nproc()),
    )


def _bundle(**overrides):
    bundle = {
        "mission_id": "mission-7",
        "objective": "invoke tools under a permit",
        "invariants": ["never_shell_out_unsandboxed"],
        # Empty lists impose no capability/domain restriction here; the
        # restrictive case is covered by the security-regression suite.
        "allowed_domains": [],
        "capabilities": [],
        "budgets": {},
        "standing_mandates": [],
        "policy_version": "1.0.0",
    }
    bundle.update(overrides)
    return bundle


def _policy(tmp_path, *, mandates=(), version="1.0.0"):
    path = tmp_path / f"policy-{version}-{len(mandates)}.json"
    envelope = sign_envelope(
        _bundle(policy_version=version, standing_mandates=list(mandates)),
        secret=DEV_SECRET,
    )
    path.write_text(json.dumps(envelope), encoding="utf-8")
    return MissionPolicyStore(secret=DEV_SECRET, allow_dev_hmac=True).load(path)


def _service(tmp_path, manifest, *, mandates=(), version="1.0.0"):
    issuer = PermitIssuer(ISSUER_KEY)
    policy = _policy(tmp_path, mandates=mandates, version=version)
    return PermitService(issuer, policy, manifest=manifest)


@pytest.fixture
def service(tmp_path):
    return _service(tmp_path, broker_manifest())


@pytest.fixture
def broker(tmp_path, service):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return ToolBroker(
        service,
        audit=AuditLog(":memory:"),
        workspace=workspace,
        manifest=broker_manifest(),
    )


class TestPermitGate:
    def test_no_permit_is_rejected_without_a_call(self, broker):
        with pytest.raises(ToolNotPermitted):
            broker.invoke(None, "shell", "run", {"command": "true"})
        assert len(broker.audit) == 0

    def test_consumed_permit_is_rejected(self, tmp_path):
        manifest = broker_manifest()
        service = _service(tmp_path, manifest)
        broker = ToolBroker(
            service, audit=AuditLog(":memory:"), workspace=tmp_path / "ws", manifest=manifest
        )
        broker.register_effect("control", "reflect", lambda args: {})
        permit = service.issue(tool="control", op="reflect", args={}, goal_id="goal-1")

        broker.invoke(permit, "control", "reflect", {})
        assert service.consumed(permit.nonce) is True
        with pytest.raises(PermitReplayError):
            broker.invoke(permit, "control", "reflect", {})

    def test_permit_service_is_required(self):
        local = ToolBroker(None)
        with pytest.raises(ToolNotPermitted):
            local.invoke(object(), "shell", "run", {"command": "true"})

    def test_valid_permit_runs_registered_effect_and_consumes_nonce(self, tmp_path):
        manifest = broker_manifest()
        manifest.register(
            ToolSpec(
                "probe",
                {
                    "record": OperationSpec(
                        name="record",
                        input_schema={"value": FieldSpec("int", min=0, max=100)},
                        effects=("a probe value is recorded",),
                        risk_class="A1",
                        reversibility="reversible",
                        idempotency="idempotent",
                        compensation="clear the recorded probe value",
                    )
                },
            )
        )
        service = _service(tmp_path, manifest)
        calls: list[int] = []
        broker = ToolBroker(
            service, audit=AuditLog(":memory:"), workspace=tmp_path / "ws", manifest=manifest
        )
        broker.register_effect(
            "probe", "record", lambda args: calls.append(args["value"]) or {"recorded": args["value"]}
        )
        permit = service.issue(tool="probe", op="record", args={"value": 1}, goal_id="goal-1")

        first = broker.invoke(permit, "probe", "record", {"value": 1})
        assert first.executed is True
        assert first.applied == {"recorded": 1}
        assert calls == [1]
        assert service.consumed(permit.nonce) is True
        assert len(broker.audit) == 1

        with pytest.raises(PermitReplayError):
            broker.invoke(permit, "probe", "record", {"value": 1})
        assert calls == [1]

    def test_argument_substitution_is_rejected_before_any_effect(self, tmp_path):
        manifest = broker_manifest()
        service = _service(tmp_path, manifest)
        calls: list[dict] = []
        broker = ToolBroker(
            service, audit=AuditLog(":memory:"), workspace=tmp_path / "ws", manifest=manifest
        )
        broker.register_effect("state", "set_load_target", lambda args: calls.append(args) or {"ok": True})
        permit = service.issue(
            tool="state", op="set_load_target", args={"load": 0.5}, goal_id="goal-1"
        )
        with pytest.raises(PermitVerificationError):
            broker.invoke(permit, "state", "set_load_target", {"load": 0.9})
        assert calls == []

    def test_authorised_but_unregistered_effect_fails_closed(self, tmp_path):
        manifest = broker_manifest()
        service = _service(tmp_path, manifest)
        broker = ToolBroker(
            service, audit=AuditLog(":memory:"), workspace=tmp_path / "ws", manifest=manifest
        )
        permit = service.issue(
            tool="state", op="set_load_target", args={"load": 0.5}, goal_id="goal-1"
        )
        with pytest.raises(ToolExecutionError):
            broker.invoke(permit, "state", "set_load_target", {"load": 0.5})


class TestMockedRiskClasses:
    def test_a2_effect_is_recorded_but_never_executed(self, tmp_path):
        manifest = broker_manifest()
        service = _service(tmp_path, manifest)
        executed: list[dict] = []
        broker = ToolBroker(
            service, audit=AuditLog(":memory:"), workspace=tmp_path / "ws", manifest=manifest
        )
        # If the broker ever performed the effect, this adapter would fire.
        broker.register_effect(
            "goal", "complete", lambda args: executed.append(args) or {"completed": True}
        )
        permit = service.issue(tool="goal", op="complete", args={"goal_id": 7}, goal_id="goal-1")

        result = broker.invoke(permit, "goal", "complete", {"goal_id": 7})

        assert result.simulated is True
        assert result.executed is False
        assert result.risk_class == "A2"
        assert executed == []
        assert result.applied["intended_effect"] == "goal.complete"
        events = broker.audit.events()
        assert len(events) == 1
        assert events[0].outcome == "simulated:A2"
        assert events[0].permit_id == permit.permit_id
        assert events[0].governor_result["executed"] is False

    def test_a3_effect_is_recorded_but_never_executed(self, tmp_path):
        manifest = broker_manifest()
        service = _service(tmp_path, manifest, mandates=["memory.forget"])
        executed: list[dict] = []
        broker = ToolBroker(
            service, audit=AuditLog(":memory:"), workspace=tmp_path / "ws", manifest=manifest
        )
        broker.register_effect(
            "memory", "forget", lambda args: executed.append(args) or {"deleted": True}
        )
        permit = service.issue(
            tool="memory", op="forget", args={"memory_id": 1}, goal_id="goal-1"
        )

        result = broker.invoke(permit, "memory", "forget", {"memory_id": 1})

        assert result.simulated is True
        assert result.executed is False
        assert result.risk_class == "A3"
        assert executed == []
        assert broker.audit.events()[0].outcome == "simulated:A3"

    def test_simulation_still_consumes_the_permit(self, tmp_path):
        manifest = broker_manifest()
        service = _service(tmp_path, manifest)
        broker = ToolBroker(
            service, audit=AuditLog(":memory:"), workspace=tmp_path / "ws", manifest=manifest
        )
        permit = service.issue(tool="goal", op="complete", args={"goal_id": 3}, goal_id="goal-1")
        broker.invoke(permit, "goal", "complete", {"goal_id": 3})
        assert service.consumed(permit.nonce) is True
        with pytest.raises(PermitReplayError):
            broker.invoke(permit, "goal", "complete", {"goal_id": 3})


class TestPermitGatedEffectPath:
    """The executor's in-process effects run only under a verified permit."""

    def test_invoke_effect_requires_a_permit(self, broker):
        broker.register_effect("control", "reflect", lambda args: {})
        with pytest.raises(ToolNotPermitted):
            broker.invoke_effect(None, "control", "reflect", {})
        assert len(broker.audit) == 0

    def test_invoke_effect_runs_under_a_permit_and_audits(self, tmp_path):
        manifest = broker_manifest()
        service = _service(tmp_path, manifest)
        broker = ToolBroker(
            service, audit=AuditLog(":memory:"), workspace=tmp_path / "ws", manifest=manifest
        )
        broker.register_effect("control", "reflect", lambda args: {"value": 5})
        permit = service.issue(tool="control", op="reflect", args={}, goal_id="goal-1")

        result = broker.invoke_effect(permit, "control", "reflect", {})
        assert result.executed is True
        assert result.recorded is True
        assert result.applied == {"value": 5}
        assert len(broker.audit) == 1
        assert broker.audit.events()[0].outcome == "executed"
        # Single-use: the same permit cannot drive a second effect.
        with pytest.raises(PermitReplayError):
            broker.invoke_effect(permit, "control", "reflect", {})

    def test_invoke_effect_without_adapter_is_rejected(self, tmp_path):
        manifest = broker_manifest()
        service = _service(tmp_path, manifest)
        broker = ToolBroker(service, audit=AuditLog(":memory:"), manifest=manifest)
        permit = service.issue(
            tool="state", op="set_load_target", args={"load": 0.5}, goal_id="goal-1"
        )
        with pytest.raises(ToolExecutionError):
            broker.invoke_effect(permit, "state", "set_load_target", {"load": 0.5})


class TestSandboxShape:
    def test_shell_fails_closed_without_bwrap(self, tmp_path, service):
        broker = ToolBroker(
            service,
            audit=AuditLog(":memory:"),
            workspace=tmp_path / "ws",
            bwrap_path="/nonexistent/definitely-not-bwrap",
        )
        permit = service.issue(
            tool="shell", op="run", args={"command": "true"}, goal_id="goal-1"
        )
        with pytest.raises(ToolExecutionError):
            broker.invoke(permit, "shell", "run", {"command": "true"})
        assert broker.audit.events()[0].outcome == "sandbox-unavailable"

    def test_sandbox_argv_declares_the_isolation(self, broker):
        argv = broker._sandbox_argv("echo hi")
        assert argv[0] == DEFAULT_BWRAP
        assert "--unshare-all" in argv
        assert "--die-with-parent" in argv
        for path in ("/usr", "/lib", "/bin", "/etc"):
            assert path in argv
        assert argv.count("--ro-bind") >= 1
        assert "--tmpfs" in argv
        assert "--bind" in argv
        assert str(broker.workspace) in argv
        assert argv[-3:] == ["/bin/sh", "-c", "echo hi"]

    def test_sandbox_argv_hardens_the_sandbox(self, broker):
        argv = broker._sandbox_argv("echo hi")
        assert "--unshare-all" in argv
        assert "--unshare-pid" in argv
        assert "--new-session" in argv
        assert "--die-with-parent" in argv
        assert argv[argv.index("--cap-drop") + 1] == "ALL"
        # An explicit separator keeps the command from ever reading as a flag.
        assert argv[-4] == "--"
        assert argv[-3:] == ["/bin/sh", "-c", "echo hi"]

    def test_sandbox_argv_pins_a_synthesized_environment(self, broker):
        argv = broker._sandbox_argv("true")
        assert _setenv_pairs(argv) == dict(MINIMAL_ENV)

    def test_sandbox_argv_mounts_tmpfs_for_tmp_and_home(self, broker):
        argv = broker._sandbox_argv("true")
        mounts = [argv[i + 1] for i, item in enumerate(argv) if item == "--tmpfs"]
        assert "/tmp" in mounts
        assert "/home" in mounts


class TestSandboxHardening:
    """Process-plumbing properties, exercised via the fake bwrap harness.

    These run even where bubblewrap is unavailable because they only cover what
    the broker itself owns: stdin, the synthesized environment, rlimits, output
    caps and the process-group kill. The namespace/bind escape proofs remain in
    :class:`TestSandboxEscape`.
    """

    def _run(self, service, broker, command, **limit_overrides):
        target = broker
        if limit_overrides:
            target = ToolBroker(
                service,
                audit=AuditLog(":memory:"),
                workspace=broker.workspace,
                manifest=broker.manifest,
                bwrap_path=str(broker._bwrap),
                limits=replace(broker.limits, **limit_overrides),
            )
        permit = service.issue(tool="shell", op="run", args={"command": command}, goal_id="goal-1")
        return target.invoke(permit, "shell", "run", {"command": command})

    def test_stdin_comes_from_dev_null(self, service, fake_broker):
        result = self._run(service, fake_broker, "readlink /proc/self/fd/0")
        assert result.ok is True
        assert result.stdout.strip() == "/dev/null"

    def test_inherited_environment_variables_are_dropped(self, service, fake_broker, monkeypatch):
        monkeypatch.setenv("RLR_TOOLBROKER_CANARY", "leaked-secret")
        result = self._run(
            service,
            fake_broker,
            "printenv RLR_TOOLBROKER_CANARY || echo absent; printenv HOME; printenv PATH",
        )
        assert "leaked-secret" not in result.stdout
        assert "absent" in result.stdout
        assert "/home" in result.stdout
        assert "/usr/bin" in result.stdout

    def test_stdout_is_capped(self, service, fake_broker):
        result = self._run(
            service,
            fake_broker,
            "head -c 1000000 /dev/zero | tr '\\0' X",
            max_output_bytes=4096,
        )
        assert len(result.stdout) <= 4096
        assert result.applied["stdout_truncated"] is True

    def test_stderr_is_capped(self, service, fake_broker):
        result = self._run(
            service,
            fake_broker,
            "head -c 1000000 /dev/zero | tr '\\0' Y 1>&2",
            max_output_bytes=4096,
        )
        assert len(result.stderr) <= 4096
        assert result.applied["stderr_truncated"] is True

    def test_file_size_rlimit_caps_writes(self, service, fake_broker):
        target = fake_broker.workspace / "big.bin"
        result = self._run(
            service,
            fake_broker,
            f"head -c {8 * 1024 * 1024} /dev/zero > {target}",
            file_size_bytes=1024 * 1024,
        )
        assert result.ok is False
        assert target.stat().st_size <= 1024 * 1024

    def test_wall_timeout_kills_the_whole_process_group(self, service, fake_broker):
        pidfile = fake_broker.workspace / "child.pid"
        command = f"sleep 300 & echo $! > {pidfile}; sleep 300"
        started = time.monotonic()
        result = self._run(service, fake_broker, command, wall_timeout_seconds=2)
        assert time.monotonic() - started < 20
        assert result.ok is False
        assert result.exit_code is None
        assert "timeout" in result.note.lower()
        child_pid = int(pidfile.read_text(encoding="utf-8").strip())
        deadline = time.monotonic() + 5
        while os.path.exists(f"/proc/{child_pid}") and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not os.path.exists(f"/proc/{child_pid}")


@requires_bwrap
class TestSandboxEscape:
    def _run(self, service, broker, command, **limit_overrides):
        limits = ResourceLimits(**limit_overrides) if limit_overrides else broker.limits
        target = (
            broker
            if not limit_overrides
            else ToolBroker(
                service, audit=AuditLog(":memory:"), workspace=broker.workspace, limits=limits
            )
        )
        permit = service.issue(tool="shell", op="run", args={"command": command}, goal_id="goal-1")
        return target.invoke(permit, "shell", "run", {"command": command})

    def test_network_egress_fails(self, service, broker):
        result = self._run(service, broker, "curl -sS --max-time 5 http://example.com")
        assert result.ok is False

    def test_write_outside_workspace_fails(self, service, broker):
        probe = "/etc/rlr_toolbroker_escape_probe"
        result = self._run(service, broker, f"printf x > {probe}")
        assert result.ok is False
        assert not os.path.exists(probe)

    def test_fork_bomb_is_contained(self, service, broker):
        """A 500-fork storm stops, says why it stopped, and leaves nothing running.

        The contract is deliberately load-independent. ``RLIMIT_NPROC`` counts
        every process the *user* owns, so whether the storm reaches 500 or is
        cut short depends on the host -- and both outcomes pass here. What has
        to hold on every host is that the storm stops instead of running away,
        that being cut short shows up as a fork diagnostic rather than as
        silence, that the shell which started it survives to report, and that
        nothing it forked outlives the run.

        The storm therefore runs in a subshell of its own: a refused fork is
        fatal to ``/bin/sh`` -- dash exits the whole shell on EAGAIN, printing
        nothing after it -- so the storm has to live somewhere that can die
        alone. Only then does the outer shell's ``parent-ok`` mean what it
        says.
        """
        token = uuid.uuid4().hex
        pidfile = broker.workspace / "bomb.pids"
        errfile = broker.workspace / "bomb.err"
        nsfile = broker.workspace / "bomb.ns"
        # Both files exist before the storm runs: a storm refused on its very
        # first fork leaves nothing behind for the test to read.
        pidfile.write_text("", encoding="utf-8")
        errfile.write_text("", encoding="utf-8")
        command = (
            # The pid namespace the sandbox gave this run, so the host-side
            # check below knows which processes, if any, are still in it.
            f"readlink /proc/self/ns/pid > {nsfile}; "
            f"( i=0; while [ $i -lt 500 ]; do (:) & echo $! >> {pidfile}; "
            f"i=$((i+1)); done; echo '{token} forks='$i ) 2> {errfile}; "
            f"bomb_rc=$?; echo '{token} bomb-rc='$bomb_rc; "
            f"echo '{token} parent-ok'"
        )
        # The rlimit is held at the host maximum so the storm is measured
        # against the sandbox, not against how loaded the machine happens to be.
        result = self._run(service, broker, command, nproc=_permissive_nproc())

        assert result.exit_code is not None, result.note
        assert "timeout" not in result.note.lower(), result.note
        # The reporting shell outlived the storm: the storm died alone.
        assert f"{token} parent-ok" in result.stdout
        started = [int(item) for item in pidfile.read_text(encoding="utf-8").split()]
        assert len(started) <= 500
        captured = result.stdout + result.stderr + errfile.read_text(encoding="utf-8")
        forks = _token_field(result.stdout, token, "forks")
        bomb_rc = int(_token_field(result.stdout, token, "bomb-rc"))
        if forks is None or int(forks) < 500:
            # Cut short. The only reason accepted here is a fork the kernel or
            # the rlimit refused, and the shell has to have said which.
            assert bomb_rc != 0, f"the storm stopped after {forks} forks with rc 0"
            assert (
                "fork" in captured.lower() or "temporarily unavailable" in captured.lower()
            ), captured
        else:
            assert int(forks) == 500
            assert bomb_rc == 0

        # Nothing the storm forked outlived the run. The sandbox unshares the
        # pid namespace, so the pids the storm recorded are namespace-local
        # numbers that mean nothing on this host; the namespace itself is what
        # the host can still see, and the kernel takes it down with the run's
        # last process. If the run did share this test's pid namespace, then
        # the recorded pids are real host pids and are polled one by one.
        namespace = nsfile.read_text(encoding="utf-8").strip()
        assert namespace, "the run did not report the pid namespace it ran in"
        if namespace == _pid_namespace(os.getpid()):
            survivors = _wait_for_no_survivors(
                lambda: [pid for pid in started if os.path.exists(f"/proc/{pid}")]
            )
        else:
            survivors = _wait_for_no_survivors(lambda: _in_namespace(namespace))
        assert survivors == []

    def test_long_sleep_is_killed_by_the_wall_timeout(self, service, broker):
        started = time.monotonic()
        result = self._run(service, broker, "sleep 120", wall_timeout_seconds=2)
        elapsed = time.monotonic() - started
        assert result.ok is False
        assert result.exit_code is None
        assert "timeout" in result.note.lower()
        assert elapsed < 20

    def test_inherited_environment_variables_are_dropped_under_bwrap(
        self, service, broker, monkeypatch
    ):
        monkeypatch.setenv("RLR_TOOLBROKER_CANARY", "leaked-secret")
        result = self._run(
            service, broker, "printenv RLR_TOOLBROKER_CANARY || echo absent; printenv HOME"
        )
        assert "leaked-secret" not in result.stdout
        assert "absent" in result.stdout
        assert "/home" in result.stdout

    def test_file_size_rlimit_caps_writes_under_bwrap(self, service, broker):
        target = broker.workspace / "big.bin"
        result = self._run(
            service,
            broker,
            f"head -c {8 * 1024 * 1024} /dev/zero > {target}",
            file_size_bytes=1024 * 1024,
        )
        assert result.ok is False
        assert target.stat().st_size <= 1024 * 1024

    def test_stdout_is_capped_under_bwrap(self, service, broker):
        result = self._run(
            service,
            broker,
            "head -c 1000000 /dev/zero | tr '\\0' X",
            max_output_bytes=4096,
        )
        assert len(result.stdout) <= 4096
        assert result.applied["stdout_truncated"] is True

"""The per-run pids cgroup: build it, fail closed, report it, remove it.

``RLIMIT_NPROC`` cannot be the hard per-run PID cap the sandbox needs: it counts
every process the *user* owns, so on a busy host it fires for unrelated reasons
and, once the count is already past the limit, it stops nothing. The pids
controller is enforced by the kernel per cgroup, which is what the broker now
delegates to: one child directory per run, ``tasks_max`` written to
``pids.max`` before the spawn, the child moved in from ``preexec_fn``, and every
pid left behind killed and the child removed afterwards.

Two files carry the security argument of the whole feature, and both are
driven hostile below. ``cgroup.procs`` is written by the run's own children,
so the drain must not take its contents as instructions -- not the broker's own
pid, not ``0`` or ``-1`` (which ``os.kill`` reads as a process group and as
every process on the host), not a value past the kernel's pid ceiling that
raises ``OverflowError`` instead of failing with ESRCH. And the *startup*
preflight cannot be fooled by a directory that merely holds files named
``pids.max`` and ``cgroup.procs``: the mount the base sits on is read out of
``/proc/self/mountinfo`` and has to be the pids controller itself.

The unit tests below drive the broker against a ``tmp_path`` base, and a plain
directory is no longer close enough to stand in for the pids controller: the
broker writes ``pids.max`` and moves the pid in through ``cgroup.procs``
without ever creating them, because an ``O_CREAT`` open would make ordinary
files under those names, report ``tasks_max`` back, and cap nothing at all. So
the ``cgroupfs`` fixture supplies the two behaviours a delegated controller has
and a directory does not -- the control files appear when the child is made, and
``rmdir`` takes the child away even though they are still listed inside it -- and
the suite still needs neither root nor a live cgroup. One test asks for no such
fixture: a base that is only a directory has to be refused, and that refusal is
the point. What even the fixture cannot show -- the kernel refusing the fork
past the cap -- is the single integration test, which runs only when
``RLR_TEST_PIDS_CGROUP`` names a real, writable cgroup base.
"""

from __future__ import annotations

import importlib.util
import json
import os
import resource
import signal
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from reflective_loop_runtime.kernel import toolbroker
from reflective_loop_runtime.kernel.audit import AuditLog
from reflective_loop_runtime.kernel.permit import PermitIssuer, PermitService
from reflective_loop_runtime.kernel.policy import MissionPolicyStore, sign_envelope
from reflective_loop_runtime.kernel.toolbroker import (
    ResourceLimits,
    ToolBroker,
    ToolExecutionError,
    broker_manifest,
)

DEV_SECRET = "dev-only-shared-secret"
ISSUER_KEY = b"governor-owned-hmac-key-0123456789"

# A stand-in for bubblewrap: it isolates nothing, it only mimics the option
# parsing and execs the command. That is the honest harness here -- what is
# under test is the cap, and a real namespace would only make it harder to tell
# which of the two stopped the forks.
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


def _write_fake_bwrap(path: Path) -> Path:
    path.write_text(_FAKE_BWRAP, encoding="utf-8")
    path.chmod(0o755)
    return path


def _permissive_nproc() -> int:
    """The host's NPROC hard limit.

    ``RLIMIT_NPROC`` counts every process the *user* owns, so the broker's small
    production default can starve even a trivial command on a busy host -- and
    then the test would be measuring the rlimit instead of the cgroup. The two
    limits are independent, so the tests hold the rlimit at the host maximum and
    leave the cap as the only thing under test.
    """
    _, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    if hard == resource.RLIM_INFINITY:
        return 4096
    return int(hard)


def _service(tmp_path: Path) -> PermitService:
    bundle = {
        "mission_id": "mission-7",
        "objective": "cap every sandboxed run",
        "invariants": ["never_shell_out_unsandboxed"],
        "allowed_domains": [],
        "capabilities": [],
        "budgets": {},
        "standing_mandates": [],
        "policy_version": "1.0.0",
    }
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(sign_envelope(bundle, secret=DEV_SECRET)), encoding="utf-8")
    policy = MissionPolicyStore(secret=DEV_SECRET, allow_dev_hmac=True).load(path)
    return PermitService(PermitIssuer(ISSUER_KEY), policy, manifest=broker_manifest())


@pytest.fixture
def service(tmp_path):
    return _service(tmp_path)


@pytest.fixture
def fake_bwrap(tmp_path):
    return _write_fake_bwrap(tmp_path / "fake-bwrap")


@pytest.fixture
def cgroupfs(tmp_path, monkeypatch):
    """Make this test's ``tmp_path`` answer the way a delegated pids cgroup does.

    Two behaviours, both the kernel's and neither a plain directory's. On the
    broker's ``mkdir`` of a run's child the controller populates it with its
    control files, which is exactly what the broker now requires before it will
    write to them. And ``rmdir`` of a child succeeds even though ``pids.max``
    and ``cgroup.procs`` are still listed inside it: a plain directory refuses
    until they are gone, and the production code must never unlink the
    controller's own files to make that happen. So this fixture creates the
    control files when the child is made and drops them only as the ``rmdir``
    reaches them, scoped to this ``tmp_path`` so nothing else on the filesystem
    is touched. It returns the path of a base, which the test creates where it
    needs it.
    """
    control_files = ("pids.max", "cgroup.procs", "pids.events")
    kernel_mkdir = toolbroker._mkdir_pids_child

    def mkdir_pids_child(child: Path) -> None:
        kernel_mkdir(child)
        for name in control_files:
            (child / name).write_text("", encoding="utf-8")

    plain_rmdir = Path.rmdir

    def rmdir(self: Path) -> None:
        if tmp_path == self.parent or tmp_path in self.parents:
            for entry in self.iterdir():
                if entry.is_file():
                    entry.unlink()
        plain_rmdir(self)

    monkeypatch.setattr(toolbroker, "_mkdir_pids_child", mkdir_pids_child)
    monkeypatch.setattr(Path, "rmdir", rmdir)
    return tmp_path / "pids"


def _broker(
    service: PermitService,
    tmp_path: Path,
    fake_bwrap: Path,
    *,
    pids_cgroup: Path,
    tasks_max: int = 64,
    wall_timeout_seconds: float = 30.0,
) -> ToolBroker:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return ToolBroker(
        service,
        audit=AuditLog(":memory:"),
        workspace=workspace,
        manifest=broker_manifest(),
        bwrap_path=str(fake_bwrap),
        limits=ResourceLimits(
            nproc=_permissive_nproc(),
            tasks_max=tasks_max,
            wall_timeout_seconds=wall_timeout_seconds,
        ),
        pids_cgroup=pids_cgroup,
    )


def _run(broker: ToolBroker, service: PermitService, command: str):
    permit = service.issue(
        tool="shell", op="run", args={"command": command}, goal_id="goal-1"
    )
    return broker.invoke(permit, "shell", "run", {"command": command})


def _last_audit(broker: ToolBroker) -> dict:
    events = broker.audit.events()
    assert events, "a refused call must leave an audit row"
    return dict(events[-1].governor_result)


def _token_field(stdout: str, token: str, name: str) -> str | None:
    """What the run reported under ``name``, or ``None`` if it reported nothing.

    Every line a stormed shell prints is tagged with the run's own token, so
    one run's output can never be read as another's -- and "the line is not
    there" stays a possible answer instead of being guessed at.
    """
    prefix = f"{token} {name}="
    for line in stdout.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return None


class TestPidsChildLifecycle:
    """The child exists for the run, carries the cap, and holds nothing else."""

    def test_child_is_created_and_capped_before_the_command_runs(
        self, cgroupfs, tmp_path, service, fake_bwrap
    ):
        # The base sits inside the workspace so the command can read it: that is
        # how the test observes the child while the sandboxed command is running.
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        base = workspace / "pids"
        base.mkdir()
        broker = ToolBroker(
            service,
            audit=AuditLog(":memory:"),
            workspace=workspace,
            manifest=broker_manifest(),
            bwrap_path=str(fake_bwrap),
            limits=ResourceLimits(nproc=_permissive_nproc(), tasks_max=17),
            pids_cgroup=base,
        )

        # pids.max must already hold the cap by the time the command starts, and
        # cgroup.procs must already hold the value the broker moved the pid in
        # with. The name of the child is asked for separately -- cat reports the
        # control files, not the directory they were found in -- and the echo
        # ends the last of those lines, because a cgroup control file carries no
        # trailing newline for the next line to start on.
        result = _run(
            broker,
            service,
            f"cat {base}/run-*/pids.max {base}/run-*/cgroup.procs; echo; ls {base}",
        )

        assert result.ok is True, result.stderr
        cap, moved, listing = result.stdout.split()
        assert cap == "17"
        assert moved == "0"
        assert listing.startswith("run-")
        # The directory the command saw is the one the result reports.
        assert result.applied["pids_cgroup"]["cgroup"] == listing

    def test_result_reports_the_cap_the_run_was_given(
        self, cgroupfs, tmp_path, service, fake_bwrap
    ):
        base = cgroupfs
        base.mkdir()
        broker = _broker(service, tmp_path, fake_bwrap, pids_cgroup=base, tasks_max=17)

        result = _run(broker, service, "true")

        report = result.applied["pids_cgroup"]
        assert report["tasks_max"] == 17
        assert report["cgroup"].startswith("run-")
        # A plain directory has no kernel counter, so the reported number is the
        # honest "nothing hit the cap"; the integration test is where a non-zero
        # value is proved.
        assert report["pids_max_events"] == 0
        assert result.as_dict()["applied"]["pids_cgroup"]["tasks_max"] == 17

    def test_events_counter_is_read_from_the_control_file(self, tmp_path, service, fake_bwrap):
        base = tmp_path / "pids"
        base.mkdir()
        broker = _broker(service, tmp_path, fake_bwrap, pids_cgroup=base, tasks_max=17)
        child = base / "run-handmade"
        child.mkdir()
        (child / "pids.events").write_text("low 0\nhigh 0\nmax 7\n", encoding="utf-8")
        (child / "cgroup.procs").write_text("", encoding="utf-8")

        report = broker._reap_pids_child(child)

        assert report["tasks_max"] == 17
        assert report["pids_max_events"] == 7
        assert report["cgroup"] == "run-handmade"
        assert report["drain"]["complete"] is False
        assert report["drain"]["remove_error"] == "OSError"
        # A plain directory is not a cgroup: rmdir cannot remove one that still
        # holds the stand-in control files, which is why the child going away is
        # asserted on a real cgroup in the integration test below.

    def test_empty_child_is_removed(self, tmp_path, service, fake_bwrap):
        base = tmp_path / "pids"
        base.mkdir()
        broker = _broker(service, tmp_path, fake_bwrap, pids_cgroup=base)
        child = base / "run-empty"
        child.mkdir()

        report = broker._reap_pids_child(child)

        assert report["cgroup"] == "run-empty"
        assert report["pids_max_events"] == 0
        assert not child.exists()

    def test_each_run_gets_its_own_child(self, service, fake_bwrap, cgroupfs):
        base = cgroupfs
        base.mkdir()
        broker = _broker(service, base.parent, fake_bwrap, pids_cgroup=base)

        first = _run(broker, service, "true")
        second = _run(broker, service, "true")

        # Two runs, two children: a run never shares -- and so never inherits
        # the pids or the leftovers of -- the cgroup of the run before it.
        assert first.applied["pids_cgroup"]["cgroup"] != second.applied["pids_cgroup"]["cgroup"]
        # And each child is gone once its run is over, so the base keeps
        # nothing. That removal is an rmdir that succeeds while the
        # controller's files are still listed inside the child, which is what
        # cgroupfs does and what the fixture models; the production code
        # deletes nothing there, and must not.
        assert list(base.iterdir()) == []
        assert not (base / first.applied["pids_cgroup"]["cgroup"]).exists()
        assert not (base / second.applied["pids_cgroup"]["cgroup"]).exists()


class TestPidsCgroupFailsClosed:
    """A configured cap that cannot be enforced refuses the call."""

    def test_missing_base_directory_is_refused(self, tmp_path, service, fake_bwrap):
        base = tmp_path / "not-created"
        broker = _broker(service, tmp_path, fake_bwrap, pids_cgroup=base)

        with pytest.raises(ToolExecutionError, match="pids cgroup"):
            _run(broker, service, "true")

        row = _last_audit(broker)
        assert row["outcome"] == "sandbox-unavailable"
        assert row["reason"].startswith("pids-cgroup:")
        assert row["executed"] is False
        assert not base.exists()

    @pytest.mark.skipif(
        not hasattr(os, "geteuid") or os.geteuid() == 0,
        reason="root ignores directory permissions, so the refusal cannot be reached",
    )
    def test_unwritable_base_directory_is_refused(self, tmp_path, service, fake_bwrap):
        base = tmp_path / "read-only"
        base.mkdir()
        base.chmod(0o555)
        try:
            broker = _broker(service, tmp_path, fake_bwrap, pids_cgroup=base)
            with pytest.raises(ToolExecutionError, match="pids cgroup"):
                _run(broker, service, "true")
        finally:
            base.chmod(0o755)

        row = _last_audit(broker)
        assert row["outcome"] == "sandbox-unavailable"
        assert row["reason"].startswith("pids-cgroup:")
        assert row["executed"] is False
        assert list(base.iterdir()) == []

    def test_a_plain_directory_is_refused(self, tmp_path, service, fake_bwrap):
        """A base that is only a directory must not report a cap it cannot set.

        Nothing here stands in for cgroupfs, which is the point: with an
        ``O_CREAT`` open the broker would make plain files named ``pids.max``
        and ``cgroup.procs``, hand back ``tasks_max=64`` in every result, and
        the kernel would cap nothing. The failure has to be loud and the base
        has to keep nothing.
        """
        base = tmp_path / "not-a-cgroup"
        base.mkdir()
        broker = _broker(service, tmp_path, fake_bwrap, pids_cgroup=base)

        with pytest.raises(ToolExecutionError, match="pids cgroup"):
            _run(broker, service, "true")

        row = _last_audit(broker)
        assert row["outcome"] == "sandbox-unavailable"
        assert row["reason"] == "pids-cgroup:not-a-cgroup"
        assert row["executed"] is False
        # The refused child was taken back down, so nothing named after a
        # control file was left where the next run would mistake it for one.
        assert list(base.iterdir()) == []

    def test_an_unconfigured_broker_still_runs(self, tmp_path, service, fake_bwrap):
        """The cap is opt-in: with no base configured nothing changes."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        broker = ToolBroker(
            service,
            audit=AuditLog(":memory:"),
            workspace=workspace,
            manifest=broker_manifest(),
            bwrap_path=str(fake_bwrap),
            limits=ResourceLimits(nproc=_permissive_nproc()),
        )

        result = _run(broker, service, "true")

        assert result.ok is True
        assert "pids_cgroup" not in result.applied


_TEST_PIDS_CGROUP = (os.environ.get("RLR_TEST_PIDS_CGROUP") or "").strip()

requires_pids_cgroup = pytest.mark.skipif(
    not _TEST_PIDS_CGROUP,
    reason="set RLR_TEST_PIDS_CGROUP to a writable pids-controller base to run this",
)


@requires_pids_cgroup
def test_a_fork_storm_is_cut_short_by_the_pids_cap(tmp_path, service, fake_bwrap):
    """A 200-fork storm under ``tasks_max=16``, cut short by the kernel.

    Only a real cgroup can show the property the unit tests above cannot: that
    the fork past the cap failed because of the controller, on a host of any
    load, and that nothing the storm started outlived the run.

    The storm runs in a subshell of its own, because a refused fork is fatal
    to ``/bin/sh``: dash exits the whole shell with EAGAIN and prints nothing
    after it. Put in the main shell, the storm would take the reporting lines
    down with it and the run could not tell "the cap stopped this" from "the
    shell died first". Inside the subshell, the outer shell always survives to
    say so -- which is the whole point of the test.
    """
    base = Path(_TEST_PIDS_CGROUP)
    assert base.is_dir(), "RLR_TEST_PIDS_CGROUP is not a directory"
    assert os.access(base, os.W_OK | os.X_OK), "RLR_TEST_PIDS_CGROUP is not writable"
    pidfile = tmp_path / "started.pids"
    errfile = tmp_path / "fork.err"
    # Both files exist before the storm runs: a storm refused on its very first
    # fork leaves nothing behind for the test to read.
    pidfile.write_text("", encoding="utf-8")
    errfile.write_text("", encoding="utf-8")
    token = uuid.uuid4().hex
    # Each iteration records the pid the fork produced, so what actually started
    # is measured rather than inferred from the loop counter.
    command = (
        f"( i=0; while [ $i -lt 200 ]; do sleep 5 & echo $! >> {pidfile}; "
        f"i=$((i+1)); done; echo '{token} forks='$i ) 2> {errfile}; "
        f"bomb_rc=$?; echo '{token} bomb-rc='$bomb_rc; echo '{token} parent-ok'"
    )
    broker = _broker(
        service, tmp_path, fake_bwrap,
        pids_cgroup=base, tasks_max=16, wall_timeout_seconds=45.0,
    )

    result = _run(broker, service, command)

    report = result.applied["pids_cgroup"]
    assert report["tasks_max"] == 16
    assert result.exit_code is not None, result.note
    assert "timeout" not in result.note.lower(), result.note
    # The reporting shell outlived the storm: the cap stopped the storm alone.
    assert f"{token} parent-ok" in result.stdout
    captured = result.stdout + result.stderr + errfile.read_text(encoding="utf-8")
    forks = _token_field(result.stdout, token, "forks")
    bomb_rc = int(_token_field(result.stdout, token, "bomb-rc"))
    # The controller refused the fork past the cap, and counted it.
    assert report["pids_max_events"] >= 1
    # The storm was cut short -- either before it could report, or short of
    # 200 -- and the shell said why when it gave up.
    assert forks is None or int(forks) < 200, result.stdout
    assert bomb_rc != 0, f"the storm stopped after {forks} forks with rc 0"
    assert (
        "fork" in captured.lower() or "temporarily unavailable" in captured.lower()
    ), captured
    started = {int(item) for item in pidfile.read_text(encoding="utf-8").split()}
    # The cap counts the reporting shell and the storm with everything they
    # started, so it cannot admit more than tasks_max processes in all.
    assert 0 < len(started) <= 16
    # Nothing the storm started outlived the run, and the child went with it.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and any(
        os.path.exists(f"/proc/{pid}") for pid in started
    ):
        time.sleep(0.05)
    assert not [pid for pid in started if os.path.exists(f"/proc/{pid}")]
    assert not (base / report["cgroup"]).exists()
    assert not list(base.glob("run-*"))


def _agent_runtime(name: str = "pids_agent_runtime"):
    """``deploy/agent_runtime.py`` loaded as a module, as startup does."""
    path = Path(__file__).resolve().parents[2] / "deploy" / "agent_runtime.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mountinfo(mount_point: Path, fstype: str, super_options: str) -> str:
    """One ``/proc/self/mountinfo`` record, in the kernel's own field order.

    ``id parent maj:min root mount-point options - fstype source super-options``.
    The cgroup v1 layout puts the controllers in the super options after the
    source, which is exactly where :func:`_mount_enforces_pids` looks.
    """
    return (
        f"25 22 0:24 / {mount_point} rw,nosuid,nodev,noexec,relatime shared:7 "
        f"- {fstype} cgroup {super_options}\n"
    )


# Well under the production drain of 5s, and well over the cost of the reads
# and the rmdir the drain does first: exceeding it means the loop polled out
# the whole wait for a file it had no right to clear.
PIDS_DRAIN_BUDGET_SECONDS = 1.0


class TestHostileCgroupProcsIsNotSignalled:
    """The reaper drains a cgroup; it does not obey the file naming it.

    ``cgroup.procs`` is the one input to the drain the run's own children can
    reach, so its contents deserve the same suspicion as anything else a
    sandboxed command produces. ``os.kill`` is replaced by a recorder, so the
    tests observe what the reaper *tried* to signal without anything actually
    being signalled -- and without the tests having to believe their own pid
    is unreachable.
    """

    @staticmethod
    def _child(tmp_path: Path, body: str, name: str) -> Path:
        child = tmp_path / name
        child.mkdir()
        # No trailing newline, exactly as the kernel writes the file.
        (child / "cgroup.procs").write_text(body, encoding="utf-8")
        return child

    @staticmethod
    def _recorder(
        monkeypatch, *, raises: BaseException | None = None
    ) -> list[tuple[int, int]]:
        calls: list[tuple[int, int]] = []

        def fake_kill(pid, sig):
            calls.append((pid, sig))
            if raises is not None:
                raise raises

        monkeypatch.setattr(toolbroker.os, "kill", fake_kill)
        # The drain polls the file the recorder cannot change, so the wait is
        # collapsed to a single pass: the assertions are about who was signalled.
        monkeypatch.setattr(toolbroker, "PIDS_CGROUP_DRAIN_SECONDS", 0.0)
        return calls

    def test_a_hostile_procs_file_cannot_steer_the_reaper(
        self, tmp_path, service, fake_bwrap, monkeypatch
    ):
        """Own pid, group leader, 0, -1, an overflow and garbage: none signalled."""
        broker = _broker(service, tmp_path, fake_bwrap, pids_cgroup=tmp_path / "pids")
        own = os.getpid()
        leader = os.getpgid(0)
        # The ceiling is itself a legal pid value, so the boundary is inclusive
        # and this one straggler -- far above anything a host hands out -- is
        # neither the broker's own pid nor its group leader.
        straggler = toolbroker.PID_MAX_LIMIT
        child = self._child(
            tmp_path,
            "\n".join(
                [
                    str(own),
                    str(leader),
                    "0",
                    "-1",
                    "99999999999999999999",
                    str(toolbroker.PID_MAX_LIMIT + 1),
                    "not-a-pid",
                    str(straggler),
                ]
            ),
            "run-hostile",
        )
        calls = self._recorder(monkeypatch)

        report = broker._reap_pids_child(child)

        signalled = [pid for pid, _ in calls]
        assert own not in signalled, "the reaper signalled the process doing the reaping"
        assert leader not in signalled, "the reaper signalled its own group leader"
        assert 0 not in signalled and -1 not in signalled, "os.kill(0) hits a group"
        assert 99999999999999999999 not in signalled
        assert toolbroker.PID_MAX_LIMIT + 1 not in signalled
        # The filter is not vacuous: the one pid that really is in the cgroup
        # was still killed, with SIGKILL, and the run reported as it always did.
        assert calls == [(straggler, signal.SIGKILL)]
        assert report["cgroup"] == "run-hostile"
        assert report["pids_max_events"] == 0
        # Not asserted gone: a plain directory still holds the stand-in
        # cgroup.procs, and rmdir refuses while that file is inside it.

    def test_an_overflow_out_of_kill_cannot_fail_the_run(
        self, tmp_path, service, fake_bwrap, monkeypatch
    ):
        """``OverflowError`` is swallowed like ESRCH: the run already finished."""
        broker = _broker(service, tmp_path, fake_bwrap, pids_cgroup=tmp_path / "pids")
        child = self._child(tmp_path, "4242\n", "run-overflow")
        calls = self._recorder(monkeypatch, raises=OverflowError("pid too large for C int"))

        report = broker._reap_pids_child(child)

        assert calls == [(4242, signal.SIGKILL)]
        assert report["cgroup"] == "run-overflow"
        assert report["tasks_max"] == toolbroker.DEFAULT_TASKS_MAX
        # Not asserted gone: a plain directory still holds the stand-in
        # cgroup.procs, and rmdir refuses while that file is inside it.

    def test_a_file_naming_only_this_process_stops_the_drain_at_once(
        self, tmp_path, service, fake_bwrap, monkeypatch
    ):
        """Nothing signalable left: break, rather than poll out the whole wait."""
        broker = _broker(service, tmp_path, fake_bwrap, pids_cgroup=tmp_path / "pids")
        child = self._child(tmp_path, f"{os.getpid()}\n{os.getpgid(0)}\n", "run-self")
        calls: list[tuple[int, int]] = []
        monkeypatch.setattr(toolbroker.os, "kill", lambda pid, sig: calls.append((pid, sig)))

        started = time.monotonic()
        report = broker._reap_pids_child(child)
        elapsed = time.monotonic() - started

        assert calls == []
        assert elapsed < PIDS_DRAIN_BUDGET_SECONDS, (
            "the drain waited out a file it cannot clear"
        )
        assert report["cgroup"] == "run-self"
        # Not asserted gone: a plain directory still holds the stand-in
        # cgroup.procs, and rmdir refuses while that file is inside it.

    def test_the_filter_states_its_own_boundary(self):
        """The rules the drain relies on, checked directly rather than via os.kill."""
        protected = toolbroker._protected_pids()

        assert os.getpid() in protected
        assert os.getpgid(0) in protected
        assert all(pid > 0 for pid in protected)
        signalable = toolbroker._signalable_pids(
            [0, -1, -4242, 4242, 99999999999999999999,
             toolbroker.PID_MAX_LIMIT, toolbroker.PID_MAX_LIMIT + 1],
            frozenset({4242}),
        )
        # One protected, one past the ceiling: the rest are all dropped.
        assert signalable == [toolbroker.PID_MAX_LIMIT]


class TestStartupPreflightNeedsARealCgroupMount:
    """The control files are necessary; the mount under them is the evidence.

    A directory can hold any name it likes, so a base with hand-made files
    named ``pids.max`` and ``cgroup.procs`` passes an existence check while the
    kernel enforces nothing there -- and the runtime would start reporting a cap
    per run that stops no fork. The preflight therefore asks
    ``/proc/self/mountinfo`` which controller the base resolves through, and
    refuses anything that is not the pids one.
    """

    def _base(self, tmp_path: Path, *, control_files: bool = True) -> Path:
        base = tmp_path / "pids" / "rlr-sandbox"
        base.mkdir(parents=True)
        if control_files:
            for name in toolbroker.PIDS_CONTROL_FILES:
                (base / name).write_text("", encoding="utf-8")
        return base

    def test_a_plain_directory_of_fake_control_files_is_refused(self, tmp_path, monkeypatch):
        module = _agent_runtime()
        base = tmp_path / "fake-pids"
        base.mkdir()
        for name in toolbroker.PIDS_CONTROL_FILES:
            (base / name).write_text("", encoding="utf-8")
        monkeypatch.setenv("RLR_SANDBOX_PIDS_CGROUP", str(base))
        # The mountinfo the reader is handed covers the base with tmpfs, which
        # is what tmp_path is on any host that is not already a cgroup.
        monkeypatch.setattr(
            module,
            "_read_mountinfo",
            lambda: _mountinfo(tmp_path, "tmpfs", "rw,mode=755"),
        )

        with pytest.raises(RuntimeError, match="not under a pids cgroup mount"):
            module._pids_cgroup_env()

    def test_a_base_under_a_cgroup_pids_mount_is_accepted(self, tmp_path, monkeypatch):
        module = _agent_runtime()
        base = self._base(tmp_path)
        monkeypatch.setenv("RLR_SANDBOX_PIDS_CGROUP", str(base))
        # The cgroup record comes first and the tmpfs record second: the mount
        # that decides is the deepest one above the base, not the first listed.
        monkeypatch.setattr(
            module,
            "_read_mountinfo",
            lambda: _mountinfo(base.parent, "cgroup", "rw,pids")
            + _mountinfo(tmp_path, "tmpfs", "rw,mode=755"),
        )

        assert module._pids_cgroup_env() == base

    def test_a_cgroup2_mount_is_accepted(self, tmp_path, monkeypatch):
        """On the unified hierarchy the fstype alone is the controller."""
        module = _agent_runtime()
        base = self._base(tmp_path)
        monkeypatch.setenv("RLR_SANDBOX_PIDS_CGROUP", str(base))
        monkeypatch.setattr(
            module,
            "_read_mountinfo",
            lambda: _mountinfo(base.parent, "cgroup2", "rw,nsdelegate"),
        )

        assert module._pids_cgroup_env() == base

    def test_a_v1_mount_of_another_controller_is_refused(self, tmp_path, monkeypatch):
        """Same fstype, no pids controller: the files there enforce nothing."""
        module = _agent_runtime()
        base = self._base(tmp_path)
        monkeypatch.setenv("RLR_SANDBOX_PIDS_CGROUP", str(base))
        monkeypatch.setattr(
            module,
            "_read_mountinfo",
            lambda: _mountinfo(base.parent, "cgroup", "rw,cpu,memory"),
        )

        with pytest.raises(RuntimeError, match="not under a pids cgroup mount"):
            module._pids_cgroup_env()

    def test_the_control_file_checks_are_kept(self, tmp_path, monkeypatch):
        """A real pids mount with no control files in the base is still refused."""
        module = _agent_runtime()
        base = self._base(tmp_path, control_files=False)
        monkeypatch.setenv("RLR_SANDBOX_PIDS_CGROUP", str(base))
        monkeypatch.setattr(
            module,
            "_read_mountinfo",
            lambda: _mountinfo(base.parent, "cgroup", "rw,pids"),
        )

        with pytest.raises(RuntimeError, match="not a pids cgroup"):
            module._pids_cgroup_env()

    def test_an_unreadable_mount_table_refuses_rather_than_assumes(self, tmp_path, monkeypatch):
        """No table means no evidence, so there is no cap to claim."""
        module = _agent_runtime()
        base = self._base(tmp_path)
        monkeypatch.setenv("RLR_SANDBOX_PIDS_CGROUP", str(base))
        monkeypatch.setattr(module, "_read_mountinfo", lambda: "")

        with pytest.raises(RuntimeError, match="not under a pids cgroup mount"):
            module._pids_cgroup_env()

    def test_off_linux_the_mount_table_is_not_the_gate(self, tmp_path, monkeypatch):
        """The check is Linux-only; elsewhere the file checks still stand."""
        module = _agent_runtime()
        base = self._base(tmp_path)
        monkeypatch.setenv("RLR_SANDBOX_PIDS_CGROUP", str(base))
        monkeypatch.setattr(module, "sys", SimpleNamespace(platform="darwin"))
        # Would refuse on Linux, so the platform skip is what this asserts.
        monkeypatch.setattr(module, "_read_mountinfo", lambda: "")

        assert module._pids_cgroup_env() == base

    def test_an_unset_variable_fails_closed_unless_explicitly_exempt(
        self, monkeypatch, caplog
    ):
        module = _agent_runtime()
        monkeypatch.delenv("RLR_SANDBOX_PIDS_CGROUP", raising=False)
        monkeypatch.delenv("RLR_SANDBOX_PIDS_CGROUP_EXEMPT", raising=False)
        with pytest.raises(RuntimeError, match="mandatory"):
            module._pids_cgroup_env()
        monkeypatch.setenv("RLR_SANDBOX_PIDS_CGROUP_EXEMPT", "yes")
        with pytest.raises(RuntimeError, match="mandatory"):
            module._pids_cgroup_env()
        monkeypatch.setenv("RLR_SANDBOX_PIDS_CGROUP_EXEMPT", "1")
        assert module._pids_cgroup_env() is None
        assert "DISABLED" in caplog.text

    def test_the_deepest_mount_decides_and_garbage_is_ignored(self, tmp_path):
        module = _agent_runtime()
        table = (
            "this line is not a mountinfo record\n"
            "25 22 0:24 / rw\n"
            "25 22 0:24 /sys/fs/cgroup\n"
            + _mountinfo(Path("/sys/fs/cgroup"), "cgroup2", "rw")
            + _mountinfo(Path("/sys/fs/cgroup/rlr-sandbox"), "tmpfs", "rw,mode=755")
        )
        # The cgroup mount is the ancestor; the tmpfs on top of the base is the
        # filesystem the base actually lives on, so it is the one that decides.
        assert module._mount_enforces_pids(Path("/sys/fs/cgroup/rlr-sandbox"), table) is False
        assert module._mount_enforces_pids(Path("/sys/fs/cgroup/other"), table) is True
        assert module._mount_for(Path("/nowhere"), table) is None

    def test_the_reader_reads_the_kernel_table_on_this_host(self):
        module = _agent_runtime()

        if not sys.platform.startswith("linux"):
            pytest.skip("mountinfo is a Linux interface")

        assert module.PROC_MOUNTINFO == "/proc/self/mountinfo"
        # The real reader is allowed to be empty only where there is no /proc.
        if Path(module.PROC_MOUNTINFO).exists():
            assert isinstance(module._read_mountinfo(), str)


class TestG2ContainmentAccounting:
    """Negative cases must report failure instead of declaring a clean drain."""

    def test_unreadable_cgroup_procs_is_incomplete(
        self, tmp_path, service, fake_bwrap, monkeypatch
    ):
        broker = _broker(
            service, tmp_path, fake_bwrap, pids_cgroup=tmp_path / "pids"
        )
        child = tmp_path / "run-unreadable"
        child.mkdir()

        def unreadable(path):
            raise PermissionError("denied")

        monkeypatch.setattr(toolbroker, "_cgroup_procs", unreadable)
        report = broker._reap_pids_child(child)
        assert report["drain"]["complete"] is False
        assert report["drain"]["read_error"] == "PermissionError"

    def test_survivor_and_failed_rmdir_are_incomplete(
        self, tmp_path, service, fake_bwrap, monkeypatch
    ):
        broker = _broker(
            service, tmp_path, fake_bwrap, pids_cgroup=tmp_path / "pids"
        )
        child = tmp_path / "run-survivor"
        child.mkdir()
        (child / "cgroup.procs").write_text("4242\n", encoding="utf-8")
        monkeypatch.setattr(
            toolbroker.os, "kill",
            lambda pid, sig: (_ for _ in ()).throw(ProcessLookupError()),
        )
        monkeypatch.setattr(toolbroker, "PIDS_CGROUP_DRAIN_SECONDS", 0)
        report = broker._reap_pids_child(child)
        assert not report["drain"]["complete"]
        assert report["drain"]["remaining_pids"] == [4242]
        assert report["drain"]["remove_error"] == "OSError"

    def test_failed_drain_marks_tool_unsuccessful_and_audits(
        self, service, fake_bwrap, cgroupfs, monkeypatch
    ):
        base = cgroupfs
        base.mkdir()
        broker = _broker(service, base.parent, fake_bwrap, pids_cgroup=base)
        actual = broker._reap_pids_child

        def forced_incomplete(child):
            report = actual(child)
            report["drain"]["complete"] = False
            report["drain"]["remove_error"] = "PermissionError"
            return report

        monkeypatch.setattr(broker, "_reap_pids_child", forced_incomplete)
        result = _run(broker, service, "true")
        assert result.exit_code == 0
        assert result.ok is False
        assert "containment-incomplete" in result.note
        assert not result.applied["pids_cgroup"]["drain"]["complete"]
        assert _last_audit(broker)["outcome"] == "containment-incomplete"

    def test_cgroup_control_fd_closed_in_child_and_nproc_branches(
        self, service, fake_bwrap, tmp_path, monkeypatch
    ):
        broker = _broker(
            service, tmp_path, fake_bwrap, pids_cgroup=tmp_path / "pids"
        )
        calls = []
        monkeypatch.setattr(
            toolbroker.resource, "setrlimit",
            lambda key, value: calls.append((key, value)),
        )
        monkeypatch.setattr(
            toolbroker.os, "write",
            lambda fd, data: calls.append(("write", fd, data)),
        )
        monkeypatch.setattr(
            toolbroker.os, "close",
            lambda fd: calls.append(("close", fd)),
        )
        broker._preexec_fn(None)()
        nocgroup = next(
            value for key, value in calls
            if key == resource.RLIMIT_NPROC
        )
        assert nocgroup == (
            broker.limits.nproc, broker.limits.nproc
        )
        calls.clear()
        broker._preexec_fn(1234)()
        hard = resource.getrlimit(resource.RLIMIT_NPROC)[1]
        withcgroup = next(
            value for key, value in calls
            if key == resource.RLIMIT_NPROC
        )
        assert withcgroup == (hard, hard)
        assert ("write", 1234, b"0") in calls
        assert ("close", 1234) in calls
        assert calls.index(("write", 1234, b"0")) < calls.index(
            ("close", 1234)
        )


@requires_pids_cgroup
@pytest.mark.skipif(
    not Path("/usr/bin/bwrap").is_file(),
    reason="real bubblewrap not installed on this host",
)
def test_real_bwrap_cannot_see_writable_cgroup_control_fd(
    tmp_path, service
):
    """Real sandbox plus delegated controller, not a fake-bwrap FD test."""
    base = Path(_TEST_PIDS_CGROUP)
    broker = _broker(
        service, tmp_path, Path("/usr/bin/bwrap"), pids_cgroup=base
    )
    command = (
        'for f in /proc/self/fd/*; do '
        'case "$(readlink "$f" 2>/dev/null)" in '
        '*cgroup.procs*) echo CONTROL_FD_LEAK;; esac; '
        'done; echo FD_SCAN_OK'
    )
    result = _run(broker, service, command)
    assert result.ok, result.as_dict()
    assert "FD_SCAN_OK" in result.stdout
    assert "CONTROL_FD_LEAK" not in result.stdout
    assert result.applied["pids_cgroup"]["drain"]["complete"]

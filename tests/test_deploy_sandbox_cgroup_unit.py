"""The shipped ``rlr-sandbox-cgroup`` unit must build a base the broker can use.

The pids cgroup the broker caps every sandboxed run in is created once, at boot,
by this oneshot, so ``deploy/`` is committed as the source of truth for it and
its claims are guarded like any other boundary:

* the base directory and its own control files are handed to the runtime user --
  measured on the Jetson (cgroup v1), mkdir under that base and writing a child's
  ``pids.max`` then both work, because the kernel gives the creator ownership of
  the control files it populates. A recursive ``chown`` would be a claim about a
  subtree the kernel does not need owned, so it is not made.
* the user is one environment variable, not a name baked into the command.
* ``Before=`` is ordering only. The gate is the ``agent_runtime`` preflight, so
  the comment above it says so rather than implying systemd enforces anything.
"""

from __future__ import annotations

from pathlib import Path

_UNIT = Path(__file__).resolve().parent.parent / "deploy" / "rlr-sandbox-cgroup.service"

_BASE = "/sys/fs/cgroup/pids/rlr-sandbox"


def _lines() -> list[str]:
    return _UNIT.read_text(encoding="utf-8").splitlines()


def _exec_start() -> str:
    for line in _lines():
        if line.startswith("ExecStart="):
            return line
    raise AssertionError("rlr-sandbox-cgroup.service has no ExecStart")


def test_base_is_created_and_handed_to_the_runtime_user():
    command = _exec_start()
    assert f"mkdir -p {_BASE}" in command
    assert f"chown ${{RLR_SANDBOX_USER}} {_BASE}" in command
    # The base's own control files are chowned by name: they are what the
    # runtime preflight checks for and what the broker writes.
    assert f"{_BASE}/pids.max" in command
    assert f"{_BASE}/cgroup.procs" in command


def test_the_chown_claims_no_subtree():
    """A recursive chown promises more than the kernel requires.

    Ownership of a child cgroup's control files follows from creating it, so
    recursing would be a claim about runs that do not exist yet -- and one that
    would silently cover anything else that ever appeared under the base.
    """
    assert "chown -R" not in _exec_start()


def test_the_runtime_user_is_configurable():
    text = _UNIT.read_text(encoding="utf-8")
    assert "Environment=RLR_SANDBOX_USER=vitor" in text
    # The command reads that variable; it does not name a user of its own.
    assert "vitor:" not in _exec_start() and "vitor " not in _exec_start()


def test_ordering_is_not_offered_as_the_gate():
    lines = _lines()
    assert "Before=user@1000.service" in lines
    index = lines.index("Before=user@1000.service")
    comment = " ".join(
        line.strip().lstrip("#").strip()
        for line in lines[max(0, index - 5):index]
        if line.strip().startswith("#")
    )
    lowered = comment.lower()
    # The comment says what Before= is (ordering) and where the real gate is.
    assert "ordering" in lowered
    assert "preflight" in lowered

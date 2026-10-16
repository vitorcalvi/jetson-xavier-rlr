"""The shipped ``rlr-advisor`` unit must bound the advisor's host memory.

The 2026-10-06 Jetson soak showed ``llama-server``'s in-RAM prompt/KV cache
growing to ~4.9 GiB ``RssAnon`` (upstream ``--cache-ram`` default 8192 MiB) and
pushing ``MemAvailable`` under 500 MiB, which risks an OOM kill of the RLR loop.
``deploy/`` is committed as the source of truth for the soak install, so this
guards the bound: the prompt cache stays small and the service cgroup caps keep
any residual growth from touching ``rlr.service`` or ``rlr-simsoak.service``.
"""

from __future__ import annotations

from pathlib import Path

_UNIT = Path(__file__).resolve().parent.parent / "deploy" / "rlr-advisor.service"


def _text() -> str:
    return _UNIT.read_text(encoding="utf-8")


def _exec_start() -> str:
    for line in _text().splitlines():
        if line.startswith("ExecStart="):
            return line
    raise AssertionError("rlr-advisor.service has no ExecStart")


def test_advisor_prompt_cache_is_bounded():
    assert "--cache-ram 256" in _exec_start()


def test_advisor_context_and_slot_are_unchanged():
    exec_start = _exec_start()
    assert "-c 1024" in exec_start
    assert "-np 1" in exec_start


def test_advisor_cgroup_caps_and_restart_policy_are_set():
    text = _text()
    assert "MemoryAccounting=yes" in text
    assert "MemoryHigh=2200M" in text
    assert "MemoryMax=2600M" in text
    assert "Restart=on-failure" in text

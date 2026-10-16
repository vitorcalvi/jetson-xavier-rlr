"""Redteam harness tests: the attack battery, its statuses, and TOON output.

The battery runs against the real kernel APIs, so these tests double as an
integration check that the permit/tool/audit/kill/plan surfaces still refuse the
hostile inputs the harness asserts on. Bubblewrap-dependent rows are only
asserted as an explicit ``SKIPPED_NO_BWRAP`` when the sandbox is absent, so the
suite is green on hosts with and without user namespaces.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reflective_loop_runtime import redteam
from reflective_loop_runtime.kernel.toolbroker import ToolResult
from reflective_loop_runtime.redteam import (
    ATTACKS,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_SKIPPED_NO_BWRAP,
    Attack,
    _Ctx,
    _execute,
    run_attacks,
    toon_encode,
)

BWRAP_ATTACKS = {attack.name for attack in ATTACKS if attack.requires_bwrap}


def test_battery_has_eighteen_unique_attacks():
    names = [attack.name for attack in ATTACKS]
    assert len(ATTACKS) == 18
    assert len(set(names)) == 18
    assert BWRAP_ATTACKS, "at least one attack must exercise the sandbox"


def test_every_attack_passes_or_is_explicitly_skipped(tmp_path):
    results = run_attacks(workdir=tmp_path)
    assert len(results) == 18
    for result in results:
        assert result.status in (STATUS_PASS, STATUS_SKIPPED_NO_BWRAP), result
        assert result.detail  # every row explains itself
    if not redteam._bwrap_available():
        skipped = {r.name for r in results if r.status == STATUS_SKIPPED_NO_BWRAP}
        assert skipped == BWRAP_ATTACKS


def test_missing_kernel_surface_is_fail_not_skip(monkeypatch, tmp_path):
    monkeypatch.setattr(redteam, "missing_kernel_surfaces", lambda: ["kernel.permit.PermitService"])
    results = run_attacks(workdir=tmp_path)
    assert len(results) == 1
    assert results[0].status == STATUS_FAIL
    assert "kernel.permit.PermitService" in results[0].detail


def test_a_raising_attack_is_reported_as_fail(tmp_path):
    def explode(ctx: _Ctx) -> str:
        raise ImportError("surface is unavailable")

    result = _execute(Attack("missing_surface", "A1", explode), _Ctx(workdir=tmp_path))
    assert result.status == STATUS_FAIL
    assert "ImportError" in result.detail


def test_bwrap_attack_skips_when_sandbox_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(redteam, "_bwrap_available", lambda: False)
    attack = Attack("needs_bwrap", "A1", lambda ctx: "never runs", requires_bwrap=True)
    result = _execute(attack, _Ctx(workdir=tmp_path))
    assert result.status == STATUS_SKIPPED_NO_BWRAP


def test_main_exits_zero_and_prints_the_toon_table(tmp_path, capsys):
    code = redteam.main(["--workdir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "attacks[18]{name,risk_class,status,detail}:" in out
    assert "failed: 0" in out
    for attack in ATTACKS:
        assert attack.name in out


def test_main_never_prints_key_material(tmp_path, capsys):
    redteam.main(["--workdir", str(tmp_path)])
    out = capsys.readouterr().out
    assert redteam.DEV_SECRET not in out
    assert redteam.ISSUER_KEY.decode() not in out


def test_json_mode_reports_the_summary(tmp_path, capsys):
    code = redteam.main(["--workdir", str(tmp_path), "--json"])
    document = json.loads(capsys.readouterr().out)
    assert code == 0
    assert document["summary"]["failed"] == 0
    assert len(document["attacks"]) == 18
    assert document["redteam"]["exit_code"] == 0


def test_toon_encode_tabular_and_scalars():
    text = toon_encode(
        {"a": 1, "b": "x,y", "rows": [{"k": "v", "n": 2}, {"k": "w", "n": 3}]}
    )
    assert "a: 1" in text
    assert '"x,y"' in text
    assert "rows[2]{k,n}:" in text
    assert "  v,2" in text
    assert "  w,3" in text


def _simulate_sandbox(monkeypatch, *, host_write: bool) -> None:
    """Make the write-escape probe runnable and script where its write lands.

    ``host_write=False`` models the real broker: the probe path is shadowed by
    the sandbox's private ``/tmp`` tmpfs, so ``printf`` exits 0 but the host
    filesystem never changes. ``host_write=True`` models a broken sandbox (or a
    host path that is bind-mounted writable), where the write does reach the
    host. Only the host filesystem decides, which is what the attack must key on.
    """
    monkeypatch.setattr(redteam, "_bwrap_available", lambda: True)
    monkeypatch.setattr(redteam.ToolBroker, "bwrap_available", lambda self: True)

    def fake_invoke(self, permit, tool, op, args=None):
        command = str((args or {}).get("command", ""))
        if "rlr-sandbox-ok" in command:
            return ToolResult(
                tool=tool, op=op, risk_class="A1", executed=True, simulated=False,
                recorded=True, ok=True, stdout="rlr-sandbox-ok",
            )
        target = command.split(">", 1)[-1].strip()
        if host_write and target:
            Path(target).write_text("x", encoding="utf-8")
        return ToolResult(
            tool=tool, op=op, risk_class="A1", executed=True, simulated=False,
            recorded=True, ok=True,
        )

    monkeypatch.setattr(redteam.ToolBroker, "invoke", fake_invoke)


def test_write_escape_attack_fails_when_a_host_write_succeeds(monkeypatch, tmp_path):
    _simulate_sandbox(monkeypatch, host_write=True)
    ctx = _Ctx(workdir=tmp_path)
    with pytest.raises(redteam.AttackFailure):
        redteam._attack_sandbox_write_escape_blocked(ctx)
    assert (tmp_path / "escape-probe").exists()


def test_write_escape_attack_reports_fail_in_the_battery(monkeypatch, tmp_path):
    _simulate_sandbox(monkeypatch, host_write=True)
    attack = next(a for a in ATTACKS if a.name == "sandbox_write_escape_blocked")
    result = _execute(attack, _Ctx(workdir=tmp_path))
    assert result.status == STATUS_FAIL
    assert "outside" in result.detail


def test_write_escape_attack_passes_on_a_private_tmpfs_write_only(monkeypatch, tmp_path):
    # ``result.ok`` is True (the command exited 0) but no host file appeared, so
    # a private tmpfs write must not be mistaken for an escape.
    _simulate_sandbox(monkeypatch, host_write=False)
    detail = redteam._attack_sandbox_write_escape_blocked(_Ctx(workdir=tmp_path))
    assert not (tmp_path / "escape-probe").exists()
    assert "bind-mounted workspace" in detail


def test_lan_network_boundary_attack_passes_without_real_network(tmp_path):
    attack = next(a for a in ATTACKS if a.name == "lan_advisor_network_boundary")
    result = _execute(attack, _Ctx(workdir=tmp_path))
    assert result.status == STATUS_PASS
    assert "redirects disabled" in result.detail
    assert "redact key" in result.detail

# ruff: noqa: E402
"""Static and behavioral checks for non-invasive injection boundaries."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy"))
from bridge_hook_patch import integrate_runner


def source_fixture() -> str:
    return (
        """
state=controller.startup_reconcile()
""".replace("state=", "        state=")
        + """        gate=StrictSessionQuoteGate()
            state=controller.state_row(); mode=state['state']
                                     now=now,headers=api.market_headers())['SPY']
                closed=[bar for bar in snap.bars if bar.ts+60<=now]
                        controller.execute_entry(cycle_id,gate_value)
"""
    )


def test_hook_is_opt_in_and_no_broker_override():
    src = source_fixture()
    transformed = integrate_runner(src)
    assert "_toon_bridge.on_market_snapshot" in transformed
    assert "_toon_bridge.on_stage2_signal" in transformed
    assert transformed.count("controller.execute_entry(cycle_id,gate_value)") == 1
    assert "controller.cancel" not in transformed
    assert "controller._transition" not in transformed
    assert integrate_runner(transformed) == transformed


def test_fails_closed_on_unknown_stage2_revision():
    with pytest.raises(RuntimeError, match="refusing unsafe hook"):
        integrate_runner("def loop(): pass")

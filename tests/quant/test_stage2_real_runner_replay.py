"""R1: run *actual* legacy Stage2 loop for 100 halted ticks, side-by-side.

Its broker and clock are test doubles. No credentials, HTTP or order POST.
The real Stage2 code is imported; all control decisions and persistent outputs
must match bit-for-bit with a shadow observer attached or disabled.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

DEPLOY = Path(__file__).resolve().parents[2] / "deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))


class StopReplay(Exception):
    pass


class ControllerMock:
    def __init__(self):
        self.observations = []
        self.broker_calls = 0
        self._state = {
            "state": "HALT_ENTRIES",
            "transition_seq": 3,
            "completed_cycles": 0,
            "held_qty": "0",
            "paper_orders_submitted": 0,
        }

    def startup_reconcile(self):
        return dict(self._state)

    def state_row(self):
        self.observations.append(dict(self._state))
        return dict(self._state)

    def execute_entry(self, *args, **kwargs):
        self.broker_calls += 1
        raise AssertionError("broker entry must never occur in HALT_ENTRIES")


class APIStub:
    def __init__(self, *args, **kwargs):
        self.called = 0

    def account_active(self):
        return True

    def get_positions(self):
        self.called += 1
        raise AssertionError("no extra broker GET under HALT_ENTRIES")

    def get_orders(self, **kwargs):
        self.called += 1
        raise AssertionError("no extra broker GET under HALT_ENTRIES")


class ShadowRecorder:
    def __init__(self):
        self.ticks = []

    def on_controller_tick(self, **kwargs):
        self.ticks.append(kwargs)
        return True


def run_real_stage2_100_tick(tmp_path: Path, monkeypatch, enabled: bool):
    runner = pytest.importorskip("stage2_paper_runner")
    bridge_mod = importlib.import_module("stage2_toon_bridge")
    ledger = tmp_path / ("enabled" if enabled else "baseline")
    home = ledger.parent
    (home / "flags").mkdir(parents=True, exist_ok=True)
    (home / "flags/HALT_ENTRIES").write_text("HALTED\n")
    recorder = ShadowRecorder()
    controller = ControllerMock()
    api = APIStub()
    count = [0]

    def fake_sleep(_):
        count[0] += 1
        if count[0] >= 100:
            raise StopReplay()

    with monkeypatch.context() as m:
        m.setattr(runner, "HOME", home)
        m.setattr(runner, "LEDGER", ledger)
        m.setattr(runner, "load", lambda: {"ledger": str(ledger), "service": runner.UNIT})
        m.setattr(runner, "_load_trading_policy_or_raise", lambda _: None)
        m.setattr(runner, "_check_alpaca_credentials", lambda _: ("MOCK", "MOCK"))
        m.setattr(runner, "Stage2AlpacaPaperAdapter", lambda *a, **k: api)
        m.setattr(runner, "Stage2DurableController", lambda *a, **k: controller)
        m.setattr(runner.time, "time", lambda: 1_792_121_800.0)
        m.setattr(runner.time, "monotonic", lambda: 100.0)
        m.setattr(runner.time, "sleep", fake_sleep)
        m.setenv("RLR_TRADER_ASSET_CLASS", "us_equity")
        m.setenv("TOON_V2_SHADOW_ENABLED", "true" if enabled else "false")
        m.setenv("TOON_V2_OVERRIDE_STAGE2", "false")
        m.setattr(
            bridge_mod.Stage2ShadowInterceptor,
            "from_env",
            classmethod(
                lambda cls, **kwargs: (
                    recorder if enabled else SimpleNamespace(on_controller_tick=lambda **_: False)
                )
            ),
        )
        with pytest.raises(StopReplay):
            runner.loop()
    assert count[0] == 100
    assert (home / "flags/HALT_ENTRIES").exists()
    return {
        "states": controller.observations,
        "broker_calls": controller.broker_calls,
        "api_get_calls": api.called,
        "runtime_tick": (ledger / "runtime_tick.json").read_bytes(),
        "shadow_timestamps": [x["timestamp_ms"] for x in recorder.ticks],
    }


def test_r1_actual_stage2_main_100_tick_bit_identical(tmp_path, monkeypatch):
    base = run_real_stage2_100_tick(tmp_path / "baseline", monkeypatch, False)
    shadow = run_real_stage2_100_tick(tmp_path / "shadow", monkeypatch, True)
    # Shadow's private audit is excluded; every legacy controller state,
    # observation, broker-call count and runtime heartbeat must match exactly.
    for key in ("states", "broker_calls", "api_get_calls", "runtime_tick"):
        assert base[key] == shadow[key], key
    assert base["broker_calls"] == shadow["broker_calls"] == 0
    assert base["api_get_calls"] == shadow["api_get_calls"] == 0
    assert len(shadow["shadow_timestamps"]) == 100

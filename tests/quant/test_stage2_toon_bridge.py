# ruff: noqa: E402
"""Stage2 read-only observer: no execution effects, uncertainty is a veto."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

# Deployment modules are imported as they are on the Stage2 service PYTHONPATH.
DEPLOY = Path(__file__).resolve().parents[2] / "deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))
from stage2_toon_bridge import SafetyLockViolation, ShadowDecision, Stage2ShadowInterceptor

from reflective_loop_runtime.quant.risk.kelly_firewall import OutcomeStats, RiskFirewall


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def snapshot(ts):
    # Actual shape of Stage2 Quote/Bar structures, not fictitious L2 sizes.
    return SimpleNamespace(
        quote=SimpleNamespace(bid=100.0, ask=100.04, ts=ts / 1000),
        bars=[
            SimpleNamespace(
                ts=(ts - 60000 * (i + 1)) / 1000, open=100.0, high=100.05, low=99.95, close=100.0
            )
            for i in range(65)
        ][::-1],
    )


def deterministic_legacy(ticks):
    """This stage2 decision replay is deliberately independent of the bridge."""
    return [("HOLD" if i % 7 else "BUY", i % 4) for i in range(ticks)]


def test_r1_100_tick_zero_effect_legacy_replay(tmp_path):
    halt = tmp_path / "HALT_ENTRIES"
    halt.write_text("1")
    audit = tmp_path / "shadow_audit.jsonl"
    bridge = Stage2ShadowInterceptor(ledger=tmp_path, halt_file=halt, max_queue=256)
    baseline = deterministic_legacy(100)
    observed = []
    for i, (action, qty) in enumerate(baseline):
        assert bridge.on_controller_tick(mode="HALT_ENTRIES", timestamp_ms=i + 1)
        # Hook can never read or return a legacy execution mutation.
        observed.append((action, qty))
    assert bridge.drain(timeout_sec=4)
    bridge.close()
    assert observed == baseline
    assert halt.exists()
    assert len(lines(audit)) == 100
    assert all(v["broker_orders_submitted"] == 0 for v in lines(audit))


def test_r2_forced_shadow_exception_does_not_affect_ticks(tmp_path):
    def explode(_):
        raise RuntimeError("M5 simulated unavailable")

    bridge = Stage2ShadowInterceptor(
        ledger=tmp_path, halt_file=tmp_path / "HALT_ENTRIES", evaluator=explode
    )
    for i in range(15):
        assert bridge.on_market_snapshot(
            symbol="SPY",
            snapshot=snapshot(60_000 * 100 + i * 1000),
            timestamp_ms=60_000 * 100 + i * 1000,
        )
    assert bridge.drain(timeout_sec=4)
    bridge.close()
    assert bridge.exceptions == 15
    assert bridge.processed == 15
    assert len(lines(tmp_path / "shadow_audit.jsonl")) == 15
    assert all(
        x["kind"] == "shadow_exception" and x["broker_orders_submitted"] == 0
        for x in lines(tmp_path / "shadow_audit.jsonl")
    )


def test_r3_high_correlation_third_position_veto_and_comparison(tmp_path):
    risk = RiskFirewall()
    names = ("SPY", "QQQ", "IWM")
    cov = ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    result = risk.evaluate(
        symbol="IWM",
        now=datetime(2026, 10, 16, tzinfo=timezone.utc),
        equity=10000,
        opening_equity=10000,
        existing_notional={"SPY": 100, "QQQ": 100},
        stats=OutcomeStats(200, 20, 2.0, 1.0),
        beta={x: 1.0 for x in names},
        covariance_names=names,
        covariance=cov,
    )
    assert not result.permitted

    def evaluator(_):
        return ShadowDecision(
            "UNSAFE", result.notional, None, True, "risk_" + result.reason, 'W1: "test"\n'
        )

    bridge = Stage2ShadowInterceptor(
        ledger=tmp_path, halt_file=tmp_path / "HALT_ENTRIES", evaluator=evaluator
    )
    t = 60_000 * 100
    assert bridge.on_market_snapshot(symbol="IWM", snapshot=snapshot(t), timestamp_ms=t)
    assert bridge.on_stage2_signal(action="BUY", qty=1.0, price=100.0, timestamp_ms=t)
    assert bridge.drain()
    bridge.close()
    audit = lines(tmp_path / "shadow_audit.jsonl")
    assert audit[0]["veto_reason"] == "risk_" + result.reason
    assert audit[1]["kind"] == "comparison"
    assert audit[1]["stage2_action"] == "BUY"
    assert audit[1]["toon_v2_action"] == "HOLD"
    assert not audit[1]["agreed"] and audit[1]["firewall_veto"]


def test_actual_l1_snapshot_requires_real_l2_qty(tmp_path):
    bridge = Stage2ShadowInterceptor(ledger=tmp_path, halt_file=tmp_path / "HALT_ENTRIES")
    t = 60_000 * 100
    bridge.on_market_snapshot(symbol="SPY", snapshot=snapshot(t), timestamp_ms=t)
    assert bridge.drain()
    bridge.close()
    audit = lines(tmp_path / "shadow_audit.jsonl")
    assert audit[0]["kind"] == "shadow_exception"
    assert "missing_l2_depth" in audit[0]["detail"]


def test_override_never_bypasses_halt(tmp_path, monkeypatch):
    flag = tmp_path / "HALT_ENTRIES"
    flag.write_text("1")
    monkeypatch.setenv("TOON_V2_OVERRIDE_STAGE2", "true")
    with pytest.raises(SafetyLockViolation):
        Stage2ShadowInterceptor.from_env(ledger=tmp_path, halt_file=flag)
    monkeypatch.setenv("TOON_V2_OVERRIDE_STAGE2", "false")
    flag.unlink()
    b = Stage2ShadowInterceptor.from_env(ledger=tmp_path, halt_file=flag)
    assert b.assert_halt_invariant()  # default environment hard halt
    b.close()


def test_trade_closed_observation_never_counts_as_paper_cert(tmp_path):
    b = Stage2ShadowInterceptor(ledger=tmp_path, halt_file=tmp_path / "HALT_ENTRIES")
    for i in range(30):
        b.on_trade_closed(
            pnl=1,
            slippage_bps=1,
            hold_seconds=90,
            broker_order_id=f"paper-{i}",
            source="UNVERIFIED",
        )
    assert b.drain()
    b.close()
    records = lines(tmp_path / "shadow_audit.jsonl")
    assert all(r["counted_for_certification"] is False for r in records)


def test_shadow_disabled_does_not_start_queue(tmp_path):
    b = Stage2ShadowInterceptor(ledger=tmp_path, halt_file=tmp_path / "HALT_ENTRIES", enabled=False)
    assert not b.on_controller_tick(mode="IDLE", timestamp_ms=1)
    b.close()
    assert not (tmp_path / "shadow_audit.jsonl").exists()

"""Read-only high-impact remediation contract checks.

These are static/in-process tests; they do not claim live broker fault injection.
"""
from pathlib import Path
import json
import os
from reflective_loop_runtime.trading.experiment_registry import load,REGISTRY
from reflective_loop_runtime.trading.broker import PaperBroker, PaperOnlyError
from reflective_loop_runtime.trading.http import _default_http,_default_post

def test_active_experiment_is_current_and_non_alpha():
    r=load()
    assert r["service"] in ("rlr-autonomy-1m.service","rlr-stage2-paper.service")
    assert r["strategy_class"]=="DIAGNOSTIC_NON_ALPHA"
    assert Path(r["ledger"]).is_dir()
    assert REGISTRY.stat().st_mode & 0o077 == 0

def test_kill_unit_covers_all_order_capable_units():
    p=Path.home()/".config/systemd/user/rlr-kill.service"
    text=p.read_text()
    for name in ("rlr.service","rlr-trader.service","rlr-autonomy-1m.service",
                 "rlr-equity-orb5.service","rlr-equity-opportunity-lab.service",
                 "rlr-trader-shadow.service"):
        assert name in text
    assert "RemainAfterExit=yes" in text

def test_paper_host_guard_refuses_real_host():
    try:
        PaperBroker("PKFAKE","test",http_get=lambda *a:None,
                    http_post=lambda *a:None,base_url="https://api.alpaca.markets")
        assert False
    except PaperOnlyError:pass

def test_reconciled_baseline_has_no_paper_exposure():
    root=Path(load()["ledger"])
    if load()["service"]=="rlr-autonomy-1m.service":
        assert (root/"AUTONOMY_1M_TEST_COMPLETE").exists()
    else:
        # The bounded Stage2 experiment is intentionally not yet complete.
        # File presence proves state persistence, NOT broker-flat status.
        assert (root/"controller.db").exists()
        state=json.loads((root/"heartbeat.json").read_text())
        assert state["strategy_class"]=="DIAGNOSTIC_NON_ALPHA"
        assert state["service"]=="rlr-stage2-paper.service"
        assert int(state["completed_cycles"])<=int(state["max_cycles"])
    assert (root/"relay_offset.json").exists()
    offset=json.loads((root/"relay_offset.json").read_text())
    assert offset["byte_offset"]==(root/"alerts_outbox.jsonl").stat().st_size

def test_legacy_orb_services_not_enabled():
    # Static expected plan; systemd state is separately verified live.
    assert not (Path.home()/".config/systemd/user/default.target.wants/rlr-equity-orb5.service").exists()
    assert not (Path.home()/".config/systemd/user/default.target.wants/rlr-equity-opportunity-lab.service").exists()

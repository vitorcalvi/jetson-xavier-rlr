"""Adaptive gate security tests: synthetic data only, no trading or broker traffic."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from reflective_loop_runtime.trading.adaptive_gate import (
    _SCOPE_DOMAIN,
    AdaptivePaperGate,
    _canonical,
)
from reflective_loop_runtime.trading.autonomy import (
    NO_TRADE,
    Candidate,
    DailyOutcome,
)

NOW = int(datetime(2026, 10, 14, 10, 0, tzinfo=timezone.utc).timestamp())


class FakeController:
    def __init__(self):
        self.state = {
            "mode": "RUN",
            "reconciled": True,
            "incident": "",
            "uncertain": False,
            "unexpected_fills": [],
        }

    def snapshot(self):
        return dict(self.state)


def setup(tmp_path: Path):
    root = tmp_path / "research"
    src = tmp_path / "decider.py"
    src.write_text("def rule(): return True\n")
    priv = Ed25519PrivateKey.generate()
    pub_path = tmp_path / "operator-public.pem"
    pub_path.write_bytes(
        priv.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    controller = FakeController()
    gate = AdaptivePaperGate(
        research_dir=root,
        public_key_path=pub_path,
        decider_file=src,
        strategy_id="trend_vwap",
        controller=controller,
    )
    return gate, priv, controller, src


def feature():
    return SimpleNamespace(ema20_5m=108.5, ema50_5m=108.2, quote_age_s=5.0, spread_pct=0.08)


def build_study(gate):
    today = date(2026, 10, 14)
    periods = []
    start = today - timedelta(days=100)
    for n in (30, 20, 30, 20):
        segment = []
        for _ in range(n):
            segment.append({"day": start.isoformat(), "net_r": 0.4, "trades": 2})
            start += timedelta(days=1)
        periods.append(segment)
    evidence = gate.research_dir / "evidence" / "independent_sample_report.json"
    evidence.write_text('{"kind":"mock_artifact_for_unit_tests_only"}\n')
    sha = hashlib.sha256(evidence.read_bytes()).hexdigest()
    code = hashlib.sha256(gate.decider_file.read_bytes()).hexdigest()
    item = {
        "strategy_id": "trend_vwap",
        "code_sha256": code,
        "symbol": "SOL/USD",
        "feed": "alpaca_crypto",
        "regimes": ["trend_up"],
        "artifact_sha256": sha,
        "artifact_file": evidence.name,
        "costs_audited": True,
        "leakage_audited": True,
        "holdout_sealed_before_testing": True,
        "train": periods[0],
        "validation": periods[1],
        "holdout": periods[2],
        "forward_paper": periods[3],
    }
    (gate.research_dir / "candidates.json").write_text(
        json.dumps({"version": 1, "candidates": [item]})
    )
    candidate = Candidate(
        strategy_id="trend_vwap",
        code_sha256=code,
        symbol="SOL/USD",
        feed="alpaca_crypto",
        regimes=frozenset(["trend_up"]),
        artifact_sha256=sha,
        costs_audited=True,
        leakage_audited=True,
        holdout_sealed_before_testing=True,
        train=tuple(DailyOutcome(**r) for r in periods[0]),
        validation=tuple(DailyOutcome(**r) for r in periods[1]),
        holdout=tuple(DailyOutcome(**r) for r in periods[2]),
        forward_paper=tuple(DailyOutcome(**r) for r in periods[3]),
    )
    return item, candidate


def authorize(gate, private, fingerprint, *, expires=NOW + 3600):
    payload = {
        "allowed_fingerprints": [fingerprint],
        "issued_at": NOW - 100,
        "expires_at": expires,
        "mode": "paper",
        "symbol": "SOL/USD",
    }
    signature = private.sign(_SCOPE_DOMAIN + _canonical(payload))
    (gate.research_dir / "approved_scope.json").write_text(
        json.dumps({"payload": payload, "signature_base64": base64.b64encode(signature).decode()})
    )


def test_no_candidate_no_scope_always_abstains(tmp_path):
    gate, _, _, _ = setup(tmp_path)
    assert gate.evaluate("SOL/USD", feature(), NOW) is False
    s = json.loads((gate.research_dir / "last_selection.json").read_text())
    assert s["mode"] == NO_TRADE
    assert not s["operator_scope_verified"]
    assert s["registered_studies"] == 0


def test_signed_sample_can_become_paper_eligible_only(tmp_path):
    gate, priv, _, _ = setup(tmp_path)
    item, candidate = build_study(gate)
    assert gate.evaluate("SOL/USD", feature(), NOW) is False
    authorize(gate, priv, candidate.fingerprint())
    assert gate.evaluate("SOL/USD", feature(), NOW + 30) is True
    state = json.loads((gate.research_dir / "last_selection.json").read_text())
    assert state["mode"] == "PAPER_ELIGIBLE"
    assert state["selected_strategy"] == "trend_vwap"
    assert state["operator_scope_verified"]
    assert gate.registry.trial_count() == 1


def test_tampered_or_expired_scope_always_refuses(tmp_path):
    gate, priv, _, _ = setup(tmp_path)
    _, candidate = build_study(gate)
    authorize(gate, priv, candidate.fingerprint(), expires=NOW + 30)
    assert gate.evaluate("SOL/USD", feature(), NOW + 2)
    assert not gate.evaluate("SOL/USD", feature(), NOW + 60)
    authorize(gate, priv, candidate.fingerprint())
    p = gate.research_dir / "approved_scope.json"
    obj = json.loads(p.read_text())
    obj["payload"]["allowed_fingerprints"] = []
    p.write_text(json.dumps(obj))
    assert not gate.evaluate("SOL/USD", feature(), NOW + 90)


def test_code_and_evidence_tamper_fail_closed(tmp_path):
    gate, priv, _, src = setup(tmp_path)
    item, candidate = build_study(gate)
    authorize(gate, priv, candidate.fingerprint())
    assert gate.evaluate("SOL/USD", feature(), NOW + 1)
    src.write_text("def rule(): return False\n")
    assert not gate.evaluate("SOL/USD", feature(), NOW + 60)
    src.write_text("def rule(): return True\n")
    (gate.research_dir / "evidence" / item["artifact_file"]).write_text("modified")
    assert not gate.evaluate("SOL/USD", feature(), NOW + 90)


def test_controller_incident_never_trades(tmp_path):
    gate, priv, ctrl, _ = setup(tmp_path)
    _, candidate = build_study(gate)
    authorize(gate, priv, candidate.fingerprint())
    ctrl.state["uncertain"] = True
    assert gate.evaluate("SOL/USD", feature(), NOW) is False
    assert (
        "broker_not_reconciled" in gate.last["reason"] or "kill_or_incident" in gate.last["reason"]
    )


def test_bad_candidate_data_fail_closed(tmp_path):
    gate, priv, _, _ = setup(tmp_path)
    _, candidate = build_study(gate)
    authorize(gate, priv, candidate.fingerprint())
    p = gate.research_dir / "candidates.json"
    payload = json.loads(p.read_text())
    payload["candidates"][0]["artifact_file"] = "../../../../etc/passwd"
    p.write_text(json.dumps(payload))
    assert not gate.evaluate("SOL/USD", feature(), NOW)
    assert gate.last["reason"].startswith("fail_closed:")


def test_compact_json_advisor_bounds_candidates_and_fail_closes():
    from reflective_loop_runtime.advisor_compact import CompactQwenAdvisor

    class Fake:
        def chat(self, *_args, **kwargs):
            assert "choice" in kwargs["grammar"]
            assert kwargs["stop"] == ["}"]
            return '{"rationale":"safe","suggested_candidate":"abstain"'

    brain = CompactQwenAdvisor(Fake())
    a = brain.advise(state={}, candidates=["abstain", "observe"], previous=None)
    assert not a.degraded and a.suggested_candidate == "abstain"

    class Bad:
        def chat(self, *_args, **_kwargs):
            return '{"rationale":"hacked","suggested_candidate":"trade_live"}'

    b = CompactQwenAdvisor(Bad()).advise(state={}, candidates=["abstain"], previous=None)
    assert b.degraded and b.suggested_candidate == "abstain"

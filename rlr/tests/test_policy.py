import json
import shutil

import pytest

from rlr.governor import Governor
from rlr.policy import PolicyEngine
from rlr.store import Store


def sample(temp=60, ram_percent=20):
    return {"zones_c": [temp], "ram_used_mib": ram_percent,
            "ram_total_mib": 100, "ram_percent": ram_percent,
            "acquisition_stalled": False}


def engine(tmp_path, **kwargs):
    store = Store(tmp_path / "brain.db")
    return store, PolicyEngine(store, Governor(store), **kwargs)


def test_opa_allow_cap_and_deny(tmp_path):
    if not shutil.which("opa"):
        pytest.skip("OPA binary unavailable")
    store, policy = engine(tmp_path)
    settings = {"load_target": 100}
    allow = policy.decide({"action": "observe", "rationale": "look"}, sample(), settings)
    cap = policy.decide({"action": "set_load_target", "percent": 90, "rationale": "load"},
                        sample(71), settings)
    deny = policy.decide({"action": "set_load_target", "percent": 90, "rationale": "load"},
                         sample(78), settings)
    assert allow.decision == "ALLOW"
    assert cap.decision == "CAP" and cap.effective_action["percent"] == 50
    assert deny.decision == "DENY"
    store.close()


def test_opa_unavailable_and_malformed_fail_closed(tmp_path):
    store, policy = engine(tmp_path, opa_path="/definitely/missing/opa")
    decision = policy.decide({"action": "observe", "rationale": "look"}, sample(), {})
    assert decision.decision == "ABSTAIN"
    assert decision.effective_action["action"] == "abstain"
    safe = policy.decide({"action": "cooldown", "seconds": 5, "rationale": "safe"},
                         {"zones_c": []}, {})
    assert safe.decision == "ALLOW"
    store.close()


def test_policy_request_is_persisted(tmp_path):
    if not shutil.which("opa"):
        pytest.skip("OPA binary unavailable")
    store, policy = engine(tmp_path)
    request_id = "11111111-1111-1111-1111-111111111111"
    result = policy.decide({"action": "observe", "rationale": "look"}, sample(), {},
                           request_id=request_id)
    row = store.policy_decision(request_id)
    assert row["decision"] == result.decision
    assert row["policy_version"].startswith("rlr-opa-v1:")
    store.close()

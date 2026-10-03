"""Deterministic policy decision contract backed by version-controlled OPA/Rego."""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

DECISIONS = {"ALLOW", "CAP", "ABSTAIN", "DENY"}
SAFE_ACTIONS = {"abstain", "cooldown"}


@dataclass(frozen=True)
class PolicyDecision:
    decision: str
    requested_action: dict
    effective_action: dict
    reason_code: str
    reason: str
    policy_version: str

    def __post_init__(self):
        if self.decision not in DECISIONS:
            raise ValueError("invalid policy decision")

    def as_dict(self):
        return asdict(self)


class PolicyEngine:
    """Narrow OPA adapter. OPA failure is fail-closed for normal actions."""
    def __init__(self, store, governor, opa_path="opa", policy_path=None, timeout_s=0.25):
        self.store = store
        self.governor = governor
        self.opa_path = str(opa_path)
        root = Path(__file__).resolve().parents[2]
        self.policy_path = Path(policy_path) if policy_path else root / "policy" / "rlr.rego"
        self.timeout_s = float(timeout_s)
        try:
            digest = hashlib.sha256(self.policy_path.read_bytes()).hexdigest()[:12]
        except OSError:
            digest = "missing"
        self.policy_version = "rlr-opa-v1:" + digest

    def _persist(self, request_id, decision):
        if request_id:
            self.store.add_policy_decision(request_id, decision.as_dict())
        return decision

    def _emergency(self, action, request_id=None):
        decision = PolicyDecision(
            "ALLOW", dict(action), dict(action), "emergency_safe_action",
            "Emergency deterministic safe action does not depend on OPA availability.",
            self.policy_version)
        return self._persist(request_id, decision)

    def _fail_closed(self, action, request_id, reason_code="policy_engine_failure"):
        effective = {"action": "abstain", "reason": reason_code,
                     "rationale": "Policy evaluation failed; abstained without acting."}
        decision = PolicyDecision(
            "ABSTAIN", dict(action), effective, reason_code,
            "Policy engine was unavailable, malformed, or timed out; failed closed.",
            self.policy_version)
        return self._persist(request_id, decision)

    def decide(self, action, sample, settings, safety_state="NORMAL", request_id=None):
        action = dict(action or {})
        if action.get("action") in SAFE_ACTIONS:
            return self._emergency(action, request_id=request_id)
        payload = {
            "proposal": action,
            "telemetry": dict(sample or {}),
            "settings": dict(settings or {}),
            "safety_state": safety_state,
            "recent_actions": self.store.recent_actions(8),
            "recent_governor_incidents": self.store.recent_incidents(8),
            "measured_capabilities": self.store.capabilities(),
            "execution_failures": self.store.recent_execution_failures(5),
            "seconds_since_aggressive_action": self.store.seconds_since_aggressive_action(),
        }
        try:
            proc = subprocess.run(
                [self.opa_path, "eval", "--format=json", "--data", str(self.policy_path),
                 "--stdin-input", "data.rlr.decision"],
                input=json.dumps(payload, ensure_ascii=False), capture_output=True,
                text=True, timeout=self.timeout_s, shell=False)
            if proc.returncode != 0:
                return self._fail_closed(action, request_id)
            raw = json.loads(proc.stdout)
            value = raw["result"][0]["expressions"][0]["value"]
            if not isinstance(value, dict) or value.get("decision") not in DECISIONS:
                return self._fail_closed(action, request_id)
            decision = PolicyDecision(
                value["decision"], action, dict(value.get("effective_action") or {}),
                str(value.get("reason_code") or "opa_decision"),
                str(value.get("reason") or "OPA policy decision"), self.policy_version)
            return self._persist(request_id, decision)
        except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, IndexError,
                TypeError, json.JSONDecodeError):
            return self._fail_closed(action, request_id)

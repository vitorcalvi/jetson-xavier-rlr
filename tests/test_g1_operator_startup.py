"""G1 S1 deploy startup must require the exact bounded operator authority."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from reflective_loop_runtime.authority import (
    HAVE_CRYPTOGRAPHY,
    load_operator_policy,
    self_maintenance_policy_bundle,
)
from reflective_loop_runtime.kernel.policy import sign_envelope

pytestmark = pytest.mark.skipif(
    not HAVE_CRYPTOGRAPHY, reason="cryptography required for production authority"
)

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _signed(tmp_path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private = Ed25519PrivateKey.generate()
    public = tmp_path / "operator-public.pem"
    policy = tmp_path / "policy.json"
    public.write_bytes(
        private.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    bundle = self_maintenance_policy_bundle(lifetime_seconds=24 * 3600)
    policy.write_text(
        json.dumps(sign_envelope(bundle, private_key=private, key_id="test")),
        encoding="utf-8",
    )
    return policy, public


class _StoppedKill:
    engaged = False
    reason = ""

    @classmethod
    def from_config(cls, config, permits=None):
        return cls()

    def check(self):
        return True


def test_agent_runtime_requires_both_operator_paths(monkeypatch, tmp_path):
    module = _load(ROOT / "deploy" / "agent_runtime.py", "g1_agent_missing")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("RLR_POLICY_PATH", raising=False)
    monkeypatch.delenv("RLR_POLICY_PUBLIC_KEY_PATH", raising=False)
    with pytest.raises(RuntimeError, match="requires RLR_POLICY_PATH"):
        module.main()


def test_agent_runtime_accepts_exact_reverified_policy_digest(monkeypatch, tmp_path):
    module = _load(ROOT / "deploy" / "agent_runtime.py", "g1_agent_exact")
    policy_path, public_path = _signed(tmp_path)
    expected = load_operator_policy(policy_path, public_path)

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RLR_POLICY_PATH", str(policy_path))
    monkeypatch.setenv("RLR_POLICY_PUBLIC_KEY_PATH", str(public_path))
    monkeypatch.setenv("RLR_ADVISOR_ENABLE", "0")
    monkeypatch.setenv("RLR_SANDBOX_PIDS_CGROUP_EXEMPT", "1")
    monkeypatch.setattr(module, "ProcTelemetry", lambda: object())
    monkeypatch.setattr(module, "KillSwitch", _StoppedKill)

    class FakeRuntime:
        def __init__(self, config, **kwargs):
            reloaded = load_operator_policy(
                config.policy_path, config.policy_public_key_path
            )
            self.permits = SimpleNamespace(policy=reloaded)

    monkeypatch.setattr(module, "ReflectiveLoopRuntime", FakeRuntime)
    assert module.main() == 0
    assert expected.digest == load_operator_policy(policy_path, public_path).digest


def test_agent_runtime_refuses_policy_digest_mismatch(monkeypatch, tmp_path):
    module = _load(ROOT / "deploy" / "agent_runtime.py", "g1_agent_mismatch")
    policy_path, public_path = _signed(tmp_path)
    expected = load_operator_policy(policy_path, public_path)
    different = dataclasses.replace(expected, mission_id="different-mission")

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RLR_POLICY_PATH", str(policy_path))
    monkeypatch.setenv("RLR_POLICY_PUBLIC_KEY_PATH", str(public_path))
    monkeypatch.setenv("RLR_ADVISOR_ENABLE", "0")
    monkeypatch.setenv("RLR_SANDBOX_PIDS_CGROUP_EXEMPT", "1")
    monkeypatch.setattr(module, "ProcTelemetry", lambda: object())

    class FakeRuntime:
        def __init__(self, config, **kwargs):
            self.permits = SimpleNamespace(policy=different)

    monkeypatch.setattr(module, "ReflectiveLoopRuntime", FakeRuntime)
    with pytest.raises(RuntimeError, match="does not match"):
        module.main()


def test_healthcheck_rejects_missing_then_accepts_valid_authority(
    monkeypatch, tmp_path
):
    module = _load(ROOT / "deploy" / "healthcheck.py", "g1_health")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("RLR_POLICY_PATH", raising=False)
    monkeypatch.delenv("RLR_POLICY_PUBLIC_KEY_PATH", raising=False)
    assert module.main() == 1

    policy_path, public_path = _signed(tmp_path)
    monkeypatch.setenv("RLR_POLICY_PATH", str(policy_path))
    monkeypatch.setenv("RLR_POLICY_PUBLIC_KEY_PATH", str(public_path))
    heartbeat = tmp_path / "rlr" / "heartbeat.txt"
    heartbeat.parent.mkdir(parents=True)
    heartbeat.write_text("healthy\n", encoding="utf-8")
    assert module.main() == 0

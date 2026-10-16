"""An ENFORCE deployment must fail loudly, not run OBSERVATION_ONLY.

ENFORCE needs an operator signature, so it needs the optional ``cryptography``
backend. The silent fallback -- a deployment that configures a signed policy
and quietly ends up with no authority at all -- is the failure this guards.
Nothing here reaches the network: keys and envelopes are made in-process.
"""

from __future__ import annotations

import json

import pytest

from reflective_loop_runtime import runtime as runtime_module
from reflective_loop_runtime.config import Config
from reflective_loop_runtime.kernel.policy import MODE_ENFORCE, MODE_OBSERVATION_ONLY
from reflective_loop_runtime.runtime import (
    PolicyEnforcementUnavailable,
    ReflectiveLoopRuntime,
    _default_permits,
    _policy_requests_enforce,
    policy_mode,
)


@pytest.fixture
def without_cryptography(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend the optional backend was never installed."""
    monkeypatch.setattr(runtime_module, "HAVE_CRYPTOGRAPHY", False)


def _enforce_config(tmp_path, **overrides) -> Config:
    return Config(
        data_dir=tmp_path / "data",
        policy_path=tmp_path / "policy.json",
        policy_public_key_path=tmp_path / "operator_pub.pem",
        **overrides,
    )


class TestEnforceWithoutCryptography:
    def test_configured_policy_refuses_to_start(self, tmp_path, without_cryptography):
        config = _enforce_config(tmp_path)
        with pytest.raises(PolicyEnforcementUnavailable) as excinfo:
            policy_mode(config)
        message = str(excinfo.value)
        # The operator has to be told which dependency is missing and how to
        # install it; "cryptography" alone is not enough.
        assert "ENFORCE" in message
        assert "cryptography" in message
        assert "[crypto]" in message

    def test_runtime_construction_refuses_to_start(self, tmp_path, without_cryptography):
        with pytest.raises(PolicyEnforcementUnavailable, match=r"\[crypto\]"):
            ReflectiveLoopRuntime(_enforce_config(tmp_path))

    def test_permit_service_construction_refuses_to_start(self, tmp_path, without_cryptography):
        config = _enforce_config(tmp_path)
        config.ensure_dirs()
        with pytest.raises(PolicyEnforcementUnavailable):
            _default_permits(config)

    def test_no_enforce_config_keeps_the_observation_only_fallback(
        self, tmp_path, without_cryptography
    ):
        # Nothing asked for ENFORCE, so the pre-existing fail-closed fallback
        # still applies: no operator key, no authority, loop still starts.
        config = Config(data_dir=tmp_path / "data")
        assert _policy_requests_enforce(config) is False
        assert policy_mode(config) == MODE_OBSERVATION_ONLY
        assert ReflectiveLoopRuntime(config).run_id


class TestEnforceWithCryptography:
    def test_signed_policy_file_is_loaded_as_enforce(self, tmp_path):
        pytest.importorskip("cryptography")
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        from reflective_loop_runtime.kernel.policy import sign_envelope

        private = Ed25519PrivateKey.generate()
        envelope = sign_envelope(
            {
                "mission_id": "test-mission",
                "objective": "verify a configured policy",
                "invariants": ["no_effect_without_permit"],
                "allowed_domains": [],
                "capabilities": [],
                "budgets": {"scheduler_tick": 10.0},
                "standing_mandates": [],
                "policy_version": "1.0.0",
            },
            private_key=private,
            key_id="test-operator",
        )
        policy_file = tmp_path / "policy.json"
        policy_file.write_text(json.dumps(envelope), encoding="utf-8")
        key_file = tmp_path / "operator_pub.pem"
        key_file.write_bytes(
            private.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        config = Config(
            data_dir=tmp_path / "data",
            policy_path=policy_file,
            policy_public_key_path=key_file,
        )
        assert _policy_requests_enforce(config) is True
        assert policy_mode(config) == MODE_ENFORCE

    def test_unverifiable_policy_is_not_silently_downgraded(self, tmp_path):
        pytest.importorskip("cryptography")
        policy_file = tmp_path / "policy.json"
        policy_file.write_text('{"bundle": {"mission_id": "unsigned"}}', encoding="utf-8")
        config = Config(data_dir=tmp_path / "data", policy_path=policy_file)
        with pytest.raises(PolicyEnforcementUnavailable, match="ENFORCE"):
            policy_mode(config)

    def test_default_config_enforces_with_its_own_operator_key(self, tmp_path):
        pytest.importorskip("cryptography")
        assert policy_mode(Config(data_dir=tmp_path / "data")) == MODE_ENFORCE


class TestPolicyConfig:
    def test_public_key_without_policy_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="requires policy_path"):
            Config(data_dir=tmp_path / "data", policy_public_key_path=tmp_path / "k.pem")

    def test_policy_path_alone_counts_as_an_enforce_request(self, tmp_path):
        assert _policy_requests_enforce(Config(policy_path=tmp_path / "p.json")) is True
        assert _policy_requests_enforce(Config()) is False

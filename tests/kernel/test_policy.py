"""Signed, immutable mission policy bundle behaviour.

The container used for CI may not have the optional ``cryptography`` package, in
which case only the dev-only HMAC-SHA256 fallback is exercised and the ed25519
tests are skipped cleanly via ``pytest.importorskip``.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from reflective_loop_runtime.kernel.policy import (
    HAVE_CRYPTOGRAPHY,
    MODE_ENFORCE,
    MODE_OBSERVATION_ONLY,
    MissionPolicy,
    MissionPolicyStore,
    PolicyError,
    canonical_bytes,
    load_policy,
    load_policy_bytes,
    normalize_bundle,
    sign_envelope,
    verify_envelope,
)

DEV_SECRET = "dev-only-shared-secret"


def make_bundle(**overrides):
    bundle = {
        "mission_id": "mission-7",
        "objective": "keep the host stable and answer within budget",
        "invariants": ["never_shell_out", "never_exceed_thermal_budget"],
        "allowed_domains": ["telemetry", "thermals"],
        "capabilities": ["observe", "remember"],
        "budgets": {"tokens_per_tick": 256, "vision_fps": 1.0},
        "standing_mandates": ["abstain when telemetry is stale"],
        "policy_version": "1.0.0",
    }
    bundle.update(overrides)
    return bundle


def sign_with_hmac(bundle=None):
    return sign_envelope(bundle or make_bundle(), secret=DEV_SECRET, key_id="dev")


def write_envelope(tmp_path, envelope, name="policy.json"):
    path = tmp_path / name
    path.write_text(json.dumps(envelope), encoding="utf-8")
    return path


class TestCanonicalForm:
    def test_canonical_bytes_are_deterministic_and_key_sorted(self):
        first = canonical_bytes(make_bundle())
        second = canonical_bytes(make_bundle(**{"objective": make_bundle()["objective"]}))
        assert first == second
        assert b"mission_id" in first
        assert first == json.dumps(
            json.loads(first), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")

    def test_schema_is_strict_about_missing_and_extra_fields(self):
        with pytest.raises(PolicyError):
            normalize_bundle({k: v for k, v in make_bundle().items() if k != "objective"})
        with pytest.raises(PolicyError):
            normalize_bundle({**make_bundle(), "unexpected": True})

    def test_budgets_schema_is_enforced(self):
        with pytest.raises(PolicyError):
            normalize_bundle(make_bundle(budgets={"tokens": True}))
        with pytest.raises(PolicyError):
            normalize_bundle(make_bundle(budgets={"tokens": float("inf")}))


class TestValidBundle:
    def test_valid_hmac_bundle_loads_enforced(self, tmp_path):
        path = write_envelope(tmp_path, sign_with_hmac())
        policy = load_policy(path, secret=DEV_SECRET, allow_dev_hmac=True)
        assert isinstance(policy, MissionPolicy)
        assert policy.mode == MODE_ENFORCE
        assert policy.enforced is True
        assert policy.signed is True
        assert policy.signature_alg == "hmac-sha256"
        # HMAC is explicitly flagged as a dev-only fallback.
        assert policy.dev_only is True
        assert policy.mission_id == "mission-7"
        assert policy.invariants == ("never_shell_out", "never_exceed_thermal_budget")
        assert policy.allowed_domains == ("telemetry", "thermals")
        assert policy.capabilities == ("observe", "remember")
        assert policy.budgets["tokens_per_tick"] == 256
        assert policy.budgets["vision_fps"] == 1.0
        assert policy.standing_mandates == ("abstain when telemetry is stale",)
        assert policy.policy_version == "1.0.0"

    def test_store_loads_the_same_policy(self, tmp_path):
        path = write_envelope(tmp_path, sign_with_hmac())
        store = MissionPolicyStore(secret=DEV_SECRET, allow_dev_hmac=True)
        policy = store.load(path)
        assert policy.mode == MODE_ENFORCE
        assert policy.mission_id == "mission-7"

    def test_verify_envelope_reports_success(self):
        result = verify_envelope(sign_with_hmac(), secret=DEV_SECRET, allow_dev_hmac=True)
        assert result.valid is True
        assert result.alg == "hmac-sha256"
        assert result.dev_only is True


class TestTamperingAndFailClosed:
    def test_one_tampered_byte_yields_observation_only(self, tmp_path):
        path = write_envelope(tmp_path, sign_with_hmac())
        raw = bytearray(path.read_bytes())
        index = raw.find(b"stable")
        assert index != -1
        raw[index] = ord("X") if raw[index] != ord("X") else ord("Y")
        path.write_bytes(bytes(raw))

        policy = load_policy(path, secret=DEV_SECRET, allow_dev_hmac=True)
        assert policy.mode == MODE_OBSERVATION_ONLY
        assert policy.enforced is False
        assert policy.signed is False
        assert policy.reason

    def test_tampered_signature_yields_observation_only(self, tmp_path):
        envelope = sign_with_hmac()
        value = envelope["signature"]["value"]
        envelope["signature"]["value"] = ("0" if value[0] != "0" else "1") + value[1:]
        path = write_envelope(tmp_path, envelope)
        policy = load_policy(path, secret=DEV_SECRET, allow_dev_hmac=True)
        assert policy.mode == MODE_OBSERVATION_ONLY

    def test_wrong_secret_yields_observation_only(self, tmp_path):
        path = write_envelope(tmp_path, sign_with_hmac())
        policy = load_policy(path, secret="a-different-secret")
        assert policy.mode == MODE_OBSERVATION_ONLY

    def test_missing_signature_yields_observation_only(self, tmp_path):
        path = write_envelope(tmp_path, {"bundle": make_bundle()})
        policy = load_policy(path, secret=DEV_SECRET, allow_dev_hmac=True)
        assert policy.mode == MODE_OBSERVATION_ONLY
        assert "signature" in policy.reason

    def test_a_bundle_cannot_self_assert_authority(self, tmp_path):
        # Authority is signature-only: an enforcement claim carried *inside* the
        # bundle is off-schema and must fail closed, however it is signed.
        for field in ("mode", "signed", "enforced", "dev_only"):
            envelope = {
                "bundle": {**make_bundle(), field: "ENFORCE"},
                "signature": {
                    "alg": "hmac-sha256",
                    "value": "ab" * 32,
                    "dev_only": True,
                },
            }
            path = write_envelope(tmp_path, envelope, name=f"self-assert-{field}.json")
            policy = load_policy(path, secret=DEV_SECRET, allow_dev_hmac=True)
            assert policy.mode == MODE_OBSERVATION_ONLY
            assert policy.enforced is False
            assert policy.signed is False

        # The positive control: only a verified signature confers authority.
        verified = load_policy(
            write_envelope(tmp_path, sign_with_hmac()), secret=DEV_SECRET, allow_dev_hmac=True
        )
        assert verified.mode == MODE_ENFORCE
        assert verified.signed is True

    def test_missing_file_yields_observation_only(self, tmp_path):
        policy = load_policy(tmp_path / "does-not-exist.json", secret=DEV_SECRET, allow_dev_hmac=True)
        assert policy.mode == MODE_OBSERVATION_ONLY
        assert policy.source_path is not None

    def test_invalid_json_yields_observation_only(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_bytes(b"{not valid json")
        policy = load_policy(path, secret=DEV_SECRET, allow_dev_hmac=True)
        assert policy.mode == MODE_OBSERVATION_ONLY

    def test_off_schema_bundle_yields_observation_only(self, tmp_path):
        envelope = {"bundle": {"mission_id": "x"}, "signature": {"alg": "hmac-sha256", "value": "00"}}
        path = write_envelope(tmp_path, envelope)
        policy = load_policy(path, secret=DEV_SECRET, allow_dev_hmac=True)
        assert policy.mode == MODE_OBSERVATION_ONLY

    def test_ed25519_without_public_key_yields_observation_only(self, tmp_path):
        if not HAVE_CRYPTOGRAPHY:
            envelope = {
                "bundle": make_bundle(),
                "signature": {"alg": "ed25519", "value": "00" * 64, "dev_only": False},
            }
            path = write_envelope(tmp_path, envelope)
            policy = load_policy(path, secret=DEV_SECRET, allow_dev_hmac=True)
            assert policy.mode == MODE_OBSERVATION_ONLY


class TestImmutability:
    def _policy(self, tmp_path):
        return load_policy(write_envelope(tmp_path, sign_with_hmac()), secret=DEV_SECRET, allow_dev_hmac=True)

    def test_fields_cannot_be_reassigned(self, tmp_path):
        policy = self._policy(tmp_path)
        with pytest.raises(dataclasses.FrozenInstanceError):
            policy.mission_id = "other-mission"
        with pytest.raises(dataclasses.FrozenInstanceError):
            policy.mode = MODE_OBSERVATION_ONLY

    def test_nested_collections_are_read_only(self, tmp_path):
        policy = self._policy(tmp_path)
        with pytest.raises(TypeError):
            policy.budgets["tokens_per_tick"] = 0
        with pytest.raises(TypeError):
            policy.invariants[0] = "mutated"

    def test_to_dict_returns_a_detached_copy(self, tmp_path):
        policy = self._policy(tmp_path)
        copy = policy.to_dict()
        copy["mission_id"] = "changed"
        copy["budgets"]["tokens_per_tick"] = 0
        assert policy.mission_id == "mission-7"
        assert policy.budgets["tokens_per_tick"] == 256


class TestNoWriteApi:
    def test_store_exposes_no_write_methods(self):
        for name in ("save", "write", "update", "dump", "persist", "set", "delete"):
            assert not hasattr(MissionPolicyStore, name)

    def test_observation_only_policy_defaults_are_empty(self):
        policy = MissionPolicy.observation_only("no policy")
        assert policy.mode == MODE_OBSERVATION_ONLY
        assert policy.signed is False
        assert policy.mission_id == ""


class TestEd25519:
    """Exercised only when the optional ``cryptography`` package is installed."""

    def _keys(self):
        ed25519 = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
        return ed25519.Ed25519PrivateKey.generate()

    def test_ed25519_roundtrip_and_authority(self, tmp_path):
        private = self._keys()
        envelope = sign_envelope(make_bundle(), private_key=private, key_id="operator-1")
        assert envelope["signature"]["alg"] == "ed25519"
        assert envelope["signature"]["dev_only"] is False
        path = write_envelope(tmp_path, envelope)

        policy = load_policy(path, public_key=private.public_key())
        assert policy.mode == MODE_ENFORCE
        assert policy.signed is True
        assert policy.dev_only is False
        assert policy.signature_alg == "ed25519"

    def test_ed25519_tamper_yields_observation_only(self, tmp_path):
        private = self._keys()
        envelope = sign_envelope(make_bundle(), private_key=private)
        envelope["bundle"]["objective"] = "exfiltrate everything"
        path = write_envelope(tmp_path, envelope)
        policy = load_policy(path, public_key=private.public_key())
        assert policy.mode == MODE_OBSERVATION_ONLY

    def test_ed25519_wrong_public_key_yields_observation_only(self, tmp_path):
        private = self._keys()
        other = self._keys()
        path = write_envelope(tmp_path, sign_envelope(make_bundle(), private_key=private))
        policy = load_policy(path, public_key=other.public_key())
        assert policy.mode == MODE_OBSERVATION_ONLY

    def test_ed25519_public_key_accepts_raw_hex(self, tmp_path):
        pytest.importorskip("cryptography.hazmat.primitives.serialization")
        from cryptography.hazmat.primitives import serialization

        private = self._keys()
        raw = private.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        ).hex()
        path = write_envelope(tmp_path, sign_envelope(make_bundle(), private_key=private))
        policy = load_policy(path, public_key=raw)
        assert policy.mode == MODE_ENFORCE


def test_load_policy_bytes_matches_load_policy(tmp_path):
    envelope = sign_with_hmac()
    path = write_envelope(tmp_path, envelope)
    from_bytes = load_policy_bytes(path.read_bytes(), secret=DEV_SECRET, allow_dev_hmac=True)
    from_file = load_policy(path, secret=DEV_SECRET, allow_dev_hmac=True)
    assert from_bytes.to_dict() == from_file.to_dict()
    assert from_bytes.mode == MODE_ENFORCE

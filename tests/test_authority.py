"""G1 S1: external Ed25519 authority is bounded, dated and fail-closed."""

from __future__ import annotations

import dataclasses
import json
from copy import deepcopy

import pytest

from reflective_loop_runtime.authority import (
    CLOCK_SKEW_SECONDS,
    MAX_POLICY_LIFETIME_SECONDS,
    REQUIRED_INVARIANTS,
    SELF_MAINTENANCE_BUDGET_MAX,
    SELF_MAINTENANCE_CAPABILITIES,
    SELF_MAINTENANCE_DOMAINS,
    AuthorityError,
    load_operator_policy,
    require_self_maintenance_scope,
    self_maintenance_policy_bundle,
)
from reflective_loop_runtime.kernel.policy import HAVE_CRYPTOGRAPHY, MODE_ENFORCE, sign_envelope

pytestmark = pytest.mark.skipif(
    not HAVE_CRYPTOGRAPHY, reason="cryptography required for production authority"
)

NOW = 2_000_000_000


def _signed(tmp_path, bundle=None):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    tmp_path.mkdir(parents=True, exist_ok=True)
    private = Ed25519PrivateKey.generate()
    public_path = tmp_path / "operator-public.pem"
    public_path.write_bytes(
        private.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            sign_envelope(
                deepcopy(
                    bundle
                    or self_maintenance_policy_bundle(
                        issued_at=NOW - 60, lifetime_seconds=3600
                    )
                ),
                private_key=private,
                key_id="test-operator",
            )
        ),
        encoding="utf-8",
    )
    return policy_path, public_path


def _rejects(tmp_path, mutate, match=None):
    bundle = self_maintenance_policy_bundle(issued_at=NOW - 60, lifetime_seconds=3600)
    mutate(bundle)
    policy, public = _signed(tmp_path, bundle)
    with pytest.raises(AuthorityError, match=match):
        load_operator_policy(policy, public, now=NOW)


def test_default_policy_is_ed25519_enforce_and_minimally_bounded(tmp_path):
    policy_path, public_path = _signed(tmp_path)
    policy = load_operator_policy(policy_path, public_path, now=NOW)
    assert policy.mode == MODE_ENFORCE
    assert policy.enforced and policy.signed and not policy.dev_only
    assert policy.signature_alg == "ed25519"
    assert set(policy.allowed_domains) == set(SELF_MAINTENANCE_DOMAINS)
    assert set(policy.capabilities) == set(SELF_MAINTENANCE_CAPABILITIES)
    assert "telemetry.sample" not in policy.capabilities
    assert "control.cooldown" in policy.capabilities
    assert set(policy.invariants) >= set(REQUIRED_INVARIANTS)
    assert set(policy.budgets) == set(SELF_MAINTENANCE_BUDGET_MAX)
    assert policy.standing_mandates == ()


def test_empty_or_overbroad_domains_and_capabilities_are_rejected(tmp_path):
    _rejects(tmp_path / "empty-domain", lambda b: b.update(allowed_domains=[]), "self-maintenance")
    _rejects(
        tmp_path / "shell-domain",
        lambda b: b.update(allowed_domains=list(b["allowed_domains"]) + ["shell"]),
        "domain ceiling",
    )
    _rejects(tmp_path / "empty-cap", lambda b: b.update(capabilities=[]), "self-maintenance")
    _rejects(
        tmp_path / "shell-cap",
        lambda b: b.update(capabilities=list(b["capabilities"]) + ["shell.run"]),
        "capability ceiling",
    )
    _rejects(
        tmp_path / "a3-cap",
        lambda b: b.update(capabilities=list(b["capabilities"]) + ["memory.forget"]),
        "capability ceiling",
    )


def test_missing_invariant_wrong_mission_and_any_mandate_are_rejected(tmp_path):
    _rejects(
        tmp_path / "mission",
        lambda b: b.update(mission_id="other"),
        "mission id",
    )
    _rejects(
        tmp_path / "inv",
        lambda b: b.update(invariants=[x for x in b["invariants"] if x != "audit_before_effect"]),
        "safety invariants",
    )
    _rejects(
        tmp_path / "mandate",
        lambda b: b.update(standing_mandates=["maintenance.unit_restart"]),
        "standing A3",
    )


@pytest.mark.parametrize(
    "name,mutate",
    [
        ("missing", lambda d: d.pop("io_read")),
        ("extra", lambda d: d.__setitem__("api_calls", 1.0)),
        ("zero", lambda d: d.__setitem__("io_read", 0.0)),
        ("negative", lambda d: d.__setitem__("io_read", -1.0)),
        ("nan", lambda d: d.__setitem__("io_read", "nan")),
        ("inf", lambda d: d.__setitem__("io_read", "inf")),
        (
            "over",
            lambda d: d.__setitem__(
                "io_read", SELF_MAINTENANCE_BUDGET_MAX["io_read"] + 1.0
            ),
        ),
    ],
)
def test_budget_map_cannot_escape_ceiling(tmp_path, name, mutate):
    def change(bundle):
        budgets = dict(bundle["budgets"])
        mutate(budgets)
        bundle["budgets"] = budgets

    _rejects(tmp_path / name, change, "budget")


def test_dev_hmac_and_wrong_ed25519_key_are_rejected(tmp_path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    public = Ed25519PrivateKey.generate().public_key()
    public_path = tmp_path / "public.pem"
    public_path.write_bytes(
        public.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    hmac_path = tmp_path / "hmac.json"
    hmac_path.write_text(
        json.dumps(
            sign_envelope(
                self_maintenance_policy_bundle(
                    issued_at=NOW - 60, lifetime_seconds=3600
                ),
                secret=b"x" * 32,
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(AuthorityError):
        load_operator_policy(hmac_path, public_path, now=NOW)

    policy_path, _ = _signed(tmp_path / "signed")
    with pytest.raises(AuthorityError):
        load_operator_policy(policy_path, public_path, now=NOW)


def test_symlinked_policy_or_public_key_is_rejected(tmp_path):
    policy, public = _signed(tmp_path / "real")
    policy_link = tmp_path / "policy-link"
    public_link = tmp_path / "public-link"
    policy_link.symlink_to(policy)
    public_link.symlink_to(public)
    with pytest.raises(AuthorityError, match="symlink"):
        load_operator_policy(policy_link, public, now=NOW)
    with pytest.raises(AuthorityError, match="symlink"):
        load_operator_policy(policy, public_link, now=NOW)


def test_expired_and_not_yet_valid_signed_policies_are_rejected(tmp_path):
    expired = self_maintenance_policy_bundle(
        issued_at=NOW - 3600, lifetime_seconds=60
    )
    p1, k1 = _signed(tmp_path / "expired", expired)
    with pytest.raises(AuthorityError, match="expired"):
        load_operator_policy(p1, k1, now=NOW)

    future = self_maintenance_policy_bundle(
        issued_at=NOW + CLOCK_SKEW_SECONDS + 1, lifetime_seconds=3600
    )
    p2, k2 = _signed(tmp_path / "future", future)
    with pytest.raises(AuthorityError, match="not yet valid"):
        load_operator_policy(p2, k2, now=NOW)


def test_validity_window_is_signed_and_tampering_breaks_authority(tmp_path):
    policy_path, public_path = _signed(tmp_path)
    document = json.loads(policy_path.read_text(encoding="utf-8"))
    version = document["bundle"]["policy_version"]
    assert "issued_at=" in version and "not_after=" in version
    document["bundle"]["policy_version"] = version.replace("not_after=", "not_after=9")
    policy_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(AuthorityError, match="Ed25519 ENFORCE"):
        load_operator_policy(policy_path, public_path, now=NOW)


def test_lifetime_above_ceiling_is_refused_before_signing():
    with pytest.raises(AuthorityError, match="lifetime"):
        self_maintenance_policy_bundle(
            issued_at=NOW,
            lifetime_seconds=MAX_POLICY_LIFETIME_SECONDS + 1,
        )


def test_nonfinite_budget_is_rejected_after_verified_load(tmp_path):
    policy_path, public_path = _signed(tmp_path)
    policy = load_operator_policy(policy_path, public_path, now=NOW)
    budgets = dict(policy.budgets)
    budgets["io_read"] = float("nan")
    forged_view = dataclasses.replace(policy, budgets=budgets)
    with pytest.raises(AuthorityError, match="finite"):
        require_self_maintenance_scope(forged_view, now=NOW)

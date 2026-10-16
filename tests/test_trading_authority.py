"""RLR paper-trading operator policy: signed, scoped, mutually exclusive.

The trader lane shares the kernel policy verifier with the G1 self-maintenance
runtime but receives its own externally signed Ed25519 envelope. These tests
prove that:

* ``load_trading_policy`` accepts a fresh, in-window, paper-trading bundle;
* the self-maintenance bundle is rejected by ``load_trading_policy``;
* a paper-trading bundle is rejected by ``load_operator_policy``;
* expired or tampered paper-trading envelopes fail closed;
* the trusted-host signer keeps its default behaviour (self-maintenance) when
  ``--domain`` is omitted.

No network is touched. The offline JSON fixture under
``tests/trading/fixtures/`` anchors the lane fixture tree.
"""

from __future__ import annotations

import importlib.util
import json
import time
from copy import deepcopy
from pathlib import Path

import pytest

from reflective_loop_runtime.authority import (
    TRADING_BUDGET_MAX,
    TRADING_CAPABILITIES,
    TRADING_DOMAINS,
    AuthorityError,
    load_operator_policy,
    load_trading_policy,
    paper_trading_policy_bundle,
    self_maintenance_policy_bundle,
)
from reflective_loop_runtime.kernel.policy import (
    HAVE_CRYPTOGRAPHY,
    MODE_ENFORCE,
    sign_envelope,
)

pytestmark = pytest.mark.skipif(
    not HAVE_CRYPTOGRAPHY, reason="cryptography required for production authority"
)

# The lane fixture tree must be real so the trader lane and other test lanes can
# anchor shared offline data here.
FIXTURES_DIR = Path(__file__).resolve().parent / "trading" / "fixtures"
SAMPLE_MARKET_FIXTURE = FIXTURES_DIR / "sample_market.json"

NOW = 2_000_000_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_ed25519_keypair():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private = Ed25519PrivateKey.generate()
    public_pem = private.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private, public_pem


def _write_policy(
    tmp_path: Path,
    bundle: dict,
    private,
    key_id: str,
    *,
    public_name: str = "trading-public.pem",
    policy_name: str = "policy.json",
) -> tuple[Path, Path]:
    from cryptography.hazmat.primitives import serialization

    tmp_path.mkdir(parents=True, exist_ok=True)
    public_path = tmp_path / public_name
    public_path.write_bytes(
        private.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    policy_path = tmp_path / policy_name
    policy_path.write_text(
        json.dumps(sign_envelope(deepcopy(bundle), private_key=private, key_id=key_id)),
        encoding="utf-8",
    )
    return policy_path, public_path


def _write_envelope(
    tmp_path: Path,
    envelope: dict,
    *,
    public_pem: bytes,
    public_name: str = "trading-public.pem",
    policy_name: str = "policy.json",
) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    public_path = tmp_path / public_name
    public_path.write_bytes(public_pem)
    policy_path = tmp_path / policy_name
    policy_path.write_text(json.dumps(envelope), encoding="utf-8")
    return policy_path, public_path


def _signed_trading(tmp_path: Path, *, bundle=None, issued_at=None, lifetime=3600):
    if bundle is None:
        if issued_at is None:
            issued_at = NOW - 60
        bundle = paper_trading_policy_bundle(issued_at=issued_at, lifetime_seconds=lifetime)
    private, public_pem = _generate_ed25519_keypair()
    return _write_policy(tmp_path, bundle, private, "trading-operator") + (private, public_pem)


def _signed_self_maintenance(tmp_path: Path):
    bundle = self_maintenance_policy_bundle(issued_at=NOW - 60, lifetime_seconds=3600)
    private, public_pem = _generate_ed25519_keypair()
    return _write_policy(tmp_path, bundle, private, "g1-operator") + (private, public_pem)


# ---------------------------------------------------------------------------
# Named tests
# ---------------------------------------------------------------------------


def test_trading_policy_signed_and_scoped(tmp_path):
    """A fresh, in-window paper-trading envelope verifies and stays scoped."""
    assert SAMPLE_MARKET_FIXTURE.is_file(), "trading fixture tree must exist"
    policy_path, public_path, _private, _public_pem = _signed_trading(tmp_path)

    policy = load_trading_policy(policy_path, public_path, now=NOW)

    assert policy.mode == MODE_ENFORCE
    assert policy.enforced is True
    assert policy.signed is True
    assert policy.dev_only is False
    assert policy.signature_alg == "ed25519"
    assert policy.mission_id == "rlr-paper-trading"
    assert set(policy.allowed_domains) == set(TRADING_DOMAINS)
    assert set(policy.capabilities) == set(TRADING_CAPABILITIES)
    assert set(policy.budgets) == set(TRADING_BUDGET_MAX)
    for resource, value in policy.budgets.items():
        assert 0.0 < float(value) <= TRADING_BUDGET_MAX[resource]


def test_trading_policy_rejects_self_maintenance_bundle(tmp_path):
    """A self-maintenance envelope must not satisfy the trading scope check."""
    policy_path, public_path, _private, _public_pem = _signed_self_maintenance(tmp_path)

    with pytest.raises(AuthorityError, match="mission id"):
        load_trading_policy(policy_path, public_path, now=NOW)


def test_self_maintenance_loader_rejects_trading_bundle(tmp_path):
    """A paper-trading envelope must not satisfy the G1 self-maintenance scope."""
    policy_path, public_path, _private, _public_pem = _signed_trading(tmp_path)

    with pytest.raises(AuthorityError, match="mission id"):
        load_operator_policy(policy_path, public_path, now=NOW)


def test_trading_policy_rejects_expired(tmp_path):
    """An expired paper-trading envelope is rejected by the trading loader."""
    expired = paper_trading_policy_bundle(
        issued_at=NOW - 3600,
        lifetime_seconds=60,
    )
    policy_path, public_path, _private, _public_pem = _signed_trading(tmp_path, bundle=expired)

    with pytest.raises(AuthorityError, match="expired"):
        load_trading_policy(policy_path, public_path, now=NOW)


def test_trading_policy_rejects_tampered(tmp_path):
    """Editing the signed bundle breaks the Ed25519 ENFORCE verification."""
    policy_path, public_path, _private, public_pem = _signed_trading(tmp_path)

    document = json.loads(policy_path.read_text(encoding="utf-8"))
    version = document["bundle"]["policy_version"]
    assert "issued_at=" in version and "not_after=" in version
    document["bundle"]["policy_version"] = version.replace("not_after=", "not_after=9")
    tampered_path, tampered_public = _write_envelope(
        tmp_path / "tampered",
        document,
        public_pem=public_pem,
        public_name=public_path.name,
        policy_name=policy_path.name,
    )

    with pytest.raises(AuthorityError, match="Ed25519 ENFORCE"):
        load_trading_policy(tampered_path, tampered_public, now=NOW)


def test_signer_domain_flag_default_unchanged(tmp_path):
    """The trusted-host signer must keep the default self-maintenance behaviour."""
    spec = importlib.util.spec_from_file_location(
        "g1_signer",
        Path(__file__).resolve().parents[1] / "deploy" / "sign_self_maintenance_policy.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = [
        "--operator-key-env",
        str(tmp_path / "operator-key.env"),
        "--public-key-out",
        str(tmp_path / "operator-public.pem"),
        "--policy-out",
        str(tmp_path / "policy.json"),
        "--generate",
        "--valid-hours",
        "24",
    ]
    assert module.main(args) == 0

    document = json.loads((tmp_path / "policy.json").read_text(encoding="utf-8"))
    bundle = document["bundle"]
    signature = document["signature"]
    assert bundle["mission_id"] == "g1-self-maintenance"
    assert signature["key_id"] == "g1-operator"
    assert "alpaca_paper" not in bundle["allowed_domains"]
    assert "alpaca_paper.order" not in bundle["capabilities"]

    # The signed self-maintenance envelope loads through the G1 loader at the
    # wall clock the signer just stamped; the trading loader rejects it because
    # the mission id is the self-maintenance one, not the paper-trading one.
    now = int(time.time())
    policy = load_operator_policy(
        tmp_path / "policy.json",
        tmp_path / "operator-public.pem",
        now=now,
    )
    assert policy.enforced is True
    with pytest.raises(AuthorityError, match="mission id"):
        load_trading_policy(
            tmp_path / "policy.json",
            tmp_path / "operator-public.pem",
            now=now,
        )


def test_trading_policy_scope_matches_kernel_permit(tmp_path):
    """A signed paper-trading bundle scopes exactly the kernel-permit domain/capability.

    Signing the bundle with a test Ed25519 key, loading it with
    ``load_trading_policy`` and issuing a permit for ``alpaca_paper.order`` must
    succeed end-to-end so the trading policy scope stays aligned with the
    permit kernel: domain == tool name, capability == "tool.op".
    """
    from reflective_loop_runtime.kernel.permit import PermitIssuer, PermitService
    from reflective_loop_runtime.trading.loop import trading_manifest

    policy_path, public_path, _private, _public_pem = _signed_trading(
        tmp_path,
        bundle=paper_trading_policy_bundle(issued_at=NOW - 60, lifetime_seconds=3600),
    )

    policy = load_trading_policy(policy_path, public_path, now=NOW)
    assert set(policy.allowed_domains) == {"alpaca_paper"}
    assert set(policy.capabilities) == {"alpaca_paper.order"}

    service = PermitService(
        PermitIssuer(),
        policy,
        manifest=trading_manifest(),
        db_path=tmp_path / "p.db",
    )
    try:
        permit = service.issue(
            tool="alpaca_paper",
            op="order",
            args={"symbol": "BTC/USD", "side": "buy", "notional": 10.0, "client_order_id": "rlr-x"},
            goal_id="t",
        )
    finally:
        service.close()

    assert permit.tool == "alpaca_paper"
    assert permit.op == "order"
    assert permit.capability == "alpaca_paper.order"

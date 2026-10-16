"""Strict operator-policy preflight for G1 self-maintenance.

Production G1 accepts only an externally signed Ed25519 ENFORCE policy whose
scope, budgets and signed validity window stay inside this deploy ceiling.
The kernel runtime independently re-verifies the same signed envelope.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from .kernel.policy import (
    ALG_ED25519,
    HAVE_CRYPTOGRAPHY,
    MODE_ENFORCE,
    MissionPolicy,
    load_policy_bytes,
    verified_authority,
)

SELF_MAINTENANCE_DOMAINS = frozenset({"control", "maintenance"})
SELF_MAINTENANCE_CAPABILITIES = frozenset(
    {
        # Reversible in-process self-protection; no host authority.
        "control.cooldown",
        # Host-facing authority is maintenance-only and explicit.
        "maintenance.unit_status",
        "maintenance.unit_restart",
        "maintenance.heartbeat_status",
        "maintenance.disk_usage",
        "maintenance.rotate_logs",
        "maintenance.archive_old",
        "maintenance.adjust_config",
    }
)
REQUIRED_INVARIANTS = frozenset(
    {
        "no_effect_without_permit",
        "audit_before_effect",
        "own_user_units_only",
        "own_rlr_paths_only",
        "never_delete_use_archive_rename",
        "never_expand_model_capabilities",
        "no_a3_actions",
    }
)

# Every resource used by the planned maintenance manifest has a finite 24h
# ceiling with headroom.
SELF_MAINTENANCE_BUDGET_MAX = {
    "io_read": 250_000.0,
    "scheduler_tick": 100_000.0,
}

POLICY_VERSION_PREFIX = "g1-self-maintenance-3"
DEFAULT_POLICY_LIFETIME_SECONDS = 7 * 24 * 60 * 60
MAX_POLICY_LIFETIME_SECONDS = 30 * 24 * 60 * 60
CLOCK_SKEW_SECONDS = 5 * 60


class AuthorityError(RuntimeError):
    """Production operator authority is absent, invalid, expired, or too broad."""


def _version_with_window(issued_at: int, not_after: int) -> str:
    return (
        f"{POLICY_VERSION_PREFIX};issued_at={int(issued_at)};"
        f"not_after={int(not_after)}"
    )


def self_maintenance_policy_bundle(
    *,
    issued_at: int | float | None = None,
    lifetime_seconds: int | float = DEFAULT_POLICY_LIFETIME_SECONDS,
) -> dict[str, Any]:
    """Return the bounded G1 bundle with a validity window inside signed bytes."""
    start = int(time.time() if issued_at is None else issued_at)
    if isinstance(lifetime_seconds, bool):
        raise AuthorityError("policy lifetime must be finite and positive")
    try:
        lifetime = int(lifetime_seconds)
    except (TypeError, ValueError) as exc:
        raise AuthorityError("policy lifetime must be finite and positive") from exc
    if lifetime <= 0 or lifetime > MAX_POLICY_LIFETIME_SECONDS:
        raise AuthorityError("policy lifetime exceeds the deploy ceiling")
    end = start + lifetime
    return {
        "mission_id": "g1-self-maintenance",
        "objective": (
            "keep the Jetson reflective-loop runtime healthy using only bounded "
            "self-maintenance authority over its own user services, files, logs, "
            "heartbeat, and allowlisted configuration knobs"
        ),
        "invariants": sorted(REQUIRED_INVARIANTS),
        "allowed_domains": sorted(SELF_MAINTENANCE_DOMAINS),
        "capabilities": sorted(SELF_MAINTENANCE_CAPABILITIES),
        "budgets": {
            "io_read": 200_000.0,
            "scheduler_tick": 90_000.0,
        },
        "standing_mandates": [],
        "policy_version": _version_with_window(start, end),
    }


# Compatibility/reference value for tests and callers that want a current
# bounded bundle. Signers should call self_maintenance_policy_bundle() at sign
# time so the window starts when the artifact is authored.
SELF_MAINTENANCE_POLICY_BUNDLE = self_maintenance_policy_bundle()


def _regular_nonsymlink(path: Path, label: str) -> None:
    if path.is_symlink():
        raise AuthorityError(f"{label} may not be a symlink")
    if not path.is_file():
        raise AuthorityError(f"{label} is unavailable")


def _require_budget_ceiling(policy: MissionPolicy) -> None:
    budgets = dict(policy.budgets)
    required = set(SELF_MAINTENANCE_BUDGET_MAX)
    if set(budgets) != required:
        raise AuthorityError("operator policy budget keys do not match the deploy ceiling")
    for resource, ceiling in SELF_MAINTENANCE_BUDGET_MAX.items():
        value = budgets.get(resource)
        if isinstance(value, bool):
            raise AuthorityError("operator policy budget values must be finite numbers")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise AuthorityError("operator policy budget values must be finite numbers") from exc
        if not math.isfinite(number) or number <= 0.0:
            raise AuthorityError("operator policy budget values must be finite and positive")
        if number > ceiling:
            raise AuthorityError("operator policy exceeds the deploy budget ceiling")


def _require_validity_window(policy_version: str, *, now: float | None = None) -> None:
    parts = policy_version.split(";")
    if len(parts) != 3 or parts[0] != POLICY_VERSION_PREFIX:
        raise AuthorityError("operator policy version/validity format is invalid")
    fields: dict[str, str] = {}
    for item in parts[1:]:
        key, sep, value = item.partition("=")
        if not sep or key in fields:
            raise AuthorityError("operator policy validity format is invalid")
        fields[key] = value
    if set(fields) != {"issued_at", "not_after"}:
        raise AuthorityError("operator policy validity fields are incomplete")
    try:
        issued_at = int(fields["issued_at"])
        not_after = int(fields["not_after"])
    except (TypeError, ValueError) as exc:
        raise AuthorityError("operator policy validity timestamps are invalid") from exc
    if not_after <= issued_at:
        raise AuthorityError("operator policy validity window is invalid")
    if not_after - issued_at > MAX_POLICY_LIFETIME_SECONDS:
        raise AuthorityError("operator policy validity exceeds the deploy ceiling")

    current = time.time() if now is None else float(now)
    if not math.isfinite(current):
        raise AuthorityError("current time is invalid")
    if issued_at > current + CLOCK_SKEW_SECONDS:
        raise AuthorityError("operator policy is not yet valid")
    if not_after <= current:
        raise AuthorityError("operator policy has expired")


def require_self_maintenance_scope(
    policy: MissionPolicy,
    *,
    now: float | None = None,
) -> None:
    domains = set(policy.allowed_domains)
    capabilities = set(policy.capabilities)
    invariants = set(policy.invariants)

    if policy.mission_id != "g1-self-maintenance":
        raise AuthorityError("operator policy mission id is not G1 self-maintenance")
    if not domains or "maintenance" not in domains:
        raise AuthorityError("operator policy is not scoped to self-maintenance")
    if not capabilities or not any(item.startswith("maintenance.") for item in capabilities):
        raise AuthorityError("operator policy has no self-maintenance capabilities")
    if not domains <= SELF_MAINTENANCE_DOMAINS:
        raise AuthorityError("operator policy exceeds the deploy domain ceiling")
    if not capabilities <= SELF_MAINTENANCE_CAPABILITIES:
        raise AuthorityError("operator policy exceeds the deploy capability ceiling")
    if not REQUIRED_INVARIANTS <= invariants:
        raise AuthorityError("operator policy is missing required safety invariants")
    if policy.standing_mandates:
        raise AuthorityError("self-maintenance policy may not carry standing A3 mandates")
    _require_budget_ceiling(policy)
    _require_validity_window(policy.policy_version, now=now)


def load_operator_policy(
    policy_path: str | Path,
    public_key_path: str | Path,
    *,
    now: float | None = None,
) -> MissionPolicy:
    """Load and strictly require bounded, dated Ed25519 ENFORCE authority."""
    if not HAVE_CRYPTOGRAPHY:
        raise AuthorityError("ed25519 backend unavailable; production authority disabled")

    policy_file = Path(policy_path).expanduser()
    public_file = Path(public_key_path).expanduser()
    _regular_nonsymlink(policy_file, "operator policy")
    _regular_nonsymlink(public_file, "operator public key")

    try:
        policy_bytes = policy_file.read_bytes()
        public_pem = public_file.read_bytes()
    except OSError as exc:
        raise AuthorityError("operator policy material is unavailable") from exc

    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        public_key = serialization.load_pem_public_key(public_pem)
        if not isinstance(public_key, Ed25519PublicKey):
            raise AuthorityError("operator public key is not Ed25519")
    except AuthorityError:
        raise
    except (ValueError, TypeError, ImportError) as exc:
        raise AuthorityError("operator public key is malformed") from exc

    policy = load_policy_bytes(
        policy_bytes,
        public_key=public_key,
        allow_dev_hmac=False,
        source_path=str(policy_file),
    )
    if (
        policy.mode != MODE_ENFORCE
        or not policy.enforced
        or not policy.signed
        or policy.signature_alg != ALG_ED25519
        or policy.dev_only
        or not verified_authority(policy)
    ):
        raise AuthorityError("operator policy did not verify as Ed25519 ENFORCE authority")
    require_self_maintenance_scope(policy, now=now)
    return policy


# ---------------------------------------------------------------------------
# Paper-trading operator policy ceiling (separate envelope, separate key id).
# ---------------------------------------------------------------------------
# The RLR trader lane receives its own externally signed Ed25519 ENFORCE policy
# with its own scope, budgets, and dated validity window. The bundle and
# authority checks live here next to ``self_maintenance_policy_bundle`` so the
# signer and the loader share one module; nothing in this block touches the
# existing self-maintenance helpers, constants, or scope checks above.

TRADING_DOMAINS = frozenset({"alpaca_paper"})
TRADING_CAPABILITIES = frozenset({"alpaca_paper.order"})
TRADING_BUDGET_MAX = {
    "orders_per_day": 100.0,
    "notional_usd_per_day": 50000.0,
}
TRADING_POLICY_VERSION_PREFIX = "rlr-paper-trading-1"


def _trading_version_with_window(issued_at: int, not_after: int) -> str:
    return (
        f"{TRADING_POLICY_VERSION_PREFIX};issued_at={int(issued_at)};"
        f"not_after={int(not_after)}"
    )


def paper_trading_policy_bundle(
    *,
    issued_at: int | float | None = None,
    lifetime_seconds: int | float = DEFAULT_POLICY_LIFETIME_SECONDS,
) -> dict[str, Any]:
    """Return the bounded paper-trading bundle with a dated validity window."""
    start = int(time.time() if issued_at is None else issued_at)
    if isinstance(lifetime_seconds, bool):
        raise AuthorityError("policy lifetime must be finite and positive")
    try:
        lifetime = int(lifetime_seconds)
    except (TypeError, ValueError) as exc:
        raise AuthorityError("policy lifetime must be finite and positive") from exc
    if lifetime <= 0 or lifetime > MAX_POLICY_LIFETIME_SECONDS:
        raise AuthorityError("policy lifetime exceeds the deploy ceiling")
    end = start + lifetime
    return {
        "mission_id": "rlr-paper-trading",
        "objective": (
            "drive the RLR paper-trading operator within bounded budgets, "
            "exercising only the trading domain with the paper.order capability"
        ),
        "invariants": [],
        "allowed_domains": sorted(TRADING_DOMAINS),
        "capabilities": sorted(TRADING_CAPABILITIES),
        "budgets": {
            "orders_per_day": 40.0,
            "notional_usd_per_day": 10000.0,
        },
        "standing_mandates": [],
        "policy_version": _trading_version_with_window(start, end),
    }


def _require_trading_budget_ceiling(policy: MissionPolicy) -> None:
    budgets = dict(policy.budgets)
    if not budgets:
        raise AuthorityError("operator policy has no trading budgets")
    for resource, value in budgets.items():
        if resource not in TRADING_BUDGET_MAX:
            raise AuthorityError("operator policy exceeds the trading budget ceiling")
        if isinstance(value, bool):
            raise AuthorityError("operator policy budget values must be finite numbers")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise AuthorityError("operator policy budget values must be finite numbers") from exc
        if not math.isfinite(number) or number <= 0.0:
            raise AuthorityError("operator policy budget values must be finite and positive")
        if number > TRADING_BUDGET_MAX[resource]:
            raise AuthorityError("operator policy exceeds the trading budget ceiling")


def _require_trading_validity_window(
    policy_version: str,
    *,
    now: float | None = None,
) -> None:
    parts = policy_version.split(";")
    if len(parts) != 3 or parts[0] != TRADING_POLICY_VERSION_PREFIX:
        raise AuthorityError("operator policy version/validity format is invalid")
    fields: dict[str, str] = {}
    for item in parts[1:]:
        key, sep, value = item.partition("=")
        if not sep or key in fields:
            raise AuthorityError("operator policy validity format is invalid")
        fields[key] = value
    if set(fields) != {"issued_at", "not_after"}:
        raise AuthorityError("operator policy validity fields are incomplete")
    try:
        issued_at = int(fields["issued_at"])
        not_after = int(fields["not_after"])
    except (TypeError, ValueError) as exc:
        raise AuthorityError("operator policy validity timestamps are invalid") from exc
    if not_after <= issued_at:
        raise AuthorityError("operator policy validity window is invalid")
    if not_after - issued_at > MAX_POLICY_LIFETIME_SECONDS:
        raise AuthorityError("operator policy validity exceeds the deploy ceiling")

    current = time.time() if now is None else float(now)
    if not math.isfinite(current):
        raise AuthorityError("current time is invalid")
    if issued_at > current + CLOCK_SKEW_SECONDS:
        raise AuthorityError("operator policy is not yet valid")
    if not_after <= current:
        raise AuthorityError("operator policy has expired")


def require_trading_scope(
    policy: MissionPolicy,
    *,
    now: float | None = None,
) -> None:
    domains = set(policy.allowed_domains)
    capabilities = set(policy.capabilities)

    if policy.mission_id != "rlr-paper-trading":
        raise AuthorityError("operator policy mission id is not rlr-paper-trading")
    if domains != set(TRADING_DOMAINS):
        raise AuthorityError("operator policy is not scoped to paper-trading")
    if not capabilities or not capabilities <= set(TRADING_CAPABILITIES):
        raise AuthorityError("operator policy exceeds the trading capability ceiling")
    _require_trading_budget_ceiling(policy)
    _require_trading_validity_window(policy.policy_version, now=now)


def load_trading_policy(
    policy_path: str | Path,
    public_key_path: str | Path,
    *,
    now: float | None = None,
) -> MissionPolicy:
    """Load and strictly require a bounded, dated Ed25519 ENFORCE trading policy."""
    if not HAVE_CRYPTOGRAPHY:
        raise AuthorityError("ed25519 backend unavailable; production authority disabled")

    policy_file = Path(policy_path).expanduser()
    public_file = Path(public_key_path).expanduser()
    _regular_nonsymlink(policy_file, "operator policy")
    _regular_nonsymlink(public_file, "operator public key")

    try:
        policy_bytes = policy_file.read_bytes()
        public_pem = public_file.read_bytes()
    except OSError as exc:
        raise AuthorityError("operator policy material is unavailable") from exc

    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        public_key = serialization.load_pem_public_key(public_pem)
        if not isinstance(public_key, Ed25519PublicKey):
            raise AuthorityError("operator public key is not Ed25519")
    except AuthorityError:
        raise
    except (ValueError, TypeError, ImportError) as exc:
        raise AuthorityError("operator public key is malformed") from exc

    policy = load_policy_bytes(
        policy_bytes,
        public_key=public_key,
        allow_dev_hmac=False,
        source_path=str(policy_file),
    )
    if (
        policy.mode != MODE_ENFORCE
        or not policy.enforced
        or not policy.signed
        or policy.signature_alg != ALG_ED25519
        or policy.dev_only
        or not verified_authority(policy)
    ):
        raise AuthorityError("operator policy did not verify as Ed25519 ENFORCE authority")
    require_trading_scope(policy, now=now)
    return policy

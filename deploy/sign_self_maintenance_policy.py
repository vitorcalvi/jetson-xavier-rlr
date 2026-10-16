#!/usr/bin/env python3
"""Sign the bounded operator policy on a trusted operator host.

Private Ed25519 material lives only in the chmod-0600 operator env file. The
Jetson receives only the signed envelope and public PEM. No private bytes are
printed or copied by this tool.

The default ``--domain self-maintenance`` signs the G1 self-maintenance bundle
that the existing operator runtime already trusts. ``--domain trading`` instead
signs the RLR paper-trading bundle with its own scope, budgets, and dated
validity window; the resulting envelope has key id ``trading-operator`` and is
intended for the trader lane, not the self-maintenance kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import time
from pathlib import Path

from reflective_loop_runtime.authority import (
    DEFAULT_POLICY_LIFETIME_SECONDS,
    MAX_POLICY_LIFETIME_SECONDS,
    paper_trading_policy_bundle,
    self_maintenance_policy_bundle,
)
from reflective_loop_runtime.kernel.policy import HAVE_CRYPTOGRAPHY, sign_envelope

_PRIVATE_VAR = "RLR_OPERATOR_ED25519_PRIVATE_KEY_HEX"

_DOMAIN_SELF_MAINTENANCE = "self-maintenance"
_DOMAIN_TRADING = "trading"
_DOMAIN_CHOICES = (_DOMAIN_SELF_MAINTENANCE, _DOMAIN_TRADING)


def _reject_symlink_chain(path: Path) -> None:
    path = path.expanduser()
    current = path
    while True:
        if current.is_symlink():
            raise SystemExit(f"refusing symlink path: {path}")
        if current == current.parent:
            break
        current = current.parent


def _canonical(path: Path) -> str:
    return str(path.expanduser().resolve(strict=False))


def _require_distinct(*paths: Path) -> None:
    canonical = [_canonical(path) for path in paths]
    if len(canonical) != len(set(canonical)):
        raise SystemExit("output/private-key paths must be distinct")


def _read_env(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise SystemExit("operator key env must be a regular non-symlink file")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o600:
        raise SystemExit("refusing operator key env with permissions other than 0600")
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if sep:
            values[key.strip()] = value.strip()
    return values


def _flags(base: int) -> int:
    return base | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)


def _write_exclusive_0600(path: Path, payload: bytes) -> None:
    path = path.expanduser()
    _reject_symlink_chain(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), _flags(os.O_WRONLY | os.O_CREAT | os.O_EXCL), 0o600)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.chmod(path, 0o600)


def _atomic_write_0600(path: Path, payload: bytes) -> None:
    path = path.expanduser()
    _reject_symlink_chain(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise SystemExit(f"refusing unsafe output target: {path}")
    tmp = path.parent / f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    fd = os.open(str(tmp), _flags(os.O_WRONLY | os.O_CREAT | os.O_EXCL), 0o600)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    os.chmod(path, 0o600)


def _bundle_for_domain(domain: str, *, issued_at: int, lifetime_seconds: int) -> tuple[dict, str]:
    if domain == _DOMAIN_SELF_MAINTENANCE:
        return (
            self_maintenance_policy_bundle(
                issued_at=issued_at,
                lifetime_seconds=lifetime_seconds,
            ),
            "g1-operator",
        )
    if domain == _DOMAIN_TRADING:
        return (
            paper_trading_policy_bundle(
                issued_at=issued_at,
                lifetime_seconds=lifetime_seconds,
            ),
            "trading-operator",
        )
    raise SystemExit(f"unknown domain: {domain!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator-key-env", required=True)
    parser.add_argument("--public-key-out", required=True)
    parser.add_argument("--policy-out", required=True)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument(
        "--domain",
        choices=_DOMAIN_CHOICES,
        default=_DOMAIN_SELF_MAINTENANCE,
        help=(
            "policy domain to sign; 'self-maintenance' (default) produces the "
            "G1 self-maintenance bundle, 'trading' produces the RLR "
            "paper-trading bundle"
        ),
    )
    parser.add_argument(
        "--valid-hours",
        type=float,
        default=DEFAULT_POLICY_LIFETIME_SECONDS / 3600.0,
        help="signed policy lifetime; maximum 30 days",
    )
    args = parser.parse_args(argv)

    if not HAVE_CRYPTOGRAPHY:
        raise SystemExit("cryptography is required; refusing dev HMAC fallback")
    if not (0.0 < args.valid_hours <= MAX_POLICY_LIFETIME_SECONDS / 3600.0):
        raise SystemExit("valid-hours must be positive and within the deploy ceiling")

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key_env = Path(args.operator_key_env).expanduser()
    public_out = Path(args.public_key_out).expanduser()
    policy_out = Path(args.policy_out).expanduser()
    _require_distinct(key_env, public_out, policy_out)
    for path in (key_env, public_out, policy_out):
        _reject_symlink_chain(path)

    if key_env.exists():
        private_hex = _read_env(key_env).get(_PRIVATE_VAR, "")
        if not private_hex:
            raise SystemExit("operator key env does not contain the Ed25519 private key")
        try:
            private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_hex))
        except (ValueError, TypeError) as exc:
            raise SystemExit("operator private key is malformed") from exc
    else:
        if not args.generate:
            raise SystemExit("operator key env missing; use --generate only on the trusted host")
        private = Ed25519PrivateKey.generate()
        raw_private = private.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        _write_exclusive_0600(
            key_env,
            f"{_PRIVATE_VAR}={raw_private.hex()}\n".encode(),
        )

    public_pem = private.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    bundle, key_id = _bundle_for_domain(
        args.domain,
        issued_at=int(time.time()),
        lifetime_seconds=int(args.valid_hours * 3600),
    )
    envelope = sign_envelope(
        bundle,
        private_key=private,
        key_id=key_id,
    )
    _atomic_write_0600(public_out, public_pem)
    _atomic_write_0600(
        policy_out,
        (json.dumps(envelope, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )

    fingerprint = hashlib.sha256(public_pem).hexdigest()[:16]
    print(f"policy_written={policy_out}")
    print(f"public_key_written={public_out}")
    print(f"public_fingerprint={fingerprint}")
    print(f"domain={args.domain}")
    print(f"key_id={key_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

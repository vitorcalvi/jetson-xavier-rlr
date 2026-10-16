"""Signed, immutable mission policy bundles for the reflective-loop kernel.

A *mission policy* is the operator's standing authority over the loop: what the
mission is, which domains it may observe or act in, which capabilities it may
invoke, which budgets bound it, and which mandates always apply. It is delivered
as a canonical-JSON bundle plus a detached signature, and is exposed to the rest
of the runtime as a frozen, read-only object.

Trust model
-----------
* A bundle is only *enforced* when its signature verifies against operator key
  material. Every other outcome -- an unreadable or missing file, malformed
  JSON, a missing or invalid signature, or a bundle that does not match the
  canonical field schema -- fails closed to ``OBSERVATION_ONLY``.
* There is deliberately **no write API**. This module can read, verify, and (as
  a pure function, for authoring and tests) sign an envelope; it never mutates
  or persists policy state.
* The HMAC-SHA256 path is a **dev-only** fallback for environments without the
  optional ``cryptography`` dependency. HMAC is a symmetric MAC, not a public
  key signature: anyone holding the shared secret can forge bundles. Policies
  verified this way are flagged ``dev_only=True`` and carry no real authority.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

try:  # pragma: no cover - environment dependent
    from cryptography.exceptions import InvalidSignature as _InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )

    HAVE_CRYPTOGRAPHY = True
except Exception:  # pragma: no cover - optional dependency absent
    _InvalidSignature = None
    Ed25519PrivateKey = None  # type: ignore[assignment]
    Ed25519PublicKey = None  # type: ignore[assignment]
    HAVE_CRYPTOGRAPHY = False

# Enforcement modes. ``ENFORCE`` carries operator authority; anything that cannot
# be fully trusted degrades to ``OBSERVATION_ONLY`` (the loop may observe but its
# mission-specific authority is withheld).
MODE_ENFORCE = "ENFORCE"
MODE_OBSERVATION_ONLY = "OBSERVATION_ONLY"

# The exact, canonical bundle schema. A signed bundle must carry precisely these
# fields; extra or missing keys are a schema violation and fail closed.
BUNDLE_FIELDS = (
    "mission_id",
    "objective",
    "invariants",
    "allowed_domains",
    "capabilities",
    "budgets",
    "standing_mandates",
    "policy_version",
)
TEXT_FIELDS = ("mission_id", "objective", "policy_version")
LIST_FIELDS = ("invariants", "allowed_domains", "capabilities", "standing_mandates")

ENVELOPE_BUNDLE_KEY = "bundle"
ENVELOPE_SIGNATURE_KEY = "signature"
ALG_ED25519 = "ed25519"
ALG_HMAC_SHA256 = "hmac-sha256"

# HMAC is documented as a dev-only convenience; it is never a production secret.
HMAC_IS_DEV_ONLY = True


class PolicyError(ValueError):
    """Raised when key material or a bundle cannot be interpreted safely."""


# ---------------------------------------------------------------------------
# Canonical encoding
# ---------------------------------------------------------------------------


def canonical_json(value: Any) -> str:
    """Deterministic JSON: sorted keys, compact separators, UTF-8 friendly."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_bytes(value: Any) -> bytes:
    """Canonical UTF-8 bytes used as the signed payload."""
    return canonical_json(value).encode("utf-8")


def normalize_bundle(bundle: Any) -> dict[str, Any]:
    """Validate and normalise a bundle to its canonical, signable form.

    Returns a fresh dict with text, list, and budget fields coerced to canonical
    types. Raises ``PolicyError`` for anything off-schema so callers can fail
    closed.
    """
    if not isinstance(bundle, Mapping):
        raise PolicyError("bundle must be a JSON object")
    keys = set(bundle)
    required = set(BUNDLE_FIELDS)
    if keys != required:
        missing = sorted(required - keys)
        extra = sorted(keys - required)
        raise PolicyError(f"bundle fields do not match schema (missing={missing}, extra={extra})")

    clean: dict[str, Any] = {}
    for name in TEXT_FIELDS:
        value = bundle[name]
        if not isinstance(value, str) or not value.strip():
            raise PolicyError(f"{name} must be non-empty text")
        clean[name] = value.strip()
    for name in LIST_FIELDS:
        value = bundle[name]
        if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
            raise PolicyError(f"{name} must be a list of strings")
        items: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise PolicyError(f"{name} entries must be strings")
            items.append(item)
        clean[name] = items

    budgets = bundle["budgets"]
    if not isinstance(budgets, Mapping):
        raise PolicyError("budgets must be an object")
    clean_budgets: dict[str, Any] = {}
    for key, value in budgets.items():
        if not isinstance(key, str) or not key:
            raise PolicyError("budget keys must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise PolicyError(f"budget {key!r} must be a number or string")
        if isinstance(value, float) and not math.isfinite(value):
            raise PolicyError(f"budget {key!r} must be a finite number")
        clean_budgets[key] = value
    clean["budgets"] = clean_budgets
    return clean


# ---------------------------------------------------------------------------
# Signature verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Verification:
    """Outcome of checking an envelope's detached signature."""

    valid: bool
    reason: str = ""
    alg: str | None = None
    dev_only: bool = False


def _secret_bytes(secret: Any) -> bytes:
    if isinstance(secret, str):
        return secret.encode("utf-8")
    if isinstance(secret, (bytes, bytearray)):
        return bytes(secret)
    raise PolicyError("secret must be str or bytes")


def _hmac_signature(secret: Any, payload: bytes) -> str:
    return _hmac.new(_secret_bytes(secret), payload, hashlib.sha256).hexdigest()


def _key_bytes(raw: Any, size: int, kind: str) -> bytes:
    if isinstance(raw, str):
        try:
            data = bytes.fromhex(raw)
        except ValueError as exc:
            raise PolicyError(f"ed25519 {kind} key must be hexadecimal") from exc
    elif isinstance(raw, (bytes, bytearray)):
        data = bytes(raw)
    else:
        raise PolicyError(f"unsupported ed25519 {kind} key type")
    if len(data) != size:
        raise PolicyError(f"ed25519 {kind} key must be {size} bytes")
    return data


def _ed25519_private_key(raw: Any) -> Any:
    if HAVE_CRYPTOGRAPHY and isinstance(raw, Ed25519PrivateKey):
        return raw
    if not HAVE_CRYPTOGRAPHY:  # pragma: no cover - environment dependent
        raise PolicyError("ed25519 requires the optional 'cryptography' dependency")
    return Ed25519PrivateKey.from_private_bytes(_key_bytes(raw, 32, "private"))


def _ed25519_public_key(raw: Any) -> Any:
    if HAVE_CRYPTOGRAPHY and isinstance(raw, Ed25519PublicKey):
        return raw
    if HAVE_CRYPTOGRAPHY and isinstance(raw, Ed25519PrivateKey):
        return raw.public_key()
    if not HAVE_CRYPTOGRAPHY:  # pragma: no cover - environment dependent
        raise PolicyError("ed25519 requires the optional 'cryptography' dependency")
    return Ed25519PublicKey.from_public_bytes(_key_bytes(raw, 32, "public"))


class KeyedHmacSigner:
    """HMAC-SHA256 whose key lives only inside opaque ``hashlib`` state objects.

    The key is absorbed into the inner/outer padded SHA-256 states at
    construction and the Python-visible key is discarded, so a secret never
    appears in an attribute, a closure cell, a bound method's ``__self__``,
    ``__dict__`` or the GC referent graph: ``hashlib`` exposes no referents for
    its buffered state, so a traversal from a model-facing object cannot recover
    it. ``sign``/``verify`` reproduce the exact HMAC-SHA256 construction
    (``H((key ^ opad) || H((key ^ ipad) || msg))``), so digests are unchanged.
    """

    __slots__ = ("_inner", "_outer")

    _BLOCK = 64

    def __init__(self, key: bytes):
        raw = bytes(key)
        if len(raw) > self._BLOCK:
            raw = hashlib.sha256(raw).digest()
        padded = raw + b"\x00" * (self._BLOCK - len(raw))
        inner = hashlib.sha256()
        outer = hashlib.sha256()
        inner.update(bytes(byte ^ 0x36 for byte in padded))
        outer.update(bytes(byte ^ 0x5C for byte in padded))
        self._inner = inner
        self._outer = outer

    def sign(self, payload: bytes) -> str:
        inner = self._inner.copy()
        inner.update(payload)
        outer = self._outer.copy()
        outer.update(inner.digest())
        return outer.hexdigest()

    def verify(self, payload: bytes, mac: Any) -> bool:
        if not isinstance(mac, str) or not mac:
            return False
        try:
            return _hmac.compare_digest(self.sign(payload), mac)
        except TypeError:
            # compare_digest rejects non-ASCII strings; a malformed MAC is a
            # verification failure, never an exception that skips later checks.
            return False


def verify_envelope(
    envelope: Any,
    *,
    public_key: Any = None,
    secret: Any = None,
    allow_dev_hmac: bool = False,
) -> Verification:
    """Check an envelope's signature without mutating anything.

    Never raises: a malformed envelope, missing key material, or a signature
    mismatch is reported as ``Verification(valid=False, ...)``.

    Authority is signature-only by default. The symmetric HMAC-SHA256 path is a
    *dev-only* convenience: it grants no authority unless the caller opts in with
    ``allow_dev_hmac=True``, and even then the resulting policy is flagged
    ``dev_only``. This keeps a leaked dev shared secret from equalling operator
    authority.
    """
    if not isinstance(envelope, Mapping):
        return Verification(False, "envelope must be a JSON object")
    signature = envelope.get(ENVELOPE_SIGNATURE_KEY)
    if not isinstance(signature, Mapping):
        return Verification(False, "signature is missing")
    alg = signature.get("alg")
    value = signature.get("value")
    dev_only = bool(signature.get("dev_only", alg == ALG_HMAC_SHA256))
    if not isinstance(value, str) or not value:
        return Verification(False, "signature value is missing", alg, dev_only)

    try:
        payload = canonical_bytes(normalize_bundle(envelope.get(ENVELOPE_BUNDLE_KEY)))
    except PolicyError as exc:
        return Verification(False, f"invalid bundle: {exc}", alg, dev_only)

    if alg == ALG_ED25519:
        if not HAVE_CRYPTOGRAPHY:
            return Verification(False, "ed25519 unavailable (cryptography not installed)", alg)
        if public_key is None:
            return Verification(False, "ed25519 verification requires a public key", alg)
        try:
            key = _ed25519_public_key(public_key)
            raw_signature = bytes.fromhex(value)
            key.verify(raw_signature, payload)
        except PolicyError as exc:
            return Verification(False, f"ed25519 verification error: {exc}", alg)
        except ValueError as exc:
            return Verification(False, f"ed25519 signature is malformed: {exc}", alg)
        except _InvalidSignature:
            return Verification(False, "ed25519 signature mismatch", alg)
        return Verification(True, "", alg)

    if alg == ALG_HMAC_SHA256:
        if not allow_dev_hmac:
            return Verification(
                False,
                "dev-only HMAC authority is disabled; provide an operator public key "
                "(or opt in with allow_dev_hmac)",
                alg,
                True,
            )
        if secret is None:
            return Verification(False, "hmac verification requires a secret", alg, True)
        try:
            expected = _hmac_signature(secret, payload)
        except PolicyError as exc:
            return Verification(False, f"hmac verification error: {exc}", alg, True)
        if not _hmac.compare_digest(expected, value):
            return Verification(False, "hmac signature mismatch", alg, True)
        return Verification(True, "", alg, True)

    return Verification(False, f"unknown signature algorithm {alg!r}", alg, dev_only)


# ---------------------------------------------------------------------------
# Signing (pure authoring/test helper; not a store write path)
# ---------------------------------------------------------------------------


def sign_envelope(
    bundle: Mapping[str, Any],
    *,
    private_key: Any = None,
    secret: Any = None,
    key_id: str = "",
) -> dict[str, Any]:
    """Build a signed envelope for ``bundle`` as a pure function.

    Prefers ed25519 when a ``private_key`` is supplied and ``cryptography`` is
    importable; otherwise signs with the dev-only HMAC-SHA256 fallback when a
    ``secret`` is supplied. The returned envelope is plain, serialisable data —
    writing it anywhere is the caller's concern; this module has no write API.
    """
    clean = normalize_bundle(bundle)
    payload = canonical_bytes(clean)
    if private_key is not None:
        if not HAVE_CRYPTOGRAPHY:  # pragma: no cover - environment dependent
            raise PolicyError("ed25519 signing requires the optional 'cryptography' dependency")
        signature = _ed25519_private_key(private_key).sign(payload)
        block = {
            "alg": ALG_ED25519,
            "key_id": str(key_id),
            "value": signature.hex(),
            "dev_only": False,
        }
    elif secret is not None:
        block = {
            "alg": ALG_HMAC_SHA256,
            "key_id": str(key_id),
            "value": _hmac_signature(secret, payload),
            "dev_only": HMAC_IS_DEV_ONLY,
        }
    else:
        raise PolicyError("signing requires a private_key (ed25519) or a secret (hmac)")
    return {ENVELOPE_BUNDLE_KEY: clean, ENVELOPE_SIGNATURE_KEY: block}


# ---------------------------------------------------------------------------
# Immutable policy object
# ---------------------------------------------------------------------------


# An opaque, module-private marker. Only the verified loader attaches it, so a
# policy assembled any other way -- including a hand-built dataclass that sets
# ``mode``/``signed`` directly -- can never present itself as authoritative.
_AUTHORITY_SENTINEL = object()


def _authority_content_digest(policy: MissionPolicy) -> str:
    """SHA-256 over the canonical signed fields of ``policy`` as they stand now."""
    material = {
        "mission_id": policy.mission_id,
        "objective": policy.objective,
        "invariants": list(policy.invariants),
        "allowed_domains": list(policy.allowed_domains),
        "capabilities": list(policy.capabilities),
        "budgets": dict(policy.budgets),
        "standing_mandates": list(policy.standing_mandates),
        "policy_version": policy.policy_version,
    }
    return hashlib.sha256(canonical_bytes(material)).hexdigest()


def _mark_verified_authority(policy: MissionPolicy) -> None:
    """Attach the provenance token to a policy whose signature just verified.

    The token is not a dataclass field, so it cannot be supplied through the
    ordinary constructor, and the content digest binds it to the exact bundle
    bytes that were verified -- editing any authority-bearing field afterwards
    invalidates it.
    """
    object.__setattr__(policy, "_authority", _AUTHORITY_SENTINEL)
    object.__setattr__(policy, "_authority_digest", _authority_content_digest(policy))


def verified_authority(policy: Any) -> bool:
    """Whether ``policy`` carries a signature-verification provenance token.

    Returns ``False`` for anything that was not produced by verifying an
    operator signature over the canonical bundle bytes: a directly constructed
    ``MissionPolicy`` (even one claiming ``ENFORCE``/``signed``), a
    ``dataclasses.replace``d copy, or a policy whose fields were mutated after
    verification.
    """
    if not isinstance(policy, MissionPolicy):
        return False
    if getattr(policy, "_authority", None) is not _AUTHORITY_SENTINEL:
        return False
    try:
        digest = _authority_content_digest(policy)
    except Exception:  # noqa: BLE001 - an unreadable policy has no authority
        return False
    return getattr(policy, "_authority_digest", None) == digest


@dataclass(frozen=True)
class MissionPolicy:
    """A frozen, read-only view of a mission policy bundle.

    Nested collections are also immutable (tuples and a read-only mapping), so a
    loaded policy cannot be altered after construction.
    """

    mission_id: str = ""
    objective: str = ""
    invariants: tuple[str, ...] = ()
    allowed_domains: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    budgets: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    standing_mandates: tuple[str, ...] = ()
    policy_version: str = ""
    mode: str = MODE_OBSERVATION_ONLY
    signed: bool = False
    signature_alg: str | None = None
    dev_only: bool = False
    reason: str = ""
    source_path: str | None = None

    @classmethod
    def from_bundle(
        cls,
        clean: Mapping[str, Any],
        *,
        mode: str,
        verification: Verification | None = None,
        reason: str = "",
        source_path: str | None = None,
    ) -> MissionPolicy:
        return cls(
            mission_id=clean["mission_id"],
            objective=clean["objective"],
            invariants=tuple(clean["invariants"]),
            allowed_domains=tuple(clean["allowed_domains"]),
            capabilities=tuple(clean["capabilities"]),
            budgets=MappingProxyType(dict(clean["budgets"])),
            standing_mandates=tuple(clean["standing_mandates"]),
            policy_version=clean["policy_version"],
            mode=mode,
            signed=bool(verification and verification.valid),
            signature_alg=verification.alg if verification else None,
            dev_only=bool(verification and verification.dev_only),
            reason=reason,
            source_path=source_path,
        )

    @classmethod
    def observation_only(cls, reason: str = "", *, source_path: str | None = None) -> MissionPolicy:
        """Build the fail-closed policy used for any untrusted input."""
        return cls(mode=MODE_OBSERVATION_ONLY, reason=reason, source_path=source_path)

    @property
    def enforced(self) -> bool:
        return self.mode == MODE_ENFORCE

    @property
    def digest(self) -> str:
        """A stable SHA-256 over the whole authority-bearing policy.

        Covers the canonical bundle *and* the enforcement mode, so a permit can
        bind the exact authority it was issued under: swapping in a different
        policy with the same ``policy_version`` changes the digest and the permit
        no longer verifies.
        """
        material = {
            "mode": self.mode,
            "mission_id": self.mission_id,
            "objective": self.objective,
            "invariants": list(self.invariants),
            "allowed_domains": list(self.allowed_domains),
            "capabilities": list(self.capabilities),
            "budgets": dict(self.budgets),
            "standing_mandates": list(self.standing_mandates),
            "policy_version": self.policy_version,
        }
        return hashlib.sha256(canonical_json(material).encode("utf-8")).hexdigest()

    def allows_domain(self, tool: str) -> bool:
        """Whether the policy names ``tool`` as an allowed domain.

        An empty ``allowed_domains`` list imposes no restriction (the operator
        declined to scope domains); a non-empty list is a closed set.
        """
        return not self.allowed_domains or tool in self.allowed_domains

    def allows_capability(self, capability: str, tool: str = "", op: str = "") -> bool:
        """Whether the policy grants ``capability`` for ``tool.op``.

        An empty ``capabilities`` list imposes no restriction. Otherwise a grant
        may be the abstract capability, the tool name, or the exact ``tool.op``.
        """
        if not self.capabilities:
            return True
        candidates = {capability}
        if tool:
            candidates.add(tool)
            if op:
                candidates.add(f"{tool}.{op}")
        return any(candidate in self.capabilities for candidate in candidates if candidate)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, mutable copy for logging/inspection only."""
        return {
            "mission_id": self.mission_id,
            "objective": self.objective,
            "invariants": list(self.invariants),
            "allowed_domains": list(self.allowed_domains),
            "capabilities": list(self.capabilities),
            "budgets": dict(self.budgets),
            "standing_mandates": list(self.standing_mandates),
            "policy_version": self.policy_version,
            "mode": self.mode,
            "signed": self.signed,
            "signature_alg": self.signature_alg,
            "dev_only": self.dev_only,
        }


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_policy_bytes(
    data: bytes,
    *,
    public_key: Any = None,
    secret: Any = None,
    allow_dev_hmac: bool = False,
    source_path: str | None = None,
) -> MissionPolicy:
    """Load a policy from raw bytes, failing closed to ``OBSERVATION_ONLY``."""
    try:
        text = data.decode("utf-8")
    except (AttributeError, UnicodeDecodeError) as exc:
        return MissionPolicy.observation_only(f"policy is not valid UTF-8: {exc}",
                                              source_path=source_path)
    try:
        envelope = json.loads(text)
    except ValueError as exc:
        return MissionPolicy.observation_only(f"policy is not valid JSON: {exc}",
                                              source_path=source_path)
    if not isinstance(envelope, Mapping):
        return MissionPolicy.observation_only("policy envelope must be a JSON object",
                                              source_path=source_path)

    raw_bundle = envelope.get(ENVELOPE_BUNDLE_KEY)
    if not isinstance(raw_bundle, Mapping):
        return MissionPolicy.observation_only("policy envelope is missing its bundle",
                                              source_path=source_path)
    try:
        clean = normalize_bundle(raw_bundle)
    except PolicyError as exc:
        return MissionPolicy.observation_only(f"invalid bundle: {exc}", source_path=source_path)

    verification = verify_envelope(
        envelope, public_key=public_key, secret=secret, allow_dev_hmac=allow_dev_hmac
    )
    if not verification.valid:
        reason = verification.reason or "signature could not be verified"
        return MissionPolicy.from_bundle(
            clean, mode=MODE_OBSERVATION_ONLY, verification=verification,
            reason=reason, source_path=source_path,
        )
    # Authority is granted here, and only here: a successful signature check
    # over the canonical bundle bytes attaches the provenance token the permit
    # gate requires. ``from_bundle`` itself never grants authority.
    policy = MissionPolicy.from_bundle(
        clean, mode=MODE_ENFORCE, verification=verification, source_path=source_path,
    )
    _mark_verified_authority(policy)
    return policy


def load_policy(
    path: str | os.PathLike[str],
    *,
    public_key: Any = None,
    secret: Any = None,
    allow_dev_hmac: bool = False,
) -> MissionPolicy:
    """Read and verify a policy file.

    A missing/unreadable file, malformed content, or a bad/missing signature all
    yield a frozen policy in ``OBSERVATION_ONLY`` mode rather than raising.
    """
    source = os.fspath(path)
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        return MissionPolicy.observation_only(f"policy file is unreadable: {exc}",
                                              source_path=source)
    return load_policy_bytes(
        data, public_key=public_key, secret=secret,
        allow_dev_hmac=allow_dev_hmac, source_path=source,
    )


class MissionPolicyStore:
    """Read-only policy store bound to operator verification key material.

    The store can only *load* policies; there is intentionally no save/update
    path, so runtime code cannot rewrite the authority it is subject to.
    """

    def __init__(
        self,
        *,
        public_key: Any = None,
        secret: Any = None,
        allow_dev_hmac: bool = False,
    ):
        self._public_key = public_key
        self._secret = secret
        self._allow_dev_hmac = bool(allow_dev_hmac)

    def load(self, path: str | os.PathLike[str]) -> MissionPolicy:
        return load_policy(
            path, public_key=self._public_key, secret=self._secret,
            allow_dev_hmac=self._allow_dev_hmac,
        )

    def load_bytes(self, data: bytes) -> MissionPolicy:
        return load_policy_bytes(
            data, public_key=self._public_key, secret=self._secret,
            allow_dev_hmac=self._allow_dev_hmac,
        )


__all__ = [
    "ALG_ED25519",
    "ALG_HMAC_SHA256",
    "BUNDLE_FIELDS",
    "ENVELOPE_BUNDLE_KEY",
    "ENVELOPE_SIGNATURE_KEY",
    "HAVE_CRYPTOGRAPHY",
    "HMAC_IS_DEV_ONLY",
    "KeyedHmacSigner",
    "MODE_ENFORCE",
    "MODE_OBSERVATION_ONLY",
    "MissionPolicy",
    "MissionPolicyStore",
    "PolicyError",
    "Verification",
    "canonical_bytes",
    "canonical_json",
    "load_policy",
    "load_policy_bytes",
    "normalize_bundle",
    "sign_envelope",
    "verified_authority",
    "verify_envelope",
]

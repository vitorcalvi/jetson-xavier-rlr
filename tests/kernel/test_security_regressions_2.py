"""Second security-regression pass over the K11 kernel review findings.

Every test here must fail against the pre-fix kernel and pass against the
hardened one. The findings covered are:

* F10b -- policy authority may only come from verifying the operator signature
  over the canonical bundle bytes; a self-declared ``ENFORCE``/``signed`` policy
  is refused, and editing a verified policy's fields revokes its authority.
* F11  -- the HMAC signing key must not be recoverable from the model-facing
  ``PermitService`` through closures, bound methods, ``__dict__``,
  ``__closure__`` cell contents or a ``gc.get_referents`` walk.
* N1   -- a model reflection may only reach memory through the permit-gated
  ``memory.remember`` path, never a direct store write.
* N2   -- every manifest cost unit must be a budget resource; an unknown unit is
  refused at construction.
* N3   -- the runtime ToolBroker is built with a persistent, keyed audit log.
* F3   -- capability names must match the manifest operation exactly; malformed
  names without a dot are refused.
* F17  -- the sandboxed shell path writes its audit row before the effect and
  fails closed when that write fails.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import types
from pathlib import Path

import pytest

from reflective_loop_runtime.kernel.audit import AuditLog
from reflective_loop_runtime.kernel.budget import RESOURCES, BudgetExceeded, BudgetService
from reflective_loop_runtime.kernel.manifest import (
    DEFAULT_MANIFEST,
    CostModel,
    ManifestError,
)
from reflective_loop_runtime.kernel.permit import (
    PermitDenied,
    PermitIssuer,
    PermitService,
)
from reflective_loop_runtime.kernel.policy import (
    MODE_ENFORCE,
    MissionPolicy,
    MissionPolicyStore,
    sign_envelope,
)
from reflective_loop_runtime.kernel.toolbroker import ToolBroker, broker_manifest

DEV_SECRET = "dev-only-shared-secret"
ISSUER_KEY = b"kernel-regressions-2-issuer-key-0123456789"


def _verified_policy(
    tmp_path: Path, *, version: str = "1.0.0", budgets=None
) -> MissionPolicy:
    bundle = {
        "mission_id": "mission-r2",
        "objective": "second regression pass",
        "invariants": ["never_shell_out_unsandboxed"],
        "allowed_domains": [],
        "capabilities": [],
        "budgets": dict(budgets or {}),
        "standing_mandates": [],
        "policy_version": version,
    }
    envelope = sign_envelope(bundle, secret=DEV_SECRET)
    path = tmp_path / f"policy-{version}.json"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    return MissionPolicyStore(secret=DEV_SECRET, allow_dev_hmac=True).load(path)


# ---------------------------------------------------------------------------
# F10b -- authority is signature provenance, not a self-declared flag
# ---------------------------------------------------------------------------


def test_f10b_hand_built_enforce_policy_is_refused(tmp_path):
    forged = MissionPolicy(mode=MODE_ENFORCE, signed=True)
    # The dataclass still *claims* everything a naive check would look for.
    assert forged.enforced is True
    assert forged.signed is True

    with pytest.raises(PermitDenied, match="signature"):
        PermitService(PermitIssuer(ISSUER_KEY), forged)

    service = PermitService(PermitIssuer(ISSUER_KEY), _verified_policy(tmp_path))
    with pytest.raises(PermitDenied, match="signature"):
        service.set_policy(forged)
    # The original verified authority is untouched and still mints permits.
    assert service.policy.enforced is True


def test_f10b_direct_construction_and_replace_lose_authority(tmp_path):
    verified = _verified_policy(tmp_path)
    replaced = dataclasses.replace(verified, mode=MODE_ENFORCE, signed=True)
    assert replaced.enforced is True
    with pytest.raises(PermitDenied, match="signature"):
        PermitService(PermitIssuer(ISSUER_KEY), replaced)


def test_f10b_post_verification_field_edit_loses_authority(tmp_path):
    verified = _verified_policy(tmp_path)
    # A frozen dataclass still yields to object.__setattr__; the content digest
    # must notice that the authority-bearing fields no longer match the signed
    # bytes.
    object.__setattr__(verified, "capabilities", ("memory.forget",))
    with pytest.raises(PermitDenied, match="signature"):
        PermitService(PermitIssuer(ISSUER_KEY), verified)


# ---------------------------------------------------------------------------
# F11 -- the signing key is unreachable from the permit service
# ---------------------------------------------------------------------------


def _key_reachable(root: object, key: bytes, *, max_depth: int = 6) -> bool:
    """Bounded referent walk that prunes module objects.

    Module globals are not part of the graph the runtime hands to a model, and
    walking them would stray into this test module's own constants.
    """
    seen: set[int] = set()
    stack: list[tuple[object, int]] = [(root, 0)]
    while stack:
        obj, depth = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, (bytes, bytearray)) and key in bytes(obj):
            return True
        if depth >= max_depth or isinstance(obj, types.ModuleType):
            continue
        try:
            refs = gc.get_referents(obj)
        except Exception:  # noqa: BLE001 - non-traversable objects expose nothing
            continue
        for ref in refs:
            stack.append((ref, depth + 1))
    return False


def test_f11_signing_key_is_unreachable_from_the_service(tmp_path):
    key = b"local-only-issuer-key-0123456789abcdef"
    service = PermitService(PermitIssuer(key), _verified_policy(tmp_path))

    # Bound-method path.
    assert getattr(service._sign, "__self__", None) is None
    assert getattr(service._verify, "__self__", None) is None

    # Closure-cell path: cells must hold the keyed signer, never the key bytes.
    for callable_ in (service._sign, service._verify):
        for cell in getattr(callable_, "__closure__", None) or ():
            try:
                contents = cell.cell_contents
            except ValueError:  # pragma: no cover - empty cell
                continue
            assert not (isinstance(contents, (bytes, bytearray)) and key in bytes(contents))

    # __dict__ / repr / gc traversal paths.
    assert key.decode() not in repr(vars(service))
    assert not _key_reachable(service, key)
    assert not _key_reachable(service._sign, key)
    assert not _key_reachable(service._verify, key)

    # The service still signs and verifies.
    permit = service.issue(
        tool="state", op="set_load_target", args={"load": 0.5}, goal_id="g-f11"
    )
    service.verify_and_consume(permit, "state", {"load": 0.5}, op="set_load_target")


# ---------------------------------------------------------------------------
# N1 -- reflections reach memory only through the permit gate
# ---------------------------------------------------------------------------


class _ReflectDecider:
    name = "r2-reflect"

    def score(self, candidates, state, *, context=""):
        from reflective_loop_runtime.contracts import ScoredDecision

        ids = [candidate.id for candidate in candidates]
        target = "reflect" if "reflect" in ids else ids[0]
        probabilities = {candidate: 0.0 for candidate in ids}
        probabilities[target] = 1.0
        return ScoredDecision(choice=target, confidence=1.0, probabilities=probabilities)


class _ReflectBrain:
    name = "r2-reflect-brain"

    def __init__(self, summary: str):
        self.summary = summary

    def reflect(self, *, state, context=""):
        return self.summary

    def advise(self, *, state, candidates, previous, context=""):
        return None


def _warm_sample():
    return {
        "temperature_c": 50.0,
        "zones_c": [50.0],
        "ram_used_mib": 1024.0,
        "ram_total_mib": 4096.0,
        "sampled_monotonic": None,
        "age_seconds": 0.0,
    }


def test_n1_reflection_memory_write_is_permit_gated(runtime_factory):
    marker = "n1-reflection-marker-7"
    runtime = runtime_factory(decider=_ReflectDecider(), slow_brain=_ReflectBrain(marker))
    result = runtime.tick(_warm_sample(), capabilities={"vision"}, structured=True)
    assert result.decision["choice"] == "reflect"

    memories = [str(row["text"]) for row in runtime.store.iter_table("memories")]
    assert any(marker in text for text in memories), "reflection did not reach memory"

    # The write is visible in the broker's keyed audit chain as a permit-gated
    # memory.remember, proving it did not bypass the Governor.
    assert runtime.audit is not None and runtime.audit.keyed is True
    audited = [
        event for event in runtime.audit.events()
        if event.governor_result.get("tool") == "memory"
        and event.governor_result.get("op") == "remember"
    ]
    assert audited, "reflection memory write is missing from the broker audit chain"

    # With authority withheld, a reflection must not produce a memory row.
    before = runtime.store.counts()["memories"]
    runtime.permits.set_policy(MissionPolicy.observation_only("regression withholds"))
    runtime._do_reflection({"goal_digest": "withheld"}, "")
    assert runtime.store.counts()["memories"] == before


# ---------------------------------------------------------------------------
# N2 -- every manifest cost unit is a budget resource
# ---------------------------------------------------------------------------


def test_n2_unknown_cost_unit_is_refused():
    with pytest.raises(ManifestError, match="budget resource"):
        CostModel("telepathy", 1.0)
    # The explicit unmetered sentinel stays valid.
    assert CostModel("none", 0.0).unit == "none"


def test_n2_default_manifest_uses_only_budget_resources():
    for unit in ("scheduler_tick", "store_write", "store_read", "io_read", "message"):
        assert unit in RESOURCES
    for tool_name in DEFAULT_MANIFEST.tool_names:
        for op_name, op in DEFAULT_MANIFEST.tool(tool_name).operations.items():
            if op.cost_model.unit != "none":
                assert op.cost_model.unit in RESOURCES, f"{tool_name}.{op_name}"


def test_n2_new_resource_units_are_enforceable_by_the_budget(tmp_path):
    # store_write: memory.remember is priced at one store_write, so a one-unit
    # cap must be exhausted by the first manifest-derived reservation.
    policy = _verified_policy(tmp_path, version="sw", budgets={"store_write": 1})
    budget = BudgetService(tmp_path / "sw-budget.db", policy)
    service = PermitService(
        PermitIssuer(ISSUER_KEY),
        policy,
        manifest=broker_manifest(),
        budget=budget,
        db_path=tmp_path / "sw-permits.db",
    )
    service.issue(
        tool="memory", op="remember", args={"text": "a", "importance": 3}, goal_id="g-n2"
    )
    with pytest.raises(BudgetExceeded):
        service.issue(
            tool="memory", op="remember", args={"text": "b", "importance": 3}, goal_id="g-n2"
        )

    # scheduler_tick: state.set_load_target is priced at one scheduler_tick.
    policy2 = _verified_policy(tmp_path, version="st", budgets={"scheduler_tick": 1})
    budget2 = BudgetService(tmp_path / "st-budget.db", policy2)
    service2 = PermitService(
        PermitIssuer(ISSUER_KEY),
        policy2,
        manifest=broker_manifest(),
        budget=budget2,
        db_path=tmp_path / "st-permits.db",
    )
    service2.issue(tool="state", op="set_load_target", args={"load": 0.5}, goal_id="g-n2")
    with pytest.raises(BudgetExceeded):
        service2.issue(tool="state", op="set_load_target", args={"load": 0.5}, goal_id="g-n2")


# ---------------------------------------------------------------------------
# N3 -- the runtime broker's audit chain is persistent and keyed
# ---------------------------------------------------------------------------


def test_n3_runtime_broker_audit_is_persistent_and_keyed(runtime_factory):
    runtime = runtime_factory()
    audit = runtime.executor.broker.audit
    assert audit is runtime.audit
    assert audit.keyed is True

    data_dir = Path(runtime.config.data_dir)
    assert (data_dir / "audit.db").exists()
    key_file = data_dir / "audit.key"
    assert key_file.exists()
    assert len(key_file.read_bytes()) == 32
    assert (key_file.stat().st_mode & 0o777) == 0o600

    runtime.audit.append(outcome="n3-regression", policy_version="1.0.0")
    reopened = AuditLog(str(data_dir / "audit.db"), key=key_file.read_bytes())
    try:
        assert reopened.verify_chain().ok is True
    finally:
        reopened.close()


# ---------------------------------------------------------------------------
# F3 -- capability names must match the manifest exactly
# ---------------------------------------------------------------------------


def test_f3_capability_must_match_the_manifest_operation(tmp_path):
    service = PermitService(PermitIssuer(ISSUER_KEY), _verified_policy(tmp_path))

    # A malformed name with no dot is refused rather than treated as a wildcard.
    with pytest.raises(PermitDenied, match="capability"):
        service.issue(
            tool="state", op="set_load_target", args={"load": 0.5}, goal_id="g-f3",
            capability="observe",
        )
    # Another operation of the *same* tool is refused too: it is not the op.
    with pytest.raises(PermitDenied, match="does not match"):
        service.issue(
            tool="state", op="set_load_target", args={"load": 0.5}, goal_id="g-f3",
            capability="state.cooldown",
        )
    permit = service.issue(
        tool="state", op="set_load_target", args={"load": 0.5}, goal_id="g-f3",
        capability="state.set_load_target",
    )
    assert permit.capability == "state.set_load_target"


# ---------------------------------------------------------------------------
# F17 -- the sandboxed shell path audits before it spawns
# ---------------------------------------------------------------------------


def test_f17_shell_audits_before_effect_and_fails_closed(tmp_path, monkeypatch):
    manifest = broker_manifest()
    service = PermitService(
        PermitIssuer(ISSUER_KEY), _verified_policy(tmp_path), manifest=manifest
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()
    broker = ToolBroker(
        service,
        audit=AuditLog(":memory:", key=b"f17-audit-key-0123456789abcdef"),
        workspace=workspace,
        manifest=manifest,
    )

    spawned: list[str] = []

    def _boom_popen(*args, **kwargs):  # pragma: no cover - must never run
        spawned.append("spawned")
        raise AssertionError("Popen ran before the audit row was committed")

    def _fail_record(*args, **kwargs):
        raise RuntimeError("audit storage unavailable")

    monkeypatch.setattr(broker, "bwrap_available", lambda: True)
    monkeypatch.setattr(broker, "_record", _fail_record)
    monkeypatch.setattr(
        "reflective_loop_runtime.kernel.toolbroker.subprocess.Popen", _boom_popen
    )

    permit = service.issue(
        tool="shell", op="run", args={"command": "true"}, goal_id="g-f17"
    )
    with pytest.raises(RuntimeError, match="audit storage unavailable"):
        broker.invoke(permit, "shell", "run", {"command": "true"})
    assert spawned == [], "the command spawned despite the audit write failing"

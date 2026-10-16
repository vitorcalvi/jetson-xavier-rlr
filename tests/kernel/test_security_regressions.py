"""Regression tests for the kernel security review (findings F1-F18).

Each test pins one finding from the security review of the MissionPolicy ->
Budget -> Governor -> ExecutionPermit -> ToolBroker chain and fails if the fix
is reverted. The review comment itself was not reachable from this environment,
so every finding is stated inline with the requirement it enforces.
"""

from __future__ import annotations

import json

import pytest

from reflective_loop_runtime.config import Config
from reflective_loop_runtime.executor import Executor, effect_request
from reflective_loop_runtime.kernel.audit import (
    AuditLog,
    compute_event_hash,
    verify_chain,
)
from reflective_loop_runtime.kernel.budget import BudgetService
from reflective_loop_runtime.kernel.manifest import (
    CostModel,
    FieldSpec,
    ManifestError,
    OperationSpec,
    ToolManifest,
    ToolSpec,
)
from reflective_loop_runtime.kernel.permit import (
    PermitDenied,
    PermitIssuer,
    PermitService,
    PermitVerificationError,
)
from reflective_loop_runtime.kernel.policy import (
    MODE_ENFORCE,
    MODE_OBSERVATION_ONLY,
    load_policy,
    sign_envelope,
)
from reflective_loop_runtime.kernel.toolbroker import (
    ToolBroker,
    ToolBrokerError,
    ToolNotPermitted,
    broker_manifest,
)
from reflective_loop_runtime.store import Store

DEV_SECRET = "dev-only-shared-secret"
ISSUER_KEY = b"security-regression-issuer-key-01"


def _bundle(**overrides):
    bundle = {
        "mission_id": "mission-sec",
        "objective": "exercise the security boundary",
        "invariants": ["never_shell_out"],
        "allowed_domains": [],
        "capabilities": [],
        "budgets": {},
        "standing_mandates": [],
        "policy_version": "1.0.0",
    }
    bundle.update(overrides)
    return bundle


def _policy(tmp_path, *, name="policy.json", allow_dev_hmac=True, **overrides):
    path = tmp_path / name
    path.write_text(
        json.dumps(sign_envelope(_bundle(**overrides), secret=DEV_SECRET)), encoding="utf-8"
    )
    return load_policy(path, secret=DEV_SECRET, allow_dev_hmac=allow_dev_hmac)


def _probe_spec(risk_class="A1"):
    return ToolSpec(
        "probe",
        {
            "record": OperationSpec(
                name="record",
                input_schema={"value": FieldSpec("int", min=0, max=10)},
                preconditions=("a probe is allowed",),
                effects=("a probe value is recorded",),
                risk_class=risk_class,
                reversibility="reversible",
                idempotency="idempotent",
                compensation="erase the probe",
                cost_model=CostModel("api_calls", 1.0, "one probe"),
            )
        },
    )


def _meter_manifest():
    return ToolManifest([_probe_spec("A1")])


def _permit_service(tmp_path, *, policy=None, manifest=None, budget=None, name="policy.json"):
    policy = policy if policy is not None else _policy(tmp_path, name=name)
    if manifest is None:
        manifest = broker_manifest()
    service = PermitService(
        PermitIssuer(ISSUER_KEY), policy, manifest=manifest, budget=budget
    )
    return policy, manifest, service


def _issue(service, tool="state", op="set_load_target", args=None, **kwargs):
    return service.issue(
        tool=tool, op=op, args=args if args is not None else {"load": 0.5},
        goal_id="goal-sec", **kwargs,
    )


def _execute(executor, service, action, *, tick=None):
    """Issue the exact permit for ``action`` and execute it under that permit."""
    tick = executor.current_tick if tick is None else tick
    request = effect_request(action, tick)
    permit = None
    if request is not None:
        tool, op, args = request
        permit = _issue(service, tool=tool, op=op, args=args)
    return executor.execute(action, permit=permit, tick=tick)


# ---------------------------------------------------------------------------
# F1 - no permit-less side-effect path on the model-facing broker
# ---------------------------------------------------------------------------
def test_f1_permit_bearing_broker_refuses_permit_less_effects(tmp_path):
    policy, manifest, service = _permit_service(tmp_path)
    broker = ToolBroker(service, audit=AuditLog(":memory:"), manifest=manifest)
    broker.register_effect("state", "set_load_target", lambda args: {"load_target": args["load"]})

    # The permit-verifying effect path refuses a call with no permit before the
    # adapter can run.
    with pytest.raises(ToolNotPermitted):
        broker.invoke_effect(None, "state", "set_load_target", {"load": 0.5})

    # There is no permit-less API left on any broker: the trusted-local escape
    # hatch (``for_effects`` / ``apply_effect``) is gone entirely.
    assert not hasattr(broker, "apply_effect")
    assert not hasattr(ToolBroker, "for_effects")
    assert not hasattr(broker, "trusted_local")

    # A broker built without a permit service can be inspected but can never
    # perform an effect: its permit-verifying path refuses outright.
    local = ToolBroker(None, manifest=manifest)
    local.register_effect("state", "set_load_target", lambda args: {"ok": True})
    with pytest.raises(ToolNotPermitted):
        local.invoke_effect(object(), "state", "set_load_target", {"load": 0.5})

    # With a real permit the same effect is allowed on the permit-bearing broker.
    permit = _issue(service)
    result = broker.invoke_effect(permit, "state", "set_load_target", {"load": 0.5})
    assert result.executed is True


# ---------------------------------------------------------------------------
# F2 - a permit is signed over its exact operation
# ---------------------------------------------------------------------------
def test_f2_permit_is_bound_to_its_operation(tmp_path):
    _, _, service = _permit_service(tmp_path)
    permit = _issue(service)
    assert permit.op == "set_load_target"

    # Same tool, same args, but a different operation: the op binding refuses it.
    with pytest.raises(PermitVerificationError, match="operation"):
        service.verify_and_consume(
            permit, "state", {"load": 0.5}, op="cooldown"
        )
    assert not service.consumed(permit.nonce)

    # Naming no operation at all is also a failure (fail closed).
    with pytest.raises(PermitVerificationError):
        service.verify_and_consume(permit, "state", {"load": 0.5})


# ---------------------------------------------------------------------------
# F3 - the capability is signed and may not be borrowed from another tool
# ---------------------------------------------------------------------------
def test_f3_capability_is_bound_and_not_borrowable(tmp_path):
    policy = _policy(tmp_path, capabilities=["memory.forget"])
    _, _, service = _permit_service(tmp_path, policy=policy)
    # A capability that names another tool cannot be used to mint this permit.
    with pytest.raises(PermitDenied, match="does not belong to tool"):
        _issue(service, capability="memory.forget")

    # The capability is covered by the MAC: editing it invalidates the permit.
    permissive = _policy(tmp_path, name="permissive.json")
    _, _, service = _permit_service(tmp_path, policy=permissive)
    permit = _issue(service, capability="state.set_load_target")
    import dataclasses

    tampered = dataclasses.replace(permit, capability="root")
    with pytest.raises(PermitVerificationError):
        service.verify_and_consume(tampered, "state", {"load": 0.5}, op="set_load_target")


# ---------------------------------------------------------------------------
# F4 - a permit is bound to the manifest it was issued under
# ---------------------------------------------------------------------------
def test_f4_permit_is_bound_to_the_manifest_digest(tmp_path):
    policy = _policy(tmp_path)
    strong = ToolManifest([_probe_spec("A1")])
    issuer = PermitIssuer(ISSUER_KEY)
    source = PermitService(issuer, policy, manifest=strong)
    permit = source.issue(tool="probe", op="record", args={"value": 1}, goal_id="g")

    weakened = ToolManifest([_probe_spec("A2")])
    target = PermitService(issuer, policy, manifest=weakened)
    # A permit minted by a different service instance is refused outright before
    # any binding is considered (kernel/test_permit.py::TestPermitInstanceBinding
    # ), so this exercises the manifest check by having one instance run against
    # two manifests: the target adopts the issuing identity and it is the digest
    # -- not the instance -- that has to refuse the permit.
    target._instance_id = source.instance_id
    with pytest.raises(PermitVerificationError, match="manifest"):
        target.verify_and_consume(permit, "probe", {"value": 1}, op="record")
    assert not target.consumed(permit.nonce)


# ---------------------------------------------------------------------------
# F5 - a permit is bound to the policy it was issued under
# ---------------------------------------------------------------------------
def test_f5_permit_is_bound_to_the_policy_digest(tmp_path):
    issuer = PermitIssuer(ISSUER_KEY)
    original = _policy(tmp_path, name="p1.json")
    swapped = _policy(tmp_path, name="p2.json", objective="a different mission entirely")

    service = PermitService(issuer, original, manifest=broker_manifest())
    permit = _issue(service)

    # Same policy_version, different authority -> the digest check refuses it.
    # The one instance adopts the new policy, because a permit minted by another
    # instance is refused outright before any binding is considered.
    service.set_policy(swapped)
    with pytest.raises(PermitVerificationError, match="policy"):
        service.verify_and_consume(permit, "state", {"load": 0.5}, op="set_load_target")


# ---------------------------------------------------------------------------
# F6 - policy capabilities are enforced at issuance
# ---------------------------------------------------------------------------
def test_f6_policy_capabilities_are_enforced_at_issuance(tmp_path):
    policy = _policy(tmp_path, capabilities=["observe"])
    _, _, service = _permit_service(tmp_path, policy=policy)
    with pytest.raises(PermitDenied, match="capability"):
        _issue(service)

    # A granted capability issues normally.
    granted = _policy(tmp_path, name="granted.json", capabilities=["state.set_load_target"])
    _, _, service = _permit_service(tmp_path, policy=granted)
    assert _issue(service).capability == "state.set_load_target"


# ---------------------------------------------------------------------------
# F7 - policy allowed_domains are enforced at issuance
# ---------------------------------------------------------------------------
def test_f7_policy_allowed_domains_are_enforced_at_issuance(tmp_path):
    policy = _policy(tmp_path, allowed_domains=["telemetry"])
    _, _, service = _permit_service(tmp_path, policy=policy)
    with pytest.raises(PermitDenied, match="allowed_domains"):
        _issue(service)

    scoped = _policy(tmp_path, name="scoped.json", allowed_domains=["state"])
    _, _, service = _permit_service(tmp_path, policy=scoped)
    assert _issue(service).tool == "state"


# ---------------------------------------------------------------------------
# F8 - budget amounts come from the manifest, never the caller
# ---------------------------------------------------------------------------
def test_f8_budget_amounts_are_derived_from_the_manifest(tmp_path):
    manifest = _meter_manifest()
    policy = _policy(tmp_path, budgets={"api_calls": 10})
    budget = BudgetService(tmp_path / "budget.db", policy)
    _, _, service = _permit_service(tmp_path, policy=policy, manifest=manifest, budget=budget)

    # Caller-supplied amounts are refused outright.
    with pytest.raises(PermitDenied, match="manifest"):
        service.issue(
            tool="probe", op="record", args={"value": 1}, goal_id="g",
            amounts={"api_calls": 999},
        )

    permit = service.issue(tool="probe", op="record", args={"value": 1}, goal_id="g")
    assert budget.scope_usage("goal", "g", "api_calls") == 1
    assert budget.reservation_state(permit.budget_reservation) == "reserved"


# ---------------------------------------------------------------------------
# F9 - every budget scope level is charged
# ---------------------------------------------------------------------------
def test_f9_all_budget_scope_levels_are_charged(tmp_path):
    manifest = _meter_manifest()
    policy = _policy(tmp_path, budgets={"api_calls": 10})
    budget = BudgetService(tmp_path / "budget.db", policy)
    _, _, service = _permit_service(tmp_path, policy=policy, manifest=manifest, budget=budget)
    permit = service.issue(tool="probe", op="record", args={"value": 1}, goal_id="goal-9")

    assert budget.scope_usage("global", "global", "api_calls") == 1
    assert budget.scope_usage("mission", "mission-sec", "api_calls") == 1
    assert budget.scope_usage("goal", "goal-9", "api_calls") == 1
    assert budget.scope_usage("permit", permit.permit_id, "api_calls") == 1


# ---------------------------------------------------------------------------
# F10 - a released reservation can no longer authorise execution
# ---------------------------------------------------------------------------
def test_f10_released_reservation_rejects_execution(tmp_path):
    manifest = _meter_manifest()
    policy = _policy(tmp_path, budgets={"api_calls": 10})
    budget = BudgetService(tmp_path / "budget.db", policy)
    _, _, service = _permit_service(tmp_path, policy=policy, manifest=manifest, budget=budget)
    permit = service.issue(tool="probe", op="record", args={"value": 1}, goal_id="g")

    budget.release(permit.budget_reservation)
    assert budget.reservation_state(permit.budget_reservation) == "released"
    with pytest.raises(PermitVerificationError, match="not open|reservation"):
        service.verify_and_consume(permit, "probe", {"value": 1}, op="record")
    assert not service.consumed(permit.nonce)


# ---------------------------------------------------------------------------
# F11 - the signing key is not reachable from the model-facing service
# ---------------------------------------------------------------------------
def test_f11_signing_key_is_not_reachable_from_the_service(tmp_path):
    policy = _policy(tmp_path)
    issuer = PermitIssuer(ISSUER_KEY)
    service = PermitService(issuer, policy, manifest=broker_manifest())

    assert "_issuer" not in vars(service)
    assert not hasattr(service, "_issuer")
    # The signing callable is a plain closure, not a bound method that would
    # expose ``__self__._key``.
    assert getattr(service._sign, "__self__", None) is None
    assert ISSUER_KEY.decode() not in repr(vars(service))


# ---------------------------------------------------------------------------
# F12 - the replay ledger is persistent by default
# ---------------------------------------------------------------------------
def test_f12_replay_ledger_is_persistent_by_default(tmp_path):
    policy = _policy(tmp_path)
    issuer = PermitIssuer(ISSUER_KEY)
    first = PermitService(issuer, policy, manifest=broker_manifest())
    second = PermitService(issuer, policy, manifest=broker_manifest())

    assert first.db_path is not None
    assert first.db_path == second.db_path

    permit = _issue(first)
    first.verify_and_consume(permit, "state", {"load": 0.5}, op="set_load_target")
    # The ledger is shared, so the nonce one service burned is already burned for
    # every service over the same policy -- persistence is what this asserts. (A
    # permit minted by the *other* instance is refused outright on that ground
    # alone, so presenting it to ``second`` would prove nothing about the ledger:
    # see kernel/test_permit.py::TestPermitInstanceBinding.)
    assert second.consumed(permit.nonce) is True


# ---------------------------------------------------------------------------
# F13 - authority is signature-only; dev HMAC is never enforced by default
# ---------------------------------------------------------------------------
def test_f13_dev_hmac_is_not_enforced_by_default(tmp_path):
    path = tmp_path / "hmac.json"
    path.write_text(
        json.dumps(sign_envelope(_bundle(), secret=DEV_SECRET)), encoding="utf-8"
    )

    default = load_policy(path, secret=DEV_SECRET)
    assert default.mode == MODE_OBSERVATION_ONLY
    assert default.enforced is False

    opt_in = load_policy(path, secret=DEV_SECRET, allow_dev_hmac=True)
    assert opt_in.mode == MODE_ENFORCE
    assert opt_in.dev_only is True

    # A real signature is authoritative with no dev opt-in.
    cryptography = pytest.importorskip(
        "cryptography.hazmat.primitives.asymmetric.ed25519"
    )
    private = cryptography.Ed25519PrivateKey.generate()
    signed = tmp_path / "ed25519.json"
    signed.write_text(
        json.dumps(sign_envelope(_bundle(), private_key=private)), encoding="utf-8"
    )
    verified = load_policy(signed, public_key=private.public_key())
    assert verified.mode == MODE_ENFORCE
    assert verified.dev_only is False


# ---------------------------------------------------------------------------
# F14 - shell.read cannot write through the sandbox
# ---------------------------------------------------------------------------
def test_f14_shell_read_gets_a_read_only_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    broker = ToolBroker(None, workspace=workspace, manifest=broker_manifest())

    def _contains(argv, sequence):
        return any(
            argv[index : index + len(sequence)] == sequence
            for index in range(len(argv) - len(sequence) + 1)
        )

    read_only = broker._sandbox_argv("true", read_only=True)
    read_write = broker._sandbox_argv("true")
    bind = ["--bind", str(workspace), str(workspace)]
    ro_bind = ["--ro-bind", str(workspace), str(workspace)]

    assert _contains(read_only, ro_bind)
    assert not _contains(read_only, bind)
    assert _contains(read_write, bind)
    assert not _contains(read_write, ro_bind)


# ---------------------------------------------------------------------------
# F15 - the manifest and adapter registry are sealed after construction
# ---------------------------------------------------------------------------
def test_f15_manifest_and_adapters_are_sealed_after_use(tmp_path):
    _, manifest, service = _permit_service(tmp_path)
    broker = ToolBroker(service, audit=AuditLog(":memory:"), manifest=manifest)
    broker.register_effect("state", "set_load_target", lambda args: {"ok": True})
    assert broker.sealed is False

    permit = _issue(service)
    broker.invoke(permit, "state", "set_load_target", {"load": 0.5})

    assert broker.sealed is True
    assert manifest.sealed is True
    with pytest.raises(ToolBrokerError, match="sealed"):
        broker.register_effect("probe", "record", lambda args: {})
    with pytest.raises(ManifestError, match="sealed"):
        manifest.register(_probe_spec())


# ---------------------------------------------------------------------------
# F16 - the audit chain is keyed (tamper-proof, not just tamper-evident)
# ---------------------------------------------------------------------------
def test_f16_audit_chain_is_keyed(tmp_path):
    key = b"k" * 32
    path = tmp_path / "audit.db"
    log = AuditLog(path, key=key)
    try:
        log.append(goal_version="g", plan_version="p", outcome="executed")
        assert log.keyed is True
        assert log.verify_chain().ok

        events = log.events()
        # Recomputing the chain without the key does not verify a keyed chain.
        assert verify_chain(events, key=None).ok is False
        assert verify_chain(events, key=key).ok is True

        # An attacker who recomputes a forged row with the unkeyed digest gets a
        # different hash, so the edit is detected.
        event = events[0]
        forged = dict(event.payload())
        forged["outcome"] = "forged"
        assert compute_event_hash(event.prev_hash, forged) != event.event_hash
    finally:
        log.close()


# ---------------------------------------------------------------------------
# F17 - the audit row is written before the effect, and fails closed
# ---------------------------------------------------------------------------
def test_f17_audit_before_effect_fails_closed(tmp_path, monkeypatch):
    _, manifest, service = _permit_service(tmp_path)
    broker = ToolBroker(service, audit=AuditLog(":memory:"), manifest=manifest)
    ran = []
    broker.register_effect("state", "set_load_target", lambda args: ran.append(args) or {"ok": True})
    permit = _issue(service)

    def _audit_down(*args, **kwargs):
        raise RuntimeError("audit log is unavailable")

    monkeypatch.setattr(broker._audit, "append", _audit_down)
    with pytest.raises(RuntimeError):
        broker.invoke(permit, "state", "set_load_target", {"load": 0.5})
    assert ran == []  # the effect never ran because the audit write failed first


# ---------------------------------------------------------------------------
# F18 - memory.recall is read-only
# ---------------------------------------------------------------------------
def test_f18_memory_recall_is_read_only(tmp_path):
    _, manifest, service = _permit_service(tmp_path, name="f18-policy.json")
    store = Store(Config(data_dir=tmp_path, tick_seconds=1.0).db_path)
    try:
        broker = ToolBroker(service, audit=AuditLog(":memory:"), manifest=manifest)
        executor = Executor(store, broker=broker)
        _execute(
            executor, service,
            {"action": "remember", "rationale": "x", "text": "seed note", "importance": 3},
        )
        before = store.counts()
        result = _execute(
            executor, service,
            {"action": "recall", "rationale": "x", "query": "seed"},
        )
        assert result.ok is True
        assert result.applied["matches"] == 1
        # A read that matches must still not write an episode or any other row.
        assert store.counts() == before
    finally:
        store.close()

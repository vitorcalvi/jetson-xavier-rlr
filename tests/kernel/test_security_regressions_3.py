"""Third security-regression pass: the remaining permit-gate bypasses.

Every test here must fail against the pre-fix kernel and pass against the
hardened one. The findings covered are:

* F1a -- the model-facing ``Executor`` published its ``Store`` and kept the raw,
  permit-free ``_effect_*`` adapters bound to itself.
* F1b -- ``ToolBroker._perform``, the dispatch that actually calls an adapter,
  trusted whatever permit it was handed and never verified one itself.
* N4  -- a reflection was turned into a memory write even when the ``reflect``
  effect it followed had itself been refused.
* N5  -- the default ``ReflectiveLoopRuntime`` minted permits with no
  ``BudgetService``, so its mission policy's budgets were never enforced.
* F11 -- key material stayed reachable from model-facing objects: the
  ``PermitIssuer`` cached the raw HMAC key, the ``AuditLog`` cached the raw chain
  key, the runtime published that log, and the runtime's default policy fell
  back to a dev-HMAC signature that became ENFORCE.
* K9e1 residual -- ``Executor._store`` was still a live ``Store`` handle on the
  model-facing object.
"""

from __future__ import annotations

import gc
import json
import types
from pathlib import Path

import pytest

from reflective_loop_runtime import runtime as runtime_module
from reflective_loop_runtime.config import Config
from reflective_loop_runtime.executor import Executor, _EffectAdapters, effect_request
from reflective_loop_runtime.kernel.audit import AuditLog
from reflective_loop_runtime.kernel.budget import BudgetExceeded, BudgetService
from reflective_loop_runtime.kernel.permit import (
    PermitError,
    PermitIssuer,
    PermitService,
)
from reflective_loop_runtime.kernel.policy import load_policy, sign_envelope
from reflective_loop_runtime.kernel.toolbroker import (
    ToolBroker,
    ToolNotPermitted,
    broker_manifest,
)
from reflective_loop_runtime.store import Store

DEV_SECRET = "dev-only-shared-secret"
ISSUER_KEY = b"security-regression-3-issuer-key"


def _policy(tmp_path: Path, *, version: str = "1.0.0", budgets=None):
    bundle = {
        "mission_id": "mission-r3",
        "objective": "close the remaining permit-gate bypasses",
        "invariants": ["no_effect_without_permit"],
        "allowed_domains": [],
        "capabilities": [],
        "budgets": dict(budgets or {}),
        "standing_mandates": [],
        "policy_version": version,
    }
    path = tmp_path / f"policy-{version}.json"
    path.write_text(
        json.dumps(sign_envelope(bundle, secret=DEV_SECRET)), encoding="utf-8"
    )
    return load_policy(path, secret=DEV_SECRET, allow_dev_hmac=True)


def _executor(tmp_path: Path) -> tuple[Executor, Store, PermitService]:
    manifest = broker_manifest()
    service = PermitService(
        PermitIssuer(ISSUER_KEY),
        _policy(tmp_path),
        manifest=manifest,
        db_path=tmp_path / "permits.db",
    )
    broker = ToolBroker(service, audit=AuditLog(":memory:"), manifest=manifest)
    store = Store(tmp_path / "store.db")
    return Executor(store, broker=broker), store, service


# ---------------------------------------------------------------------------
# Object-graph reachability helpers
#
# ``material`` may be key bytes (compared by value) or a plain object such as a
# ``Store`` (compared by identity). The walk covers every route a model would
# have: ``dir()`` attribute names including underscore and dunder ones,
# ``__dict__``, ``__slots__``, closure cells, container items and
# ``gc.get_referents``.
# ---------------------------------------------------------------------------


def _reachable_objects(root, *, stop=(), limit=200_000):
    """Every object reachable from ``root`` through attributes and the GC graph.

    Types, modules and function ``__globals__`` are code namespaces rather than
    object-graph edges -- following them would report whatever unrelated module
    happens to hold the same bytes -- so they terminate the walk. ``stop`` names
    any other class that terminates it: the broker is the documented,
    permit-verified choke point that has to hold the effect adapters.
    """
    seen: set[int] = set()
    found: list[object] = []
    stack: list[object] = [root]
    while stack and len(found) < limit:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        found.append(obj)
        if stop and isinstance(obj, stop):
            continue
        if isinstance(obj, (type, types.ModuleType)):
            continue
        children: list[object] = []
        if isinstance(obj, dict):
            children.extend(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            children.extend(obj)
        elif isinstance(obj, types.FunctionType):
            children.extend(cell.cell_contents for cell in (obj.__closure__ or ()))
        else:
            state = getattr(obj, "__dict__", None)
            if isinstance(state, dict):
                children.extend(state.values())
            for slot in getattr(type(obj), "__slots__", ()) or ():
                children.append(getattr(obj, slot, None))
            children.extend(gc.get_referents(obj))
        for child in children:
            if child is not None and id(child) not in seen:
                stack.append(child)
    return found


def _is_material(value, material) -> bool:
    if value is material:
        return True
    if isinstance(material, (bytes, bytearray)) and isinstance(
        value, (bytes, bytearray, memoryview)
    ):
        try:
            return bytes(value) == bytes(material)
        except TypeError:  # pragma: no cover - non-contiguous buffer
            return False
    return False


def _leaks(root, material, *, stop=()) -> list[str]:
    """Describe how ``material`` is reachable from ``root``, by name or by graph."""
    hits: list[str] = []
    for name in dir(root):
        try:
            value = getattr(root, name)
        except Exception:  # pragma: no cover - defensive; a property may raise
            continue
        if _is_material(value, material):
            hits.append(f"attribute {name!r}")
    for obj in _reachable_objects(root, stop=stop):
        if _is_material(obj, material):
            hits.append(f"{type(obj).__name__} in the GC referent graph")
    return hits


# ---------------------------------------------------------------------------
# F1a -- the executor publishes neither the Store nor a raw effect adapter
# ---------------------------------------------------------------------------


def test_f1a_executor_publishes_no_store_and_no_raw_adapters(tmp_path):
    executor, store, service = _executor(tmp_path)

    # The model-facing object hands out no Store handle...
    assert not hasattr(executor, "store")
    assert "store" not in vars(executor)
    # ...and none of the permit-free effect implementations is reachable on it,
    # so there is no ``executor._effect_remember({...})`` side-effect door.
    assert [name for name in dir(executor) if name.startswith("_effect")] == []

    # K9e1 residual: the Store is not held under *any* name -- not through
    # ``dir()`` (underscore names included), ``__dict__``, ``__slots__``, a
    # closure, or ``gc.get_referents``. The broker terminates the walk because
    # it is the permit-verified choke point that must hold the adapters; the
    # adapters themselves must still be unreachable from the executor.
    assert _leaks(executor, store, stop=(ToolBroker,)) == []
    reachable = _reachable_objects(executor, stop=(ToolBroker,))
    assert not [obj for obj in reachable if isinstance(obj, _EffectAdapters)]
    assert not [obj for obj in reachable if isinstance(obj, Store)]

    # The permit gate still works: the same effect runs when the Governor's
    # authority is presented, so the adapters were hidden, not removed.
    action = {"action": "remember", "rationale": "f1a", "text": "gated", "importance": 3}
    tool, op, args = effect_request(action, 0)
    permit = service.issue(tool=tool, op=op, args=args, goal_id="goal-r3")
    result = executor.execute(action, permit=permit, tick=0)
    assert result.ok is True
    assert store.counts()["memories"] == 1

    # Without that permit nothing is applied.
    denied = executor.execute(action, tick=0)
    assert denied.ok is False
    assert store.counts()["memories"] == 1


def test_f1a_model_facing_state_carries_no_privileged_object(runtime_factory):
    runtime = runtime_factory()
    privileged = (Store, Executor, ToolBroker, PermitService, BudgetService)
    sample = {
        "temperature_c": 50.0,
        "zones_c": [50.0],
        "ram_used_mib": 1024.0,
        "ram_total_mib": 4096.0,
        "sampled_monotonic": None,
        "age_seconds": 0.0,
    }

    seen: list[object] = []

    def walk(value, depth=0):
        if depth > 4:
            return
        if isinstance(value, dict):
            seen.extend(value.values())
            for item in value.values():
                walk(item, depth + 1)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item, depth + 1)

    state = runtime.build_state(sample, capabilities={"vision"})
    walk(state)
    assert not [item for item in seen if isinstance(item, privileged)]
    runtime.store.close()


# ---------------------------------------------------------------------------
# F1b -- the dispatch that calls the adapter verifies the permit itself
# ---------------------------------------------------------------------------


def _counting_broker(tmp_path: Path, calls: list[dict]):
    manifest = broker_manifest()
    service = PermitService(
        PermitIssuer(ISSUER_KEY),
        _policy(tmp_path),
        manifest=manifest,
        db_path=tmp_path / "permits.db",
    )
    broker = ToolBroker(service, audit=AuditLog(":memory:"), manifest=manifest)

    def _apply(args):
        calls.append(dict(args))
        return {"load_target": args["load"]}

    broker.register_effect("state", "set_load_target", _apply)
    return broker, service


def test_f1b_perform_refuses_to_run_an_adapter_on_an_unverified_permit(tmp_path):
    calls: list[dict] = []
    broker, service = _counting_broker(tmp_path, calls)

    permit = service.issue(
        tool="state", op="set_load_target", args={"load": 0.5}, goal_id="goal-r3"
    )
    assert broker.invoke(permit, "state", "set_load_target", {"load": 0.5}).executed
    assert len(calls) == 1

    # The nonce is spent, so reaching the dispatch directly must not run the
    # adapter a second time: ``_perform`` verifies rather than trusting.
    with pytest.raises(PermitError):
        broker._perform(permit, "state", "set_load_target", {"load": 0.9}, "A1")
    assert len(calls) == 1

    # No permit at all, and a broker with no permit service, are both refused
    # before the adapter is reached.
    with pytest.raises(ToolNotPermitted):
        broker._perform(None, "state", "set_load_target", {"load": 0.9}, "A1")
    local_calls: list[dict] = []
    local = ToolBroker(None, audit=AuditLog(":memory:"), manifest=broker_manifest())
    local.register_effect(
        "state", "set_load_target", lambda args: local_calls.append(dict(args))
    )
    with pytest.raises(ToolNotPermitted):
        local._perform(object(), "state", "set_load_target", {"load": 0.9}, "A1")
    assert len(calls) == 1
    assert local_calls == []


# ---------------------------------------------------------------------------
# N4 -- reflection memory only follows a successful reflect effect
# ---------------------------------------------------------------------------


class _ReflectDecider:
    name = "r3-reflect"

    def score(self, candidates, state, *, context=""):
        from reflective_loop_runtime.contracts import ScoredDecision

        ids = [candidate.id for candidate in candidates]
        target = "reflect" if "reflect" in ids else ids[0]
        probabilities = {candidate: 0.0 for candidate in ids}
        probabilities[target] = 1.0
        return ScoredDecision(choice=target, confidence=1.0, probabilities=probabilities)


class _ReflectBrain:
    name = "r3-reflect-brain"

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


def test_n4_reflection_is_dropped_when_the_reflect_effect_fails(runtime_factory, monkeypatch):
    marker = "n4-reflection-marker-9"
    runtime = runtime_factory(decider=_ReflectDecider(), slow_brain=_ReflectBrain(marker))

    # Authority for ``control.reflect`` is withheld; the memory.remember permit
    # the reflection would need is still granted normally.
    real_authorize = runtime.governor.authorize

    def authorize(action, **kwargs):
        if action.get("action") == "reflect":
            return None
        return real_authorize(action, **kwargs)

    monkeypatch.setattr(runtime.governor, "authorize", authorize)
    result = runtime.tick(_warm_sample(), capabilities={"vision"}, structured=True)

    assert result.decision["choice"] == "reflect"
    assert result.execution["ok"] is False
    assert runtime.store.counts()["memories"] == 0
    runtime.store.close()


def test_n4_successful_reflect_still_writes_through_the_permit_gate(runtime_factory):
    marker = "n4-reflection-marker-ok"
    runtime = runtime_factory(decider=_ReflectDecider(), slow_brain=_ReflectBrain(marker))
    result = runtime.tick(_warm_sample(), capabilities={"vision"}, structured=True)

    assert result.execution["ok"] is True
    memories = [str(row["text"]) for row in runtime.store.iter_table("memories")]
    assert any(marker in text for text in memories)
    audited = [
        event for event in runtime.audit.events()
        if event.governor_result.get("tool") == "memory"
        and event.governor_result.get("op") == "remember"
    ]
    assert audited, "the reflection write did not pass through the broker"
    runtime.store.close()


# ---------------------------------------------------------------------------
# N5 -- the default runtime enforces its mission policy's budgets
# ---------------------------------------------------------------------------


def test_n5_default_runtime_wires_a_budget_service(runtime_factory):
    runtime = runtime_factory()
    budget = runtime.budget
    assert isinstance(budget, BudgetService)
    assert budget.caps(), "the runtime policy declares no budgets to enforce"

    permit = runtime.permits.issue(
        tool="state", op="set_load_target", args={"load": 0.5}, goal_id="goal-n5"
    )
    assert permit.budget_reservation is not None
    assert budget.reservation_state(permit.budget_reservation) == "reserved"
    # state.set_load_target is priced at one scheduler_tick.
    assert budget.scope_usage("goal", "goal-n5", "scheduler_tick") == 1.0

    # The declared cap is a real ceiling, not decoration.
    with pytest.raises(BudgetExceeded):
        budget.reserve({"goal": "goal-n5"}, {"scheduler_tick": 10_000_000.0})
    runtime.store.close()


# ---------------------------------------------------------------------------
# F11 -- no key material is reachable from a model-facing object
# ---------------------------------------------------------------------------


def test_f11_permit_issuer_key_is_not_reachable_from_the_service(tmp_path):
    executor, _, service = _executor(tmp_path)
    issuer = PermitIssuer(ISSUER_KEY)

    # Pre-fix the issuer kept ``_key`` and the log kept ``_key`` as raw bytes;
    # the key is now absorbed into opaque ``hashlib`` state that reports no GC
    # referents, so neither an attribute walk nor ``gc.get_referents`` finds it.
    for label, root in (("PermitIssuer", issuer), ("PermitService", service)):
        assert _leaks(root, ISSUER_KEY) == [], f"issuer key reachable from the {label}"

    # The issuer is itself kept private: the service holds only closed callables.
    assert not hasattr(service, "_issuer")
    assert ISSUER_KEY.decode() not in repr(vars(service))

    # Refusing to expose the key did not break signing: the permit still verifies
    # and still gates the effect.
    action = {"action": "remember", "rationale": "f11", "text": "signed", "importance": 3}
    tool, op, args = effect_request(action, 0)
    permit = service.issue(tool=tool, op=op, args=args, goal_id="goal-r3")
    assert executor.execute(action, permit=permit, tick=0).ok is True


def test_f11_audit_chain_key_is_not_reachable_from_the_runtime_objects(runtime_factory):
    runtime = runtime_factory()
    chain_key = (Path(runtime.config.data_dir) / "audit.key").read_bytes()
    assert runtime.audit is not None and runtime.audit.keyed is True

    # Every model-facing object the runtime hands out -- including the broker it
    # publishes -- must keep the chain key out of reach.
    for label, root in (
        ("runtime", runtime),
        ("executor", runtime.executor),
        ("broker", runtime.executor.broker),
        ("permit service", runtime.permits),
        ("audit log", runtime.audit),
    ):
        assert _leaks(root, chain_key) == [], f"chain key reachable from the {label}"

    # The chain is still keyed, so the links are HMACs and still verify.
    runtime.audit.append(outcome="f11-regression", policy_version="1.0.0")
    assert runtime.audit.verify_chain().ok is True
    runtime.store.close()


def test_f11_dev_hmac_policy_is_never_enforced_in_the_runtime_default_path(
    monkeypatch, tmp_path
):
    # Without the optional ed25519 backend there is no operator key. The runtime
    # must not borrow authority from the shared dev secret.
    monkeypatch.setattr(runtime_module, "HAVE_CRYPTOGRAPHY", False)

    policy = runtime_module._runtime_policy()
    assert policy.enforced is False, "a dev-HMAC signature became operator authority"
    assert policy.dev_only is True

    config = Config(data_dir=tmp_path / "dev-hmac")
    config.ensure_dirs()
    permits, budget = runtime_module._default_permits(config)
    assert permits.policy.enforced is False
    assert budget.caps() == {}
    with pytest.raises(PermitError, match="not enforced"):
        permits.issue(
            tool="state", op="set_load_target", args={"load": 0.5}, goal_id="goal-f11"
        )

    # And the same holds for a default runtime built in that environment.
    runtime = runtime_module.ReflectiveLoopRuntime(config)
    assert runtime.permits.policy.enforced is False
    runtime.store.close()


def test_f11_default_runtime_policy_is_enforced_with_an_operator_signature():
    # The hardening is not "stop enforcing": with an operator key the default
    # runtime policy is still ENFORCE and still issues permits.
    policy = runtime_module._runtime_policy()
    if not runtime_module.HAVE_CRYPTOGRAPHY:  # pragma: no cover - environment dependent
        pytest.skip("ed25519 backend unavailable; the default policy cannot be enforced")
    assert policy.enforced is True

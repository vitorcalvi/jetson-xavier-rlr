"""Execution permits: issuance authority, verification, and single-use nonces.

Policies are signed with the dev-only HMAC-SHA256 fallback so the suite runs
without the optional ``cryptography`` package; that still yields an ``ENFORCE``
policy, which is what the permit service requires.
"""

from __future__ import annotations

import dataclasses
import json
import threading
from datetime import datetime, timedelta, timezone

import pytest

from reflective_loop_runtime.kernel.budget import BudgetExceeded, BudgetService
from reflective_loop_runtime.kernel.permit import (
    PERMIT_FIELDS,
    ExecutionPermit,
    PermitClockError,
    PermitDenied,
    PermitError,
    PermitIssuer,
    PermitReplayError,
    PermitService,
    PermitVerificationError,
    args_hash,
)
from reflective_loop_runtime.kernel.policy import (
    MissionPolicy,
    MissionPolicyStore,
    canonical_bytes,
    sign_envelope,
)

DEV_SECRET = "dev-only-shared-secret"
ISSUER_KEY = b"governor-owned-hmac-key-0123456789"


def make_bundle(**overrides):
    # Empty capability/domain lists impose no restriction; the dedicated
    # security-regression suite exercises the restrictive case.
    bundle = {
        "mission_id": "mission-7",
        "objective": "authorise bounded tool calls",
        "invariants": ["never_shell_out"],
        "allowed_domains": [],
        "capabilities": [],
        "budgets": {},
        "standing_mandates": [],
        "policy_version": "1.0.0",
    }
    bundle.update(overrides)
    return bundle


def enforced_policy(tmp_path, *, version="1.0.0", mandates=(), budgets=None, name="policy.json"):
    path = tmp_path / name
    envelope = sign_envelope(
        make_bundle(
            policy_version=version,
            standing_mandates=list(mandates),
            budgets=dict(budgets or {}),
        ),
        secret=DEV_SECRET,
    )
    path.write_text(json.dumps(envelope), encoding="utf-8")
    return MissionPolicyStore(secret=DEV_SECRET, allow_dev_hmac=True).load(path)


@pytest.fixture
def issuer():
    return PermitIssuer(ISSUER_KEY)


@pytest.fixture
def policy(tmp_path):
    return enforced_policy(tmp_path)


@pytest.fixture
def service(policy, issuer):
    return PermitService(issuer, policy)


class TestPermitShape:
    def test_field_set_and_frozen(self):
        permit = ExecutionPermit(permit_id="p1")
        assert PERMIT_FIELDS == (
            "permit_id",
            "nonce",
            "goal_id",
            "plan_id",
            "step_id",
            "tool",
            "op",
            "capability",
            "normalized_args_hash",
            "manifest_digest",
            "policy_digest",
            "state_version",
            "policy_version",
            "risk_class",
            "budget_reservation",
            "issued_at",
            "expires_at",
            "idempotency_key",
            "instance_id",
            "mac",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            permit.tool = "memory"

    def test_payload_excludes_the_mac(self):
        permit = ExecutionPermit(permit_id="p1", nonce="n", mac="deadbeef")
        assert "mac" not in permit.payload()
        assert permit.as_dict()["mac"] == "deadbeef"


class TestIssuance:
    def test_issue_binds_tool_args_policy_and_risk(self, service, policy):
        permit = service.issue(
            tool="state",
            op="set_load_target",
            args={"load": 0.5},
            goal_id="goal-1",
            plan_id="plan-1",
            step_id="step-1",
            capability="state.set_load_target",
        )
        assert isinstance(permit, ExecutionPermit)
        assert permit.tool == "state"
        assert permit.goal_id == "goal-1"
        assert permit.plan_id == "plan-1"
        assert permit.step_id == "step-1"
        assert permit.capability == "state.set_load_target"
        assert permit.risk_class == "A1"
        assert permit.normalized_args_hash == args_hash({"load": 0.5})
        assert permit.policy_version == policy.policy_version
        assert permit.issued_at and permit.expires_at
        assert permit.idempotency_key
        assert permit.mac

    def test_capability_defaults_to_tool_dot_op(self, service):
        permit = service.issue(tool="state", op="set_load_target", args={"load": 0.5},
                               goal_id="goal-1")
        assert permit.capability == "state.set_load_target"

    def test_unknown_tool_is_denied(self, service):
        with pytest.raises(PermitDenied):
            service.issue(tool="shell", op="run", args={}, goal_id="goal-1")

    def test_off_schema_args_are_denied(self, service):
        with pytest.raises(PermitDenied):
            service.issue(tool="state", op="set_load_target", args={"load": 99.0},
                          goal_id="goal-1")
        with pytest.raises(PermitDenied):
            service.issue(tool="state", op="set_load_target", args={"load": 0.5, "x": 1},
                          goal_id="goal-1")

    def test_observation_only_policy_issues_nothing(self, issuer):
        service = PermitService(issuer, MissionPolicy.observation_only("no policy"))
        with pytest.raises(PermitDenied):
            service.issue(tool="telemetry", op="sample", args={}, goal_id="goal-1")

    def test_goal_id_is_required(self, service):
        with pytest.raises(PermitDenied):
            service.issue(tool="telemetry", op="sample", args={}, goal_id="")

    def _costed_service(self, tmp_path, issuer, *, budgets, name):
        from reflective_loop_runtime.kernel.manifest import (
            CostModel,
            FieldSpec,
            OperationSpec,
            ToolManifest,
            ToolSpec,
        )

        manifest = ToolManifest(
            [
                ToolSpec(
                    "meter",
                    {
                        "use": OperationSpec(
                            name="use",
                            input_schema={"n": FieldSpec("int", min=0, max=100)},
                            effects=("one metered call is consumed",),
                            risk_class="A1",
                            reversibility="reversible",
                            idempotency="idempotent",
                            compensation="refund the metered call",
                            cost_model=CostModel("api_calls", 1.0, "one metered call"),
                        )
                    },
                )
            ]
        )
        policy = enforced_policy(tmp_path, budgets=budgets, name=name)
        budget = BudgetService(tmp_path / f"{name}.db", policy)
        service = PermitService(issuer, policy, budget=budget, manifest=manifest)
        return budget, service

    def test_budget_is_reserved_from_the_manifest(self, tmp_path, issuer):
        budget, service = self._costed_service(
            tmp_path, issuer, budgets={"api_calls": 10}, name="b1.json"
        )
        permit = service.issue(tool="meter", op="use", args={"n": 1}, goal_id="goal-1")
        assert permit.budget_reservation
        # The amount comes from the manifest cost model, and is charged at every
        # level of the hierarchy.
        assert budget.scope_usage("goal", "goal-1", "api_calls") == 1
        assert budget.scope_usage("permit", permit.permit_id, "api_calls") == 1

    def test_caller_supplied_amounts_are_refused(self, tmp_path, issuer):
        budget, service = self._costed_service(
            tmp_path, issuer, budgets={"api_calls": 10}, name="b2.json"
        )
        with pytest.raises(PermitDenied):
            service.issue(
                tool="meter", op="use", args={"n": 1}, goal_id="goal-1",
                amounts={"api_calls": 1},
            )

    def test_budget_cap_blocks_issuance(self, tmp_path, issuer):
        budget, service = self._costed_service(
            tmp_path, issuer, budgets={"api_calls": 1}, name="b3.json"
        )
        service.issue(tool="meter", op="use", args={"n": 1}, goal_id="goal-1")
        with pytest.raises(BudgetExceeded):
            service.issue(tool="meter", op="use", args={"n": 1}, goal_id="goal-1")


class TestGovernorOnlyIssuance:
    def test_issuer_never_exposes_its_key(self, issuer):
        import gc

        assert not hasattr(issuer, "key")
        assert not hasattr(issuer, "hmac_key")
        assert ISSUER_KEY.decode() not in repr(issuer)
        assert ISSUER_KEY.decode() not in str(issuer)
        # The key is not merely private: no attribute of the issuer holds it, so
        # there is no ``_key`` slot to walk to and the GC referent graph exposes
        # nothing either -- it lives only inside opaque ``hashlib`` state.
        assert "_key" not in dir(issuer)
        assert ISSUER_KEY not in [
            referent
            for referent in gc.get_referents(issuer)
            if isinstance(referent, (bytes, bytearray))
        ]

    def test_service_has_no_public_issuer_or_key_attribute(self, service):
        assert not hasattr(service, "issuer")
        assert not hasattr(service, "key")
        assert not hasattr(service, "hmac_key")
        assert "issuer" not in {name for name in dir(service) if not name.startswith("_")}

    def test_a_forged_issuer_cannot_sign_for_the_real_one(self, policy, issuer):
        forger = PermitIssuer(b"a-different-key")
        forged = PermitService(forger, policy)
        permit = forged.issue(tool="telemetry", op="sample", args={}, goal_id="goal-1")
        real = PermitService(issuer, policy)
        with pytest.raises(PermitVerificationError):
            real.verify_and_consume(permit, "telemetry", {}, op="sample")


class TestVerification:
    def _issued(self, service, **overrides):
        params = dict(tool="state", op="set_load_target", args={"load": 0.5}, goal_id="goal-1")
        params.update(overrides)
        return service.issue(**params)

    def test_valid_permit_is_consumed_once(self, service):
        permit = self._issued(service)
        service.verify_and_consume(permit, "state", {"load": 0.5}, op="set_load_target")
        assert service.consumed(permit.nonce)

    def test_replay_is_rejected(self, service):
        permit = self._issued(service)
        service.verify_and_consume(permit, "state", {"load": 0.5}, op="set_load_target")
        with pytest.raises(PermitReplayError):
            service.verify_and_consume(permit, "state", {"load": 0.5}, op="set_load_target")

    def test_argument_substitution_is_rejected(self, service):
        permit = self._issued(service)
        with pytest.raises(PermitVerificationError):
            service.verify_and_consume(permit, "state", {"load": 0.9}, op="set_load_target")
        assert not service.consumed(permit.nonce)

    def test_tool_substitution_is_rejected(self, service):
        permit = self._issued(service)
        with pytest.raises(PermitVerificationError):
            service.verify_and_consume(permit, "telemetry", {"load": 0.5}, op="set_load_target")

    def test_expired_permit_is_rejected(self, service):
        permit = self._issued(service, ttl_seconds=-1)
        with pytest.raises(PermitVerificationError):
            service.verify_and_consume(permit, "state", {"load": 0.5}, op="set_load_target")
        assert not service.consumed(permit.nonce)

    def test_tampered_mac_is_rejected(self, service):
        permit = self._issued(service)
        tampered = dataclasses.replace(permit, risk_class="A0")
        with pytest.raises(PermitVerificationError):
            service.verify_and_consume(tampered, "state", {"load": 0.5}, op="set_load_target")
        assert not service.consumed(permit.nonce)

    def test_wrong_policy_version_is_rejected(self, tmp_path, issuer):
        old = enforced_policy(tmp_path, version="1.0.0", name="old.json")
        new = enforced_policy(tmp_path, version="2.0.0", name="new.json")
        service = PermitService(issuer, old)
        permit = service.issue(tool="state", op="set_load_target", args={"load": 0.5},
                               goal_id="goal-1")
        # A permit is bound to the policy version in force when it was minted, so
        # the same service adopting a newer policy rejects it. The refusal is
        # reached within one instance because a permit is refused outright by any
        # other one (see TestPermitInstanceBinding).
        service.set_policy(new)
        with pytest.raises(PermitVerificationError, match="policy version"):
            service.verify_and_consume(permit, "state", {"load": 0.5}, op="set_load_target")
        assert not service.consumed(permit.nonce)

    def test_non_permit_is_rejected(self, service):
        with pytest.raises(PermitVerificationError):
            service.verify_and_consume({"permit_id": "x"}, "state", {}, op="set_load_target")


class TestA3Mandate:
    def test_a3_without_mandate_is_denied(self, service):
        with pytest.raises(PermitDenied):
            service.issue(tool="memory", op="forget", args={"memory_id": 1}, goal_id="goal-1")

    def test_a3_with_matching_mandate_is_allowed(self, tmp_path, issuer):
        policy = enforced_policy(tmp_path, mandates=["memory.forget"], name="mandate.json")
        service = PermitService(issuer, policy)
        permit = service.issue(tool="memory", op="forget", args={"memory_id": 1},
                               goal_id="goal-1")
        assert permit.risk_class == "A3"
        service.verify_and_consume(permit, "memory", {"memory_id": 1}, op="forget")

    def test_a3_mandate_must_match_the_operation(self, tmp_path, issuer):
        policy = enforced_policy(tmp_path, mandates=["memory.remember"], name="m2.json")
        service = PermitService(issuer, policy)
        with pytest.raises(PermitDenied):
            service.issue(tool="memory", op="forget", args={"memory_id": 1}, goal_id="goal-1")

    def test_non_a3_needs_no_mandate(self, service):
        permit = service.issue(tool="memory", op="recall", args={"query": "x"}, goal_id="goal-1")
        assert permit.risk_class == "A0"


def test_args_hash_is_deterministic_and_key_sensitive():
    assert args_hash({"b": 1, "a": 2}) == args_hash({"a": 2, "b": 1})
    assert args_hash({"a": 1}) != args_hash({"a": 2})
    assert args_hash(None) == args_hash({})


class _MutableClock:
    """A controllable clock for exercising exact expiry boundaries."""

    def __init__(self, moment: datetime):
        self.now = moment

    def __call__(self):
        return self.now.isoformat()

    def set(self, moment: datetime) -> None:
        self.now = moment


class _MutableMonotonic:
    """A controllable monotonic counter for driving elapsed process time.

    :func:`time.monotonic` is the one clock a wall-clock change cannot step, so
    the tests drive it directly instead of sleeping.
    """

    def __init__(self, seconds: float = 0.0):
        self.seconds = float(seconds)

    def __call__(self) -> float:
        return self.seconds

    def advance(self, seconds: float) -> None:
        self.seconds += float(seconds)


class TestPermitHardening:
    """The nine hardening properties, each checked by a dedicated test."""

    def _issued(self, service, **overrides):
        params = dict(
            tool="state", op="set_load_target", args={"load": 0.5}, goal_id="goal-1"
        )
        params.update(overrides)
        return service.issue(**params)

    # -- 1. domain-separated, length-prefixed MAC payload -------------------
    def test_mac_payload_is_domain_separated_and_length_prefixed(self, service, issuer):
        from reflective_loop_runtime.kernel.permit import (
            _PERMIT_MAC_DOMAIN,
            _UNSIGNED_FIELDS,
            _signed_bytes,
        )

        permit = self._issued(service)
        signed = _signed_bytes(permit)
        assert permit.mac == issuer.sign(signed)

        # Domain separation: the first length-prefixed chunk is the domain tag,
        # so the MAC could not be replayed for another object.
        head = len(_PERMIT_MAC_DOMAIN).to_bytes(8, "big") + _PERMIT_MAC_DOMAIN
        assert signed.startswith(head)
        assert permit.mac != issuer.sign(canonical_bytes(permit.payload()))

        # Length prefixes: every chunk decodes back to exactly one field part.
        position = len(head)
        chunks = []
        while position < len(signed):
            size = int.from_bytes(signed[position : position + 8], "big")
            chunks.append(signed[position + 8 : position + 8 + size])
            position += 8 + size
        assert position == len(signed)
        names = [chunk.decode("utf-8") for chunk in chunks[0::2]]
        assert tuple(names) == _UNSIGNED_FIELDS

    # -- 2. compare_digest first, before any DB work ------------------------
    def test_mac_is_checked_before_any_db_work(self, service, monkeypatch):
        permit = self._issued(service)
        tampered = dataclasses.replace(permit, mac="0" * 64)
        reached = []

        def _fail(*args, **kwargs):
            reached.append(args)
            raise AssertionError("nonce transaction ran before the MAC check")

        monkeypatch.setattr(service, "_consume_nonce", _fail)
        with pytest.raises(PermitVerificationError):
            service.verify_and_consume(tampered, "state", {"load": 0.5}, op="set_load_target")
        assert reached == []

    def test_non_ascii_mac_is_a_verification_failure_not_an_exception(self, service):
        permit = self._issued(service)
        with pytest.raises(PermitVerificationError):
            service.verify_and_consume(
                dataclasses.replace(permit, mac="\u00e9" * 64), "state", {"load": 0.5},
                op="set_load_target",
            )
        assert not service.consumed(permit.nonce)

    # -- 3. canonical args JSON rejects NaN, Infinity, non-string keys ------
    def test_args_hash_rejects_nan_infinity_and_non_string_keys(self):
        for bad in (
            {"load": float("nan")},
            {"load": float("inf")},
            {"load": float("-inf")},
            {1: "x"},
            {"outer": {2: "x"}},
            {"outer": [float("nan")]},
        ):
            with pytest.raises(PermitError):
                args_hash(bad)

    # -- 4. atomic burn under concurrency -----------------------------------
    def test_concurrent_consumers_yield_exactly_one_winner(self, tmp_path, issuer, policy):
        service = PermitService(issuer, policy, db_path=tmp_path / "permits.db")
        permit = self._issued(service)
        workers = 8
        barrier = threading.Barrier(workers)
        outcomes: list[str] = []

        def consume():
            barrier.wait()
            try:
                service.verify_and_consume(permit, "state", {"load": 0.5}, op="set_load_target")
                outcomes.append("winner")
            except PermitReplayError:
                outcomes.append("replay")

        threads = [threading.Thread(target=consume) for _ in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert outcomes.count("winner") == 1
        assert outcomes.count("replay") == workers - 1
        assert service.consumed(permit.nonce)

    # -- 5. A3 mandate lookup inside the nonce transaction ------------------
    def test_a3_mandate_is_rechecked_in_the_consumption_transaction(self, tmp_path, issuer):
        with_mandate = enforced_policy(
            tmp_path, mandates=["memory.forget"], name="a3-with.json"
        )
        without = enforced_policy(tmp_path, version="1.0.0", mandates=[], name="a3-without.json")
        service = PermitService(issuer, with_mandate, db_path=tmp_path / "a3.db")
        permit = self._issued(
            service, tool="memory", op="forget", args={"memory_id": 1}
        )
        assert permit.risk_class == "A3"

        service.set_policy(without)
        with pytest.raises(PermitVerificationError):
            service.verify_and_consume(permit, "memory", {"memory_id": 1}, op="forget")
        assert not service.consumed(permit.nonce)

    # -- 6. no nonce consumed when any check fails --------------------------
    def test_no_nonce_is_consumed_when_any_check_fails(self, service):
        permit = self._issued(service)
        with pytest.raises(PermitVerificationError):
            service.verify_and_consume(permit, "telemetry", {"load": 0.5}, op="set_load_target")  # wrong tool
        assert not service.consumed(permit.nonce)

        with pytest.raises(PermitVerificationError):
            service.verify_and_consume(permit, "state", {"load": 0.9}, op="set_load_target")  # wrong args
        assert not service.consumed(permit.nonce)

        with pytest.raises(PermitVerificationError):
            service.verify_and_consume(
                permit, "state", {"load": 0.5}, op="set_load_target", expected_state_version="other"
            )
        assert not service.consumed(permit.nonce)

        # The intact permit still works after every failed attempt.
        service.verify_and_consume(permit, "state", {"load": 0.5}, op="set_load_target")
        assert service.consumed(permit.nonce)

    # -- 7. invalid at exactly expires_at, injectable clock -----------------
    def test_permit_is_invalid_at_exactly_expires_at(self, policy, issuer, tmp_path):
        start = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        clock = _MutableClock(start)
        # Its own ledger, like TestClockRollback below. The high-water mark is
        # persistent on purpose -- it is what makes expiry monotonic across a
        # restart -- so a fake clock parked in January would otherwise leave a
        # mark in the shared default database that expires every permit this
        # file mints, in the next test of the class as much as in the next run
        # of the suite. The boundary under test is the permit's, not the
        # ledger's, so the ledger is the test's to throw away.
        service = PermitService(issuer, policy, clock=clock, db_path=tmp_path / "permits.db")
        permit = self._issued(service, ttl_seconds=10)
        expiry = datetime.fromisoformat(permit.expires_at)
        assert expiry == start + timedelta(seconds=10)

        clock.set(expiry - timedelta(microseconds=1))
        service.verify_and_consume(permit, "state", {"load": 0.5}, op="set_load_target")
        assert service.consumed(permit.nonce)

        clock.set(start)
        boundary = self._issued(service, ttl_seconds=10)
        assert datetime.fromisoformat(boundary.expires_at) == expiry
        clock.set(expiry)
        with pytest.raises(PermitVerificationError):
            service.verify_and_consume(boundary, "state", {"load": 0.5}, op="set_load_target")
        assert not service.consumed(boundary.nonce)

    # -- 8. budget_reservation and state_version are MAC-covered ------------
    def test_budget_reservation_and_state_version_are_mac_covered(self, service):
        permit = self._issued(service, state_version="sv-1")
        assert permit.state_version == "sv-1"
        for field_name, value in (
            ("state_version", "sv-2"),
            ("budget_reservation", "reservation-9"),
        ):
            tampered = dataclasses.replace(permit, **{field_name: value})
            with pytest.raises(PermitVerificationError):
                service.verify_and_consume(tampered, "state", {"load": 0.5}, op="set_load_target")
            assert not service.consumed(permit.nonce)

    # -- 9. optional expected_state_version check ---------------------------
    def test_optional_expected_state_version_check(self, service):
        permit = self._issued(service, state_version="sv-7")
        with pytest.raises(PermitVerificationError):
            service.verify_and_consume(
                permit, "state", {"load": 0.5}, op="set_load_target", expected_state_version="sv-8"
            )
        assert not service.consumed(permit.nonce)

        service.verify_and_consume(
            permit, "state", {"load": 0.5}, op="set_load_target", expected_state_version="sv-7"
        )
        assert service.consumed(permit.nonce)

    def test_expected_state_version_is_optional(self, service):
        permit = self._issued(service, state_version="anything")
        service.verify_and_consume(permit, "state", {"load": 0.5}, op="set_load_target")
        assert service.consumed(permit.nonce)


class TestClockRollback:
    """Expiry must hold when the wall clock is rolled back (expiry is monotonic).

    The wall clock is attacker-reachable, so a permit may not be judged by it
    alone: the permit store keeps a high-water mark of the latest moment it ever
    observed, an in-process anchor advances that mark by elapsed monotonic time,
    expiry is checked against the effective now, and a clock further behind it
    than the tolerated skew fails closed.
    """

    START = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    Y2K = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    def _service(self, tmp_path, issuer, policy, clock, name="permits.db", monotonic=None):
        extra = {} if monotonic is None else {"monotonic": monotonic}
        return PermitService(issuer, policy, clock=clock, db_path=tmp_path / name, **extra)

    def _issued(self, service, **overrides):
        params = dict(
            tool="state", op="set_load_target", args={"load": 0.5}, goal_id="goal-1"
        )
        params.update(overrides)
        return service.issue(**params)

    # (a) an expired permit stays expired after the clock is rolled back.
    def test_expired_permit_stays_rejected_after_the_clock_is_rolled_back(
        self, tmp_path, issuer, policy
    ):
        clock = _MutableClock(self.START)
        service = self._service(tmp_path, issuer, policy, clock)
        permit = self._issued(service, ttl_seconds=10)
        expiry = datetime.fromisoformat(permit.expires_at)

        # Walk past the expiry normally: the rejection also raises the mark.
        clock.set(expiry + timedelta(seconds=1))
        with pytest.raises(PermitVerificationError, match="expired"):
            service.verify_and_consume(permit, "state", {"load": 0.5}, op="set_load_target")

        # Now roll the wall clock back to the year 2000 and try again: the same
        # permit must not be resurrected by the older clock.
        clock.set(self.Y2K)
        with pytest.raises(PermitVerificationError):
            service.verify_and_consume(permit, "state", {"load": 0.5}, op="set_load_target")
        assert not service.consumed(permit.nonce)

    # (b) a rollback beyond the tolerated skew fails closed, distinctly.
    def test_rollback_beyond_the_skew_fails_closed_with_a_clock_error(
        self, tmp_path, issuer, policy
    ):
        clock = _MutableClock(self.START)
        service = self._service(tmp_path, issuer, policy, clock)
        permit = self._issued(service, ttl_seconds=3600)
        service.verify_and_consume(permit, "state", {"load": 0.5}, op="set_load_target")

        # The permit is still live at the mark, but the clock now reads an hour
        # earlier: that is a rollback, not jitter, and it is refused outright.
        fresh = self._issued(service, ttl_seconds=3600)
        clock.set(self.START - timedelta(seconds=3600))
        with pytest.raises(PermitClockError, match="clock rollback"):
            service.verify_and_consume(
                fresh, "state", {"load": 0.5}, op="set_load_target"
            )
        assert not service.consumed(fresh.nonce)
        # The refusal is an ordinary permit rejection: it stays inside the
        # existing hierarchy and therefore the ordinary record/audit path.
        assert isinstance(PermitClockError("x"), PermitVerificationError)
        assert isinstance(PermitClockError("x"), PermitError)

    # (c) jitter well inside the skew is tolerated: a valid permit still works.
    def test_small_backwards_jitter_still_verifies_a_valid_permit(
        self, tmp_path, issuer, policy
    ):
        clock = _MutableClock(self.START)
        service = self._service(tmp_path, issuer, policy, clock)
        permit = self._issued(service, ttl_seconds=600)
        expiry = datetime.fromisoformat(permit.expires_at)

        # Push the mark forward, then step the clock 120s back: inside the skew.
        clock.set(self.START + timedelta(seconds=300))
        clock.set(self.START + timedelta(seconds=180))
        assert expiry > self.START + timedelta(seconds=180)

        service.verify_and_consume(permit, "state", {"load": 0.5}, op="set_load_target")
        assert service.consumed(permit.nonce)

    # (d) the mark outlives the service that wrote it.
    def test_high_water_mark_persists_across_a_new_service_on_the_same_database(
        self, tmp_path, issuer, policy
    ):
        import sqlite3

        clock = _MutableClock(self.START)
        db_path = tmp_path / "permits.db"
        first = PermitService(issuer, policy, clock=clock, db_path=db_path)
        permit = self._issued(first, ttl_seconds=3600)
        first.verify_and_consume(permit, "state", {"load": 0.5}, op="set_load_target")
        first.close()

        with sqlite3.connect(db_path) as connection:
            row = connection.execute("SELECT observed_at FROM clock_hwm").fetchone()
        assert row is not None
        assert datetime.fromisoformat(row[0]) == self.START

        # A fresh service on the same ledger still knows how far time had run:
        # the clock it is handed here reads a whole year earlier than the mark,
        # and even a permit it minted itself cannot be authorised under it.
        rolled_back = _MutableClock(self.START - timedelta(days=365))
        second = PermitService(issuer, policy, clock=rolled_back, db_path=db_path)
        probe = self._issued(second, ttl_seconds=3600)
        with pytest.raises(PermitClockError, match="clock rollback"):
            second.verify_and_consume(probe, "state", {"load": 0.5}, op="set_load_target")
        assert not second.consumed(probe.nonce)
        second.close()


    # (e) F2: an idle store must not hand a dead permit back its life.
    def test_idle_process_time_rejects_a_permit_revived_by_a_backwards_step(
        self, tmp_path, issuer, policy
    ):
        clock = _MutableClock(self.START)
        monotonic = _MutableMonotonic()
        service = self._service(tmp_path, issuer, policy, clock, "idle.db", monotonic)
        permit = self._issued(service, ttl_seconds=10)
        expiry = datetime.fromisoformat(permit.expires_at)
        assert expiry == self.START + timedelta(seconds=10)

        # The store then sits idle for 20s of process time: nothing issues,
        # verifies, or reads the permit database, so the persisted mark still
        # reads the instant of issue and the permit is ten seconds dead.
        monotonic.advance(20.0)

        # Stepping the wall clock back 25s is inside the tolerated skew -- not a
        # rollback the clock check would catch -- and used to revive the permit.
        clock.set(self.START - timedelta(seconds=25))
        with pytest.raises(PermitVerificationError, match="expired"):
            service.verify_and_consume(permit, "state", {"load": 0.5}, op="set_load_target")
        assert not service.consumed(permit.nonce)

    def test_idle_process_time_expires_a_permit_with_no_clock_step_at_all(
        self, tmp_path, issuer, policy
    ):
        clock = _MutableClock(self.START)
        monotonic = _MutableMonotonic()
        service = self._service(tmp_path, issuer, policy, clock, "elapsed.db", monotonic)
        short = self._issued(service, ttl_seconds=10)
        long = self._issued(service, ttl_seconds=120)

        # The wall clock never moves: only elapsed process time does.
        monotonic.advance(30.0)
        with pytest.raises(PermitVerificationError, match="expired"):
            service.verify_and_consume(short, "state", {"load": 0.5}, op="set_load_target")
        assert not service.consumed(short.nonce)

        # A permit still inside its life at the same elapsed time is unaffected.
        service.verify_and_consume(long, "state", {"load": 0.5}, op="set_load_target")
        assert service.consumed(long.nonce)

    def test_effective_time_cannot_be_walked_back_by_a_second_clock_step(
        self, tmp_path, issuer, policy
    ):
        clock = _MutableClock(self.START)
        monotonic = _MutableMonotonic()
        service = self._service(tmp_path, issuer, policy, clock, "rewind.db", monotonic)
        permit = self._issued(service, ttl_seconds=10)

        monotonic.advance(20.0)
        clock.set(self.START - timedelta(seconds=25))
        with pytest.raises(PermitVerificationError, match="expired"):
            service.verify_and_consume(permit, "state", {"load": 0.5}, op="set_load_target")

        # Rolling the clock back must not lower the effective time: stepping it
        # forward again to a moment before the expiry does not resurrect a
        # permit that already died, however the clock is then moved.
        clock.set(self.START + timedelta(seconds=5))
        with pytest.raises(PermitVerificationError, match="expired"):
            service.verify_and_consume(permit, "state", {"load": 0.5}, op="set_load_target")
        assert not service.consumed(permit.nonce)

    # (f) F3: the mark follows the instant of issue, not just the clock reading.
    def test_mark_records_the_instant_of_issue_ahead_of_the_clock(
        self, tmp_path, issuer, policy
    ):
        import sqlite3

        clock = _MutableClock(self.START)
        monotonic = _MutableMonotonic()
        service = self._service(tmp_path, issuer, policy, clock, "issued-at.db", monotonic)
        issued_at = self.START + timedelta(seconds=200)
        permit = self._issued(service, ttl_seconds=10, now=issued_at)
        assert permit.issued_at == issued_at.isoformat()

        # Minting is an observation of time, and the observation is the instant
        # minted -- the local clock is 200s behind it and must not decide.
        with sqlite3.connect(tmp_path / "issued-at.db") as connection:
            row = connection.execute("SELECT observed_at FROM clock_hwm").fetchone()
        assert datetime.fromisoformat(row[0]) == issued_at

        # So a later clock, stepped back 150s -- ordinary skew, not a rollback --
        # still cannot hand this permit the life its own instant already spent.
        clock.set(self.START + timedelta(seconds=50))
        monotonic.advance(15.0)
        with pytest.raises(PermitVerificationError, match="expired"):
            service.verify_and_consume(permit, "state", {"load": 0.5}, op="set_load_target")
        assert not service.consumed(permit.nonce)

    # (g) F9: an untrusted clock is reported as a rollback, not as an expiry.
    def test_clock_error_is_not_masked_by_the_expiry_diagnosis(
        self, tmp_path, issuer, policy
    ):
        clock = _MutableClock(self.START)
        monotonic = _MutableMonotonic()
        service = self._service(tmp_path, issuer, policy, clock, "diagnosis.db", monotonic)
        permit = self._issued(service, ttl_seconds=10)

        # Let the permit die normally, so "expired" alone would be a sufficient
        # -- and uninformative -- diagnosis for what follows.
        monotonic.advance(60.0)
        with pytest.raises(PermitVerificationError, match="expired"):
            service.verify_and_consume(permit, "state", {"load": 0.5}, op="set_load_target")

        # With the clock an hour behind, expiry is guaranteed; the finding is the
        # clock, so the clock error must be the one that surfaces.
        clock.set(self.START - timedelta(seconds=3600))
        with pytest.raises(PermitClockError, match="clock rollback"):
            service.verify_and_consume(permit, "state", {"load": 0.5}, op="set_load_target")
        assert not service.consumed(permit.nonce)


class TestPermitInstanceBinding:
    """A permit is bound to the one service instance that minted it.

    No clock a process can read measures the time that process spent dead. The
    persisted mark is a *floor*, not a measurement: after a restart the mark
    enters the effective now as a bare timestamp, so a wall clock parked on the
    instant of issue gives ``rollback == 0``, raises no clock error, and would
    hand back the life of a permit that died hours ago. The gap is therefore
    closed structurally, not by another clock reading: every permit carries the
    random identity of its issuer under the MAC, and any other instance refuses
    it before its age is ever considered.
    """

    START = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

    def _issued(self, service, **overrides):
        params = dict(
            tool="state", op="set_load_target", args={"load": 0.5}, goal_id="goal-1"
        )
        params.update(overrides)
        return service.issue(**params)

    def _service(self, tmp_path, issuer, policy, clock, name="permits.db", monotonic=None):
        extra = {} if monotonic is None else {"monotonic": monotonic}
        return PermitService(issuer, policy, clock=clock, db_path=tmp_path / name, **extra)

    def test_a_restarted_service_refuses_the_permits_of_the_instance_it_replaced(
        self, tmp_path, issuer, policy
    ):
        clock = _MutableClock(self.START)
        db_path = tmp_path / "restart.db"
        before = self._service(tmp_path, issuer, policy, clock, db_path.name,
                                _MutableMonotonic())
        permit = self._issued(before, ttl_seconds=5)
        assert permit.expires_at == (self.START + timedelta(seconds=5)).isoformat()
        before.close()  # the restart: a new process over the same ledger

        # 2467s of real time pass while the wall clock stays where the mark is.
        # The restarted process has no monotonic anchor to advance time with and
        # nothing on this machine observes the downtime, so its effective now is
        # the mark, the rollback is zero and no clock error is raised: expiry
        # alone would accept this permit, hours past its five-second life.
        monotonic = _MutableMonotonic()
        after = self._service(tmp_path, issuer, policy, clock, "restart.db", monotonic)
        monotonic.advance(2467.0)
        with pytest.raises(PermitVerificationError, match="previous service instance"):
            after.verify_and_consume(permit, "state", {"load": 0.5}, op="set_load_target")
        assert not after.consumed(permit.nonce)
        after.close()

    def test_a_second_service_on_the_same_ledger_refuses_the_permit(
        self, tmp_path, issuer, policy
    ):
        clock = _MutableClock(self.START)
        db_path = tmp_path / "shared.db"
        first = self._service(tmp_path, issuer, policy, clock, db_path.name)
        second = self._service(tmp_path, issuer, policy, clock, db_path.name)
        assert first.instance_id != second.instance_id

        permit = self._issued(first, ttl_seconds=3600)
        with pytest.raises(PermitVerificationError, match="previous service instance"):
            second.verify_and_consume(permit, "state", {"load": 0.5}, op="set_load_target")
        assert not second.consumed(permit.nonce)

        # It is the *issuer*, not the ledger, that is refused: the second service
        # mints and consumes its own permits over the very same database.
        own = self._issued(second, ttl_seconds=3600)
        second.verify_and_consume(own, "state", {"load": 0.5}, op="set_load_target")
        assert second.consumed(own.nonce)
        assert own.instance_id == second.instance_id
        first.close()
        second.close()

    def test_tampering_with_the_instance_field_breaks_the_mac(self, service):
        permit = self._issued(service)
        assert permit.instance_id == service.instance_id
        assert permit.payload()["instance_id"] == service.instance_id
        assert permit.as_dict()["instance_id"] == service.instance_id

        # One guaranteed-different id, so the case cannot flake on a hex
        # string that happens to contain no letters.
        other = ("1" if permit.instance_id[0] != "1" else "2")
        other += permit.instance_id[1:]
        for forged in ("", "0" * 32, other):
            tampered = dataclasses.replace(permit, instance_id=forged)
            with pytest.raises(PermitVerificationError, match="MAC"):
                service.verify_and_consume(
                    tampered, "state", {"load": 0.5}, op="set_load_target"
                )
        assert not service.consumed(permit.nonce)

    def test_the_instance_is_checked_before_the_clock(self, tmp_path, issuer, policy):
        clock = _MutableClock(self.START)
        db_path = tmp_path / "order.db"
        first = self._service(tmp_path, issuer, policy, clock, db_path.name)
        permit = self._issued(first, ttl_seconds=3600)
        first.close()

        # Even where the wall clock is untrusted the diagnosis is the instance:
        # the permit is refused as another instance's, so a caller cannot mistake
        # a restart for a clock fault (or the other way round).
        rolled_back = _MutableClock(self.START - timedelta(days=365))
        second = self._service(tmp_path, issuer, policy, rolled_back, db_path.name)
        with pytest.raises(PermitVerificationError, match="previous service instance") as caught:
            second.verify_and_consume(permit, "state", {"load": 0.5}, op="set_load_target")
        assert not isinstance(caught.value, PermitClockError)
        assert not second.consumed(permit.nonce)
        second.close()

    def test_a_permit_with_no_instance_is_refused(self, service):
        # A permit minted before the binding existed carries an empty instance
        # id, and fails closed rather than being treated as untagged.
        permit = self._issued(service)
        legacy = dataclasses.replace(permit, instance_id="")
        with pytest.raises(PermitVerificationError, match="MAC"):
            service.verify_and_consume(legacy, "state", {"load": 0.5}, op="set_load_target")

        # Re-signed by the Governor it owns, the same permit is still not this
        # instance's: the MAC check cannot be used to smuggle one in.
        from reflective_loop_runtime.kernel.permit import _permit_mac

        forged = dataclasses.replace(permit, instance_id="", mac="")
        forged = dataclasses.replace(forged, mac=_permit_mac(service._sign, forged))
        with pytest.raises(PermitVerificationError, match="previous service instance"):
            service.verify_and_consume(forged, "state", {"load": 0.5}, op="set_load_target")
        assert not service.consumed(permit.nonce)


class TestPolicyAuthorityIsSignatureOnly:
    """F10: a mission policy may not assert its own authority."""

    def test_set_policy_refuses_a_self_asserted_enforce_policy(self, service):
        # A policy object that claims ENFORCE but was never produced by a
        # verified operator signature carries no authority.
        forged = dataclasses.replace(
            MissionPolicy.observation_only("hand-built"), mode="ENFORCE"
        )
        assert forged.enforced is True
        assert forged.signed is False
        with pytest.raises(PermitDenied, match="signature"):
            service.set_policy(forged)
        # The original authority is unchanged and still mints permits.
        assert service.policy.enforced is True

    def test_set_policy_accepts_a_verified_policy_and_a_downgrade(self, service, tmp_path):
        verified = enforced_policy(tmp_path, version="2.0.0", name="verified.json")
        service.set_policy(verified)
        assert service.policy.policy_version == "2.0.0"

        # Downgrading to observation-only withholds authority and is allowed.
        service.set_policy(MissionPolicy.observation_only("kill switch engaged"))
        assert service.policy.enforced is False
        with pytest.raises(PermitDenied):
            service.issue(
                tool="state", op="set_load_target", args={"load": 0.5}, goal_id="goal-1"
            )

"""Hierarchical budget accounting: caps, reservation lifecycle, and racing.

The container used for CI may not have the optional ``cryptography`` package, so
policies are signed with the dev-only HMAC-SHA256 fallback. That still loads in
``ENFORCE`` mode, which is all the budget service needs to grant caps.
"""

from __future__ import annotations

import json
import threading

import pytest

from reflective_loop_runtime.kernel.budget import (
    RESOURCES,
    SCOPE_LEVELS,
    BudgetError,
    BudgetExceeded,
    BudgetService,
)
from reflective_loop_runtime.kernel.policy import (
    MissionPolicy,
    MissionPolicyStore,
    sign_envelope,
)

DEV_SECRET = "dev-only-shared-secret"


def make_bundle(**overrides):
    bundle = {
        "mission_id": "mission-7",
        "objective": "stay within the operator's budget",
        "invariants": ["never_shell_out"],
        "allowed_domains": ["telemetry"],
        "capabilities": ["observe"],
        "budgets": {},
        "standing_mandates": [],
        "policy_version": "1.0.0",
    }
    bundle.update(overrides)
    return bundle


def enforced_policy(tmp_path, budgets, *, version="1.0.0"):
    """Sign a bundle and load it through the read-only store, as the kernel does."""
    path = tmp_path / f"policy-{version}.json"
    envelope = sign_envelope(
        make_bundle(budgets=budgets, policy_version=version), secret=DEV_SECRET
    )
    path.write_text(json.dumps(envelope), encoding="utf-8")
    return MissionPolicyStore(secret=DEV_SECRET, allow_dev_hmac=True).load(path)


def service(tmp_path, budgets, *, name="budget.db", version="1.0.0"):
    policy = enforced_policy(tmp_path, budgets, version=version)
    assert policy.enforced
    return BudgetService(tmp_path / name, policy)


GLOBAL = {"global": "g"}


class TestCapsComeFromPolicy:
    def test_bare_resource_key_caps_every_level(self, tmp_path):
        budget = service(tmp_path, {"money": 100})
        assert budget.cap_for("global", "money") == 100
        assert budget.cap_for("mission", "money") == 100
        assert budget.cap_for("goal", "money") == 100
        assert budget.cap_for("permit", "money") == 100

    def test_level_scoped_keys_apply_to_that_level_only(self, tmp_path):
        budget = service(tmp_path, {"goal:llm_tokens": 10, "permit.api_calls": 5})
        assert budget.cap_for("goal", "llm_tokens") == 10
        assert budget.cap_for("global", "llm_tokens") is None
        assert budget.cap_for("permit", "api_calls") == 5
        assert budget.cap_for("goal", "api_calls") is None

    def test_unrelated_policy_metadata_is_ignored(self, tmp_path):
        budget = service(tmp_path, {"tokens_per_tick": 256, "vision_fps": 1.0})
        assert budget.caps() == {}

    def test_store_is_the_only_cap_source_and_has_no_write_api(self):
        for name in ("set_cap", "raise_cap", "update_cap", "add_cap", "set_budget"):
            assert not hasattr(BudgetService, name)

    def test_observation_only_policy_grants_nothing(self, tmp_path):
        budget = BudgetService(tmp_path / "obs.db", MissionPolicy.observation_only("no policy"))
        assert budget.cap_for("global", "money") == 0.0
        with pytest.raises(BudgetExceeded):
            budget.reserve(GLOBAL, {"money": 1})
        # An empty reservation still has nothing to charge.
        assert budget.reserve(GLOBAL, {})

    def test_resources_and_levels_are_the_documented_sets(self):
        assert set(RESOURCES) == {
            "money",
            "llm_tokens",
            "api_calls",
            "shell_cpu_seconds",
            "wall_clock_seconds",
            "scheduler_tick",
            "store_write",
            "store_read",
            "io_read",
            "message",
            "concurrency",
        }
        assert SCOPE_LEVELS == ("global", "mission", "goal", "permit")


class TestReservationLifecycle:
    def test_reserve_charges_every_provided_level(self, tmp_path):
        budget = service(tmp_path, {"money": 100})
        scopes = {"global": "g", "mission": "m", "goal": "goal-1"}
        reservation = budget.reserve(scopes, {"money": 25})
        assert isinstance(reservation, str) and reservation
        for level, scope_id in scopes.items():
            assert budget.scope_usage(level, scope_id, "money") == 25
        assert budget.reservation_state(reservation) == "reserved"

    def test_cap_is_enforced_per_level(self, tmp_path):
        budget = service(tmp_path, {"goal:llm_tokens": 10, "money": 100})
        scopes = {"global": "g", "goal": "goal-1"}
        budget.reserve(scopes, {"llm_tokens": 4})
        with pytest.raises(BudgetExceeded):
            budget.reserve(scopes, {"llm_tokens": 7})

    def test_a_rejected_reserve_charges_nothing(self, tmp_path):
        budget = service(tmp_path, {"goal:llm_tokens": 10})
        scopes = {"global": "g", "goal": "goal-1"}
        budget.reserve(scopes, {"llm_tokens": 4})
        with pytest.raises(BudgetExceeded):
            budget.reserve(scopes, {"llm_tokens": 7})
        assert budget.scope_usage("global", "g", "llm_tokens") == 4
        assert budget.scope_usage("goal", "goal-1", "llm_tokens") == 4

    def test_commit_refunds_the_remainder(self, tmp_path):
        budget = service(tmp_path, {"money": 100})
        reservation = budget.reserve(GLOBAL, {"money": 40})
        budget.commit(reservation, {"money": 12})
        assert budget.scope_usage("global", "g", "money") == 12
        assert budget.reservation_state(reservation) == "committed"

    def test_release_frees_everything(self, tmp_path):
        budget = service(tmp_path, {"money": 100})
        reservation = budget.reserve(GLOBAL, {"money": 40})
        budget.release(reservation)
        assert budget.scope_usage("global", "g", "money") == 0
        assert budget.reservation_state(reservation) == "released"

    def test_reservation_is_terminal(self, tmp_path):
        budget = service(tmp_path, {"money": 100})
        reservation = budget.reserve(GLOBAL, {"money": 40})
        budget.commit(reservation, {"money": 1})
        with pytest.raises(BudgetError):
            budget.commit(reservation, {"money": 1})
        with pytest.raises(BudgetError):
            budget.release(reservation)

    def test_unknown_and_malformed_operations_raise(self, tmp_path):
        budget = service(tmp_path, {"money": 100})
        with pytest.raises(BudgetError):
            budget.commit("nope", {"money": 0})
        with pytest.raises(BudgetError):
            budget.release("nope")
        with pytest.raises(BudgetError):
            budget.reserve(GLOBAL, {"unobtainium": 1})
        with pytest.raises(BudgetError):
            budget.reserve(GLOBAL, {"money": 0})
        with pytest.raises(BudgetError):
            budget.reserve({}, {"money": 1})
        with pytest.raises(BudgetError):
            budget.reserve({"bogus": "x"}, {"money": 1})

    def test_commit_cannot_claim_more_than_reserved(self, tmp_path):
        budget = service(tmp_path, {"money": 100})
        reservation = budget.reserve(GLOBAL, {"money": 5})
        with pytest.raises(BudgetExceeded):
            budget.commit(reservation, {"money": 6})
        assert budget.reservation_state(reservation) == "reserved"

    def test_commit_cannot_name_never_reserved_resource(self, tmp_path):
        budget = service(tmp_path, {"money": 100})
        reservation = budget.reserve(GLOBAL, {"money": 5})
        with pytest.raises(BudgetError):
            budget.commit(reservation, {"llm_tokens": 1})


class TestConcurrencyGauge:
    def test_concurrency_cap_bounds_in_flight_reservations(self, tmp_path):
        budget = service(tmp_path, {"concurrency": 2})
        first = budget.reserve(GLOBAL, {"concurrency": 1})
        second = budget.reserve(GLOBAL, {"concurrency": 1})
        with pytest.raises(BudgetExceeded):
            budget.reserve(GLOBAL, {"concurrency": 1})
        budget.release(second)
        third = budget.reserve(GLOBAL, {"concurrency": 1})
        assert third
        assert budget.scope_usage("global", "g", "concurrency") == 2
        budget.release(first)
        assert budget.scope_usage("global", "g", "concurrency") == 1

    def test_commit_releases_the_concurrency_gauge(self, tmp_path):
        budget = service(tmp_path, {"concurrency": 1})
        reservation = budget.reserve(GLOBAL, {"concurrency": 1})
        budget.commit(reservation, {"concurrency": 1})
        assert budget.scope_usage("global", "g", "concurrency") == 0
        assert budget.reserve(GLOBAL, {"concurrency": 1})


class TestBoundedDB:
    def test_memory_database_is_rejected(self):
        with pytest.raises(BudgetError):
            BudgetService(":memory:", MissionPolicy.observation_only("x"))


def test_fifty_threads_racing_never_exceed_a_cap(tmp_path):
    budget = service(tmp_path, {"money": 10})
    threads = []
    barrier = threading.Barrier(50)
    grants: list[str] = []
    denials: list[str] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        try:
            barrier.wait(timeout=30)
        except threading.BrokenBarrierError:  # pragma: no cover - defensive
            return
        try:
            reservation = budget.reserve(GLOBAL, {"money": 1})
        except BudgetExceeded:
            with lock:
                denials.append(f"racer-{index}")
        else:
            with lock:
                grants.append(reservation)

    for index in range(50):
        thread = threading.Thread(target=worker, args=(index,))
        threads.append(thread)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert len(grants) == 10
    assert len(denials) == 40
    assert len({budget.reservation_state(r) for r in grants}) == 1
    total = budget.scope_usage("global", "g", "money")
    assert total == 10
    assert total <= budget.cap_for("global", "money")

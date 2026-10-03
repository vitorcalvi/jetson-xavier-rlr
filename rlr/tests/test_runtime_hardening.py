import shutil

import pytest

from rlr.runtime import Config, ReflectiveLoop


class NoLLM:
    def complete(self, *_args):
        raise AssertionError("LLM should not be called")


def sample(temp=60):
    return {"zones_c": [temp], "ram_used_mib": 20, "ram_total_mib": 100,
            "ram_percent": 20, "sampled_monotonic": 1.0,
            "acquisition_stalled": False, "backend": "test"}


def test_trusted_request_id_prevents_duplicate_internal_side_effect(tmp_path):
    if not shutil.which("opa"):
        pytest.skip("OPA binary unavailable")
    loop = ReflectiveLoop(NoLLM(), Config(str(tmp_path), self_report_every=0),
                          telemetry_provider=lambda: sample())
    request_id = "55555555-5555-5555-5555-555555555555"
    action = {"action": "set_load_target", "percent": 40, "rationale": "reduce"}
    first = loop._execute(action, sample(), request_id=request_id)
    second = loop._execute(action, sample(), request_id=request_id)
    assert first["outcome"] == "applied"
    assert second["idempotent"] is True
    assert loop.store.db.execute("SELECT COUNT(*) FROM actions WHERE request_id=?", (request_id,)).fetchone()[0] == 1
    loop.close()

def test_opa_cap_changes_effective_runtime_setting(tmp_path):
    if not shutil.which("opa"):
        pytest.skip("OPA binary unavailable")
    loop = ReflectiveLoop(NoLLM(), Config(str(tmp_path), self_report_every=0),
                          telemetry_provider=lambda: sample(71))
    action = {"action": "set_load_target", "percent": 90, "rationale": "load"}
    result = loop._execute(action, sample(71))
    assert result["decision"] == "CAP"
    assert result["effective_action"]["percent"] == 50
    assert loop.settings["load_target"] == 50
    assert result["postcondition"] == "APPLIED_VERIFIED"
    loop.close()


def test_missing_opa_fails_closed_for_normal_action(tmp_path):
    config = Config(str(tmp_path), self_report_every=0, opa_path="/missing/opa")
    loop = ReflectiveLoop(NoLLM(), config, telemetry_provider=lambda: sample())
    result = loop._execute({"action": "observe", "rationale": "look"}, sample())
    assert result["decision"] == "ABSTAIN"
    assert result["outcome"].startswith("abstained:policy_engine_failure")
    loop.close()

def test_privileged_control_path_is_policy_checked_and_journaled(tmp_path):
    if not shutil.which("opa"):
        pytest.skip("OPA binary unavailable")
    from rlr.executor import ExecutionResult

    class FakeExecutor:
        def __init__(self):
            self.calls = 0
        def execute(self, request_id, operation, args):
            self.calls += 1
            return ExecutionResult(request_id, operation, "APPLIED_VERIFIED",
                                   observed_state={"active_state": "active"})

    executor = FakeExecutor()
    loop = ReflectiveLoop(NoLLM(), Config(str(tmp_path), self_report_every=0),
                          telemetry_provider=lambda: sample(), executor=executor)
    request_id = "66666666-6666-6666-6666-666666666666"
    result = loop.execute_privileged("manage_service",
                                     {"unit": "rlr-inference.service", "state": "active"},
                                     sample(), request_id=request_id)
    assert result["decision"] == "ALLOW"
    assert result["outcome"] == "APPLIED_VERIFIED"
    assert executor.calls == 1
    assert loop.store.policy_decision(request_id)["decision"] == "ALLOW"
    assert loop.store.execution_result(request_id)["outcome"] == "APPLIED_VERIFIED"
    again = loop.execute_privileged("manage_service",
                                    {"unit": "rlr-inference.service", "state": "active"},
                                    sample(), request_id=request_id)
    assert again["idempotent"] is True
    assert executor.calls == 1
    loop.close()


def test_model_schema_does_not_expose_privileged_operations():
    from rlr.actions import ACTION_FIELDS
    assert "manage_service" not in ACTION_FIELDS
    assert "set_nvpmodel" not in ACTION_FIELDS
    assert "set_cgroup_value" not in ACTION_FIELDS

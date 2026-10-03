from types import SimpleNamespace

from rlr.executor import TrustedExecutor
from rlr.store import Store


class SequenceRunner:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        assert kwargs.get("shell") is False
        return self.responses.pop(0)


def response(code=0, out="", err=""):
    return SimpleNamespace(returncode=code, stdout=out, stderr=err)


def test_unknown_operation_is_not_executed(tmp_path):
    runner = SequenceRunner([])
    store = Store(tmp_path / "brain.db")
    executor = TrustedExecutor(store=store, runner=runner)
    result = executor.execute("11111111-1111-1111-1111-111111111111", "shell", {"cmd": "true"})
    assert result.outcome == "ABSTAINED"
    assert runner.calls == []
    store.close()

def test_nvpmodel_is_verified_and_idempotent(tmp_path):
    runner = SequenceRunner([
        response(out="NV Power Mode: MODE_15W\n0\n"),
        response(),
        response(out="NV Power Mode: MODE_10W_DESKTOP\n5\n"),
    ])
    store = Store(tmp_path / "brain.db")
    executor = TrustedExecutor(store=store, runner=runner, allowed_nvp_modes={5})
    request_id = "22222222-2222-2222-2222-222222222222"
    first = executor.execute(request_id, "set_nvpmodel", {"mode": 5})
    second = executor.execute(request_id, "set_nvpmodel", {"mode": 5})
    assert first.outcome == "APPLIED_VERIFIED"
    assert second.outcome == "APPLIED_VERIFIED"
    assert len(runner.calls) == 3
    store.close()


def test_postcondition_mismatch_is_detected(tmp_path):
    runner = SequenceRunner([
        response(out="inactive\n"), response(), response(out="inactive\n"),
    ])
    store = Store(tmp_path / "brain.db")
    executor = TrustedExecutor(store=store, runner=runner,
                               allowed_services={"rlr-inference.service"})
    result = executor.execute("33333333-3333-3333-3333-333333333333",
                              "manage_service",
                              {"unit": "rlr-inference.service", "state": "active"})
    assert result.outcome == "FAILED_POSTCONDITION"
    store.close()

def test_cgroup_write_is_allowlisted_and_read_back(tmp_path):
    root = tmp_path / "cgroup"
    root.mkdir()
    control = root / "memory.max"
    control.write_text("max")
    executor = TrustedExecutor(cgroup_root=str(root))
    result = executor.execute("44444444-4444-4444-4444-444444444444",
                              "set_cgroup_value", {"name": "memory.max", "value": "4096"})
    assert result.outcome == "APPLIED_VERIFIED"
    assert control.read_text() == "4096"

from rlr.governor import Governor
from rlr.safety import SafetyStateMachine
from rlr.store import Store


def sample(temp=60, stalled=False):
    return {"zones_c": [temp], "ram_total_mib": 100, "ram_used_mib": 20,
            "backend": "test", "acquisition_stalled": stalled}


def test_cooldown_hysteresis_and_persistence(tmp_path):
    store = Store(tmp_path / "brain.db")
    machine = SafetyStateMachine(store, Governor(store))
    assert machine.evaluate(sample(79)) == "COOLDOWN"
    assert machine.evaluate(sample(72)) == "COOLDOWN"
    assert machine.evaluate(sample(69)) == "NORMAL"
    states = [row[0] for row in store.db.execute(
        "SELECT to_state FROM safety_transitions ORDER BY id")]
    assert states == ["COOLDOWN", "NORMAL"]
    store.close()


def test_restart_reconstructs_cooldown(tmp_path):
    path = tmp_path / "brain.db"
    store = Store(path)
    first = SafetyStateMachine(store, Governor(store))
    assert first.evaluate(sample(79)) == "COOLDOWN"
    store.close()
    reopened = Store(path)
    second = SafetyStateMachine(reopened, Governor(reopened))
    assert second.state == "COOLDOWN"
    assert second.evaluate(sample(72)) == "COOLDOWN"
    reopened.close()


def test_stalled_telemetry_enters_safe_mode(tmp_path):
    store = Store(tmp_path / "brain.db")
    machine = SafetyStateMachine(store, Governor(store))
    assert machine.evaluate(sample(60, stalled=True)) == "SAFE_MODE"
    store.close()


def test_repeated_execution_failures_enter_safe_mode(tmp_path):
    store = Store(tmp_path / "brain.db")
    machine = SafetyStateMachine(store, Governor(store))
    assert machine.evaluate(sample(60), execution_failures=3) == "SAFE_MODE"
    store.close()

import json

from rlr.forecast import ForecastEngine
from rlr.store import Store


class ForecastLLM:
    model = "Qwen2.5-1.5B-test"

    def __init__(self, value=50.0, invalid=False):
        self.value = value
        self.invalid = invalid
        self.prompts = []

    def complete(self, prompt, schema, schema_name):
        self.prompts.append((schema_name, prompt))
        if self.invalid:
            return {"wrong": 1}
        return {"forecast_temp_c": self.value}


def _sample(t, temp):
    return {"captured_monotonic": float(t), "sampled_monotonic": float(t),
            "zones_c": [float(temp), float(temp) - 1.0],
            "ram_used_mib": 1000.0, "ram_total_mib": 8192.0,
            "power_mode": "MODE_10W_DESKTOP", "fan_rpm": 0}


def _persist_and_run(store, engine, t, temp):
    sample = _sample(t, temp)
    store.add_telemetry(sample)
    return engine.on_sample(sample)


def _stable_rows(store):
    return [
        {k: row[k] for k in ("target_at", "horizon_s", "window_s",
                              "predicted_temp_c", "observed_temp_c", "error_c",
                              "valid", "method", "model_version")}
        for row in store.forecast_rows()
    ]


def test_forecast_persisted_before_future_observation_then_scored(tmp_path):
    store = Store(tmp_path / "brain.db")
    llm = ForecastLLM(value=50.0)
    engine = ForecastEngine(store, llm, window_s=60, horizon_s=60)

    assert _persist_and_run(store, engine, 0, 40)["issued"] == []
    assert _persist_and_run(store, engine, 30, 43)["issued"] == []
    issued = _persist_and_run(store, engine, 60, 46)["issued"]
    assert len(issued) == 3

    before = [row for row in store.forecast_rows() if row["target_at"] == 120.0]
    assert len(before) == 3
    assert all(row["observed_temp_c"] is None and row["error_c"] is None for row in before)
    assert store.db.execute(
        "SELECT COUNT(*) FROM telemetry WHERE captured_monotonic >= 120").fetchone()[0] == 0

    schema_name, prompt = llm.prompts[-1]
    assert schema_name == "rlr_temperature_forecast"
    payload = json.loads(prompt.split("\n", 1)[1])
    assert max(point["captured_monotonic"] for point in payload["history"]) == 60.0
    assert payload["target_at"] == 120.0

    _persist_and_run(store, engine, 120, 52)
    settled = {row["method"]: row for row in store.forecast_rows()
               if row["target_at"] == 120.0}
    assert settled["persistence"]["predicted_temp_c"] == 46.0
    assert settled["persistence"]["error_c"] == 6.0
    assert round(settled["linear"]["predicted_temp_c"], 6) == 52.0
    assert round(settled["linear"]["error_c"], 6) == 0.0
    assert settled["rlr_model"]["observed_temp_c"] == 52.0
    assert settled["rlr_model"]["error_c"] == 2.0
    store.close()


def test_invalid_model_forecast_is_never_scored(tmp_path):
    store = Store(tmp_path / "brain.db")
    engine = ForecastEngine(store, ForecastLLM(invalid=True), window_s=60, horizon_s=60)
    _persist_and_run(store, engine, 0, 40)
    _persist_and_run(store, engine, 30, 43)
    _persist_and_run(store, engine, 60, 46)

    model = next(row for row in store.forecast_rows() if row["method"] == "rlr_model")
    assert model["valid"] == 0
    assert model["predicted_temp_c"] is None
    _persist_and_run(store, engine, 120, 52)
    model = next(row for row in store.forecast_rows()
                 if row["method"] == "rlr_model" and row["target_at"] == 120.0)
    assert model["observed_temp_c"] is None
    assert model["error_c"] is None
    store.close()


def test_forecast_replay_is_deterministic_excluding_creation_timestamp(tmp_path):
    outputs = []
    for name in ("a.db", "b.db"):
        store = Store(tmp_path / name)
        engine = ForecastEngine(store, ForecastLLM(value=50.0), window_s=60, horizon_s=60)
        for t, temp in ((0, 40), (30, 43), (60, 46), (120, 52)):
            _persist_and_run(store, engine, t, temp)
        outputs.append(_stable_rows(store))
        store.close()
    assert outputs[0] == outputs[1]


def test_forecast_schema_migration_preserves_core_schema(tmp_path):
    store = Store(tmp_path / "brain.db")
    tables = {row[0] for row in store.db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert "forecasts" in tables
    assert store.db.execute(
        "SELECT 1 FROM schema_migrations WHERE version=4").fetchone() is not None
    columns = {row[1] for row in store.db.execute("PRAGMA table_info(forecasts)")}
    assert {"created_at", "target_at", "horizon_s", "window_s", "predicted_temp_c",
            "observed_temp_c", "error_c", "valid", "method", "model_version"} <= columns
    store.close()


def test_new_engine_ignores_persisted_pending_rows_from_old_session(tmp_path):
    store = Store(tmp_path / "brain.db")
    store.add_forecast(9999.0, 60, 60, 45.0, "persistence", "old")
    engine = ForecastEngine(store, ForecastLLM(value=50.0), window_s=60, horizon_s=60)
    _persist_and_run(store, engine, 0, 40)
    _persist_and_run(store, engine, 30, 43)
    result = _persist_and_run(store, engine, 60, 46)
    assert len(result["issued"]) == 3
    current = [row for row in store.forecast_rows() if row["target_at"] == 120.0]
    assert len(current) == 3
    old = next(row for row in store.forecast_rows() if row["model_version"] == "old")
    assert old["observed_temp_c"] is None
    store.close()


def test_allow_issue_false_settles_but_does_not_call_model(tmp_path):
    store = Store(tmp_path / "brain.db")
    llm = ForecastLLM(value=50.0)
    engine = ForecastEngine(store, llm, window_s=60, horizon_s=60)
    _persist_and_run(store, engine, 0, 40)
    _persist_and_run(store, engine, 30, 43)
    sample = _sample(60, 46)
    store.add_telemetry(sample)
    result = engine.on_sample(sample, allow_issue=False)
    assert result["issued"] == []
    assert llm.prompts == []
    store.close()

from rlr.governor import Governor
from rlr.store import Store


def make_governor(tmp_path):
    store = Store(tmp_path / "brain.db")
    return store, Governor(store)


def sample(temp=60, ram=1200, total=6833):
    return {"zones_c": [temp, temp - 1], "ram_used_mib": ram, "ram_total_mib": total,
            "sampled_monotonic": 1e20}


def test_thermal_caps_veto_vision_and_reply_tokens(tmp_path):
    store, governor = make_governor(tmp_path)
    assert "cap" in governor.veto({"action": "set_vision_fps", "fps": 2, "rationale": "x"}, sample(75), {})
    assert "cap" in governor.veto({"action": "set_reply_tokens", "tokens": 96, "rationale": "x"}, sample(75), {})
    assert governor.veto({"action": "set_vision_fps", "fps": 5, "rationale": "x"}, sample(60), {}) is None
    store.close()


def test_missing_telemetry_requires_abstention(tmp_path):
    store, governor = make_governor(tmp_path)
    action = {"action": "set_vision_fps", "fps": 1, "rationale": "x"}
    assert governor.veto(action, {"zones_c": []}, {})
    store.close()


def test_abstain_is_allowed_when_telemetry_is_missing(tmp_path):
    """The fail-safe action must not crash or be vetoed when evidence is absent."""
    store, governor = make_governor(tmp_path)
    action = {"action": "abstain", "reason": "telemetry unavailable", "rationale": "fail closed"}
    assert governor.veto(action, {"zones_c": []}, {}) is None
    store.close()


def test_missing_capture_timestamp_is_stale(tmp_path):
    """Unknown freshness must be treated as unsafe rather than silently fresh."""
    store, governor = make_governor(tmp_path)
    unsafe = {"zones_c": [60], "ram_used_mib": 1200, "ram_total_mib": 6833}
    assert "capture time" in governor.stale_reason(unsafe, 123.0)
    store.close()

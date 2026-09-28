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

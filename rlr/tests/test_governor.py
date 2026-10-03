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


def test_max_temp_filters_non_numeric_zone_values(tmp_path):
    """``Governor.max_temp`` must treat a corrupt sensor reading
    (a non-numeric value in ``zones_c``) as "unknown" rather than raise.

    A bare ``TypeError`` from ``max()`` (raised when a zone value is
    ``None`` or a string) would propagate out of ``preflight()`` and
    ``_tick()`` as an uncaught traceback the frozen external harness
    records as ``run_status:error``. The thermal gate is the
    ``too_hot_to_start:<temp>`` gate specifically, so any other
    failure mode must not surface to the harness at all. The
    defensive filter strips non-numeric zone values before calling
    ``max()``; if no numeric zones remain the result is ``None``
    (same shape as an empty zones list).
    """
    # Mixed numeric + None: only the numeric value contributes.
    assert Governor.max_temp({"zones_c": [60.0, None]}) == 60.0
    # Mixed numeric + string: the string is dropped, the numeric value wins.
    assert Governor.max_temp({"zones_c": [60.5, "62.0"]}) == 60.5
    # All non-numeric: equivalent to no telemetry, returns None.
    assert Governor.max_temp({"zones_c": [None, "bad", True]}) is None
    # A boolean in a zones list is rejected (boolean is technically
    # int but never a sensor-side temperature reading).
    assert Governor.max_temp({"zones_c": [True]}) is None
    # Empty zones list: still None.
    assert Governor.max_temp({"zones_c": []}) is None
    # Missing zones_c key: still None.
    assert Governor.max_temp({}) is None
    # Pure numeric stays available at the recorded broken-fan idle values.
    assert Governor.max_temp({"zones_c": [59.5, 60.5, 61.0, 61.5, 62.0, 62.5]}) == 62.5

def test_missing_timestamp_fails_closed(tmp_path):
    store, governor = make_governor(tmp_path)
    action = {"action": "set_load_target", "percent": 50, "rationale": "x"}
    reason = governor.veto(action, {"zones_c": [60], "ram_used_mib": 20, "ram_total_mib": 100}, {})
    assert "timestamp" in reason
    store.close()


def test_abstain_is_always_allowed_as_fail_safe(tmp_path):
    """The no-op fail-safe must survive missing evidence and thermal cooldown."""
    store, governor = make_governor(tmp_path)
    action = {"action": "abstain", "reason": "unsafe evidence", "rationale": "fail closed"}
    assert governor.veto(action, {"zones_c": []}, {}) is None
    assert governor.veto(action, sample(80), {}) is None
    store.close()


def test_missing_capture_timestamp_is_stale(tmp_path):
    """Unknown freshness must be treated as unsafe rather than silently fresh."""
    store, governor = make_governor(tmp_path)
    unsafe = {"zones_c": [60], "ram_used_mib": 1200, "ram_total_mib": 6833}
    reason = governor.stale_reason(unsafe, 123.0)
    assert "capture time" in reason and "timestamp" in reason
    store.close()

import rlr.telemetry as telemetry


class FakeBoard:
    stats = {"Temp CPU": 55.5, "GPU": 42.0, "Fan pwmfan0": 1200.0,
             "Power VDD_IN": 7.5, "NV Power Mode": "MODE_10W_DESKTOP"}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def ok(self):
        return True


def test_jtop_normalization():
    provider = telemetry.JtopTelemetryProvider(factory=lambda: FakeBoard())
    sample = provider.sample()
    assert sample["backend"] == "jtop"
    assert sample["temperatures"]["Temp CPU"] == 55.5
    assert sample["fan_rpm"] == 1200.0
    assert sample["power_w"] == 7.5
    assert sample["ram_total_mib"] is not None

def test_missing_fan_sensor_is_none(monkeypatch):
    monkeypatch.setattr(telemetry.glob, "glob", lambda _pattern: [])
    assert telemetry._fan_rpm() is None


def test_auto_provider_falls_back():
    class Broken:
        def sample(self):
            raise RuntimeError("jtop unavailable")

    class Fallback:
        def sample(self):
            return {"backend": "sysfs", "fan_rpm": None}

    sample = telemetry.AutoTelemetryProvider(Broken(), Fallback()).sample()
    assert sample == {"backend": "sysfs", "fan_rpm": None}


def test_acquisition_stall_is_explicit(monkeypatch):
    times = iter([10.0, 10.75])
    monkeypatch.setattr(telemetry.time, "monotonic", lambda: next(times))
    provider = telemetry.JtopTelemetryProvider(factory=lambda: FakeBoard(), stall_ms=500)
    sample = provider.sample()
    assert sample["capture_duration_ms"] == 750.0
    assert sample["acquisition_stalled"] is True

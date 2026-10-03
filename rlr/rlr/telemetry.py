"""Normalized read-only telemetry with jtop primary and sysfs fallback."""
import glob
import os
import re
import subprocess
import time
from pathlib import Path


class TelemetryProvider:
    """Interface for one bounded telemetry acquisition."""
    backend = "unknown"

    def sample(self):
        raise NotImplementedError


def _meminfo():
    values = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                values[key] = int(value.strip().split()[0]) / 1024.0
    except (OSError, ValueError):
        pass
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    used = (total - available) if total is not None and available is not None else None
    swap_total = values.get("SwapTotal")
    swap_free = values.get("SwapFree")
    swap_used = (swap_total - swap_free) if swap_total is not None and swap_free is not None else None
    return used, total, {"used_mib": swap_used, "total_mib": swap_total}


def _power_mode():
    try:
        result = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            return result.stdout.strip()[:400] or None
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _loads():
    gpu = dla = None
    paths = (glob.glob("/sys/devices/*gpu*/load") + glob.glob("/sys/devices/*/gpu*/load") +
             glob.glob("/sys/devices/platform/*gpu*/load"))
    for path in paths:
        try:
            gpu = float(Path(path).read_text().strip())
            if gpu > 100:
                gpu /= 10.0
            break
        except (OSError, ValueError):
            pass
    for path in glob.glob("/sys/devices/*dla*/load") + glob.glob("/sys/devices/platform/*dla*/load"):
        try:
            dla = float(Path(path).read_text().strip())
            break
        except (OSError, ValueError):
            pass
    return gpu, dla


def _fan_rpm():
    candidates = glob.glob("/sys/devices/platform/pwm-fan/hwmon/hwmon*/fan1_input")
    candidates += glob.glob("/sys/class/hwmon/hwmon*/fan1_input")
    for path in candidates:
        try:
            return float(Path(path).read_text().strip())
        except (OSError, ValueError):
            pass
    return None


def _temperatures():
    values = {}
    for zone in glob.glob("/sys/devices/virtual/thermal/thermal_zone*"):
        temp_path = Path(zone) / "temp"
        type_path = Path(zone) / "type"
        try:
            raw = float(temp_path.read_text().strip())
            value = raw / 1000.0 if abs(raw) > 1000 else raw
            name = type_path.read_text().strip() if type_path.exists() else Path(zone).name
            if -40 <= value <= 150:
                values[name] = value
        except (OSError, ValueError):
            pass
    return values


def _base_sample(backend, start, temperatures, ram_used, ram_total, swap,
                 gpu, dla, fan_rpm, power_mode, power_w=None, stall_ms=500.0):
    duration_ms = max(0.0, (time.monotonic() - start) * 1000.0)
    zones = list(temperatures.values())
    ram_percent = (ram_used / ram_total * 100.0) if ram_used is not None and ram_total else None
    try:
        load_average = list(os.getloadavg())
    except (AttributeError, OSError):
        load_average = None
    return {"captured_monotonic": start, "capture_duration_ms": duration_ms,
            "backend": backend, "temperatures": temperatures, "zones_c": zones,
            "ram_used_mib": ram_used, "ram_total_mib": ram_total,
            "ram_percent": ram_percent, "gpu_load": gpu, "dla_load": dla,
            "fan_rpm": fan_rpm, "power_mode": power_mode, "power_w": power_w,
            "swap": swap, "load_average": load_average,
            "acquisition_stalled": duration_ms > float(stall_ms)}


class SysfsTelemetryProvider(TelemetryProvider):
    backend = "sysfs"

    def __init__(self, stall_ms=500.0):
        self.stall_ms = stall_ms

    def sample(self):
        started = time.monotonic()
        ram_used, ram_total, swap = _meminfo()
        gpu, dla = _loads()
        return _base_sample(self.backend, started, _temperatures(), ram_used, ram_total,
                            swap, gpu, dla, _fan_rpm(), _power_mode(),
                            stall_ms=self.stall_ms)


class JtopTelemetryProvider(TelemetryProvider):
    backend = "jtop"

    def __init__(self, factory=None, stall_ms=500.0):
        self.factory = factory
        self.stall_ms = stall_ms
    def _factory(self):
        if self.factory is not None:
            return self.factory
        from jtop import jtop
        return lambda: jtop(interval=1.0)

    def sample(self):
        started = time.monotonic()
        with self._factory()() as board:
            if hasattr(board, "ok") and not board.ok():
                raise RuntimeError("jtop is not ready")
            stats = dict(getattr(board, "stats", {}) or {})
        temperatures = {}
        for key, value in stats.items():
            if "temp" in str(key).lower() and isinstance(value, (int, float)) and not isinstance(value, bool):
                temperatures[str(key)] = float(value)
        ram_used, ram_total, swap = _meminfo()
        gpu = _first_number(stats, ("GPU", "GPU Load", "GR3D_FREQ"))
        dla_values = [_first_number(stats, ("DLA0", "DLA1")), _first_number(stats, ("DLA",))]
        dla = next((v for v in dla_values if v is not None), None)
        fan = _first_number(stats, ("Fan", "Fan pwmfan0", "fan"))
        power = _first_number(stats, ("Power TOT", "Power VDD_IN", "power"))
        mode = _first_text(stats, ("NV Power Mode", "nvp model", "power mode")) or _power_mode()
        return _base_sample(self.backend, started, temperatures, ram_used, ram_total,
                            swap, gpu, dla, fan, mode, power_w=power,
                            stall_ms=self.stall_ms)


def _first_number(mapping, keys):
    lowered = {str(k).lower(): v for k, v in mapping.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _first_text(mapping, keys):
    lowered = {str(k).lower(): v for k, v in mapping.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


class AutoTelemetryProvider(TelemetryProvider):
    backend = "auto"

    def __init__(self, jtop_provider=None, fallback=None):
        self.jtop_provider = jtop_provider or JtopTelemetryProvider()
        self.fallback = fallback or SysfsTelemetryProvider()

    def sample(self):
        try:
            return self.jtop_provider.sample()
        except Exception:
            return self.fallback.sample()


_DEFAULT_PROVIDER = None


def sample_telemetry():
    """Compatibility entry point used by the existing ReflectiveLoop."""
    global _DEFAULT_PROVIDER
    if _DEFAULT_PROVIDER is None:
        _DEFAULT_PROVIDER = AutoTelemetryProvider()
    return _DEFAULT_PROVIDER.sample()


def summarize_mode(mode):
    if not mode:
        return None
    match = re.search(r"MODE_[A-Z0-9_]+", mode)
    return match.group(0) if match else mode

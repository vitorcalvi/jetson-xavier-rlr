"""Read-only local telemetry probes with injectable test providers."""
import glob
import re
import subprocess
from pathlib import Path


def _meminfo():
    values = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("MemTotal", "MemAvailable"):
                values[key] = int(value.strip().split()[0]) / 1024.0
    except (OSError, ValueError):
        pass
    if "MemTotal" in values and "MemAvailable" in values:
        return values["MemTotal"] - values["MemAvailable"], values["MemTotal"]
    return None, None


def _power_mode():
    try:
        result = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            return result.stdout.strip()[:200] or None
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
    # Xavier DLA load is not exposed consistently through sysfs/tegrastats.
    for path in glob.glob("/sys/devices/*dla*/load") + glob.glob("/sys/devices/platform/*dla*/load"):
        try:
            dla = float(Path(path).read_text().strip())
            break
        except (OSError, ValueError):
            pass
    return gpu, dla


def _fan_rpm():
    candidates = glob.glob("/sys/devices/platform/pwm-fan/hwmon/hwmon*/fan1_input")
    for path in candidates:
        try:
            return float(Path(path).read_text().strip())
        except (OSError, ValueError):
            pass
    return 0.0


def sample_telemetry():
    zones = []
    for path in glob.glob("/sys/devices/virtual/thermal/thermal_zone*/temp"):
        try:
            raw = float(Path(path).read_text().strip())
            zones.append(raw / 1000.0 if raw > 1000 else raw)
        except (OSError, ValueError):
            pass
    ram_used, ram_total = _meminfo()
    gpu, dla = _loads()
    return {"zones_c": zones, "ram_used_mib": ram_used, "ram_total_mib": ram_total,
            "power_mode": _power_mode(), "gpu_load": gpu, "dla_load": dla,
            "fan_rpm": _fan_rpm()}


def summarize_mode(mode):
    if not mode:
        return None
    match = re.search(r"MODE_[A-Z0-9_]+", mode)
    return match.group(0) if match else mode

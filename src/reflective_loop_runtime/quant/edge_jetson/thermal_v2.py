"""Fail-closed Jetson background-training throttle; never changes power modes.

No hardware controls are performed by default. The hot-path router stays
independent; a background trainer must call permitted() each work cycle.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ThermalGovernor:
    thermal_root: Path = Path("/sys/devices/virtual/thermal")
    meminfo: Path = Path("/proc/meminfo")
    pause_c: int = 75
    resume_c: int = 68
    max_memory_fraction: float = 0.80
    paused: bool = True
    reason: str = "unknown"
    samples: int = 0

    def check(self) -> bool:
        self.samples += 1
        if (
            not 0 < self.resume_c < self.pause_c <= 100
            or not 0.5 <= self.max_memory_fraction <= 0.95
        ):
            self.paused = True
            self.reason = "invalid_config"
            return False
        try:
            sensors = [
                int(f.read_text().strip()) for f in self.thermal_root.glob("thermal_zone*/temp")
            ]
            if not sensors or any(t < 0 or t > 125000 for t in sensors):
                raise ValueError("missing/inconsistent sysfs thermal data")
            info = {}
            for line in self.meminfo.read_text().splitlines():
                fields = line.split()
                if len(fields) >= 2 and fields[0] in ("MemTotal:", "MemAvailable:"):
                    info[fields[0]] = int(fields[1])
            total = info["MemTotal:"]
            available = info["MemAvailable:"]
            if total <= 0 or not 0 <= available <= total:
                raise ValueError("invalid memory status")
            frac = 1 - available / total
            max_c = max(sensors) / 1000
        except (OSError, ValueError, KeyError, UnicodeError) as exc:
            self.paused = True
            self.reason = "telemetry_unavailable_" + type(exc).__name__
            return False
        if max_c >= self.pause_c or frac >= self.max_memory_fraction:
            self.paused = True
            self.reason = "thermal_or_memory_cap"
            return False
        if self.paused and (max_c > self.resume_c or frac >= self.max_memory_fraction - 0.05):
            self.reason = "cooldown_required"
            return False
        self.paused = False
        self.reason = "allowed_background_only"
        return True


def safe_cpu_affinity(cpus: tuple[int, ...] = (0, 1), *, apply: bool = False) -> tuple[int, ...]:
    """Dry-run returns intersection with allowed cgroup/CPU affinity set.

    Never pins the calling service unless explicitly requested; on some Jetson
    environments logical CPUs 0-1 are reserved or outside container cpusets.
    """
    allowed = (
        set(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else set(range(os.cpu_count() or 1))
    )
    selected = tuple(sorted(allowed.intersection(cpus)))
    if not selected:
        raise RuntimeError("requested CPUs unavailable in current cpuset")
    if apply:
        os.sched_setaffinity(0, set(selected))
    return selected


def tensorrt_export_command(
    onnx_model: Path, output: Path, *, fp16: bool = True
) -> tuple[str, ...] | None:
    """Return offline builder arguments when trtexec is available. No build occurs."""
    if not onnx_model.is_file() or onnx_model.suffix != ".onnx":
        raise ValueError("ONNX source missing")
    binary = shutil.which("trtexec")
    if binary is None:
        return None
    return (binary, "--onnx=" + str(onnx_model), "--saveEngine=" + str(output)) + (
        ("--fp16",) if fp16 else ()
    )


def inference_backend(*, trt_available: bool, onnx_available: bool, torch_available: bool) -> str:
    return (
        "tensorrt_fp16"
        if trt_available
        else ("onnx_cpu" if onnx_available else ("torch_cpu" if torch_available else "unavailable"))
    )

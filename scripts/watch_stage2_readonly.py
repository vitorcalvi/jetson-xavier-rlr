#!/usr/bin/env python3
"""Read-only Jetson Stage2 48-hour health sampler; no broker/network writes.

Does not claim live market-feed shadow certification, trading profitability,
zero memory leaks, or actual paper-broker fill validation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

HALT = Path("/home/vitor/rlr/flags/HALT_ENTRIES")
LEDGER = Path("/home/vitor/rlr/trader-autonomy-1m-2cycle-20261015")
UNIT = "rlr-stage2-paper.service"


class SafetyMonitorViolation(RuntimeError):
    """Safety invariant violated; stop monitoring rather than report success."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SafetyMonitorViolation(f"unreadable telemetry: {path.name}") from exc
    if not isinstance(payload, dict):
        raise SafetyMonitorViolation(f"invalid telemetry: {path.name}")
    return payload


def _positive_pid() -> int:
    try:
        process = subprocess.run(
            ["systemctl", "--user", "show", UNIT, "--property=MainPID", "--value"],
            capture_output=True,
            text=True,
            check=True,
            timeout=3,
        )
        value = int(process.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise SafetyMonitorViolation("Stage2 service PID unavailable") from exc
    if value <= 1:
        raise SafetyMonitorViolation("Stage2 service is not running")
    return value


def _proc_metrics(pid: int, proc_root: Path) -> tuple[float, int]:
    path = proc_root / str(pid)
    try:
        status = (path / "status").read_text(encoding="utf-8")
        rss_kb = next(
            int(line.split()[1]) for line in status.splitlines() if line.startswith("VmRSS:")
        )
        fd_count = sum(1 for _ in (path / "fd").iterdir())
    except (OSError, StopIteration, ValueError) as exc:
        raise SafetyMonitorViolation("Stage2 /proc metrics unavailable") from exc
    return round(rss_kb / 1024, 3), fd_count


def _thermals(thermal_root: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    for temp_file in sorted(thermal_root.glob("thermal_zone*/temp")):
        try:
            raw = int(temp_file.read_text().strip())
        except (OSError, ValueError) as exc:
            raise SafetyMonitorViolation("thermal sensor unreadable") from exc
        if raw < -40000 or raw > 200000:
            raise SafetyMonitorViolation("thermal sensor out of range")
        result[temp_file.parent.name] = round(raw / 1000, 3)
    if not result:
        raise SafetyMonitorViolation("no sysfs temperature sensors")
    return result


def sample(
    *,
    halt: Path = HALT,
    ledger: Path = LEDGER,
    proc_root: Path = Path("/proc"),
    thermal_root: Path = Path("/sys/devices/virtual/thermal"),
    pid_lookup: Callable[[], int] = _positive_pid,
    epoch: Callable[[], float] = time.time,
) -> dict[str, Any]:
    # Every read is independent; no broker credentials, orders, or transports.
    if not halt.is_file():
        raise SafetyMonitorViolation("HALT_ENTRIES absent")
    now = epoch()
    safety = _read_json(ledger / "independent_safety_status.json")
    heartbeat = _read_json(ledger / "heartbeat.json")
    runtime_tick = _read_json(ledger / "runtime_tick.json")
    if safety.get("state") != "HALT_ENTRIES" or heartbeat.get("controller_state") != "HALT_ENTRIES":
        raise SafetyMonitorViolation("controller not in durable HALT_ENTRIES")
    if safety.get("mode") != "READ_ONLY" or safety.get("paper_only") is not True:
        raise SafetyMonitorViolation("safety watcher mode not read-only paper")
    if safety.get("issues") != []:
        raise SafetyMonitorViolation("safety watcher reported issues")
    if safety.get("broker_positions") != 0 or safety.get("broker_open_orders") != 0:
        raise SafetyMonitorViolation("unexpected broker exposure")
    tick_time = runtime_tick.get("ts")
    if not isinstance(tick_time, (float, int)) or not 0 <= now - tick_time <= 30:
        raise SafetyMonitorViolation("stale or future runtime tick")
    watcher_time = safety.get("utc")
    if not isinstance(watcher_time, str):
        raise SafetyMonitorViolation("watcher timestamp absent")
    from datetime import datetime

    try:
        dt = datetime.fromisoformat(watcher_time.replace("Z", "+00:00"))
        safety_age = now - dt.timestamp()
    except ValueError as exc:
        raise SafetyMonitorViolation("watcher timestamp invalid") from exc
    if not 0 <= safety_age <= 120:
        raise SafetyMonitorViolation("safety watcher stale")
    pid = pid_lookup()
    rss_mb, fds = _proc_metrics(pid, proc_root)
    temps = _thermals(thermal_root)
    return {
        "epoch": round(now, 3),
        "stage2_pid": pid,
        "stage2_rss_mb": rss_mb,
        "stage2_fd_count": fds,
        "max_thermal_c": max(temps.values()),
        "thermals_c": temps,
        "halt_verified": True,
        "controller_state": "HALT_ENTRIES",
        "watcher_age_seconds": round(safety_age, 3),
        "runtime_tick_age_seconds": round(now - tick_time, 3),
        "broker_positions": 0,
        "broker_open_orders": 0,
        "paper_orders_sent_by_monitor": 0,
    }


def _write_record(path: Path, record: dict[str, Any]) -> None:
    """Atomic, append-only newline-delimited TOON-style scalar record."""
    if path.suffix != ".toon":
        raise ValueError("output extension must be .toon")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(mode=0o600, exist_ok=True)
    os.chmod(path, 0o600)
    with path.open("a", encoding="utf-8") as out:
        out.write("sample:\n")
        for key, value in record.items():
            out.write(
                "  " + key + ": " + json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n"
            )
        out.flush()
        os.fsync(out.fileno())


def monitor(
    *,
    duration_hours: float,
    every_seconds: float,
    out: Path,
    once: bool = False,
    sample_fn: Callable[[], dict[str, Any]] = sample,
) -> dict[str, Any]:
    if not 0 < duration_hours <= 48 or not 0 < every_seconds <= 3600:
        raise ValueError("invalid duration / sample interval")
    started = time.monotonic()
    count = 0
    first: dict[str, Any] | None = None
    previous: dict[str, Any] | None = None
    while True:
        row = sample_fn()
        if first is None:
            first = row
        if row["max_thermal_c"] >= 80.0:
            raise SafetyMonitorViolation("Stage2 hardware thermal cap exceeded")
        if previous is not None and row["stage2_pid"] != previous["stage2_pid"]:
            raise SafetyMonitorViolation("Stage2 process restarted during soak")
        previous = row
        count += 1
        metrics = dict(row)
        metrics["samples"] = count
        metrics["rss_drift_mb"] = round(row["stage2_rss_mb"] - first["stage2_rss_mb"], 3)
        metrics["fd_delta"] = row["stage2_fd_count"] - first["stage2_fd_count"]
        metrics["elapsed_seconds"] = round(time.monotonic() - started, 3)
        _write_record(out, metrics)
        if once or time.monotonic() - started >= duration_hours * 3600:
            return metrics
        time.sleep(
            min(every_seconds, max(0.0, duration_hours * 3600 - (time.monotonic() - started)))
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=48)
    parser.add_argument("--sample-seconds", type=float, default=3600)
    parser.add_argument(
        "--out", type=Path, default=Path("/home/vitor/rlr/monitoring/toon22_stage2_health.toon")
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    try:
        print(
            json.dumps(
                monitor(
                    duration_hours=args.hours,
                    every_seconds=args.sample_seconds,
                    out=args.out,
                    once=args.once,
                ),
                sort_keys=True,
            ),
            flush=True,
        )
    except (SafetyMonitorViolation, ValueError, OSError) as exc:
        print("MONITOR_FAIL_CLOSED:" + str(exc), file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

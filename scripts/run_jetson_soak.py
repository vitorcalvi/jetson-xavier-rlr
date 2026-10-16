#!/usr/bin/env python3
"""Offline 100k logical-tick soak; no broker transport or service mutation."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
import time
import tracemalloc
from pathlib import Path

from reflective_loop_runtime.quant.edge_jetson.runtime_opt import (
    toon_decode,
    toon_encode,
)
from reflective_loop_runtime.quant.edge_jetson.thermal_v2 import ThermalGovernor


def rss_mb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError("RSS unavailable")


def fd_count() -> int:
    return len(list(Path("/proc/self/fd").iterdir()))


def stage_a(ticks: int = 100_000, output: Path | None = None) -> dict:
    if not 10 <= ticks <= 10_000_000:
        raise ValueError("invalid tick count")
    state = {
        "W1": {
            "symbol": "SPY",
            "mode": "SHADOW",
            "bars": [
                {
                    "t": i,
                    "o": 100 + i / 100,
                    "h": 101 + i / 100,
                    "l": 99 + i / 100,
                    "c": 100.2 + i / 100,
                    "v": 12345 + i,
                }
                for i in range(8)
            ],
        }
    }
    for i in range(50):
        state["W1"]["seq"] = i
        assert toon_decode(toon_encode(state)) == state
    governor = ThermalGovernor()
    with tempfile.TemporaryDirectory(prefix="toon22-") as temporary:
        with sqlite3.connect(str(Path(temporary) / "log.sqlite")) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE ticks(id INTEGER PRIMARY KEY, mode TEXT NOT NULL)")
            start_fds = fd_count()
            tracemalloc.start()
            start_rss = rss_mb()
            start_memory = tracemalloc.get_traced_memory()[0]
            start = time.perf_counter()
            thermal_pauses = 0
            for i in range(ticks):
                state["W1"]["seq"] = i
                if i % 32 == 0 and toon_decode(toon_encode(state)) != state:
                    raise RuntimeError("lossy serializer")
                if i % 128 == 0:
                    db.execute("INSERT INTO ticks VALUES (?,?)", (i, "HOLD"))
                if i % 4096 == 0:
                    db.commit()
                    db.execute("PRAGMA wal_checkpoint(PASSIVE)")
                if i % 1000 == 0 and not governor.check():
                    thermal_pauses += 1
            db.commit()
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            elapsed = time.perf_counter() - start
            final_memory, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            drift = rss_mb() - start_rss
            fd_delta = fd_count() - start_fds
            result = {
                "ticks": ticks,
                "logical_hours": ticks * 1.728 / 3600,
                "wall_seconds": round(elapsed, 3),
                "serializer_roundtrips": (ticks + 31) // 32,
                "rss_drift_mb": round(drift, 3),
                "tracemalloc_drift_mb": round((final_memory - start_memory) / 1048576, 4),
                "tracemalloc_peak_mb": round(peak_memory / 1048576, 4),
                "fd_delta": fd_delta,
                "thermal_pause_samples": thermal_pauses,
                "exceptions": 0,
                "pass": drift < 5 and fd_delta == 0,
                "scope": "simulated ticks, not full 48h main-loop certification",
            }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def stage_b(
    hours: float = 48,
    interval_seconds: int = 3600,
    output: Path = Path("artifacts/stage_b_health.toon"),
    halt_file: Path = Path("/home/vitor/rlr/flags/HALT_ENTRIES"),
) -> None:
    if hours <= 0 or hours > 48 or interval_seconds <= 0 or not halt_file.is_file():
        raise RuntimeError("HALT_ENTRIES absent or invalid schedule")
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    while time.monotonic() - start < hours * 3600:
        if not halt_file.is_file():
            raise RuntimeError("halt flag removed")
        governor = ThermalGovernor()
        training_allowed = governor.check()
        event = {
            "epoch_ns": time.time_ns(),
            "elapsed_seconds": time.monotonic() - start,
            "rss_mb": rss_mb(),
            "fd_count": fd_count(),
            "halt_verified": True,
            "broker_orders_placed_by_monitor": 0,
            "background_training_allowed": training_allowed,
            "thermal_reason": governor.reason,
        }
        with output.open("a", encoding="utf-8") as file:
            file.write(toon_encode(event))
            file.flush()
            os.fsync(file.fileno())
        time.sleep(min(interval_seconds, max(0.01, hours * 3600 - (time.monotonic() - start))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("a", "b"), required=True)
    parser.add_argument("--ticks", type=int, default=100_000)
    parser.add_argument("--hours", type=float, default=48)
    parser.add_argument("--interval-seconds", type=int, default=3600)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--halt-file", type=Path, default=Path("/home/vitor/rlr/flags/HALT_ENTRIES")
    )
    arguments = parser.parse_args()
    if arguments.stage == "a":
        print(json.dumps(stage_a(arguments.ticks, arguments.output), indent=2))
    else:
        stage_b(
            arguments.hours,
            arguments.interval_seconds,
            arguments.output or Path("artifacts/stage_b_health.toon"),
            arguments.halt_file,
        )


if __name__ == "__main__":
    main()

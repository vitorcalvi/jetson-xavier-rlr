"""Independent, fail-closed 48h Stage2 audit evaluator. No broker or service writes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SOURCE = Path("/home/vitor/rlr/monitoring/toon22_stage2_health.toon")
VERDICT = Path("/home/vitor/rlr/monitoring/toon23_soak_final_verdict.toon")
HALT = Path("/home/vitor/rlr/flags/HALT_ENTRIES")
EXPECTED_PID = 2355006


def parse_samples(raw: str) -> list[dict]:
    rows: list[dict] = []
    current: dict = {}
    for line in raw.splitlines():
        if line == "sample:":
            if current:
                rows.append(current)
            current = {}
        elif line.startswith("  ") and ": " in line:
            key, val = line.strip().split(": ", 1)
            if key in current:
                raise ValueError("duplicate sample key")
            current[key] = json.loads(val)
        elif line.strip():
            raise ValueError("unexpected monitor record")
    if current:
        rows.append(current)
    return rows


def evaluate(rows: list[dict]) -> dict:
    required = (
        "epoch",
        "stage2_pid",
        "stage2_rss_mb",
        "stage2_fd_count",
        "max_thermal_c",
        "halt_verified",
        "broker_positions",
        "broker_open_orders",
    )
    if not rows or any(any(k not in r for k in required) for r in rows):
        return {"status": "FAIL", "reason": "missing_samples_or_fields", "samples": len(rows)}
    first, last = rows[0], rows[-1]
    duration = last["epoch"] - first["epoch"]
    pids = {r["stage2_pid"] for r in rows}
    safe = all(
        r["halt_verified"] is True
        and r["broker_positions"] == r["broker_open_orders"] == 0
        and r["max_thermal_c"] < 80
        for r in rows
    )
    if pids != {EXPECTED_PID} or not safe:
        state = "FAIL"
        reason = "pid_or_halt_or_exposure_or_thermal_invariant"
    elif len(rows) < 576 or duration < 48 * 3600:
        state = "PENDING"
        reason = "elapsed_or_sample_count_insufficient"
    else:
        state = "PASS"
        reason = "observational_soak_complete"
    first_rss = float(first["stage2_rss_mb"])
    first_fd = int(first["stage2_fd_count"])
    return {
        "status": state,
        "reason": reason,
        "samples": len(rows),
        "elapsed_hours": round(duration / 3600, 4),
        "start_epoch": first["epoch"],
        "last_epoch": last["epoch"],
        "stage2_pid": EXPECTED_PID,
        "max_rss_growth_mb": round(max(float(r["stage2_rss_mb"]) - first_rss for r in rows), 4),
        "last_rss_drift_mb": round(float(last["stage2_rss_mb"]) - first_rss, 4),
        "max_fd_delta": max(int(r["stage2_fd_count"]) - first_fd for r in rows),
        "final_fd_delta": int(last["stage2_fd_count"]) - first_fd,
        "max_thermal_c": max(float(r["max_thermal_c"]) for r in rows),
        "halt_preserved": all(r["halt_verified"] is True for r in rows),
        "broker_orders_by_notifier": 0,
    }


def run(source: Path = SOURCE, verdict_file: Path = VERDICT) -> dict:
    if not HALT.is_file():
        raise RuntimeError("fatal: HALT_ENTRIES is absent")
    rows = parse_samples(source.read_text(encoding="utf-8"))
    result = evaluate(rows)
    if result["status"] == "PENDING":
        return result
    if verdict_file.exists():
        # Never overwrite an earlier completion/failure outcome.
        return result
    message = (
        "FINAL_SOAK_VERDICT\n"
        + "\n".join(
            f"  {key}: {json.dumps(val, separators=(',', ':'))}" for key, val in result.items()
        )
        + "\n"
    )
    verdict_file.parent.mkdir(parents=True, exist_ok=True)
    import tempfile

    with tempfile.NamedTemporaryFile(
        "w", dir=verdict_file.parent, prefix=".soak_verdict_", delete=False
    ) as fp:
        fp.write(message)
        fp.flush()
        os.fsync(fp.fileno())
        temp = Path(fp.name)
    if verdict_file.exists():
        temp.unlink()
        return result
    os.replace(temp, verdict_file)
    # Journal and wall are optional; the verdict file is always durable.
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--verdict", type=Path, default=VERDICT)
    args = parser.parse_args()
    try:
        result = run(args.source, args.verdict)
        print("TOON23_SOAK_STATE " + json.dumps(result, sort_keys=True), flush=True)
        return 1 if result["status"] == "FAIL" else 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        print("TOON23_SOAK_FAIL_CLOSED " + str(exc), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""User-level RLR health probe (Jetson D1).

Reads the runtime heartbeat file and reports whether the loop is alive and
advancing, and asserts that the deployment holds real authority: the mission
policy must load as ENFORCE. Never modifies system state, never needs sudo,
never prints secrets.

Exit code 0 means healthy, 1 means degraded or stopped.
"""

from __future__ import annotations

import os
import time
from pathlib import Path


def policy_mode() -> str:
    """Verify the bounded, dated operator policy used by production G1."""
    from reflective_loop_runtime.authority import load_operator_policy

    policy_raw = (os.environ.get("RLR_POLICY_PATH") or "").strip()
    public_raw = (os.environ.get("RLR_POLICY_PUBLIC_KEY_PATH") or "").strip()
    if not policy_raw or not public_raw:
        raise RuntimeError(
            "production G1 requires RLR_POLICY_PATH and RLR_POLICY_PUBLIC_KEY_PATH"
        )
    return load_operator_policy(policy_raw, public_raw).mode


def main() -> int:
    home = Path(os.environ.get("HOME", str(Path.home())))
    heartbeat = Path(
        os.environ.get("RLR_HEARTBEAT_FILE", str(home / "rlr" / "heartbeat.txt"))
    )
    max_age = float(os.environ.get("RLR_HEARTBEAT_MAX_AGE", "15.0"))

    # The loop must hold real authority: any other policy mode means the
    # deployment is not enforcing the mission policy it believes it enforces.
    try:
        mode = policy_mode()
    except Exception as exc:  # noqa: BLE001 - any failure here is unhealthy
        print(f"UNHEALTHY: policy mode unavailable: {type(exc).__name__}: {exc}")
        return 1
    if mode != "ENFORCE":
        print(f"UNHEALTHY: policy mode {mode} (expected ENFORCE)")
        return 1

    if not heartbeat.exists():
        print(f"UNHEALTHY: heartbeat {heartbeat} missing")
        return 1

    mtime = heartbeat.stat().st_mtime
    age = time.time() - mtime
    try:
        line = heartbeat.read_text(encoding="utf-8").rsplit("\n", 1)[-2]
    except (IndexError, OSError):
        line = ""

    if age > max_age:
        print(f"UNHEALTHY: heartbeat stale ({age:.1f}s > {max_age}s); last={line}")
        return 1

    print(f"HEALTHY: heartbeat age={age:.1f}s; {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

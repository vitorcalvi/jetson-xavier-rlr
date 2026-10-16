#!/usr/bin/env python3
"""User-level RLR paper-trader health probe.

Reads the trader heartbeat file and reports whether the loop is alive
within the configured freshness window. Exit code 0 means the heartbeat
is fresh; non-zero means the trader is stale or missing. The probe never
modifies system state, never needs sudo, and never prints secrets.

Environment variables (all optional):

* ``RLR_TRADER_HEARTBEAT_FILE`` -- override the heartbeat path
  (default: ``$HOME/rlr/trader/heartbeat.json``).
* ``RLR_TRADER_HEARTBEAT_MAX_AGE`` -- override the freshness window in
  seconds (default: 180).
* ``RLR_TRADER_DATA_DIR`` -- override the data directory when no
  explicit heartbeat path is set (default: ``$HOME/rlr/trader``).
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

DEFAULT_MAX_AGE_SECONDS = 180.0


def _data_dir(env: Mapping[str, str]) -> Path:
    """Resolve the trader data directory from ``env`` / ``HOME``."""
    raw = (env.get("RLR_TRADER_DATA_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser()
    home = (env.get("HOME") or str(Path.home())).strip()
    return Path(home) / "rlr" / "trader"


def heartbeat_path(env: Mapping[str, str]) -> Path:
    """The path the trader writes its heartbeat to."""
    raw = (env.get("RLR_TRADER_HEARTBEAT_FILE") or "").strip()
    if raw:
        return Path(raw).expanduser()
    return _data_dir(env) / "heartbeat.json"


def max_age_seconds(env: Mapping[str, str]) -> float:
    """The freshness window in seconds; defaults to :data:`DEFAULT_MAX_AGE_SECONDS`."""
    raw = (env.get("RLR_TRADER_HEARTBEAT_MAX_AGE") or "").strip()
    if not raw:
        return float(DEFAULT_MAX_AGE_SECONDS)
    try:
        value = float(raw)
    except ValueError:
        return float(DEFAULT_MAX_AGE_SECONDS)
    return value if value > 0 else float(DEFAULT_MAX_AGE_SECONDS)


def _read_heartbeat(path: Path) -> Mapping[str, Any] | None:
    """Return the parsed heartbeat payload, or ``None`` on read/decode error."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        payload = json.loads(text)
    except ValueError:
        return None
    if not isinstance(payload, Mapping):
        return None
    return payload


def main(env: Mapping[str, str] | None = None) -> int:
    """Probe the trader heartbeat; return 0 on healthy, 1 on stale/missing."""
    env_map: Mapping[str, str] = os.environ if env is None else env
    path = heartbeat_path(env_map)
    max_age = float(max_age_seconds(env_map))

    if not path.exists():
        print(f"UNHEALTHY: heartbeat {path} missing", file=sys.stderr)
        return 1

    age = time.time() - path.stat().st_mtime
    if age > max_age:
        print(
            f"UNHEALTHY: heartbeat stale ({age:.1f}s > {max_age:.1f}s)",
            file=sys.stderr,
        )
        return 1

    payload = _read_heartbeat(path)
    if payload is None:
        # A heartbeat the trader just wrote should parse; if it does not,
        # the loop is misconfigured. The freshness check above already passed,
        # so report degraded but do not crash the probe.
        print(
            f"UNHEALTHY: heartbeat {path} is not parseable JSON",
            file=sys.stderr,
        )
        return 1

    open_positions = payload.get("open_positions") or {}
    if not isinstance(open_positions, dict):
        open_positions = {}
    realized = payload.get("realized_r_today")
    halted = bool(payload.get("halted"))

    # Legacy healthcheck is process-aliveness-only unless explicitly run
    # as an entry-authority check. Never report RUN from just halted=False.
    controller_mode = payload.get("controller_mode", "UNREPORTED")
    entries_blocked = payload.get("entries_blocked")
    require_run = env_map.get("RLR_TRADER_REQUIRE_RUN", "") == "1"
    if require_run and (controller_mode != "RUN" or entries_blocked is not False):
        print(f"ENTRY_BLOCKED: age={age:.1f}s controller_mode={controller_mode} "
              f"incident={payload.get('controller_incident', 'unknown')}",file=sys.stderr)
        return 2
    label = "PROCESS_HEALTHY_ENTRIES_BLOCKED" if entries_blocked else "HEALTHY"
    print(f"{label}: heartbeat age={age:.1f}s halted={halted} "
          f"controller_mode={controller_mode} entries_blocked={entries_blocked} "
          f"positions={sorted(open_positions.keys())} "
          f"realized_r_today={realized}")
    return 0


__all__ = [
    "DEFAULT_MAX_AGE_SECONDS",
    "heartbeat_path",
    "max_age_seconds",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Append one TOON metrics line for the RLR overnight soak (Jetson D2).

Reads user-systemd unit state, per-process RSS, the CPU thermal zone, free RAM,
verifies the live runtime's keyed audit hash chain, and folds in the newest
simsoak result's permits/policy counters. Prints a single space-separated
``key=value`` line (the same TOON-ish scalar style the runtime heartbeat uses).

Never needs sudo, never prints secrets (the audit key is loaded from disk and
never echoed), and never modifies state other than appending the metrics line.
"""

from __future__ import annotations

import os
import re
import sqlite3
import subprocess
import time
from pathlib import Path

HOME = Path(os.environ.get("HOME", str(Path.home())))
SOAK = Path(os.environ.get("RLR_SOAK_DIR", str(HOME / "rlr" / "soak")))
DATA = Path(os.environ.get("RLR_DATA_DIR", str(HOME / "rlr" / "data")))
HEARTBEAT = Path(os.environ.get("RLR_HEARTBEAT_FILE", str(HOME / "rlr" / "heartbeat.txt")))

UNITS = {
    "rlr": "rlr.service",
    "advisor": "rlr-advisor.service",
    "simsoak": "rlr-simsoak.service",
}


def _sh(*args: str) -> str:
    try:
        return subprocess.run(
            list(args), capture_output=True, text=True, timeout=15, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _unit_props(unit: str) -> dict[str, str]:
    out = _sh(
        "systemctl", "--user", "show",
        "-p", "ActiveState", "-p", "SubState", "-p", "NRestarts", "-p", "MainPID",
        unit,
    )
    props: dict[str, str] = {}
    for line in out.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            props[key] = value
    return props


def _rss_mib(pid_raw: str) -> float:
    """Sum VmRSS (MiB) over a pid and all of its descendants."""
    try:
        root = int(pid_raw)
    except (TypeError, ValueError):
        return 0.0
    children: dict[int, list[int]] = {}
    rss: dict[int, float] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        close = stat.rfind(")")
        try:
            ppid = int(stat[close + 2:].split()[1])
        except (IndexError, ValueError):
            continue
        children.setdefault(ppid, []).append(int(entry.name))
        try:
            status = (entry / "status").read_text()
        except OSError:
            continue
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                rss[int(entry.name)] = int(line.split()[1]) / 1024.0
                break
    total = 0.0
    stack = [root]
    seen: set[int] = set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        total += rss.get(cur, 0.0)
        stack.extend(children.get(cur, []))
    return round(total, 1)


def _cpu_temp_c() -> float | None:
    base = Path("/sys/class/thermal")
    fallback: float | None = None
    for zone in sorted(base.glob("thermal_zone*")):
        try:
            kind = (zone / "type").read_text().strip()
            value = int((zone / "temp").read_text().strip()) / 1000.0
        except (OSError, ValueError):
            continue
        if kind == "CPU-therm":
            return round(value, 1)
        if fallback is None:
            fallback = round(value, 1)
    return fallback


def _ram_avail_mib() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return round(int(line.split()[1]) / 1024.0, 1)
    except (OSError, ValueError):
        pass
    return None


def _heartbeat_age_s() -> float | None:
    try:
        return round(time.time() - HEARTBEAT.stat().st_mtime, 1)
    except OSError:
        return None


def _audit_verify() -> tuple[str, str]:
    """Verify the live runtime's keyed audit chain; returns (ok, reason)."""
    db = DATA / "audit.db"
    if not db.exists():
        return "none", "audit_db_absent"
    try:
        from reflective_loop_runtime.kernel.audit import AuditLog
    except Exception:  # noqa: BLE001 - report, never crash the sampler
        return "none", "import_failed"
    key = None
    key_file = DATA / "audit.key"
    if key_file.exists():
        try:
            key = key_file.read_bytes()
        except OSError:
            key = None
    try:
        log = AuditLog(str(db), key=key)
        verdict = log.verify_chain()
        log.close()
    except Exception as exc:  # noqa: BLE001
        return "false", type(exc).__name__
    reason = str(getattr(verdict, "reason", "") or ("ok" if verdict else "broken"))
    return ("true" if verdict else "false"), reason


_METRIC_RE = re.compile(r"^\s{2}([a-z_]+):\s*(.*)$", re.MULTILINE)


def _latest_soak_metrics() -> dict[str, str]:
    latest = SOAK / "simsoak.latest.toon"
    if not latest.exists():
        return {}
    try:
        text = latest.read_text()
    except OSError:
        return {}
    values = {m.group(1): m.group(2).strip() for m in _METRIC_RE.finditer(text)}
    age = round(time.time() - latest.stat().st_mtime, 1)
    out = {
        "permits_issued": values.get("permits_issued", "none"),
        "permits_consumed": values.get("permits_consumed", "none"),
        "policy_violations": values.get("policy_violations", "none"),
        "soak_audit_ok": values.get("audit_chain_verified", "none"),
        "soak_age_s": str(age),
    }
    return out


def _permits_db_counts() -> tuple[str, str]:
    """Fallback (consumed, issued) from the live runtime's permits.db.

    Consumed nonces are persisted; issued permits are bearer tokens that are not
    persisted, so the issued count is only known from a completed soak result.
    """
    db = DATA / "permits.db"
    if not db.exists():
        return "none", "none"
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            consumed = str(con.execute("select count(*) from used_nonces").fetchone()[0])
        except sqlite3.Error:
            consumed = "none"
        con.close()
    except sqlite3.Error:
        return "none", "none"
    return consumed, "none"


def main() -> int:
    line: dict[str, str] = {"ts": str(int(time.time()))}
    mains: dict[str, str] = {}
    for label, unit in UNITS.items():
        props = _unit_props(unit)
        line[f"{label}_state"] = props.get("ActiveState", "unknown")
        line[f"{label}_sub"] = props.get("SubState", "unknown")
        line[f"{label}_restarts"] = props.get("NRestarts", "none")
        mains[label] = props.get("MainPID", "0")

    hb = _heartbeat_age_s()
    line["heartbeat_age_s"] = "none" if hb is None else str(hb)
    line["rlr_rss_mib"] = str(_rss_mib(mains.get("rlr", "0")))
    line["llama_rss_mib"] = str(_rss_mib(mains.get("advisor", "0")))
    temp = _cpu_temp_c()
    line["cpu_temp_c"] = "none" if temp is None else str(temp)
    ram = _ram_avail_mib()
    line["ram_avail_mib"] = "none" if ram is None else str(ram)
    audit_ok, audit_reason = _audit_verify()
    line["audit_ok"] = audit_ok
    line["audit_reason"] = audit_reason.replace(" ", "_") or "none"

    soak = _latest_soak_metrics()
    if soak.get("permits_consumed", "none") != "none":
        line["permits_issued"] = soak["permits_issued"]
        line["permits_consumed"] = soak["permits_consumed"]
        line["policy_violations"] = soak["policy_violations"]
    else:
        consumed, issued = _permits_db_counts()
        line["permits_issued"] = issued
        line["permits_consumed"] = consumed
        line["policy_violations"] = "none"
    line["soak_audit_ok"] = soak.get("soak_audit_ok", "none")
    line["soak_age_s"] = soak.get("soak_age_s", "none")

    rendered = " ".join(f"{k}={v}" for k, v in line.items())
    SOAK.mkdir(parents=True, exist_ok=True)
    with (SOAK / "metrics.toon").open("a", encoding="utf-8") as fh:
        fh.write(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

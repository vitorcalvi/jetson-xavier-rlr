
"""Regression coverage for persisted clock ordering and bounded rollback safety."""
from __future__ import annotations

import json
import multiprocessing
import random
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from reflective_loop_runtime.kernel.permit import (
    PermitIssuer,
    PermitService,
    PermitVerificationError,
    _TimeHighWaterMark,
)
from reflective_loop_runtime.kernel.policy import MissionPolicyStore, sign_envelope

_START = datetime(2026, 6, 1, 12, tzinfo=timezone.utc)
_ISSUER = b"g2-clock-fuzz-key-0123456789012345678"
_POLICY_SECRET = "g2-clock-fuzz-dev-policy"


class Clock:
    def __init__(self, moment: datetime = _START):
        self.now = moment

    def __call__(self) -> str:
        return self.now.isoformat()


class Ticks:
    def __init__(self) -> None:
        self.seconds = 0.0

    def __call__(self) -> float:
        return self.seconds


def _policy(tmp_path: Path):
    bundle = {
        "mission_id": "g2-clock-regression",
        "objective": "prevent permit-clock rollback",
        "invariants": ["never_shell_out_unsandboxed"],
        "allowed_domains": [],
        "capabilities": [],
        "budgets": {},
        "standing_mandates": [],
        "policy_version": "1.0.0",
    }
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(sign_envelope(bundle, secret=_POLICY_SECRET)))
    return MissionPolicyStore(secret=_POLICY_SECRET, allow_dev_hmac=True).load(path)


def _writer(db_path: str, offsets: list[float], start_flag) -> None:
    connection = sqlite3.connect(db_path, timeout=30, isolation_level=None)
    connection.execute("PRAGMA busy_timeout=30000")
    hwm = _TimeHighWaterMark(connection, threading.RLock(), "shared-clock")
    start_flag.wait(30)
    for offset in offsets:
        hwm.observe(_START + timedelta(seconds=offset))
    connection.close()


def test_two_services_preserve_newer_hwm_under_older_writer(tmp_path):
    policy = _policy(tmp_path)
    clock = Clock()
    db_path = tmp_path / "shared.db"
    first = PermitService(PermitIssuer(_ISSUER), policy, clock=clock, db_path=db_path)
    second = PermitService(PermitIssuer(_ISSUER), policy, clock=clock, db_path=db_path)
    newer = _START + timedelta(hours=3, microseconds=1)
    older = _START + timedelta(minutes=12)
    first._hwm.observe(newer)
    second._hwm.observe(older)
    assert first._hwm.observed() == newer
    assert second._hwm.observed() == newer
    first.close()
    second.close()


def test_existing_offset_timestamp_is_normalized_on_open(tmp_path):
    db_path = tmp_path / "old.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE clock_hwm (source TEXT PRIMARY KEY, observed_at TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO clock_hwm VALUES (?, ?)",
            ("shared-clock", "2026-06-01T14:00:00+02:00"),
        )
    connection = sqlite3.connect(db_path, isolation_level=None)
    hwm = _TimeHighWaterMark(connection, threading.RLock(), "shared-clock")
    assert hwm.observed() == _START
    hwm.observe(_START - timedelta(hours=1))
    assert hwm.observed() == _START
    raw = connection.execute("SELECT observed_at FROM clock_hwm").fetchone()[0]
    assert raw == "2026-06-01T12:00:00.000000+00:00"
    connection.close()


def test_multiprocess_writers_retain_true_maximum(tmp_path):
    db_path = tmp_path / "shared-processes.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE clock_hwm (source TEXT PRIMARY KEY, observed_at TEXT NOT NULL)"
        )
    ctx = multiprocessing.get_context("spawn")
    event = ctx.Event()
    values = [list(range(start, 500, 5)) for start in range(5)]
    processes = [
        ctx.Process(target=_writer, args=(str(db_path), offsets, event))
        for offsets in values
    ]
    for process in processes:
        process.start()
    event.set()
    for process in processes:
        process.join(timeout=25)
        if process.is_alive():
            process.terminate()
            process.join(5)
        assert process.exitcode == 0
    with sqlite3.connect(db_path) as connection:
        stamp = connection.execute(
            "SELECT observed_at FROM clock_hwm WHERE source='shared-clock'"
        ).fetchone()[0]
    assert datetime.fromisoformat(stamp) == _START + timedelta(seconds=499)


@pytest.mark.parametrize("seed", list(range(300)))
def test_fixed_seed_clock_sequences_never_revive_true_expired_permits(tmp_path, seed):
    rng = random.Random(20261011 + seed)
    policy = _policy(tmp_path)
    issuer = PermitIssuer(_ISSUER)
    db_path = tmp_path / "case.db"
    wall = Clock()
    ticks = Ticks()
    true_elapsed = 0.0
    current = PermitService(
        issuer, policy, db_path=db_path, clock=wall, monotonic=ticks
    )
    permits: list[tuple[object, float]] = []
    for _ in range(15):
        operation = rng.choice(("issue", "idle", "back", "forward", "verify", "restart"))
        if operation == "issue":
            ttl = rng.choice((2.0, 5.0, 20.0, 120.0))
            permit = current.issue(
                tool="state", op="set_load_target",
                args={"load": 0.5}, goal_id="clock-fuzz", ttl_seconds=ttl
            )
            permits.append((permit, true_elapsed + ttl))
        elif operation == "idle":
            delta = rng.uniform(0.1, 400.0)
            ticks.seconds += delta
            true_elapsed += delta
        elif operation == "back":
            wall.now -= timedelta(seconds=rng.choice((1, 100, 10000, 315360000)))
        elif operation == "forward":
            wall.now += timedelta(seconds=rng.choice((1, 12, 200, 40000)))
        elif operation == "restart":
            current.close()
            ticks = Ticks()
            current = PermitService(
                issuer, policy, db_path=db_path, clock=wall, monotonic=ticks
            )
        elif permits:
            permit, true_expiry = rng.choice(permits)
            try:
                current.verify_and_consume(
                    permit, "state", {"load": 0.5}, op="set_load_target"
                )
            except PermitVerificationError:
                pass
            else:
                assert true_elapsed < true_expiry, (
                    f"seed={seed}: expired permit authorized after {true_elapsed}s"
                )
    current.close()


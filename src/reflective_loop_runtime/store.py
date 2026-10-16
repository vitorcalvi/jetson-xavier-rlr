"""SQLite persistence for RLR telemetry, episodes, memories, goals, actions,
decisions, governor vetoes, and outcomes.

The store is append-mostly and entirely stdlib. It is intentionally generic so
that hardware-specific telemetry lives in adapters, not here.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    started_at  TEXT NOT NULL,
    metadata    TEXT
);
CREATE TABLE IF NOT EXISTS telemetry (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL,
    tick          INTEGER NOT NULL,
    sampled_at    TEXT NOT NULL,
    sampled_mono  REAL,
    temperature_c REAL,
    ram_used_mib  REAL,
    ram_total_mib REAL,
    thermal_level TEXT,
    source        TEXT,
    raw_json      TEXT
);
CREATE TABLE IF NOT EXISTS episodes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL,
    tick       INTEGER NOT NULL,
    kind       TEXT NOT NULL,
    content    TEXT NOT NULL,
    confidence REAL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL,
    tick       INTEGER NOT NULL,
    text       TEXT NOT NULL,
    importance INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS goals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL,
    created_tick INTEGER NOT NULL,
    text         TEXT NOT NULL,
    status       TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    closed_at    TEXT
);
CREATE TABLE IF NOT EXISTS decisions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL,
    tick          INTEGER NOT NULL,
    phase         TEXT NOT NULL,
    engine        TEXT NOT NULL,
    choice        TEXT,
    confidence    REAL,
    probabilities TEXT,
    mode          TEXT,
    latency_ms    REAL,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS actions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL,
    tick       INTEGER NOT NULL,
    name       TEXT NOT NULL,
    args_json  TEXT NOT NULL,
    outcome    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS vetoes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    tick        INTEGER NOT NULL,
    action_name TEXT NOT NULL,
    reason      TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS outcomes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    tick        INTEGER NOT NULL,
    action_id   INTEGER,
    result_json TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""


class Store:
    def __init__(self, path: str | Path, run_id: str | None = None):
        self.path = Path(path)
        self.run_id = run_id or uuid.uuid4().hex
        self.current_tick = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.migrate()
        self.begin_run()

    # -- lifecycle ---------------------------------------------------------
    def migrate(self) -> None:
        self.db.executescript(SCHEMA)
        self.db.commit()

    def begin_run(self, metadata: dict[str, Any] | None = None) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO runs (run_id, started_at, metadata) VALUES (?, ?, ?)",
            (self.run_id, utc_now(), json.dumps(metadata or {}, sort_keys=True)),
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    # -- writes ------------------------------------------------------------
    def record_telemetry(self, sample: dict[str, Any], source: str | None = None) -> int:
        cur = self.db.execute(
            """INSERT INTO telemetry
               (run_id, tick, sampled_at, sampled_mono, temperature_c,
                ram_used_mib, ram_total_mib, thermal_level, source, raw_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self.run_id,
                self.current_tick,
                sample.get("sampled_at") or utc_now(),
                sample.get("sampled_monotonic"),
                sample.get("temperature_c"),
                sample.get("ram_used_mib"),
                sample.get("ram_total_mib"),
                sample.get("thermal_level"),
                source,
                json.dumps(sample, sort_keys=True, default=str),
            ),
        )
        self.db.commit()
        return int(cur.lastrowid)

    def add_episode(self, kind: str, content: str, confidence: float | None = None) -> int:
        cur = self.db.execute(
            """INSERT INTO episodes (run_id, tick, kind, content, confidence, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (self.run_id, self.current_tick, kind, content, confidence, utc_now()),
        )
        self.db.commit()
        return int(cur.lastrowid)

    def add_memory(self, text: str, importance: int) -> int:
        cur = self.db.execute(
            """INSERT INTO memories (run_id, tick, text, importance, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (self.run_id, self.current_tick, text, int(importance), utc_now()),
        )
        self.db.commit()
        return int(cur.lastrowid)

    def recall_memories(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        # Deterministic, dependency-free recall: substring match, else recent by
        # importance then recency. Never raises on odd input.
        tokens = [t for t in (query or "").lower().split() if t]
        rows = self.db.execute(
            "SELECT id, tick, text, importance, created_at FROM memories WHERE run_id=?",
            (self.run_id,),
        ).fetchall()
        scored: list[tuple[int, float, int, sqlite3.Row]] = []
        for row in rows:
            haystack = row["text"].lower()
            hits = sum(1 for t in tokens if t in haystack)
            scored.append((hits, row["importance"], row["id"], row))
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return [dict(row) for _, _, _, row in scored[:limit]]

    def add_goal(self, text: str) -> int:
        cur = self.db.execute(
            """INSERT INTO goals (run_id, created_tick, text, status, created_at)
               VALUES (?, ?, ?, 'active', ?)""",
            (self.run_id, self.current_tick, text, utc_now()),
        )
        self.db.commit()
        return int(cur.lastrowid)

    def complete_goal(self, goal_id: int) -> bool:
        cur = self.db.execute(
            "UPDATE goals SET status='completed', closed_at=? WHERE id=? AND run_id=? AND status='active'",
            (utc_now(), int(goal_id), self.run_id),
        )
        self.db.commit()
        return cur.rowcount > 0

    def active_goals(self) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT id, text, created_tick FROM goals WHERE run_id=? AND status='active' ORDER BY id",
            (self.run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def record_decision(
        self,
        *,
        phase: str,
        engine: str,
        choice: str | None,
        confidence: float | None,
        probabilities: dict[str, float] | None,
        mode: str,
        latency_ms: float,
    ) -> int:
        cur = self.db.execute(
            """INSERT INTO decisions
               (run_id, tick, phase, engine, choice, confidence, probabilities, mode, latency_ms, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self.run_id,
                self.current_tick,
                phase,
                engine,
                choice,
                confidence,
                json.dumps(probabilities or {}, sort_keys=True),
                mode,
                latency_ms,
                utc_now(),
            ),
        )
        self.db.commit()
        return int(cur.lastrowid)

    def record_action(self, name: str, args: dict[str, Any], outcome: str) -> int:
        # ``args`` excludes the action name; rationale is retained for audit.
        cur = self.db.execute(
            """INSERT INTO actions (run_id, tick, name, args_json, outcome, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                self.run_id,
                self.current_tick,
                name,
                json.dumps(args, sort_keys=True, default=str),
                outcome,
                utc_now(),
            ),
        )
        self.db.commit()
        return int(cur.lastrowid)

    def record_veto(self, action_name: str, reason: str) -> int:
        cur = self.db.execute(
            """INSERT INTO vetoes (run_id, tick, action_name, reason, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (self.run_id, self.current_tick, action_name, reason, utc_now()),
        )
        self.db.commit()
        return int(cur.lastrowid)

    def record_outcome(self, action_id: int | None, result: dict[str, Any]) -> int:
        cur = self.db.execute(
            """INSERT INTO outcomes (run_id, tick, action_id, result_json, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                self.run_id,
                self.current_tick,
                action_id,
                json.dumps(result, sort_keys=True, default=str),
                utc_now(),
            ),
        )
        self.db.commit()
        return int(cur.lastrowid)

    # -- reads -------------------------------------------------------------
    def recent_actions(self, limit: int = 5, *, exclude_vetoed: bool = True) -> list[dict[str, Any]]:
        clause = "AND outcome!='vetoed'" if exclude_vetoed else ""
        rows = self.db.execute(
            f"SELECT id, tick, name, args_json, outcome FROM actions "
            f"WHERE run_id=? {clause} ORDER BY id DESC LIMIT ?",
            (self.run_id, int(limit)),
        ).fetchall()
        return [
            {"id": r["id"], "tick": r["tick"], "name": r["name"],
             "args": json.loads(r["args_json"]), "outcome": r["outcome"]}
            for r in rows
        ]

    def recent_episodes(self, limit: int = 12) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT id, tick, kind, content, confidence, created_at FROM episodes "
            "WHERE run_id=? ORDER BY id DESC LIMIT ?",
            (self.run_id, int(limit)),
        ).fetchall()
        return [dict(row) for row in rows]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for table in ("telemetry", "episodes", "memories", "goals", "decisions",
                      "actions", "vetoes", "outcomes"):
            out[table] = int(
                self.db.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE run_id=?", (self.run_id,)
                ).fetchone()[0]
            )
        return out

    def iter_table(self, table: str) -> Iterable[sqlite3.Row]:
        allowed = {"telemetry", "episodes", "memories", "goals", "decisions",
                   "actions", "vetoes", "outcomes", "runs"}
        if table not in allowed:
            raise ValueError(f"unknown table {table!r}")
        return self.db.execute(f"SELECT * FROM {table} WHERE run_id=?", (self.run_id,)).fetchall()

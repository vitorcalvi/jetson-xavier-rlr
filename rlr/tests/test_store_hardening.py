import sqlite3

from rlr.store import Store


def create_v2(path):
    db = sqlite3.connect(str(path))
    db.executescript("""
        CREATE TABLE capabilities (name TEXT PRIMARY KEY, value REAL NOT NULL,
          unit TEXT NOT NULL, source TEXT NOT NULL, measured_at TEXT NOT NULL);
        CREATE TABLE telemetry (id INTEGER PRIMARY KEY, ts TEXT NOT NULL,
          zones_c_json TEXT NOT NULL, ram_used_mib REAL, ram_total_mib REAL,
          power_mode TEXT, gpu_load REAL, dla_load REAL, fan_rpm REAL);
        CREATE TABLE episodes (id INTEGER PRIMARY KEY, ts TEXT NOT NULL, kind TEXT NOT NULL,
          detections_json TEXT, transcript TEXT, frame_path TEXT, audio_path TEXT, content TEXT);
        CREATE TABLE actions (id INTEGER PRIMARY KEY, ts TEXT NOT NULL, name TEXT NOT NULL,
          args_json TEXT NOT NULL, rationale TEXT NOT NULL, outcome TEXT NOT NULL,
          valid INTEGER NOT NULL, veto_reason TEXT);
        CREATE TABLE reflections (id INTEGER PRIMARY KEY, ts TEXT NOT NULL, summary TEXT NOT NULL);
        CREATE TABLE goals (id INTEGER PRIMARY KEY, created_at TEXT NOT NULL,
          completed_at TEXT, text TEXT NOT NULL, status TEXT NOT NULL);
        CREATE VIRTUAL TABLE memories USING fts5(text, importance UNINDEXED, ts UNINDEXED);
        CREATE TABLE ticks (id INTEGER PRIMARY KEY, ts TEXT NOT NULL,
          duration_ms REAL NOT NULL, outcome TEXT NOT NULL, error TEXT);
        PRAGMA user_version=2;
    """)
    db.execute("INSERT INTO actions(ts,name,args_json,rationale,outcome,valid) VALUES(?,?,?,?,?,?)",
               ("2026-10-03T00:00:00+00:00", "observe", "{}", "legacy", "observed", 1))
    db.commit()
    db.close()


def test_current_v2_schema_migrates_without_data_loss(tmp_path):
    path = tmp_path / "brain.db"
    create_v2(path)
    store = Store(path)
    assert store.db.execute("PRAGMA user_version").fetchone()[0] == 2
    assert store.db.execute("SELECT version FROM schema_migrations").fetchone()[0] == 3
    assert store.db.execute("SELECT COUNT(*) FROM actions").fetchone()[0] == 1
    tables = {row[0] for row in store.db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"policy_decisions", "execution_results", "safety_transitions"} <= tables
    columns = {row[1] for row in store.db.execute("PRAGMA table_info(telemetry)")}
    assert {"backend", "capture_duration_ms", "acquisition_stalled"} <= columns
    mode = store.db.execute("PRAGMA journal_mode").fetchone()[0].lower()
    assert mode == "wal"
    assert store.db.execute("PRAGMA busy_timeout").fetchone()[0] >= 5000
    store.close()

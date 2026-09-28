"""SQLite persistence for the RLR self-model and episodic memory."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


CAPABILITIES = [
    ("qwen_0_5b_generation", 41.99, "tokens/s", "bench/RESULTS.md §A, solo tg128"),
    ("qwen_1_5b_generation", 15.98, "tokens/s", "bench/RESULTS.md §A, solo tg128"),
    ("qwen_1_5b_generation_combined", 11.29, "tokens/s", "bench/RESULTS.md §D, combined"),
    ("llama_3_2_3b_generation", 9.47, "tokens/s", "bench/RESULTS.md §A, bounded batch"),
    ("qwen_1_5b_prompt_processing", 396.44, "tokens/s", "bench/RESULTS.md §A, solo pp512"),
    ("yolov8n_gpu_qps", 67.8935, "queries/s", "bench/RESULTS.md §B, FP16"),
    ("yolov8n_dla_qps", 43.426, "queries/s", "bench/RESULTS.md §B, FP16 with GPU fallback"),
    ("resnet50_gpu_qps", 128.165, "queries/s", "bench/RESULTS.md §B, FP16"),
    ("qwen_1_5b_peak_ram", 3029, "MiB", "bench/RESULTS.md §A"),
    ("resnet50_peak_ram", 4751, "MiB", "bench/RESULTS.md §B"),
    ("yolov8n_peak_ram", 3282, "MiB", "bench/RESULTS.md §B"),
    ("combined_peak_ram", 4421, "MiB", "bench/RESULTS.md §D"),
    ("qwen_1_5b_power", 8.241, "W", "bench/RESULTS.md §E, VDD_IN"),
    ("qwen_1_5b_efficiency", 1.93, "tokens/J", "bench/RESULTS.md §E"),
    ("resnet50_power", 9.575, "W", "bench/RESULTS.md §E, VDD_IN"),
    ("resnet50_efficiency", 13.43, "frames/J", "bench/RESULTS.md §E"),
    ("yolov8n_dla_power", 6.199, "W", "bench/RESULTS.md §E, VDD_IN"),
    ("yolov8n_dla_efficiency", 5.75, "frames/J", "bench/RESULTS.md §E"),
    ("combined_max_temperature", 64.5, "°C", "bench/RESULTS.md §D"),
    ("historical_fan_rpm", 0, "RPM", "bench/RESULTS.md, broken fan record"),
    ("thermal_conserve_threshold", 70, "°C", "RLR v1a safety policy"),
    ("thermal_degraded_threshold", 75, "°C", "RLR v1a safety policy"),
    ("thermal_cooldown_threshold", 78, "°C", "RLR v1a safety policy"),
    ("thermal_resume_threshold", 70, "°C", "RLR v1a safety policy"),
]


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.migrate()
        self.seed_capabilities()

    def close(self):
        self.db.close()

    def migrate(self):
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version > 2:
            raise RuntimeError("database schema is newer than this RLR build")
        if version == 0:
            with self.db:
                self.db.executescript("""
                    CREATE TABLE capabilities (
                        name TEXT PRIMARY KEY, value REAL NOT NULL, unit TEXT NOT NULL,
                        source TEXT NOT NULL, measured_at TEXT NOT NULL
                    );
                    CREATE TABLE telemetry (
                        id INTEGER PRIMARY KEY, ts TEXT NOT NULL,
                        zones_c_json TEXT NOT NULL, ram_used_mib REAL, ram_total_mib REAL,
                        power_mode TEXT, gpu_load REAL, dla_load REAL, fan_rpm REAL
                    );
                    CREATE TABLE episodes (
                        id INTEGER PRIMARY KEY, ts TEXT NOT NULL, kind TEXT NOT NULL,
                        detections_json TEXT, transcript TEXT, frame_path TEXT,
                        audio_path TEXT, content TEXT
                    );
                    CREATE TABLE actions (
                        id INTEGER PRIMARY KEY, ts TEXT NOT NULL, name TEXT NOT NULL,
                        args_json TEXT NOT NULL, rationale TEXT NOT NULL,
                        outcome TEXT NOT NULL, valid INTEGER NOT NULL,
                        veto_reason TEXT
                    );
                    CREATE TABLE reflections (
                        id INTEGER PRIMARY KEY, ts TEXT NOT NULL, summary TEXT NOT NULL
                    );
                    CREATE TABLE goals (
                        id INTEGER PRIMARY KEY, created_at TEXT NOT NULL,
                        completed_at TEXT, text TEXT NOT NULL, status TEXT NOT NULL
                    );
                    CREATE VIRTUAL TABLE memories USING fts5(
                        text, importance UNINDEXED, ts UNINDEXED
                    );
                    PRAGMA user_version=1;
                """)
            version = 1
        if version < 2:
            with self.db:
                self.db.execute("""CREATE TABLE ticks (
                    id INTEGER PRIMARY KEY, ts TEXT NOT NULL,
                    duration_ms REAL NOT NULL, outcome TEXT NOT NULL,
                    error TEXT
                )""")
                self.db.execute("PRAGMA user_version=2")

    def add_tick(self, duration_ms, outcome, error=None):
        with self.db:
            cur = self.db.execute(
                "INSERT INTO ticks(ts,duration_ms,outcome,error) VALUES(?,?,?,?)",
                (utc_now(), duration_ms, outcome, error))
        return cur.lastrowid

    def seed_capabilities(self):
        now = utc_now()
        with self.db:
            self.db.executemany(
                "INSERT OR IGNORE INTO capabilities(name,value,unit,source,measured_at) VALUES(?,?,?,?,?)",
                [(name, value, unit, source, now) for name, value, unit, source in CAPABILITIES],
            )

    def capabilities(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM capabilities ORDER BY name")]

    def add_telemetry(self, sample):
        with self.db:
            cur = self.db.execute("""INSERT INTO telemetry
                (ts,zones_c_json,ram_used_mib,ram_total_mib,power_mode,gpu_load,dla_load,fan_rpm)
                VALUES(?,?,?,?,?,?,?,?)""", (
                sample.get("ts") or utc_now(), json.dumps(sample.get("zones_c", [])),
                sample.get("ram_used_mib"), sample.get("ram_total_mib"), sample.get("power_mode"),
                sample.get("gpu_load"), sample.get("dla_load"), sample.get("fan_rpm", 0)))
        sample = dict(sample)
        sample["id"] = cur.lastrowid
        return sample

    def latest_telemetry(self):
        row = self.db.execute("SELECT * FROM telemetry ORDER BY id DESC LIMIT 1").fetchone()
        if row is None:
            return None
        value = dict(row)
        value["zones_c"] = json.loads(value.pop("zones_c_json"))
        return value

    def add_episode(self, kind, detections=None, transcript=None, frame_path=None, audio_path=None, content=None):
        with self.db:
            cur = self.db.execute("""INSERT INTO episodes
                (ts,kind,detections_json,transcript,frame_path,audio_path,content)
                VALUES(?,?,?,?,?,?,?)""", (utc_now(), kind,
                json.dumps(detections) if detections is not None else None,
                transcript, frame_path, audio_path, content))
        return cur.lastrowid

    def recent_episodes(self, limit=12):
        rows = self.db.execute("SELECT * FROM episodes ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        result = []
        for row in reversed(rows):
            item = dict(row)
            item["detections"] = json.loads(item.pop("detections_json")) if item["detections_json"] else None
            result.append(item)
        return result

    def add_action(self, name, args, rationale, outcome, valid=True, veto_reason=None):
        with self.db:
            cur = self.db.execute("""INSERT INTO actions
                (ts,name,args_json,rationale,outcome,valid,veto_reason) VALUES(?,?,?,?,?,?,?)""",
                (utc_now(), name, json.dumps(args, ensure_ascii=False), rationale, outcome,
                 int(bool(valid)), veto_reason))
        return cur.lastrowid

    def add_reflection(self, summary):
        with self.db:
            cur = self.db.execute("INSERT INTO reflections(ts,summary) VALUES(?,?)", (utc_now(), summary))
        return cur.lastrowid

    def latest_reflection(self):
        row = self.db.execute("SELECT * FROM reflections ORDER BY id DESC LIMIT 1").fetchone()
        return dict(row) if row else None

    def active_goals(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM goals WHERE status='active' ORDER BY id")]

    def add_goal(self, text):
        with self.db:
            cur = self.db.execute("INSERT INTO goals(created_at,text,status) VALUES(?,?, 'active')", (utc_now(), text))
        return cur.lastrowid

    def complete_goal(self, goal_id):
        with self.db:
            cur = self.db.execute("UPDATE goals SET status='complete',completed_at=? WHERE id=? AND status='active'",
                                  (utc_now(), goal_id))
        return cur.rowcount > 0

    def remember(self, text, importance):
        stamp = utc_now()
        with self.db:
            cur = self.db.execute("INSERT INTO memories(text,importance,ts) VALUES(?,?,?)", (text, importance, stamp))
        self.add_episode("memory", content=text)
        return cur.lastrowid

    def recall(self, query, limit=5):
        terms = [part.replace('"', ' ') for part in query.split() if part.strip()]
        if not terms:
            return []
        match = " AND ".join('"%s"' % term for term in terms)
        try:
            rows = self.db.execute("SELECT text,importance,ts FROM memories WHERE memories MATCH ? ORDER BY rank LIMIT ?",
                                   (match, limit)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [dict(r) for r in rows]

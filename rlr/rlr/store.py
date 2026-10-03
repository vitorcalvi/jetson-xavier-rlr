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
        self.db = sqlite3.connect(str(self.path), timeout=5.0)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.execute("PRAGMA journal_mode=WAL")
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
                        args_json TEXT NOT NULL, rationale TEXT NOT NULL, outcome TEXT NOT NULL,
                        valid INTEGER NOT NULL, veto_reason TEXT
                    );
                    CREATE TABLE reflections (id INTEGER PRIMARY KEY, ts TEXT NOT NULL, summary TEXT NOT NULL);
                    CREATE TABLE goals (
                        id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, completed_at TEXT,
                        text TEXT NOT NULL, status TEXT NOT NULL
                    );
                    CREATE VIRTUAL TABLE memories USING fts5(text, importance UNINDEXED, ts UNINDEXED);
                    PRAGMA user_version=1;
                """)
            version = 1
        if version < 2:
            with self.db:
                self.db.execute("""CREATE TABLE ticks (
                    id INTEGER PRIMARY KEY, ts TEXT NOT NULL, duration_ms REAL NOT NULL,
                    outcome TEXT NOT NULL, error TEXT)""")
                self.db.execute("PRAGMA user_version=2")
            version = 2
        marker = self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
        ).fetchone()
        applied = None
        if marker is not None:
            applied = self.db.execute("SELECT 1 FROM schema_migrations WHERE version=3").fetchone()
        if applied is None:
            with self.db:
                self.db.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
                for ddl in (
                    "ALTER TABLE telemetry ADD COLUMN backend TEXT",
                    "ALTER TABLE telemetry ADD COLUMN captured_monotonic REAL",
                    "ALTER TABLE telemetry ADD COLUMN capture_duration_ms REAL",
                    "ALTER TABLE telemetry ADD COLUMN ram_percent REAL",
                    "ALTER TABLE telemetry ADD COLUMN power_w REAL",
                    "ALTER TABLE telemetry ADD COLUMN swap_json TEXT",
                    "ALTER TABLE telemetry ADD COLUMN load_average_json TEXT",
                    "ALTER TABLE telemetry ADD COLUMN acquisition_stalled INTEGER NOT NULL DEFAULT 0",
                    "ALTER TABLE actions ADD COLUMN request_id TEXT",
                    "ALTER TABLE actions ADD COLUMN policy_version TEXT",
                    "ALTER TABLE actions ADD COLUMN postcondition_result TEXT"
                ):
                    try:
                        self.db.execute(ddl)
                    except sqlite3.OperationalError as exc:
                        if "duplicate column" not in str(exc).lower():
                            raise
                self.db.executescript("""
                    CREATE TABLE IF NOT EXISTS policy_decisions (
                        id INTEGER PRIMARY KEY, ts TEXT NOT NULL, request_id TEXT NOT NULL UNIQUE,
                        decision TEXT NOT NULL, requested_action_json TEXT NOT NULL,
                        effective_action_json TEXT NOT NULL, reason_code TEXT NOT NULL,
                        reason TEXT NOT NULL, policy_version TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS execution_results (
                        request_id TEXT PRIMARY KEY, ts TEXT NOT NULL, operation TEXT NOT NULL,
                        outcome TEXT NOT NULL, exit_code INTEGER, detail TEXT, observed_state_json TEXT
                    );
                    CREATE TABLE IF NOT EXISTS safety_transitions (
                        id INTEGER PRIMARY KEY, ts TEXT NOT NULL, from_state TEXT, to_state TEXT NOT NULL,
                        reason TEXT NOT NULL, temperature_c REAL, telemetry_backend TEXT
                    );
                    CREATE INDEX IF NOT EXISTS idx_actions_request_id ON actions(request_id);
                """)
                self.db.execute("INSERT OR IGNORE INTO schema_migrations(version,applied_at) VALUES(3,?)",
                                (utc_now(),))

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
                (ts,zones_c_json,ram_used_mib,ram_total_mib,power_mode,gpu_load,dla_load,fan_rpm,
                 backend,captured_monotonic,capture_duration_ms,ram_percent,power_w,swap_json,
                 load_average_json,acquisition_stalled)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                sample.get("ts") or utc_now(), json.dumps(sample.get("zones_c", [])),
                sample.get("ram_used_mib"), sample.get("ram_total_mib"), sample.get("power_mode"),
                sample.get("gpu_load"), sample.get("dla_load"), sample.get("fan_rpm"),
                sample.get("backend"), sample.get("captured_monotonic"),
                sample.get("capture_duration_ms"), sample.get("ram_percent"), sample.get("power_w"),
                json.dumps(sample.get("swap")) if sample.get("swap") is not None else None,
                json.dumps(sample.get("load_average")) if sample.get("load_average") is not None else None,
                int(bool(sample.get("acquisition_stalled")))))
        sample = dict(sample)
        sample["id"] = cur.lastrowid
        if cur.lastrowid and cur.lastrowid % 500 == 0:
            self.prune_routine_data()
        return sample

    def latest_telemetry(self):
        row = self.db.execute("SELECT * FROM telemetry ORDER BY id DESC LIMIT 1").fetchone()
        if row is None:
            return None
        value = dict(row)
        value["zones_c"] = json.loads(value.pop("zones_c_json"))
        value["temperatures"] = {"zone_%d" % i: t for i, t in enumerate(value["zones_c"])}
        value["swap"] = json.loads(value.pop("swap_json")) if value.get("swap_json") else None
        value["load_average"] = json.loads(value.pop("load_average_json")) if value.get("load_average_json") else None
        value["acquisition_stalled"] = bool(value.get("acquisition_stalled"))
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

    def add_action(self, name, args, rationale, outcome, valid=True, veto_reason=None,
                   request_id=None, policy_version=None, postcondition_result=None):
        with self.db:
            cur = self.db.execute("""INSERT INTO actions
                (ts,name,args_json,rationale,outcome,valid,veto_reason,request_id,policy_version,postcondition_result)
                VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (utc_now(), name, json.dumps(args, ensure_ascii=False), rationale, outcome,
                 int(bool(valid)), veto_reason, request_id, policy_version, postcondition_result))
        return cur.lastrowid

    def recent_actions(self, limit=8):
        rows = self.db.execute("""SELECT ts,name,args_json,outcome,rationale,veto_reason,request_id,policy_version
                                 FROM actions WHERE valid=1 ORDER BY id DESC LIMIT ?""",
                               (limit,)).fetchall()
        return [dict(row) for row in rows]

    def recent_execution_failures(self, limit=5):
        rows = self.db.execute("SELECT outcome FROM execution_results ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return sum(1 for row in rows if row[0] in ("FAILED_EXECUTION", "FAILED_POSTCONDITION"))

    def recent_incidents(self, limit=8):
        kinds = ("governor", "governor_veto", "policy_deny", "action_error",
                 "thermal_watchdog", "self_report_error", "reflection_error")
        marks = ",".join("?" for _ in kinds)
        rows = self.db.execute(
            "SELECT ts,kind,content FROM episodes WHERE kind IN (%s) ORDER BY id DESC LIMIT ?" % marks,
            kinds + (int(limit),)).fetchall()
        return [dict(row) for row in rows]

    def seconds_since_aggressive_action(self):
        row = self.db.execute("""SELECT ts FROM actions
                               WHERE name='set_load_target' AND valid=1 AND outcome LIKE 'applied%'
                               ORDER BY id DESC LIMIT 1""").fetchone()
        if row is None:
            return None
        try:
            stamp = datetime.fromisoformat(row[0])
            now = datetime.now(timezone.utc)
            return max(0.0, (now - stamp).total_seconds())
        except (TypeError, ValueError):
            return None

    def add_reflection(self, summary):
        with self.db:
            cur = self.db.execute("INSERT INTO reflections(ts,summary) VALUES(?,?)", (utc_now(), summary))
        return cur.lastrowid

    def update_reflection_summary(self, reflection_id, summary):
        with self.db:
            self.db.execute("UPDATE reflections SET summary=? WHERE id=?", (summary, reflection_id))
        return reflection_id

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

    def add_policy_decision(self, request_id, decision):
        with self.db:
            cur = self.db.execute("""INSERT OR REPLACE INTO policy_decisions
                (ts,request_id,decision,requested_action_json,effective_action_json,
                 reason_code,reason,policy_version) VALUES(?,?,?,?,?,?,?,?)""", (
                utc_now(), request_id, decision["decision"],
                json.dumps(decision["requested_action"], ensure_ascii=False),
                json.dumps(decision["effective_action"], ensure_ascii=False),
                decision["reason_code"], decision["reason"], decision["policy_version"]))
        return cur.lastrowid

    def policy_decision(self, request_id):
        row = self.db.execute("SELECT * FROM policy_decisions WHERE request_id=?", (request_id,)).fetchone()
        if row is None:
            return None
        value = dict(row)
        value["requested_action"] = json.loads(value.pop("requested_action_json"))
        value["effective_action"] = json.loads(value.pop("effective_action_json"))
        return value

    def add_execution_result(self, result):
        observed = result.get("observed_state")
        with self.db:
            self.db.execute("""INSERT OR IGNORE INTO execution_results
                (request_id,ts,operation,outcome,exit_code,detail,observed_state_json)
                VALUES(?,?,?,?,?,?,?)""", (
                result["request_id"], utc_now(), result["operation"], result["outcome"],
                result.get("exit_code"), result.get("detail"),
                json.dumps(observed, ensure_ascii=False) if observed is not None else None))

    def execution_result(self, request_id):
        row = self.db.execute("SELECT * FROM execution_results WHERE request_id=?", (request_id,)).fetchone()
        if row is None:
            return None
        value = dict(row)
        raw = value.pop("observed_state_json")
        value["observed_state"] = json.loads(raw) if raw else None
        return value

    def add_safety_transition(self, old_state, new_state, reason,
                              temperature_c=None, telemetry_backend=None):
        with self.db:
            cur = self.db.execute("""INSERT INTO safety_transitions
                (ts,from_state,to_state,reason,temperature_c,telemetry_backend)
                VALUES(?,?,?,?,?,?)""", (utc_now(), old_state, new_state, reason,
                temperature_c, telemetry_backend))
        return cur.lastrowid

    def latest_safety_state(self):
        row = self.db.execute("SELECT to_state FROM safety_transitions ORDER BY id DESC LIMIT 1").fetchone()
        return row[0] if row else None

    def prune_routine_data(self, telemetry_keep=10000, ticks_keep=10000):
        telemetry_keep = max(100, int(telemetry_keep))
        ticks_keep = max(100, int(ticks_keep))
        with self.db:
            self.db.execute("""DELETE FROM telemetry WHERE id <= COALESCE(
                (SELECT id FROM telemetry ORDER BY id DESC LIMIT 1 OFFSET ?), -1)""",
                            (telemetry_keep,))
            self.db.execute("""DELETE FROM ticks WHERE id <= COALESCE(
                (SELECT id FROM ticks ORDER BY id DESC LIMIT 1 OFFSET ?), -1)""",
                            (ticks_keep,))

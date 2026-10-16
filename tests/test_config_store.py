"""Config validation and SQLite persistence behaviour."""

from __future__ import annotations

import pytest
from conftest import make_config

from reflective_loop_runtime.config import Config
from reflective_loop_runtime.store import Store


class TestConfig:
    def test_default_db_lives_in_data_dir(self, tmp_path):
        config = Config(data_dir=tmp_path)
        assert config.db_path.parent == tmp_path

    def test_db_outside_data_dir_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            Config(data_dir=tmp_path / "data", db_path=tmp_path / "elsewhere.db")

    def test_bad_confidence_threshold_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            Config(data_dir=tmp_path, confidence_threshold=1.5)

    def test_bad_tick_seconds_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            Config(data_dir=tmp_path, tick_seconds=0.0)

    def test_non_monotonic_thermal_thresholds_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            Config(data_dir=tmp_path, conserve_c=80.0, degraded_c=70.0)

    def test_ensure_dirs_creates_directory(self, tmp_path):
        config = Config(data_dir=tmp_path / "nested" / "data")
        config.ensure_dirs()
        assert config.data_dir.is_dir()


class TestStore:
    def test_domain_tables_exist(self, tmp_path):
        store = Store(make_config(tmp_path).db_path)
        try:
            names = {
                row["name"]
                for row in store.db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        finally:
            store.close()
        for table in (
            "telemetry",
            "episodes",
            "memories",
            "goals",
            "decisions",
            "actions",
            "vetoes",
            "outcomes",
            "runs",
        ):
            assert table in names

    def test_roundtrip_and_counts(self, tmp_path):
        store = Store(make_config(tmp_path).db_path)
        try:
            store.current_tick = 1
            store.record_telemetry({"temperature_c": 55.0, "ram_total_mib": 8192})
            store.add_episode("telemetry", "digest", None)
            store.add_memory("a lesson", 3)
            goal = store.add_goal("stay cool")
            store.record_decision(
                phase="p",
                engine="e",
                choice="observe",
                confidence=0.6,
                probabilities={"observe": 1.0},
                mode="fast",
                latency_ms=0.5,
            )
            action_id = store.record_action("observe", {}, "executed")
            store.record_outcome(action_id, {"ok": True})
            counts = store.counts()
            assert counts["telemetry"] == 1
            assert counts["memories"] == 1
            assert counts["goals"] == 1
            assert store.complete_goal(goal) is True
            assert store.active_goals() == []
        finally:
            store.close()

    def test_recall_orders_by_relevance_then_importance(self, tmp_path):
        store = Store(make_config(tmp_path).db_path)
        try:
            store.add_memory("thermal cooldown worked", 2)
            store.add_memory("unrelated note", 5)
            hits = store.recall_memories("thermal")
            assert hits[0]["text"] == "thermal cooldown worked"
        finally:
            store.close()

    def test_recent_actions_excludes_vetoed(self, tmp_path):
        store = Store(make_config(tmp_path).db_path)
        try:
            store.record_action("cooldown", {"seconds": 30}, "vetoed")
            store.record_action("observe", {}, "executed")
            names = [a["name"] for a in store.recent_actions()]
            assert "cooldown" not in names
            assert "observe" in names
        finally:
            store.close()

    def test_iter_table_rejects_unknown_table(self, tmp_path):
        store = Store(make_config(tmp_path).db_path)
        try:
            with pytest.raises(ValueError):
                store.iter_table("users; DROP TABLE runs")
        finally:
            store.close()

from rlr.store import Store


def test_migration_and_seeded_capabilities(tmp_path):
    path = tmp_path / "nested" / "brain.db"
    store = Store(path)
    assert store.db.execute("PRAGMA user_version").fetchone()[0] == 2
    tables = {row[0] for row in store.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"capabilities", "telemetry", "episodes", "actions", "reflections", "goals", "memories", "ticks"} <= tables
    assert store.db.execute("SELECT value FROM capabilities WHERE name='qwen_1_5b_generation'").fetchone()[0] == 15.98
    store.close()
    reopened = Store(path)
    assert len(reopened.capabilities()) >= 10
    reopened.close()


def test_fts_recall(tmp_path):
    store = Store(tmp_path / "brain.db")
    store.remember("The Xavier fan is broken and reports zero RPM", 5)
    found = store.recall("Xavier fan")
    assert found and "zero RPM" in found[0]["text"]
    store.close()


def test_tick_latency_and_failure_are_recorded(tmp_path):
    from rlr.runtime import Config, ReflectiveLoop

    class BrokenLLM:
        def complete(self, *_args):
            raise RuntimeError("inference failed")

    loop = ReflectiveLoop(BrokenLLM(), Config(str(tmp_path), self_report_every=0),
                          lambda: {"zones_c": [60], "ram_used_mib": 100,
                                   "ram_total_mib": 6833, "power_mode": "MODE_10W_DESKTOP"})
    try:
        try:
            loop.tick()
        except RuntimeError:
            pass
        else:
            assert False, "expected model failure"
        row = loop.store.db.execute("SELECT duration_ms,outcome,error FROM ticks").fetchone()
        assert row[0] >= 0
        assert row[1] == "error"
        assert row[2] == "RuntimeError: inference failed"
    finally:
        loop.close()

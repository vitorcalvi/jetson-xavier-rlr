"""Run-scoped autonomy metrics for one RLR run directory.

Reads only the artifacts that belong to AUTONOMY_DIR, so a run is never
inflated by historical rows in the shared evidence database. Read-only apart
from the explicit recall probes, which are recorded as episodes.
"""
import csv
import json
import math
import os
import re
import sqlite3
import statistics
import sys
from pathlib import Path

ROOT = Path(os.environ.get("AUTONOMY_DIR", "/models/rlr/autonomy-run"))
CODE = os.environ.get("AUTONOMY_CODE", "/home/vitor/jetson-xavier-rlr/rlr")
DB = ROOT / "brain.db"
sys.path.insert(0, CODE)

con = sqlite3.connect(str(DB))
con.row_factory = sqlite3.Row

ticks = con.execute("SELECT duration_ms,outcome,error FROM ticks ORDER BY id").fetchall()
total_actions = con.execute("SELECT COUNT(*) FROM actions").fetchone()[0]
valid_actions = con.execute("SELECT COALESCE(SUM(valid),0) FROM actions").fetchone()[0]
by_type = {row["name"]: row["n"] for row in
           con.execute("SELECT name,COUNT(*) AS n FROM actions GROUP BY name ORDER BY name")}
vetoes = con.execute("SELECT COUNT(*) FROM actions WHERE outcome='vetoed'").fetchone()[0]
veto_reasons = [row[0] for row in con.execute(
    "SELECT DISTINCT veto_reason FROM actions WHERE outcome='vetoed' AND veto_reason IS NOT NULL")]

latencies = sorted(row["duration_ms"] for row in ticks)


def percentile(values, fraction):
    if not values:
        return None
    return values[max(0, math.ceil(fraction * len(values)) - 1)]


temperatures = []
for (encoded,) in con.execute("SELECT zones_c_json FROM telemetry"):
    temperatures.extend(json.loads(encoded))
ram = [row[0] for row in
       con.execute("SELECT ram_used_mib FROM telemetry WHERE ram_used_mib IS NOT NULL")]

scores = []
for (content,) in con.execute("SELECT content FROM episodes WHERE kind='self_report'"):
    try:
        scores.append(json.loads(content)["score"])
    except Exception:
        pass
temp_mae = [s["temperature_abs_error_c"] for s in scores if s.get("temperature_abs_error_c") is not None]
ram_mae = [s["ram_abs_error_mib"] for s in scores if s.get("ram_abs_error_mib") is not None]
mode_match = [s["power_mode_match"] for s in scores if s.get("power_mode_match") is not None]
goal_match = [s["active_goal_match"] for s in scores if s.get("active_goal_match") is not None]

watch_path = ROOT / "thermal.csv"
watch_temps, watch_ram, pauses = [], [], 0
if watch_path.exists():
    for row in csv.DictReader(open(watch_path)):
        if row.get("max_temp_c"):
            watch_temps.append(float(row["max_temp_c"]))
        if row.get("ram_used_mib") and float(row["ram_used_mib"]) >= 0:
            watch_ram.append(float(row["ram_used_mib"]))
        pauses += int(row.get("paused") or 0)

from rlr.store import Store  # noqa: E402

store = Store(str(DB))
lookups = []
for query in ("10W thermal", "goal", "RAM"):
    hits = store.recall(query)
    lookups.append({"query": query, "hits": hits})
    store.add_episode("recall_lookup", content=json.dumps(lookups[-1], ensure_ascii=False))
store.close()

goals = [dict(row) for row in con.execute("SELECT id,text,status FROM goals ORDER BY id")]
reflections = [row[0] for row in con.execute("SELECT summary FROM reflections ORDER BY id")]
goals_chosen = con.execute(
    "SELECT COUNT(*) FROM episodes WHERE kind='governor_veto'").fetchone()[0]

# Reflection scoring lives in rlr_scoring.py so it can be unit tested; a rule
# only ever run against one real model output is easy to get permissively wrong.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rlr_scoring import score_reflections  # noqa: E402

capability_table = [(row[0], row[1]) for row in
                    con.execute("SELECT name,value FROM capabilities")]
reflection_scores = score_reflections(reflections, by_type, goals, temperatures, capability_table)
grounded = [score for score in reflection_scores if score["grounded"]]
# Fabrications that pass every other rule: a real capability name bound to the
# wrong number, or a first-person claim of an action the run never applied.
fabricated = [score for score in reflection_scores
              if score.get("misattributed_capabilities") or score.get("fabricated_actions")]
reflect_actions = con.execute(
    "SELECT COUNT(*) FROM actions WHERE name='reflect'").fetchone()[0]
reflect_outcomes = {row["outcome"]: row["n"] for row in con.execute(
    "SELECT outcome,COUNT(*) AS n FROM actions WHERE name='reflect' GROUP BY outcome")}
reflect_failed_ticks = con.execute(
    "SELECT COUNT(*) FROM ticks WHERE outcome='reflection_failed'").fetchone()[0]
# Reflections the model produced that the runtime rejected for length. A high
# rate here means the budget is being enforced, not that reflection is absent.
reflection_rejected = [row["content"] for row in
                       con.execute("SELECT content FROM episodes WHERE kind='reflection_invalid'")]
over_budget = [re.search(r"is (\d+) words", text) for text in reflection_rejected]
over_budget = [int(m.group(1)) for m in over_budget if m]

probe = {}
if (ROOT / "probe.json").exists():
    try:
        probe = json.loads((ROOT / "probe.json").read_text())
    except Exception:
        probe = {"error": "probe.json is unreadable"}
discrimination = probe.get("discrimination") or {}

def supervisor_line():
    """Last supervisor log line, or null when the run left no supervisor log."""
    try:
        return (ROOT / "supervisor.log").read_text().strip().splitlines()[-1]
    except (OSError, IndexError):
        return None


print(json.dumps({
    "run_dir": str(ROOT),
    "supervisor": supervisor_line(),
    "ticks": len(ticks),
    "tick_errors": sum(1 for row in ticks if row["error"]),
    "tick_outcomes": {row["outcome"]: row["n"] for row in
                      con.execute("SELECT outcome,COUNT(*) AS n FROM ticks GROUP BY outcome ORDER BY outcome")},
    "actions_total": total_actions,
    "actions_valid": valid_actions,
    "action_validity_rate_pct": round(100.0 * valid_actions / total_actions, 2) if total_actions else None,
    "actions_by_type": by_type,
    "governor_vetoes": vetoes,
    "veto_reasons": veto_reasons,
    "governor_episodes": goals_chosen,
    "abstains": by_type.get("abstain", 0),
    "latency_p50_ms": round(statistics.median(latencies), 2) if latencies else None,
    "latency_p95_ms": round(percentile(latencies, 0.95), 2) if latencies else None,
    "max_latency_ms": round(max(latencies), 2) if latencies else None,
    "peak_ram_db_mib": round(max(ram), 1) if ram else None,
    "peak_ram_watch_mib": round(max(watch_ram), 1) if watch_ram else None,
    "max_temp_db_c": round(max(temperatures), 1) if temperatures else None,
    "max_temp_watch_c": round(max(watch_temps), 1) if watch_temps else None,
    "watchdog_samples": len(watch_temps),
    "thermal_pause_samples": pauses,
    "self_reports": len(scores),
    "self_report_abstentions": sum(1 for s in scores if s.get("abstained")),
    "temperature_mae_c": round(statistics.mean(temp_mae), 3) if temp_mae else None,
    "ram_mae_mib": round(statistics.mean(ram_mae), 2) if ram_mae else None,
    "power_mode_accuracy_pct": round(100 * statistics.mean(mode_match), 1) if mode_match else None,
    # Copy metric only: the goal was in the prompt. The withheld condition
    # below is the one that can show internal state tracking.
    "active_goal_copy_accuracy_pct": round(100 * statistics.mean(goal_match), 1) if goal_match else None,
    # The ablation lives under the discrimination result in probe.json.
    "goal_recall_withheld": discrimination.get("goal_recall_withheld"),
    "goal_recall_abstained": discrimination.get("goal_recall_abstained"),
    "goals": goals,
    "reflect_actions_requested": reflect_actions,
    "reflect_outcomes": reflect_outcomes,
    "reflection_failed_ticks": reflect_failed_ticks,
    "reflection_rejected_over_budget": len(over_budget),
    "reflection_word_counts_rejected": sorted(over_budget),
    "reflection_count": len(reflections),
    "reflection_grounded_count": len(grounded),
    "reflection_grounded_pct": round(100.0 * len(grounded) / len(reflection_scores), 1)
        if reflection_scores else None,
    "reflection_fabricated_count": len(fabricated),
    "reflection_distinct_count": len(set(reflections)),
    "reflection_scores": reflection_scores,
    "reflection_excerpts": [text[:200] for text in reflections[:3]],
    "probe": probe,
    "recall_examples": lookups,
}, ensure_ascii=False, indent=2))
con.close()

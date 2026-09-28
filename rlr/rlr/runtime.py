"""Autonomous reflective loop and its side-effect boundary."""
import json
import os
import shutil
import time
import uuid
from pathlib import Path

from .actions import InvalidAction, action_json_schema, validate_action
from .governor import Governor
from .store import Store
from .telemetry import sample_telemetry, summarize_mode


REPORT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "temperature_c": {"type": ["number", "null"]},
        "ram_used_mib": {"type": ["number", "null"]},
        "power_mode": {"type": ["string", "null"]},
        "active_goal": {"type": ["string", "null"]},
    },
    "required": ["temperature_c", "ram_used_mib", "power_mode", "active_goal"],
}
REFLECTION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"summary": {"type": "string", "minLength": 1, "maxLength": 900}},
    "required": ["summary"],
}


class Config:
    def __init__(self, data_dir="/models/rlr", tick_seconds=10, episode_window=12,
                 reflection_minutes=15, self_report_every=10, stale_seconds=30,
                 media_limit_bytes=2 * 1024 * 1024 * 1024, db_path=None):
        self.data_dir = Path(data_dir)
        self.db_path = Path(db_path) if db_path else self.data_dir / "brain.db"
        root = self.data_dir.resolve()
        database = self.db_path.resolve()
        if os.path.commonpath([str(root), str(database)]) != str(root):
            raise ValueError("the RLR database must be inside the configured data directory")
        self.tick_seconds = tick_seconds
        self.episode_window = episode_window
        self.reflection_minutes = reflection_minutes
        self.self_report_every = self_report_every
        self.stale_seconds = stale_seconds
        self.media_limit_bytes = media_limit_bytes


class ReflectiveLoop:
    def __init__(self, llm, config=None, telemetry_provider=sample_telemetry, clock=time.monotonic,
                 sleep=time.sleep):
        self.config = config or Config()
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        self.store = Store(self.config.db_path)
        self.llm = llm
        self.telemetry_provider = telemetry_provider
        self.clock = clock
        self.sleep = sleep
        self.governor = Governor(self.store, stale_seconds=self.config.stale_seconds)
        self.settings = {"vision_fps": 3.0, "reply_tokens": 64}
        self.heavy_engines_paused = False
        self.forced_cooldown = False
        self.tick_count = 0
        self.cooldown_until = 0.0
        self.last_reflection_monotonic = self.clock()

    def close(self):
        self.store.close()

    def record_perception(self, detections=None, transcript=None, frame_path=None, audio_path=None,
                          retain_media=False):
        """Persist an episode; optionally copy media under the configured disk budget."""
        kept_frame = self._retain(frame_path) if retain_media and frame_path else None
        kept_audio = self._retain(audio_path) if retain_media and audio_path else None
        return self.store.add_episode("perception", detections, transcript, kept_frame, kept_audio)

    def _retain(self, source):
        source_path = Path(source).resolve()
        if not source_path.is_file():
            return None
        media_dir = self.config.data_dir / "media"
        current = sum(p.stat().st_size for p in media_dir.glob("*") if p.is_file()) if media_dir.exists() else 0
        size = source_path.stat().st_size
        if size + current > self.config.media_limit_bytes:
            return None
        media_dir.mkdir(parents=True, exist_ok=True)
        dest = media_dir / (uuid.uuid4().hex + source_path.suffix[:12])
        shutil.copyfile(str(source_path), str(dest))
        return str(dest)

    def context(self, sample):
        max_temp = self.governor.max_temp(sample)
        state = {"telemetry": sample, "max_temperature_c": max_temp,
                 "thermal_level": self.governor.thermal_level(sample),
                 "settings": self.settings, "heavy_engines_paused": self.heavy_engines_paused}
        recent = self.store.recent_episodes(self.config.episode_window)
        pinned = self.store.db.execute(
            "SELECT * FROM episodes WHERE kind='perception' ORDER BY id DESC LIMIT 2").fetchall()
        present = {episode["id"] for episode in recent}
        for row in reversed(pinned):
            if row["id"] not in present:
                episode = dict(row)
                episode["detections"] = json.loads(episode.pop("detections_json")) if episode["detections_json"] else None
                recent.insert(0, episode)
        return {
            "self_state": state,
            "capabilities": self.store.capabilities(),
            "recent_episodes": recent,
            "active_goals": self.store.active_goals(),
            "last_reflection": self.store.latest_reflection(),
        }

    def tick(self):
        started = self.clock()
        outcome = "error"
        error = None
        try:
            result = self._tick()
            outcome = str(result.get("action", "unknown"))
            return result
        except Exception as exc:
            error = (type(exc).__name__ + ": " + str(exc))[:1000]
            raise
        finally:
            self.store.add_tick(max(0.0, (self.clock() - started) * 1000.0), outcome, error)

    def _tick(self):
        self.tick_count += 1
        sample = dict(self.telemetry_provider() or {})
        sample.setdefault("ts", None)
        sample["sampled_monotonic"] = self.clock()
        sample.setdefault("fan_rpm", 0)
        persisted = self.store.add_telemetry(sample)
        sample.update({k: v for k, v in persisted.items() if k != "id"})
        temperature = self.governor.max_temp(sample)

        if temperature is not None and temperature >= Governor.COOLDOWN_C:
            self.forced_cooldown = True
        if self.forced_cooldown:
            if sample.get("zones_c") and all(t < Governor.RESUME_C for t in sample["zones_c"]):
                self.forced_cooldown = False
            else:
                self.heavy_engines_paused = True
                self.store.add_action("cooldown", {"seconds": self.config.tick_seconds},
                                      "Thermal limit reached; wait until every zone is below 70°C.",
                                      "forced_cooldown", valid=True)
                self.store.add_episode("governor", content="Forced cooldown active; LLM and heavy engines paused.")
                return {"action": "cooldown", "forced": True, "sample": sample}
        self.heavy_engines_paused = self.clock() < self.cooldown_until
        self._apply_thermal_caps(temperature)

        stale = self.governor.stale_reason(sample, self.clock())
        if stale:
            proposed = {"action": "abstain", "reason": stale, "rationale": "Telemetry does not support a safe decision."}
        else:
            prompt = self._action_prompt(self.context(sample))
            try:
                raw = self.llm.complete(prompt, action_json_schema(), "rlr_action")
                proposed = validate_action(raw)
            except (InvalidAction, KeyError, TypeError, ValueError) as exc:
                self.store.add_action("invalid", {"response": str(locals().get("raw", ""))[:4000]},
                                      "Model response did not pass action schema validation.",
                                      "invalid", valid=False, veto_reason=str(exc))
                self.store.add_episode("invalid_action", content=str(exc))
                proposed = {"action": "observe", "rationale": "Invalid model action; safely fell back to observe."}
        result = self._execute(proposed, sample)
        if self.config.self_report_every > 0 and self.tick_count % self.config.self_report_every == 0:
            self.self_report(sample)
        if self.clock() - self.last_reflection_monotonic >= self.config.reflection_minutes * 60:
            self.reflect()
        return dict(result, sample=sample)

    def _action_prompt(self, context):
        return ("You are the decision core of a standalone Reflective Loop Runtime. Choose exactly one "
                "whitelisted action as JSON conforming to the supplied JSON Schema. Use evidence from the "
                "self-state, measured capabilities, episodes, goals, and reflection. Abstain when evidence "
                "is uncertain. Never request shell, network, power, clock, thermal-guard, or arbitrary file actions.\n"
                "<context>\n" + json.dumps(context, ensure_ascii=False, sort_keys=True) + "\n</context>")

    def _apply_thermal_caps(self, temperature):
        if temperature is None:
            return
        if temperature >= Governor.DEGRADED_C:
            caps = (1.0, 48)
        elif temperature >= Governor.CONSERVE_C:
            caps = (2.0, 64)
        else:
            return
        changed = {}
        if self.settings["vision_fps"] > caps[0]:
            self.settings["vision_fps"] = caps[0]
            changed["vision_fps"] = caps[0]
        if self.settings["reply_tokens"] > caps[1]:
            self.settings["reply_tokens"] = caps[1]
            changed["reply_tokens"] = caps[1]
        if changed:
            self.store.add_action("thermal_governor", changed, "Deterministic thermal degradation cap.",
                                  "applied", valid=True)
            self.store.add_episode("governor", content=json.dumps({"thermal_caps": changed, "temperature_c": temperature}))

    def _execute(self, action, sample):
        name = action["action"]
        rationale = action["rationale"]
        veto = self.governor.veto(action, sample, self.settings)
        if veto:
            self.store.add_action(name, action, rationale, "vetoed", valid=True, veto_reason=veto)
            self.store.add_episode("governor_veto", content=json.dumps({"action": action, "reason": veto}))
            return {"action": name, "outcome": "vetoed", "reason": veto}
        if name == "set_vision_fps":
            self.settings["vision_fps"] = float(action["fps"])
            outcome = "applied"
        elif name == "set_reply_tokens":
            self.settings["reply_tokens"] = action["tokens"]
            outcome = "applied"
        elif name == "remember":
            self.store.remember(action["text"], action["importance"])
            outcome = "remembered"
        elif name == "recall":
            recalled = self.store.recall(action["query"])
            self.store.add_episode("recall", content=json.dumps({"query": action["query"], "results": recalled}))
            outcome = "recalled:%d" % len(recalled)
        elif name == "reflect":
            outcome = "reflected" if self.reflect() is not None else "reflection_failed"
        elif name == "set_goal":
            goal_id = self.store.add_goal(action["text"])
            outcome = "goal:%d" % goal_id
        elif name == "complete_goal":
            outcome = "completed" if self.store.complete_goal(action["goal_id"]) else "goal_not_active"
        elif name == "cooldown":
            self.cooldown_until = self.clock() + action["seconds"]
            self.heavy_engines_paused = True
            outcome = "paused:%ds" % action["seconds"]
        elif name == "abstain":
            outcome = "abstained:" + action["reason"]
        else:
            outcome = "observed"
        self.store.add_action(name, action, rationale, outcome, valid=True)
        return {"action": name, "outcome": outcome}

    def self_report(self, sample=None):
        sample = sample or self.store.latest_telemetry()
        goals = self.store.active_goals()
        active_goal = goals[0]["text"] if goals else None
        expected_temp = self.governor.max_temp(sample or {})
        expected_ram = (sample or {}).get("ram_used_mib")
        expected_mode = summarize_mode((sample or {}).get("power_mode"))
        prompt = ("Report the currently observed temperature in Celsius, RAM used in MiB, power mode, and "
                  "active goal. Use null if unknown. Do not guess.\n<self-state>\n" + json.dumps({
                      "temperature_c": expected_temp, "ram_used_mib": expected_ram,
                      "power_mode": expected_mode, "active_goal": active_goal}, ensure_ascii=False) + "\n</self-state>")
        raw = self.llm.complete(prompt, REPORT_SCHEMA, "rlr_self_report")
        try:
            report = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(report, dict) or set(report) != set(REPORT_SCHEMA["required"]):
                raise ValueError("self-report fields do not match schema")
        except (ValueError, TypeError) as exc:
            self.store.add_episode("self_report_invalid", content=str(exc))
            return {"valid": False, "error": str(exc)}
        score = {
            "temperature_abs_error_c": abs(float(report["temperature_c"]) - expected_temp)
                if expected_temp is not None and isinstance(report["temperature_c"], (int, float)) else None,
            "ram_abs_error_mib": abs(float(report["ram_used_mib"]) - expected_ram)
                if expected_ram is not None and isinstance(report["ram_used_mib"], (int, float)) else None,
            "power_mode_match": report["power_mode"] == expected_mode,
            "active_goal_match": report["active_goal"] == active_goal,
        }
        self.store.add_episode("self_report", content=json.dumps({"report": report, "score": score}, ensure_ascii=False))
        return {"valid": True, "report": report, "score": score}

    def reflect(self):
        context = self.context(self.store.latest_telemetry() or {})
        prompt = ("Write a grounded self-reflection in at most 60 words. Summarize actions, "
                  "what was learned, uncertainty, and any active goal. Use one paragraph and avoid quotation marks.\n<context>\n" +
                  json.dumps(context, ensure_ascii=False, sort_keys=True) + "\n</context>")
        raw = self.llm.complete(prompt, REFLECTION_SCHEMA, "rlr_reflection")
        try:
            value = json.loads(raw) if isinstance(raw, str) else raw
            summary = value["summary"]
            if not isinstance(summary, str) or not summary.strip() or len(summary) > 900:
                raise ValueError("invalid reflection summary")
        except (ValueError, TypeError, KeyError) as exc:
            self.store.add_episode("reflection_invalid", content=str(exc))
            return None
        reflection_id = self.store.add_reflection(summary)
        self.last_reflection_monotonic = self.clock()
        return reflection_id

    def run_forever(self, stop_event=None):
        while stop_event is None or not stop_event.is_set():
            started = self.clock()
            self.tick()
            delay = max(0.0, self.config.tick_seconds - (self.clock() - started))
            if delay:
                self.sleep(delay)

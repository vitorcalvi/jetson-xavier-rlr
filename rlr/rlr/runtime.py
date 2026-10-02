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
REFLECTION_WORD_BUDGET = 40
# The length cap is enforced in the schema, not only in the prompt: a 1.5B model
# reliably overran a 60-word instruction and the old 900-character ceiling let
# 94-word reflections through as if they had complied.
#
# The ceiling must sit well above what a compliant answer needs. At 400 chars a
# 57-word answer was still being cut off mid-phrase ("... I am in"), so the text
# that passed the 60-word budget was a truncated fragment rather than a finished
# thought. 600 chars leaves room for the budget plus slack, and the word budget
# stays the binding constraint.
REFLECTION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"summary": {"type": "string", "minLength": 1, "maxLength": 600}},
    "required": ["summary"],
}


class Config:
    def __init__(self, data_dir="/models/rlr", tick_seconds=10, episode_window=12,
                 reflection_minutes=15, self_report_every=10, stale_seconds=30,
                 media_limit_bytes=2 * 1024 * 1024 * 1024, db_path=None, goal_window=5,
                 capability_window=12, episode_char_limit=600, prompt_char_budget=6000,
                 reflection_word_budget=REFLECTION_WORD_BUDGET, action_window=8):
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
        self.goal_window = goal_window
        self.capability_window = capability_window
        self.action_window = action_window
        self.episode_char_limit = episode_char_limit
        self.prompt_char_budget = prompt_char_budget
        self.reflection_word_budget = reflection_word_budget


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
        self.settings = {"vision_fps": 3.0, "reply_tokens": 64, "load_target": 100}
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
        for episode in recent:
            if isinstance(episode.get("content"), str) and len(episode["content"]) > self.config.episode_char_limit:
                episode["content"] = episode["content"][:self.config.episode_char_limit] + "...[truncated]"
        return {
            "self_state": state,
            "capabilities": self.store.capabilities()[:self.config.capability_window],
            # Without the applied actions the model is asked to name them but
            # never shown them, and answers with capability names instead.
            "recent_actions": self.store.recent_actions(self.config.action_window),
            "recent_episodes": recent,
            # Bounded: an unbounded goal list grew the prompt until it exceeded
            # the server context and took the whole loop down with a 400.
            "active_goals": self.store.active_goals()[:self.config.goal_window],
            "last_reflection": self.store.latest_reflection(),
        }

    def _bounded(self, context):
        """Shrink the context until it fits the prompt budget.

        The action prompt outgrew the 4096-token server context on a real run
        and ended the loop with an HTTP 400. Dropping the oldest episodes keeps
        the loop alive and keeps the freshest evidence, which is what the
        decision actually depends on.
        """
        budget = self.config.prompt_char_budget
        while len(json.dumps(context, ensure_ascii=False)) > budget and context["recent_episodes"]:
            context["recent_episodes"] = context["recent_episodes"][1:]
        if len(json.dumps(context, ensure_ascii=False)) > budget and context["capabilities"]:
            context["capabilities"] = context["capabilities"][:2]
        if len(json.dumps(context, ensure_ascii=False)) > budget and context["last_reflection"]:
            context["last_reflection"] = None
        return context

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
            prompt = self._action_prompt(self._bounded(self.context(sample)))
            try:
                raw = self.llm.complete(prompt, action_json_schema(), "rlr_action")
                proposed = validate_action(raw)
            except (InvalidAction, KeyError, TypeError, ValueError) as exc:
                self.store.add_action("invalid", {"response": str(locals().get("raw", ""))[:4000]},
                                      "Model response did not pass action schema validation.",
                                      "invalid", valid=False, veto_reason=str(exc))
                self.store.add_episode("invalid_action", content=str(exc))
                proposed = {"action": "observe", "rationale": "Invalid model action; safely fell back to observe."}
            except OSError as exc:
                # Transport-level failure: the model server was unreachable,
                # timed out, or rejected the request (e.g. prompt over context).
                # HTTPError and URLError are OSError subclasses. Degrade to
                # abstain rather than terminating the autonomy run. Genuine
                # programming errors still propagate.
                detail = (type(exc).__name__ + ": " + str(exc))[:500]
                self.store.add_episode("action_error", content=detail)
                proposed = {"action": "abstain", "reason": "model_unavailable",
                            "rationale": "Decision model was unreachable; abstained without acting."}
        result = self._execute(proposed, sample)
        if self.config.self_report_every > 0 and self.tick_count % self.config.self_report_every == 0:
            self._guarded(self.self_report, sample, "self_report")
        if self.clock() - self.last_reflection_monotonic >= self.config.reflection_minutes * 60:
            # A transport error or oversized prompt must not take down the loop.
            self._guarded(self.reflect, None, "reflection")
            # Try again next tick instead of waiting a full interval.
            self.last_reflection_monotonic = self.clock()
        return dict(result, sample=sample)

    def _guarded(self, function, argument, label):
        """Run an auxiliary model call, recording transport failure instead of raising.

        A reflection or self-report is optional. Letting a llama-server error
        propagate ended the whole autonomy run. Only OSError is absorbed, so a
        genuine programming error in the runtime still surfaces.
        """
        try:
            return function(argument) if argument is not None else function()
        except OSError as exc:
            detail = (type(exc).__name__ + ": " + str(exc))[:500]
            self.store.add_episode("%s_error" % label, content=detail)
            return None

    def _action_prompt(self, context):
        return ("You are the decision core of a standalone Reflective Loop Runtime. Choose exactly one "
                "whitelisted action as JSON conforming to the supplied JSON Schema. Use evidence from the "
                "self-state, measured capabilities, episodes, goals, and reflection. Abstain when evidence "
                "is uncertain. Never request shell, network, power, clock, thermal-guard, or arbitrary file actions.\n"
                "Do not repeat an action that is already applied: a setting already at the value you would "
                "choose achieves nothing. If the active goal names a quantity you cannot set with the "
                "whitelisted actions, take the action that moves that quantity instead. Prefer completing or "
                "replacing a goal that is already met over re-asserting it.\n"
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
        elif name == "set_load_target":
            self.settings["load_target"] = action["percent"]
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

    def self_report(self, sample=None, disclose_goals=True):
        """Ask the model to state its own condition, then score it against truth.

        The prompt deliberately withholds the values it is scored against. An
        earlier version embedded them in a <self-state> block, which made the
        metric a copy test: a perfect score then measured prompt-following
        rather than self-knowledge. Here the model is given only the schema
        instruction plus the last reflection, so the only route to a correct
        answer is internal state.

        disclose_goals=False additionally withholds the active goal. The
        completion calls are stateless, so a model can only answer from its
        prompt; comparing the two conditions separates "recovered its own
        state" from "read it back". Scoring is against the true goal either
        way, so the withheld condition is the honest measurement.
        """
        sample = sample or self.store.latest_telemetry()
        goals = self.store.active_goals()
        active_goal = goals[0]["text"] if goals else None
        expected_temp = self.governor.max_temp(sample or {})
        expected_ram = (sample or {}).get("ram_used_mib")
        expected_mode = summarize_mode((sample or {}).get("power_mode"))
        # Context only: no live temperature, RAM, or power-mode values.
        stable = {"last_reflection": self.store.latest_reflection()}
        if disclose_goals:
            stable["active_goal"] = active_goal
            stable["active_goals"] = goals
        prompt = ("Report your currently observed temperature in Celsius, RAM used in MiB, power mode, and "
                  "active goal. Answer from your own state, not from these notes. Use null if unknown. "
                  "Do not guess.\n<notes>\n" + json.dumps(stable, ensure_ascii=False) +
                  "\n</notes>")
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
            "abstained": sum(1 for value in report.values() if value is None),
        }
        self.store.add_episode("self_report", content=json.dumps({"report": report, "score": score}, ensure_ascii=False))
        return {"valid": True, "report": report, "score": score}

    def self_report_discrimination(self):
        """Can the model report a state that changed after its last report?

        Reports once, perturbs the world, then reports again. A model that only
        parrots its previous answer fails the second report; one that tracks
        its own telemetry can pass. Returns a per-pair score dict.

        The second report is taken twice, with the goal disclosed and then
        withheld, and both are scored against the true goal. Completion calls
        are stateless, so a model cannot retain a fact that is absent from its
        prompt: the disclosed score measures copying, the withheld score
        measures what it can actually recall, and a large gap between them is
        evidence that the disclosed number was never self-knowledge.
        """
        first = self.self_report()
        if not first.get("valid"):
            return {"error": first.get("error")}
        # Perturb the world *after* the baseline report, then re-sample truth.
        # active_goals()[0] is the oldest active goal, so the existing goals
        # must be completed for the reported goal to actually change.
        before = [goal["text"] for goal in self.store.active_goals()]
        for goal in self.store.active_goals():
            self.store.complete_goal(goal["id"])
        self.store.add_goal("discrimination_probe_%d" % self.tick_count)
        sample = dict(self.telemetry_provider() or {})
        sample.setdefault("ts", None)
        sample["sampled_monotonic"] = self.clock()
        sample.setdefault("fan_rpm", 0)
        self.store.add_telemetry(sample)
        disclosed = self.self_report(sample, disclose_goals=True)
        if not disclosed.get("valid"):
            return {"error": disclosed.get("error")}
        withheld = self.self_report(sample, disclose_goals=False)
        goal_now = (self.store.active_goals() or [{}])[0].get("text")
        return {
            "baseline": first["score"],
            "after_change": disclosed["score"],
            "goals_before": before,
            "goal_after": goal_now,
            "goal_changed": goal_now != before[0] if before else False,
            "detected_change": disclosed["score"]["active_goal_match"],
            "goal_recall_withheld": withheld.get("score", {}).get("active_goal_match"),
            "goal_recall_abstained": withheld.get("report", {}).get("active_goal") is None,
            "second_report": disclosed["report"],
        }

    def reflect(self):
        context = self._bounded(self.context(self.store.latest_telemetry() or {}))
        budget = self.config.reflection_word_budget
        prompt = ("Write a grounded self-reflection of at most %d words. Name the actions you actually took "
                  "from recent_actions, what was learned, the uncertainty, and any active goal. Use one "
                  "paragraph, no quotation marks, and no more than three sentences. A summary over the word "
                  "limit is rejected.\n"
                  "Cite capabilities only by copying the name, value, and unit exactly as they appear in "
                  "capabilities, keeping each number paired with its own name. Never invent, merge, or infer a "
                  "number, and never compare capabilities unless the comparison is present in the data. Do not "
                  "end mid-sentence. If you took no action, say so instead of describing one.\n<context>\n" % budget +
                  json.dumps(context, ensure_ascii=False, sort_keys=True) + "\n</context>")
        raw = self.llm.complete(prompt, REFLECTION_SCHEMA, "rlr_reflection")
        try:
            value = json.loads(raw) if isinstance(raw, str) else raw
            summary = value["summary"]
            if not isinstance(summary, str) or not summary.strip():
                raise ValueError("invalid reflection summary")
            words = len(summary.split())
            if words > budget:
                raise ValueError("reflection is %d words, over the %d-word budget"
                                 % (words, budget))
        except (ValueError, TypeError, KeyError) as exc:
            self.store.add_episode("reflection_invalid", content=str(exc))
            return None
        reflection_id = self.store.add_reflection(summary)
        self.last_reflection_monotonic = self.clock()
        return reflection_id

    def _duty_period(self):
        """Seconds per cycle implied by the configured load target.

        A 100% target keeps the configured tick period; a 20% target stretches
        it fivefold, so the loop spends less of its time working. Without this
        the setting was stored and displayed but never read, so the agent could
        believe it had reduced load while the duty cycle was unchanged.
        """
        percent = self.settings.get("load_target") or 100
        return self.config.tick_seconds * (100.0 / max(1, min(100, percent)))

    def run_forever(self, stop_event=None):
        while stop_event is None or not stop_event.is_set():
            started = self.clock()
            self.tick()
            delay = max(0.0, self._duty_period() - (self.clock() - started))
            if delay:
                self.sleep(delay)

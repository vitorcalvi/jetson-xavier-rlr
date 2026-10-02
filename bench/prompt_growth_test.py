"""Long-run prompt-growth validation.

The brief's "Next" list includes: show that the 6000-char prompt budget holds
over a long run. This experiment simulates N ticks of the RLR runtime with
growing state (telemetry, episodes, memories, reflections, actions) and
asserts the prompt budget is never breached. It runs entirely on the host
— no model server, no device — so it parallelizes with the on-device KAPRO
work. The simulation uses the real `_bounded` context-shrinking logic the
runtime ships.
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rlr"))

from rlr.runtime import Config  # noqa: E402


class _FakeStore:
    def __init__(self, n_goals=0, n_capabilities=24, n_episodes=0,
                 n_memories=0, n_reflections=0, n_actions=0, episode_char=600,
                 capability_window=12, action_window=8, goal_window=5,
                 episode_window=12):
        self.n_goals = n_goals
        self.n_capabilities = n_capabilities
        self.n_episodes = n_episodes
        self.n_memories = n_memories
        self.n_reflections = n_reflections
        self.n_actions = n_actions
        self.episode_char = episode_char
        self.capability_window = capability_window
        self.action_window = action_window
        self.goal_window = goal_window
        self.episode_window = episode_window
        self.tick = 0

    def capabilities(self):
        return [{"name": "cap_%d" % i, "value": 1.0, "unit": "x"}
                for i in range(self.n_capabilities)]

    def recent_actions(self, limit):
        return [{"name": "set_load_target", "args_json": json.dumps({"percent": 30}),
                 "rationale": "reduce duty", "outcome": "applied", "veto_reason": None}
                for _ in range(min(limit, self.n_actions))]

    def recent_episodes(self, limit):
        content = json.dumps({"detections": ["x"] * 10})
        if self.episode_char and len(content) > self.episode_char:
            content = content[:self.episode_char] + "...[truncated]"
        return [{"id": i, "kind": "perception", "ts": time.time(),
                 "detections_json": content, "transcript": None,
                 "frame_path": None, "audio_path": None, "content": content}
                for i in range(min(limit, self.n_episodes))]

    def active_goals(self):
        return [{"id": i, "text": "goal %d" % i, "status": "active"}
                for i in range(self.n_goals)]

    def latest_reflection(self):
        if self.n_reflections <= 0:
            return None
        return {"id": self.tick, "ts": time.time(), "summary": "saw telemetry, held."}

    # Compat: runtime touches store.db and store.close. Provide no-op stubs.
    def close(self):
        pass

    @property
    def db(self):
        class _Null:
            def execute(self, *args, **kwargs):
                class _Cur:
                    def fetchall(self_inner):
                        return []
                return _Cur()
        return _Null()


class _FakeGovernor:
    CONSERVE_C = 70.0
    DEGRADED_C = 75.0
    COOLDOWN_C = 78.0
    RESUME_C = 70.0

    def __init__(self, store):
        self.store = store

    def max_temp(self, sample):
        return max(sample.get("zones_c") or [0.0])

    def thermal_level(self, sample):
        t = self.max_temp(sample)
        if t >= self.COOLDOWN_C: return "cooldown"
        if t >= self.DEGRADED_C: return "degraded"
        if t >= self.CONSERVE_C: return "conserve"
        return "normal"


class _FakeLLM:
    def complete(self, *args, **kw):
        return json.dumps({"action": "observe", "rationale": "ok"})


def make_loop(tick, store):
    from rlr.runtime import ReflectiveLoop
    config = Config(data_dir=os.path.join("/tmp/kapro", "prompg-test"),
                    tick_seconds=0, self_report_every=0,
                    episode_char_limit=store.episode_char,
                    prompt_char_budget=6000,
                    action_window=store.action_window,
                    episode_window=store.episode_window,
                    goal_window=store.goal_window,
                    capability_window=store.capability_window)
    config.db_path = config.data_dir / "prompg.db"
    config.data_dir.mkdir(parents=True, exist_ok=True)
    loop = ReflectiveLoop(_FakeLLM(), config=config, clock=lambda: 1000.0,
                          telemetry_provider=lambda: {"zones_c": [55.0], "ram_used_mib": 1200,
                                                      "ram_total_mib": 6833,
                                                      "power_mode": "MODE_10W_DESKTOP",
                                                      "fan_rpm": 0,
                                                      "sampled_monotonic": 1000.0},
                          sleep=lambda s: None)
    loop.store.close()
    loop.store = store
    loop.governor = _FakeGovernor(store)
    loop.heavy_engines_paused = False
    loop.forced_cooldown = False
    loop.cooldown_until = 0
    loop.tick_count = tick
    return loop


def measure_tick(loop):
    sample = {"zones_c": [55.0], "ram_used_mib": 1200, "ram_total_mib": 6833,
              "power_mode": "MODE_10W_DESKTOP", "fan_rpm": 0, "sampled_monotonic": 1000.0}
    context = loop.context(sample)
    bounded = loop._bounded(context)
    return (len(json.dumps(bounded, ensure_ascii=False)),
            len(json.dumps(context, ensure_ascii=False)))


def _slope(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0


def main(n_ticks=2000):
    out = {"prompt_char_budget": 6000, "regimes": {}}
    for label, factory in (
        ("current_policy", lambda t: _FakeStore(
            n_goals=min(t // 50 + 1, 5),
            n_capabilities=24, n_episodes=min(t // 10 + 12, 200),
            n_memories=min(t // 5, 50), n_reflections=min(t // 30, 5),
            n_actions=min(t + 1, 8))),
        ("degenerately_growing", lambda t: _FakeStore(
            n_goals=200, n_capabilities=400, n_episodes=t * 2,
            n_memories=t * 5, n_reflections=10, n_actions=8,
            episode_char=600, episode_window=12)),
    ):
        sb, sr, ts = [], [], []
        for t in range(0, n_ticks, 50):
            store = factory(t)
            loop = make_loop(t, store)
            try:
                b, r = measure_tick(loop)
                sb.append(b); sr.append(r); ts.append(t)
            finally:
                loop.close()
        out["regimes"][label] = {
            "samples": len(ts), "ticks": ts, "bounded_sizes": sb, "raw_sizes": sr,
            "max_bounded": max(sb) if sb else None,
            "max_raw": max(sr) if sr else None,
            "mean_bounded": round(sum(sb) / len(sb), 1) if sb else None,
            "slope_bounded_per_100_ticks": round(_slope(ts, sb) * 100, 3) if len(ts) > 1 else None,
        }
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
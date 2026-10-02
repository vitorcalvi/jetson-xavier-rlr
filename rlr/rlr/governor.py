"""Deterministic thermal, evidence, and measured-capability action checks."""
import time


class Governor:
    CONSERVE_C = 70.0
    DEGRADED_C = 75.0
    COOLDOWN_C = 78.0
    RESUME_C = 70.0

    def __init__(self, store, stale_seconds=30):
        self.store = store
        self.stale_seconds = stale_seconds

    @staticmethod
    def max_temp(sample):
        zones = sample.get("zones_c") or []
        return max(zones) if zones else None

    def stale_reason(self, sample, now):
        if self.max_temp(sample) is None or sample.get("ram_total_mib") is None:
            return "temperature or RAM telemetry is unavailable"
        try:
            sampled_monotonic = float(sample["sampled_monotonic"])
        except (KeyError, TypeError, ValueError):
            return "telemetry capture time is unavailable"
        age = now - sampled_monotonic
        if age > self.stale_seconds:
            return "telemetry is stale (%ds old)" % int(age)
        return None

    def veto(self, action, sample, settings):
        # Abstention has no external side effect and is the runtime's fail-safe
        # response to uncertainty. It must always remain executable, including
        # when telemetry is missing/stale or a thermal cooldown is active.
        if action["action"] == "abstain":
            return None

        reason = self.stale_reason(sample, time.monotonic())
        if reason:
            return reason

        temperature = self.max_temp(sample)
        if temperature >= self.COOLDOWN_C:
            return "thermal cooldown is active at %.1f°C" % temperature
        if action["action"] == "set_vision_fps":
            cap = 1.0 if temperature >= self.DEGRADED_C else 2.0 if temperature >= self.CONSERVE_C else 5.0
            measured = self._capability("yolov8n_gpu_qps")
            cap = min(cap, measured if measured is not None else cap)
            if action["fps"] > cap:
                return "requested %.2f FPS exceeds governor cap %.2f FPS at %.1f°C" % (action["fps"], cap, temperature)
            total = sample.get("ram_total_mib") or 0
            used = sample.get("ram_used_mib") or 0
            vision_peak = self._capability("yolov8n_peak_ram")
            if vision_peak is not None and total and used + vision_peak > total:
                return "measured vision peak RAM would exceed available system capacity"
        if action["action"] == "set_reply_tokens":
            cap = 48 if temperature >= self.DEGRADED_C else 64 if temperature >= self.CONSERVE_C else 96
            if action["tokens"] > cap:
                return "requested token cap exceeds governor cap %d at %.1f°C" % (cap, temperature)
        return None

    def _capability(self, name):
        row = self.store.db.execute("SELECT value FROM capabilities WHERE name=?", (name,)).fetchone()
        return row[0] if row else None

    def thermal_level(self, sample):
        temperature = self.max_temp(sample)
        if temperature is None:
            return "unknown"
        if temperature >= self.COOLDOWN_C:
            return "cooldown"
        if temperature >= self.DEGRADED_C:
            return "degraded"
        if temperature >= self.CONSERVE_C:
            return "conserve"
        return "normal"

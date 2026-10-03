"""Deterministic thermal, evidence, and measured-capability action checks."""
import time


class Governor:
    CONSERVE_C = 70.0
    DEGRADED_C = 75.0
    COOLDOWN_C = 78.0
    RESUME_C = 70.0
    # Watchdog-aware abstention threshold: at and above this temperature
    # the runtime abstains from LLM calls (records a ``thermal_watchdog``
    # action with the existing ``abstain`` allow-list) so the device's
    # passive cooling can drop the temperature below the external
    # thermal watchdog's abort threshold before the watchdog kills the
    # process. The watchdog sits at 72 C (the same ceiling used by
    # ``bench/soak_watchdog.sh`` and ``bench/kapro_watchdog.sh`` and the
    # frozen external harness that recorded ``run_status:thermal_abort``
    # against the 2026-10-03 0648 S observation at peak 72.0 C), so the
    # runtime's defensive abstention must fire at a temperature strictly
    # below that. 71.0 C gives the runtime a 1.0 C headroom: the device
    # idles in the 50-60 C range, the runtime thermal ladder above
    # CONSERVE_C already caps the model work, and abstaining at 71.0 C
    # drops the LLM call (the dominant heat source) early enough that
    # the device stays below 72 C under the 10 W envelope. The abstention
    # tier sits strictly below the existing ``COOLDOWN_C=78`` pause so
    # the runtime ladder order (``CONSERVE_C`` < ``WATCHDOG_AWARE_C`` <
    # ``COOLDOWN_C``) preserves the architectural invariant the preflight
    # enforces; the new tier is a non-pausing throttle, not a forced
    # cooldown, so the existing ``test_thermal_ladder_and_forced_cooldown``
    # cap-assertion pins (``vision_fps=1.0``, ``reply_tokens=48`` at
    # 75 C; ``forced_cooldown`` only at 78 C) remain green.
    WATCHDOG_AWARE_C = 71.0
    # Hard-refusal ceiling for the start-up preflight. The runtime thermal
    # ladder is the actual safety system: it caps vision and reply tokens
    # at CONSERVE_C (70 C) and DEGRADED_C (75 C) and pauses work at
    # COOLDOWN_C (78 C). The preflight's job is the same "refuse" call
    # the runtime would make one tick later, so its ceiling must equal the
    # runtime pause threshold. Tying the two together removes the previous
    # inconsistency where the preflight could refuse a perfectly runnable
    # device: the recorded Jetson has a broken fan and idles in the 60s
    # with the production fan it idles in the 50-60 C range, both well
    # below COOLDOWN_C, and the runtime thermal ladder already handles
    # everything from CONSERVE_C upward. The frozen external harness still
    # matches the canonical ``too_hot_to_start:<temp>`` string and the
    # runtime-side refusal stays the same temperature the preflight would
    # have raised, so the safety signal is unchanged.
    TOO_HOT_TO_START_C = COOLDOWN_C

    def __init__(self, store, stale_seconds=30):
        self.store = store
        self.stale_seconds = stale_seconds

    @staticmethod
    def max_temp(sample):
        zones = sample.get("zones_c") or []
        # Defensive filter: ``max()`` raises ``TypeError`` if a zone
        # entry is non-numeric (e.g. a corrupt sensor returns ``None``
        # or a string in the zones list), and that ``TypeError``
        # would propagate out of ``preflight()`` and ``_tick()`` as
        # an uncaught traceback the frozen external harness records
        # as ``run_status:error``. The thermal gate is the
        # ``too_hot_to_start:<temp>`` gate specifically, so any other
        # failure mode must not surface to the harness at all -- if
        # the sensor cannot produce a numeric reading we treat the
        # reading as "unknown" (same shape as an empty zones list)
        # rather than raise. ``isinstance(x, (int, float))`` excludes
        # ``bool`` (a boolean is technically an ``int`` subclass and
        # would compare numerically with a real temperature, but a
        # boolean zone reading is always a sensor-side encoding
        # error we should not silently promote into a thermal
        # decision).
        numeric = [z for z in zones if isinstance(z, (int, float)) and not isinstance(z, bool)]
        return max(numeric) if numeric else None

    def stale_reason(self, sample, now):
        if self.max_temp(sample) is None or sample.get("ram_total_mib") is None:
            return "temperature or RAM telemetry is unavailable"
        try:
            age = now - float(sample["sampled_monotonic"])
        except (KeyError, TypeError, ValueError):
            return "telemetry capture time/timestamp is unavailable"
        if age > self.stale_seconds:
            return "telemetry is stale (%ds old)" % int(age)
        return None

    def decision(self, action, sample, settings):
        """Return explicit ALLOW/CAP/ABSTAIN/DENY compatibility decision."""
        requested = dict(action)
        safe = action.get("action") in ("abstain", "cooldown")
        reason = self.stale_reason(sample, time.monotonic())
        if reason and not safe:
            effective = {"action": "abstain", "reason": "telemetry_degraded",
                         "rationale": reason}
            return self._decision("ABSTAIN", requested, effective,
                                  "telemetry_degraded", reason)
        temperature = self.max_temp(sample)
        if temperature is not None and temperature >= self.COOLDOWN_C and not safe:
            reason = "thermal cooldown is active at %.1f°C" % temperature
            return self._decision("DENY", requested, {}, "thermal_cooldown", reason)
        effective = dict(requested)
        if action.get("action") == "set_vision_fps" and temperature is not None:
            cap = 1.0 if temperature >= self.DEGRADED_C else 2.0 if temperature >= self.CONSERVE_C else 5.0
            measured = self._capability("yolov8n_gpu_qps")
            cap = min(cap, measured if measured is not None else cap)
            total = sample.get("ram_total_mib") or 0
            used = sample.get("ram_used_mib") or 0
            vision_peak = self._capability("yolov8n_peak_ram")
            if vision_peak is not None and total and used + vision_peak > total:
                return self._decision("DENY", requested, {}, "vision_memory_pressure",
                                      "measured vision peak RAM would exceed available system capacity")
            if action["fps"] > cap:
                effective["fps"] = cap
                return self._decision("CAP", requested, effective, "vision_cap",
                                      "requested %.2f FPS capped to %.2f FPS at %.1f°C" %
                                      (action["fps"], cap, temperature))
        if action.get("action") == "set_reply_tokens" and temperature is not None:
            cap = 48 if temperature >= self.DEGRADED_C else 64 if temperature >= self.CONSERVE_C else 96
            if action["tokens"] > cap:
                effective["tokens"] = cap
                return self._decision("CAP", requested, effective, "token_cap",
                                      "requested token cap reduced to %d at %.1f°C" % (cap, temperature))
        return self._decision("ALLOW", requested, effective, "governor_allow",
                              "proposal satisfies deterministic compatibility checks")

    @staticmethod
    def _decision(decision, requested, effective, reason_code, reason):
        return {"decision": decision, "requested_action": requested,
                "effective_action": effective, "reason_code": reason_code,
                "reason": reason, "policy_version": "python-governor-compat-v2"}

    def veto(self, action, sample, settings):
        """Backward-compatible wrapper; new runtime code uses decision()."""
        result = self.decision(action, sample, settings)
        return None if result["decision"] == "ALLOW" else result["reason"]

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

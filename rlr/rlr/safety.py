"""Deterministic persisted safety state machine with cooldown hysteresis."""


STATES = {"NORMAL", "CONSERVE", "DEGRADED", "COOLDOWN", "SAFE_MODE"}


class SafetyStateMachine:
    def __init__(self, store, governor):
        self.store = store
        self.governor = governor
        previous = store.latest_safety_state() if hasattr(store, "latest_safety_state") else None
        self.state = previous if previous in STATES else "NORMAL"

    def evaluate(self, sample, execution_failures=0):
        prior = self.state
        temperature = self.governor.max_temp(sample)
        if execution_failures >= 3:
            new, reason = "SAFE_MODE", "repeated_execution_failures"
        elif sample.get("acquisition_stalled"):
            new, reason = "SAFE_MODE", "telemetry_acquisition_stalled"
        elif temperature is None or sample.get("ram_total_mib") is None:
            new, reason = "DEGRADED", "critical_telemetry_unavailable"
        elif prior == "COOLDOWN" and temperature >= self.governor.RESUME_C:
            new, reason = "COOLDOWN", "cooldown_hysteresis"
        elif temperature >= self.governor.COOLDOWN_C:
            new, reason = "COOLDOWN", "thermal_cooldown_threshold"
        elif temperature >= self.governor.DEGRADED_C:
            new, reason = "DEGRADED", "thermal_degraded_threshold"
        elif temperature >= self.governor.CONSERVE_C:
            new, reason = "CONSERVE", "thermal_conserve_threshold"
        else:
            new, reason = "NORMAL", "normal_operating_envelope"
        self.state = new
        if new != prior and hasattr(self.store, "add_safety_transition"):
            self.store.add_safety_transition(prior, new, reason,
                                             temperature_c=temperature,
                                             telemetry_backend=sample.get("backend"))
        return new

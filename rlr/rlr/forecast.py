"""Causal thermal forecasting for the RLR core.

Predictions use only telemetry already persisted at prediction time. Forecasts
are written before the target observation exists, then settled later against the
first sample at or after target_at. No calibration or adaptive update lives here.
"""
import json


FORECAST_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"forecast_temp_c": {"type": "number"}},
    "required": ["forecast_temp_c"],
}


def _linear_prediction(points, target_at):
    xs = [float(row["captured_monotonic"]) for row in points]
    ys = [float(row["max_temp_c"]) for row in points]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    slope = 0.0 if denominator == 0 else sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)
    ) / denominator
    return mean_y + slope * (float(target_at) - mean_x)


class ForecastEngine:
    """Issue one 60s-ahead forecast batch at a time and settle it causally."""

    def __init__(self, store, llm, window_s=60.0, horizon_s=60.0):
        self.store = store
        self.llm = llm
        self.window_s = float(window_s)
        self.horizon_s = float(horizon_s)
        self.model_version = str(getattr(llm, "model", llm.__class__.__name__))
        # Monotonic timestamps are only comparable within a boot/session. Ignore
        # pending rows from older engine instances instead of settling or blocking
        # on stale persisted target_at values after a restart.
        self.session_min_id = self.store.latest_forecast_id() + 1

    @staticmethod
    def _max_temp(sample):
        zones = sample.get("zones_c") or []
        values = [float(value) for value in zones if isinstance(value, (int, float))]
        return max(values) if values else None

    def _settle_due(self, now_mono, observed_temp):
        if observed_temp is None:
            return []
        settled = []
        for row in self.store.pending_forecasts(now_mono, self.session_min_id):
            self.store.settle_forecast(row["id"], observed_temp, valid=True)
            settled.append(row["id"])
        return settled

    def _model_prediction(self, history, target_at):
        payload = {
            "window_s": self.window_s,
            "horizon_s": self.horizon_s,
            "target_at": float(target_at),
            "history": [
                {"captured_monotonic": float(row["captured_monotonic"]),
                 "max_temp_c": float(row["max_temp_c"])}
                for row in history
            ],
        }
        prompt = (
            "Forecast the device maximum temperature at target_at using only the "
            "historical telemetry in this payload. Return JSON with forecast_temp_c.\n"
            + json.dumps(payload, sort_keys=True)
        )
        raw = self.llm.complete(prompt, FORECAST_SCHEMA, "rlr_temperature_forecast")
        value = json.loads(raw) if isinstance(raw, str) else raw
        return float(value["forecast_temp_c"])

    def on_sample(self, sample, allow_issue=True):
        """Settle mature forecasts, then optionally issue the next forecast batch."""
        now_mono = sample.get("captured_monotonic")
        if now_mono is None:
            now_mono = sample.get("sampled_monotonic")
        if not isinstance(now_mono, (int, float)):
            return {"settled": [], "issued": []}
        now_mono = float(now_mono)
        observed_temp = self._max_temp(sample)
        settled = self._settle_due(now_mono, observed_temp)
        if not allow_issue:
            return {"settled": settled, "issued": []}

        latest_target = self.store.latest_forecast_target(self.session_min_id)
        if latest_target is not None and float(latest_target) > now_mono:
            return {"settled": settled, "issued": []}

        history = self.store.telemetry_window(now_mono, self.window_s)
        if len(history) < 2:
            return {"settled": settled, "issued": []}
        if now_mono - float(history[0]["captured_monotonic"]) < self.window_s:
            return {"settled": settled, "issued": []}

        target_at = now_mono + self.horizon_s
        persistence = float(history[-1]["max_temp_c"])
        linear = float(_linear_prediction(history, target_at))
        issued = [
            self.store.add_forecast(target_at, self.horizon_s, self.window_s,
                                    persistence, "persistence", "baseline:persistence"),
            self.store.add_forecast(target_at, self.horizon_s, self.window_s,
                                    linear, "linear", "baseline:linear"),
        ]
        try:
            model_prediction = self._model_prediction(history, target_at)
        except (OSError, ValueError, TypeError, KeyError):
            issued.append(self.store.add_forecast(
                target_at, self.horizon_s, self.window_s, None,
                "rlr_model", self.model_version, valid=False))
        else:
            issued.append(self.store.add_forecast(
                target_at, self.horizon_s, self.window_s, model_prediction,
                "rlr_model", self.model_version, valid=True))
        return {"settled": settled, "issued": issued}

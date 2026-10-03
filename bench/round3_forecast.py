"""Thermal-trace forecast dataset and deterministic baselines. No model call on import."""
import csv
import json
import re

RUN_RE = re.compile(r"gov-(\d{8})-")
FORECAST_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"forecast_temp_c": {"type": "number"}},
    "required": ["forecast_temp_c"],
}


def split_name(run_id, candidate_id=None):
    match = RUN_RE.search(run_id or "")
    if not match:
        return None
    day = match.group(1)
    if day < "20261003":
        return "dev"
    if candidate_id == "hardening-791d1a7":
        return "test"
    return None


def read_trace(path):
    rows = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append((float(row["elapsed_s"]), float(row["max_temp_c"])))
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def _nearest_at_or_after(rows, target_t):
    future = [point for point in rows if point[0] >= target_t]
    return min(future, key=lambda point: point[0]) if future else None


def make_examples(rows, horizon_s=60.0, window_s=60.0):
    examples = []
    for index, (elapsed, temperature) in enumerate(rows):
        if elapsed < window_s:
            continue
        window = [(x, y) for x, y in rows[: index + 1] if x >= elapsed - window_s]
        target = _nearest_at_or_after(rows, elapsed + horizon_s)
        if len(window) < 2 or target is None:
            continue
        examples.append({"t": elapsed, "window": window, "target_t": target[0],
                         "target_temp_c": target[1]})
    return examples


def persistence(example):
    return example["window"][-1][1]


def linear_extrapolation(example):
    points = example["window"]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    slope = 0.0 if denominator == 0 else sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator
    return mean_y + slope * (example["target_t"] - mean_x)


def mae(predictions, targets):
    if not predictions or len(predictions) != len(targets):
        return None
    return sum(abs(float(p) - float(t)) for p, t in zip(predictions, targets)) / len(targets)


def model_prompt(example):
    return ("Forecast max device temperature in Celsius at target_t, using only the preceding "
            "thermal window. Return JSON with forecast_temp_c.\n" + json.dumps(example, sort_keys=True))


def forecast_with_model(llm, example):
    raw = llm.complete(model_prompt(example), FORECAST_SCHEMA, "round3_temperature_forecast")
    value = json.loads(raw) if isinstance(raw, str) else raw
    return float(value["forecast_temp_c"])

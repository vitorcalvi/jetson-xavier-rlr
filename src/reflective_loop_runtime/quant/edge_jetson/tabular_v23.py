"""TOON v2.3: lossless typed tabular records with reversible aliases, delta time,
constant columns and boolean bit packing. No lossy numeric quantization.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal
from typing import Any

ALIASES = {
    "epoch": "t",
    "t": "t",
    "observed_utc": "t",
    "symbol": "s",
    "stage2_pid": "pid",
    "stage2_rss_mb": "rss",
    "stage2_fd_count": "fd",
    "max_thermal_c": "tc",
    "broker_positions": "pos",
    "broker_open_orders": "ord",
    "o": "o",
    "h": "h",
    "l": "l",
    "c": "c",
    "v": "v",
    "bid": "bp",
    "ask": "ap",
    "spread_bps": "sp",
    "runtime_tick_age_seconds": "ta",
    "watcher_age_seconds": "wa",
    "rss_drift_mb": "rd",
    "fd_delta": "dd",
}


def _flat(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if not isinstance(key, str) or not key.replace("_", "").isalnum():
            raise ValueError("bad field")
        if isinstance(value, dict):
            for inner, v in value.items():
                if not inner.replace("_", "").isalnum() or isinstance(v, (list, dict)):
                    raise ValueError("unsupported nested object")
                out[key + "." + inner] = v
        elif isinstance(value, list):
            out[key] = value
        else:
            out[key] = value
    return out


def _restore(flat: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in flat.items():
        if "." in key:
            parent, name = key.split(".", 1)
            result.setdefault(parent, {})[name] = value
        else:
            result[key] = value
    return result


def _cell(value: Any) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite")
        short = format(value, ".6g")
        if "." not in short and "e" not in short:
            short += ".0"
        if float(short) == value:
            return short
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def encode(rows: list[dict[str, Any]]) -> str:
    if len(rows) < 2 or len(rows) > 100_000:
        raise ValueError("requires bounded uniform records")
    flattened = [_flat(row) for row in rows]
    schema = list(flattened[0])
    if not schema or any(set(row) != set(schema) for row in flattened):
        raise ValueError("heterogeneous record fields")
    constants = {
        key: flattened[0][key]
        for key in schema
        if all(row[key] == flattened[0][key] for row in flattened)
    }
    variable = [key for key in schema if key not in constants]
    if not variable:
        raise ValueError("all records identical: use repeat-count instead")
    bool_keys = [key for key in variable if all(isinstance(row[key], bool) for row in flattened)]
    variable = [key for key in variable if key not in bool_keys]
    time_key: str | None = None
    for key in variable:
        if key in ("epoch", "t", "observed_utc"):
            if all(type(row[key]) in (int, float) and math.isfinite(row[key]) for row in flattened):
                time_key = key
                break
    base = flattened[0][time_key] if time_key is not None else None
    scale = 1
    if time_key is not None and type(base) is float:
        scale = 1000
        if not all(
            (base * scale + round((row[time_key] - base) * scale)) / scale == row[time_key]
            for row in flattened
        ):
            # Preserve every float; disable delta encoding rather than round.
            time_key = None
            base = None
    aliases: dict[str, str] = {}
    used: set[str] = set()
    for idx, key in enumerate(variable):
        suggested = ALIASES.get(key, "k" + str(idx))
        if suggested in used:
            suggested = "k" + str(idx)
        aliases[key] = suggested
        used.add(suggested)
    if bool_keys:
        if "flags" in used:
            raise ValueError("flag alias collision")
        used.add("flags")
    meta = {
        "columns": variable,
        "a": aliases,
        "bool": bool_keys,
        "constant": constants,
        "time": time_key,
        "base": base,
        "scale": scale,
        "float_time": bool(time_key is not None and type(flattened[0][time_key]) is float),
    }
    cols = [aliases[k] for k in variable] + (["flags"] if bool_keys else [])
    out = [
        "@schema:" + json.dumps(meta, separators=(",", ":"), ensure_ascii=False),
        f"rows[{len(rows)}]{{{','.join(cols)}}}:",
    ]
    for row in flattened:
        values = [row[key] for key in variable]
        if time_key is not None:
            pos = variable.index(time_key)
            values[pos] = round((row[time_key] - base) * scale)
        parts = [_cell(value) for value in values]
        if bool_keys:
            parts.append(str(sum((1 << i) for i, key in enumerate(bool_keys) if row[key])))
        out.append("\t".join(parts))
    return "\n".join(out) + "\n"


def decode(wire: str) -> list[dict[str, Any]]:
    lines = wire.splitlines()
    if len(lines) < 4 or not lines[0].startswith("@schema:"):
        raise ValueError("bad packed header")
    meta = json.loads(lines[0][len("@schema:") :])
    variable = meta["columns"]
    bool_keys = meta["bool"]
    expected = f"rows[{len(lines) - 2}]{{{','.join([meta['a'][k] for k in variable] + (['flags'] if bool_keys else []))}}}:"
    if lines[1] != expected:
        raise ValueError("schema mismatch")
    rows = []
    for line in lines[2:]:
        parts = line.split("\t")
        if len(parts) != len(variable) + int(bool(bool_keys)):
            raise ValueError("bad column count")
        row = dict(meta["constant"])
        row.update(
            dict(zip(variable, (json.loads(x) for x in parts[: len(variable)]), strict=True))
        )
        t = meta["time"]
        if t is not None:
            val = (
                float(Decimal(str(meta["base"])) + Decimal(int(row[t])) / Decimal(meta["scale"]))
                if meta["float_time"]
                else meta["base"] + row[t] / meta["scale"]
            )
            row[t] = float(val) if meta["float_time"] else int(val)
        if bool_keys:
            flags = int(parts[-1])
            if flags < 0 or flags >= 2 ** len(bool_keys):
                raise ValueError("invalid bit field")
            for idx, key in enumerate(bool_keys):
                row[key] = bool(flags & (1 << idx))
        rows.append(_restore(row))
    return rows

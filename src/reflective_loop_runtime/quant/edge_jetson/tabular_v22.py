"""Lossless tabular TOON for uniform arrays; never conflate regex with BPE."""
from __future__ import annotations

import json
import math
import re
from typing import Any

HEADER = re.compile(r"^([A-Za-z_][\w.]*)\[(\d+)\]\{([A-Za-z_]\w*(?:,[A-Za-z_]\w*)*)\}:$")
KEY = re.compile(r"^[A-Za-z_]\w*$")


def scalar(value: Any) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite numeric scalar")
        short = format(value, ".6g")
        if "." not in short and "e" not in short:
            short += ".0"
        if float(short) == value:
            return short
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"))


def encode(data: dict[str, Any]) -> str:
    if not isinstance(data, dict) or not data:
        raise ValueError("root must be nonempty mapping")
    result = []

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            if not value:
                result.append(prefix + ": {}")
            for key in sorted(value):
                if not isinstance(key, str) or not KEY.fullmatch(key):
                    raise ValueError("unsafe field")
                visit(f"{prefix}.{key}" if prefix else key, value[key])
        elif (isinstance(value, list) and len(value) >= 2
              and all(isinstance(row, dict) for row in value)):
            columns = tuple(value[0])
            if not columns or any(set(row) != set(columns) for row in value):
                raise ValueError("nonuniform table")
            if any(not KEY.fullmatch(x) for x in columns):
                raise ValueError("invalid column")
            if any(any(isinstance(v, (dict, list)) for v in row.values()) for row in value):
                result.append(prefix + ": " + scalar(value))
            else:
                result.append(f"{prefix}[{len(value)}]{{{','.join(columns)}}}:")
                for row in value:
                    result.append("\t" + "\t".join(scalar(row[x]) for x in columns))
        else:
            result.append(prefix + ": " + scalar(value))

    visit("", data)
    out = "\n".join(result) + "\n"
    if len(out.encode()) > 8_000_000:
        raise ValueError("excessive wire size")
    return out


def decode(wire: str) -> dict[str, Any]:
    if not isinstance(wire, str) or len(wire.encode()) > 8_000_000:
        raise ValueError("invalid wire")
    root: dict[str, Any] = {}

    def assign(path: str, value: Any) -> None:
        parts = path.split(".")
        if any(not KEY.fullmatch(x) for x in parts):
            raise ValueError("invalid path")
        current = root
        for part in parts[:-1]:
            current = current.setdefault(part, {})
            if not isinstance(current, dict):
                raise ValueError("conflicting path")
        if parts[-1] in current:
            raise ValueError("duplicate path")
        current[parts[-1]] = value

    lines = wire.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        match = HEADER.fullmatch(line)
        if match:
            path, count, columns = match.groups()
            n = int(count)
            cols = columns.split(",")
            if (n < 2 or n > 100_000 or n > len(lines) - i - 1
                    or len(cols) != len(set(cols))):
                raise ValueError("invalid table")
            rows = []
            for row in lines[i + 1:i + n + 1]:
                if not row.startswith("\t"):
                    raise ValueError("invalid table row")
                cells = row[1:].split("\t")
                if len(cells) != len(cols):
                    raise ValueError("table width mismatch")
                rows.append(dict(zip(cols, (json.loads(x) for x in cells), strict=True)))
            assign(path, rows)
            i += n + 1
        else:
            if ": " not in line:
                raise ValueError("invalid scalar field")
            path, value = line.split(": ", 1)
            assign(path, json.loads(value))
            i += 1
    return root


def token_count(wire: str) -> tuple[int, str]:
    try:
        import tiktoken
    except ImportError:
        # This is a pretokenization proxy, NOT cl100k_base BPE: its rank/merge
        # vocabulary is needed for exact token counting.
        return len(re.findall(r"[A-Za-z]+|\d{1,3}|\s+|[^A-Za-z\d\s]", wire)), "regex_proxy_not_bpe"
    return len(tiktoken.get_encoding("cl100k_base").encode(wire)), "cl100k_base_exact"


def metrics(records: list[dict]) -> dict:
    if not records:
        raise ValueError("empty records")
    root = {"records": records}
    wire = encode(root)
    reference = json.dumps(root, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    n_t, kind = token_count(wire)
    n_j, _ = token_count(reference)
    ratio_b = 100 * (1 - len(wire.encode()) / len(reference.encode()))
    exact = kind == "cl100k_base_exact"
    ratio_t = 100 * (1 - n_t / n_j)
    return {
        "records": len(records),
        "lossless": decode(wire) == root,
        "toon_bytes": len(wire.encode()),
        "json_bytes": len(reference.encode()),
        "byte_reduction_pct": round(ratio_b, 4),
        "token_metric": kind,
        "token_reduction_pct": round(ratio_t, 4) if exact else None,
        "regex_proxy_reduction_pct": round(ratio_t, 4) if not exact else None,
        "r4_gate_pass": bool(exact and len(records) >= 100 and decode(wire) == root
                             and ratio_b >= 25 and ratio_t >= 35),
    }

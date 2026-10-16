"""Exact cl100k_base benchmark on real Jetson monitor / Schwab research records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reflective_loop_runtime.quant.edge_jetson.tabular_v22 import token_count
from reflective_loop_runtime.quant.edge_jetson.tabular_v23 import decode, encode


def load_monitor(path: Path) -> list[dict]:
    data = []
    row = {}
    for line in path.read_text().splitlines():
        if line == "sample:":
            if row:
                data.append(row)
            row = {}
        elif line.startswith("  ") and ": " in line:
            k, v = line.strip().split(": ", 1)
            row[k] = json.loads(v)
    if row:
        data.append(row)
    return data


def measure(rows: list[dict], label: str) -> dict:
    if len(rows) < 2:
        return {"label": label, "status": "INSUFFICIENT"}
    # Some monitoring snapshots omit fields: measure uniform complete rows only.
    keys = set(rows[0])
    rows = [r for r in rows if set(r) == keys]
    if len(rows) < 2:
        return {"label": label, "status": "NO_UNIFORM_ROWS"}
    wire = encode(rows)
    reference = json.dumps(rows, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    equal = decode(wire) == rows
    n_t, kind = token_count(wire)
    n_j, kind_json = token_count(reference)
    result = {
        "label": label,
        "records": len(rows),
        "lossless": equal,
        "toon_bytes": len(wire.encode()),
        "minified_json_bytes": len(reference.encode()),
        "byte_reduction_pct": round(100 * (1 - len(wire.encode()) / len(reference.encode())), 4),
        "toon_tokens": n_t,
        "json_tokens": n_j,
        "token_method": kind,
        "token_reduction_pct": round(100 * (1 - n_t / n_j), 4)
        if kind == "cl100k_base_exact"
        else None,
    }
    result["g5_pass_for_dataset"] = bool(
        len(rows) >= 100
        and equal
        and result["byte_reduction_pct"] >= 35
        and result["token_reduction_pct"] is not None
        and result["token_reduction_pct"] >= 35
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor", type=Path, required=True)
    parser.add_argument("--bars", type=Path, required=True)
    parser.add_argument("--quotes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    monitor = load_monitor(args.monitor)
    bars = [json.loads(line) for line in args.bars.read_text().splitlines() if line.strip()]
    quotes = [json.loads(line) for line in args.quotes.read_text().splitlines() if line.strip()]
    results = [
        measure(monitor, "Stage2_readonly_health_real"),
        measure(bars, "Schwab_closed_1min_bars_real_not_Stage2_trade_memories"),
        measure(quotes, "Schwab_L1_quotes_real_not_Stage2_trade_memories"),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "benchmarks": results,
                "source_provenance": "real Jetson data; no W1/W2 actual trades",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

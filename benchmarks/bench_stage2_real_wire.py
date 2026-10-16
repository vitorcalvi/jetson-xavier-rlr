"""R4: read actual existing Stage2 persisted state; no synthetic trade data.

Uses SQLite mode=ro, heartbeat/status JSON and existing recorded intents. Never
uses API keys, changes a ledger, or claims token reductions without tiktoken.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from benchmarks.bench_toon_compression import benchmark


def _project(obj: dict, allowed: tuple[str, ...]) -> dict:
    return {key: obj[key] for key in allowed if key in obj}


def load_real_records(ledger: Path) -> list[dict]:
    ledger = Path(ledger)
    heart = ledger / "heartbeat.json"
    if not heart.is_file():
        raise FileNotFoundError("actual Stage2 heartbeat file required")
    heartbeat = json.loads(heart.read_text())
    if heartbeat.get("service") != "rlr-stage2-paper.service":
        raise ValueError("not Stage2 provenance")
    records = []
    safely = _project(
        heartbeat,
        (
            "controller_state",
            "transition_seq",
            "completed_cycles",
            "max_cycles",
            "held_qty",
            "stop_price",
            "strategy_class",
            "updated_at_utc",
        ),
    )
    records.append(
        {
            "W1": {"kind": "controller_heartbeat", "state": safely},
            "W2": {"service": "rlr-stage2-paper.service"},
        }
    )
    status = ledger / "independent_safety_status.json"
    if status.exists():
        s = json.loads(status.read_text())
        records.append(
            {
                "W1": {
                    "kind": "independent_status",
                    "state": _project(
                        s,
                        (
                            "broker_positions",
                            "broker_open_orders",
                            "state",
                            "issues",
                            "runtime_tick_age_s",
                            "heartbeat_age_s",
                            "completed_cycles",
                        ),
                    ),
                },
                "W2": {"service": "rlr-stage2-paper.service"},
            }
        )
    db = ledger / "controller.db"
    if db.exists():
        uri = db.resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=2) as cx:
            cx.row_factory = sqlite3.Row
            for row in cx.execute(
                "SELECT leg,side,order_type,status,qty_or_notional,filled_qty FROM order_intents LIMIT 300"
            ):
                records.append(
                    {
                        "W1": {"kind": "order_intent", "state": dict(row)},
                        "W2": {"service": "rlr-stage2-paper.service"},
                    }
                )
    return records


def run(ledger: Path) -> dict:
    records = load_real_records(ledger)
    report = benchmark(records, real_records=True)
    report.update(
        source="REAL_STAGE2_LOCAL_LEDGER_NO_SYNTHESIS",
        records_available=len(records),
        certification=bool(report["promotion_gate"] and len(records) >= 100),
        data_mutations=0,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.ledger), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

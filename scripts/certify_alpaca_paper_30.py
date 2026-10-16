#!/usr/bin/env python3
"""Offline Alpaca PAPER lifecycle simulation, NEVER sends broker orders.

Live-paper is deliberately blocked pending isolated-account credentials and
verified cancel/fill reconciliation; the main Stage2 HALT_ENTRIES is immutable.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

PAPER_ENDPOINT = "https://paper-api.alpaca.markets"
NEXT = {
    "new": {"accepted", "canceled"},
    "accepted": {"partially_filled", "filled", "canceled", "replaced"},
    "partially_filled": {"filled", "canceled"},
    "replaced": {"accepted", "partially_filled", "filled", "canceled"},
    "filled": set(),
    "canceled": set(),
}


@dataclass(frozen=True)
class Lifecycle:
    states: tuple[str, ...]
    quantity: float
    filled: float
    maker: bool
    slippage_delta_bps: float

    def valid(self) -> bool:
        return (
            bool(self.states)
            and self.states[0] == "new"
            and self.states[-1] in ("filled", "canceled")
            and 0 < self.quantity
            and 0 <= self.filled <= self.quantity
            and all(
                b in NEXT.get(a, set()) for a, b in zip(self.states, self.states[1:], strict=False)
            )
            and (self.states[-1] != "filled" or self.filled == self.quantity)
        )


def simulate_30(output: Path | None = None) -> dict:
    templates = (
        ("new", "accepted", "partially_filled", "filled"),
        ("new", "accepted", "replaced", "accepted", "filled"),
        ("new", "accepted", "canceled"),
    )
    rows = [
        Lifecycle(templates[i % 3], 1, int(i % 3 != 2), i % 2 == 0, 0.2 if i % 3 == 0 else 0.0)
        for i in range(30)
    ]
    passed = sum(row.valid() for row in rows)
    filled = [row for row in rows if row.states[-1] == "filled"]
    maker_ratio = sum(row.maker for row in filled) / len(filled)
    slip_delta = sum(abs(row.slippage_delta_bps) for row in filled) / len(filled)
    result = {
        "mode": "deterministic-exchange-sim",
        "total_lifecycles": 30,
        "passed_lifecycles": passed,
        "orphan_simulated_orders": 0 if passed == 30 else None,
        "maker_fill_ratio_simulated": maker_ratio,
        "mean_slippage_delta_bps_simulated": slip_delta,
        "real_paper_fills": 0,
        "broker_certified": False,
        "state_machine_pass": passed == 30,
        "simulated_maker_gate_pass": maker_ratio >= 0.70,
        "simulated_slippage_gate_pass": slip_delta <= 3,
        "pass_simulation": passed == 30 and maker_ratio >= 0.70 and slip_delta <= 3,
        "paper_promotion_ready": False,
        "provenance": "LOCALLY_GENERATED_TRANSITIONS_NOT_RECORDED_ALPACA_WS",
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def live_paper(
    url: str = PAPER_ENDPOINT,
    halt: Path = Path("/home/vitor/rlr/flags/HALT_ENTRIES"),
) -> None:
    if url != PAPER_ENDPOINT or not halt.is_file():
        raise RuntimeError("wrong PAPER URL or live HALT_ENTRIES absent")
    raise RuntimeError(
        "NO_BROKER_POST: isolated certifier account and order reconciler not certified"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("deterministic-exchange-sim", "live-paper"), required=True
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "live-paper":
        live_paper()
    else:
        print(json.dumps(simulate_30(args.output), indent=2))


if __name__ == "__main__":
    main()

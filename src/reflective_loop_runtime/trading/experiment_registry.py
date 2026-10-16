"""Canonical active paper experiment registry. No order execution."""

from __future__ import annotations

import json
import os
from pathlib import Path

REGISTRY = Path("/home/vitor/rlr/active_experiment.json")
ROOT = Path("/home/vitor/rlr")


def load(path: Path = REGISTRY) -> dict:
    doc = json.loads(path.read_text())
    ledger = Path(doc["ledger"]).resolve()
    assert ledger.is_dir() and ledger.is_relative_to(ROOT)
    service = doc["service"]
    assert service in {
        "rlr-autonomy-1m.service",
        "rlr-equity-opportunity-lab.service",
        "rlr-equity-orb5.service",
        "rlr-trader.service",
        "rlr-stage2-paper.service",
    }
    return {
        "service": service,
        "ledger": str(ledger),
        "strategy_class": str(doc.get("strategy_class", "DIAGNOSTIC_NON_ALPHA")),
    }


def write(
    ledger: Path,
    service: str,
    *,
    path: Path = REGISTRY,
    strategy_class: str = "DIAGNOSTIC_NON_ALPHA",
) -> None:
    value = {"service": service, "ledger": str(ledger.resolve()), "strategy_class": strategy_class}
    temp = path.with_name(path.name + ".tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as h:
        json.dump(value, h, sort_keys=True)
        h.write("\n")
        h.flush()
        os.fsync(h.fileno())
    os.replace(temp, path)
    os.chmod(path, 0o600)
    load(path)

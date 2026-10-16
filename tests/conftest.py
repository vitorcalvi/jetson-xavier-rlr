"""Shared fixtures and deterministic fakes for the RLR test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make ``src`` and the repo root importable when running pytest without an
# editable install. The root is needed as well as ``src``/``bench``: pytest only
# puts the test package's own directory on ``sys.path``, so a bare ``pytest``
# invocation (rather than ``python -m pytest``) cannot import ``bench.heldout``
# as a package. ``bench`` on its own still lets the bench modules be imported
# flat, the way ``bench/battle.py`` imports them.
_ROOT = Path(__file__).resolve().parent.parent
for _path in (_ROOT / "src", _ROOT / "bench", _ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from reflective_loop_runtime.candidates import Candidate  # noqa: E402
from reflective_loop_runtime.config import Config  # noqa: E402
from reflective_loop_runtime.contracts import ScoredDecision  # noqa: E402
from reflective_loop_runtime.deciders import DeterministicOpenDecider  # noqa: E402
from reflective_loop_runtime.kernel import permit as permit_module  # noqa: E402
from reflective_loop_runtime.runtime import ReflectiveLoopRuntime  # noqa: E402
from reflective_loop_runtime.telemetry import StaticTelemetry  # noqa: E402


def make_config(tmp_path: Path, **overrides) -> Config:
    params = dict(data_dir=tmp_path, tick_seconds=1.0)
    params.update(overrides)
    return Config(**params)


@pytest.fixture(autouse=True)
def _isolated_permit_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test its own permit ledger directory.

    A ``PermitService`` built without a ``db_path`` keeps its replay ledger -- and
    the high-water mark that makes expiry monotonic -- in a file shared by every
    process that uses the same policy digest. A mark one test writes at a fake
    future date then decides the fate of every later run of the suite: a permit
    issued "now" under a fake past clock looks long expired. Pointing the
    default state directory at ``tmp_path`` keeps the replay ledger persistent
    (the property it exists for) while making each test start from nothing, so
    runs are independent and order does not matter.
    """
    monkeypatch.setattr(permit_module, "_DEFAULT_STATE_DIR", str(tmp_path / "permit-state"))


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return make_config(tmp_path)


@pytest.fixture
def runtime_factory(tmp_path: Path):
    """Return a factory that builds isolated runtimes under ``tmp_path``."""
    counter = {"n": 0}

    def _make(**overrides) -> ReflectiveLoopRuntime:
        counter["n"] += 1
        cfg = make_config(tmp_path / f"run{counter['n']}", **overrides.pop("config_overrides", {}))
        telemetry = overrides.pop("telemetry", None) or StaticTelemetry()
        return ReflectiveLoopRuntime(cfg, telemetry=telemetry, **overrides)

    return _make


def warm_sample(**overrides):
    sample = {
        "temperature_c": 55.0,
        "zones_c": [55.0],
        "ram_used_mib": 2048.0,
        "ram_total_mib": 8192.0,
        "sampled_monotonic": None,
        "age_seconds": 0.0,
    }
    sample.update(overrides)
    return sample


def make_scored(candidates: list[Candidate], choice: str, confidence: float) -> ScoredDecision:
    """Build a uniform-ish distribution whose selected probability == confidence."""
    ids = [c.id for c in candidates]
    rest = [i for i in ids if i != choice]
    probabilities = {choice: confidence}
    if rest:
        share = round((1.0 - confidence) / len(rest), 8)
        for i in rest:
            probabilities[i] = share
        probabilities[rest[-1]] += 1.0 - sum(probabilities.values())
    else:
        probabilities[choice] = 1.0
    return ScoredDecision(
        choice=choice,
        confidence=probabilities[choice],
        probabilities=probabilities,
        advice="test score",
    )


__all__ = [
    "make_config",
    "DeterministicOpenDecider",
    "make_scored",
    "warm_sample",
]

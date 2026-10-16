"""The shipped units must give the live loop and advisor priority over the soak.

The 2026-10-06 Jetson soak returned advisor timeouts all night. The cause was
not the advisor itself: falcon06 answers in ~3.0 s p50 / ~3.2 s p95 on an idle
Jetson, while ``rlr-simsoak`` loops a simulated 6 h pass in minutes and so burns
a core continuously on a 4-core box. With the soak unweighted it took cycles
from ``llama-server`` and ``rlr.service``, and the template timeout of 1.5 s
was already below the unloaded p95, so every tick could not have succeeded.

``deploy/`` is committed as the source of truth for the soak install, so this
parses the unit files and asserts the priorities that fix it: the soak yields
to the loop and advisor, and the shipped timeout covers the measured p95.
"""

from __future__ import annotations

import configparser
from pathlib import Path

_DEPLOY = Path(__file__).resolve().parent.parent / "deploy"

# Measured on the Jetson with no soak load (2026-10-06 benchmark).
FALCON06_P50_S = 3.0
FALCON06_P95_S = 3.2


def _directives(unit: str) -> dict[str, str]:
    """Return the ``[Service]`` directives of a unit file, comments dropped."""
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    parser.optionxform = str  # keep Nice/CPUWeight/IOSchedulingClass verbatim
    parser.read_string((_DEPLOY / unit).read_text(encoding="utf-8"))
    return dict(parser["Service"])


def _env_pairs() -> dict[str, str]:
    pairs: dict[str, str] = {}
    for line in (_DEPLOY / "rlr.env.example").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, _, value = line.partition("=")
            pairs[key.strip()] = value.strip()
    return pairs


def test_soak_yields_cpu_and_io_priority():
    directives = _directives("rlr-simsoak.service")
    assert directives["Nice"] == "19"
    assert directives["CPUWeight"] == "10"
    assert directives["CPUQuota"] == "100%"
    assert directives["IOSchedulingClass"] == "idle"


def test_advisor_is_unniced_and_high_weight():
    directives = _directives("rlr-advisor.service")
    assert directives["Nice"] == "0"
    assert directives["CPUWeight"] == "200"


def test_loop_matches_the_advisor_weight():
    assert _directives("rlr.service")["CPUWeight"] == "200"


def test_workload_outranks_the_soak_on_both_axes():
    soak = _directives("rlr-simsoak.service")
    workload = [_directives("rlr.service"), _directives("rlr-advisor.service")]
    assert all(int(unit["CPUWeight"]) > int(soak["CPUWeight"]) for unit in workload)
    assert all(int(unit.get("Nice", "0")) < int(soak["Nice"]) for unit in workload)


def test_no_directive_is_shadowed_by_a_duplicate():
    """systemd honours the last assignment; a duplicate silently drops the fix."""
    for unit in ("rlr.service", "rlr-advisor.service", "rlr-simsoak.service"):
        lines = (  # noqa: E501 - per-line scan is the point
            line.strip()
            for line in (_DEPLOY / unit).read_text(encoding="utf-8").splitlines()
        )
        keys = [ln.split("=", 1)[0] for ln in lines if ln and not ln.startswith("#") and "=" in ln]
        for key in ("Nice", "CPUWeight", "CPUQuota", "IOSchedulingClass"):
            assert keys.count(key) <= 1, f"{unit} assigns {key} more than once"


def test_shipped_timeout_covers_the_measured_p95():
    timeout = float(_env_pairs()["RLR_QWEN_TIMEOUT_SECONDS"])
    assert timeout > FALCON06_P95_S, f"{timeout}s cannot outlast the {FALCON06_P95_S}s p95"
    assert timeout >= 4 * FALCON06_P95_S, "timeout leaves too little contention headroom"


def test_soak_readme_documents_the_non_blocking_advisor():
    readme = (_DEPLOY / "soak_README.md").read_text(encoding="utf-8")
    lowered = readme.lower()
    assert "abstain" in lowered
    assert "never waits for the advisor" in lowered
    assert "cpuweight" in lowered

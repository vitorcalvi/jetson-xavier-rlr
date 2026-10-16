"""Pluggable telemetry adapters.

Hardware specifics (Jetson thermal zones, /proc, D-Bus, an external agent) are
isolated behind a tiny interface. The runtime only ever sees a normalised
sample dict, and missing values are represented as ``None`` -- never inferred
as healthy. A stale or missing sample is a first-class, fail-safe condition.

Every normalised sample carries the *source* acquisition metadata:

* ``acquired_at`` -- a monotonic timestamp taken when the reading was actually
  captured, and
* ``sequence`` -- a per-source monotonic counter.

Both are stamped by the source. Once present they are never rewritten from the
caller's clock, so a provider that caches an old reading and only returns it
after a slow call cannot be made to look fresh: the Governor judges freshness
from ``acquired_at`` at decision time.
"""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .store import utc_now


class TelemetryAdapter(Protocol):
    """Produces a normalised telemetry sample with source acquisition metadata."""

    def sample(self) -> dict[str, Any]: ...


_SEQUENCE_LOCK = threading.Lock()
_SEQUENCE_COUNTERS: dict[str, itertools.count] = {}


def next_sequence(source: str) -> int:
    """Return the next acquisition sequence number for ``source``.

    Sequences are owned by the source: they advance when the source captures a
    reading, so a cached sample keeps the sequence of the moment it was taken
    rather than the moment it was handed to the caller.
    """
    with _SEQUENCE_LOCK:
        counter = _SEQUENCE_COUNTERS.get(source)
        if counter is None:
            counter = itertools.count(1)
            _SEQUENCE_COUNTERS[source] = counter
        return next(counter)


def _normalise(
    raw: dict[str, Any],
    *,
    now_wall: float,
    now_mono: float,
    source: str,
) -> dict[str, Any]:
    """Coerce an adapter result into the canonical sample shape.

    ``acquired_at``/``sequence`` set by the source are preserved verbatim;
    ``now_mono`` is only a last-resort default for adapters that carry no
    acquisition metadata of their own. Stamping the return time here would let
    a slow provider that returns a cached old sample masquerade as fresh.
    """
    zones = raw.get("zones_c")
    if zones is not None and not isinstance(zones, (list, tuple)):
        zones = [zones]
    temperature = raw.get("temperature_c")
    if temperature is None and zones:
        try:
            temperature = max(float(z) for z in zones if z is not None)
        except (TypeError, ValueError):
            temperature = None
    acquired_at = _num(raw.get("acquired_at"))
    if acquired_at is None:
        acquired_at = _num(raw.get("sampled_monotonic"))
    if acquired_at is None:
        acquired_at = _num(now_mono)
    sequence = _sequence(raw.get("sequence"))
    if sequence is None:
        sequence = next_sequence(source)
    sample = {
        "temperature_c": _num(temperature),
        "ram_used_mib": _num(raw.get("ram_used_mib")),
        "ram_total_mib": _num(raw.get("ram_total_mib")),
        "zones_c": [_num(z) for z in (zones or [])],
        "sampled_at": raw.get("sampled_at") or utc_now(),
        "sampled_monotonic": acquired_at,
        "acquired_at": acquired_at,
        "sequence": sequence,
        "age_seconds": _num(raw.get("age_seconds")),
        "source": source,
    }
    return sample


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def _sequence(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class StaticTelemetry:
    """Deterministic adapter for tests and offline simulation."""

    temperature_c: float | None = 55.0
    ram_used_mib: float | None = 1024.0
    ram_total_mib: float | None = 8192.0
    age_seconds: float | None = 0.0
    source: str = "static"
    _zones: list[float] = field(default_factory=list)

    def sample(self) -> dict[str, Any]:
        now = time.monotonic()
        acquired_at = now
        if self.age_seconds:
            # Report an artificially old capture time to exercise staleness.
            acquired_at = now - float(self.age_seconds)
        raw: dict[str, Any] = {
            "temperature_c": self.temperature_c,
            "ram_used_mib": self.ram_used_mib,
            "ram_total_mib": self.ram_total_mib,
            "sampled_monotonic": acquired_at,
            "acquired_at": acquired_at,
            "sequence": next_sequence(self.source),
        }
        return _normalise(raw, now_wall=time.time(), now_mono=now, source=self.source)


@dataclass
class ProcTelemetry:
    """Portable Linux adapter: memory from /proc, temperature from sysfs.

    Every field degrades to ``None`` when unavailable. It never raises; a
    broken read becomes a missing value that the Governor treats as unsafe.
    """

    thermal_root: str = "/sys/class/thermal"
    meminfo_path: str = "/proc/meminfo"
    temperature_c: float | None = None
    zone_glob: str = "thermal_zone*/temp"
    excluded_zone_types: tuple[str, ...] = ("PMIC-Die",)

    def sample(self) -> dict[str, Any]:
        now = time.monotonic()
        zones: list[float] = []
        root = Path(self.thermal_root)
        if root.is_dir():
            for path in sorted(root.glob(self.zone_glob)):
                zone_type = _read_zone_type(path)
                if zone_type is not None and zone_type in self.excluded_zone_types:
                    continue
                value = _read_millidegree(path)
                if value is not None:
                    zones.append(value)
        used = total = None
        try:
            meminfo = Path(self.meminfo_path).read_text(encoding="utf-8", errors="replace")
            values = {}
            for line in meminfo.splitlines():
                if ":" in line:
                    key, rest = line.split(":", 1)
                    values[key.strip()] = rest.strip().split()[0]
            if "MemTotal" in values:
                total = float(values["MemTotal"]) / 1024.0
            if "MemAvailable" in values and total is not None:
                available = float(values["MemAvailable"]) / 1024.0
                used = max(0.0, total - available)
        except (OSError, ValueError):
            pass
        temperature = self.temperature_c
        if temperature is None and zones:
            temperature = max(zones)
        raw = {
            "temperature_c": temperature,
            "zones_c": zones,
            "ram_used_mib": used,
            "ram_total_mib": total,
            "sampled_monotonic": now,
            "acquired_at": now,
            "sequence": next_sequence("proc"),
        }
        return _normalise(raw, now_wall=time.time(), now_mono=now, source="proc")


@dataclass
class FileTelemetry:
    """Reads a JSON sample from disk (e.g. a sidecar writing telemetry).

    Missing/old files surface as ``age_seconds`` so staleness is explicit. The
    file's modification time is the acquisition time: the adapter maps it onto
    the monotonic clock rather than pretending the read happened just now.
    """

    path: str | Path

    def sample(self) -> dict[str, Any]:
        import json

        now = time.monotonic()
        raw: dict[str, Any] = {}
        age = None
        acquired_at = now
        try:
            p = Path(self.path)
            stat = p.stat()
            age = max(0.0, time.time() - stat.st_mtime)
            acquired_at = now - age
            raw = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raw = {}
        except (OSError, ValueError):
            raw = {}
            age = None
            acquired_at = now
        if "acquired_at" not in raw:
            raw["acquired_at"] = acquired_at
        if "sampled_monotonic" not in raw:
            raw["sampled_monotonic"] = acquired_at
        if "sequence" not in raw:
            raw["sequence"] = next_sequence("file")
        sample = _normalise(raw, now_wall=time.time(), now_mono=now, source="file")
        if age is not None:
            sample["age_seconds"] = age
            sample["sampled_monotonic"] = acquired_at
            sample["acquired_at"] = acquired_at
        return sample


def _read_millidegree(path: Path) -> float | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
        value = float(raw)
    except (OSError, ValueError):
        return None
    # Most Linux thermal zones report milli-degrees; some report degrees.
    if abs(value) > 1000:
        value = value / 1000.0
    return value


def _read_zone_type(path: Path) -> str | None:
    """Return the stripped ``type`` sibling of a thermal zone temp file.

    Returns ``None`` when the ``type`` file cannot be read, so the caller can
    default to keeping the zone rather than silently dropping real readings.
    """
    try:
        return (path.parent / "type").read_text(encoding="utf-8").strip()
    except OSError:
        return None

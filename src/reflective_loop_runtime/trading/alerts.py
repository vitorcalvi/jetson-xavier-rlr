"""Trader alerts (P2 of the RLR autonomy plan): an append-only outbox.

The trading package has no network egress except Alpaca (static guard), so
``Alerter.send`` only appends a JSON line to a local outbox file; the separate
``deploy/alert_relay.py`` service posts it to ntfy. ``send`` never raises.
``AlertingLedger`` wraps the trade ledger and turns notable journal entries
into alerts, so the reviewed loop code needs no changes.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Mapping
from typing import Any

_SECRET_PATTERNS = (
    re.compile(r"PK[A-Z0-9]{10,}"),
    re.compile(r"[A-Za-z0-9+/=_-]{30,}"),
)
_PRIORITIES = {"min", "low", "default", "high", "urgent"}

# journal kind -> (priority, short label)
_ALERT_KINDS: dict[str, tuple[str, str]] = {
    "prelude_failed": ("high", "controller halted"),
    "reconcile_failed": ("high", "broker reconcile failed"),
    "kill_at_dispatch": ("high", "send refused at dispatch"),
    "controller_record_failed": ("high", "controller record failed"),
    "recovered_fill_ledger_failed": ("urgent", "recovered fill not in ledger"),
    "unknown_submit": ("urgent", "unknown order outcome"),
    "position_held_outside_session": ("urgent", "position held after close"),
    "session_flatten": ("default", "session flatten"),
    "market_error": ("low", "market data error"),
    "halt": ("low", "kill file present"),
}


def _redact(text: str) -> str:
    for pat in _SECRET_PATTERNS:
        text = pat.sub("<redacted>", text)
    return text[:1000]


def _append_outbox(path: str, record: dict[str, Any]) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


class Alerter:
    def __init__(
        self,
        outbox: str | None,
        *,
        sink: Callable[[dict[str, Any]], None] | None = None,
        min_interval_s: float = 60.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._outbox = (outbox or "").strip()
        self._sink = sink or (lambda rec: _append_outbox(self._outbox, rec))
        self._min_interval_s = float(min_interval_s)
        self._clock = clock
        self._last: dict[str, float] = {}

    @property
    def enabled(self) -> bool:
        return bool(self._outbox)

    def send(self, kind: str, message: str, *, priority: str = "default",
             dedupe_key: str | None = None) -> bool:
        if not self._outbox:
            return False
        try:
            key = dedupe_key or kind
            now = self._clock()
            if now - self._last.get(key, -1e18) < self._min_interval_s:
                return False
            self._last[key] = now
            self._sink({
                "ts": now,
                "title": f"RLR {kind}",
                "priority": priority if priority in _PRIORITIES else "default",
                "tags": kind,
                "message": _redact(message),
            })
            return True
        except Exception:  # noqa: BLE001 - alerting must never affect trading
            return False


def from_env(env: Mapping[str, str], *, data_dir: Any = None) -> Alerter:
    """Enabled when RLR_ALERTS=1: outbox at RLR_ALERT_OUTBOX or <data_dir>/alerts_outbox.jsonl."""
    if (env.get("RLR_ALERTS") or "") != "1":
        return Alerter(None)
    path = env.get("RLR_ALERT_OUTBOX") or (os.path.join(str(data_dir), "alerts_outbox.jsonl") if data_dir else "")
    return Alerter(path or None)


class AlertingLedger:
    """Ledger proxy: forwards everything, alerts on notable journal entries."""

    def __init__(self, ledger: Any, alerter: Alerter, *, label: str = "trader") -> None:
        self._ledger = ledger
        self._alerter = alerter
        self._label = label

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ledger, name)

    def journal(self, entry: Mapping[str, Any]) -> Any:
        result = self._ledger.journal(entry)
        try:
            self._maybe_alert(entry)
        except Exception:  # noqa: BLE001
            pass
        return result

    def _maybe_alert(self, e: Mapping[str, Any]) -> None:
        kind = str(e.get("kind", ""))
        if kind == "decision" and e.get("executed"):
            d = e.get("decision") or {}
            f = e.get("features") or {}
            self._alerter.send(
                "order",
                f"[{self._label}] {e.get('symbol')} {d.get('action')} ({d.get('reason')}) "
                f"px={f.get('price')} stop={d.get('stop')} target={d.get('target')}",
                priority="high",
                dedupe_key=f"order-{e.get('ts')}-{e.get('symbol')}",
            )
            return
        if kind == "decision" and e.get("veto") == "daily_loss_halt":
            self._alerter.send("daily_halt", f"[{self._label}] daily loss limit reached: entries blocked today",
                               priority="high", dedupe_key="daily_halt")
            return
        if kind in _ALERT_KINDS:
            priority, label = _ALERT_KINDS[kind]
            rest = {k: v for k, v in e.items() if k not in ("kind", "ts", "tick")}
            self._alerter.send(kind, f"[{self._label}] {label}: {rest}", priority=priority)


__all__ = ["Alerter", "AlertingLedger", "from_env"]

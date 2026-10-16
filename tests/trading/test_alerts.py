"""P2 alerts: outbox writer (no network in src), relay outside the package."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from reflective_loop_runtime.trading.alerts import Alerter, AlertingLedger, from_env

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deploy"))
import alert_relay  # noqa: E402


def _alerter(sent, **kw):
    return Alerter("unused", sink=sent.append, **kw)


def test_disabled_is_noop(tmp_path) -> None:
    assert Alerter(None).send("order", "x") is False
    assert from_env({}, data_dir=tmp_path).enabled is False


def test_writes_outbox_record(tmp_path) -> None:
    a = from_env({"RLR_ALERTS": "1"}, data_dir=tmp_path)
    assert a.send("order", "IREN buy", priority="high")
    rec = json.loads((tmp_path / "alerts_outbox.jsonl").read_text().strip())
    assert (
        rec["title"] == "RLR order" and rec["priority"] == "high" and rec["message"] == "IREN buy"
    )


def test_rate_limit_dedupes() -> None:
    sent: list = []
    t = [0.0]
    a = _alerter(sent, clock=lambda: t[0], min_interval_s=60)
    assert a.send("halt", "a") and not a.send("halt", "b")
    t[0] = 61
    assert a.send("halt", "c") and len(sent) == 2


def test_never_raises_on_sink_error() -> None:
    def boom(rec):
        raise OSError("disk full")

    assert Alerter("x", sink=boom).send("order", "m") is False


def test_redacts_secrets() -> None:
    sent: list = []
    _alerter(sent).send("order", "key PK43OYMSMYKRXGAMEPNI3R3Z7G secret " + "a" * 40)
    assert "PK43" not in sent[0]["message"] and "<redacted>" in sent[0]["message"]


def test_ledger_proxy_alerts_orders_and_incidents() -> None:
    sent: list = []

    class L:
        def __init__(self):
            self.rows = []

        def journal(self, e):
            self.rows.append(e)

        def closed_trades(self):
            return ["ok"]

    base = L()
    led = AlertingLedger(base, _alerter(sent, min_interval_s=0), label="paper")
    led.journal(
        {
            "kind": "decision",
            "ts": 1,
            "symbol": "IREN",
            "executed": True,
            "decision": {"action": "enter_long", "reason": "orb5_breakout"},
            "features": {"price": 35},
        }
    )
    led.journal({"kind": "decision", "ts": 2, "executed": False, "veto": "daily_loss_halt"})
    led.journal({"kind": "prelude_failed", "ts": 3, "stage": "reconcile"})
    led.journal({"kind": "decision", "ts": 4, "executed": False})
    assert [r["tags"] for r in sent] == ["order", "daily_halt", "prelude_failed"]
    assert len(base.rows) == 4 and led.closed_trades() == ["ok"]


def test_relay_posts_new_lines_once_and_retries_on_failure(tmp_path) -> None:
    box = tmp_path / "alerts_outbox.jsonl"
    a = from_env({"RLR_ALERTS": "1"}, data_dir=tmp_path)
    a.send("order", "one")
    a.send("halt", "two")
    posted: list = []
    calls = {"n": 0}

    def flaky(url, rec):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("ntfy down")
        posted.append(rec["message"])

    assert alert_relay.relay_once(str(box), "https://ntfy.example/t", flaky) == 1
    assert alert_relay.relay_once(str(box), "https://ntfy.example/t", flaky) == 1
    assert posted == ["one", "two"]
    assert alert_relay.relay_once(str(box), "https://ntfy.example/t", flaky) == 0

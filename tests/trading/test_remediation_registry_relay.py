import json

from deploy.alert_relay import relay_once
from reflective_loop_runtime.trading.experiment_registry import load


def test_registry_roundtrip(tmp_path):
    # test path must be beneath /home/vitor/rlr for safety
    assert load.__module__.endswith("experiment_registry")


def test_relay_offset_retries_on_failure(tmp_path):
    p = tmp_path / "alerts_outbox.jsonl"
    p.write_text('{"message":"buy","title":"SPY"}\n{"message":"sell","title":"SPY"}\n')
    seen = []

    def first(url, rec, *, key):
        seen.append((rec["message"], key))
        if rec["message"] == "sell":
            raise ConnectionError

    assert relay_once(p, "https://example.test", first) == 1
    x = json.loads((tmp_path / "relay_offset.json").read_text())
    assert x["byte_offset"] < p.stat().st_size

    def next_delivery(url, rec, *, key):
        seen.append((rec["message"], key))

    assert relay_once(p, "https://example.test", next_delivery) == 1
    x = json.loads((tmp_path / "relay_offset.json").read_text())
    assert x["byte_offset"] == p.stat().st_size
    assert seen[-1][0] == "sell"
    assert relay_once(p, "https://example.test", next_delivery) == 0

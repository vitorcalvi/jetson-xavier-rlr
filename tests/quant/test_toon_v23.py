"""Offline regression for TOON v2.3; no external orders or services."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reflective_loop_runtime.quant.edge_jetson.tabular_v23 import decode, encode
from reflective_loop_runtime.quant.execution.microstructure_v22 import BBO, OFIWindow
from reflective_loop_runtime.quant.execution.smart_router import Book, Level
from scripts.certify_alpaca_paper_30_v23 import PaperRefusal, synthetic_limit, validate_host
from scripts.toon23_soak_notifier import evaluate
from scripts.verify_real_l1_l2_ic import score


def test_lossless_alias_time_bool_bitfield_and_book_arrays():
    rows = [
        {
            "epoch": 1792147074.015 + i * 300,
            "stage2_pid": 2355006,
            "halt_verified": i != 2,
            "thermals_c": {"thermal_zone0": 29.1 + i, "thermal_zone1": 30.0},
            "bids": [[81600.11, 1.025], [81600.10, 0.5 + i]],
            "broker_positions": 0,
        }
        for i in range(10)
    ]
    assert decode(encode(rows)) == rows


def test_roundtrip_rejects_missing_uniform_and_bad_header():
    with pytest.raises(ValueError):
        encode([{"x": 1}, {"y": 2}])
    with pytest.raises(ValueError):
        decode("bad\n")
    with pytest.raises(ValueError):
        encode([{"x": float("nan")}, {"x": 1.0}])


def test_paper_host_guard_and_passive_limit_not_a_maker_guarantee():
    for url in (
        "http://paper-api.alpaca.markets",
        "https://api.alpaca.markets",
        "https://paper-api.alpaca.markets.example.net",
    ):
        with pytest.raises(PaperRefusal):
            validate_host(url)
    assert synthetic_limit(80000, 80001, 2, 5, "buy") < 80001
    assert synthetic_limit(80000, 80001, 2, 5, "sell") > 80000
    with pytest.raises(PaperRefusal):
        synthetic_limit(80000, 80000.01, 2, 5, "buy")


def test_soak_evaluator_fails_if_halt_or_exposure_changes():
    rows = [
        {
            "epoch": 1000 + i * 301,
            "stage2_pid": 2355006,
            "stage2_rss_mb": 45.0 + i / 100000,
            "stage2_fd_count": 40,
            "max_thermal_c": 50,
            "halt_verified": True,
            "broker_positions": 0,
            "broker_open_orders": 0,
        }
        for i in range(576)
    ]
    assert evaluate(rows)["status"] == "PASS"
    rows[-1]["broker_positions"] = 1
    assert evaluate(rows)["status"] == "FAIL"
    rows[-1]["broker_positions"] = 0
    rows[-1]["stage2_pid"] = 999
    assert evaluate(rows)["status"] == "FAIL"


def test_soak_evaluator_pending_when_less_than_48h():
    rows = [
        {
            "epoch": i * 300,
            "stage2_pid": 2355006,
            "stage2_rss_mb": 45.0,
            "stage2_fd_count": 7,
            "max_thermal_c": 50,
            "halt_verified": True,
            "broker_positions": 0,
            "broker_open_orders": 0,
        }
        for i in range(576)
    ]
    assert evaluate(rows)["status"] == "PENDING"


def test_recorded_genuine_book_and_bbo_replay_is_offline():
    source = Path(__file__).parent / "fixtures" / "recorded_binance_spot_20261016_sample.jsonl"
    events = [json.loads(line) for line in source.read_text().splitlines()]
    assert len(events) == 120
    assert all(x["source"] == "Binance public spot WS" for x in events)
    bbo = OFIWindow(5)
    valid = 0
    for e in events:
        if e["symbol"] != "btcusdt":
            continue
        if e["tier"] == "L2":
            book = Book(
                tuple(Level(*b) for b in e["bids"]),
                tuple(Level(*a) for a in e["asks"]),
                e["capture_ns"] // 1_000_000,
                e["capture_ns"],
            )
            assert -1 <= book.imbalance(5) <= 1
            valid += 1
        elif e["tier"] == "L1":
            tick = BBO(
                e["symbol"],
                e["capture_ns"] // 1_000_000,
                e["capture_ns"],
                e["bid"],
                e["bid_qty"],
                e["ask"],
                e["ask_qty"],
                "Binance public spot WS",
            )
            if bbo.last is None or tick.timestamp_ms > bbo.last.timestamp_ms:
                bbo.update(tick)
    assert valid == 12
    assert len(bbo.events) == 5


def test_g4_no_claim_of_verified_maker_expectancy_from_quotes():
    source = Path(__file__).parent / "fixtures" / "recorded_binance_spot_20261016_sample.jsonl"
    result = score(source)
    assert result["genuine_snapshot_count"] == 60
    assert result["maker_expectancy_net_verified"] is False
    assert result["gate_5000"] is False

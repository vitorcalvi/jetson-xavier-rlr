"""Tests for :mod:`reflective_loop_runtime.trading.ledger`.

The ledger is the source of truth for trade persistence; the four named tests
below are the contract: R-multiple arithmetic, the secret-scrubbing journal,
the UTC-day risk aggregation, and the open-position round-trip.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from reflective_loop_runtime.trading.ledger import Ledger
from reflective_loop_runtime.trading.model import Position, RiskState

# -- helpers ---------------------------------------------------------------


def _ts(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> float:
    """Return a UTC epoch second for the given calendar fields."""
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc).timestamp()


def _make_ledger(tmp_path: Path, *, name: str = "ledger.db") -> Ledger:
    db = tmp_path / name
    journal = tmp_path / f"{name}.jsonl"
    return Ledger(db, journal)


# -- test_r_multiple_computation -------------------------------------------


def test_r_multiple_computation(tmp_path: Path) -> None:
    """``close_trade`` must compute R as ``(exit - entry) / (entry - stop)``.

    The exercise covers a profitable 1R, a break-even 0R, a losing -1R and a
    profitable 2R so a sign-flip or divide-by-zero regression fails loudly.
    """
    led = _make_ledger(tmp_path)

    # Long at 100, stop at 99 -> 1 unit of risk. Exit at 101 -> 1R profit.
    led.open_trade(
        symbol="AAA",
        entry_ts=_ts(2026, 10, 11, 14, 0),
        entry_px=100.0,
        qty=10.0,
        stop=99.0,
        target=103.0,
        atr=1.0,
        entry_order_id="o-1",
    )
    win = led.close_trade(
        symbol="AAA",
        exit_ts=_ts(2026, 10, 11, 14, 5),
        exit_px=101.0,
        reason="target",
        exit_order_id="x-1",
    )
    assert win["r_multiple"] == pytest.approx(1.0)

    # Exit exactly at entry -> 0R.
    led.open_trade(
        symbol="BBB",
        entry_ts=_ts(2026, 10, 11, 14, 0),
        entry_px=50.0,
        qty=4.0,
        stop=48.0,
        target=55.0,
        atr=1.0,
        entry_order_id="o-2",
    )
    flat = led.close_trade(
        symbol="BBB",
        exit_ts=_ts(2026, 10, 11, 14, 10),
        exit_px=50.0,
        reason="scratch",
        exit_order_id="x-2",
    )
    assert flat["r_multiple"] == pytest.approx(0.0)

    # Loss: exit at stop -> -1R.
    led.open_trade(
        symbol="CCC",
        entry_ts=_ts(2026, 10, 11, 14, 0),
        entry_px=200.0,
        qty=2.0,
        stop=198.0,
        target=210.0,
        atr=1.0,
        entry_order_id="o-3",
    )
    loss = led.close_trade(
        symbol="CCC",
        exit_ts=_ts(2026, 10, 11, 14, 20),
        exit_px=198.0,
        reason="stop",
        exit_order_id="x-3",
    )
    assert loss["r_multiple"] == pytest.approx(-1.0)

    # Bigger win: 2R profit on a 1-unit risk.
    led.open_trade(
        symbol="DDD",
        entry_ts=_ts(2026, 10, 11, 14, 0),
        entry_px=20.0,
        qty=5.0,
        stop=18.0,
        target=24.0,
        atr=1.0,
        entry_order_id="o-4",
    )
    big = led.close_trade(
        symbol="DDD",
        exit_ts=_ts(2026, 10, 11, 14, 30),
        exit_px=24.0,
        reason="target",
        exit_order_id="x-4",
    )
    assert big["r_multiple"] == pytest.approx(2.0)

    led.close()


# -- test_keys_never_in_journal --------------------------------------------


def test_keys_never_in_journal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Full values and >= 6-char substrings of the Alpaca secrets must be scrubbed.

    We plant both full values and partial substrings into a record, then read
    back the journal file line-by-line to confirm none of them leaked through.
    The keys themselves are also temporarily removed from the environment after
    the test so other tests are not polluted.
    """
    full_key = "PKTESTZZ1234567890ABCDEFGHIJ"
    full_secret = "abcdefghijklmnopqrstuvwxyz1234567890ABCD"  # 38 chars
    monkeypatch.setenv("ALPACA_PAPER_KEY", full_key)
    monkeypatch.setenv("ALPACA_PAPER_SECRET", full_secret)

    led = _make_ledger(tmp_path)
    # A substring of length >= 6 from the middle of the key.
    key_substring = full_key[5:11]
    secret_substring = full_secret[10:18]
    payload = {
        "event": "tick",
        "note": "alive",
        "token": full_key,
        "nested": {"secret": full_secret, "frag": key_substring},
        "more": [secret_substring, "harmless"],
    }
    led.journal(payload)
    led.close()

    journal_file = tmp_path / "ledger.db.jsonl"
    raw = journal_file.read_text(encoding="utf-8")
    assert full_key not in raw
    assert full_secret not in raw
    assert key_substring not in raw
    assert secret_substring not in raw

    # Round-trip through JSON: each line must still be a valid JSON object and
    # must not contain either of the keys.
    for line in raw.splitlines():
        decoded = json.loads(line)
        assert "PKTESTZZ1234567890ABCDEFGHIJ" not in json.dumps(decoded)
    assert "[REDACTED]" in raw


# -- test_risk_state_counts_utc_day ----------------------------------------


def test_risk_state_counts_utc_day(tmp_path: Path) -> None:
    """``risk_state`` must aggregate over the UTC day boundary only.

    A trade entered on 2026-10-10 must NOT appear in the day-2 aggregate;
    the per-day R and entry counts must come from the same UTC-day window.
    """
    led = _make_ledger(tmp_path)

    # Day 1: a winning trade and a losing trade.
    led.open_trade(
        symbol="WIN",
        entry_ts=_ts(2026, 10, 10, 14, 0),
        entry_px=100.0,
        qty=1.0,
        stop=99.0,
        target=105.0,
        atr=1.0,
        entry_order_id="ow",
    )
    led.close_trade(
        symbol="WIN",
        exit_ts=_ts(2026, 10, 10, 14, 5),
        exit_px=101.0,
        reason="target",
        exit_order_id="xw",
    )
    led.open_trade(
        symbol="LOSE",
        entry_ts=_ts(2026, 10, 10, 15, 0),
        entry_px=100.0,
        qty=1.0,
        stop=99.0,
        target=105.0,
        atr=1.0,
        entry_order_id="ol",
    )
    led.close_trade(
        symbol="LOSE",
        exit_ts=_ts(2026, 10, 10, 15, 30),
        exit_px=99.0,
        reason="stop",
        exit_order_id="xl",
    )

    # Day 2: a still-open entry that started yesterday and one that started today.
    led.open_trade(
        symbol="HOLD",
        entry_ts=_ts(2026, 10, 11, 9, 0),
        entry_px=100.0,
        qty=1.0,
        stop=99.0,
        target=110.0,
        atr=1.0,
        entry_order_id="oh",
    )
    led.open_trade(
        symbol="TODAY",
        entry_ts=_ts(2026, 10, 11, 13, 0),
        entry_px=200.0,
        qty=2.0,
        stop=198.0,
        target=210.0,
        atr=1.0,
        entry_order_id="ot",
    )
    # Close TODAY at a 0.5R profit so realized_r_today > 0.
    led.close_trade(
        symbol="TODAY",
        exit_ts=_ts(2026, 10, 11, 13, 5),
        exit_px=201.0,
        reason="target",
        exit_order_id="xt",
    )

    now = _ts(2026, 10, 11, 16, 0)
    state: RiskState = led.risk_state(now, account_active=True, kill=False)

    # Two entries on/after the UTC day start of Oct 11 (HOLD + TODAY).
    assert state.entries_today == 2
    # Only TODAY closed today at +0.5R.
    assert state.realized_r_today == pytest.approx(0.5)
    # HOLD is still open; TODAY is closed.
    assert "HOLD" in state.open_positions
    assert "TODAY" not in state.open_positions
    # The losing symbol on the current UTC day only shows up after a same-day
    # loss; the previous-day LOSE must not pollute the dict.
    assert "LOSE" not in state.last_loss_ts

    # Now: a same-day loss sets last_loss_ts for that symbol.
    led.open_trade(
        symbol="NEWLOSS",
        entry_ts=_ts(2026, 10, 11, 14, 0),
        entry_px=50.0,
        qty=1.0,
        stop=49.0,
        target=60.0,
        atr=1.0,
        entry_order_id="onl",
    )
    led.close_trade(
        symbol="NEWLOSS",
        exit_ts=_ts(2026, 10, 11, 14, 30),
        exit_px=49.0,
        reason="stop",
        exit_order_id="xnl",
    )
    state2 = led.risk_state(now, account_active=False, kill=True)
    assert state2.entries_today == 3
    # +0.5R (TODAY) and -1.0R (NEWLOSS).
    assert state2.realized_r_today == pytest.approx(-0.5)
    assert "NEWLOSS" in state2.last_loss_ts
    assert state2.account_active is False
    assert state2.kill is True

    led.close()


# -- test_open_positions_roundtrip -----------------------------------------


def test_open_positions_roundtrip(tmp_path: Path) -> None:
    """``open_positions`` must reflect every ``open_trade`` and drop closed ones.

    We open three trades, close one, then assert the snapshot reflects two
    open positions with the exact field values used at entry. A frozen
    ``Position`` is returned, so mutating it must raise.
    """
    led = _make_ledger(tmp_path)

    led.open_trade(
        symbol="ABC",
        entry_ts=_ts(2026, 10, 11, 14, 0),
        entry_px=100.0,
        qty=10.0,
        stop=99.0,
        target=103.0,
        atr=1.25,
        entry_order_id="oid-a",
    )
    led.open_trade(
        symbol="XYZ",
        entry_ts=_ts(2026, 10, 11, 14, 1),
        entry_px=42.0,
        qty=3.0,
        stop=40.0,
        target=50.0,
        atr=0.75,
        entry_order_id="oid-x",
    )
    led.open_trade(
        symbol="FOO",
        entry_ts=_ts(2026, 10, 11, 14, 2),
        entry_px=7.5,
        qty=20.0,
        stop=7.0,
        target=8.5,
        atr=0.3,
        entry_order_id="oid-f",
    )

    snapshot = led.open_positions()
    assert set(snapshot.keys()) == {"ABC", "XYZ", "FOO"}

    abc = snapshot["ABC"]
    assert isinstance(abc, Position)
    assert abc.symbol == "ABC"
    assert abc.qty == pytest.approx(10.0)
    assert abc.entry_px == pytest.approx(100.0)
    assert abc.stop == pytest.approx(99.0)
    assert abc.target == pytest.approx(103.0)
    assert abc.entry_ts == pytest.approx(_ts(2026, 10, 11, 14, 0))
    assert abc.atr == pytest.approx(1.25)

    # Frozen dataclass: mutating must raise.
    with pytest.raises(dataclasses.FrozenInstanceError):
        abc.qty = 0  # type: ignore[misc]

    # Close one, the snapshot must drop it.
    led.close_trade(
        symbol="FOO",
        exit_ts=_ts(2026, 10, 11, 14, 10),
        exit_px=8.0,
        reason="target",
        exit_order_id="xid-f",
    )
    snapshot2 = led.open_positions()
    assert set(snapshot2.keys()) == {"ABC", "XYZ"}

    led.close()

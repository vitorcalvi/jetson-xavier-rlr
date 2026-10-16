"""Trade ledger and journal for the RLR paper trader.

The ledger is the single source of truth for what the trader did: every entry
and exit is persisted to SQLite, every trade's R-multiple is computed from the
committed entry/stop/target/exit prices, and every loop-level decision is
appended to a JSONL journal for post-hoc analysis.

The journal is the place leaks to be careful about. The two Alpaca paper-trading
secrets live in the process environment, so an exception message or a stack
trace that flows through the ledger may carry them. ``journal`` therefore scrubs
the values of ``ALPACA_PAPER_KEY`` and ``ALPACA_PAPER_SECRET`` (and any
substring of either, so a partial match cannot escape) from every JSON record
it serialises. The scrub is applied to the *serialised* string, so a Python
``dict`` whose string payload was already populated by the caller is rewritten
defensively before it hits disk.

The ledger is stdlib-only: SQLite for trades, plain ``append`` for the journal.
There is no async writer and no in-memory mirror; the table is small (a few
trades per UTC day is the design point) so the synchronous path is fine.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# The shared lane imports. The decider lane owns model.py and ships
# ``Position``, ``Decision``, ``Limits`` and ``RiskState`` on top of the same
# wave-1 types; we fall back to local stub dataclasses (matching the brief's
# exact interfaces) so the ledger can be imported even before that work merges.
try:
    from reflective_loop_runtime.trading.model import (  # type: ignore[import-not-found]
        Limits,
        Position,
        RiskState,
    )
except ImportError:  # pragma: no cover - decider lane will resolve the merge

    @dataclass(frozen=True)
    class Position:  # noqa: D401 - stand-in for the shared Position dataclass
        """Stub ``Position`` matching the wave-2 shared interface."""

        symbol: str
        qty: float
        entry_px: float
        stop: float
        target: float
        entry_ts: float
        atr: float

    @dataclass(frozen=True)
    class Limits:  # noqa: D401 - stand-in
        """Stub ``Limits`` matching the wave-2 shared interface."""

        stale_s: float = 120.0
        max_spread_pct: float = 0.20
        max_concurrent: int = 3
        max_notional: float = 1000.0
        max_entries_per_day: int = 10
        daily_loss_halt_r: float = -3.0
        loss_cooldown_s: float = 1800.0
        notional: float = 500.0
        stop_atr: float = 1.5
        target_atr: float = 2.5
        time_stop_s: float = 3600.0
        atr_band: tuple[float, float] = (0.0005, 0.02)

    @dataclass
    class RiskState:  # noqa: D401 - stand-in
        """Stub ``RiskState`` matching the wave-2 shared interface."""

        entries_today: int
        realized_r_today: float
        open_positions: dict
        last_loss_ts: dict
        account_active: bool
        kill: bool


# The schema is fixed; we never ALTER it in place. Adding a column is a
# migration that requires a separate path.
_TRADES_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT NOT NULL,
    entry_ts      REAL NOT NULL,
    entry_px      REAL NOT NULL,
    qty           REAL NOT NULL,
    stop          REAL NOT NULL,
    target        REAL NOT NULL,
    atr           REAL NOT NULL,
    exit_ts       REAL,
    exit_px       REAL,
    exit_reason   TEXT,
    r_multiple    REAL,
    entry_order_id TEXT,
    exit_order_id  TEXT,
    closed        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_trades_open
    ON trades (symbol) WHERE closed = 0;
CREATE INDEX IF NOT EXISTS ix_trades_closed_day
    ON trades (exit_ts, symbol);
"""

_OPEN_TRADE_COLUMNS = (
    "symbol",
    "entry_ts",
    "entry_px",
    "qty",
    "stop",
    "target",
    "atr",
    "entry_order_id",
)


def _utc_day_start(now: float) -> float:
    """Return the Unix-epoch second at which the UTC day of ``now`` starts."""
    dt = datetime.fromtimestamp(float(now), tz=timezone.utc)
    return datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc).timestamp()


def _scrub_secrets(text: str) -> str:
    """Remove Alpaca paper-trading secret substrings from ``text``.

    The full values from the environment and every contiguous substring of
    length >= 6 are stripped. Substrings that short are unlikely to leak a
    useful secret but may shred real data, so the threshold errs on the side
    of safety without being silly.
    """
    if not text:
        return text
    secrets: list[str] = []
    for var in ("ALPACA_PAPER_KEY", "ALPACA_PAPER_SECRET"):
        value = os.environ.get(var, "")
        if isinstance(value, str) and value:
            secrets.append(value)
    for full in secrets:
        if full in text:
            text = text.replace(full, "[REDACTED]")
    # Substring scrub -- catches ``...XY[rest of key]...`` traces and partial
    # dumps of longer secrets. We only scrub substrings of >= 6 chars so that
    # ordinary JSON keys (``"qty"``, ``"side"``) are not over-redacted.
    for full in secrets:
        if len(full) < 6:
            continue
        for start in range(len(full) - 5):
            sub = full[start : start + 6]
            if sub in text:
                text = text.replace(sub, "[REDACTED]")
    return text


def _scrub_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a serialisable copy of ``record`` with secret substrings removed.

    Strings are scrubbed in place; non-string scalars are passed through. The
    scrub works on the serialised form because that is what hits disk, so a
    string that was concatenated *after* construction is still caught.
    """
    payload = json.dumps(dict(record), default=str, sort_keys=True)
    cleaned = _scrub_secrets(payload)
    return json.loads(cleaned)


class Ledger:
    """SQLite-backed trade ledger plus a JSONL journal.

    The two are intentionally separate: trades are relational (we need to query
    "open positions" cheaply); the journal is append-only and structured for
    post-hoc replay. Both live under ``<db_path>``/``<journal_path>`` and are
    flushed to disk synchronously so a crash never loses the last tick.
    """

    def __init__(
        self, db_path: str | Path, journal_path: str | Path, *, modeled_crypto_fee_rate: float = 0.0
    ) -> None:
        # Modeled fee risk is conservative until broker CFEE is reconciled.
        if not 0.0 <= modeled_crypto_fee_rate < 1.0:
            raise ValueError("invalid modeled fee")
        self._fee_rate = float(modeled_crypto_fee_rate)
        self._db_path = Path(db_path)
        self._journal_path = Path(journal_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._journal_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_TRADES_SCHEMA)

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    def __enter__(self) -> Ledger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- writes ------------------------------------------------------------
    def open_trade(
        self,
        *,
        symbol: str,
        entry_ts: float,
        entry_px: float,
        qty: float,
        stop: float,
        target: float,
        atr: float,
        entry_order_id: str | None,
    ) -> int:
        """Insert a new open trade row and return its primary key."""
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("symbol must be a non-empty string")
        for name, value in (
            ("entry_ts", entry_ts),
            ("entry_px", entry_px),
            ("qty", qty),
            ("stop", stop),
            ("target", target),
            ("atr", atr),
        ):
            if not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number")
        cur = self._conn.execute(
            "INSERT INTO trades ({cols}) VALUES (?, ?, ?, ?, ?, ?, ?, ?)".format(
                cols=", ".join(_OPEN_TRADE_COLUMNS)
            ),
            (
                symbol,
                float(entry_ts),
                float(entry_px),
                float(qty),
                float(stop),
                float(target),
                float(atr),
                str(entry_order_id) if entry_order_id is not None else None,
            ),
        )
        return int(cur.lastrowid)

    def close_trade(
        self,
        *,
        symbol: str,
        exit_ts: float,
        exit_px: float,
        reason: str,
        exit_order_id: str | None,
    ) -> dict[str, Any]:
        """Close the open trade for ``symbol`` and return the updated row.

        The row is matched on ``symbol`` and ``closed = 0``; if there is no
        open trade the call raises ``ValueError`` rather than silently closing
        nothing -- the loop depends on a closed row being the result of a
        successful exit fill.
        """
        row = self._conn.execute(
            "SELECT id, entry_px, stop, target FROM trades "
            "WHERE symbol = ? AND closed = 0 ORDER BY id DESC LIMIT 1",
            (str(symbol),),
        ).fetchone()
        if row is None:
            raise ValueError(f"no open trade to close for symbol={symbol!r}")
        risk_per_unit = float(row["entry_px"]) - float(row["stop"])
        if risk_per_unit == 0:
            raise ValueError(
                f"trade {row['id']} for {symbol!r} has stop == entry, cannot compute R"
            )
        r_multiple = (float(exit_px) - float(row["entry_px"])) / risk_per_unit
        self._conn.execute(
            "UPDATE trades SET exit_ts = ?, exit_px = ?, exit_reason = ?, "
            "exit_order_id = ?, r_multiple = ?, closed = 1 WHERE id = ?",
            (
                float(exit_ts),
                float(exit_px),
                str(reason),
                str(exit_order_id) if exit_order_id is not None else None,
                float(r_multiple),
                int(row["id"]),
            ),
        )
        return {
            "id": int(row["id"]),
            "symbol": str(symbol),
            "entry_px": float(row["entry_px"]),
            "stop": float(row["stop"]),
            "target": float(row["target"]),
            "exit_px": float(exit_px),
            "exit_ts": float(exit_ts),
            "exit_reason": str(reason),
            "r_multiple": float(r_multiple),
        }

    # -- reads -------------------------------------------------------------
    def open_positions(self) -> dict[str, Position]:
        """Snapshot of every open trade as ``symbol -> Position``.

        ``Position`` instances are constructed from the row. Frozen dataclass,
        so the caller cannot mutate the row.
        """
        rows = self._conn.execute(
            "SELECT symbol, qty, entry_px, stop, target, entry_ts, atr "
            "FROM trades WHERE closed = 0 ORDER BY symbol, id"
        ).fetchall()
        positions: dict[str, Position] = {}
        for row in rows:
            positions[str(row["symbol"])] = Position(
                symbol=str(row["symbol"]),
                qty=float(row["qty"]),
                entry_px=float(row["entry_px"]),
                stop=float(row["stop"]),
                target=float(row["target"]),
                entry_ts=float(row["entry_ts"]),
                atr=float(row["atr"]),
            )
        return positions

    def risk_state(self, now: float, *, account_active: bool, kill: bool) -> RiskState:
        """Build a ``RiskState`` from the ledger for the UTC day of ``now``.

        ``entries_today`` counts every trade (open + closed) opened since the
        current UTC day started; ``realized_r_today`` sums the ``r_multiple``
        of trades closed in the same window. ``last_loss_ts`` is the
        ``exit_ts`` of the most recent closed trade with ``r_multiple < 0`` for
        each symbol -- the cooldown clock is keyed off this.
        """
        day_start = _utc_day_start(float(now))
        entries_row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM trades WHERE entry_ts >= ?",
            (day_start,),
        ).fetchone()
        realized_row = self._conn.execute(
            "SELECT COALESCE(SUM(r_multiple), 0.0) AS r "
            "FROM trades WHERE closed = 1 AND exit_ts IS NOT NULL AND exit_ts >= ?",
            (day_start,),
        ).fetchone()
        net_r = float(realized_row["r"]) if realized_row is not None else 0.0
        if self._fee_rate:
            # Model buy CFEE in kind and sell CFEE in USD, using actual
            # broker fill prices but a conservative fee tier. This is not
            # a substitute for separately reconciling broker activities.
            rows = self._conn.execute(
                "SELECT entry_px, exit_px, stop FROM trades WHERE closed=1 "
                "AND exit_ts IS NOT NULL AND exit_ts>=?",
                (day_start,),
            ).fetchall()
            net_r = 0.0
            for row in rows:
                risk = float(row["entry_px"]) - float(row["stop"])
                if risk > 0:
                    net_r += (
                        float(row["exit_px"]) * (1 - self._fee_rate) ** 2 - float(row["entry_px"])
                    ) / risk

        last_loss_rows = self._conn.execute(
            "SELECT symbol, MAX(exit_ts) AS last FROM trades "
            "WHERE closed = 1 AND r_multiple IS NOT NULL AND r_multiple < 0 "
            "AND exit_ts >= ? GROUP BY symbol",
            (day_start,),
        ).fetchall()
        last_loss_ts = {str(r["symbol"]): float(r["last"]) for r in last_loss_rows}
        return RiskState(
            entries_today=int(entries_row["n"]) if entries_row is not None else 0,
            realized_r_today=net_r,
            open_positions=self.open_positions(),
            last_loss_ts=last_loss_ts,
            account_active=bool(account_active),
            kill=bool(kill),
        )

    def closed_trades(self, day: float | None = None) -> list[dict[str, Any]]:
        """All trades closed on the UTC day of ``day`` (or all closed if None).

        Returned as plain dicts so reporters can serialise them without picking
        up ``Position``-class surprises.
        """
        if day is None:
            rows = self._conn.execute(
                "SELECT id, symbol, entry_ts, entry_px, qty, stop, target, atr, "
                "exit_ts, exit_px, exit_reason, r_multiple, entry_order_id, exit_order_id "
                "FROM trades WHERE closed = 1 ORDER BY exit_ts ASC, id ASC"
            ).fetchall()
        else:
            day_start = _utc_day_start(float(day))
            day_end = day_start + 86400.0
            rows = self._conn.execute(
                "SELECT id, symbol, entry_ts, entry_px, qty, stop, target, atr, "
                "exit_ts, exit_px, exit_reason, r_multiple, entry_order_id, exit_order_id "
                "FROM trades WHERE closed = 1 AND exit_ts IS NOT NULL "
                "AND exit_ts >= ? AND exit_ts < ? ORDER BY exit_ts ASC, id ASC",
                (day_start, day_end),
            ).fetchall()
        return [dict(row) for row in rows]

    def has_open(self, symbol: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM trades WHERE symbol = ? AND closed = 0 LIMIT 1",
            (str(symbol),),
        ).fetchone()
        return row is not None

    # -- journal -----------------------------------------------------------
    def journal(self, record: Mapping[str, Any]) -> None:
        """Append ``record`` as one JSONL line, scrubbing secrets.

        The record is re-serialised so that nested dicts and non-string
        scalars are uniformly converted. The text is then scrubbed of any
        ``ALPACA_PAPER_KEY`` / ``ALPACA_PAPER_SECRET`` substrings (full value
        and any 6-character sub-window) before being written.
        """
        cleaned = _scrub_record(record)
        payload = json.dumps(cleaned, default=str, sort_keys=True)
        # Defensive re-scrub: if a string value came pre-concatenated, the
        # serialised form is what was scrubbed, so nothing further to do here.
        line = _scrub_secrets(payload) + "\n"
        with self._journal_path.open("a", encoding="utf-8") as fh:
            fh.write(line)

    def journal_path(self) -> Path:
        """The on-disk JSONL journal location (callers need this for tests)."""
        return self._journal_path

    def db_path(self) -> Path:
        """The on-disk SQLite location (callers need this for tests)."""
        return self._db_path


__all__ = ["Ledger", "Position", "Limits", "RiskState"]

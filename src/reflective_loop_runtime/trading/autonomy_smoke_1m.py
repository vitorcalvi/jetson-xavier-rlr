"""Bounded 1-minute Alpaca PAPER autonomy smoke test.

Not an alpha strategy. One entry per *service's ledger lifetime*, due to a
separate immutable completion marker gate wired by deploy/trader_runtime.py.
The autonomous state machine opens using a fresh valid quote + completed 1m
bar, then exits after 2 minutes regardless of profit/loss. This tests real
order, fill, ledger, controller, and exit end-to-end; not historical returns.
"""
from __future__ import annotations
import math
from reflective_loop_runtime.trading.model import Decision, Limits, Position
from reflective_loop_runtime.trading.types import Snapshot

SMOKE_LIMITS = dict(
    notional=25.0,
    max_notional=25.0,
    max_concurrent=1,
    max_entries_per_day=1,
    time_stop_s=120.0,
    max_spread_pct=0.20,
)

def decide_autonomy_smoke_1m(snap: Snapshot, position: Position | None, limits: Limits, *, now: float) -> Decision:
    symbol = snap.symbol
    if position is not None:
        if now-position.entry_ts >= limits.time_stop_s:
            return Decision(symbol=symbol, action="exit", reason="autonomy_1m_time_exit")
        return Decision(symbol=symbol, action="hold", reason="autonomy_1m_in_position")
    if not snap.bars or not snap.quote:
        return Decision(symbol=symbol, action="hold", reason="autonomy_1m_no_market_data")
    b = snap.bars[-1];q=snap.quote
    if not (b.ts+60 <= now and 0<=now-(b.ts+60)<=180 and
            math.isfinite(b.c) and b.c>0 and b.v>0):
        return Decision(symbol=symbol, action="hold", reason="autonomy_1m_stale_or_incomplete_bar")
    if not (math.isfinite(q.bid) and math.isfinite(q.ask)
            and 0<q.bid<=q.ask and -10<=now-q.ts<=20
            and (q.ask-q.bid)/((q.ask+q.bid)/2)*100<=limits.max_spread_pct):
        return Decision(symbol=symbol, action="hold", reason="autonomy_1m_invalid_quote")
    # These wide levels are operational guardrails; this test is not
    # attempting to optimize price or return.
    return Decision(symbol=symbol, action="enter_long", reason="autonomy_1m_paper_cycle",
                    stop=b.c*0.98, target=b.c*1.02,
                    notional=limits.notional)

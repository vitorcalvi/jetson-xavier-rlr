"""Deterministic risk governor for the RLR paper trader.

The governor is the final authority on whether a :class:`Decision` is
allowed to flow on to the broker. It is intentionally tiny and side-
effect-free: it reads risk state, returns ``(allowed, veto_code)``, and
performs no clock reads, no logging, and no broker calls.

Veto ladder
-----------
For ``action == "hold"`` the governor always returns ``(True, None)``;
holding is free and the lane needs the heartbeat.

For ``action == "exit"`` the governor also always returns
``(True, None)`` -- even when ``state.kill`` is set. Exits are
unconditional: a kill switch must not be allowed to strand an open
position.

For ``action == "enter_long"`` the vetoes are checked in this exact
order so a log reader can tell *which* trip-wire fired:

1. ``"kill_file"``           -- ``state.kill`` is set
2. ``"account_not_active"``  -- broker account is not trading-enabled
3. ``"stale_data"``          -- ``f.last_bar_age_s > limits.stale_s``
4. ``"spread"``              -- ``f.spread_pct`` is ``None`` or above cap
5. ``"one_per_symbol"``      -- already long this symbol
6. ``"max_concurrent"``      -- too many open positions across the book
7. ``"notional"``            -- order notional over the per-trade cap
8. ``"max_entries_per_day"`` -- daily entry budget exhausted
9. ``"daily_loss_halt"``     -- realized R has tripped the daily floor
10. ``"loss_cooldown"``      -- too soon since this symbol's last loss
"""

from __future__ import annotations

import math

from reflective_loop_runtime.trading.model import (
    ACTION_ENTER_LONG,
    ACTION_EXIT,
    ACTION_HOLD,
    Decision,
    Limits,
    RiskState,
)
from reflective_loop_runtime.trading.types import Features

# Veto codes are kept as module-level constants so tests (and any future
# structured logging) can reference them by name rather than by string.
VETO_KILL = "kill_file"
VETO_ACCOUNT = "account_not_active"
VETO_STALE = "stale_data"
VETO_SPREAD = "spread"
VETO_ONE_PER_SYMBOL = "one_per_symbol"
VETO_MAX_CONCURRENT = "max_concurrent"
VETO_NOTIONAL = "notional"
VETO_MAX_ENTRIES = "max_entries_per_day"
VETO_DAILY_LOSS = "daily_loss_halt"
VETO_LOSS_COOLDOWN = "loss_cooldown"


def _spread_check(f: Features, max_spread_pct: float) -> bool:
    """``True`` when the spread is acceptable (finite and within cap)."""
    if f.spread_pct is None:
        return False
    try:
        return math.isfinite(f.spread_pct) and 0.0 <= f.spread_pct <= max_spread_pct
    except TypeError:
        return False


def review(
    d: Decision,
    f: Features,
    state: RiskState,
    limits: Limits,
    *,
    now: float,
) -> tuple[bool, str | None]:
    """Return ``(allowed, veto_code)`` for ``d`` under the current ``state``.

    ``veto_code`` is ``None`` when ``allowed`` is ``True``. The function
    performs no I/O -- the caller is responsible for recording the veto
    and for translating ``(True, None)`` into a broker action.
    """
    if d.action == ACTION_HOLD:
        return True, None

    if d.action == ACTION_EXIT:
        # Exits are intentionally never vetoed: a stuck position is
        # worse than the daily-loss cap we already blew through.
        return True, None

    if d.action != ACTION_ENTER_LONG:
        # Forward-compatible: an unknown action is refused with a
        # dedicated code so a misbehaving caller cannot masquerade as
        # a hold or an enter.
        return False, "unknown_action"

    if state.kill:
        return False, VETO_KILL

    if not state.account_active:
        return False, VETO_ACCOUNT

    if f.last_bar_age_s > limits.stale_s:
        return False, VETO_STALE

    if not _spread_check(f, limits.max_spread_pct):
        return False, VETO_SPREAD

    if d.symbol in state.open_positions:
        return False, VETO_ONE_PER_SYMBOL

    if len(state.open_positions) >= limits.max_concurrent:
        return False, VETO_MAX_CONCURRENT

    if d.notional is not None and d.notional > limits.max_notional:
        return False, VETO_NOTIONAL

    if state.entries_today >= limits.max_entries_per_day:
        return False, VETO_MAX_ENTRIES

    if state.realized_r_today <= limits.daily_loss_halt_r:
        return False, VETO_DAILY_LOSS

    last_loss = state.last_loss_ts.get(d.symbol, -math.inf)
    if now - last_loss < limits.loss_cooldown_s:
        return False, VETO_LOSS_COOLDOWN

    return True, None


__all__ = [
    "review",
    "VETO_KILL",
    "VETO_ACCOUNT",
    "VETO_STALE",
    "VETO_SPREAD",
    "VETO_ONE_PER_SYMBOL",
    "VETO_MAX_CONCURRENT",
    "VETO_NOTIONAL",
    "VETO_MAX_ENTRIES",
    "VETO_DAILY_LOSS",
    "VETO_LOSS_COOLDOWN",
]

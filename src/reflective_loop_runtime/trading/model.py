"""Data models for the decider and governor of the RLR paper trader.

This module owns the lane-internal dataclasses that flow between the
:mod:`decider` and the :mod:`governor`. The lane's contract with other
lanes is :mod:`reflective_loop_runtime.trading.types`; everything here
builds on those frozen shapes.

The shapes are deliberately small and frozen: a :class:`Position` is a
plain record of an open trade, a :class:`Decision` is what the decider
asks the governor to evaluate, :class:`Limits` parameterizes both, and
:class:`RiskState` is the per-day risk counters the governor reads.

Everything is keyword-only where it can be, so accidental positional
construction cannot silently swap ``stop`` and ``target`` (which would
be a catastrophic trading bug).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Position:
    """A single open long position.

    ``stop`` is the price at which the decider must exit for loss
    control; ``target`` is the price at which it must exit for profit
    taking. ``entry_ts`` is the epoch seconds when the position was
    opened and is used by the time-stop check. ``atr`` is the ATR(14)
    at entry, retained so the governor can reason about the stop
    distance without recomputing it.
    """

    symbol: str
    qty: float
    entry_px: float
    stop: float
    target: float
    entry_ts: float
    atr: float


# Action literals kept as plain strings so the lanes upstream/downstream
# can compare against the brief's literals without an import dance.
ACTION_HOLD = "hold"
ACTION_ENTER_LONG = "enter_long"
ACTION_EXIT = "exit"


@dataclass(frozen=True)
class Decision:
    """What the decider wants to do, plus the protective levels it suggests.

    ``action`` is one of ``"hold"``, ``"enter_long"``, ``"exit"``. ``reason``
    is a short human-readable code describing why the action was chosen --
    useful in logs and in the governor's veto accounting.

    ``stop``, ``target``, and ``notional`` are only meaningful for
    ``enter_long``; they are ``None`` for ``hold`` and ``exit``. Keeping
    the same dataclass for all three actions means the governor sees a
    single, well-typed payload.
    """

    symbol: str
    action: str
    reason: str
    stop: float | None = None
    target: float | None = None
    notional: float | None = None


@dataclass(frozen=True)
class Limits:
    """All numeric knobs governing the decider and the governor.

    Defaults match the Wave-2 brief so a caller can pass no arguments to
    get a safe starting policy, and so tests can override only the
    field they exercise. ``atr_band`` is the inclusive ``(low, high)``
    ratio of ATR14 / price below which volatility is too thin and
    above which it is too wild for the long-only setup; the decider
    refuses to enter outside the band.

    ``min_stop_pct`` is the floor on the stop distance as a fraction of
    entry price -- protects against stops smaller than the round-trip
    cost in calm markets. ``cost_mult`` scales the quoted spread into
    a minimum stop distance so a wide-spread entry cannot produce a
    stop that loses money before the protective level ever fires.
    ``target_r`` sets the take-profit as a multiple of the chosen stop
    distance (an R-multiple), replacing the older ``target_atr`` knob.
    """

    stale_s: float = 240.0
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
    min_stop_pct: float = 0.004
    cost_mult: float = 4.0
    target_r: float = 1.6
    # Hardened paper policy is explicitly enabled by the deployment.
    hardened_entries: bool = False
    vwap_failure_exit: bool = False
    crypto_fee_rate: float = 0.0
    min_net_target_r: float = 0.50
    min_reclaim_spread_mult: float = 1.25
    min_ret_5m: float = 0.0005
    max_quote_age_s: float = 90.0


@dataclass
class RiskState:
    """Per-day risk counters used by the governor.

    ``entries_today`` is the number of entries already approved since
    the daily reset; ``realized_r_today`` is the cumulative realized R
    (multiple of the per-trade risk unit, negative when losing).
    ``open_positions`` maps symbol -> position metadata so the governor
    can enforce the one-per-symbol cap without re-deriving it from the
    broker. ``last_loss_ts`` maps symbol -> epoch of its most recent
    loss; missing entries are treated as ``-inf`` for the cooldown
    arithmetic. ``account_active`` is the live-trading flag from the
    broker account record; ``kill`` is the external hard kill switch
    (a separate authority file writes this; the governor only reads).
    """

    entries_today: int = 0
    realized_r_today: float = 0.0
    open_positions: dict[str, Any] = field(default_factory=dict)
    last_loss_ts: dict[str, float] = field(default_factory=dict)
    account_active: bool = True
    kill: bool = False


__all__ = [
    "Position",
    "Decision",
    "Limits",
    "RiskState",
    "ACTION_HOLD",
    "ACTION_ENTER_LONG",
    "ACTION_EXIT",
]

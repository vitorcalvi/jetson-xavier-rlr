"""Tests for the deterministic risk governor.

The governor reviews a :class:`Decision` against the live :class:`RiskState`
and :class:`Limits`, returning ``(allowed, veto_code)``. The tests below
exercise every named veto and confirm the order in which they fire.
"""

from __future__ import annotations

from reflective_loop_runtime.trading.governor import (
    VETO_ACCOUNT,
    VETO_DAILY_LOSS,
    VETO_KILL,
    VETO_LOSS_COOLDOWN,
    VETO_MAX_CONCURRENT,
    VETO_MAX_ENTRIES,
    VETO_NOTIONAL,
    VETO_ONE_PER_SYMBOL,
    VETO_SPREAD,
    VETO_STALE,
    review,
)
from reflective_loop_runtime.trading.model import (
    ACTION_ENTER_LONG,
    ACTION_EXIT,
    ACTION_HOLD,
    Decision,
    Limits,
    Position,
    RiskState,
)
from reflective_loop_runtime.trading.types import Features

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _features(
    *,
    symbol: str = "BTC/USD",
    price: float = 100.0,
    ema20: float | None = 101.0,
    ema50: float | None = 100.0,
    atr14: float | None = 1.0,
    spread_pct: float | None = 0.05,
    crossed: bool = True,
    age_s: float = 5.0,
) -> Features:
    """Defaults are tuned so a Decision of ``enter_long`` is otherwise allowed."""
    return Features(
        symbol=symbol,
        price=price,
        ema20_5m=ema20,
        ema50_5m=ema50,
        vwap=price - 1.0 if crossed else price + 1.0,
        atr14=atr14,
        ret_60m=0.01,
        spread_pct=spread_pct,
        crossed_above_vwap=crossed,
        last_bar_age_s=age_s,
    )


def _enter(symbol: str = "BTC/USD", *, notional: float = 500.0) -> Decision:
    """Build an enter_long Decision that the decider would emit on the happy path."""
    return Decision(
        symbol=symbol,
        action=ACTION_ENTER_LONG,
        reason="trend_vwap_reclaim",
        stop=98.5,
        target=102.5,
        notional=notional,
    )


def _state(**overrides) -> RiskState:
    """A risk state with everything permissive by default."""
    base = dict(
        entries_today=0,
        realized_r_today=0.0,
        open_positions={},
        last_loss_ts={},
        account_active=True,
        kill=False,
    )
    base.update(overrides)
    return RiskState(**base)


def _limits(**overrides) -> Limits:
    base = dict(
        stale_s=240.0,
        max_spread_pct=0.20,
        max_concurrent=3,
        max_notional=1000.0,
        max_entries_per_day=10,
        daily_loss_halt_r=-3.0,
        loss_cooldown_s=1800.0,
        notional=500.0,
    )
    base.update(overrides)
    return Limits(**base)


# ---------------------------------------------------------------------------
# Hold/exit: never vetoed
# ---------------------------------------------------------------------------


def test_hold_is_always_allowed() -> None:
    """``hold`` actions always pass: they cost nothing and produce no order."""
    d = Decision(symbol="BTC/USD", action=ACTION_HOLD, reason="in_position")
    f = _features()
    # Even with the kill file set, hold still passes.
    state = _state(kill=True, account_active=False)
    limits = _limits(stale_s=0.0, max_spread_pct=-1.0)
    allowed, veto = review(d, f, state, limits, now=0.0)
    assert allowed is True
    assert veto is None


def test_exit_allowed_under_kill_file() -> None:
    """Exits must always pass, even with the kill file set and every other
    trip-wire tripped -- a kill switch cannot strand an open position."""
    d = Decision(symbol="BTC/USD", action=ACTION_EXIT, reason="stop")
    f = _features(age_s=10_000.0, spread_pct=5.0)  # stale, wide spread
    state = _state(
        kill=True,
        account_active=False,
        entries_today=10**6,
        realized_r_today=-1e9,
        open_positions={"BTC/USD": Position("BTC/USD", 1.0, 100.0, 95.0, 110.0, 0.0, 1.0)},
        last_loss_ts={"BTC/USD": 0.0},
    )
    limits = _limits(
        stale_s=1.0,
        max_spread_pct=0.01,
        max_concurrent=0,
        max_notional=0.0,
        max_entries_per_day=0,
        daily_loss_halt_r=1e9,
        loss_cooldown_s=10_000.0,
    )
    allowed, veto = review(d, f, state, limits, now=0.0)
    assert allowed is True
    assert veto is None


# ---------------------------------------------------------------------------
# enter_long vetoes, in the order the brief defines
# ---------------------------------------------------------------------------


def test_kill_file_veto() -> None:
    d = _enter()
    state = _state(kill=True)
    allowed, veto = review(d, _features(), state, _limits(), now=0.0)
    assert allowed is False
    assert veto == VETO_KILL


def test_account_not_active_veto() -> None:
    d = _enter()
    state = _state(account_active=False)
    allowed, veto = review(d, _features(), state, _limits(), now=0.0)
    assert allowed is False
    assert veto == VETO_ACCOUNT


def test_stale_bars_veto() -> None:
    """Fresh bars only; an old bar means the governor refuses to enter."""
    d = _enter()
    # age_s=300 is well past the 240s threshold: stale.
    f = _features(age_s=300.0)
    allowed, veto = review(d, f, _state(), _limits(stale_s=240.0), now=0.0)
    assert allowed is False
    assert veto == VETO_STALE


def test_fresh_bars_pass() -> None:
    """A bar whose age is under the 240s threshold is not vetoed for staleness."""
    d = _enter()
    f = _features(age_s=200.0)
    allowed, veto = review(d, f, _state(), _limits(stale_s=240.0), now=0.0)
    # Some other downstream veto could still fire, but not STALE.
    assert veto != VETO_STALE


def test_spread_veto() -> None:
    """Vetoed when spread is missing OR wider than the cap.

    Two cases per brief: ``None`` spread, and a finite spread over the cap.
    """
    d = _enter()
    # Missing spread -> veto.
    f_none = _features(spread_pct=None)
    allowed, veto = review(d, f_none, _state(), _limits(max_spread_pct=0.20), now=0.0)
    assert allowed is False
    assert veto == VETO_SPREAD

    # Spread at the cap is allowed (inclusive), over is not.
    f_at = _features(spread_pct=0.20)
    allowed, veto = review(d, f_at, _state(), _limits(max_spread_pct=0.20), now=0.0)
    assert allowed is True
    assert veto is None

    f_over = _features(spread_pct=0.25)
    allowed, veto = review(d, f_over, _state(), _limits(max_spread_pct=0.20), now=0.0)
    assert allowed is False
    assert veto == VETO_SPREAD


def test_one_position_per_symbol() -> None:
    """Already long this symbol? Refuse to add a second position in it."""
    d = _enter(symbol="BTC/USD")
    open_pos = {"BTC/USD": Position("BTC/USD", 1.0, 100.0, 95.0, 110.0, 0.0, 1.0)}
    allowed, veto = review(d, _features(), _state(open_positions=open_pos), _limits(), now=0.0)
    assert allowed is False
    assert veto == VETO_ONE_PER_SYMBOL


def test_max_concurrent_veto() -> None:
    d = _enter(symbol="XRP/USD")
    open_positions = {
        sym: Position(sym, 1.0, 100.0, 95.0, 110.0, 0.0, 1.0)
        for sym in ("BTC/USD", "ETH/USD", "SOL/USD")
    }
    allowed, veto = review(
        d, _features(symbol="XRP/USD"), _state(open_positions=open_positions),
        _limits(max_concurrent=3), now=0.0,
    )
    assert allowed is False
    assert veto == VETO_MAX_CONCURRENT

    # Just under the cap is fine.
    open_positions.pop("SOL/USD")
    allowed, veto = review(
        d, _features(symbol="XRP/USD"), _state(open_positions=open_positions),
        _limits(max_concurrent=3), now=0.0,
    )
    assert allowed is True
    assert veto is None


def test_notional_veto() -> None:
    """Order notional above ``max_notional`` is refused."""
    d = _enter(notional=1500.0)
    allowed, veto = review(d, _features(), _state(), _limits(max_notional=1000.0), now=0.0)
    assert allowed is False
    assert veto == VETO_NOTIONAL

    # Notional exactly at the cap passes (the check is ``>``, not ``>=``).
    d_at = _enter(notional=1000.0)
    allowed, veto = review(d_at, _features(), _state(), _limits(max_notional=1000.0), now=0.0)
    assert allowed is True
    assert veto is None


def test_max_entries_per_day() -> None:
    """Once today's entry budget is spent, no more entries until reset."""
    d = _enter()
    state = _state(entries_today=10)
    allowed, veto = review(
        d, _features(), state, _limits(max_entries_per_day=10), now=0.0,
    )
    assert allowed is False
    assert veto == VETO_MAX_ENTRIES

    # One under the budget is still ok.
    state_ok = _state(entries_today=9)
    allowed, veto = review(
        d, _features(), state_ok, _limits(max_entries_per_day=10), now=0.0,
    )
    assert allowed is True
    assert veto is None


def test_daily_loss_halt() -> None:
    """Realized R at or below the daily floor halts new entries."""
    d = _enter()
    # Floor reached.
    state = _state(realized_r_today=-3.0)
    allowed, veto = review(
        d, _features(), state, _limits(daily_loss_halt_r=-3.0), now=0.0,
    )
    assert allowed is False
    assert veto == VETO_DAILY_LOSS

    # Worse than the floor -- still halted.
    state_worse = _state(realized_r_today=-5.0)
    allowed, veto = review(
        d, _features(), state_worse, _limits(daily_loss_halt_r=-3.0), now=0.0,
    )
    assert allowed is False
    assert veto == VETO_DAILY_LOSS

    # Just above the floor -- still allowed.
    state_ok = _state(realized_r_today=-2.99)
    allowed, veto = review(
        d, _features(), state_ok, _limits(daily_loss_halt_r=-3.0), now=0.0,
    )
    assert allowed is True
    assert veto is None


def test_loss_cooldown() -> None:
    """A loss on this symbol blocks a new entry for ``loss_cooldown_s``."""
    d = _enter(symbol="BTC/USD")
    now = 10_000.0
    state = _state(last_loss_ts={"BTC/USD": now - 100.0})
    allowed, veto = review(
        d, _features(), state, _limits(loss_cooldown_s=1800.0), now=now,
    )
    assert allowed is False
    assert veto == VETO_LOSS_COOLDOWN

    # Outside the window -> allowed.
    state_old = _state(last_loss_ts={"BTC/USD": now - 10_000.0})
    allowed, veto = review(
        d, _features(), state_old, _limits(loss_cooldown_s=1800.0), now=now,
    )
    assert allowed is True
    assert veto is None

    # No prior loss recorded for this symbol -> allowed (treated as -inf).
    state_none = _state(last_loss_ts={})
    allowed, veto = review(
        d, _features(), state_none, _limits(loss_cooldown_s=1800.0), now=now,
    )
    assert allowed is True
    assert veto is None


# ---------------------------------------------------------------------------
# Veto ordering: kill first, then account, then stale, then spread, ...
# ---------------------------------------------------------------------------


def test_kill_takes_precedence_over_account_veto() -> None:
    """When the kill file is set, the account flag no longer matters.

    The brief orders ``kill_file`` before ``account_not_active``, so the
    governor must report ``kill_file`` even with the account flagged.
    """
    d = _enter()
    f = _features()
    state = _state(kill=True, account_active=False)
    allowed, veto = review(d, f, state, _limits(), now=0.0)
    assert allowed is False
    assert veto == VETO_KILL


def test_account_takes_precedence_over_stale_veto() -> None:
    """``account_not_active`` fires before ``stale_data``."""
    d = _enter()
    f = _features(age_s=10_000.0)
    state = _state(account_active=False)
    allowed, veto = review(d, f, state, _limits(stale_s=10.0), now=0.0)
    assert allowed is False
    assert veto == VETO_ACCOUNT


def test_stale_takes_precedence_over_spread_veto() -> None:
    """``stale_data`` fires before ``spread``."""
    d = _enter()
    f = _features(age_s=10_000.0, spread_pct=99.0)
    state = _state()
    allowed, veto = review(d, f, state, _limits(stale_s=10.0, max_spread_pct=1.0), now=0.0)
    assert allowed is False
    assert veto == VETO_STALE


def test_unknown_action_is_vetoed() -> None:
    """An action outside the brief's vocabulary is refused cleanly."""
    d = Decision(symbol="BTC/USD", action="weird", reason="x")
    allowed, veto = review(d, _features(), _state(), _limits(), now=0.0)
    assert allowed is False
    assert veto is not None


def test_stale_threshold_is_240() -> None:
    """The default staleness threshold is 240 seconds."""
    assert Limits().stale_s == 240.0


def test_negative_spread_never_authorizes_entry() -> None:
    from reflective_loop_runtime.trading.governor import review, VETO_SPREAD
    from reflective_loop_runtime.trading.model import Decision
    d = Decision(symbol="IREN", action="enter_long", reason="orb5", notional=500)
    f = _features(spread_pct=-200.0)
    allowed, veto = review(d, f, _state(), _limits(max_spread_pct=0.20), now=0.0)
    assert (allowed, veto) == (False, VETO_SPREAD)

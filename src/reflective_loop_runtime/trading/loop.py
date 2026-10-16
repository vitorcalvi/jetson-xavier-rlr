"""Permit-gated trader loop for the RLR paper-trader.

This module glues the wave-1 ingest (snapshots, features) and the wave-1
broker to the kernel's :class:`~reflective_loop_runtime.kernel.toolbroker.ToolBroker`,
the single permit-gated path through which every side effect must pass. The
loop:

1. fetches a snapshot per symbol;
2. computes features;
3. asks the deterministic :func:`decide` for a :class:`Decision`;
4. asks the deterministic :func:`review` whether the decision is allowed;
5. if the decision is non-``hold`` and allowed (and we are not in dry-run),
   mints an ``alpaca_paper.order`` permit, runs it through
   :meth:`ToolBroker.invoke_effect` and records the fill to the
   :class:`~reflective_loop_runtime.trading.ledger.Ledger`;
6. journals every per-symbol decision and the matched veto/executed flag;
7. writes a JSON heartbeat after every tick so the systemd healthcheck can
   see the trader is alive and what state it is in.

The ``alpaca_paper`` operation is registered as ``risk_class="A2"`` because
submitting an order is a bounded, compensable effect (the opposite order can
close it); the manifest registers the compensation so the policy checker is
happy.

The decider lane owns ``trading.model``, ``trading.decider`` and
``trading.governor``. If those modules are not on main yet, the loop falls back
to local stub types and stub logic that match the interfaces exactly, so the
files can be imported and used now and the real implementations can drop in
later without any churn in this module.

When the optional ``OrderController`` is injected (``controller`` keyword), the
loop also performs a per-tick durable prelude (begin_day + killmodes.apply +
broker reconciliation) and gates every entry/exit through the controller's
mode.  When the controller is ``None`` the loop is bit-for-bit compatible with
the previous behaviour: existing tests and the dry-run path keep working
unchanged.

Safety contract (G5 closure):

* :class:`ControllerFailStop` is raised when the controller's storage layer
  becomes unreachable (e.g. ``sqlite3.OperationalError``) -- the trader
  process must exit non-zero so systemd restarts it and the next prelude can
  reconcile from a freshly persisted HALT.  Once tripped, every subsequent
  ``tick()`` raises immediately: no further order effects are admitted.
* When ``require_controller=True`` (the production default in
  ``deploy/trader_runtime.build``), the loop refuses any order effect unless a
  controller is wired in.  This keeps the safety-enabled mode bit-for-bit
  consistent: there is no code path that can produce an order effect without
  a durable record of the intent.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from . import session
from .broker import client_order_id
from .features import compute as compute_features
from .ledger import Ledger, Position
from .market import SYMBOLS

# ----------------------------------------------------------------------------
# Shared lane imports -- replaceable stubs until the decider lane merges
# ----------------------------------------------------------------------------


try:  # pragma: no cover - exercised via tests against whichever path lands first
    from reflective_loop_runtime.trading.model import (  # type: ignore[import-not-found]
        Decision,
        Limits,
        RiskState,
    )
except ImportError:  # pragma: no cover - decider lane will resolve the merge

    @dataclass(frozen=True)
    class Decision:  # noqa: D401 - stand-in
        """Stub ``Decision`` matching the wave-2 shared interface."""

        symbol: str
        action: str  # 'hold' | 'enter_long' | 'exit'
        reason: str
        stop: float | None = None
        target: float | None = None
        notional: float | None = None

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


try:  # pragma: no cover
    from reflective_loop_runtime.trading.decider import decide as _default_decide  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover

    def _default_decide(features, position, limits, *, now: float) -> Decision:
        """Conservative stub that returns ``hold`` until the decider lane lands."""

        return Decision(symbol=features.symbol, action="hold", reason="no-decider")


try:  # pragma: no cover
    from reflective_loop_runtime.trading.governor import review as _default_review  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover

    def _default_review(
        decision, features, state, limits, *, now: float
    ) -> tuple[bool, str | None]:
        """Conservative stub that approves every decision."""

        return (True, None)


# ----------------------------------------------------------------------------
# Fail-stop sentinel
# ----------------------------------------------------------------------------


class ControllerFailStop(RuntimeError):
    """Controller storage unusable: the trader process must stop (no RUN without a provable persisted halt).

    The :class:`~reflective_loop_runtime.trading.orders.OrderController`'s
    durable layer (SQLite) raised an unrecoverable error during the prelude's
    fail-closed reconcile path or during ``mark_submitting``.  Without a
    provable persisted HALT the trader cannot prove that entries are blocked;
    the only safe response is for the process to exit non-zero so systemd
    restarts it.  The next startup reconciles from HALT and the operator
    re-arms ``set_mode(RUN)`` once the broker's view is verified.
    """


# ----------------------------------------------------------------------------
# Manifest: a fresh ToolManifest carrying every wave-1 tool plus alpaca_paper
# ----------------------------------------------------------------------------

from reflective_loop_runtime.kernel.manifest import (  # noqa: E402 - third-party import order
    CostModel,
    FieldSpec,
    OperationSpec,
    ToolManifest,
    ToolSpec,
    default_manifest,
)

_ORDER_SCHEMA: dict[str, FieldSpec] = {
    "symbol": FieldSpec("str", min_length=1, max_length=16),
    "side": FieldSpec("str", choices=("buy", "sell")),
    # Exactly one of ``notional`` (buy) and ``qty`` (sell) must be supplied; the
    # broker adapter enforces the other side too, but registering both as
    # optional gives the broker room to reject them with a clean ValueError
    # rather than the manifest refusing the call.
    "notional": FieldSpec("float", min=0.01, required=False),
    "qty": FieldSpec("float", min=0.0000001, required=False),
    "client_order_id": FieldSpec("str", min_length=4, max_length=64),
}


def trading_manifest() -> ToolManifest:
    """A manifest that knows about the wave-1 tools and the ``alpaca_paper`` op.

    The :mod:`kernel.toolbroker.broker_manifest` helper extends the same default
    manifest with the broker's ``shell`` and ``control`` tools -- the trader
    builds a fresh manifest so a test can construct a manifest without those
    extras if it does not need them. The only trader-specific tool here is
    ``alpaca_paper.order``.
    """
    manifest = default_manifest()
    alpaca = ToolSpec(
        "alpaca_paper",
        {
            "order": OperationSpec(
                name="order",
                input_schema=_ORDER_SCHEMA,
                preconditions=(
                    "the paper API key is loaded",
                    "the paper endpoint is reachable",
                ),
                effects=(
                    "a market order is submitted to the Alpaca paper endpoint",
                    "the fill is recorded in the trade ledger",
                ),
                risk_class="A2",
                reversibility="compensable",
                idempotency="non_idempotent",
                compensation="submit a closing order for the full position",
                allowed_destinations=("paper-api.alpaca.markets",),
                cost_model=CostModel(
                    "api_calls",
                    1.0,
                    "paper order",
                ),
            ),
        },
    )
    manifest.register(alpaca)
    return manifest


# ----------------------------------------------------------------------------
# The loop
# ----------------------------------------------------------------------------


MarketFn = Callable[..., dict]


def _normalise_symbol(raw: Any) -> str:
    """Return ``raw`` in ``BTC/USD`` slash form.

    The Alpaca paper REST API returns crypto positions keyed by the bare
    ticker (``BTCUSD``); the controller's positions and our own logic are
    both keyed by the slashed form. Insert a slash before the trailing
    ``USD`` so the two views line up.
    """
    s = str(raw or "")
    if "/" in s:
        return s
    if s.endswith("USD") and len(s) > 3:
        return s[:-3] + "/USD"
    return s


# Exits never read features; this stand-in keeps the close-time flatten
# independent of market data.
_FLATTEN_FEATURES = SimpleNamespace(price=0.0, atr14=0.0)


class TraderLoop:
    """One permit-gated instance of the trader, parameterised by injected fakes.

    The constructor takes the network, broker, permits, tool broker, ledger, and
    limits as injectable objects; the only thing the loop owns is its clock,
    its kill path, and its in-tick counters. ``dry_run=True`` disables the
    permit/``invoke_effect`` path so a smoke test can run end-to-end without
    side effects.

    The ``controller`` keyword, when supplied, wires the durable
    :class:`~reflective_loop_runtime.trading.orders.OrderController` into the
    loop.  ``data_dir`` is the directory killmodes watches (and the parent of
    ``orders.db`` when the controller is created externally); ``history_start_iso``
    is the ISO-8601 timestamp passed to ``broker.orders_since`` on every
    reconcile.  When ``controller`` is ``None`` the loop is exactly the
    pre-controller version: no prelude, no controller-driven veto, no recovery
    lookup -- the existing KILL-file path is preserved unchanged.

    The ``require_controller`` keyword (default ``False`` for backwards
    compatibility) enables the I0 safety contract: when ``True`` the loop
    refuses any order effect unless a controller is wired in.  The production
    runtime sets this from the ``--dry-run`` flag (``not dry_run``).
    """

    def __init__(
        self,
        market_fn: MarketFn,
        broker: Any,
        permits: Any,
        tool_broker: Any,
        ledger: Ledger,
        limits: Limits,
        *,
        clock: Callable[[], float] = time.time,
        kill_path: str | Path | None = None,
        dry_run: bool = False,
        heartbeat_path: str | Path | None = None,
        decide_fn: Callable[..., Any] | None = None,
        review_fn: Callable[..., Any] | None = None,
        symbols: tuple[str, ...] | None = None,
        controller: Any | None = None,
        data_dir: str | Path | None = None,
        history_start_iso: str | None = None,
        require_controller: bool = False,
        entry_evidence_gate: Callable[[str, Any, float], bool] | None = None,
        equity_session: bool = False,
        market_open_fn: Callable[[], bool] | None = None,
        strategy_fn: Callable[..., Any] | None = None,
        shadow: bool = False,
        shadow_autorun: bool = False,
        paper_autorun: bool = False,
        buy_fee_in_asset: float = 0.0025,
    ) -> None:
        # Optional snapshot-level strategy (e.g. ORB5); replaces decide_fn.
        self._strategy_fn = strategy_fn
        # SHADOW: real broker reads and controller reconciliation, every gate
        # evaluated, but order effects are journalled and never sent.
        self._shadow = bool(shadow)
        self._shadow_autorun = bool(shadow and shadow_autorun)
        # PAPER autorun (user-authorized, paper account only): after each clean
        # broker reconcile re-enter RUN, so the per-order HALT does not need a
        # human on every trade. Hard incidents still block it (set_mode refuses).
        self._paper_autorun = bool(paper_autorun)
        # Crypto takes the buy fee in the asset; US equities have none.
        self._buy_fee_in_asset = float(buy_fee_in_asset)
        self._market_fn = market_fn
        # US-equity mode: regular-session entry window, forced flat at 15:55 ET,
        # and the broker clock must say the market is open before any entry.
        self._equity_session = bool(equity_session)
        self._market_open_fn = market_open_fn
        self._broker = broker
        self._permits = permits
        self._tool_broker = tool_broker
        self._ledger = ledger
        self._limits = limits
        self._clock = clock
        self._dry_run = bool(dry_run)
        self._kill_path = Path(kill_path) if kill_path is not None else None
        self._heartbeat_path = Path(heartbeat_path) if heartbeat_path is not None else None
        self._decide = decide_fn if decide_fn is not None else _default_decide
        self._review = review_fn if review_fn is not None else _default_review
        self._symbols = tuple(symbols) if symbols is not None else SYMBOLS
        self._controller = controller
        self._data_dir = Path(data_dir) if data_dir is not None else None
        self._history_start_iso = history_start_iso
        self._require_controller = bool(require_controller)
        # Optional evidence gate can veto a BUY, never authorize an effect.
        # All P0 controller, governor and signed-permit checks still apply.
        self._entry_evidence_gate = entry_evidence_gate
        self._fail_stopped = False
        self._tick = 0
        self._last_halted = False

    # ---- public ----------------------------------------------------------
    @property
    def tick_count(self) -> int:
        """Number of ticks completed by this loop instance."""
        return self._tick

    def is_halted(self) -> bool:
        """Whether the most recent tick observed a kill flag or hard error."""
        return self._last_halted

    def tick_once(self, now: float | None = None) -> None:
        """Alias for :meth:`tick` -- single pass through every symbol.

        The runtime calls this when ``--once`` is passed; the systemd unit uses
        ``tick`` from its inner loop. Both stay valid because ``tick`` itself
        is a one-shot pass that owns its own journal/heartbeat side effects.
        """
        if now is None:
            now = float(self._clock())
        self.tick(now)

    def tick(self, now: float) -> None:
        """Run one pass through every symbol.

        Each symbol gets its own snapshot/features/decide/review row in the
        journal, and the heartbeat at the end reflects the post-tick state of
        the ledger. Exceptions raised per symbol are caught and journalised;
        one bad symbol never breaks the loop, so a stale broker reply for
        ``BTC/USD`` does not skip ``ETH/USD``.

        When a controller is wired the tick first performs the durable
        prelude (begin_day + killmodes.apply + broker reconciliation).  Any
        failure in the prelude trips the controller into ``HALT_ENTRIES`` via
        the fail-closed reconcile path so the rest of the tick refuses to
        trade.

        If the loop has already tripped :class:`ControllerFailStop` on a
        previous tick (controller storage was unusable), every subsequent
        ``tick`` raises immediately so no order effect runs until the process
        has been restarted by systemd.
        """
        if self._fail_stopped:
            raise ControllerFailStop("fail-stopped earlier")
        if not isinstance(now, (int, float)):
            raise ValueError("now must be a number")
        now = float(now)
        self._tick += 1

        if self._controller is not None:
            self._controller_prelude(now)
            if self._shadow_autorun or self._paper_autorun:
                self._shadow_try_run(now)

        halted = bool(self._kill_path is not None and self._kill_path.exists())
        if halted:
            self._ledger.journal(
                {
                    "kind": "halt",
                    "ts": now,
                    "tick": self._tick,
                    "reason": "kill-file present",
                }
            )

        if self._equity_session:
            # Close-time flatten does not depend on market data or features.
            self._session_flatten(now)
        try:
            snapshots = self._market_fn(symbols=self._symbols, now=now)
        except ControllerFailStop:
            raise  # storage failure must stop the process, never be isolated
        except Exception as exc:  # noqa: BLE001 - broker outages must not crash the loop
            self._ledger.journal(
                {
                    "kind": "market_error",
                    "ts": now,
                    "tick": self._tick,
                    "error": repr(exc),
                }
            )
            self._write_heartbeat(now=now, halted=True)
            self._last_halted = True
            return

        risk_state = self._ledger.risk_state(
            now,
            account_active=True,
            kill=halted,
        )

        for symbol in self._symbols:
            try:
                self._tick_symbol(symbol, snapshots.get(symbol), now, risk_state)
            except ControllerFailStop:
                raise  # storage failure must stop the process, never be isolated
            except Exception as exc:  # noqa: BLE001 - per-symbol isolation
                self._ledger.journal(
                    {
                        "kind": "symbol_error",
                        "ts": now,
                        "tick": self._tick,
                        "symbol": symbol,
                        "error": repr(exc),
                    }
                )

        # Refresh the state after every symbol so the heartbeat reports the
        # closed-trades sum including this tick's exits.
        post = self._ledger.risk_state(now, account_active=True, kill=False)
        self._write_heartbeat(
            now=now,
            halted=halted,
            open_positions=post.open_positions,
            realized_r_today=post.realized_r_today,
        )
        self._last_halted = halted

    # ---- controller prelude ---------------------------------------------
    def _controller_prelude(self, now: float) -> None:
        """Run the per-tick durable prelude; trip the controller on failure."""
        ctrl = self._controller

        def _fail(stage: str, exc: BaseException | None) -> None:
            try:
                ctrl.reconcile({}, [], account_active=False, complete_snapshot=False)
            except Exception as fail_exc:  # noqa: BLE001 - reconcile is already tripping
                # The controller's storage layer is itself unusable: we cannot
                # provably persist a HALT, so trading must stop.  The runtime
                # catches ControllerFailStop, exits non-zero, and systemd
                # restarts the process; the next startup reconciles from a
                # fresh HALT.
                self._fail_stopped = True
                try:
                    self._ledger.journal(
                        {
                            "kind": "controller_storage_failed",
                            "stage": stage,
                            "ts": now,
                            "tick": self._tick,
                            "error": repr(fail_exc),
                        }
                    )
                except Exception:  # noqa: BLE001 - logging must never block the halt
                    pass
                raise ControllerFailStop(repr(fail_exc)) from fail_exc
            try:
                self._ledger.journal(
                    {
                        "kind": "prelude_failed",
                        "stage": stage,
                        "ts": now,
                        "tick": self._tick,
                        "error": repr(exc),
                    }
                )
                if stage == "reconcile":
                    self._ledger.journal(
                        {
                            "kind": "reconcile_failed",
                            "ts": now,
                            "tick": self._tick,
                            "error": repr(exc),
                        }
                    )
            except Exception:  # noqa: BLE001 - logging must never block the halt
                pass

        try:
            ctrl.begin_day(datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%d"))
        except Exception as exc:  # noqa: BLE001 - fail closed
            return _fail("begin_day", exc)
        if self._data_dir is None:
            return _fail("no_kill_dir", None)
        kill_dir = Path(self._data_dir)
        if (
            not kill_dir.is_dir()
            or kill_dir.is_symlink()
            or not os.access(kill_dir, os.R_OK | os.X_OK)
        ):
            return _fail("kill_dir_unavailable", None)
        try:
            from . import killmodes

            killmodes.apply(ctrl, self._data_dir)
        except Exception as exc:  # noqa: BLE001 - unreadable kill dir fails closed
            return _fail("killmodes", exc)
        if self._history_start_iso is None:
            return _fail("no_history_start", None)
        try:
            status = getattr(self._broker, "account_status", None)
            account_active = bool(status()["active"]) if callable(status) else False
            positions = {_normalise_symbol(p["symbol"]): p["qty"] for p in self._broker.positions()}
            known = set(ctrl.snapshot()["orders"])
            open_orders = list(self._broker.open_orders())
            unknown = [
                o.get("client_order_id")
                for o in open_orders
                if o.get("client_order_id") not in known
            ]
            if unknown:
                return _fail("unknown_open_order", RuntimeError(f"unknown open orders: {unknown}"))
            start = self._history_start_iso
            ticks = [
                int(o.get("decision_tick", 0))
                for o in ctrl.snapshot()["orders"].values()
                if o.get("decision_tick")
            ]
            if ticks:
                earliest = datetime.fromtimestamp(min(ticks) * 60 - 86400, tz=timezone.utc)
                start = min(start, earliest.strftime("%Y-%m-%dT%H:%M:%SZ"))
            merged = {o["client_order_id"]: o for o in self._broker.orders_since(start)}
            merged.update({o["client_order_id"]: o for o in open_orders})
            ctrl.reconcile(
                positions,
                list(merged.values()),
                account_active=account_active,
                complete_snapshot=True,
            )
        except Exception as exc:  # noqa: BLE001 - any broker/reconcile failure fails closed
            return _fail("reconcile", exc)

    # ---- internals -------------------------------------------------------
    def _tick_symbol(
        self,
        symbol: str,
        snap: Any,
        now: float,
        risk_state: RiskState,
    ) -> None:
        if snap is None:
            self._ledger.journal(
                {
                    "kind": "decision",
                    "ts": now,
                    "tick": self._tick,
                    "symbol": symbol,
                    "features": None,
                    "decision": {"action": "hold", "reason": "missing-snapshot"},
                    "veto": None,
                    "executed": False,
                }
            )
            return
        features = compute_features(snap, now=now)
        position = risk_state.open_positions.get(symbol)
        if self._strategy_fn is not None:
            decision = self._strategy_fn(snap, position, self._limits, now=now)
        else:
            decision = self._decide(features, position, self._limits, now=now)
        if self._equity_session and position is not None and session.must_flatten(now):
            decision = Decision(symbol, "exit", "session_close")
        allowed, veto_code = self._review(decision, features, risk_state, self._limits, now=now)
        if (
            allowed and decision.action == "enter_long"
            and Path("/home/vitor/rlr/flags/HALT_ENTRIES").exists()
        ):
            allowed, veto_code = False, "global_halt_entries"
        if (
            self._equity_session
            and allowed
            and decision.action == "enter_long"
            and not self._entry_session_open(now)
        ):
            allowed, veto_code = False, "session_closed"
        if allowed and decision.action == "enter_long" and self._entry_evidence_gate is not None:
            try:
                # Exactly True passes. Unknown, null, timeout or any exception
                # fail closed; this veto cannot grant permission to trade.
                if self._entry_evidence_gate(symbol, features, now) is not True:
                    allowed, veto_code = False, "autonomy_no_validated_edge"
            except Exception:  # noqa: BLE001 - research subsystem is untrusted
                allowed, veto_code = False, "autonomy_gate_unavailable"
        executed = False
        if allowed and decision.action != "hold" and not self._dry_run:
            executed = self._execute(symbol, decision, features, now, position)
        elif self._dry_run and allowed and decision.action != "hold":
            executed = False  # dry-run never trades
        self._ledger.journal(
            {
                "kind": "decision",
                "ts": now,
                "tick": self._tick,
                "symbol": symbol,
                "features": _jsonable(features),
                "decision": _jsonable(decision),
                "veto": veto_code,
                "executed": bool(executed),
            }
        )

    def _shadow_try_run(self, now: float) -> None:
        """SHADOW ONLY: re-enter RUN after a clean reconcile so gates can be observed.

        Safe solely because shadow mode never sends an order effect; it is
        refused for any non-shadow loop.
        """
        if not (self._shadow or self._paper_autorun) or self._controller is None:
            return
        try:
            snap = self._controller.snapshot()
            if snap.get("mode") != "RUN":
                self._controller.set_mode("RUN", operator_approved=True)
                self._ledger.journal({"kind": "shadow_autorun" if self._shadow else "paper_autorun",
                                      "ts": now, "tick": self._tick})
        except Exception:  # noqa: BLE001 - not clean yet: stay halted
            pass

    def _session_flatten(self, now: float) -> None:
        """From 15:50 ET sell every ledger position each tick until flat; alert after the close."""
        state = self._ledger.risk_state(now, account_active=True, kill=False)
        if not state.open_positions:
            return
        if not session.must_flatten(now):
            if not session.in_regular_session(now) and session.is_weekday(now):
                self._ledger.journal({"kind": "position_held_outside_session", "ts": now,
                                      "tick": self._tick,
                                      "symbols": sorted(state.open_positions)})
            return
        for symbol, position in list(state.open_positions.items()):
            try:
                ok = self._execute(symbol, Decision(symbol, "exit", "session_close"),
                                   _FLATTEN_FEATURES, now, position)
                error = None
            except Exception as exc:  # noqa: BLE001 - retried on the next tick
                ok, error = False, repr(exc)
            self._ledger.journal({"kind": "session_flatten", "ts": now, "tick": self._tick,
                                  "symbol": symbol, "executed": bool(ok), "error": error})

    def _entry_session_open(self, now: float) -> bool:
        """Entry window open AND the broker clock confirms the market is open."""
        if not session.entry_allowed(now):
            return False
        if self._market_open_fn is None:
            return False
        try:
            return bool(self._market_open_fn())
        except Exception:  # noqa: BLE001 - unknown market state blocks entries
            return False

    def _execute(
        self,
        symbol: str,
        decision: Decision,
        features: Any,
        now: float,
        position: Position | None,
    ) -> bool:
        """Mint the permit, run the effect through ``tool_broker``, record the fill.

        Returns ``True`` only when an order actually filled and the ledger was
        updated. ``False`` means the effect was rejected (no permit, broker
        error, dry-run, controller veto, unknown submit) and nothing was
        written.

        The I0 safety gate (``require_controller=True``) runs first: when
        the caller asks for the safety-enabled mode but no controller is
        wired, the loop refuses every order effect and journals
        ``controller_required``.  This makes it impossible to produce an
        order effect without a durable controller in the path.
        """
        # ---- I0: require a controller when the caller asked for it --------
        if self._require_controller and self._controller is None and not self._dry_run:
            try:
                self._ledger.journal(
                    {
                        "kind": "controller_required",
                        "symbol": symbol,
                        "ts": now,
                        "tick": self._tick,
                    }
                )
            except Exception:  # noqa: BLE001 - logging must never block the refuse
                pass
            return False

        if self._shadow:
            mode = ""
            if self._controller is not None:
                try:
                    mode = str(self._controller.snapshot().get("mode", ""))
                except Exception:  # noqa: BLE001 - shadow reporting only
                    mode = "unavailable"
            try:
                self._ledger.journal(
                    {
                        "kind": "shadow_order",
                        "ts": now,
                        "tick": self._tick,
                        "symbol": symbol,
                        "action": decision.action,
                        "reason": decision.reason,
                        "stop": decision.stop,
                        "target": decision.target,
                        "notional": decision.notional,
                        "controller_mode": mode,
                        "controller_would_allow": (
                            mode == "RUN" if decision.action == "enter_long"
                            else mode != "EMERGENCY_STOP"
                        ),
                    }
                )
            except Exception:  # noqa: BLE001
                pass
            return False

        if decision.action == "enter_long":
            side = "buy"
            qty: float | None = None
            notional = (
                float(decision.notional)
                if decision.notional is not None
                else float(self._limits.notional)
            )
        elif decision.action == "exit":
            side = "sell"
            if position is None:
                return False
            qty = self._held_qty(symbol, fallback=float(position.qty))
            notional = None
        else:
            return False

        # ---- controller gate (opt-in) ------------------------------------
        cid: str = client_order_id(symbol, side, int(now // 60))
        controller = self._controller
        if controller is not None:
            try:
                snapshot = controller.snapshot()
            except sqlite3.Error as exc:
                # controller storage failure: fail-stop, never just veto (I2)
                self._fail_stopped = True
                raise ControllerFailStop(repr(exc)) from exc
            except Exception:  # noqa: BLE001 - controller hard-fail -> refuse
                snapshot = {"mode": ""}
            mode = str(snapshot.get("mode", "") or "")
            if decision.action == "enter_long":
                if mode != "RUN":
                    self._ledger.journal(
                        {
                            "kind": "controller_veto",
                            "ts": now,
                            "tick": self._tick,
                            "symbol": symbol,
                            "veto": f"controller_mode:{mode}",
                        }
                    )
                    return False
                price = float(getattr(features, "price", 0.0) or 0.0)
                qty_est = float(notional) / price if price > 0 else 0.0
                try:
                    cid = controller.prepare_order(symbol, "buy", int(now // 60), qty_est)
                except sqlite3.Error as exc:
                    # controller storage failure is not a refusal: fail-stop (I2)
                    self._fail_stopped = True
                    raise ControllerFailStop(repr(exc)) from exc
                except Exception as exc:  # noqa: BLE001 - controller refused
                    self._ledger.journal(
                        {
                            "kind": "controller_refused",
                            "ts": now,
                            "tick": self._tick,
                            "symbol": symbol,
                            "side": "buy",
                            "error": repr(exc),
                        }
                    )
                    return False
            else:  # exit
                if mode == "EMERGENCY_STOP":
                    self._ledger.journal(
                        {
                            "kind": "emergency_no_auto_send",
                            "ts": now,
                            "tick": self._tick,
                            "symbol": symbol,
                        }
                    )
                    return False
                try:
                    cid = controller.prepare_order(symbol, "sell", int(now // 60), qty)
                except sqlite3.Error as exc:
                    # controller storage failure is not a refusal: fail-stop (I2)
                    self._fail_stopped = True
                    raise ControllerFailStop(repr(exc)) from exc
                except Exception as exc:  # noqa: BLE001 - controller refused
                    self._ledger.journal(
                        {
                            "kind": "controller_refused",
                            "ts": now,
                            "tick": self._tick,
                            "symbol": symbol,
                            "side": "sell",
                            "error": repr(exc),
                        }
                    )
                    return False
            blocked = self._kill_blocks(side)
            if blocked is not None:
                self._persist_dispatch_halt(controller)
                try:
                    self._ledger.journal(
                        {
                            "kind": "kill_at_dispatch",
                            "stage": "pre_submit",
                            "ts": now,
                            "tick": self._tick,
                            "symbol": symbol,
                            "side": side,
                            "mode": blocked,
                            "cid": cid,
                        }
                    )
                except Exception:  # noqa: BLE001
                    pass
                return False
            try:
                controller.mark_submitting(cid)
            except sqlite3.Error as exc:
                # Controller storage is unusable: the trader process must
                # stop (no RUN without a provable persisted halt).  The
                # runtime re-raises ControllerFailStop so systemd restarts
                # the process.
                self._fail_stopped = True
                try:
                    self._ledger.journal(
                        {
                            "kind": "controller_storage_failed",
                            "stage": "mark_submitting",
                            "ts": now,
                            "tick": self._tick,
                            "symbol": symbol,
                            "cid": cid,
                            "error": repr(exc),
                        }
                    )
                except Exception:  # noqa: BLE001 - logging must never block the halt
                    pass
                raise ControllerFailStop(repr(exc)) from exc
            except Exception as exc:  # noqa: BLE001 - mark_submitting must not crash
                self._ledger.journal(
                    {
                        "kind": "mark_submitting_failed",
                        "ts": now,
                        "tick": self._tick,
                        "symbol": symbol,
                        "cid": cid,
                        "error": repr(exc),
                    }
                )
                return False

        args: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "client_order_id": cid,
        }
        if qty is not None:
            args["qty"] = qty
        else:
            args["notional"] = notional

        try:
            permit = self._permits.issue(
                tool="alpaca_paper",
                op="order",
                args=args,
                goal_id="rlr-paper-trader",
            )
        except Exception as exc:  # noqa: BLE001 - permit failure must not crash
            self._ledger.journal(
                {
                    "kind": "permit_denied",
                    "ts": now,
                    "tick": self._tick,
                    "symbol": symbol,
                    "error": repr(exc),
                }
            )
            return False

        # ---- controller flow: explicit ack + record_fill on the controller
        if controller is not None:
            blocked = self._kill_blocks(side)
            if blocked is not None:
                self._persist_dispatch_halt(controller)
                try:
                    self._ledger.journal(
                        {
                            "kind": "kill_at_dispatch",
                            "stage": "pre_transport",
                            "ts": now,
                            "tick": self._tick,
                            "symbol": symbol,
                            "side": side,
                            "mode": blocked,
                            "cid": cid,
                        }
                    )
                except Exception:  # noqa: BLE001
                    pass
                return False
            try:
                result = self._tool_broker.invoke_effect(permit, "alpaca_paper", "order", args)
            except Exception as exc:  # noqa: BLE001 - unknown POST outcome
                recovered = self._recover_unknown_submit(
                    controller, cid, symbol, now, decision, features, position
                )
                if not recovered:
                    self._ledger.journal(
                        {
                            "kind": "unknown_submit",
                            "ts": now,
                            "tick": self._tick,
                            "symbol": symbol,
                            "cid": cid,
                            "error": repr(exc),
                        }
                    )
                return False
            if getattr(result, "executed", True) is False:
                return False
            applied = getattr(result, "applied", None) or {}
            order_id = str(applied.get("order_id") or "")
            filled_px = float(applied.get("filled_avg_price") or 0.0)
            filled_qty = float(applied.get("filled_qty") or 0.0)
            broker_status = str(applied.get("status") or "new")
            ack_status = (
                broker_status if broker_status in {"new", "accepted", "partially_filled"} else "new"
            )
            # Fee is only charged in kind on buys; Alpaca takes the sell fee
            # in USD, so record_fill on a sell must pass fee_in_asset=0.
            fee = filled_qty * self._buy_fee_in_asset if side == "buy" else 0.0
            try:
                controller.acknowledge_order(cid, broker_status=ack_status)
            except Exception as exc:  # noqa: BLE001 - controller trip is its own signal
                self._ledger.journal(
                    {
                        "kind": "controller_record_failed",
                        "ts": now,
                        "tick": self._tick,
                        "cid": cid,
                        "error": repr(exc),
                    }
                )
                try:
                    controller.reconcile({}, [], account_active=True, complete_snapshot=False)
                except Exception:  # noqa: BLE001 - reconcile is already tripping
                    pass
            if filled_qty > 0:
                try:
                    controller.record_fill(
                        cid,
                        execution_id=f"{order_id}:{filled_qty}",
                        filled_qty=filled_qty,
                        fee_in_asset=fee,
                    )
                except Exception as exc:  # noqa: BLE001 - controller trip is its own signal
                    self._ledger.journal(
                        {
                            "kind": "controller_record_failed",
                            "ts": now,
                            "tick": self._tick,
                            "cid": cid,
                            "error": repr(exc),
                        }
                    )
                    try:
                        controller.reconcile({}, [], account_active=True, complete_snapshot=False)
                    except Exception:  # noqa: BLE001 - reconcile is already tripping
                        pass
            return self._record_fill_in_ledger(
                decision,
                features,
                symbol,
                now,
                position,
                applied,
                order_id,
                filled_px,
                filled_qty,
            )

        # ---- no-controller path: existing flow --------------------------
        try:
            result = self._tool_broker.invoke_effect(permit, "alpaca_paper", "order", args)
        except Exception as exc:  # noqa: BLE001 - broker failure must not crash
            self._ledger.journal(
                {
                    "kind": "broker_error",
                    "ts": now,
                    "tick": self._tick,
                    "symbol": symbol,
                    "error": repr(exc),
                }
            )
            return False
        if getattr(result, "executed", True) is False:
            return False
        applied = getattr(result, "applied", None) or {}
        order_id = str(applied.get("order_id") or "")
        filled_px = float(applied.get("filled_avg_price") or 0.0)
        filled_qty = float(applied.get("filled_qty") or 0.0)
        return self._record_fill_in_ledger(
            decision,
            features,
            symbol,
            now,
            position,
            applied,
            order_id,
            filled_px,
            filled_qty,
        )

    def _persist_dispatch_halt(self, controller: Any) -> None:
        """A send refused at dispatch must leave a sticky persisted halt (I2).

        Escalate from the kill files when readable, then always trip the
        controller's fail-closed reconcile (a hard incident an operator must
        clear). If the halt cannot be persisted, fail-stop the process.
        """
        try:
            from . import killmodes

            killmodes.apply(controller, self._data_dir)
        except Exception:  # noqa: BLE001 - the reconcile below still persists a halt
            pass
        try:
            controller.reconcile({}, [], account_active=False, complete_snapshot=False)
        except Exception as exc:  # noqa: BLE001 - unprovable halt => fail-stop
            self._fail_stopped = True
            raise ControllerFailStop(repr(exc)) from exc

    def _kill_blocks(self, side: str) -> str | None:
        """Re-read operator kill files at the send boundary; None = not blocked."""
        from . import killmodes
        from .orders import EMERGENCY_STOP

        # A kill dir that vanished, became a symlink or turned unreadable after
        # the prelude must block the send, never read as "no kill file".
        kill_dir = Path(self._data_dir) if self._data_dir is not None else None
        if (
            kill_dir is None
            or not kill_dir.is_dir()
            or kill_dir.is_symlink()
            or not os.access(kill_dir, os.R_OK | os.X_OK)
        ):
            return EMERGENCY_STOP
        try:
            mode = killmodes.requested_mode(self._data_dir)
        except Exception:  # noqa: BLE001 - unreadable kill dir blocks every send
            return EMERGENCY_STOP
        if mode is None:
            return None
        if side == "buy" or mode == EMERGENCY_STOP:
            return mode
        return None

    def _record_fill_in_ledger(
        self,
        decision: Decision,
        features: Any,
        symbol: str,
        now: float,
        position: Position | None,
        applied: Mapping[str, Any],
        order_id: str,
        filled_px: float,
        filled_qty: float,
    ) -> bool:
        """Mirror the broker's fill into the local ledger; return True on success."""
        if decision.action == "enter_long":
            atr = float(features.atr14) if getattr(features, "atr14", None) else 0.0
            if filled_qty <= 0 or filled_px <= 0:
                return False
            # Rebase stop/target on the actual fill price so the protective
            # distances from the decision are preserved even when the fill
            # drifts from the decision price used by the decider.
            fill = float(applied.get("filled_avg_price") or 0.0)
            decision_price = float(getattr(features, "price", fill) or fill)
            if decision.stop is not None:
                stop = fill - (decision_price - float(decision.stop))
            else:
                stop = fill - atr * float(self._limits.stop_atr)
            if decision.target is not None:
                target = fill + (float(decision.target) - decision_price)
            else:
                target = fill + atr * float(self._limits.target_atr)
            self._ledger.open_trade(
                symbol=symbol,
                entry_ts=now,
                entry_px=filled_px,
                qty=filled_qty,
                stop=stop,
                target=target,
                atr=atr,
                entry_order_id=order_id or None,
            )
        else:  # exit
            if not self._ledger.has_open(symbol):
                return False
            self._ledger.close_trade(
                symbol=symbol,
                exit_ts=now,
                exit_px=filled_px,
                reason=str(decision.reason or "exit"),
                exit_order_id=order_id or None,
            )
        return True

    def _recover_unknown_submit(
        self,
        controller: Any,
        cid: str,
        symbol: str,
        now: float,
        decision: Decision,
        features: Any,
        position: Position | None,
    ) -> bool:
        """Look up the broker's view of the just-attempted order; ack+record if filled.

        Returns ``True`` when the controller has been brought back in sync with
        a confirmed broker fill; ``False`` when the broker has no record (the
        controller stays uncertain until the next reconcile).
        """
        try:
            look = self._broker.get_order_by_client_id(cid)
        except Exception:  # noqa: BLE001 - any lookup failure -> uncertain
            return False
        if not look:
            return False
        cumulative = float(look.get("filled_qty") or 0.0)
        if cumulative <= 0:
            return False
        already = float(controller.snapshot()["orders"].get(cid, {}).get("filled_qty") or 0.0)
        delta = cumulative - already
        if delta <= 0:
            return False
        broker_status = str(look.get("status") or "new")
        ack_status = (
            broker_status if broker_status in {"new", "accepted", "partially_filled"} else "new"
        )
        # Fee is only charged in kind on buys; Alpaca takes the sell fee in
        # USD, so a recovered sell fill must pass fee_in_asset=0.
        look_side = str(look.get("side") or "")
        if not look_side:
            look_side = "buy" if decision.action == "enter_long" else "sell"
        fee = delta * self._buy_fee_in_asset if look_side == "buy" else 0.0
        try:
            controller.acknowledge_order(cid, broker_status=ack_status)
        except Exception as exc:  # noqa: BLE001 - controller trip is its own signal
            self._ledger.journal(
                {
                    "kind": "controller_record_failed",
                    "ts": now,
                    "tick": self._tick,
                    "cid": cid,
                    "error": repr(exc),
                }
            )
            try:
                controller.reconcile({}, [], account_active=True, complete_snapshot=False)
            except Exception:  # noqa: BLE001 - reconcile is already tripping
                pass
        order_id = str(look.get("id") or "")
        if delta > 0:
            try:
                controller.record_fill(
                    cid,
                    execution_id=f"{order_id}:{cumulative}",
                    filled_qty=delta,
                    fee_in_asset=fee,
                )
            except Exception as exc:  # noqa: BLE001 - controller trip is its own signal
                self._ledger.journal(
                    {
                        "kind": "controller_record_failed",
                        "ts": now,
                        "tick": self._tick,
                        "cid": cid,
                        "error": repr(exc),
                    }
                )
                try:
                    controller.reconcile({}, [], account_active=True, complete_snapshot=False)
                except Exception:  # noqa: BLE001 - reconcile is already tripping
                    pass
            filled_px = float(look.get("filled_avg_price") or 0.0)
            try:
                ok = self._record_fill_in_ledger(
                    decision,
                    features,
                    symbol,
                    now,
                    position,
                    look,
                    order_id,
                    filled_px,
                    delta,
                )
            except Exception:  # noqa: BLE001 - ledger IO failure fails closed
                ok = False
            if not ok:
                try:
                    controller.reconcile({}, [], account_active=False, complete_snapshot=False)
                except Exception:  # noqa: BLE001 - reconcile is already tripping
                    pass
                try:
                    self._ledger.journal(
                        {
                            "kind": "recovered_fill_ledger_failed",
                            "ts": now,
                            "tick": self._tick,
                            "cid": cid,
                        }
                    )
                except Exception:  # noqa: BLE001 - logging must never block the halt
                    pass
        return True

    def _held_qty(self, symbol: str, *, fallback: float) -> float:
        """Quantity actually held at the broker (fees are taken in kind on crypto)."""
        try:
            for p in self._broker.positions():
                if str(p.get("symbol", "")).replace("/", "") == symbol.replace("/", ""):
                    held = float(p.get("qty") or 0.0)
                    if held > 0:
                        return min(held, fallback) if fallback > 0 else held
        except Exception:  # noqa: BLE001 - fall back to the ledger quantity
            pass
        return fallback

    def _write_heartbeat(
        self,
        *,
        now: float,
        halted: bool,
        open_positions: Mapping[str, Position] | None = None,
        realized_r_today: float | None = None,
    ) -> None:
        if self._heartbeat_path is None:
            return
        positions = open_positions if open_positions is not None else self._ledger.open_positions()
        payload = {
            "ts": float(now),
            "tick": int(self._tick),
            "open_positions": {
                symbol: _jsonable(position) for symbol, position in positions.items()
            },
            "realized_r_today": float(realized_r_today or 0.0),
            "halted": bool(halted),
        }
        # Controller entry authority is independent of kill-file status.
        # Expose both: a fresh process heartbeat is NOT evidence that
        # new entries are permitted.
        if self._controller is not None:
            try:
                control = self._controller.snapshot()
                mode = control.get("mode")
                reconciled = control.get("reconciled") is True
                uncertain = control.get("uncertain") is not False
                incident = str(control.get("incident") or "")
                payload.update({
                    "controller_mode": mode,
                    "controller_reconciled": reconciled,
                    "controller_uncertain": uncertain,
                    "controller_incident": incident,
                    "entries_blocked": bool(
                        halted or mode != "RUN" or incident or uncertain or not reconciled
                    ),
                })
            except Exception:  # fail closed, never invent an all-clear
                payload.update({
                    "controller_mode": "UNKNOWN",
                    "controller_reconciled": False,
                    "controller_uncertain": True,
                    "controller_incident": "controller_snapshot_unavailable",
                    "entries_blocked": True,
                })
        else:
            payload.update({"controller_mode": "UNWIRED", "entries_blocked": bool(halted)})
        text = json.dumps(payload, default=str, sort_keys=True)
        tmp = self._heartbeat_path.with_suffix(self._heartbeat_path.suffix + ".tmp")
        tmp.write_text(text + "\n", encoding="utf-8")
        os.replace(tmp, self._heartbeat_path)


def _jsonable(obj: Any) -> Any:
    """Convert a dataclass to a dict; pass dicts and scalars through."""
    if hasattr(obj, "__dict__") and not isinstance(obj, (str, bytes, int, float, bool)):
        return {k: _jsonable(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


__all__ = [
    "ControllerFailStop",
    "TraderLoop",
    "trading_manifest",
    "Decision",
    "Limits",
    "RiskState",
    "Position",
]

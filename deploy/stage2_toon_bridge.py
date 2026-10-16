"""Stage2 -> TOON v2 observer, strictly no broker methods or control authority.

Only enqueue(), which uses put_nowait(), is called by the legacy trading thread.
Worker owns the serializer, research model, and append-only audit. Its failures
never change any broker instructions. Shadow is not paper trading certification.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Full, Queue
import json
import os
import threading
import time
from typing import Any

from reflective_loop_runtime.quant.edge_jetson.runtime_opt import toon_encode
from reflective_loop_runtime.quant.execution.smart_router import Book, Level, SmartRouter, VenueCapabilities
from reflective_loop_runtime.quant.regime.market_state import Bar
from reflective_loop_runtime.quant.risk.kelly_firewall import OutcomeStats, RiskFirewall
from reflective_loop_runtime.quant.rl_core.venue_costs import get_profile
from reflective_loop_runtime.quant.runtime_v2 import MarketTick, RuntimeConfig, UnifiedQuantRuntime


class SafetyLockViolation(RuntimeError):
    """Never interpret an environment variable as live-trading authority."""


def _enabled(name: str, default: bool) -> bool:
    value = os.environ.get(name, 'true' if default else 'false').strip().lower()
    if value not in ('true', 'false', '1', '0'):
        raise SafetyLockViolation('invalid boolean for ' + name)
    return value in ('true', '1')


@dataclass(frozen=True)
class ShadowDecision:
    regime: str
    kelly_size: float
    target_limit_price: float | None
    synthetic_post_only: bool
    veto_reason: str
    toon_memory_payload: str
    broker_orders_submitted: int = 0


@dataclass(frozen=True)
class ComparisonAudit:
    stage2_action: str
    toon_v2_action: str
    agreed: bool
    fee_adjusted_edge_bps: float | None
    firewall_veto: bool


class Stage2ShadowInterceptor:
    """Bounded asynchronous observer; never reads credentials or imports broker API.

    The main-loop hook is a nonblocking queue offer, not a synchronous 10 ms
    inference call. Data may be dropped under overload, and the drop is counted.
    A Stage2 broker trade has to be independently verified before it can become
    certification evidence. This class CANNOT promote a strategy or unlock trading.
    """

    def __init__(self, *, ledger: Path, halt_file: Path,
                 enabled: bool = True, override_requested: bool = False,
                 max_queue: int = 64, audit_path: Path | None = None,
                 evaluator: Any = None):
        self.enabled = enabled
        self.halt_file = Path(halt_file)
        self.ledger = Path(ledger)
        self.override_requested = override_requested
        self._assert_configuration()
        if not 1 <= max_queue <= 4096:
            raise ValueError('invalid queue bound')
        self.audit_path = Path(audit_path) if audit_path else self.ledger / 'shadow_audit.jsonl'
        self._q: Queue[tuple[str, dict[str, Any]] | None] = Queue(max_queue)
        self._lock = threading.Lock()
        self._closed = False
        self.dropped = 0
        self.processed = 0
        self.exceptions = 0
        self.last_decision: ShadowDecision | None = None
        self._evaluator = evaluator
        # No writes until the worker starts, and no order API is injected.
        self._worker = threading.Thread(target=self._run, name='toon-v2-readonly-shadow', daemon=True)
        if enabled:
            self._worker.start()

    @classmethod
    def from_env(cls, *, ledger: Path, halt_file: Path) -> 'Stage2ShadowInterceptor':
        return cls(ledger=ledger, halt_file=halt_file,
                   enabled=_enabled('TOON_V2_SHADOW_ENABLED', True),
                   override_requested=_enabled('TOON_V2_OVERRIDE_STAGE2', False))

    def _assert_configuration(self) -> None:
        # Both the environment and persistent file represent a safety barrier.
        # In this version, override remains impossible EVEN if they disappear.
        if self.override_requested:
            raise SafetyLockViolation('override unsupported pending independently verified certification')

    def assert_halt_invariant(self) -> bool:
        self._assert_configuration()
        # An absent file must not make the observer trade-capable.
        return _enabled('HALT_ENTRIES', True) or self.halt_file.exists()

    def _offer(self, kind: str, payload: dict[str, Any]) -> bool:
        if not self.enabled or self._closed:
            return False
        try:
            self._q.put_nowait((kind, payload))
            return True
        except Full:
            with self._lock:
                self.dropped += 1
            return False

    def on_controller_tick(self, *, mode: str, timestamp_ms: int, transition_seq: int = 0) -> bool:
        return self._offer('controller_tick', {
            'mode': str(mode), 'timestamp_ms': int(timestamp_ms), 'transition_seq': int(transition_seq)})

    def on_market_snapshot(self, *, symbol: str, snapshot: Any, timestamp_ms: int,
                           position: dict[str, Any] | None = None,
                           expected_alpha_bps: float | None = None) -> bool:
        return self._offer('snapshot', {
            'symbol': str(symbol), 'snapshot': snapshot, 'timestamp_ms': int(timestamp_ms),
            'position': position or {}, 'expected_alpha_bps': expected_alpha_bps})

    def on_stage2_signal(self, *, action: str, qty: float = 0,
                         price: float = 0, timestamp_ms: int = 0) -> bool:
        return self._offer('legacy_signal', {
            'action': str(action).upper(), 'qty': float(qty), 'price': float(price),
            'timestamp_ms': int(timestamp_ms)})

    def on_trade_closed(self, *, pnl: float, slippage_bps: float, hold_seconds: float,
                        broker_order_id: str = '', source: str = 'UNVERIFIED') -> bool:
        # Real broker closes are audit observations, not certified samples until
        # externally proven. The worker does not increment the 30-trade gate.
        return self._offer('trade_close', {
            'pnl': float(pnl), 'slippage_bps': float(slippage_bps),
            'hold_seconds': float(hold_seconds), 'broker_order_id': str(broker_order_id),
            'source': str(source), 'counted_for_certification': False})

    def _decision(self, payload: dict[str, Any]) -> ShadowDecision:
        if self._evaluator is not None:
            return self._evaluator(payload)
        snap = payload['snapshot']
        quote = getattr(snap, 'quote', None)
        bars = getattr(snap, 'bars', None)
        if quote is None or not bars:
            raise ValueError('missing_real_quote_or_bars')
        # Stage2 L1 feed may omit level sizes: do not invent L2 information.
        bid_size = getattr(quote, 'bid_size', getattr(quote, 'bid_qty', None))
        ask_size = getattr(quote, 'ask_size', getattr(quote, 'ask_qty', None))
        if bid_size is None or ask_size is None:
            raise ValueError('missing_l2_depth_no_fabricated_obi')
        timestamp = payload['timestamp_ms']
        book = Book((Level(float(quote.bid),float(bid_size)),),
                    (Level(float(quote.ask),float(ask_size)),),
                    int(float(quote.ts)*1000), sequence=max(1,int(float(quote.ts)*1000)))
        # Only completed bars, no future-derived state; as-of is the hook clock.
        hist = tuple(Bar(int(float(b.ts)*1000), float(b.open),float(b.high),
                         float(b.low),float(b.close)) for b in bars
                     if (float(b.ts)+60)*1000<=timestamp)
        if len(hist)<50:
            raise ValueError('insufficient_completed_ohlc_history')
        # Fail closed without independently verified equity, fills, beta and
        # complete covariance: the risk module will veto all entry requests.
        router = SmartRouter(VenueCapabilities('alpaca_equity',False,'day',.01,.01,True))
        runtime = UnifiedQuantRuntime(RuntimeConfig(halt_entries=True,
                        halt_file=self.halt_file),router,RiskFirewall(),
                        OutcomeStats(0,0,1.,1.),get_profile('ALPACA_EQUITIES_UNCALIBRATED'),
                        (payload['symbol'],),((1.,),), {payload['symbol']:1.},
                        equity=0., opening_equity=0.)
        obs = runtime.on_tick(MarketTick(timestamp,hist,book,payload['symbol'],
            price_sigma_1s=0., measured_rtt_ms=0.),
            cycle_id=f'shadow-{timestamp}', expected_gross_alpha_bps=None)
        memory_wire = toon_encode({'W1':{'source':'stage2_verified_snapshot'},
             'W2':{'time_ms':timestamp,'regime':obs.regime,'risk':obs.risk_status}})
        return ShadowDecision(obs.regime,0.,None,True,
             obs.execution_status, memory_wire)

    def _append_audit(self, payload: dict[str, Any]) -> None:
        self.audit_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        # JSONL envelope with a canonical TOON scalar memory payload.
        line=json.dumps(payload,sort_keys=True,default=str,separators=(',',':'))+'\n'
        with self.audit_path.open('a', encoding='utf8') as f:
            f.write(line)

    def _run(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is None:
                    return
                kind,payload=item
                try:
                    if kind == 'snapshot':
                        decision=self._decision(payload)
                        self.last_decision=decision
                        audit={'kind':'shadow_snapshot','timestamp_ms':payload['timestamp_ms'],
                               'symbol':payload['symbol'],'regime':decision.regime,
                               'kelly_size':decision.kelly_size,'target_limit_price':decision.target_limit_price,
                               'veto_reason':decision.veto_reason,
                               'toon_memory_payload':decision.toon_memory_payload,
                               'broker_orders_submitted':0}
                    elif kind == 'legacy_signal':
                        decision=self.last_decision
                        action=payload['action']
                        shadow='HOLD' if decision is None or decision.veto_reason else 'BUY'
                        audit={'kind':'comparison','timestamp_ms':payload['timestamp_ms'],
                               'stage2_action':action,'toon_v2_action':shadow,
                               'agreed':action==shadow,'firewall_veto':shadow=='HOLD',
                               'fee_adjusted_edge_bps':None,'broker_orders_submitted':0}
                    elif kind == 'trade_close':
                        audit={'kind':'trade_close_observed',**payload,'broker_orders_submitted':0}
                    else:
                        audit={'kind':kind,**payload,'broker_orders_submitted':0}
                    self._append_audit(audit)
                except Exception as exc:
                    self.exceptions += 1
                    self.last_decision=ShadowDecision('UNSAFE',0.,None,False,
                        'shadow_error_'+type(exc).__name__,'',0)
                    try:self._append_audit({'kind':'shadow_exception','error_type':type(exc).__name__,
                            'detail':str(exc)[:160],'broker_orders_submitted':0})
                    except Exception:pass
                finally:
                    self.processed += 1
            finally:
                self._q.task_done()

    def drain(self, timeout_sec: float = 2.0) -> bool:
        """Test/operator method ONLY; never invoked by the Stage2 live loop."""
        end=time.monotonic()+timeout_sec
        while self._q.unfinished_tasks and time.monotonic()<end:
            time.sleep(0.005)
        return self._q.unfinished_tasks==0

    def close(self) -> None:
        if self._closed:
            return
        if self.enabled:
            self.drain()
            try:self._q.put_nowait(None)
            except Full:pass
            self._worker.join(timeout=1.)
        self._closed=True

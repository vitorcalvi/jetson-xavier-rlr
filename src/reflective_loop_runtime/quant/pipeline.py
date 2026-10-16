"""Offline-only proof that regime, risk, routing, fills and reward compose.

There is no network client, broker key, or method for placing a live order.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from .execution.smart_router import Book, SmartRouter, PassiveIntent, RouteRejected
from .risk.kelly_firewall import RiskFirewall, OutcomeStats, Veto
from .regime.market_state import Regime
from .rl_core.reward_and_env import QueueAhead, NetRewardEnv


@dataclass(frozen=True)
class Decision:
    permitted:bool
    reason:str
    intent:PassiveIntent|None=None


def plan_research_entry(*,regime:Regime,strategy:str,router:SmartRouter,
                        book:Book,symbol:str,side:str,quantity:float,now_ms:int,
                        cycle_id:str,firewall:RiskFirewall,equity:float,
                        opening_equity:float,existing_notional:dict[str,float],
                        stats:OutcomeStats,betas:dict[str,float],
                        covariance_names:tuple[str,...],
                        covariance:tuple[tuple[float,...],...],
                        now:datetime)->Decision:
    if not regime.permits(strategy):
        return Decision(False,'regime_veto')
    risk:Veto=firewall.evaluate(symbol=symbol,now=now,equity=equity,
                opening_equity=opening_equity,existing_notional=existing_notional,
                stats=stats,beta=betas,covariance_names=covariance_names,
                covariance=covariance)
    if not risk.permitted:return Decision(False,'risk_'+risk.reason)
    try:
        intent=router.plan(symbol=symbol,side=side,quantity=min(quantity,risk.notional/book.bids[0].price),
                           book=book,now_ms=now_ms,cycle_id=cycle_id)
    except RouteRejected as exc:
        return Decision(False,'execution_'+str(exc))
    return Decision(True,'RESEARCH_ONLY_NO_BROKER_CALL',intent)


def simulate_maker_fill(intent:PassiveIntent,queue:QueueAhead,
                        observed_aggressive_volume:float,env:NetRewardEnv,
                        *,mark_price:float,now_seconds:float)->dict[str,float]:
    """Research-only: consumes supplied actual volume at the resting price."""
    if queue.remaining>intent.quantity+1e-9:
        raise ValueError('queue larger than planned order')
    filled=queue.consume(observed_aggressive_volume)
    if filled>0:
        env.trade(side=intent.side,qty=filled,price=intent.limit,maker=True)
    step=env.step(mark_price=mark_price,now_seconds=now_seconds)
    return dict(step,filled_qty=filled)

"""Deterministic fill/reward environment; broker/network independent.

Fees and funding are already accounted for in net equity. Reward NEVER deducts
those same expenses twice; the risk-sensitive penalty acts on net returns.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from math import isfinite, sqrt
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .venue_costs import VenueCostProfile


@dataclass
class QueueAhead:
    """FIFO approximation: executions first reduce observed queue ahead.

    Caller must supply only actual aggressive fills at this resting price,
    not all market volume. Queue priority can be worse in real venues.
    """
    ahead:float
    remaining:float
    filled:float=0.0

    def __post_init__(self)->None:
        if not all(isfinite(v) and v>=0 for v in (self.ahead,self.remaining,self.filled)):
            raise ValueError('invalid queue')

    def consume(self, aggressor_volume:float)->float:
        if not isfinite(aggressor_volume) or aggressor_volume<0:
            raise ValueError('invalid traded volume')
        ahead_cut=min(self.ahead,aggressor_volume)
        self.ahead-=ahead_cut
        available=aggressor_volume-ahead_cut
        executed=min(available,self.remaining)
        self.remaining-=executed
        self.filled+=executed
        return executed


@dataclass(frozen=True)
class VenueCosts:
    maker_bps:float
    taker_bps:float
    half_spread_bps:float
    impact_bps_per_unit:float
    perpetual_funding:bool=False

    def __post_init__(self)->None:
        if any(not isfinite(x) or x<0 for x in (
            self.maker_bps,self.taker_bps,self.half_spread_bps,self.impact_bps_per_unit)):
            raise ValueError('negative/nonfinite cost')


@dataclass
class NetRewardEnv:
    initial_cash:float
    venue:VenueCosts
    ew_alpha:float=.05
    downside_alpha:float=.06
    drawdown_soft:float=.02
    downside_penalty:float=1.0
    drawdown_penalty:float=4.0
    cash:float=field(init=False)
    position:float=0.0
    fees:float=0.0
    funding:float=0.0
    gross_realized:float=0.0
    _last_equity:float=field(init=False)
    _peak:float=field(init=False)
    _mean:float=0.0
    _downside2:float=0.0
    _last_time:float|None=None
    fee_profile:'VenueCostProfile|None'=None
    margin_interest_total:float=0.0

    def __post_init__(self)->None:
        if not isfinite(self.initial_cash) or self.initial_cash<=0 or not 0<self.ew_alpha<=1 or not 0<self.downside_alpha<=1:
            raise ValueError('invalid environment')
        self.cash=self._last_equity=self._peak=self.initial_cash

    def trade(self, *, side:str,qty:float,price:float, maker:bool,
              participation:float=0)->float:
        """Apply an observed fill ONCE; never places an order. Returns total cost."""
        if side not in ('buy','sell') or not all(isfinite(x) for x in (qty,price,participation)) or qty<=0 or price<=0 or participation<0:
            raise ValueError('invalid trade')
        signed=qty if side=='buy' else -qty
        slippage_bps=(0 if maker else self.venue.half_spread_bps)+self.venue.impact_bps_per_unit*participation
        execution=price*(1+(slippage_bps if side=='buy' else -slippage_bps)/10000)
        fee=(self.fee_profile.executed_fees(side=side,qty=qty,price=execution,maker=maker)
             if self.fee_profile is not None else
             qty*execution*(self.venue.maker_bps if maker else self.venue.taker_bps)/10000)
        self.cash-=signed*execution+fee
        self.position+=signed
        self.fees+=fee
        return fee + abs(qty*(execution-price))

    def step(self, *, mark_price:float, now_seconds:float,
             funding_rate_8h:float|None=None,margin_carry_hours:float=0.0)->dict[str,float]:
        if not all(isfinite(x) and x>0 for x in (mark_price,)) or not isfinite(now_seconds):
            raise ValueError('bad mark/clock')
        if self._last_time is not None and now_seconds<self._last_time:
            raise ValueError('time moved backward')
        if funding_rate_8h is not None and (not self.venue.perpetual_funding or not isfinite(funding_rate_8h)):
            raise ValueError('funding not supported or nonfinite')
        if not isfinite(margin_carry_hours) or margin_carry_hours<0:
            raise ValueError('invalid margin carry')
        if margin_carry_hours and (self.fee_profile is None or self.fee_profile.asset_class!='equity'):
            raise ValueError('margin carry requires calibrated equity profile')
        dt=0 if self._last_time is None else now_seconds-self._last_time
        self._last_time=now_seconds
        # Continuous-rate approximation of discrete 8h funding; snapshot rate
        # must cover the elapsed interval. No funding on cash equities/spot.
        if self.venue.perpetual_funding:
            if funding_rate_8h is None and dt>0:
                raise ValueError('missing derivatives funding observation')
            charge=self.position*mark_price*(funding_rate_8h or 0)*dt/(8*3600)
            self.cash-=charge
            self.funding+=charge
        if margin_carry_hours and self.fee_profile is not None:
            borrow=max(0,-self.cash)
            charge=self.fee_profile.interest(notional_borrowed=borrow,hours=margin_carry_hours)
            self.cash-=charge
            self.margin_interest_total+=charge
        equity=self.cash+self.position*mark_price
        if not isfinite(equity) or equity<=0:raise ValueError('insolvent or invalid mark')
        prior=self._last_equity
        net_return=(equity-prior)/prior
        self._mean=(1-self.ew_alpha)*self._mean+self.ew_alpha*net_return
        downside=min(net_return,0.0)**2
        self._downside2=(1-self.downside_alpha)*self._downside2+self.downside_alpha*downside
        self._peak=max(self._peak,equity)
        dd=max(0,1-equity/self._peak)
        # Differential net equity: already includes slippage, fees, funding.
        # Avoid subtracting fees again, or apparent performance gets distorted.
        net_delta=equity-prior
        reward=net_return-self.downside_penalty*downside-self.drawdown_penalty*max(0,dd-self.drawdown_soft)**2
        self._last_equity=equity
        return {'equity':equity,'net_pnl':equity-self.initial_cash,
                'net_delta':net_delta,'net_return':net_return,'reward':reward,
                'ew_downside_sortino':self._mean/sqrt(max(self._downside2,1e-12)),
                'drawdown':dd,'fees_total':self.fees,'funding_total':self.funding,
                'margin_interest_total':self.margin_interest_total}

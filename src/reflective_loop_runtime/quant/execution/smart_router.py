"""Broker-agnostic, order-free maker routing plans with fail-closed validation."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from math import ceil, exp, floor, isfinite, sqrt
from typing import Literal


class RouteRejected(ValueError):
    """The router refuses to construct an unsafe order."""


@dataclass(frozen=True)
class Level:
    price: float
    qty: float

    def __post_init__(self) -> None:
        if not all(isfinite(x) and x > 0 for x in (self.price, self.qty)):
            raise RouteRejected("invalid level")


@dataclass(frozen=True)
class Book:
    bids: tuple[Level, ...]
    asks: tuple[Level, ...]
    timestamp_ms: int
    sequence: int

    def __post_init__(self) -> None:
        if not self.bids or not self.asks or len(self.bids) > 50 or len(self.asks) > 50:
            raise RouteRejected("empty or unbounded book")
        if self.sequence < 0 or self.timestamp_ms <= 0:
            raise RouteRejected("invalid book metadata")
        if self.bids[0].price >= self.asks[0].price:
            raise RouteRejected("crossed/locked book")
        if any(a.price <= b.price for a,b in zip(self.bids, self.bids[1:])):
            raise RouteRejected("unordered bid levels")
        if any(a.price >= b.price for a,b in zip(self.asks, self.asks[1:])):
            raise RouteRejected("unordered ask levels")

    def microprice(self) -> float:
        b,a = self.bids[0], self.asks[0]
        return (b.qty*a.price + a.qty*b.price)/(b.qty+a.qty)

    def imbalance(self, depth:int=5, decay:float=.5) -> float:
        if not 1 <= depth <= 50 or not 0 <= decay <= 10:
            raise RouteRejected("invalid depth")
        b = sum(v.qty*exp(-decay*i) for i,v in enumerate(self.bids[:depth]))
        a = sum(v.qty*exp(-decay*i) for i,v in enumerate(self.asks[:depth]))
        return (b-a)/(b+a)


@dataclass(frozen=True)
class VenueCapabilities:
    name: str
    native_post_only: bool
    tif: Literal['gtx','day','gtc']
    tick: float
    lot: float
    terminal_cancel_readback: bool

    def __post_init__(self) -> None:
        if not self.name or not isfinite(self.tick) or not isfinite(self.lot) or min(self.tick,self.lot) <= 0:
            raise RouteRejected('invalid venue')
        if self.native_post_only != (self.tif == 'gtx'):
            raise RouteRejected('GTX native-post-only mismatch')


@dataclass(frozen=True)
class PassiveIntent:
    symbol: str
    side: Literal['buy','sell']
    quantity: float
    limit: float
    client_order_id: str
    tif: str
    native_post_only: bool
    quote_sequence: int
    microprice: float
    obi: float
    synthetic_guard_bps: float = 0.0
    routing_mode: str = "NATIVE_POST_ONLY"


class SmartRouter:
    """Plans only: never submits, replaces, or cancels a broker order."""

    def __init__(self, venue:VenueCapabilities, max_age_ms:int=1000,
                 max_spread_bps:float=15, max_chases:int=2,
                 max_rtt_ms:float=500.0):
        if max_age_ms <= 0 or max_spread_bps <= 0 or max_chases < 0:
            raise ValueError('invalid router guards')
        if not isfinite(max_rtt_ms) or max_rtt_ms<=0:
            raise ValueError("invalid RTT limit")
        self.venue,self.max_age_ms,self.max_spread_bps,self.max_chases = (
            venue,max_age_ms,max_spread_bps,max_chases)
        self.max_rtt_ms=max_rtt_ms

    def plan(self, *, symbol:str, side:Literal['buy','sell'], quantity:float,
             book:Book, now_ms:int, cycle_id:str, chase:int=0,
             prior_sequence:int|None=None, cancel_terminal:bool=False,
             inventory_reconciled:bool=False, require_native_post_only:bool=True,
             rtt_ms:float|None=None, sigma_price_per_sqrt_second:float|None=None,
             target_price:float|None=None, observed_obi:float|None=None,
             prev_obi:float|None=None) -> PassiveIntent:
        if not symbol or len(symbol)>16 or not symbol.replace('/','').replace('.','').isalnum():
            raise RouteRejected('invalid symbol')
        if side not in ('buy','sell') or not isfinite(quantity) or quantity<=0 or not cycle_id:
            raise RouteRejected('invalid intent')
        if not self.venue.terminal_cancel_readback:
            raise RouteRejected('terminal broker cancel readback unavailable')
        if require_native_post_only and not self.venue.native_post_only:
            raise RouteRejected('broker does not support native post-only')
        if not 0 <= now_ms-book.timestamp_ms <= self.max_age_ms:
            raise RouteRejected('book stale or future-dated')
        if prior_sequence is not None and book.sequence <= prior_sequence:
            raise RouteRejected('non-monotonic book')
        if chase<0 or chase>self.max_chases:
            raise RouteRejected('chase budget')
        if chase and not (cancel_terminal and inventory_reconciled):
            raise RouteRejected('missing terminal cancellation/inventory proof')
        bid,ask = book.bids[0].price, book.asks[0].price
        if (ask-bid)/((ask+bid)/2)*10000>self.max_spread_bps:
            raise RouteRejected('spread too wide')
        tick=self.venue.tick
        if not self.venue.native_post_only and ask-bid < 2*tick-1e-9:
            raise RouteRejected('spread below two ticks')
        obi=book.imbalance()
        if observed_obi is not None:
            if not isfinite(observed_obi) or abs(observed_obi)>1:
                raise RouteRejected('invalid observed imbalance')
            obi=observed_obi
        if prev_obi is not None:
            if not isfinite(prev_obi) or abs(prev_obi)>1:
                raise RouteRejected('invalid previous imbalance')
            if (side=='buy' and prev_obi>=0 and obi<0) or (side=='sell' and prev_obi<=0 and obi>0):
                raise RouteRejected('adverse OBI sign flip')
        # Passive LIMIT is NOT exchange-enforced post-only. This computes a
        # buffer for the observed snapshot only; network-transit price moves
        # can still create taker executions. Do not claim zero taker fills.
        synthetic=not self.venue.native_post_only
        offset=tick
        if synthetic:
            if rtt_ms is None or sigma_price_per_sqrt_second is None:
                raise RouteRejected('missing measured RTT or 1s price volatility')
            if not isfinite(rtt_ms) or not 0<rtt_ms<=self.max_rtt_ms:
                raise RouteRejected('RTT unmeasured or out of range')
            if not isfinite(sigma_price_per_sqrt_second) or sigma_price_per_sqrt_second<0:
                raise RouteRejected('invalid volatility scale')
            # sigma is in *price units* per sqrt(second), never ticks or bps.
            ticks=ceil(sigma_price_per_sqrt_second*sqrt(rtt_ms/1000)/tick-1e-12)
            offset=max(tick,ticks*tick)
        desired=book.microprice() if target_price is None else target_price
        if not isfinite(desired) or desired<=0:
            raise RouteRejected('invalid target price')
        raw=min(desired,ask-offset) if side=='buy' else max(desired,bid+offset)
        limit=(floor((raw+1e-10)/tick) if side=='buy' else ceil((raw-1e-10)/tick))*tick
        if limit<=0 or (side=='buy' and limit>=ask) or (side=='sell' and limit<=bid):
            raise RouteRejected('would cross book')
        lots=floor((quantity+1e-12)/self.venue.lot)
        if lots<1:
            raise RouteRejected('below min lot')
        coid='q'+sha256(f'{symbol}|{side}|{cycle_id}|{chase}'.encode()).hexdigest()[:31]
        return PassiveIntent(symbol,side,round(lots*self.venue.lot,10),round(limit,8),coid,
                             self.venue.tif,self.venue.native_post_only,book.sequence,
                             book.microprice(),obi, offset/((ask+bid)/2)*10000 if synthetic else 0.0,
                             'SYNTHETIC_PASSIVE_NOT_GUARANTEED' if synthetic else 'NATIVE_POST_ONLY')


def twap_slices(quantity:float, slices:int, lot:float) -> tuple[float,...]:
    """Bounded offline slicing; not an instruction to a trading venue."""
    if not all(isfinite(x) and x>0 for x in (quantity,lot)) or not 1<=slices<=100:
        raise RouteRejected('invalid slice input')
    n=floor((quantity+1e-12)/lot)
    if n<slices:
        raise RouteRejected('insufficient lots')
    q,r=divmod(n,slices)
    return tuple(round(lot*(q+(i<r)),10) for i in range(slices))


def verify_oca_siblings(orders:tuple[dict,...], *, total_exit_qty:float) -> bool:
    """Checks a shared OCA group and that at most one sibling remains active after a fill."""
    if not orders or not isfinite(total_exit_qty) or total_exit_qty<=0:
        return False
    ids=[o.get('id') for o in orders]
    groups={o.get('oca_group') for o in orders}
    if any(not x for x in ids) or len(set(ids))!=len(ids) or len(groups)!=1 or None in groups:
        return False
    try:
        filled=sum(float(o.get('filled_qty') or 0) for o in orders)
        active=[o for o in orders if o.get('status') in ('new','accepted','partially_filled','pending_cancel')]
        if filled < 0 or filled > total_exit_qty+1e-8 or (filled>0 and active):
            return False
        if any(float(o.get('qty') or 0)<=0 for o in orders):
            return False
        return True
    except (ValueError,TypeError):
        return False

"""Auditable venue fee profiles; unknown rates fail closed, not silently zero.

Default crypto tier below is the published *lowest-volume* tier, not a fixed
account fee. Equities CAT/TAF/SEC fees and margin APR need up-to-date broker
inputs. A configurable perp example is not an exchange-verified tariff.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from .reward_and_env import VenueCosts


class UnknownFees(ValueError):
    pass


@dataclass(frozen=True)
class VenueCostProfile:
    name: str
    asset_class: str
    maker_bps: float
    taker_bps: float
    funding_enabled: bool
    funding_interval_seconds: int = 28800
    sec_sell_bps: float | None = None
    taf_sell_per_share: float | None = None
    cat_per_share: float | None = None
    margin_apr: float | None = None
    source: str = 'example_not_verified'

    def __post_init__(self) -> None:
        if not self.name or self.asset_class not in ('equity','spot_crypto','perpetual'):
            raise UnknownFees('invalid venue')
        if not all(isfinite(x) and x >= 0 for x in (self.maker_bps,self.taker_bps)):
            raise UnknownFees('invalid maker/taker rate')
        if self.funding_enabled != (self.asset_class == 'perpetual'):
            raise UnknownFees('funding venue mismatch')
        if self.funding_interval_seconds<=0:
            raise UnknownFees('invalid funding interval')
        for attr in ('sec_sell_bps','taf_sell_per_share','cat_per_share','margin_apr'):
            x=getattr(self,attr)
            if x is not None and (not isfinite(x) or x<0):
                raise UnknownFees('invalid '+attr)

    def executed_fees(self, *, side:str, qty:float, price:float, maker:bool) -> float:
        if side not in ('buy','sell') or not all(isfinite(x) and x>0 for x in (qty,price)):
            raise UnknownFees('invalid execution')
        base=qty*price*(self.maker_bps if maker else self.taker_bps)/10000
        if self.asset_class == 'equity':
            if self.cat_per_share is None or self.sec_sell_bps is None or self.taf_sell_per_share is None:
                raise UnknownFees('current SEC/TAF/CAT rate must be supplied')
            base+=qty*self.cat_per_share
            if side=='sell':
                base+=qty*price*self.sec_sell_bps/10000+qty*self.taf_sell_per_share
        return base

    def interest(self, *, notional_borrowed:float, hours:float) -> float:
        if not all(isfinite(x) and x>=0 for x in (notional_borrowed,hours)):
            raise UnknownFees('invalid financing interval')
        if self.asset_class != 'equity':
            return 0.0
        if self.margin_apr is None:
            raise UnknownFees('unverified broker margin APR')
        return notional_borrowed*self.margin_apr*hours/(365*24)

    def friction_bps(self, *, maker_entry:bool, maker_exit:bool,
                     modeled_slippage_bps:float, spread_cost_bps:float,
                     per_side_regulatory_bps:float=0.0,
                     carry_bps:float=0.0) -> float:
        vals=(modeled_slippage_bps,spread_cost_bps,per_side_regulatory_bps,carry_bps)
        if any(not isfinite(x) or x<0 for x in vals):
            raise UnknownFees('invalid round-trip friction assumption')
        if self.asset_class == 'equity' and any(getattr(self,x) is None for x in
                                  ('sec_sell_bps','taf_sell_per_share','cat_per_share','margin_apr')):
            raise UnknownFees('regulatory/financing assumptions not calibrated')
        return ((self.maker_bps if maker_entry else self.taker_bps)
               +(self.maker_bps if maker_exit else self.taker_bps)
               +modeled_slippage_bps+spread_cost_bps+per_side_regulatory_bps+carry_bps)

    def expectancy_veto(self, expected_gross_alpha_bps:float,
                        *, friction_bps:float, multiple:float=1.5) -> bool:
        if not all(isfinite(x) for x in (expected_gross_alpha_bps,friction_bps,multiple)) or friction_bps<0 or multiple<1:
            raise UnknownFees('invalid expectancy assumption')
        return expected_gross_alpha_bps < multiple*friction_bps

    def to_env_costs(self, *, spread_bps:float=0.0,impact_bps_per_unit:float=0.0) -> VenueCosts:
        if self.asset_class=='equity' and any(getattr(self,x) is None for x in
                                  ('sec_sell_bps','taf_sell_per_share','cat_per_share','margin_apr')):
            raise UnknownFees('equity friction calibration required before training')
        return VenueCosts(self.maker_bps,self.taker_bps,spread_bps/2,impact_bps_per_unit,self.funding_enabled)


REGISTRY:dict[str,VenueCostProfile] = {
    'ALPACA_SPOT_CRYPTO_TIER1':VenueCostProfile('ALPACA_SPOT_CRYPTO_TIER1','spot_crypto',15,25,False,
       source='https://docs.alpaca.markets/us/v1.1/docs/crypto-fees ; verify account 30D tier'),
    'ALPACA_EQUITIES_UNCALIBRATED':VenueCostProfile('ALPACA_EQUITIES_UNCALIBRATED','equity',0,0,False,
       source='https://alpaca.markets/support/regulatory-fees ; supply current CAT/TAF/SEC and financing'),
    'CRYPTO_PERP_EXAMPLE_UNVERIFIED':VenueCostProfile('CRYPTO_PERP_EXAMPLE_UNVERIFIED','perpetual',2,5,True,
       source='example only; replace using venue/account executed fee ledger'),
}


def get_profile(name:str,*,override:VenueCostProfile|None=None)->VenueCostProfile:
    if override is not None:
        if override.name!=name:raise UnknownFees('profile name mismatch')
        return override
    if name not in REGISTRY:raise UnknownFees('unknown venue fee schedule')
    return REGISTRY[name]

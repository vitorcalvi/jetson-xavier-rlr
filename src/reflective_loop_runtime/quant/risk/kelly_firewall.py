"""Conservative research-only portfolio veto; drawdown limits are not guarantees."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import isfinite, sqrt
from .covariance_v2 import RiskMatrices,effective_positive_correlation


@dataclass(frozen=True)
class OutcomeStats:
    wins:int
    losses:int
    avg_win:float
    avg_loss:float

    @property
    def count(self)->int:
        return self.wins+self.losses

    def lower_win_bound(self)->float:
        if self.wins<0 or self.losses<0 or not all(isfinite(x) and x>0 for x in (self.avg_win,self.avg_loss)):
            raise ValueError('invalid outcomes')
        n=self.count
        if n==0:return 0.0
        p=self.wins/n;z=1.96
        return max(0,(p+z*z/(2*n)-z*sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n))


def shrink_covariance(data:dict[str,list[tuple[int,float]]], asof_ms:int,
                      *, window_ms:int=86_400_000, min_samples:int=30,
                      offdiag_shrink:float=.2)->tuple[tuple[str,...],tuple[tuple[float,...],...]]:
    """Past-only, timestamp-aligned sample covariance with diagonal shrinkage.

    This is fixed diagonal shrinkage, NOT Ledoit-Wolf. Returns are dimensionless.
    """
    names=tuple(sorted(data))
    if not names or window_ms<=0 or min_samples<3 or not 0<=offdiag_shrink<=1:
        raise ValueError('invalid covariance configuration')
    aligned={}
    for name in names:
        rows={}
        for t,v in data[name]:
            if not isfinite(v):raise ValueError('nonfinite return')
            if asof_ms-window_ms<=t<=asof_ms:
                if t in rows:raise ValueError('duplicate return timestamp')
                rows[t]=v
        aligned[name]=rows
    shared=sorted(set.intersection(*(set(aligned[n]) for n in names)))
    if len(shared)<min_samples:raise ValueError('insufficient aligned return history')
    x=[[aligned[n][t] for t in shared] for n in names]
    # Reject numerically constant return series across Python runtimes.
    # Tiny rounding residuals must never count as usable covariance evidence.
    if any(max(row)-min(row) <= 1e-12*max(1.0,max(abs(v) for v in row))
           for row in x):
        raise ValueError('zero variance or numerically degenerate returns')
    avg=[sum(row)/len(row) for row in x]
    cov=tuple(tuple(
        sum((u-avg[i])*(v-avg[j]) for u,v in zip(x[i],x[j]))/(len(shared)-1)*
        (1-offdiag_shrink if i!=j else 1)
        for j in range(len(names))) for i in range(len(names)))
    if any(cov[i][i]<=0 for i in range(len(names))):raise ValueError('zero variance')
    return names,cov


def corr(names:tuple[str,...],cov:tuple[tuple[float,...],...],left:str,right:str)->float:
    i,j=names.index(left),names.index(right)
    return max(-1.0,min(1.0,cov[i][j]/sqrt(cov[i][i]*cov[j][j])))


@dataclass(frozen=True)
class Veto:
    permitted:bool
    reason:str
    capital_fraction:float=0.0
    notional:float=0.0


@dataclass
class RiskFirewall:
    daily_loss_limit:float=.025
    single_cap:float=.02
    gross_cap:float=.12
    beta_cap:float=.12
    min_trades:int=100
    _daily_high:dict[str,float]=field(default_factory=dict,repr=False)

    def evaluate(self, *, symbol:str, now:datetime,equity:float,opening_equity:float,
                 existing_notional:dict[str,float], stats:OutcomeStats,
                 beta:dict[str,float], covariance_names:tuple[str,...],
                 covariance:tuple[tuple[float,...],...],
                 risk_matrices:RiskMatrices|None=None)->Veto:
        if now.tzinfo is None:return Veto(False,'timezone required')
        if not all(isfinite(x) and x>0 for x in (equity,opening_equity)):
            return Veto(False,'invalid account balance')
        if symbol not in covariance_names or symbol not in beta or any(
            s not in covariance_names or s not in beta or not isfinite(n) or n<0
            for s,n in existing_notional.items()):return Veto(False,'missing covariance/beta')
        if any(not isfinite(v) or v<0 for v in beta.values()):return Veto(False,'invalid beta')
        day=now.astimezone(timezone.utc).date().isoformat()
        high=max(equity,opening_equity,self._daily_high.get(day,0.0))
        self._daily_high[day]=high
        if (opening_equity-equity)/opening_equity>=self.daily_loss_limit or (
            high-equity)/high>=self.daily_loss_limit:
            return Veto(False,'drawdown circuit breaker')
        if stats.count < self.min_trades:return Veto(False,'insufficient verified sample')
        p=stats.lower_win_bound();b=stats.avg_win/stats.avg_loss
        f=max(0.0,.25*(p*b-(1-p))/b)
        if f==0:return Veto(False,'conservative expected edge nonpositive')
        for s in existing_notional:
            if risk_matrices is not None:
                if risk_matrices.names!=covariance_names:
                    return Veto(False,'risk matrix universe mismatch')
                rho=effective_positive_correlation(risk_matrices,symbol,s)
            else:
                rho=max(0.,corr(covariance_names,covariance,symbol,s))
            f*=1-rho**2
        used=sum(existing_notional.values())/equity
        beta_used=sum(v*beta[s] for s,v in existing_notional.items())/equity
        f=min(f,self.single_cap,max(0,self.gross_cap-used),
              max(0,self.beta_cap-beta_used)/max(beta[symbol],1e-6))
        if f<=1e-12:return Veto(False,'portfolio cap')
        return Veto(True,'approved_research_only',f,equity*f)


@dataclass
class ATRTrail:
    direction:str
    multiplier:float=2.0
    stop:float|None=None

    def update(self, high:float,low:float,atr:float)->float:
        if self.direction not in ('long','short') or not all(isfinite(v) and v>0 for v in (high,low,atr,self.multiplier)) or high<low:
            raise ValueError('invalid ATR input')
        next_stop=high-self.multiplier*atr if self.direction=='long' else low+self.multiplier*atr
        self.stop=next_stop if self.stop is None else (
            max(self.stop,next_stop) if self.direction=='long' else min(self.stop,next_stop))
        return self.stop

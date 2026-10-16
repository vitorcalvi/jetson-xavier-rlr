"""Past-only OHLC/L2 regime classification; insufficient evidence always abstains."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log, sqrt
from statistics import mean, pstdev
from typing import Literal

State = Literal["TREND_UP", "TREND_DOWN", "MEAN_REVERT", "UNSAFE"]


@dataclass(frozen=True)
class Bar:
    timestamp: int
    open: float
    high: float
    low: float
    close: float

    def __post_init__(self) -> None:
        if (
            self.timestamp < 0
            or not all(isfinite(x) and x > 0 for x in (self.open, self.high, self.low, self.close))
            or self.low > min(self.open, self.close)
            or self.high < max(self.open, self.close)
        ):
            raise ValueError("invalid OHLC")


def yang_zhang(bars: list[Bar], window: int = 20) -> float:
    """Annualization-free Yang-Zhang period volatility from completed OHLC bars."""
    if window < 4 or len(bars) < window + 1:
        raise ValueError("insufficient YZ bars")
    b = bars[-(window + 1) :]
    overnight = [log(b[i].open / b[i - 1].close) for i in range(1, len(b))]
    intra = [log(v.close / v.open) for v in b[1:]]
    rs = [
        log(v.high / v.close) * log(v.high / v.open) + log(v.low / v.close) * log(v.low / v.open)
        for v in b[1:]
    ]
    n = len(intra)
    k = 0.34 / (1.34 + (n + 1) / (n - 1))

    def variance(x: list[float]) -> float:
        mu = mean(x)
        return sum((v - mu) ** 2 for v in x) / (len(x) - 1)

    return sqrt(max(0, variance(overnight) + k * variance(intra) + (1 - k) * mean(rs)))


def adx(bars: list[Bar], period: int = 14) -> float:
    """Wilder-style directional movement index over past closed bars."""
    if period < 2 or len(bars) < 2 * period + 1:
        raise ValueError("insufficient ADX bars")
    bs = bars[-(2 * period + 1) :]
    trs = []
    plus = []
    minus = []
    for prev, cur in zip(bs, bs[1:], strict=False):
        up = cur.high - prev.high
        dn = prev.low - cur.low
        plus.append(up if up > dn and up > 0 else 0.0)
        minus.append(dn if dn > up and dn > 0 else 0.0)
        trs.append(max(cur.high - cur.low, abs(cur.high - prev.close), abs(cur.low - prev.close)))
    dx = []
    for start in range(period, len(trs) + 1):
        tr = sum(trs[start - period : start])
        p = sum(plus[start - period : start]) * 100 / max(tr, 1e-12)
        m = sum(minus[start - period : start]) * 100 / max(tr, 1e-12)
        dx.append(100 * abs(p - m) / max(p + m, 1e-12))
    return mean(dx[-period:])


def hurst(prices: list[float]) -> float:
    """Log-log structure-function H estimate; research diagnostic, not proof of stationarity."""
    if len(prices) < 48 or any(not isfinite(x) or x <= 0 for x in prices):
        raise ValueError("insufficient Hurst sample")
    logp = [log(p) for p in prices]
    x = []
    y = []
    for lag in (1, 2, 4, 8, 16):
        vals = [logp[i] - logp[i - lag] for i in range(lag, len(logp))]
        if len(vals) < 20:
            continue
        v = pstdev(vals)
        if v > 0:
            x.append(log(lag))
            y.append(log(v))
    if len(x) < 3:
        raise ValueError("insufficient Hurst variation")
    mx, my = mean(x), mean(y)
    slope = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=False)) / sum(
        (a - mx) ** 2 for a in x
    )
    return max(0, min(1, slope))


@dataclass(frozen=True)
class Regime:
    state: State
    yz_vol: float | None
    adx: float | None
    hurst: float | None
    obi: float
    funding_z: float | None
    reason: str

    def permits(self, strategy: str) -> bool:
        return (
            self.state == "TREND_UP"
            and strategy == "momentum_long"
            or self.state == "TREND_DOWN"
            and strategy == "momentum_short"
            or self.state == "MEAN_REVERT"
            and strategy == "mean_reversion"
        )


def classify(
    bars: list[Bar],
    *,
    asof: int,
    obi: float,
    funding_history: list[float] | None = None,
    shock_threshold: float = 0.06,
) -> Regime:
    if not isfinite(obi) or abs(obi) > 1:
        raise ValueError("invalid normalized OBI")
    if (
        not bars
        or any(b.timestamp > asof for b in bars)
        or any(a.timestamp >= b.timestamp for a, b in zip(bars, bars[1:], strict=False))
    ):
        raise ValueError("future/duplicate/nonmonotonic bar")
    fz = None
    if funding_history is not None:
        if len(funding_history) < 30 or not all(isfinite(v) for v in funding_history):
            raise ValueError("unverified funding history")
        st = pstdev(funding_history[:-1])
        fz = (funding_history[-1] - mean(funding_history[:-1])) / st if st else 0.0
    if len(bars) < 50:
        return Regime("UNSAFE", None, None, None, obi, fz, "insufficient_history")
    try:
        yz = yang_zhang(bars)
        a = adx(bars)
        h = hurst([b.close for b in bars])
    except ValueError:
        return Regime("UNSAFE", None, None, None, obi, fz, "invalid_indicators")
    if yz > shock_threshold or (fz is not None and abs(fz) > 3):
        return Regime("UNSAFE", yz, a, h, obi, fz, "volatility_or_funding_shock")
    if h > 0.55 and a >= 25:
        delta = bars[-1].close / bars[-16].close - 1
        if delta > 0 and obi >= -0.35:
            return Regime("TREND_UP", yz, a, h, obi, fz, "confirmed_uptrend")
        if delta < 0 and obi <= 0.35:
            return Regime("TREND_DOWN", yz, a, h, obi, fz, "confirmed_downtrend")
    if h < 0.45 and a < 25 and abs(obi) < 0.7:
        return Regime("MEAN_REVERT", yz, a, h, obi, fz, "low_persistence")
    return Regime("UNSAFE", yz, a, h, obi, fz, "chop_or_conflict")

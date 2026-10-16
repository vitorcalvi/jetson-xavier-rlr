"""Aligned-past-bar Ledoit-Wolf-style covariance and downside-tail correlation.

Implements a scalar-identity shrink target and finite-sample plug-in shrinkage.
Shrinking an empirical PSD covariance toward positive diagonal identity is PSD.
No global future features. Correlation calculated only on overlapping bars.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt


@dataclass(frozen=True)
class RiskMatrices:
    names: tuple[str, ...]
    covariance: tuple[tuple[float, ...], ...]
    downside_correlation: tuple[tuple[float, ...], ...]
    shrinkage: float
    common_samples: int
    tail_samples: int


def _quantile(xs: list[float], f: float) -> float:
    if not xs or not 0 <= f <= 1:
        raise ValueError("invalid quantile")
    a = sorted(xs)
    v = f * (len(a) - 1)
    i = int(v)
    return a[i] if i + 1 == len(a) else a[i] * (i + 1 - v) + a[i + 1] * (v - i)


def _cov(rows: list[list[float]]) -> list[list[float]]:
    n = len(rows)
    k = len(rows[0])
    means = [sum(row[j] for row in rows) / n for j in range(k)]
    center = [[row[j] - means[j] for j in range(k)] for row in rows]
    return [[sum(row[i] * row[j] for row in center) / n for j in range(k)] for i in range(k)]


def aligned_risk_matrices(
    data: dict[str, list[tuple[int, float]]],
    *,
    benchmark: str,
    asof_ms: int,
    window_ms: int = 86_400_000,
    min_samples: int = 30,
    min_tail: int = 8,
) -> RiskMatrices:
    if benchmark not in data or not data or min_samples < 10 or min_tail < 4 or window_ms <= 0:
        raise ValueError("invalid risk covariance settings")
    names = tuple(sorted(data))
    indexed = {}
    for name in names:
        vals = {}
        last = -1
        for ts, value in data[name]:
            if ts <= last or not isfinite(value):
                raise ValueError("out-of-order/nonfinite input")
            last = ts
            if asof_ms - window_ms < ts <= asof_ms:
                vals[ts] = value
        indexed[name] = vals
    times = sorted(set.intersection(*(set(indexed[name]) for name in names)))
    if len(times) < min_samples:
        raise ValueError("insufficient synchronized overlapping bars")
    rows = [[indexed[name][ts] for name in names] for ts in times]
    n = len(rows)
    p = len(names)
    s = _cov(rows)
    if any(s[i][i] <= 0 for i in range(p)):
        raise ValueError("zero variation")
    mu = sum(s[i][i] for i in range(p)) / p
    delta = sum((s[i][j] - (mu if i == j else 0)) ** 2 for i in range(p) for j in range(p))
    mean = [sum(row[j] for row in rows) / n for j in range(p)]
    # Ledoit-Wolf plug-in numerator for a shrinkage to mu*Identity.
    fluctuation = 0.0
    for row in rows:
        x = [row[j] - mean[j] for j in range(p)]
        fluctuation += sum((x[i] * x[j] - s[i][j]) ** 2 for i in range(p) for j in range(p))
    beta = fluctuation / (n * n)
    intensity = min(1.0, max(0.0, beta / delta)) if delta > 0 else 1.0
    cov = tuple(
        tuple((1 - intensity) * s[i][j] + (intensity * mu if i == j else 0.0) for j in range(p))
        for i in range(p)
    )
    # Left tail of the benchmark; include only bottom 20% benchmark returns.
    b = names.index(benchmark)
    cutoff = _quantile([row[b] for row in rows], 0.20)
    tail = [row for row in rows if row[b] <= cutoff]
    if len(tail) < min_tail:
        raise ValueError("insufficient benchmark downside observations")
    tail_cov = _cov(tail)
    tc = []
    for i in range(p):
        rr = []
        for j in range(p):
            if tail_cov[i][i] <= 1e-24 or tail_cov[j][j] <= 1e-24:
                if i == j:
                    raise ValueError("zero tail variation")
                rr.append(0.0)
            else:
                rr.append(
                    max(-1.0, min(1.0, tail_cov[i][j] / sqrt(tail_cov[i][i] * tail_cov[j][j])))
                )
        tc.append(tuple(rr))
    return RiskMatrices(names, cov, tuple(tc), intensity, n, len(tail))


def effective_positive_correlation(mat: RiskMatrices, a: str, b: str) -> float:
    i = mat.names.index(a)
    j = mat.names.index(b)
    standard = mat.covariance[i][j] / sqrt(mat.covariance[i][i] * mat.covariance[j][j])
    return min(1.0, max(0.0, standard, mat.downside_correlation[i][j]))

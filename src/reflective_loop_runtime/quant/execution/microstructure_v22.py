"""L1 sized-BBO order-flow imbalance and L2 weighted book indicators."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import exp, isfinite, sqrt
from statistics import mean
from reflective_loop_runtime.quant.execution.smart_router import Book


@dataclass(frozen=True)
class BBO:
    symbol: str
    timestamp_ms: int
    sequence: int
    bid: float
    bid_size: float
    ask: float
    ask_size: float
    source: str

    def __post_init__(self) -> None:
        valid = (self.symbol and self.source and self.timestamp_ms > 0
                 and self.sequence >= 0 and self.bid < self.ask)
        if not valid or not all(isfinite(x) and x > 0 for x in
                                (self.bid, self.ask, self.bid_size, self.ask_size)):
            raise ValueError("incomplete L1 data")

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def microprice(self) -> float:
        return (self.bid_size * self.ask + self.ask_size * self.bid) / (
            self.bid_size + self.ask_size
        )


def ofi_step(before: BBO, after: BBO) -> float:
    if (before.symbol != after.symbol or before.source != after.source
            or before.sequence >= after.sequence
            or before.timestamp_ms >= after.timestamp_ms):
        raise ValueError("mixed-feed or nonmonotonic quote")
    return (
        (after.bid_size if after.bid >= before.bid else 0.0)
        - (before.bid_size if after.bid <= before.bid else 0.0)
        - (after.ask_size if after.ask <= before.ask else 0.0)
        + (before.ask_size if after.ask >= before.ask else 0.0)
    )


def l2_metrics(book: Book, depth: int = 5, decay: float = 0.5) -> dict:
    if min(len(book.bids), len(book.asks)) < 2 or not 2 <= depth <= 50:
        raise ValueError("real L2 depth required")
    n = min(depth, len(book.bids), len(book.asks))
    den = num = micro = 0.0
    for i in range(n):
        weight = exp(-decay * i)
        bid, ask = book.bids[i], book.asks[i]
        den += weight * (bid.qty + ask.qty)
        num += weight * (bid.qty - ask.qty)
        micro += weight * (bid.qty * ask.price + ask.qty * bid.price)
    return {"tier": "L2", "depth": n, "obi": num / den, "microprice": micro / den}


class OFIWindow:
    def __init__(self, length: int = 5) -> None:
        if not 1 <= length <= 10000:
            raise ValueError("invalid window")
        self.last: BBO | None = None
        self.events: deque[tuple[float, float]] = deque(maxlen=length)

    def update(self, tick: BBO) -> dict:
        if self.last is None:
            self.last = tick
            return {"tier": "L1", "ofi": None, "microprice": tick.microprice, "ready": False}
        delta = ofi_step(self.last, tick)
        self.events.append((delta, tick.bid_size + tick.ask_size))
        self.last = tick
        return {
            "tier": "L1",
            "ofi": sum(x[0] for x in self.events) / (sum(x[1] for x in self.events) + 1e-9),
            "microprice": tick.microprice,
            "ready": len(self.events) == self.events.maxlen,
        }


def directional_ic(signals: list[float], mids: list[float], horizon: int = 5) -> float:
    if horizon < 1 or len(signals) != len(mids) or len(signals) <= horizon + 2:
        raise ValueError("insufficient paired observations")
    x = signals[:-horizon]
    y = [(mids[i + horizon] - mids[i]) / mids[i] for i in range(len(x))]
    if not all(isfinite(v) for v in x + y) or any(v <= 0 for v in mids):
        raise ValueError("invalid prices")
    mx, my = mean(x), mean(y)
    var_x = sum((v - mx) ** 2 for v in x)
    var_y = sum((v - my) ** 2 for v in y)
    if min(var_x, var_y) <= 0:
        raise ValueError("zero signal variance")
    return sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True)) / sqrt(
        var_x * var_y
    )

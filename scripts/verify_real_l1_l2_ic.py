"""Public Binance spot stream capture and strictly observational forward IC.

NO TRADING, no account authentication, no orders, no fabricated fill evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

SYMBOLS = ("btcusdt", "ethusdt", "solusdt", "xrpusdt", "dogeusdt")
STREAMS = [f"{s}@depth5@100ms" for s in SYMBOLS] + [f"{s}@bookTicker" for s in SYMBOLS]
URL = "wss://stream.binance.com:9443/stream?streams=" + "/".join(STREAMS)


def capture(path: Path, count: int = 6000, timeout_s: int = 600) -> dict:
    import websocket

    if not 100 <= count <= 30000 or not 1 <= timeout_s <= 900:
        raise ValueError("invalid bounds")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Append-only; do not overwrite real evidence from a prior capture.
    already = 0
    if path.exists():
        with path.open(encoding="utf8") as file:
            already = sum(1 for _ in file)
    ws = websocket.create_connection(URL, timeout=7, enable_multithread=False)
    ws.settimeout(7)
    start = time.monotonic()
    by_kind: dict[str, int] = defaultdict(int)
    failures = 0
    collected = 0
    try:
        with path.open("a", encoding="utf8") as out:
            while collected < count and time.monotonic() - start < timeout_s:
                try:
                    frame = json.loads(ws.recv())
                except TimeoutError:
                    failures += 1
                    if failures >= 3:
                        break
                    continue
                except Exception:
                    failures += 1
                    if failures >= 3:
                        break
                    continue
                name = frame.get("stream", "")
                item = frame.get("data", {})
                symbol = name.split("@")[0]
                if symbol not in SYMBOLS:
                    continue
                kind = "L2" if "@depth5" in name else "L1" if "@bookTicker" in name else None
                if kind is None:
                    continue
                # Preserve real BBO coverage while preventing ticker floods
                # from starving the minimum 5,000 genuine L2 snapshots.
                if kind == "L1" and by_kind["L1"] >= 2 * by_kind["L2"] + 200:
                    continue
                try:
                    if kind == "L2":
                        bids = [[float(p), float(q)] for p, q in item["bids"][:5]]
                        asks = [[float(p), float(q)] for p, q in item["asks"][:5]]
                        if len(bids) < 2 or len(asks) < 2 or bids[0][0] >= asks[0][0]:
                            continue
                        payload = {"bids": bids, "asks": asks}
                    else:
                        bid, ask, bqty, aqty = [float(item[k]) for k in ("b", "a", "B", "A")]
                        if not bid < ask or not min(bqty, aqty) > 0:
                            continue
                        payload = {"bid": bid, "ask": ask, "bid_qty": bqty, "ask_qty": aqty}
                    row = {
                        "source": "Binance public spot WS",
                        "stream": name,
                        "symbol": symbol,
                        "tier": kind,
                        "capture_ns": time.time_ns(),
                        "update_id": int(item.get("lastUpdateId") or item.get("u") or 0),
                        **payload,
                    }
                    out.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                    collected += 1
                    by_kind[kind] += 1
                    if collected % 100 == 0:
                        out.flush()
                        os.fsync(out.fileno())
                except (ValueError, TypeError, KeyError):
                    failures += 1
            out.flush()
            os.fsync(out.fileno())
    finally:
        ws.close()
    return {
        "recorded_this_run": collected,
        "preexisting_lines": already,
        "l1": by_kind["L1"],
        "l2": by_kind["L2"],
        "elapsed_s": round(time.monotonic() - start, 3),
        "failures": failures,
        "genuine": True,
        "broker_orders": 0,
    }


def _rank(x: list[float]) -> list[float]:
    order = sorted(range(len(x)), key=x.__getitem__)
    ranks = [0.0] * len(x)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        for k in range(i, j):
            ranks[order[k]] = (i + 1 + j) / 2
        i = j
    return ranks


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 30:
        return None
    a, b = _rank(x), _rank(y)
    ma, mb = statistics.mean(a), statistics.mean(b)
    ab = sum((v - ma) * (w - mb) for v, w in zip(a, b, strict=True))
    da = sum((v - ma) ** 2 for v in a)
    db = sum((v - mb) ** 2 for v in b)
    return ab / math.sqrt(da * db) if da > 0 and db > 0 else None


def _hac_t_stat(signals: list[float], targets: list[float], horizon: int) -> float | None:
    """Newey-West-like rank IC test statistic for serially overlapping returns."""
    if len(signals) != len(targets) or len(signals) < 30:
        return None
    x, y = _rank(signals), _rank(targets)
    mx, my = statistics.mean(x), statistics.mean(y)
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    norm = math.sqrt(sum(z * z for z in dx) * sum(z * z for z in dy))
    if norm <= 0:
        return None
    n = len(x)
    z = [dx[k] * dy[k] * n / norm for k in range(n)]
    mean_z = statistics.mean(z)
    centered = [val - mean_z for val in z]
    variance = sum(v * v for v in centered) / n
    lag = min(horizon + 10, 50, n // 4)
    for offset in range(1, lag + 1):
        ac = sum(centered[j] * centered[j - offset] for j in range(offset, n)) / n
        variance += 2 * (1 - offset / (lag + 1)) * ac
    return mean_z / math.sqrt(variance / n) if variance > 0 else None


def score(path: Path, *, fee_roundtrip_bps: float = 40) -> dict:
    from reflective_loop_runtime.quant.execution.microstructure_v22 import BBO, OFIWindow
    from reflective_loop_runtime.quant.execution.smart_router import Book, Level

    by_key: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    windows = {sym: OFIWindow(length=5) for sym in SYMBOLS}
    counts = defaultdict(int)
    first_ns = {}
    last_ns = {}
    with path.open(encoding="utf8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj.get("source") != "Binance public spot WS":
                    continue
                symbol, tier = obj["symbol"], obj["tier"]
                if symbol not in SYMBOLS or tier not in ("L1", "L2"):
                    continue
                key = (symbol, tier)
                sequence = int(obj["capture_ns"])
                if sequence <= last_ns.get(key, 0):
                    continue
                if tier == "L2":
                    bids = tuple(Level(*b) for b in obj["bids"])
                    asks = tuple(Level(*a) for a in obj["asks"])
                    book = Book(
                        bids=bids, asks=asks, timestamp_ms=sequence // 1000000, sequence=sequence
                    )
                    if min(len(bids), len(asks)) < 2:
                        continue
                    signal = book.imbalance(5)
                    mid = (bids[0].price + asks[0].price) / 2
                else:
                    tick = BBO(
                        symbol,
                        sequence // 1000000,
                        sequence,
                        obj["bid"],
                        obj["bid_qty"],
                        obj["ask"],
                        obj["ask_qty"],
                        "Binance public spot WS",
                    )
                    v = windows[symbol].update(tick)
                    if v["ofi"] is None or not v["ready"]:
                        last_ns[key] = sequence
                        continue
                    signal, mid = v["ofi"], tick.mid
                by_key[key].append((signal, mid))
                counts[tier] += 1
                first_ns.setdefault(key, sequence)
                last_ns[key] = sequence
            except (ValueError, TypeError, KeyError, ZeroDivisionError):
                continue
    result = {}
    for (sym, tier), pairs in by_key.items():
        metrics = {}
        for horizon in (1, 5, 10):
            xs = [x for x, _ in pairs[:-horizon]]
            ys = [
                (pairs[i + horizon][1] - pairs[i][1]) / pairs[i][1]
                for i in range(len(pairs) - horizon)
            ]
            rho = _spearman(xs, ys)
            # Naive iid t statistic; overlapping futures autocorrelated.
            t = (
                rho * math.sqrt((len(xs) - 2) / (1 - rho * rho))
                if rho is not None and abs(rho) < 1
                else None
            )
            hac = _hac_t_stat(xs, ys, horizon)
            hypothetical_gross = (
                statistics.mean(
                    (1 if xs[k] > 0 else -1 if xs[k] < 0 else 0) * ys[k] * 10000
                    for k in range(len(xs))
                )
                if xs
                else None
            )
            metrics[str(horizon)] = {
                "spearman_ic": round(rho, 6) if rho is not None else None,
                "t_iid_optimistic": round(t, 4) if t is not None else None,
                "t_hac_approx": round(hac, 4) if hac is not None else None,
                "hypothetical_mid_gross_bps": round(hypothetical_gross, 4)
                if hypothetical_gross is not None
                else None,
                "hypothetical_net_at_40bps": round(hypothetical_gross - fee_roundtrip_bps, 4)
                if hypothetical_gross is not None
                else None,
                "pairs": len(xs),
            }
        result[f"{sym}_{tier}"] = {"samples": len(pairs), "forward": metrics}
    return {
        "source": str(path),
        "counts": dict(counts),
        "per_stream": result,
        "genuine_snapshot_count": counts["L2"],
        "gate_5000": counts["L2"] >= 5000,
        "fee_roundtrip_bps_example_only": fee_roundtrip_bps,
        "maker_expectancy_net_verified": False,
        "maker_expectancy_note": "Hypothetical mid-mid net 40bps excludes queue/latency/slippage; maker fill expectancy NOT verified",
        "statistic_warning": "HAC t is an approximate within-sample statistic; independent out-of-sample validation remains pending",
    }


def persist_real_toon(source_path: Path, destination: Path, analysis: dict) -> int:
    """Write ACTUAL captured WS messages in append-safe TOON-style records.

    The unmodified JSONL original is retained as the primary auditable source.
    The output is derived real data, not a generated or synthetic order book.
    """
    temp = destination.with_suffix(".toon.tmp")
    n = 0
    with source_path.open(encoding="utf8") as src, temp.open("w", encoding="utf8") as output:
        output.write("provenance: Binance public spot WebSocket\n")
        output.write("broker_orders: 0\n")
        output.write("snapshot_count_L2: " + str(analysis["genuine_snapshot_count"]) + "\n")
        output.write("broker_maker_expectancy_verified: false\n")
        for line in src:
            record = json.loads(line)
            if record.get("source") != "Binance public spot WS":
                continue
            output.write("book:\n" if record.get("tier") == "L2" else "bbo:\n")
            for field in (
                "symbol",
                "stream",
                "capture_ns",
                "update_id",
                "bid",
                "ask",
                "bid_qty",
                "ask_qty",
                "bids",
                "asks",
            ):
                if field in record:
                    output.write(
                        "  "
                        + field
                        + ": "
                        + json.dumps(record[field], separators=(",", ":"), allow_nan=False)
                        + "\n"
                    )
            n += 1
        output.flush()
        os.fsync(output.fileno())
    os.replace(temp, destination)
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--target", type=int, default=10000)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--output", type=Path, default=Path("/home/vitor/rlr/monitoring/real_l1_l2_sample.jsonl")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("/home/vitor/rlr/monitoring/real_l1_l2_sample.toon")
    )
    args = parser.parse_args()
    capture_result = capture(args.output, args.target, args.timeout) if args.capture else None
    analysis = score(args.output) if args.score else None
    result = {
        "capture": capture_result,
        "analysis": analysis,
        "halt_file_checked": Path("/home/vitor/rlr/flags/HALT_ENTRIES").is_file(),
    }
    if args.report and analysis:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        persist_real_toon(args.output, args.report, analysis)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

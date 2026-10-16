"""Diagnostic recorded-feature replay; NEVER confuse this with blind backtest.

Only evaluates conditions supported by old journals. New features absent from
the record (quote age/full VWAP/two-bar confirmation) are not fabricated.
"""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path


def replay(journal: Path) -> dict:
    records=[json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
    decisions=[x for x in records if x.get("kind")=="decision" and x.get("features")
               and x.get("symbol")=="SOL/USD"]
    indexed={int(x["ts"]):x for x in decisions}
    inspections=[]
    for row in decisions:
        d=row.get("decision") or {}
        if d.get("action")!="enter_long":
            continue
        f=row["features"]
        price=f["price"]
        blockers=[]
        if not (f.get("ema20_5m") is not None and price>f["ema20_5m"]):
            blockers.append("below_ema20")
        before=indexed.get(int(row["ts"])-300)
        if before:
            ret5=price/before["features"]["price"]-1
            if ret5 < 0.0005: blockers.append("weak_ret5")
        else: ret5=None
        if f.get("ret_60m") is None or f["ret_60m"]<=0:
            blockers.append("weak_ret60")
        if f.get("spread_pct") is None or f.get("vwap") is None or (
            (price-f["vwap"])/price <=1.25*f["spread_pct"]/100
        ):
            blockers.append("reclaim_below_spread")
        stop=d.get("stop")
        target=d.get("target")
        if stop is not None and target is not None and price>stop:
            model_net_r=(target*(1-.0025)**2-price)/(price-stop)
            if model_net_r<0.5: blockers.append("insufficient_net_reward")
        else: model_net_r=None
        inspections.append({
            "ts":row["ts"],"signal_price":price,"blockers":blockers,
            "modeled_target_net_r":round(model_net_r,4) if model_net_r is not None else None,
            "ret_5m":round(ret5,6) if ret5 is not None else None,
            "missing_evidence":["full_utc_day_vwap","two_closed_bar_confirmation","quote_age"],
        })
    return {"mode":"DIAGNOSTIC_ONLY_NOT_VALIDATED_BACKTEST",
            "candidate_trades":len(inspections),
            "blocked_by_observable_gates":sum(bool(t["blockers"]) for t in inspections),
            "counts":dict(Counter(b for t in inspections for b in t["blockers"])),
            "trades":inspections,
            "forward_paper_edge_validated":False}


def main():
    if len(sys.argv)!=3:
        raise SystemExit("Usage: replay_hardened_journal.py JOURNAL OUTPUT_JSON")
    result=replay(Path(sys.argv[1]))
    p=Path(sys.argv[2]);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()

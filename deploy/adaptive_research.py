"""Research-only hourly diagnostic: Qwen may suggest experiments, never orders.

The selector remains signed and statistical. This process has no broker keys or
write access to signed scope; a model completion cannot approve itself.
"""
from __future__ import annotations
import collections
import datetime as dt
import json
import os
from pathlib import Path
import sqlite3
import time
from urllib import request

from reflective_loop_runtime.trading.experiment_registry import load as active_experiment
DIR = Path(active_experiment()["ledger"])
OUT = DIR / "adaptive_research" / "research_hypothesis.json"
ACTIONS = ("COLLECT_FORWARD_DATA","AUDIT_FEES","INVESTIGATE_RECLAIM","REDUCE_TURNOVER","NO_CHANGE")

def summarize() -> dict:
    now=time.time()
    journal=DIR/"journal.jsonl"
    counter=collections.Counter()
    ticks=0
    for line in journal.open():
        row=json.loads(line)
        if row.get("kind")!="decision" or float(row.get("ts",0)) < now-86400:
            continue
        ticks+=1
        counter[str((row.get("decision")or{}).get("reason")or "unknown")]+=1
    with sqlite3.connect(f"file:{DIR/'trades.db'}?mode=ro", uri=True) as db:
        trades=db.execute("SELECT entry_ts,entry_px,exit_px,qty,stop FROM trades "
                          "WHERE closed=1 AND entry_ts>? AND exit_px IS NOT NULL", (now-86400,)).fetchall()
    fee=0.0025 if "solusd" in str(DIR).lower() else 0.0
    net=0.0
    for _,entry,ex,qty,_ in trades:
        net+=(ex*(1-fee)**2-entry)*qty
    return {
        "window_h":24, "decision_count":ticks,"entry_count":len(trades),
        "gross_realized_trades":len(trades),
        "modeled_net_usd":round(net,3),
        "fee_verified":False,
        "reason_counts":dict(counter.most_common(8)),
        "validated_forward_edge":False,
        "active_experiment_ledger":str(DIR),
        "research_only_no_order_authority":True,
    }

def research(summary: dict) -> dict:
    schema={"type":"object","additionalProperties":False,
            "properties":{"action":{"type":"string","enum":list(ACTIONS)},
                          "note":{"type":"string","maxLength":36}},
            "required":["action","note"]}
    message={"model":"qwen35","messages":[
        {"role":"system","content":"You are a paper-only research advisor. Suggest an investigation, not a trade. Return short JSON only. The sample lacks a validated forward edge."},
        {"role":"user","content":json.dumps(summary,separators=(",",":"))[:750]}],
        "temperature":0,"max_tokens":64,
        "response_format":{"type":"json_schema","json_schema":{
            "name":"research","strict":True,"schema":schema}}}
    try:
        req=request.Request("http://127.0.0.1:8081/v1/chat/completions",
                            data=json.dumps(message).encode(),
                            headers={"Content-Type":"application/json"},method="POST")
        with request.urlopen(req,timeout=25) as response:
            obj=json.load(response)
        choice=obj["choices"][0]
        if choice.get("finish_reason")=="length":
            raise ValueError("truncated completion")
        payload=json.loads(choice["message"]["content"])
        if payload.get("action") not in ACTIONS:
            raise ValueError("unrecognized action")
        return {"action":payload["action"],
                "note":str(payload.get("note")or "")[:36],
                "model":"Qwen3.5-2B-Q4_K_M",
                "structured_json_verified":True}
    except Exception as exc:
        return {"action":"COLLECT_FORWARD_DATA",
                "note":"Model unavailable; fail closed",
                "model":"Qwen3.5-2B-Q4_K_M",
                "structured_json_verified":False,
                "error_class":type(exc).__name__}

def main():
    snapshot=summarize()
    suggestion=research(snapshot)
    report={
      "created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
      "evidence":snapshot,
      "model_hypothesis":suggestion,
      "trade_decision":"NO_TRADE",
      "permission_to_execute":False,
      "automated_strategy_changes":False,
      "comment":"Research proposals require independent forward validation, verified costs and operator-signed approval.",
    }
    OUT.parent.mkdir(exist_ok=True)
    tmp=OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    os.replace(tmp,OUT)
    print(json.dumps({"research_saved":True,"model_action":suggestion["action"],
                      "model_json_verified":suggestion["structured_json_verified"],
                      "paper_entry_approval":False}))
if __name__=="__main__":
    main()

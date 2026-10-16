#!/usr/bin/env python3
"""Stage 2 bounded paper trial: read-only independent broker reconciler."""
from __future__ import annotations
import json,os,sqlite3,time
from decimal import Decimal,InvalidOperation
from pathlib import Path
from datetime import datetime,timezone
from reflective_loop_runtime.trading.experiment_registry import load
from stage2_alpaca_paper_adapter import Stage2AlpacaPaperAdapter
ROOT=Path("/home/vitor/rlr")
def main():
    r=load()
    if r["service"]!="rlr-stage2-paper.service":
        return
    p=Path(r["ledger"])
    client=Stage2AlpacaPaperAdapter(os.environ["ALPACA_PAPER_KEY"],os.environ["ALPACA_PAPER_SECRET"])
    positions=client.get_positions();orders=client.get_orders(status="open")
    db=sqlite3.connect("file:"+str(p/"controller.db")+"?mode=ro",uri=True)
    cols=[x[1] for x in db.execute("pragma table_info(controller_state)")]
    row=db.execute("select * from controller_state where id=1").fetchone()
    if not row:raise RuntimeError("missing controller state")
    state=dict(zip(cols,row))
    hb=json.loads((p/"heartbeat.json").read_text())
    age=time.time()-datetime.fromisoformat(hb["updated_at_utc"]).timestamp()
    held=[x for x in positions if x.get("symbol")=="SPY" and float(x.get("qty") or 0)>0]
    stops=[x for x in orders if x.get("symbol")=="SPY" and x.get("side")=="sell"
           and x.get("type")=="stop" and x.get("status") in
           ("new","accepted","held","pending_new","partially_filled","pending_cancel")]
    problems=[]
    if held and not stops:problems.append("PAPER_POSITION_WITHOUT_BROKER_STOP")
    if held and state["state"]=="LONG_PROTECTED":
        if len(stops)!=1:
            problems.append("PROTECTION_STATE_MISMATCH")
        else:
            stop=stops[0]
            if stop.get("id")!=state.get("stop_order_id") or stop.get("client_order_id")!=state.get("stop_coid"):
                problems.append("STOP_IDENTITY_MISMATCH")
            try:
                held_qty=Decimal(str(held[0]["qty"]))
                total=Decimal(str(stop["qty"]))
                filled=Decimal(str(stop.get("filled_qty") or "0"))
                if total-filled != held_qty or Decimal(str(state["held_qty"]))!=held_qty:
                    problems.append("STOP_REMAINING_QTY_MISMATCH")
                if Decimal(str(stop["stop_price"]))!=Decimal(str(state["stop_price"])):
                    problems.append("STOP_PRICE_MISMATCH")
            except (InvalidOperation,KeyError,ValueError,TypeError):
                problems.append("STOP_PROTECTION_FIELDS_INVALID")
    if held and any(o.get("symbol")=="SPY" and o.get("side")=="buy" for o in orders):
        problems.append("OUTSTANDING_BUY_WITH_POSITION")
    if held and state["state"] in ("HALT_ENTRIES","EMERGENCY_STOP","RECONCILING_UNCERTAIN_STOP","BUY_PARTIAL_CANCEL_PENDING"):
        problems.append("HALTED_OR_UNCERTAIN_WITH_POSITION")
    runtime_tick=p/"runtime_tick.json"
    try:
        tick_age=time.time()-json.loads(runtime_tick.read_text())["ts"]
    except Exception:
        tick_age=999999
    if held and tick_age>20:problems.append("STAGE2_PROCESS_STALE_WITH_POSITION")
    if state["transition_seq"]!=hb["transition_seq"]:
        problems.append("HEARTBEAT_DB_SEQ_MISMATCH")
    status={"utc":datetime.now(timezone.utc).isoformat(),"mode":"READ_ONLY",
            "paper_only":True,"state":state["state"],"completed_cycles":state["completed_cycles"],
            "broker_positions":len(positions),"broker_open_orders":len(orders),
            "working_stops":len(stops),"heartbeat_age_s":round(age,2),"runtime_tick_age_s":round(tick_age,2),
            "issues":problems}
    temp=p/"independent_safety_status.tmp"
    temp.write_text(json.dumps(status,sort_keys=True)+"\n")
    os.chmod(temp,0o600)
    temp.replace(p/"independent_safety_status.json")
    if problems:
        with (p/"alerts_outbox.jsonl").open("a") as f:
            f.write(json.dumps({"title":"RLR Stage2 PAPER safety alert",
              "message":",".join(problems),"priority":"urgent","tags":"warning,paper"})+"\n")
    print(json.dumps(status,sort_keys=True))
if __name__=="__main__":main()

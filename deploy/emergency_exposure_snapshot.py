#!/usr/bin/env python3
"""Read-only broker exposure snapshot for emergency kill; never sends orders."""
from __future__ import annotations
import json, os,sys
from pathlib import Path
from datetime import datetime, timezone
from reflective_loop_runtime.trading.broker import PaperBroker
from reflective_loop_runtime.trading.http import _default_http,_default_post

def main():
    p=Path("/home/vitor/rlr/last_emergency_exposure.json")
    result={"at":datetime.now(timezone.utc).isoformat(),"paper_only":True,"safe_to_trade":False,
            "broker": "unavailable", "positions":None,"orders":None}
    try:
        b=PaperBroker(os.environ["ALPACA_PAPER_KEY"],os.environ["ALPACA_PAPER_SECRET"],
                      http_get=_default_http,http_post=_default_post)
        positions=b.positions();orders=b.open_orders()
        result.update(broker="verified",positions=[
            {"symbol":x.get("symbol"),"qty":x.get("qty")} for x in positions],
            orders=[{"symbol":x.get("symbol"),"side":x.get("side"),"status":x.get("status")}
                    for x in orders])
    except Exception as e:
        result["error_class"]=type(e).__name__
    temp=p.with_suffix(".tmp")
    temp.write_text(json.dumps(result,sort_keys=True)+"\n")
    os.chmod(temp,0o600);os.replace(temp,p)
    print("PAPER_EMERGENCY_EXPOSURE",json.dumps(result,sort_keys=True))
    return 0
if __name__=="__main__":sys.exit(main())

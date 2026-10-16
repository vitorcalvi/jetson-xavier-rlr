#!/usr/bin/env python3
"""Paper trader alerts -> ntfy; active-registry aware, durable per-ledger offsets.

When ntfy rejects a message, offset never advances. Idempotency-Key is stable
for the current message so external retries can be de-duplicated by the server.
"""
from __future__ import annotations
import hashlib
import inspect
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from reflective_loop_runtime.trading.experiment_registry import load

def _outbox(env: dict) -> Path:
    if env.get("RLR_ALERT_OUTBOX"): return Path(env["RLR_ALERT_OUTBOX"])
    return Path(load()["ledger"])/"alerts_outbox.jsonl"

def post(url: str, rec: dict, *, key: str) -> None:
    headers={"Title":str(rec.get("title","RLR alert"))[:120],
             "Priority":str(rec.get("priority","default")),
             "Tags":str(rec.get("tags","rlr"))[:100],
             "Idempotency-Key":key}
    body = str(rec.get("message",""))[:1500].encode("utf-8")
    req=urllib.request.Request(url,data=body,headers=headers,method="POST")
    with urllib.request.urlopen(req,timeout=8) as response:
        if response.status not in (200,201,202,204):
            raise RuntimeError("ntfy delivery was not acknowledged")

def relay_once(outbox: str | Path, url: str, poster=post) -> int:
    outbox = Path(outbox)
    if not outbox.exists():return 0
    off = outbox.parent/"relay_offset.json"
    try:
        state=json.loads(off.read_text());offset=int(state.get("byte_offset",0))
        if offset<0 or offset>outbox.stat().st_size:raise ValueError("offset outside file")
        if offset>0:
            with outbox.open("rb") as f:prefix=f.read(offset)
            if hashlib.sha256(prefix).hexdigest()!=state.get("prefix_sha256"):
                raise ValueError("outbox changed under existing delivery offset")
    except FileNotFoundError:offset=0
    count=0
    with outbox.open("rb") as f:
        f.seek(offset)
        while True:
            raw=f.readline()
            if not raw or not raw.endswith(b"\n"):break
            payload=json.loads(raw)
            key=hashlib.sha256((str(outbox.resolve())+":"+str(offset)+":").encode()+raw).hexdigest()
            try:
                if "key" in inspect.signature(poster).parameters:
                    poster(url,payload,key=key)
                else:  # backward compatible test/custom transport
                    poster(url,payload)
            except Exception:break
            offset=f.tell();count+=1
            with outbox.open("rb") as check:prefix=check.read(offset)
            state={"byte_offset":offset, "line_sha256":hashlib.sha256(raw).hexdigest(),
                   "prefix_sha256":hashlib.sha256(prefix).hexdigest(),"last_idempotency_key":key}
            temp=off.with_suffix(".tmp")
            temp.write_text(json.dumps(state,sort_keys=True)+"\n")
            os.chmod(temp,0o600)
            os.replace(temp,off)
    return count

def main() -> int:
    if not os.environ.get("RLR_NTFY_URL","").strip():
        print("ntfy URL missing",file=sys.stderr);return 2
    while True:
        try:
            path=_outbox(os.environ)
            count=relay_once(path,os.environ["RLR_NTFY_URL"])
            if count:print(f"ntfy acknowledged {count} paper alerts",flush=True)
        except Exception as exc:
            print("relay error: "+type(exc).__name__,file=sys.stderr,flush=True)
        if "--once" in sys.argv:return 0
        time.sleep(15)
if __name__=="__main__":raise SystemExit(main())

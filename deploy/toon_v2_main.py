#!/usr/bin/env python3
"""Non-trading quant entrypoint. Only replays local, untrusted history as shadow.

No broker imports, network, accounts, credentials, service restarts or orders.
For future live-feed integration use a separately reviewed feed-only adapter.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from sys import stderr
from benchmarks.bench_integrated_shadow import run


def main()->int:
    p=argparse.ArgumentParser(description='TOON v2 offline SHADOW-only pipeline')
    p.add_argument('--synthetic-smoke-ticks',type=int,default=500)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if not 1<=args.synthetic_smoke_ticks<=10000:
        p.error('tick count out of bounds')
    results=run(args.synthetic_smoke_ticks)
    if args.output:
        if args.output.exists():
            p.error('refusing overwrite')
        args.output.write_text(json.dumps(results,indent=2,sort_keys=True)+'\n')
    print(json.dumps(results,indent=2,sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())

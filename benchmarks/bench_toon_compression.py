"""Empirical encoder benchmark. Real payloads supplied via --input JSONL.

Synthetic examples are for smoke tests ONLY and never satisfy the official
>=100 real W1/W2 records gate. cl100k_base requires optional tiktoken package.
"""
from __future__ import annotations
import argparse,json,statistics,time,sys
from pathlib import Path
from reflective_loop_runtime.quant.edge_jetson.runtime_opt import toon_encode,toon_decode


def benchmark(records:list[dict],*,real_records:bool=False)->dict:
    try:
        import tiktoken
        encode=tiktoken.get_encoding('cl100k_base').encode
    except ImportError:
        encode=None
    if not records:raise ValueError('no W1/W2 payloads supplied')
    js_tokens=[];toon_tokens=[];js_bytes=0;toon_bytes=0;times=[];lossless=0
    for record in records:
        if 'W1' not in record or 'W2' not in record:raise ValueError('missing W1/W2')
        js=json.dumps(record,sort_keys=True,ensure_ascii=False,separators=(',',':'))
        t0=time.perf_counter_ns();wire=toon_encode(record);roundtrip=toon_decode(wire)
        times.append((time.perf_counter_ns()-t0)/1e6)
        lossless+=int(roundtrip==record)
        js_bytes+=len(js.encode());toon_bytes+=len(wire.encode())
        if encode is not None:
            js_tokens.append(len(encode(js)));toon_tokens.append(len(encode(wire)))
    return dict(real_records=real_records,count=len(records),lossless=lossless,
                byte_reduction=(1-toon_bytes/js_bytes) if js_bytes else None,
                cl100k_base_available=encode is not None,
                token_reduction=(1-sum(toon_tokens)/sum(js_tokens)) if js_tokens else None,
                serialization_p50_ms=statistics.median(times),
                serialization_p99_ms=sorted(times)[max(0,int(len(times)*.99)-1)],
                promotion_gate=bool(real_records and len(records)>=100 and lossless==len(records) and
                      encode is not None and (1-sum(toon_tokens)/sum(js_tokens))>=.35 and max(times)<.5))


def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path)
    ap.add_argument('--synthetic-count',type=int,default=100);args=ap.parse_args()
    if args.input:
        with args.input.open() as f:records=[json.loads(l) for l in f if l.strip()]
        real=True
    else:
        records=[{'W1':{'symbol':'SPY','volume':int(i*137),'time':i,'side':'buy','price':100+i*.01},
                  'W2':{'decision':'hold','notes':f'synthetic trial memory {i}','confidence':.71,
                        'features':[round((i+j)/300,6) for j in range(8)]}} for i in range(args.synthetic_count)]
        real=False
    print(json.dumps(benchmark(records,real_records=real),sort_keys=True,indent=2))


if __name__=='__main__':main()

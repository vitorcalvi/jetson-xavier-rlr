"""Deterministic TOON-compatible W1/W2 memory wire and bounded edge utilities.

This compact mapping subset uses path keys plus JSON scalar/array literals.
It does not claim to implement every upstream TOON feature. Measure actual
model token use before claiming token savings. No GPU clock changes here.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from math import exp, isfinite, sqrt, tanh
from pathlib import Path
from typing import Any
import json
import sqlite3


from reflective_loop_runtime.quant.edge_jetson.tabular_v22 import (
    decode as toon_decode,
    encode as toon_encode,
)


def wire_metrics(payload:dict[str,Any])->dict[str,int]:
    toon=toon_encode(payload).encode('utf8')
    js=json.dumps(payload,ensure_ascii=False,separators=(',',':'),sort_keys=True).encode('utf8')
    return {'toon_bytes':len(toon),'json_bytes':len(js),'saved_bytes':len(js)-len(toon)}


class RingBuffer:
    """Bounded byte ring: Python copies payloads, not CUDA zero-copy."""
    def __init__(self,capacity:int,max_item:int=8192):
        if not 1<=capacity<=100000 or not 1<=max_item<=1_000_000:
            raise ValueError('invalid buffer bounds')
        self._q:deque[bytes]=deque(maxlen=capacity)
        self.max_item=max_item

    def push(self,data:bytes)->None:
        if not isinstance(data,bytes) or len(data)>self.max_item:
            raise ValueError('oversized frame')
        self._q.append(data)

    def latest(self)->bytes|None:
        return self._q[-1] if self._q else None

    def __len__(self)->int:
        return len(self._q)


def cosine(a:list[float],b:list[float])->float:
    if len(a)!=len(b) or not a or any(not isfinite(x) for x in a+b):
        raise ValueError('invalid embeddings')
    mag=sqrt(sum(x*x for x in a)*sum(y*y for y in b))
    if mag==0:raise ValueError('zero embedding')
    return sum(x*y for x,y in zip(a,b))/mag


def memory_score(query:list[float],candidate:list[float],age_days:float,
                 realized_sortino:float,half_life_days:float=30)->float:
    if not all(isfinite(x) for x in (age_days,realized_sortino,half_life_days)) or age_days<0 or half_life_days<=0:
        raise ValueError('invalid memory metadata')
    return .60*cosine(query,candidate)+.25*exp(-.6931471805599453*age_days/half_life_days)+.15*tanh(realized_sortino)


@dataclass(frozen=True)
class Memory:
    id:str
    embedding:list[float]
    timestamp:int
    sortino:float
    payload:dict[str,Any]


class MemoryStore:
    """SQLite-only local read/write W1/W2 storage. No external vector database."""
    def __init__(self,db_path:Path):
        self.path=Path(db_path)
        if self.path.parent.exists():
            self.path.parent.mkdir(parents=True,exist_ok=True)
        else:self.path.parent.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(self.path) as c:
            c.execute('CREATE TABLE IF NOT EXISTS memory(id TEXT PRIMARY KEY,embedding TEXT NOT NULL,ts INTEGER NOT NULL,sortino REAL NOT NULL,payload TEXT NOT NULL)')

    def put(self,m:Memory)->None:
        if not m.id or len(m.id)>128 or m.timestamp<=0 or not isfinite(m.sortino):raise ValueError('invalid memory')
        cosine(m.embedding,m.embedding)
        wire=toon_encode(m.payload)
        with sqlite3.connect(self.path) as c:
            c.execute('INSERT OR REPLACE INTO memory VALUES (?,?,?,?,?)',(
                m.id,json.dumps(m.embedding),m.timestamp,m.sortino,wire))

    def recall(self,query:list[float],now:int,limit:int=5)->list[tuple[str,float,dict[str,Any]]]:
        if now<=0 or not 1<=limit<=100:raise ValueError('invalid recall parameters')
        rows=[]
        with sqlite3.connect(self.path) as c:
            for id,emb,ts,sortino,payload in c.execute('SELECT id,embedding,ts,sortino,payload FROM memory'):
                if ts>now:continue
                try:score=memory_score(query,json.loads(emb),(now-ts)/86400,sortino)
                except ValueError:continue
                rows.append((id,score,toon_decode(payload)))
        return sorted(rows,key=lambda x:(-x[1],x[0]))[:limit]


def thermal_readings(zone_root:Path=Path('/sys/class/thermal'))->dict[str,float]:
    """Read-only sysfs temperatures in °C; no forced power/clock modifications."""
    out={}
    for zone in sorted(zone_root.glob('thermal_zone*')):
        try:
            label=(zone/'type').read_text().strip()
            milli=float((zone/'temp').read_text().strip())
            if isfinite(milli):out[f'{zone.name}:{label}']=milli/1000
        except (OSError,ValueError):continue
    return out

#!/usr/bin/env python3
"""Read-only Stage 2 go/no-go probe. Never submits or cancels orders."""
import json, os, time
from datetime import datetime,timezone
from reflective_loop_runtime.trading.broker import PaperBroker
from reflective_loop_runtime.trading.http import _default_http,_default_post
from reflective_loop_runtime.trading.market import fetch_snapshots
from rlr_stage2_autonomy import StrictSessionQuoteGate
from reflective_loop_runtime.trading.experiment_registry import load

def main():
    if not os.environ.get('ALPACA_PAPER_KEY','').startswith('PK'):
        raise RuntimeError('paper key not validated')
    broker=PaperBroker(os.environ['ALPACA_PAPER_KEY'],os.environ['ALPACA_PAPER_SECRET'],http_get=_default_http,http_post=_default_post,time_in_force='day')
    acc=broker.account_status();positions=broker.positions();orders=broker.open_orders()
    now=time.time()
    headers=broker._headers()
    snapshot=fetch_snapshots(('SPY',),http=_default_http,now=now,headers=headers)['SPY']
    closed=[b for b in snapshot.bars if b.ts+60<=now]
    if snapshot.quote and closed:
        decision=StrictSessionQuoteGate().evaluate(
            snapshot.quote.bid,snapshot.quote.ask,
            datetime.fromtimestamp(snapshot.quote.ts,timezone.utc),
            datetime.fromtimestamp(closed[-1].ts+60,timezone.utc),
            datetime.fromtimestamp(now,timezone.utc))
        ok,reason=decision.allowed,decision.veto_reason
    else:ok,reason=False,'market_data_missing'
    print(json.dumps({'utc':datetime.fromtimestamp(now,timezone.utc).isoformat(),'account_active':acc['active'],'paper_host_validated':True,
       'broker_positions':len(positions),'broker_open_orders':len(orders),'one_minute_quote_gate':reason,
       'entry_market_data_valid_now':ok,'active_experiment':load()['service'],'mode':'READ_ONLY'},sort_keys=True))
if __name__=='__main__':main()

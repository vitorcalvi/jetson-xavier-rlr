"""PAPER broker certification preflight: GET-only while HALT_ENTRIES active.

No order POST, cancel, replace, account mode switching, or implicit re-arm.
B4 order lifecycle must be performed later in an explicitly supervised phase.
"""
from __future__ import annotations
import json,os,urllib.request
from pathlib import Path


def preflight(*,halt_file:Path=Path('/home/vitor/rlr/flags/HALT_ENTRIES'),
              endpoint:str='https://paper-api.alpaca.markets') ->dict:
    # Do not even allow an alternative host. Credentials are never logged.
    if endpoint!='https://paper-api.alpaca.markets':raise ValueError('hardcoded paper-only API')
    if not halt_file.exists():raise RuntimeError('HALT_ENTRIES must remain set')
    key=os.environ.get('ALPACA_PAPER_KEY','');secret=os.environ.get('ALPACA_PAPER_SECRET','')
    if not key.startswith('PK') or not secret:return {'mode':'GET_ONLY','certified':False,'reason':'paper_credentials_unavailable'}
    headers={'APCA-API-KEY-ID':key,'APCA-API-SECRET-KEY':secret}
    result={'mode':'GET_ONLY','certified':False,'halt_flag':True,'orders_sent':0}
    for path,label in (('/v2/account','account'),('/v2/positions','positions'),('/v2/orders?status=open','open_orders')):
        req=urllib.request.Request(endpoint+path,headers=headers,method='GET')
        with urllib.request.urlopen(req,timeout=8) as f:data=json.load(f)
        result[label]=data.get('status') if label=='account' else len(data)
    result['ready_for_future_supervised_test']=result['account']=='ACTIVE' and result['positions']==0 and result['open_orders']==0
    return result


if __name__=='__main__':print(json.dumps(preflight(),indent=2,sort_keys=True))

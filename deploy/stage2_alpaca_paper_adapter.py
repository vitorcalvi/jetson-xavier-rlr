"""Alpaca PAPER bridge implementing the Stage2 controller broker interface.

This adapter does not initiate orders on import. It never contacts a live
account and never retries an uncertain order POST automatically.
"""
from __future__ import annotations
from urllib.parse import quote
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from reflective_loop_runtime.trading.broker import PaperBroker,PAPER_BASE_URL
from reflective_loop_runtime.trading.http import _default_http,_default_post

from rlr_stage2_autonomy import ExplicitPaperOrderReject

class Stage2AlpacaPaperAdapter:
    base_url = PAPER_BASE_URL
    def __init__(self,key: str,secret: str):
        if not key.startswith('PK'):
            raise ValueError('paper key required')
        self.api_key_id = key
        self._api = PaperBroker(key,secret,http_get=_default_http,http_post=_default_post,time_in_force='day')

    def get_positions(self):
        return self._api.positions()

    def get_orders(self,status='open'):
        if status!='open':
            raise ValueError('only complete open-order snapshots supported')
        return self._api.open_orders()

    def get_order(self,order_id):
        return self._api.get_order(order_id)

    def get_order_by_client_order_id(self,client_order_id):
        try:
            return self._api.get_order_by_client_id(client_order_id)
        except RuntimeError as exc:
            # The stock GET transport wraps HTTP 404 in RuntimeError. Only
            # exact 404 may be interpreted as 'no such order'.
            if 'status=404' in str(exc) or 'http_status=404' in str(exc):return None
            raise

    def post_order(self,payload):
        if payload.get('time_in_force') != 'day' or payload.get('symbol') != 'SPY':
            raise ValueError('this diagnostic requires DAY SPY paper orders')
        if payload.get('side') not in ('buy','sell') or payload.get('type') not in ('market','stop'):
            raise ValueError('unapproved order shape')
        if not payload.get('client_order_id'):
            raise ValueError('idempotency key required')
        try:
            return _default_post(PAPER_BASE_URL+'/v2/orders',self._api._headers(),payload)
        except RuntimeError as exc:
            # Only explicit validation 400/403/422 is definite. HTTP 5xx,
            # timeout, TLS failure, etc., remain ambiguous and fail closed.
            import re
            match=re.search(r"status=(400|403|422)\b",str(exc))
            if match:raise ExplicitPaperOrderReject(int(match.group(1))) from exc
            raise

    def cancel_order(self,order_id):
        if not order_id or '/' in order_id or '?' in order_id:
            raise ValueError('invalid order identifier')
        current=self.get_order(order_id)
        if current.get('status') in ('canceled','filled','expired'):
            return
        req=Request(PAPER_BASE_URL+'/v2/orders/'+quote(order_id,safe=''),method='DELETE')
        for name,value in self._api._headers().items():req.add_header(name,value)
        try:
            with urlopen(req,timeout=8) as response:
                if response.status != 204:
                    raise RuntimeError('paper order cancellation not confirmed')
        except HTTPError as exc:
            # A concurrent stop fill can win the cancellation race; look up
            # actual terminal state before claiming an error or selling.
            state=self.get_order(order_id)
            if state.get('status') not in ('canceled','filled','expired'):
                raise

    def account_active(self):
        return self._api.account_status()['active']

    def market_is_open(self):
        return self._api.market_is_open()

    def market_headers(self):
        return self._api._headers()

from reflective_loop_runtime.trading.autonomy_smoke_1m import decide_autonomy_smoke_1m,SMOKE_LIMITS
from reflective_loop_runtime.trading.model import Limits,Position
from reflective_loop_runtime.trading.types import Bar,Quote,Snapshot

L=Limits(**SMOKE_LIMITS)
def snap(*,now=180.,bid=100.,ask=100.03,quote_time=None,bar_time=60.):
    return Snapshot("SPY",(Bar(ts=bar_time,o=100.,h=100.1,l=99.9,c=100.,v=1000.),),
                    Quote(bid=bid,ask=ask,ts=now if quote_time is None else quote_time),
                    now,bar_time)

def test_entry_uses_completed_one_minute_candle_and_fresh_quote():
    d=decide_autonomy_smoke_1m(snap(),None,L,now=180.)
    assert d.action=="enter_long" and d.notional==25 and d.stop==98 and d.target==102

def test_rejects_stale_crossed_or_zero_quotes():
    for kwargs in (dict(bid=100,ask=0),dict(bid=100,ask=99),dict(bid=100,ask=101),
                   dict(quote_time=110),dict(bar_time=175)):
        d=decide_autonomy_smoke_1m(snap(**kwargs),None,L,now=180.)
        assert d.action=="hold",kwargs

def test_automatic_exit_after_two_minutes_without_pnl_conditions():
    pos=Position(symbol="SPY",qty=.25,entry_px=100,stop=98,target=102,entry_ts=100,atr=.1)
    assert decide_autonomy_smoke_1m(snap(),pos,L,now=180.).action=="hold"
    d=decide_autonomy_smoke_1m(snap(),pos,L,now=220.)
    assert d.action=="exit" and d.reason=="autonomy_1m_time_exit"

def test_smoke_limit_and_scope():
    assert L.max_entries_per_day==1 and L.max_concurrent==1 and L.notional==25.0

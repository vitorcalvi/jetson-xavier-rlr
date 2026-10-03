#!/usr/bin/env python3
import argparse, json, os, random, tempfile, time, hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
CAND=Path(os.environ.get('PHASEB_CANDIDATE','/models/rlr/governor-candidates/hardening-1cdb094'))
import sys
sys.path[:0]=[str(CAND/'bench'),str(CAND/'rlr')]

import kapro_ablation as ka
import kapro_rlr as kr
import round3_baselines as rb
import round3_forecast as rf
from rlr.actions import action_json_schema, validate_action, InvalidAction
from rlr.llm import LlamaServer
from rlr.runtime import Config, ReflectiveLoop, REPORT_SCHEMA

TEMP_FLAG=70.0

def max_temp():
    vals=[]
    for p in Path('/sys/class/thermal').glob('thermal_zone*/temp'):
        try:
            typ=(p.parent/'type').read_text().strip()
            if 'PMIC' in typ: continue
            vals.append(int(p.read_text().strip())/1000.0)
        except Exception: pass
    return max(vals) if vals else None

class LoggedLLM:
    def __init__(self):
        self.inner=LlamaServer('http://127.0.0.1:8080',model='Qwen2.5-1.5B',timeout=90)
        self.calls=[]
    def complete(self,prompt,schema,schema_name):
        before=max_temp()
        if before is not None and before>=TEMP_FLAG:
            raise RuntimeError('temperature_flag_pre:%.1f'%before)
        t=time.time(); raw=self.inner.complete(prompt,schema,schema_name); after=max_temp()
        self.calls.append({'schema':schema_name,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                           'prompt':prompt,'temp_before_c':before,'temp_after_c':after,
                           'latency_ms':round((time.time()-t)*1000,2)})
        return raw

def public_cases(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def run_knowing(case,llm):
    raw=llm.complete(kr.knowing_prompt(case),kr.KNOWING_SCHEMA,'phaseb_knowing')
    value,err=kr.parse_knowing(raw)
    return value or {'required_actions':None},err,raw

def fixture_sample(case):
    return kr._Fixture(case['context'],time.monotonic)()

def b0f(case,llm):
    with tempfile.TemporaryDirectory(prefix='phaseb-b0f-') as td:
        cfg=Config(data_dir=td,db_path=str(Path(td)/'x.db'),tick_seconds=0,self_report_every=0,reflection_minutes=10**6)
        provider=kr._Fixture(case['context'],time.monotonic)
        loop=ReflectiveLoop(llm,config=cfg,telemetry_provider=provider,clock=time.monotonic,sleep=lambda _:None)
        try:
            loop.settings.update(case['context'].get('settings') or {})
            sample=provider()
            return rb.run_b0f(case,llm,loop.policy,sample,loop.settings)
        finally: loop.close()

def b1(case,llm):
    knowing,err,kraw=run_knowing(case,llm)
    with tempfile.TemporaryDirectory(prefix='phaseb-b1-') as td:
        cfg=Config(data_dir=td,db_path=str(Path(td)/'x.db'),tick_seconds=0,self_report_every=0,reflection_minutes=10**6)
        provider=kr._Fixture(case['context'],time.monotonic)
        loop=ReflectiveLoop(llm,config=cfg,telemetry_provider=provider,clock=time.monotonic,sleep=lambda _:None)
        try:
            kr.seed_store(loop,case); loop.store.add_goal(case['task'])
            sample=provider(); ctx=loop._bounded(loop.context(sample)); prompt=loop._action_prompt(ctx)
            raw=llm.complete(prompt,action_json_schema(),'phaseb_b1_action')
            try:
                act=validate_action(raw); names=[act['action']]; invalid=False
            except (InvalidAction,ValueError,TypeError,KeyError):
                names=[]; invalid=True
            return {'knowing':knowing,'knowing_error':err,'knowing_raw':kraw,'E_act':names,'raw_action':raw,'invalid':invalid}
        finally: loop.close()

def full_acting(case,llm):
    with tempfile.TemporaryDirectory(prefix='phaseb-b2-') as td:
        cfg=Config(data_dir=td,db_path=str(Path(td)/'x.db'),tick_seconds=0,self_report_every=0,reflection_minutes=10**6)
        provider=kr._Fixture(case['context'],time.monotonic)
        loop=ReflectiveLoop(llm,config=cfg,telemetry_provider=provider,clock=time.monotonic,sleep=lambda _:None)
        try:
            kr.seed_store(loop,case); loop.store.add_goal(case['task'])
            loop._kapro_marker=loop.store.db.execute('SELECT COALESCE(MAX(id),0) FROM actions').fetchone()[0]
            loop._tick(); return kr.collect_trace(loop)
        finally: loop.close()

def full_case(case,llm):
    knowing,err,kraw=run_knowing(case,llm); trace=full_acting(case,llm)
    return {'knowing':knowing,'knowing_error':err,'knowing_raw':kraw,'E_act':trace.get('E_act') or [],
            'E_vetoed':trace.get('E_vetoed') or [],'E_invalid':trace.get('E_invalid') or [],'trace':trace}

def action_mode(args):
    cases=public_cases(args.cases); out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    llm=LoggedLLM()
    with out.open('w') as f:
        for original in cases:
            case=ka.apply_condition(original,'no_self_model') if args.condition=='NO_SELF_MODEL' else original
            started=time.time(); before=max_temp(); valid=True; error=None; ann={}
            try:
                if args.condition=='B0':
                    r=rb.run_b0(case,llm); eact=r.get('E_act') or []; know=r.get('knowing') or {'required_actions':None}; invalid=False
                elif args.condition=='B0F':
                    r=b0f(case,llm); eact=r.get('E_act') or []; know=r.get('knowing') or {'required_actions':None}; invalid=bool(r.get('invalid'))
                elif args.condition=='B1':
                    r=b1(case,llm); eact=r.get('E_act') or []; know=r.get('knowing') or {'required_actions':None}; invalid=bool(r.get('invalid'))
                elif args.condition in ('B2','B2n','NO_SELF_MODEL'):
                    r=full_case(case,llm); eact=r.get('E_act') or []; know=r.get('knowing') or {'required_actions':None}; invalid=bool(r.get('E_invalid'))
                    if args.condition=='B2n': ann['b2n_action_track_manipulation_active']=False
                else: raise ValueError('bad condition')
                if know.get('required_actions') is None or invalid: valid=False
            except Exception as exc:
                r={}; eact=[]; know={'required_actions':None}; valid=False; error='%s:%s'%(type(exc).__name__,exc)
            after=max_temp(); peak=max([x for x in (before,after) if x is not None],default=None)
            if peak is not None and peak>=TEMP_FLAG: valid=False; error=error or 'temperature_flag'
            row={'case_id':original['case_id'],'seed':original['seed'],'condition':args.condition,'boundary':original['boundary'],
                 'knowing_action_set':know.get('required_actions'),'executed_action_set':eact,'valid':valid,'error':error,
                 'temp_before_c':before,'temp_after_c':after,'temp_peak_sampled_c':peak,
                 'latency_ms':round((time.time()-started)*1000,2),**ann}
            f.write(json.dumps(row,sort_keys=True)+'\n'); f.flush()
    calls=[{k:v for k,v in c.items() if k!='prompt'} for c in llm.calls]
    Path(str(out)+'.calls.json').write_text(json.dumps(calls,indent=2,sort_keys=True)+'\n')

def selfstate_one(condition,seed,llm):
    ctx=kr.kc.base_context(random.Random(seed)) if hasattr(kr,'kc') else None
    # build telemetry directly from benchmark context helper
    import kapro_cases as kc
    ctx=kc.base_context(random.Random(seed)); provider=kr._Fixture(ctx,time.monotonic)
    with tempfile.TemporaryDirectory(prefix='phaseb-self-') as td:
        cfg=Config(data_dir=td,db_path=str(Path(td)/'x.db'),tick_seconds=0,self_report_every=0)
        loop=ReflectiveLoop(llm,config=cfg,telemetry_provider=provider,clock=time.monotonic,sleep=lambda _:None)
        try:
            target='discrimination_probe_%d'%seed
            if condition in ('B2','B2n'):
                loop.tick_count=seed; loop.store.add_goal('baseline_goal_%d'%seed)
                result=loop.self_report_discrimination(no_anchor=(condition=='B2n'))
                matches=bool(result.get('goal_recall_withheld'))
                self_prompts=[c['prompt'] for c in llm.calls if c['schema']=='rlr_self_report']
                withheld=self_prompts[-1] if self_prompts else ''
                return {'seed':seed,'condition':condition,'target':target,'withheld_goal_match':matches,
                        'goal_recall_abstained':result.get('goal_recall_abstained'),
                        'withheld_prompt_contains_target':target in withheld,'result':result}
            # NO_SELF_MODEL: goal exists in world/store but no disclosed call/reflection route is provided.
            loop.store.add_goal(target); sample=provider(); result=loop.self_report(sample,disclose_goals=False)
            report=result.get('report') or {}; self_prompts=[c['prompt'] for c in llm.calls if c['schema']=='rlr_self_report']
            withheld=self_prompts[-1] if self_prompts else ''
            return {'seed':seed,'condition':condition,'target':target,'withheld_goal_match':report.get('active_goal')==target,
                    'goal_recall_abstained':report.get('active_goal') is None,
                    'withheld_prompt_contains_target':target in withheld,'result':result}
        finally: loop.close()

def selfstate_mode(args):
    llm=LoggedLLM(); out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w') as f:
        for seed in [1103,2307,4201]:
            before=max_temp(); started=time.time(); valid=True; err=None
            try: row=selfstate_one(args.condition,seed,llm)
            except Exception as exc:
                row={'seed':seed,'condition':args.condition,'target':'discrimination_probe_%d'%seed,'withheld_goal_match':False}; valid=False; err='%s:%s'%(type(exc).__name__,exc)
            after=max_temp(); peak=max([x for x in (before,after) if x is not None],default=None)
            if peak is not None and peak>=TEMP_FLAG: valid=False; err=err or 'temperature_flag'
            row.update({'valid':valid,'error':err,'temp_before_c':before,'temp_after_c':after,'temp_peak_sampled_c':peak,'latency_ms':round((time.time()-started)*1000,2)})
            f.write(json.dumps(row,sort_keys=True)+'\n'); f.flush()
    calls=[{k:v for k,v in c.items() if k!='prompt'} for c in llm.calls]
    Path(str(out)+'.calls.json').write_text(json.dumps(calls,indent=2,sort_keys=True)+'\n')

def forecast_mode(args):
    rows=rf.read_trace(args.trace); ex=rf.make_examples(rows,horizon_s=60,window_s=60)[:3]
    llm=LoggedLLM(); out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w') as f:
        for i,e in enumerate(ex):
            visible={'t':e['t'],'window':e['window'],'target_t':e['target_t']}
            prompt='Forecast max device temperature in Celsius at target_t, using only the preceding thermal window. Return JSON with forecast_temp_c.\n'+json.dumps(visible,sort_keys=True)
            before=max_temp(); started=time.time(); valid=True; err=None; pred=None
            try:
                raw=llm.complete(prompt,rf.FORECAST_SCHEMA,'rlr_temperature_forecast'); value=json.loads(raw) if isinstance(raw,str) else raw; pred=float(value['forecast_temp_c'])
            except Exception as exc: valid=False; err='%s:%s'%(type(exc).__name__,exc)
            after=max_temp(); peak=max([x for x in (before,after) if x is not None],default=None)
            if peak is not None and peak>=TEMP_FLAG: valid=False; err=err or 'temperature_flag'
            row={'example':i,'valid':valid,'error':err,'model_pred_c':pred,'persistence_pred_c':rf.persistence(e),
                 'linear_pred_c':rf.linear_extrapolation(e),'target_temp_c':e['target_temp_c'],'target_t':e['target_t'],
                 'prompt_contains_target_value':str(e['target_temp_c']) in prompt,'temp_before_c':before,'temp_after_c':after,
                 'latency_ms':round((time.time()-started)*1000,2)}
            f.write(json.dumps(row,sort_keys=True)+'\n'); f.flush()
    calls=[{k:v for k,v in c.items() if k!='prompt'} for c in llm.calls]
    Path(str(out)+'.calls.json').write_text(json.dumps(calls,indent=2,sort_keys=True)+'\n')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['action','selfstate','forecast'],required=True); p.add_argument('--condition'); p.add_argument('--cases'); p.add_argument('--trace'); p.add_argument('--out',required=True); a=p.parse_args()
    if a.mode=='action': action_mode(a)
    elif a.mode=='selfstate': selfstate_mode(a)
    else: forecast_mode(a)
if __name__=='__main__': main()

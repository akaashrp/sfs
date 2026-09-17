"""Bounded startup/memory/scheduling evidence for one serving profile; calibration prompts only.

Every coherent snapshot must report the profile's scheduler config. The evidence rule is the
profile's `bounded_smoke`: `full_prefill` (FCFS unchunked: every scheduled prompt fits whole in its
step) or `chunk_bound` (chunked prefill: no step schedules more than max_num_batched_tokens and a
prompt longer than the budget is observed being prefilled in chunks).
"""
import argparse
import asyncio
import json
import mmap
import os
from pathlib import Path
import time
from types import SimpleNamespace

from scripts.cloud.common import read,write
from scripts.cloud.serving.pool import pool
from scripts.cloud.serving.profiles import profile as load_profile


def evidence(profile, observations):
    """Count the observations proving the profile's scheduling regime; raise on a violation."""
    budget=profile.settings['max_num_batched_tokens'];checks=0;long_prompt=False
    for s in observations:
        scheduled=(s['inflight'] or {}).get('scheduled_tokens_by_request',{})
        if profile.bounded_smoke=='chunk_bound' and sum(scheduled.values())>budget:
            raise AssertionError(('step exceeds max_num_batched_tokens',sum(scheduled.values()),budget))
        for rid,n in scheduled.items():
            r=s['requests'].get(rid)
            if not r or r.get('num_output_processed_tokens',0)!=0:continue
            planned=r['num_computed_tokens'];committed=max(0,planned-n)
            remaining=max(0,r['num_prompt_tokens']-committed)
            if not remaining:continue
            if profile.bounded_smoke=='full_prefill':
                # For an unchunked prefill every scheduled prompt must fit in its entirety.
                assert n>=remaining,(rid,n,remaining);checks+=1
            elif profile.bounded_smoke=='chunk_bound':
                assert n<=budget,(rid,n,budget)
                long_prompt|=r['num_prompt_tokens']>budget
                if n<remaining:checks+=1   # a partial (chunked) prefill step
            else:raise ValueError(f'No bounded smoke rule for the {profile.name} profile')
    if profile.bounded_smoke=='chunk_bound' and not long_prompt:raise ValueError('No prompt longer than the step budget was observed')
    if not checks:raise ValueError(f'No observed {profile.bounded_smoke} scheduling evidence')
    return checks


async def measure(profile, bundle, output, path):
    import msgspec
    from scripts.runs import experiments as exp
    from scripts.runs.ministral3_methodology_stage import wait_drained
    from sfs_core.shared.shared_experiment_helpers import warm_up_instances,build_messages
    from vllm.v1.engine.snapshot_shm import read_snapshot_shm_once
    from vllm.v1.engine import _scheduler_sim
    clients,_,_=exp.load_instances(path)
    await warm_up_instances(list(clients.values()));await wait_drained(clients,timeout_s=120)
    raw=read(path);by_model={r['model_id']:r for r in raw['instances']}
    requests=[exp.ExperimentRequest(**json.loads(line)) for line in (bundle/'qwen/calibration_requests.jsonl').read_text().splitlines()]
    ordered=sorted(requests,key=lambda r:r.prompt_tokens)
    selected=[ordered[round(i*(len(ordered)-1)/7)] for i in range(8)]
    expected=profile.snapshot_config
    observations=[];responses=[];running=True
    async def capture():
        while running:
            for model,row in by_model.items():
                try:
                    with (Path('/dev/shm')/row['snapshot_shm_name']).open('rb') as f:
                        with mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as view:
                            result=read_snapshot_shm_once(SimpleNamespace(buf=view,size=len(view)))
                    if result is None:continue
                    header,payload=result
                    state=msgspec.msgpack.decode(payload)
                    config=state['config']
                    for key,value in expected.items():assert config[key]==value,(model,key,config[key],value)
                    inflight=state.get('inflight_batch') or {}
                    if inflight and (output/f'{model}-snapshot.bin').exists() is False:
                        (output/f'{model}-snapshot.msgpack').write_bytes(payload)
                        from vllm.v1.engine.snapshot_shm import encode_snapshot_shm_header
                        (output/f'{model}-snapshot.bin').write_bytes(encode_snapshot_shm_header(header)+payload)
                    observations.append({'model':model,'version':header.snapshot_version,'config':config,
                        'running':state['num_running'],'waiting':state['num_waiting'],'inflight':inflight,
                        'requests':{k:{n:r.get(n) for n in ('num_prompt_tokens','num_computed_tokens','num_prompt_processed_tokens','num_output_processed_tokens','num_cached_tokens')} for k,r in state['requests'].items()}})
                except FileNotFoundError:pass
            await asyncio.sleep(.05)
    watcher=asyncio.create_task(capture())
    async def model(client):
        async def submit(i,r):
            started=time.perf_counter()
            result=await client.submit_request(messages=build_messages(r.prompt,'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'),
                temperature=0,top_p=1,max_completion_tokens=128,
                extra_body={'request_id':f'{profile.name}-smoke-{client.model_id}-{i}','ignore_eos':True,'chat_template_kwargs':{'enable_thinking':False}})
            assert result.usage.completion_tokens==128
            responses.append({'model':client.model_id,'index':i,'request_id':r.request_id,'usage':result.usage.model_dump(),'elapsed_s':time.perf_counter()-started})
        await asyncio.gather(*(submit(i,r) for i,r in enumerate(selected)))
    try:
        await asyncio.wait_for(asyncio.gather(*(model(c) for c in clients.values())),timeout=180)
        await wait_drained(clients,timeout_s=120)
        running=False;await watcher
        if not observations:raise ValueError('No coherent GPU snapshots')
        checks=evidence(profile,observations)
        report={'status':'PASS_BOUNDED_GPU_FCFS_SMOKE' if profile.name=='fcfs' else 'PASS_BOUNDED_GPU_SMOKE','configuration_id':profile.configuration_id,
            'profile':profile.settings,'snapshot_config':expected,'models':list(by_model),'responses':responses,
            'evidence_rule':profile.bounded_smoke,'evidence_checks':checks,'maximum_running':max(s['running'] for s in observations),
            'observations':observations,'evaluation_started':False,
            'coefficient_status':raw['coefficient_status'] if profile.coefficient_policy=='canonical' else 'Placeholder coefficients; timing qualification pending'}
        if profile.bounded_smoke=='full_prefill':report['full_prefill_checks']=checks
        write(output/'gpu-smoke.json',report)
    finally:
        running=False;watcher.cancel();await asyncio.gather(watcher,return_exceptions=True)
        for c in clients.values():c.close()


def run(profile, bundle, models, output, gpus, indices):
    if profile.bounded_smoke is None:raise ValueError(f'The {profile.name} profile has no bounded GPU smoke rule; qualify it with worker qualify')
    output.mkdir(parents=True,exist_ok=False)
    write(output/'status.json',{'state':'STARTING','pid':os.getpid(),'time':time.time(),'evaluation_started':False})
    try:
        with pool(profile,bundle,models,output,gpus,indices) as (path,hardware,processes):
            asyncio.run(measure(profile,bundle,output,path))
        write(output/'status.json',{'state':'COMPLETE','time':time.time(),'evaluation_started':False})
    except BaseException as error:
        write(output/'status.json',{'state':'FAILED','error':str(error),'time':time.time()});raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',required=True,help='fcfs or chunk8192')
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--models',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',required=True);p.add_argument('--indices',default='0,1,2');a=p.parse_args()
    run(load_profile(a.profile),a.bundle,read(a.models),a.output,a.gpus.split(','),tuple(map(int,a.indices.split(','))))

if __name__=='__main__':main()

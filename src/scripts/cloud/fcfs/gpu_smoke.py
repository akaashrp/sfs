"""Bounded startup/memory/nonchunking evidence; calibration prompts only."""
import argparse
import asyncio
import json
import mmap
import os
from pathlib import Path
import time
from types import SimpleNamespace

from scripts.cloud.common import read,write,digest
from scripts.cloud.fcfs.pool import pool


async def measure(bundle, output, path):
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
                    assert config['chunked_prefill_enabled'] is False
                    assert config['max_num_batched_tokens']==65536 and config['max_num_seqs']==512
                    assert config['max_model_len']==65536 and config['long_prefill_token_threshold']==0
                    inflight=state.get('inflight_batch') or {}
                    if inflight and (output/f'{model}-snapshot.bin').exists() is False:
                        (output/f'{model}-snapshot.msgpack').write_bytes(payload)
                        from vllm.v1.engine.snapshot_shm import encode_snapshot_shm_header
                        (output/f'{model}-snapshot.bin').write_bytes(encode_snapshot_shm_header(header)+payload)
                    observations.append({'model':model,'version':header.snapshot_version,'config':config,
                        'running':state['num_running'],'waiting':state['num_waiting'],'inflight':inflight,
                        'requests':{k:{n:r.get(n) for n in ('num_prompt_tokens','num_computed_tokens','num_prompt_processed_tokens','num_output_processed_tokens')} for k,r in state['requests'].items()}})
                except FileNotFoundError:pass
            await asyncio.sleep(.05)
    watcher=asyncio.create_task(capture())
    async def model(client):
        async def submit(i,r):
            started=time.perf_counter()
            result=await client.submit_request(messages=build_messages(r.prompt,'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'),
                temperature=0,top_p=1,max_completion_tokens=128,
                extra_body={'request_id':f'fcfs-smoke-{client.model_id}-{i}','ignore_eos':True,'chat_template_kwargs':{'enable_thinking':False}})
            assert result.usage.completion_tokens==128
            responses.append({'model':client.model_id,'index':i,'request_id':r.request_id,'usage':result.usage.model_dump(),'elapsed_s':time.perf_counter()-started})
        await asyncio.gather(*(submit(i,r) for i,r in enumerate(selected)))
    try:
        await asyncio.wait_for(asyncio.gather(*(model(c) for c in clients.values())),timeout=180)
        await wait_drained(clients,timeout_s=120)
        running=False;await watcher
        if not observations:raise ValueError('No coherent GPU snapshots')
        # For an unchunked prefill every scheduled prompt must fit in its
        # entirety. The publisher records per-request in-flight scheduled work.
        checks=0
        for s in observations:
            scheduled=(s['inflight'] or {}).get('scheduled_tokens_by_request',{})
            for rid,n in scheduled.items():
                r=s['requests'].get(rid)
                if r and r.get('num_output_processed_tokens',0)==0:
                    planned=r['num_computed_tokens'];committed=max(0,planned-n)
                    remaining=max(0,r['num_prompt_tokens']-committed)
                    if remaining:
                        assert n>=remaining,(rid,n,remaining)
                        checks+=1
        if not checks:raise ValueError('No observed full-prefill scheduling evidence')
        write(output/'gpu-smoke.json',{'status':'PASS_BOUNDED_GPU_FCFS_SMOKE','models':list(by_model),'responses':responses,
            'full_prefill_checks':checks,'maximum_running':max(s['running'] for s in observations),
            'observations':observations,'evaluation_started':False,'coefficient_status':'Placeholder coefficients; timing qualification pending'})
    finally:
        running=False;watcher.cancel();await asyncio.gather(watcher,return_exceptions=True)
        for c in clients.values():c.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--models',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',required=True);p.add_argument('--indices',default='0,1,2');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'status.json',{'state':'STARTING','pid':os.getpid(),'time':time.time(),'evaluation_started':False})
    try:
        with pool(a.bundle,read(a.models),a.output,a.gpus.split(','),tuple(map(int,a.indices.split(',')))) as (path,hardware,processes):
            asyncio.run(measure(a.bundle,a.output,path))
        write(a.output/'status.json',{'state':'COMPLETE','time':time.time(),'evaluation_started':False})
    except BaseException as error:
        write(a.output/'status.json',{'state':'FAILED','error':str(error),'time':time.time()});raise

if __name__=='__main__':main()

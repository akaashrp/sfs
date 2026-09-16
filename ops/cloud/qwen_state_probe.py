#!/usr/bin/env python3
"""Short paired SFS prefix diagnostic: fresh engines versus same-process smoke."""
import argparse
import asyncio
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.cloud.common import read, write, digest
from scripts.cloud.pool import pool
from scripts.cloud.canonical_control import preflight, smoke
from scripts.cloud.worker import parse, run_point, audit_cell


def lane(o):
    from scripts.runs import experiments as exp
    root=Path(o.output); out=root/o.lane; out.mkdir(exist_ok=False)
    os.sched_setaffinity(0,set(range(0,48) if o.lane=='fresh' else range(48,96)))
    bundle=Path(o.bundle); manifest=read(bundle/'bundle.json')
    argv,evidence=preflight(bundle,manifest)
    args=parse(argv); requests,_,_=exp._build_request_set(args)
    write(out/'preflight.json',evidence)
    args.num_requests=o.count; args.utilities=['hard']; args.request_rate_qps=8.6
    gpus=['0','1','2','3'] if o.lane=='fresh' else ['4','5','6','7']
    with pool('qwen',manifest['families']['qwen'],read(o.models),bundle,out,gpus,o.state,bundle/'qwen/length') as (path,hardware,processes):
        if o.lane=='smoked':
            write(out/'status.json',{'state':'SMOKE','time':time.time()})
            asyncio.run(asyncio.wait_for(smoke(argv,bundle,manifest['families']['qwen'],path,out),timeout=200))
        write(out/'ready.json',{'time':time.time(),'history':o.lane})
        end=time.monotonic()+240
        while not (root/'start').exists():
            if time.monotonic()>end:raise TimeoutError('start barrier')
            time.sleep(1)
        clients,costs,metadata=exp.load_instances(path)
        args.per_request_wait_log=[str(out/f'wait_{m}.log') for m in manifest['families']['qwen']['models']]
        write(out/'status.json',{'state':'DIAGNOSTIC_PREFIX','time':time.time()})
        async def prefix():
            # The canonical router performs its own one-token warmup before SHM use.
            # A fresh engine has no published snapshot until that first request.
            folder=out/'prefix';folder.mkdir(exist_ok=False)
            routed=await exp.run_router_experiment(args=args,requests=requests[:o.count],
                instances=clients,instance_costs=costs,instance_metadata=metadata,
                response_map_base_path=folder/'responses.log',
                request_log_base_path=folder/'predicted_waits.log')
            payload={'config':{'request_rate_qps':8.6,'num_requests':o.count,
                'prompt_source':{'data_role':'diagnostic_prefix'}},'router':routed}
            write(folder/'point.json',payload)
            return payload
        result=asyncio.run(asyncio.wait_for(prefix(),timeout=300))
        audit=audit_cell(result,{'requests':o.count,'policy':'hard','qps':8.6})
        write(out/'result.json',{'audit':audit,'summary':result['router']['runs'][0]['summary']})
        end=time.monotonic()+330
        while not (root/'release').exists():
            if time.monotonic()>end:raise TimeoutError('release barrier')
            time.sleep(1)
    write(out/'status.json',{'state':'COMPLETE','time':time.time()})


def coordinator(o):
    root=Path(o.output);root.mkdir(parents=True,exist_ok=False); children=[]
    write(root/'status.json',{'state':'STARTING','time':time.time(),'requests_per_lane':o.count,'qps':8.6,'scope':'Diagnostic only; identical first requests of canonical holdout. Short prefixes cannot establish steady-state capacity.','runner_sha256':digest(__file__)})
    try:
        for name in ('fresh','smoked'):
            with (root/(name+'.log')).open('x') as log:
                children.append(subprocess.Popen([sys.executable,__file__,'--lane',name,'--bundle',o.bundle,'--models',o.models,'--state',o.state,'--output',o.output,'--count',str(o.count)],stdout=log,stderr=subprocess.STDOUT))
        def wait(paths,seconds):
            end=time.monotonic()+seconds
            while not all(p.exists() for p in paths):
                if any(p.poll() is not None for p in children):raise RuntimeError('Lane exited before barrier; inspect logs')
                if time.monotonic()>end:raise TimeoutError('coordinator barrier')
                time.sleep(1)
        wait([root/n/'ready.json' for n in ('fresh','smoked')],300)
        (root/'start').write_text('start\n');write(root/'status.json',{'state':'DIAGNOSTIC_PREFIX','time':time.time()})
        wait([root/n/'result.json' for n in ('fresh','smoked')],310)
        (root/'release').write_text('done\n')
        for child in children:
            if child.wait(timeout=45):raise RuntimeError('Lane failed cleanup')
        write(root/'results.json',{n:read(root/n/'result.json') for n in ('fresh','smoked')})
        write(root/'status.json',{'state':'COMPLETE','time':time.time()})
    except BaseException as e:
        write(root/'status.json',{'state':'FAILED','error':str(e),'time':time.time()});raise
    finally:
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            try:child.wait(timeout=35)
            except subprocess.TimeoutExpired:child.kill();child.wait()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--lane',choices=['fresh','smoked']);p.add_argument('--count',type=int,default=1024,choices=[1024])
    for name in ('bundle','models','state','output'):p.add_argument('--'+name,required=True)
    def stop(signum,frame):raise KeyboardInterrupt('bounded diagnostic stop')
    signal.signal(signal.SIGTERM,stop)
    o=p.parse_args();(lane if o.lane else coordinator)(o)

#!/usr/bin/env python3
"""Bounded serving diagnostic; synthetic fixed-token work, never a paper cell."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from scripts.cloud.common import read, write
from scripts.cloud.pool import pool


async def measure(instances, output):
    from openai import AsyncOpenAI
    async def model(row):
        client = AsyncOpenAI(base_url=row['address']+'/v1', api_key='diagnostic',
                             timeout=90, max_retries=0)
        # Equal requests in both phases; fixed generation length removes EOS variance.
        prompt = ' hello'*4096
        async def request(index, cap):
            started = time.perf_counter()
            first = None
            usage = None
            stream = await client.chat.completions.create(model=row['default_model'],
                messages=[{'role':'system','content':'You are a helpful assistant.'},
                          {'role':'user','content':prompt}],
                temperature=0, top_p=1, max_tokens=cap, stream=True,
                stream_options={'include_usage':True},
                extra_body={'ignore_eos':True,'chat_template_kwargs':{'enable_thinking':False}})
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content and first is None:
                    first = time.perf_counter()
                if chunk.usage is not None:
                    usage = chunk.usage.model_dump()
            elapsed = time.perf_counter()-started
            if first is None or usage is None or usage['completion_tokens']!=cap:
                raise ValueError('Diagnostic did not generate the requested fixed token budget')
            return {'index':index,'ttft_s':first-started,'elapsed_s':elapsed,'usage':usage}
        try:
            await request(-1,16)
            start = time.perf_counter()
            rows = await asyncio.gather(*(request(i,256) for i in range(32)))
            duration = time.perf_counter()-start
            result = {'requests':len(rows),'wall_s':duration,'completion_tps':8192/duration,
                'ttft_p50_s':statistics.median(r['ttft_s'] for r in rows),
                'e2e_p50_s':statistics.median(r['elapsed_s'] for r in rows),'rows':rows}
            write(output/(row['model_id']+'.json'),result)
            return row['model_id'],{k:v for k,v in result.items() if k!='rows'}
        finally:
            await client.close()
    return dict(await asyncio.gather(*(model(row) for row in instances)))


def lane(options):
    root = Path(options.output)
    output = root/options.lane
    output.mkdir(parents=True,exist_ok=False)
    os.sched_setaffinity(0,set(range(0,48) if options.lane=='a' else range(48,96)))
    manifest = read(Path(options.bundle)/'bundle.json')
    write(output/'status.json',{'state':'STARTING_SERVERS','time':time.time()})
    gpus = ['0','1','2','3'] if options.lane=='a' else ['4','5','6','7']
    with pool('qwen',manifest['families']['qwen'],read(options.models),options.bundle,
              output,gpus,options.state,Path(options.bundle)/'qwen/length') as (path,hardware,processes):
        instances = read(path)['instances']
        write(output/'ready.json',{'time':time.time(),'hardware':hardware})
        for phase in ('solo_a','dual'):
            deadline=time.monotonic()+600
            while not (root/(phase+'.start')).exists():
                if any(p.poll() is not None for p in processes):
                    raise RuntimeError('Server exited')
                if time.monotonic()>deadline:
                    raise TimeoutError('Coordinator barrier timed out')
                time.sleep(1)
            if options.lane=='a' or phase=='dual':
                write(output/'status.json',{'state':'MEASURING','phase':phase,'time':time.time()})
                result=asyncio.run(asyncio.wait_for(measure(instances,output/phase),timeout=120))
                write(output/(phase+'.json'),result)
            else:
                write(output/(phase+'.json'),{'state':'IDLE_CONTROL'})
        deadline=time.monotonic()+180
        while not (root/'release-pools').exists():
            if time.monotonic()>deadline:raise TimeoutError('Pool release barrier timed out')
            time.sleep(1)
        write(output/'status.json',{'state':'COMPLETE','time':time.time()})


def coordinator(options):
    root=Path(options.output);root.mkdir(parents=True,exist_ok=False)
    write(root/'status.json',{'state':'STARTING','pid':os.getpid(),'time':time.time(),
        'scope':'Synthetic diagnostic, 32 concurrent requests/model, 4096 repeated words, 256 forced output tokens; not an evaluation cell'})
    children=[]
    try:
        for name in ('a','b'):
            log=(root/(name+'.log')).open('x')
            child=subprocess.Popen([sys.executable,__file__,'--lane',name,'--bundle',options.bundle,
                '--models',options.models,'--state',options.state,'--output',options.output],stdout=log,stderr=subprocess.STDOUT)
            log.close();children.append(child)
        def wait_for(paths,seconds):
            deadline=time.monotonic()+seconds
            while not all(p.exists() for p in paths):
                for child in children:
                    if child.poll() is not None and child.returncode!=0:
                        raise RuntimeError('Diagnostic lane failed; inspect logs')
                if time.monotonic()>deadline:raise TimeoutError('Diagnostic stage timed out')
                time.sleep(1)
        wait_for([root/name/'ready.json' for name in ('a','b')],360)
        for phase in ('solo_a','dual'):
            write(root/'status.json',{'state':'MEASURING','phase':phase,'time':time.time()})
            (root/(phase+'.start')).write_text('start\n')
            wait_for([root/name/(phase+'.json') for name in ('a','b')],150)
        (root/'release-pools').write_text('done\n')
        for child in children:child.wait(timeout=45)
        results={name:{phase:read(root/name/(phase+'.json')) for phase in ('solo_a','dual')} for name in ('a','b')}
        write(root/'results.json',results)
        write(root/'status.json',{'state':'COMPLETE','time':time.time()})
    except BaseException as error:
        write(root/'status.json',{'state':'FAILED','error':str(error),'time':time.time()})
        raise
    finally:
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lane',choices=['a','b'])
    for name in ('bundle','models','state','output'):p.add_argument('--'+name,required=True)
    options=p.parse_args()
    def stop(signum,frame):raise KeyboardInterrupt('Diagnostic time limit or explicit stop')
    signal.signal(signal.SIGTERM,stop)
    (lane if options.lane else coordinator)(options)

"""Run the authorized prefill-throughput reruns only after primary family work.

Prepared but not auto-submitted. The controller must first establish that there
is no higher-priority runnable work; this script also requires completed primary
cells and acquires the same exclusive GPU locks as every other cloud worker.
"""
import argparse
import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
import time

from scripts.cloud.common import ROOT, read, write, validate_bundle, digest, source_hashes
from scripts.cloud.worker import execute


def main():
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['qwen','ministral'],required=True);a=p.parse_args()
    root=Path('/workspace/sfs');state=root/'state'
    campaign_path=ROOT/'scripts/cloud/baseline-campaign-20260916.json';campaign=read(campaign_path)
    primary=state/'baselines'/(a.family+'-20260916')
    if read(primary/'status.json')['state']!='COMPLETE':raise ValueError('Primary campaign is not complete')
    source=source_hashes()
    for cell in campaign['cells']:
        if cell['family']!=a.family:continue
        done=read(state/'completed'/(cell['id']+'.json'))
        if done['status']!='PASS_CELL' or done['source_sha256']!=source or digest(done['point'])!=done['point_sha256']:
            raise ValueError('Higher-priority cell has not been completed and verified')
    # A controller can hold this family for newly requested higher-value work.
    if (state/f'hold-fallback-{a.family}').exists():raise ValueError('Higher-priority work has reserved this family')
    manifest=validate_bundle(root/'bundle')
    manifest['cells']=[{k:c[k] for k in ('id','family','variant','policy','qps','requests')} for c in campaign['fallback_cells'] if c['family']==a.family]
    definition=manifest['families'][a.family]
    definition['policies']=['hard_prefill_tps'];definition['qps']=sorted({c['qps'] for c in manifest['cells']})
    output=state/'baselines'/(a.family+'-prefill-fallback-20260916')
    gpus='0,1,2,3' if a.family=='qwen' else '4,5,6'
    os.sched_setaffinity(0,set(range(0,48) if a.family=='qwen' else range(48,84)))
    options=SimpleNamespace(mode='run',family=a.family,variant='canonical',gpus=gpus,bundle=str(root/'bundle'),state=str(state),
        qualification=str(primary),campaign=str(campaign_path),cells=None)
    output.mkdir(parents=True,exist_ok=False)
    write(output/'status.json',{'state':'RUNNING','pid':os.getpid(),'started':time.time(),'purpose':'User-authorized fallback reruns after primary cells'})
    try:asyncio.run(execute(options,manifest,definition,read(root/'models.json'),output))
    except BaseException as error:
        write(output/'status.json',{'state':'FAILED','error':str(error),'ended':time.time()});raise
    write(output/'status.json',{'state':'COMPLETE','ended':time.time()})

if __name__=='__main__':main()

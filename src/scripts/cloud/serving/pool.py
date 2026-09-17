"""Isolated subset/full Qwen pool for one serving profile, sharing exclusive GPU locks with base runs."""
import contextlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request
import uuid

from scripts.cloud.common import ROOT, write, locks
from scripts.cloud.pool import hardware
from scripts.cloud.serving.profiles import instances, server_argv


@contextlib.contextmanager
def pool(profile, bundle, models, output, gpus, indices=(0,1,2), coefficients=None):
    if len(gpus)!=sum((1,1,2)[i] for i in indices):raise ValueError('GPU count must match selected canonical TP sizes')
    fingerprint=hardware(gpus)
    with locks(Path('/dev/shm')/f'sfs-cloud-locks-{os.getuid()}', ['gpu-'+r[1] for r in fingerprint['gpus']]):
        tag=profile.name.replace('_','')[:6]+uuid.uuid4().hex[:8]
        from scripts.runs.serving_ipc import qwen_ipc_environment
        env=qwen_ipc_environment(output/'ipc',os.environ)
        socks=[];processes=[];logs=[]
        try:
            for i in range(3):
                s=socket.socket();s.bind(('127.0.0.1',0));socks.append(s)
            cfg=instances(profile,bundle,[s.getsockname()[1] for s in socks],tag,coefficients)
            cfg['instances']=[r for i,r in enumerate(cfg['instances']) if i in indices]
            path=output/'instances.json';write(path,cfg);write(output/'hardware.json',fingerprint)
            offset=0
            for index,row in zip(indices,cfg['instances']):
                argv=server_argv(profile,bundle,models[row['model_id']],row,index,output)
                n=(1,1,2)[index];visible=','.join(gpus[offset:offset+n]);offset+=n
                model_env=dict(env,CUDA_VISIBLE_DEVICES=visible,VLLM_USE_V1='1',VLLM_ATTENTION_BACKEND='FLASH_ATTN',VLLM_USE_FLASHINFER_SAMPLER='0',
                    VLLM_PER_REQUEST_WAIT_LOG_PATH=str(output/f"wait_{row['model_id']}.log"))
                write(output/f"server_argv_{row['model_id']}.json",argv)
                log=(output/f"server_{row['model_id']}.log").open('x');logs.append(log)
                socks[index].close()
                processes.append(subprocess.Popen(argv,cwd=ROOT/'src',env=model_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
            deadline=time.monotonic()+600
            for row in cfg['instances']:
                while True:
                    if any(p.poll() is not None for p in processes):raise RuntimeError(f'{profile.configuration_id} server exited; inspect its retained log')
                    try:
                        with urllib.request.urlopen(row['address']+'/v1/models',timeout=2) as response:data=json.load(response)
                        if row['default_model'] in {r['id'] for r in data['data']}:break
                    except (OSError,ValueError):pass
                    if time.monotonic()>deadline:raise TimeoutError('Bounded startup exceeded ten minutes')
                    time.sleep(2)
            yield path,fingerprint,processes
        finally:
            for s in socks:s.close()
            for p in processes:
                if p.poll() is None:
                    try:os.killpg(p.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
            deadline=time.monotonic()+30
            for p in processes:
                try:p.wait(timeout=max(.1,deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGKILL);p.wait()
            for log in logs:log.close()

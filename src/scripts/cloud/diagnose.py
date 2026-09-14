"""Fail early on GPU, IPC, ABI, storage, and NCCL problems before model startup."""
import argparse
import datetime
import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
from scripts.cloud.common import ROOT, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state',required=True);p.add_argument('--output',required=True)
    p.add_argument('--nccl', action='store_true');p.add_argument('--gpus', default='2,3')
    a=p.parse_args(); state=Path(a.state).resolve();state.mkdir(parents=True,exist_ok=True)
    from scripts.runs.serving_ipc import validate_ipc_paths
    ipc=validate_ipc_paths(state/'ipc-check',state/'scratch-check',bind=True)
    import torch
    import vllm
    importlib.import_module('vllm._C')
    importlib.import_module('vllm.v1.engine._scheduler_sim')
    if not torch.cuda.is_available() or torch.cuda.device_count()<4:
        raise RuntimeError('Need at least four usable CUDA GPUs')
    if torch.__version__ != '2.8.0+cu129': raise RuntimeError('Wrong Torch ABI for the frozen wheel')
    if Path(vllm.__file__).resolve().parent.parent != ROOT/'vllm': raise RuntimeError('Wrong editable vLLM')
    if shutil.disk_usage('/dev/shm').free < 1024**3:
        raise RuntimeError('Need at least 1 GiB free /dev/shm; configure container shared memory (recommend 16 GiB)')
    for i in range(torch.cuda.device_count()):
        props=torch.cuda.get_device_properties(i)
        if 'H100' not in props.name or props.total_memory<79_000*1024**2: raise RuntimeError('Expected full H100 80GB')
        x=torch.ones(1024,device=f'cuda:{i}'); assert x.sum().item()==1024
    result={'status':'PASS_GPU_PREFLIGHT','torch':torch.__version__,'vllm':vllm.__version__,
        'vllm_import':vllm.__file__,'ipc':ipc,'disk_free_bytes':shutil.disk_usage(state).free,
        'shm_free_bytes':shutil.disk_usage('/dev/shm').free,
        'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True),
        'topology':subprocess.check_output(['nvidia-smi','topo','-m'],text=True)}
    if a.nccl:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpus)
        subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=2',
            str(ROOT/'src/scripts/cloud/nccl_probe.py')],check=True,timeout=180,env=env)
        result['nccl_pair']=a.gpus
    write(a.output,result)


if __name__=='__main__':main()

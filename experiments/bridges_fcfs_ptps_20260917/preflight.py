"""CPU-only preflight for the Bridges FCFS/hard_prefill_tps launchers.

Runs at the top of every launcher (and in the empty-environment rehearsal) so
argument, path and provenance errors surface before any GPU work. Nothing here
touches a GPU or fabricates GPU evidence.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.cloud.common import ROOT, digest, read, write, validate_bundle, source_hashes, source_digest
from scripts.cloud.fcfs.campaign import apply_campaign
from scripts.cloud.fcfs.config import CONFIG_ID, SETTINGS
from scripts.runs.serving_ipc import validate_ipc_paths

BUNDLE_SHA256 = '4e36359aaa8dce4c833b854a553b5f6afc9aafaa8f57e6ef2b58a49864bbf60d'
VLLM_COMMIT = '28bbf9226ffd607f01006aeece3996250ee3e4dc'
EXTENSIONS = ('_C.abi3.so', '_moe_C.abi3.so', '_flashmla_C.abi3.so', '_flashmla_extension_C.abi3.so', 'cumem_allocator.abi3.so')
POLICY = 'hard_prefill_tps'


def check(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['qualify', 'run'], required=True)
    p.add_argument('--storage', type=Path, required=True); p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--cells', required=True); p.add_argument('--pointer', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    storage = a.storage.resolve()
    bundle = storage/'bundle'
    check(digest(bundle/'bundle.json') == BUNDLE_SHA256, 'Bundle is not the frozen 20260916 bundle')
    manifest = validate_bundle(bundle)
    # Pinned model snapshots resolved from the Bridges HF cache, never downloaded.
    models = read(storage/'models.json')
    for model in manifest['families']['qwen']['models']:
        path = Path(models[model])
        check(path.name == manifest['models'][model]['revision'] and (path/'config.json').is_file(), f'Unpinned or incomplete snapshot: {model}')
        check(path.resolve().is_relative_to('/ocean/projects/cis250162p/aparthas/.cache/huggingface/hub'), f'Snapshot is not the shared Bridges HF cache copy: {model}')
    # Runtime: nested vLLM worktree at the submodule commit with the prebuilt extensions.
    origin = Path(importlib.util.find_spec('vllm').origin).resolve()
    check(origin == (ROOT/'vllm/vllm/__init__.py').resolve(), f'Active vLLM is not the worktree submodule: {origin}')
    vllm_commit = subprocess.check_output(['git', '-C', str(ROOT/'vllm'), 'rev-parse', 'HEAD'], text=True).strip()
    gitlink = subprocess.check_output(['git', '-C', str(ROOT), 'ls-tree', 'HEAD', 'vllm'], text=True).split()[2]
    check(vllm_commit == VLLM_COMMIT == gitlink, f'vLLM worktree {vllm_commit} / gitlink {gitlink} differ from {VLLM_COMMIT}')
    for name in EXTENSIONS:
        check((ROOT/'vllm/vllm'/name).resolve().is_file(), f'Missing compiled extension: {name}')
    # Overlay: exactly the four authorized hard_prefill_tps cells, accepted in this stage's mode.
    applied = apply_campaign(manifest, read(a.campaign), a.stage)
    cells = a.cells.split(',')
    check([c['id'] for c in applied['cells']] == cells, f"Overlay cells {[c['id'] for c in applied['cells']]} differ from launcher cells {cells}")
    check(all(c['policy'] == POLICY and c['requests'] == 16000 for c in applied['cells']), 'Overlay cell identity changed')
    check(applied['families']['qwen']['policies'] == [POLICY] and applied['families']['qwen']['qps'] == [6., 7., 8., 8.3], 'Overlay grid changed')
    # Unix socket pathnames for the worker pool and the bounded 32B smoke stay under 107 bytes.
    ipc = {'worker_pool': validate_ipc_paths(storage/'state/ipc/0123456789/ipc', storage/'state/ipc/0123456789/tmp', bind=True),
           'gpu_smoke': validate_ipc_paths(storage/'state/b99999999/ipc/ipc', storage/'state/b99999999/ipc/tmp', bind=True)}
    for prefix in ('state/ipc', 'state/b99999999'):
        subprocess.run(['rm', '-rf', str(storage/prefix)], check=True)
    # Go toolchain for the mandatory upstream selector differential test.
    go = Path(os.environ['SFS_TEST_GO'])
    check(go.is_file(), f'Missing Go compiler: {go}')
    go_version = subprocess.check_output([str(go), 'version'], text=True).strip()
    setup = storage/'setup'
    for gate in ('tests/gate.json', 'cpu-inputs.json', 'cpu-serving.json', 'fcfs-inputs.json'):
        if (setup/gate).exists():
            ipc[f'existing_{gate}'] = read(setup/gate).get('status')
    pointer = None
    if a.stage == 'run':
        # The qualification pointer is written by job A; at rehearsal time it does not exist yet.
        if a.pointer and a.pointer.exists():
            pointer = read(a.pointer)
            q = Path(pointer['qualification'])
            check((q/'qualification.json').is_file() and Path(pointer['coefficients']).is_file(), 'Pointer names missing qualification evidence')
            check(read(q/'qualification.json').get('configuration_id') == CONFIG_ID, 'Pointer qualification is not the FCFS configuration')
            check(read(q/'qualification.json').get('campaign_sha256') == digest(a.campaign), 'Pointer qualification was recorded against other overlay bytes')
    source = source_hashes()
    write(a.output, {'status': 'PASS_CPU_PREFLIGHT', 'stage': a.stage, 'gpu_executed': False, 'time': time.time(),
        'bundle_sha256': BUNDLE_SHA256, 'campaign': str(a.campaign), 'campaign_sha256': digest(a.campaign), 'cells': cells,
        'configuration_id': CONFIG_ID, 'profile': SETTINGS, 'models': models, 'vllm_commit': vllm_commit, 'vllm_origin': str(origin),
        'sfs_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'source_digest': source_digest(source), 'go_version': go_version, 'ipc': ipc, 'pointer': pointer,
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'host': os.uname().nodename})
    print(json.dumps({'preflight': str(a.output), 'stage': a.stage, 'cells': cells, 'go': go_version}), flush=True)


if __name__ == '__main__':
    main()

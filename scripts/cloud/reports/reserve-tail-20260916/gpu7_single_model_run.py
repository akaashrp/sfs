"""Bounded single-model SFS diagnostic on one GPU with dense read-only snapshot capture.

Runs ONE canonical Qwen engine (0.6B or 8B) on the given GPU and replays, through the
canonical SFS router with one instance, the subset of canonical requests that a completed
control routed to that model. It records the router's per-request wait estimates and the
engine's real TTFT, and captures coherent scheduler snapshots every --snapshot-interval
seconds via read-only mmap of the published SHM segment (no SHM ownership, no writes).

It never touches other GPUs, never modifies the repository, and does not run a control.
Use only with an explicit GPU and CPU affinity; it is a diagnostic, not an evaluation cell.
"""
import argparse
import asyncio
import json
import mmap
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

from scripts.cloud.common import ROOT, read, write
from scripts.cloud.canonical_control import preflight
from scripts.cloud.pool import server_argv, hardware
from scripts.cloud.worker import parse, run_point

HEADER = struct.Struct('<8sIIQQdQdddddddd')


def capture_loop(shm_name, out_dir, interval, stop, log):
    path = Path('/dev/shm') / shm_name
    out_dir.mkdir(exist_ok=True)
    count = 0
    while not stop.is_set():
        try:
            with path.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as view:
                for _ in range(32):
                    before = view[:HEADER.size]; v = HEADER.unpack(before)
                    if v[0] != b'VLLMSHM1' or v[3] % 2:
                        continue
                    payload = view[HEADER.size:HEADER.size + v[6]]
                    if before != view[:HEADER.size]:
                        continue
                    now = time.time_ns()
                    (out_dir / f'{now}.bin').write_bytes(before + payload)
                    with (out_dir / 'index.jsonl').open('a') as f:
                        f.write(json.dumps({'time_ns': now, 'monotonic': time.monotonic(), 'snapshot_version': v[4],
                                            'created_at': v[5], 'payload_bytes': v[6]}) + '\n')
                    count += 1
                    break
        except FileNotFoundError:
            pass
        except Exception as error:  # keep capturing; record the failure
            with (out_dir / 'capture-errors.log').open('a') as f:
                f.write(f'{time.time()} {error!r}\n')
        stop.wait(interval)
    log(f'captured {count} snapshots')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=['qwen3-0.6b', 'qwen3-8b'], required=True)
    ap.add_argument('--gpu', required=True)
    ap.add_argument('--cpus', default='84-95')
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--state', required=True)
    ap.add_argument('--models', required=True)
    ap.add_argument('--subset', required=True, help='JSON with request_ids_arrival_order per model')
    ap.add_argument('--count', type=int, required=True)
    ap.add_argument('--qps', type=float, required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--snapshot-interval', type=float, default=1.0)
    ap.add_argument('--run-timeout', type=float, default=2400)
    o = ap.parse_args()
    lo, hi = (int(x) for x in o.cpus.split('-'))
    os.sched_setaffinity(0, set(range(lo, hi + 1)))
    out = Path(o.output).resolve(); out.mkdir(parents=True, exist_ok=False)
    status = lambda state, **extra: write(out / 'status.json', dict(state=state, time=time.time(), pid=os.getpid(), **extra))
    logf = (out / 'run.log').open('a')
    def log(msg):
        logf.write(f'{time.strftime("%H:%M:%S")} {msg}\n'); logf.flush(); print(msg, flush=True)
    bundle, state = Path(o.bundle).resolve(), Path(o.state).resolve()
    manifest = read(bundle / 'bundle.json')
    definition = manifest['families']['qwen']
    status('PREFLIGHT')
    argv, evidence = preflight(bundle, manifest)
    from scripts.runs import experiments as exp
    args = parse(argv)
    requests, _, _ = exp._build_request_set(args)
    by_id = {r.request_id: r for r in requests}
    subset_ids = read(o.subset)[o.model]['request_ids_arrival_order'][:o.count]
    subset = [by_id[i] for i in subset_ids]
    write(out / 'preflight.json', dict(evidence, subset_count=len(subset), subset_source=str(o.subset),
                                       subset_sha256=__import__('hashlib').sha256(json.dumps(subset_ids).encode()).hexdigest()))
    model_paths = read(o.models)
    from scripts.runs.qwen_baselines import pool_config
    from scripts.runs.serving_ipc import qwen_ipc_environment
    tag = 'reserve' + uuid.uuid4().hex[:8]
    local = state / 'ipc' / tag
    env = qwen_ipc_environment(local, os.environ)
    sock = socket.socket(); sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    cfg = pool_config({}, (port, port, port), tag)
    index = definition['models'].index(o.model)
    row = cfg['instances'][index]
    instances = {'instances': [row], 'serving_profile': cfg['serving_profile'],
                 'instance_costs': {row['instance_id']: cfg['instance_costs'][row['instance_id']]},
                 'cost_units': cfg['cost_units'], 'canonical_manifest': None,
                 'diagnostic': 'single-model reserve-tail diagnostic; not a control'}
    path = out / 'instances.json'; write(path, instances)
    write(out / 'hardware.json', dict(hardware([o.gpu]), cpu_affinity=sorted(os.sched_getaffinity(0))))
    argv_server = server_argv('qwen', model_paths[o.model], row, index, out, bundle / 'qwen/length')
    server_env = dict(env, CUDA_VISIBLE_DEVICES=o.gpu, VLLM_USE_V1='1', VLLM_ATTENTION_BACKEND='FLASH_ATTN',
                      VLLM_USE_FLASHINFER_SAMPLER='0', VLLM_PER_REQUEST_WAIT_LOG_PATH=str(out / f'wait_{o.model}.log'),
                      XDG_CACHE_HOME=str(state / 'cache'), TRITON_CACHE_DIR=str(local / 'triton'), CUDA_CACHE_PATH=str(local / 'nv'))
    write(out / f'server_argv_{o.model}.json', argv_server)
    sock.close()
    server_log = (out / f'server_{o.model}.log').open('x')
    status('STARTING_SERVER')
    proc = subprocess.Popen(argv_server, cwd=ROOT / 'src', env=server_env, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
    stop = threading.Event(); capture = None
    try:
        deadline = time.monotonic() + 600
        while True:
            if proc.poll() is not None:
                raise RuntimeError('Server exited before readiness')
            try:
                with urllib.request.urlopen(row['address'] + '/v1/models', timeout=3) as response:
                    if o.model in {r['id'] for r in json.load(response)['data']}:
                        break
            except (OSError, ValueError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('readiness')
            time.sleep(2)
        log('server ready')
        from sfs_core.shared.shared_experiment_helpers import warm_up_instances
        from scripts.runs.ministral3_methodology_stage import wait_drained
        clients, costs, metadata = exp.load_instances(path)
        asyncio.run(warm_up_instances(list(clients.values())))
        asyncio.run(wait_drained(clients, timeout_s=120))
        capture = threading.Thread(target=capture_loop, args=(row['snapshot_shm_name'], out / 'snapshots', o.snapshot_interval, stop, log), daemon=True)
        capture.start()
        args.utilities = ['hard']; args.num_requests = len(subset); args.request_rate_qps = o.qps
        args.per_request_wait_log = [str(out / f'wait_{o.model}.log')]
        status('RUNNING', requests=len(subset), qps=o.qps)
        started = time.time()
        async def run():
            return await asyncio.wait_for(run_point('qwen', args, subset, clients, costs, metadata, out / 'measure', data_role='diagnostic'), timeout=o.run_timeout)
        result = asyncio.run(run())
        summary = result['router']['runs'][0]['summary']
        log(f'run finished in {time.time()-started:.0f}s: {json.dumps({k: summary.get(k) for k in ("succeeded_requests", "failed_requests")})}')
        write(out / 'result.json', {'summary': summary, 'elapsed_s': time.time() - started, 'requests': len(subset), 'qps': o.qps})
        status('COMPLETE')
    except BaseException as error:
        status('FAILED', error=repr(error)); log(f'FAILED {error!r}')
        raise
    finally:
        stop.set()
        if capture is not None:
            capture.join(timeout=10)
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL); proc.wait()
        server_log.close()
        log('server stopped')


if __name__ == '__main__':
    main()

"""GPU7-only bounded telemetry probe. No evaluation prompts or paper results."""
import asyncio
import json
import os
from pathlib import Path
from dataclasses import replace
import signal
import socket
import subprocess
import time
import urllib.request
import uuid

from scripts.cloud.common import ROOT, read, write, locks
from scripts.cloud.pool import config, hardware, server_argv
from scripts.cloud.worker import arguments, parse, run_point

ROOT_STATE = Path('/workspace/sfs/state')
OUTPUT = ROOT_STATE/'diagnostics/baseline-snapshot-gpu7-attempt2'
BUNDLE = Path('/workspace/sfs/bundle')

async def probe(path, definition, manifest):
    from scripts.runs import experiments as exp
    from scripts.runs.ministral3_methodology_stage import wait_drained, audit_run
    clients, costs, metadata = exp.load_instances(path)
    from sfs_core.shared.shared_experiment_helpers import warm_up_instances
    await warm_up_instances(list(clients.values()))
    await wait_drained(clients, timeout_s=120)
    observations = []
    running = True
    async def capture():
        while running:
            for c in clients.values():
                s = await c.refresh_baseline_state()
                observations.append({'elapsed':time.monotonic(), 'version':s.version, 'age_ms':s.age_ms,
                    'running':s.num_running, 'waiting':s.num_waiting,'inflight':s.inflight_total_tokens})
            await asyncio.sleep(.05)
    watcher = asyncio.create_task(capture())
    try:
        base = arguments(definition, BUNDLE, 'canonical', manifest)
        calibration = ROOT_STATE/'baselines/ministral-20260916/timing_models/methodology_calibration.json'
        deadline = time.monotonic()+600
        while not calibration.exists():
            if time.monotonic()>deadline: raise TimeoutError('Destination calibration not ready for probe')
            await asyncio.sleep(2)
        from scripts.cloud.common import set_option
        base = set_option(base, '--methodology-calibration-json', calibration)
        source = [exp.ExperimentRequest(**json.loads(line)) for line in (BUNDLE/'ministral/calibration_requests.jsonl').read_text().splitlines()]
        # Long, real held-out-from-evaluation prompts exercise chunked prefill.
        selected = sorted((r for r in source if r.prompt_tokens < 110000), key=lambda r:r.prompt_tokens, reverse=True)[:8]
        requests = [replace(r, request_id=f'probe-{i}') for i,r in enumerate(selected)]
        for policy in ('mooncake_prefill','routebalance'):
            args = parse(base)
            args.utilities = [policy]; args.num_requests = len(requests); args.request_rate_qps = 8.7875
            args.max_completion_tokens = 32
            args.per_request_wait_log = [str(OUTPUT/'wait_ministral3-8b.log')]
            write(OUTPUT/'status.json', {'state':'GPU_SMOKE','policy':policy,'pid':os.getpid()})
            payload = await asyncio.wait_for(run_point('ministral', args, requests, clients, costs, metadata,
                                            OUTPUT/policy, data_role='calibration'), timeout=180)
            audit_run(payload['router']['runs'][0], len(requests))
            write(OUTPUT/f'{policy}-summary.json', payload['router']['runs'][0]['summary'])
        write(OUTPUT/'probe.json', {'status':'PASS_BOUNDED_GPU_TELEMETRY', 'requests_per_policy':len(requests),
            'maximum_snapshot_age_ms':max(x['age_ms'] for x in observations),
            'busy_observations_older_than_1s':sum(x['age_ms']>1000 and bool(x['inflight']) for x in observations),
            'observations':observations,'calibration':'Destination calibration; single-model plumbing test only'})
    finally:
        running = False
        watcher.cancel(); await asyncio.gather(watcher,return_exceptions=True)
        for c in clients.values(): c.close()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0,set(range(84,92)))
    write(OUTPUT/'status.json', {'state':'STARTING_SERVER','pid':os.getpid()})
    manifest = read(BUNDLE/'bundle.json'); definition=manifest['families']['ministral']
    models = read('/workspace/sfs/models.json'); tag='debug'+uuid.uuid4().hex[:8]
    from scripts.runs.serving_ipc import qwen_ipc_environment
    env=qwen_ipc_environment(ROOT_STATE/'ipc'/tag,os.environ)
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    cfg=config('ministral',definition,read(BUNDLE/'ministral/bridges_metrics.json'),[port]*3,tag)
    cfg['instances']=[r for r in cfg['instances'] if r['model_id']=='ministral3-8b']
    row=cfg['instances'][0]; path=OUTPUT/'instances.json';write(path,cfg)
    env.update(CUDA_VISIBLE_DEVICES='7',VLLM_USE_V1='1',VLLM_ATTENTION_BACKEND='FLASH_ATTN',VLLM_USE_FLASHINFER_SAMPLER='0',
               VLLM_PER_REQUEST_WAIT_LOG_PATH=str(OUTPUT/'wait_ministral3-8b.log'))
    g=hardware(['7'])
    with locks(Path('/dev/shm')/f'sfs-cloud-locks-{os.getuid()}', ['gpu-'+g['gpus'][0][1]]):
        argv=server_argv('ministral',models[row['model_id']],row,0,OUTPUT,BUNDLE/'ministral/length')
        write(OUTPUT/'server_argv.json',argv)
        sock.close()
        with (OUTPUT/'server.log').open('x') as log:
            process=subprocess.Popen(argv,cwd=ROOT/'src',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            deadline=time.monotonic()+420
            while True:
                if process.poll() is not None: raise RuntimeError('Probe server exited')
                try:
                    with urllib.request.urlopen(row['address']+'/health',timeout=2) as response:
                        if response.status==200: break
                except OSError: pass
                if time.monotonic()>deadline: raise TimeoutError('Probe startup deadline')
                time.sleep(2)
            asyncio.run(probe(path,definition,manifest))
            write(OUTPUT/'status.json',{'state':'COMPLETE','time':time.time()})
        except BaseException as error:
            write(OUTPUT/'status.json',{'state':'FAILED','error':str(error),'time':time.time()})
            raise
        finally:
            if process.poll() is None: os.killpg(process.pid,signal.SIGTERM)
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()

if __name__=='__main__': main()

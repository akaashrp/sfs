"""Qualify and execute auditable cells on a locally owned cloud GPU pool."""
import argparse
import asyncio
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time
import uuid

from scripts.cloud.common import ROOT, digest, read, write, expand, set_option, source_hashes, validate_bundle, locks, portable_cache, portable_routebalance
from scripts.cloud.pool import pool, hardware


def arguments(definition, bundle, variant, manifest, qualification=None):
    argv = list(definition['experiment_argv'])
    if variant != 'canonical':
        flag = '--output-length-model-path' if variant == 'mlp_length' else '--accuracy-model-path'
        argv = set_option(argv, flag, manifest['variants'][variant])
    argv = expand(argv, bundle, {})
    family = 'qwen' if definition['models'][0].startswith('qwen') else 'ministral'
    cache = portable_cache(bundle, family)
    for flag in ('--bucket-dir', '--holdout-cache-dir', '--holdout-bucket-dir'):
        argv = set_option(argv, flag, cache)
    argv = set_option(argv, '--routebalance-predictor-path', portable_routebalance(bundle, family))
    if qualification:
        argv = set_option(argv, '--service-metrics-json', Path(qualification)/'model_metrics.json')
        argv = set_option(argv, '--methodology-calibration-json', Path(qualification)/'timing_models/methodology_calibration.json')
    else:
        # CPU ingestion does not load runtime timing heads.
        while '--methodology-calibration-json' in argv:
            i = argv.index('--methodology-calibration-json'); del argv[i:i+2]
    return argv


def parse(argv):
    from scripts.runs.experiments_sweep import _parse_experiment_args
    return _parse_experiment_args(argv)


def audit_cell(payload, cell):
    from scripts.runs.ministral3_methodology_stage import audit_run
    if len(payload['router']['runs']) != 1:
        raise ValueError('Expected exactly one policy per checkpoint')
    run = payload['router']['runs'][0]
    audit_run(run, cell['requests'])
    if run['utility'] != cell['policy'] or payload['config']['request_rate_qps'] != cell['qps']:
        raise ValueError('Cell identity mismatch')
    rows = run['per_request']
    if {r['request_id'] for r in rows} != {f'req-{i}' for i in range(cell['requests'])}:
        raise ValueError('Missing or duplicate evaluation request identities')
    summary = run['summary']
    if summary.get('failed_requests') != 0 or summary.get('succeeded_requests') != len(rows) or summary.get('system_entry_e2e_ttft_missing_count') != 0:
        raise ValueError('Incomplete requests or end-to-end TTFT')
    arrivals = [r.get('system_entry_offset_s') for r in rows]
    if any(not isinstance(t, (float, int)) or not math.isfinite(t) or t < 0 for t in arrivals):
        raise ValueError('Missing arrival telemetry')
    realized = (len(rows)-1)/(max(arrivals)-min(arrivals))
    if abs(realized/cell['qps'] - 1) > .1:
        raise ValueError('Arrival generator missed the requested rate by more than 10%')
    return {'status': 'PASS_CELL', 'requests': len(rows), 'realized_qps': realized}


async def run_point(family, args, requests, clients, costs, metadata, folder, monitor=None):
    from scripts.runs import experiments as exp
    from scripts.runs.ministral3_methodology_stage import wait_drained
    folder.mkdir(parents=True, exist_ok=False)
    await wait_drained(clients, timeout_s=120)
    function = exp.run_router_experiment
    if family == 'ministral' and args.utilities == ['vllm_sr_latency']:
        from scripts.runs.ministral3_latency import run_selector
        function = run_selector
    result = await asyncio.wait_for(function(args=args, requests=requests, instances=clients,
        instance_costs=costs, instance_metadata=metadata, response_map_base_path=folder/'responses.log',
        request_log_base_path=folder/'predicted_waits.log', trial_monitor=monitor), timeout=10800)
    await wait_drained(clients, timeout_s=120)
    config = {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
    config['instance_metadata'] = metadata
    payload = {'config': config, 'request_set': {'num_requests': len(requests)}, 'router': result}
    # Some argparse metadata contains Paths; preserve them as strings.
    payload = json.loads(json.dumps(payload, default=str))
    write(folder/'point.json', payload)
    return payload


async def calibrate(family, definition, requests, clients, base_args, output):
    """Warm shapes first, measure singleton prefill and loaded service/decode next."""
    from scripts.runs.ministral3_methodology_stage import length_stratified_requests, smoke_requests, wait_drained
    from scripts.prep.fit_methodology_calibration import fit_manifest, load_trace_rows
    from sfs_core.shared.shared_experiment_helpers import build_messages
    from sfs_core.shared.trace_theta import estimate_score_proxy_metrics_from_batch_stats
    probes, sample = length_stratified_requests(requests), smoke_requests(requests, per_bucket=128)
    recorded, services = [], {}

    async def measure(client):
        async def submit(req, rid, tokens):
            reply = await client.submit_request(messages=build_messages(req.prompt, base_args.system_prompt),
                temperature=0, top_p=1, max_completion_tokens=tokens,
                extra_body={'chat_template_kwargs': base_args.chat_template_kwargs, 'request_id': rid})
            if reply.usage is None or not reply.id:
                raise ValueError('Missing measured calibration response/usage')
            recorded.append({'model': client.model_id, 'request_id': req.request_id,
                             'probe_id': rid, 'response_id': reply.id, 'usage': reply.usage.model_dump()})
        for i, req in enumerate(probes):
            await submit(req, f'warm-shape-{client.model_id}-{i}', 1)
            await wait_drained({client.instance_id: client}, timeout_s=120)
        trace = output/f'batch_stats_{client.model_id}.csv'
        await asyncio.sleep(2)
        with trace.open() as stream:
            header = stream.readline(); skip = sum(1 for _ in stream)
        for i, req in enumerate(probes):
            await submit(req, f'prefill-probe-{client.model_id}-{i}', 1)
            await wait_drained({client.instance_id: client}, timeout_s=120)
        semaphore = asyncio.Semaphore(128)
        async def loaded(i, req):
            async with semaphore:
                await submit(req, f'loaded-{client.model_id}-{i}', 8192)
        start = time.monotonic()
        await asyncio.gather(*(loaded(i, r) for i, r in enumerate(sample)))
        elapsed = time.monotonic()-start
        await wait_drained({client.instance_id: client}, timeout_s=120)
        await asyncio.sleep(2)
        frozen = output/f'calibration_trace_{client.model_id}.csv'
        with trace.open() as source, frozen.open('x') as dest:
            next(source)
            for _ in range(skip): next(source)
            dest.write(header)
            shutil.copyfileobj(source, dest)
        rows, _ = load_trace_rows([frozen])
        positive = [r for r in rows if r['prefill'] > 0]
        proxy = estimate_score_proxy_metrics_from_batch_stats(batch_stats_csv_path=frozen, batch_stats_offset=0)
        proxy['prefill_tps'] = sum(r['prefill'] for r in positive)/sum(r['exec'] for r in positive)
        services[client.model_id] = {'service_rate_qps': len(sample)/elapsed, 'num_queries': len(sample),
            'succeeded': len(sample), 'failed': 0, 'elapsed_s': elapsed, 'score_proxy': proxy,
            'service_rate_definition': '512 calibration requests at concurrency 128 / whole-run elapsed; not router capacity',
            'traces': [str(frozen)]}
    await asyncio.gather(*(measure(client) for client in clients.values()))
    write(output/'calibration_responses.json', {'data_role': 'calibration', 'responses': recorded})
    write(output/'model_metrics.json', services)
    write(output/'service_manifest.json', {'data_role': 'calibration', 'serving_profile_verified': True,
        'serving_profile': definition['profile'], 'models': services})
    await asyncio.to_thread(fit_manifest, output/'service_manifest.json', output/'timing_models')


async def execute(options, manifest, definition, model_paths, output):
    from scripts.runs import experiments as exp
    from scripts.runs.ministral3_methodology_stage import smoke_requests, audit_run, wait_drained
    from sfs_core.shared.shared_experiment_helpers import warm_up_instances
    gpus = options.gpus.split(',')
    base = arguments(definition, options.bundle, options.variant, manifest)
    base_args = parse(base)
    source = source_hashes()
    tests_dir = Path(options.state).parent/'setup/tests'
    tests = read(tests_dir/'gate.json')
    if (tests.get('status') != 'PASS_CPU_REGRESSION' or tests.get('source_sha256') != source
            or tests.get('xml_sha256') != digest(tests_dir/'results.xml')):
        raise ValueError('Missing/stale source-bound CPU regression gate; rerun scripts/cloud/test.sh')
    cpu = read(Path(options.state).parent/'setup/cpu-inputs.json')
    if (cpu.get('status') != 'PASS_CPU_INPUTS' or cpu.get('source_sha256') != source
            or cpu.get('bundle_sha256') != digest(Path(options.bundle)/'bundle.json')):
        raise ValueError('Missing/stale destination CPU input verification; rerun prepare cpu')
    serving = read(Path(options.state).parent/'setup/cpu-serving.json')
    if (serving.get('status') != 'PASS_CPU_SERVING' or serving.get('source_sha256') != source
            or serving.get('bundle_sha256') != digest(Path(options.bundle)/'bundle.json')):
        raise ValueError('Missing/stale CPU chat, prediction and scheduler gate; rerun prepare serving')
    calibration = [exp.ExperimentRequest(**json.loads(line)) for line in
        Path(expand(definition['calibration_requests'], options.bundle, {})).read_text().splitlines()]
    length = base[base.index('--output-length-model-path')+1]
    for model in definition['models']:
        path = Path(model_paths[model])
        if path.name != manifest['models'][model]['revision']:
            raise ValueError(f'Unpinned model snapshot: {model}')
        for name in ('config.json','params.json','tokenizer_config.json','tekken.json'):
            frozen = Path(options.bundle)/'tokenizers'/model/name
            if frozen.exists() and digest(path/name) != digest(frozen):
                raise ValueError(f'Model/tokenizer configuration differs from the frozen checkpoint: {model}/{name}')
    qualification = Path(options.qualification).resolve() if options.qualification else output
    if options.mode == 'run':
        validate_release(qualification, options, source)
    with pool(options.family, definition, model_paths, options.bundle, output, gpus, options.state, length) as (instances_path, machine, processes):
        clients, costs, metadata = exp.load_instances(instances_path)
        async def heartbeat():
            while True:
                if any(p.poll() is not None for p in processes):
                    raise RuntimeError('A serving process exited during the workload')
                if shutil.disk_usage(output).free < 20*1024**3:
                    raise RuntimeError('Less than 20 GiB free; stopped to preserve artifacts')
                write(output/'heartbeat.json', {'time': time.time(), 'pid': os.getpid(), 'state': 'RUNNING'})
                await asyncio.sleep(15)
        async def workload():
            await warm_up_instances(list(clients.values()))
            await wait_drained(clients, timeout_s=120)
            if options.mode == 'qualify':
                await calibrate(options.family, definition, calibration, clients, base_args, output)
            argv = arguments(definition, options.bundle, options.variant, manifest, qualification)
            def args_for(policy, rate, count):
                args = parse(argv)
                args.utilities, args.num_requests, args.request_rate_qps = [policy], count, rate
                args.per_request_wait_log = [str(output/f'wait_{m}.log') for m in definition['models']]
                return args
            policies = definition['policies'] if options.variant == 'canonical' else ['hard']
            # Every new pool passes matching smoke, including resumption on the same host.
            smoke = smoke_requests(calibration)
            for policy in policies:
                payload = await run_point(options.family, args_for(policy, 2., 192), smoke, clients, costs, metadata,
                                          output/'smoke'/policy)
                audit_run(payload['router']['runs'][0], 192)
            if options.mode == 'qualify':
                from scripts.runs.capacity_scout import TrialMonitor, classify_trial
                load_probes = []
                for rate in (definition['qps'][0], definition['qps'][-1]):
                    folder = output/'load_probes'/f'{rate:g}'
                    monitor = TrialMonitor(folder/'events.jsonl', duration_s=360, max_outstanding=1024)
                    payload = await run_point(options.family, args_for('shortest_queue', rate, len(calibration)), calibration,
                        clients, costs, metadata, folder, monitor)
                    audit_run(payload['router']['runs'][0])
                    probe = classify_trial(monitor.events, requested_qps=rate)
                    load_probes.append(probe)
                evidence = {str(p.relative_to(output)): digest(p) for p in output.rglob('*')
                            if p.is_file() and (p.suffix in ('.json', '.jsonl') or p.name.startswith('calibration_trace_'))
                            and p.name not in ('heartbeat.json', 'status.json')}
                write(output/'qualification.json', {'status': 'GPU_MEASURED_REVIEW_REQUIRED',
                    'family': options.family, 'variant': options.variant, 'hardware': machine,
                    'source_sha256': source, 'bundle_sha256': digest(Path(options.bundle)/'bundle.json'),
                    'load_probes': load_probes, 'files': evidence,
                    'serving_coefficients': 'Canonical SFS batch coefficients retained; destination residuals require review',
                    'evaluation_started': False})
                return
            cells = [c for c in manifest['cells'] if c['family'] == options.family and c['variant'] == options.variant]
            if options.cells:
                requested = set(options.cells.split(','))
                if not requested.issubset({c['id'] for c in cells}):
                    raise ValueError('Requested cells do not match this family/variant')
                cells = [c for c in cells if c['id'] in requested]
            ledger = Path(options.state)/'completed'
            for cell in cells:
                with locks(Path(options.state)/'cell-locks', [cell['id']]):
                    done = ledger/(cell['id']+'.json')
                    if done.exists():
                        previous = read(done)
                        if previous['bundle_sha256'] != digest(Path(options.bundle)/'bundle.json'):
                            raise ValueError('Completed cell belongs to a different bundle')
                        if previous['source_sha256'] != source:
                            raise ValueError('Completed cell used different source; review before mixing implementations')
                        if digest(previous['point']) != previous['point_sha256']:
                            raise ValueError('Completed point checksum changed')
                        continue
                    validate_release(qualification, options, source_hashes())
                    args = args_for(cell['policy'], cell['qps'], cell['requests'])
                    requests, _, _ = exp._build_request_set(args)
                    if len(requests) != cell['requests']:
                        raise ValueError('Evaluation ingestion budget changed')
                    folder = output/'cells'/cell['id']
                    write(output/'active_cell.json', {'cell': cell, 'started': time.time()})
                    payload = await run_point(options.family, args, requests, clients, costs, metadata, folder)
                    audit = audit_cell(payload, cell)
                    if source_hashes() != source:
                        raise ValueError('Runtime source changed during evaluation')
                    point = folder/'point.json'
                    entry = {**audit, 'cell': cell, 'point': str(point), 'point_sha256': digest(point),
                        'source_sha256': source, 'bundle_sha256': digest(Path(options.bundle)/'bundle.json'),
                        'qualification_sha256': digest(qualification/'qualification.json'), 'hardware': machine}
                    write(folder/'audit.json', entry)
                    write(done, entry)
        watcher = asyncio.create_task(heartbeat())
        task = asyncio.create_task(workload())
        try:
            completed, _ = await asyncio.wait((watcher, task), return_when=asyncio.FIRST_COMPLETED)
            for future in completed: await future
        finally:
            for future in (watcher, task):
                future.cancel()
            await asyncio.gather(watcher, task, return_exceptions=True)
            for client in clients.values(): client.close()


def validate_release(qualification, options, source):
    report, release = read(qualification/'qualification.json'), read(qualification/'release.json')
    if (release.get('status') != 'RELEASED' or release.get('qualification_sha256') != digest(qualification/'qualification.json')
            or not release.get('timing_review') or not release.get('load_review')
            or report['source_sha256'] != source or report['family'] != options.family or report['variant'] != options.variant
            or report['bundle_sha256'] != digest(Path(options.bundle)/'bundle.json')
            or report['hardware'] != hardware(options.gpus.split(','))):
        raise ValueError('Missing/stale destination qualification and reviewed release')
    for name, expected in report['files'].items():
        if digest(qualification/name) != expected:
            raise ValueError(f'Qualification evidence changed: {name}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['qualify', 'run'])
    p.add_argument('--bundle', required=True); p.add_argument('--models', required=True)
    p.add_argument('--state', required=True); p.add_argument('--output', required=True)
    p.add_argument('--family', choices=['qwen', 'ministral'], required=True)
    p.add_argument('--variant', choices=['canonical', 'mlp_quality', 'mlp_length', 'flash_quality'], default='canonical')
    p.add_argument('--gpus', required=True); p.add_argument('--cpus'); p.add_argument('--qualification'); p.add_argument('--cells')
    options = p.parse_args()
    if options.family == 'ministral' and options.variant != 'canonical':
        p.error('Predictor ablations are Qwen only')
    if options.mode == 'run' and not options.qualification: p.error('Run requires destination qualification')
    if options.cpus:
        os.sched_setaffinity(0, {int(c) for c in options.cpus.split(',')})
    manifest = validate_bundle(options.bundle)
    output = Path(options.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    def stop(signum, frame): raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    write(output/'status.json', {'state': 'RUNNING', 'pid': os.getpid(), 'started': time.time(), 'options': vars(options)})
    try:
        asyncio.run(execute(options, manifest, manifest['families'][options.family], read(options.models), output))
    except BaseException as error:
        write(output/'status.json', {'state': 'FAILED', 'error': f'{type(error).__name__}: {error}', 'ended': time.time()})
        raise
    else:
        write(output/'status.json', {'state': 'QUALIFIED_AWAITING_REVIEW' if options.mode == 'qualify' else 'COMPLETE', 'ended': time.time()})


if __name__ == '__main__': main()

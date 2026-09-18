#!/usr/bin/env python3
"""Run only the authorized Ministral SFS cells with the frozen Bridges workload."""
import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import time

from scripts.cloud.canonical_control import gates, request_fingerprint
from scripts.cloud.common import digest, expand, locks, read, source_hashes, validate_bundle, write
from scripts.cloud.pool import pool
from scripts.cloud.worker import arguments, audit_cell, parse, run_point

RATES = (6.0125, 7.8625, 8.7875)
REFERENCE = 'fb268ebd517a0cb0ef83c408b6b9283cee10cc991035271d35c8b5dff5c797fb'


def preflight(bundle):
    from scripts.runs import experiments as exp
    manifest = validate_bundle(bundle)
    definition = manifest['families']['ministral']
    argv = arguments(definition, bundle, 'canonical', manifest)
    args = parse(argv)
    if (Path(args.accuracy_model_path) != bundle/'ministral/quality' or
            Path(args.output_length_model_path) != bundle/'ministral/length'):
        raise ValueError('Canonical Ministral predictors required')
    requests, _, _ = exp._build_request_set(args)
    if len(requests) != 8000 or request_fingerprint(requests) != REFERENCE:
        raise ValueError('Ministral request identities, count, tokens or SLOs changed')
    return manifest, definition, argv, requests


def checked_completed(path, bundle_hash, runner_hash, source):
    if not path.exists():
        return False
    record = read(path)
    if (record['bundle_sha256'] != bundle_hash or record['runner_sha256'] != runner_hash or
            record['source_sha256'] != source or digest(record['point']) != record['point_sha256']):
        raise ValueError('Existing completed cell requires provenance review; refusing rerun')
    return True


def report(bundle, payload, folder):
    from scripts.eval.augment_router_actual_accuracy import augment_file, load_req_map, load_quality_index
    target = folder/'quality.json'
    shutil.copyfile(folder/'point.json', target)
    stats = augment_file(json_path=target,
        req_maps_by_holdout={2000: load_req_map(bundle/'ministral/request_map.csv')},
        quality_index=load_quality_index(bundle/'ministral/scores'), dry_run=False)
    if any(getattr(stats, name) for name in ('skipped_reason', 'missing_req_map',
            'missing_example_id', 'unresolved_model', 'missing_quality')):
        raise ValueError(f'Incomplete quality join: {stats}')
    rows = read(target)['router']['runs'][0]['per_request']
    values = []
    for row in rows:
        gate = row['system_entry_e2e_ttft_slo_met']
        if gate != (row['system_entry_e2e_ttft_ms'] <= row['ttft_slo_ms']):
            raise ValueError('TTFT flag disagrees with measurement')
        value = (row['actual_accuracy'] - payload['config']['lambda_weight']*row['actual_cost'])*gate
        if not math.isfinite(value):
            raise ValueError('Nonfinite utility')
        values.append(value)
    result = {'status': 'PASS', 'requests': len(rows), 'qps': payload['config']['request_rate_qps'],
              'primary_judge': 'pro', 'ontimeutility': statistics.mean(values),
              'ttft_slo_attainment_pct': payload['router']['runs'][0]['summary']['system_entry_e2e_ttft_slo_attainment_pct']}
    write(folder/'result_summary.json', result)
    return result


async def workload(options, manifest, definition, argv, requests, output, source):
    from scripts.runs import experiments as exp
    from scripts.runs.ministral3_methodology_stage import smoke_requests, audit_run, wait_drained
    from scripts.runs.measured_audit import require_complete_ttft
    from sfs_core.shared.shared_experiment_helpers import warm_up_instances
    bundle, state = Path(options.bundle), Path(options.state)
    runner_hash, bundle_hash = digest(Path(__file__)), digest(bundle/'bundle.json')
    models = read(options.models)
    for model in definition['models']:
        path = Path(models[model])
        if path.name != manifest['models'][model]['revision']:
            raise ValueError('Model revision changed')
        for name in ('config.json', 'params.json', 'tokenizer_config.json', 'tekken.json'):
            frozen = bundle/'tokenizers'/model/name
            if frozen.exists() and digest(path/name) != digest(frozen):
                raise ValueError('Model/tokenizer bytes changed')
    calibration = [exp.ExperimentRequest(**json.loads(line)) for line in
        Path(expand(definition['calibration_requests'], bundle, {})).read_text().splitlines()]
    def status(value, **extra):
        write(output/'status.json', {'state': value, 'pid': os.getpid(), 'time': time.time(), **extra})
    with pool('ministral', definition, models, bundle, output, options.gpus.split(','), state,
              bundle/'ministral/length') as (instances, machine, processes):
        clients, costs, metadata = exp.load_instances(instances)
        async def run():
            await warm_up_instances(list(clients.values()))
            await wait_drained(clients, timeout_s=120)
            status('SMOKE')
            args = parse(argv)
            args.utilities, args.num_requests, args.request_rate_qps = ['hard'], 192, max(RATES)
            args.per_request_wait_log = [str(output/f'wait_{m}.log') for m in definition['models']]
            payload = await run_point('ministral', args, smoke_requests(calibration), clients, costs,
                                      metadata, output/'smoke', data_role='calibration')
            audit_run(payload['router']['runs'][0], 192)
            require_complete_ttft(payload['router']['runs'][0])
            for rate in options.qps:
                cell = next(c for c in manifest['cells'] if c['family']=='ministral' and
                    c['variant']=='canonical' and c['policy']=='hard' and c['qps']==rate)
                done = state/'completed'/(cell['id']+'.json')
                with locks(state/'cell-locks', [cell['id']]):
                    if checked_completed(done, bundle_hash, runner_hash, source):
                        continue
                    status('RUNNING_8000', qps=rate)
                    args = parse(argv)
                    args.utilities, args.num_requests, args.request_rate_qps = ['hard'], 8000, rate
                    args.per_request_wait_log = [str(output/f'wait_{m}.log') for m in definition['models']]
                    folder = output/'cells'/cell['id']
                    payload = await run_point('ministral', args, requests, clients, costs, metadata, folder)
                    audit = audit_cell(payload, cell)
                    result = report(bundle, payload, folder)
                    if source_hashes()!=source or digest(Path(__file__))!=runner_hash:
                        raise ValueError('Runtime source changed')
                    record = {**audit, 'cell': cell, 'point': str(folder/'point.json'),
                        'point_sha256': digest(folder/'point.json'), 'bundle_sha256': bundle_hash,
                        'runner_sha256': runner_hash, 'source_sha256': source, 'hardware': machine,
                        'request_identity_slo_sha256': REFERENCE, 'result': result,
                        'scope': 'Canonical SFS only; original coefficients/SLOs; destination SFS smoke'}
                    write(folder/'audit.json', record)
                    write(done, record)
                    write(output/'result_summary.json', {'status':'PARTIAL', 'last_completed': result})
        async def watch():
            while True:
                if any(p.poll() is not None for p in processes):
                    raise RuntimeError('A model server exited')
                if shutil.disk_usage(output).free < 20*1024**3:
                    raise RuntimeError('Insufficient free storage')
                await asyncio.sleep(15)
        tasks = [asyncio.create_task(run()), asyncio.create_task(watch())]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                await task
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for client in clients.values():
                client.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['preflight', 'run'])
    for name in ('bundle', 'state', 'models', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--gpus', choices=['0,1,2,3','4,5,6,7'], required=True)
    p.add_argument('--qps', type=float, nargs='+', choices=RATES, required=True)
    options = p.parse_args()
    if len(set(options.qps)) != len(options.qps):
        p.error('Duplicate rates')
    bundle, state, output = Path(options.bundle).resolve(), Path(options.state).resolve(), Path(options.output).resolve()
    options.bundle, options.state = str(bundle), str(state)
    manifest, definition, argv, requests = preflight(bundle)
    if options.mode == 'preflight':
        write(output, {'status':'PASS_CPU_MINISTRAL_SFS', 'requests':8000, 'qps':options.qps,
                      'request_identity_slo_sha256':REFERENCE, 'runner_sha256':digest(Path(__file__))})
        return
    for name in ('qps8p6', 'qps8p75'):
        if (read(state/'controls'/name/'status.json')['state']!='COMPLETE' or
                read(state/'controls'/name/'result_summary.json')['status']!='PASS'):
            raise ValueError('Both Qwen controls must complete and pass before Ministral')
    source = gates(state.parent/'setup', bundle)
    output.mkdir(parents=True, exist_ok=False)
    write(output/'status.json', {'state':'STARTING_SERVERS','pid':os.getpid(),'time':time.time()})
    try:
        asyncio.run(workload(options, manifest, definition, argv, requests, output, source))
    except BaseException as error:
        write(output/'status.json', {'state':'FAILED','error':str(error),'time':time.time()})
        raise
    write(output/'status.json', {'state':'COMPLETE','time':time.time()})


if __name__ == '__main__':
    main()

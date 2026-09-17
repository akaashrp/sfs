"""Two additive current-SFS controls, using the Bridges sweep producer unchanged."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.cloud.common import ROOT, digest, read, write, source_hashes, validate_bundle, expand
from scripts.cloud.worker import arguments, parse, run_point, audit_cell
from scripts.cloud.pool import pool, parse_remaining_length_rules, remaining_length_provenance

# Paired-run default (reserve-tail-20260916 README section 6): survival median for every running
# 0.6B request, conditioned on the prompt-length bin; 8B and 32B keep the current rule.
PAIRED_RUN_RULES = ['qwen3-0.6b=running_all:0.5:prompt_bin']
# All 16,000 identities and SLOs from the reference used by Bridges job 46116020.
REFERENCE_REQUESTS = '611f7b53a01c71c6fd8bd3d3b67249689d554b62301e9b2d4cbee402a31d4fb2'
REQUEST_FIELDS = ('request_id', 'bucket', 'prompt_tokens', 'latency_slo_ms', 'queue_slo_ms', 'ttft_slo_ms')


def request_fingerprint(requests):
    rows = [[getattr(r, key) for key in REQUEST_FIELDS] for r in requests]
    return hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()


def control_arguments(manifest, bundle):
    argv = arguments(manifest['families']['qwen'], bundle, 'canonical', manifest)
    args = parse(argv)
    if (Path(args.accuracy_model_path) != Path(bundle)/'qwen/quality'
            or Path(args.output_length_model_path) != Path(bundle)/'qwen/length'):
        raise ValueError('Control must use both canonical Qwen predictors')
    return argv


def preflight(bundle, manifest):
    from scripts.runs import experiments as exp
    argv = control_arguments(manifest, bundle)
    requests, _, _ = exp._build_request_set(parse(argv))
    fingerprint = request_fingerprint(requests)
    if len(requests) != 16000 or fingerprint != REFERENCE_REQUESTS:
        raise ValueError('Control workload or SLOs differ from the Bridges reference')
    return argv, {'status': 'PASS_CPU_CONTROL', 'requests': len(requests),
                  'request_identity_slo_sha256': fingerprint, 'reference_slurm_job': '46116020',
                  'policy': 'hard', 'predictors': 'canonical Qwen quality and length'}


def gates(setup, bundle):
    source = source_hashes()
    for name, status in (('tests/gate.json', 'PASS_CPU_REGRESSION'),
                         ('cpu-inputs.json', 'PASS_CPU_INPUTS'),
                         ('cpu-serving.json', 'PASS_CPU_SERVING')):
        gate = read(setup/name)
        if gate.get('status') != status or gate.get('source_sha256') != source:
            raise ValueError(f'Missing or stale destination gate: {name}')
        if name.startswith('tests/'):
            if gate.get('xml_sha256') != digest(setup/'tests/results.xml'):
                raise ValueError('Regression evidence changed')
        elif gate.get('bundle_sha256') != digest(bundle/'bundle.json'):
            raise ValueError('Input bundle differs from the destination gate')
    return source


async def smoke(argv, bundle, definition, instances, output):
    from scripts.runs import experiments as exp
    from scripts.runs.ministral3_methodology_stage import smoke_requests, audit_run, wait_drained
    from scripts.runs.measured_audit import require_complete_ttft
    from sfs_core.shared.shared_experiment_helpers import warm_up_instances
    clients, costs, metadata = exp.load_instances(instances)
    await warm_up_instances(list(clients.values()))
    await wait_drained(clients, timeout_s=120)
    calibration = [exp.ExperimentRequest(**json.loads(line)) for line in
                   Path(expand(definition['calibration_requests'], bundle, {})).read_text().splitlines()]
    args = parse(argv)
    args.utilities, args.num_requests, args.request_rate_qps = ['hard'], 192, 2.
    args.per_request_wait_log = [str(output/f'wait_{m}.log') for m in definition['models']]
    result = await run_point('qwen', args, smoke_requests(calibration), clients, costs, metadata,
                             output/'smoke', data_role='calibration')
    audit_run(result['router']['runs'][0], 192)
    require_complete_ttft(result['router']['runs'][0])


def report(bundle, manifest, output, qps):
    from scripts.eval.augment_router_actual_accuracy import load_req_map, load_quality_index
    from scripts.cloud.collate import observed_qwen_queries, observed_utilities
    points = list((output/'outputs').glob('*_point*.json'))
    if len(points) != 1:
        raise ValueError(f'Expected exactly one control point: {points}')
    payload = read(points[0])
    audit = audit_cell(payload, {'requests': 16000, 'policy': 'hard', 'qps': qps})
    quality = {judge: load_quality_index(bundle/'qwen'/directory) for judge, directory in
               (('pro', 'scores'), ('flash', 'scores_flash'))}
    common = observed_qwen_queries(bundle, manifest['families']['qwen']['models'], quality)
    if len(common) != 15996:
        raise ValueError('Observed judge cohort changed')
    summary = observed_utilities(payload, load_req_map(bundle/'qwen/request_map.csv'), quality, common, 'pro')
    write(output/'result_summary.json', dict(summary, status='PASS', qps=qps,
          raw_point=str(points[0]), raw_sha256=digest(points[0]), audit=audit))


def execute(options):
    bundle, state, output = Path(options.bundle).resolve(), Path(options.state).resolve(), Path(options.output).resolve()
    manifest = validate_bundle(bundle)
    argv, evidence = preflight(bundle, manifest)
    if options.mode == 'preflight':
        write(output, evidence)
        return
    setup = state.parent/'setup'
    source = gates(setup, bundle)
    definition = manifest['families']['qwen']
    models = read(options.models)
    # The final map is only published after all six requested downloads are verified.
    if set(models) != set(manifest['models']):
        raise ValueError('All campaign models must finish downloading before the controls')
    for model, path in models.items():
        if Path(path).name != manifest['models'][model]['revision']:
            raise ValueError(f'Unpinned model snapshot: {model}')
        for name in ('config.json', 'params.json', 'tokenizer_config.json', 'tekken.json'):
            frozen = bundle/'tokenizers'/model/name
            if frozen.exists() and digest(Path(path)/name) != digest(frozen):
                raise ValueError(f'Model configuration differs from bundle: {model}/{name}')
    output.mkdir(parents=True, exist_ok=False)
    write(output/'preflight.json', evidence)
    remaining_length = ({'tables': options.remaining_length_tables,
                         'rules': parse_remaining_length_rules(options.remaining_length_rules or PAIRED_RUN_RULES)}
                        if options.remaining_length_tables else None)
    write(output/'provenance.json', {'source_sha256': source, 'bundle_sha256': digest(bundle/'bundle.json'),
          'qps': options.qps, 'gpus': options.gpus, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
          'remaining_length': remaining_length_provenance(definition, remaining_length),
          'comparison_boundary': 'Same canonical SFS predictors, coefficients and SLOs; destination hardware differs from Bridges'})
    write(output/'status.json', {'state': 'STARTING_SERVERS', 'pid': os.getpid(), 'time': time.time()})
    try:
        with pool('qwen', definition, models, bundle, output, options.gpus.split(','), state,
                  bundle/'qwen/length', remaining_length) as (instances, machine, processes):
            write(output/'status.json', {'state': 'SMOKE', 'pid': os.getpid(), 'time': time.time()})
            asyncio.run(smoke(argv, bundle, definition, instances, output))
            command = [sys.executable, '-m', 'scripts.runs.experiments_sweep', '--sweep', 'qps',
                       '--qps-values', str(options.qps), '--qps-utilities', 'hard', '--output-dir',
                       str(output/'outputs'), '--output-prefix', 'current_sfs_control', *argv,
                       '--instances-config', str(instances)]
            for model in definition['models']:
                command += ['--per-request-wait-log', str(output/f'wait_{model}.log')]
            write(output/'driver_argv.json', command)
            write(output/'status.json', {'state': 'RUNNING_16000', 'pid': os.getpid(), 'time': time.time()})
            with (output/'driver.log').open('x') as log:
                subprocess.run(command, cwd=ROOT/'src', stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=10800)
            report(bundle, manifest, output, options.qps)
        write(output/'status.json', {'state': 'COMPLETE', 'time': time.time()})
        write(output/'completed.json', {'status': 'PASS', 'qps': options.qps, 'requests': 16000})
    except BaseException as error:
        write(output/'status.json', {'state': 'FAILED', 'error': str(error), 'time': time.time()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['preflight', 'run'])
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--state', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--models')
    parser.add_argument('--qps', type=float, choices=[8.6, 8.75], required=True)
    parser.add_argument('--gpus', choices=['0,1,2,3', '4,5,6,7'], required=True)
    parser.add_argument('--remaining-length-tables', help='Directory of <model>.json survival tables built by '
                        'scripts.prep.remaining_length_tables; enables the per-model remaining-length rules (default: current rule)')
    parser.add_argument('--remaining-length-rules', action='append', help='MODEL=MODE[:QUANTILE[:CONDITIONING]] with MODE in '
                        f'off/running_all/exhausted_only and CONDITIONING in model/prompt_bin; repeatable; default {PAIRED_RUN_RULES}')
    execute(parser.parse_args())

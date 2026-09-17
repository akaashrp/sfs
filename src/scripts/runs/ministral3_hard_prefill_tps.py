"""Deferred Bridges side jobs: Ministral SFS with the prefill-TPS wait estimator.

Additive entry point for the three baseline-campaign fallback cells
(``hard_prefill_tps`` at 6.0125, 7.8625 and 8.7875 QPS, 8,000 requests each),
modelled on ``ministral3_latency``. ``prepare`` freezes a side manifest on CPU
over the existing 32-cell Ministral Figure 5 manifest and its measured
calibration stage; ``smoke`` runs 192 calibration requests at 2 QPS; ``sweep``
runs the three cells. Both GPU modes use an existing three-model pool started
by the frozen ``ministral3_router_common.sh`` path and audit every point with
``scripts.cloud.worker.audit_cell`` labelled ``provider=bridges``. The legacy
manifest, the shared policy tuple and the recovery plan are never rewritten.

hard_prefill_tps is not a methodology-scheduler policy: like canonical ``hard``
it runs on the generic wait-time scheduler, with ``prefill_tps_ttft`` reading
each model's measured ``score_proxy.prefill_tps`` from the frozen service
metrics. Points go through ``scripts.cloud.worker.run_point`` (family
``ministral``), so the Ministral reliability wrapper is entered as for every
canonical cell; its scheduler substitution is inert for this policy.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time

from scripts.cloud.common import digest, read, write, source_hashes
from scripts.prep.prepare_methodology_service import MODELS, PROFILE
from scripts.runs.ministral3_latency import QPS as LEGACY_QPS
from scripts.runs.ministral3_methodology_stage import BUCKETS, POLICIES as LEGACY_POLICIES
from scripts.runs.ministral3_methodology_stage import audit_run, smoke_requests, wait_drained

POLICY = 'hard_prefill_tps'
ESTIMATOR = 'prefill_tps_ttft'
ARRIVAL_TIMING = 'absolute_schedule_thread'
PROVIDER = 'bridges'
FAMILY = 'ministral'
REQUESTS = 8000
QPS = tuple(LEGACY_QPS[:3])
SMOKE_REQUESTS, SMOKE_QPS = 192, 2.
CALIBRATION_POOL = 10000
# ops/cloud/ministral_sfs.py REFERENCE: the Vast canonical Ministral SFS cells fingerprint the
# same frozen Bridges evaluation workload (request ids, buckets, prompt tokens, SLOs).
REFERENCE = 'fb268ebd517a0cb0ef83c408b6b9283cee10cc991035271d35c8b5dff5c797fb'
STATUS = 'PASS_CPU_SIDE_MANIFEST'


def cell_id(rate):
    # IDs match scripts/cloud/baseline-campaign-20260916.json fallback_cells.
    return f'{FAMILY}-{POLICY}-{rate:g}-rerun'


def cells():
    return [{'id': cell_id(r), 'family': FAMILY, 'variant': 'canonical', 'policy': POLICY, 'qps': r, 'requests': REQUESTS}
            for r in QPS]


def option(argv, flag):
    """Effective value of a scalar CLI option (the last occurrence wins, as in argparse)."""
    value = None
    for index, item in enumerate(argv[:-1]):
        if item == flag:
            value = argv[index+1]
    return value


def prefill_tps_by_model(metrics_path):
    metrics = read(metrics_path)
    models = metrics.get('models', metrics)
    values = {}
    for model in MODELS:
        value = (models.get(model) or {}).get('score_proxy', {}).get('prefill_tps')
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'Ministral service metrics lack a positive prefill_tps for {model}')
        values[model] = float(value)
    if len(set(values.values())) != len(MODELS):
        raise ValueError('Ministral prefill_tps values must be distinct per model')
    return values


def calibration_reuse(base, calibration):
    """Reuse the measured stage behind the frozen manifest, bound to its authoritative calibration bytes."""
    stage = Path(base['stage_dir'])
    started = read(stage/'stage_started.json')
    prepared = Path(started['prepared_dir'])
    meta = read(prepared/'metadata.json')
    if (started.get('data_role') != 'calibration' or started.get('policies') != list(LEGACY_POLICIES)
            or started.get('prepared_metadata_sha256') != digest(prepared/'metadata.json')
            or meta.get('data_role') != 'calibration' or meta.get('holdout_start_index') != 0
            or meta.get('requests_sha256') != digest(calibration)):
        raise ValueError('Calibration requests differ from the measured stage')
    timing = stage/'timing_models/methodology_calibration.json'
    metrics = option(base['experiment_argv'], '--service-metrics-json')
    if option(base['experiment_argv'], '--methodology-calibration-json') != str(timing) or not metrics:
        raise ValueError('Frozen argv does not consume the measured stage calibration and service metrics')
    frozen = base['file_sha256']
    if frozen.get(str(timing)) != digest(timing) or frozen.get(str(Path(metrics).resolve())) != digest(metrics):
        raise ValueError('Stage timing or service metrics are not the frozen Figure 5 inputs')
    return {'stage': str(stage), 'prepared_dir': str(prepared), 'smoke_audit_sha256': digest(stage/'smoke_audit.json'),
            'timing': str(timing), 'timing_sha256': digest(timing), 'service_metrics': str(Path(metrics).resolve()),
            'metrics_sha256': digest(metrics), 'prefill_tps_by_model': prefill_tps_by_model(metrics),
            'note': 'Measured Bridges H100 Ministral stage reused unchanged (no calibration rerun); '
                    'hard_prefill_tps reads score_proxy.prefill_tps from these service metrics'}


def load_calibration(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if len(rows) != CALIBRATION_POOL or Counter(r['bucket'] for r in rows) != Counter({b: CALIBRATION_POOL//len(BUCKETS) for b in BUCKETS}):
        raise ValueError('Expected the frozen balanced calibration pool')
    from scripts.runs import experiments as exp
    return [exp.ExperimentRequest(**r) for r in rows]


def evaluation_requests(argv):
    """Same 8,000 request identities and SLOs as the canonical Ministral cells, on the fixed arrival producer."""
    from scripts.runs import experiments as exp
    from scripts.runs.experiments_sweep import _parse_experiment_args
    from scripts.cloud.canonical_control import request_fingerprint
    args = _parse_experiment_args(list(argv))
    if not args.decouple_arrivals or exp.ARRIVAL_TIMING_THREAD != ARRIVAL_TIMING:
        raise ValueError('Fixed dedicated-thread arrival producer required')
    requests, _, _ = exp._build_request_set(args)
    fingerprint = request_fingerprint(requests)
    if (len(requests) != REQUESTS or {r.request_id for r in requests} != {f'req-{i}' for i in range(REQUESTS)}
            or fingerprint != REFERENCE):
        raise ValueError('Ministral request identities, count, tokens or SLOs differ from the canonical reference')
    return args, requests, fingerprint


def check_base(base):
    """The legacy contract stays untouched: eight policies, four loads, 8,000 requests, Ministral profile."""
    if POLICY in base['policies'] or list(base['policies']) != list(LEGACY_POLICIES):
        raise ValueError('Legacy Ministral policy tuple must stay the eight frozen policies without hard_prefill_tps')
    if (list(base['loads']['qps_values'][:len(QPS)]) != list(QPS) or base['requests_per_cell'] != REQUESTS
            or base['serving_profile'] != PROFILE or base['matrix_cells'] != 32):
        raise ValueError('Unexpected Ministral load grid, budget or serving profile')


def prepare(legacy_path, calibration_path, output):
    from scripts.runs import ministral3_figure5 as fig
    legacy, calibration, output = Path(legacy_path).resolve(), Path(calibration_path).resolve(), Path(output).resolve()
    base = fig.load_manifest(legacy)  # verifies every frozen input and policies == POLICIES
    check_base(base)
    reuse = calibration_reuse(base, calibration)
    load_calibration(calibration)
    _, _, fingerprint = evaluation_requests(base['experiment_argv'])
    output.mkdir(parents=True, exist_ok=False)
    files = {**base['file_sha256'], str(legacy): digest(legacy), str(calibration): digest(calibration)}
    manifest = {'schema_version': 1, 'status': STATUS, 'family': FAMILY, 'provider': PROVIDER, 'policy': POLICY,
        'wait_estimator': ESTIMATOR, 'arrival_timing': ARRIVAL_TIMING, 'qps_values': list(QPS),
        'requests_per_cell': REQUESTS, 'matrix_cells': len(QPS), 'cells': cells(), 'seed': base['seed'],
        'serving_profile': PROFILE, 'experiment_argv': list(base['experiment_argv']),
        'calibration_requests': str(calibration), 'calibration_reuse': reuse,
        'legacy_manifest': str(legacy), 'legacy_manifest_sha256': digest(legacy),
        'legacy_policies': list(base['policies']), 'legacy_32_cells_unchanged': True,
        'request_identity_slo_sha256': fingerprint,
        'reference': 'ops/cloud/ministral_sfs.py REFERENCE (Vast canonical Ministral SFS cells on the same frozen workload)',
        'smoke': {'requests': SMOKE_REQUESTS, 'qps': SMOKE_QPS, 'data_role': 'calibration'},
        'file_sha256': files, 'source_sha256': source_hashes(), 'side_source_sha256': digest(Path(__file__)),
        'gpu_smoke_passed': False, 'prepared_at': time.time()}
    write(output/'manifest.json', manifest)
    return manifest


def validate(path, *, verify_files=True):
    m = read(path)
    if (m.get('status') != STATUS or m.get('family') != FAMILY or m.get('provider') != PROVIDER
            or m.get('policy') != POLICY or m.get('wait_estimator') != ESTIMATOR or m.get('arrival_timing') != ARRIVAL_TIMING
            or m.get('qps_values') != list(QPS) or m.get('requests_per_cell') != REQUESTS or m.get('matrix_cells') != len(QPS)
            or m.get('cells') != cells() or m.get('legacy_policies') != list(LEGACY_POLICIES)
            or m.get('request_identity_slo_sha256') != REFERENCE or m.get('serving_profile') != PROFILE
            or not m.get('file_sha256') or not m.get('calibration_reuse', {}).get('prefill_tps_by_model')):
        raise ValueError('Invalid frozen Ministral hard_prefill_tps side manifest')
    if m.get('side_source_sha256') != digest(Path(__file__)):
        raise ValueError('Side module changed since preparation')
    if verify_files:
        for name, expected in m['file_sha256'].items():
            if digest(name) != expected:
                raise ValueError(f'Frozen Ministral input changed: {name}')
        if m['source_sha256'] != source_hashes():
            raise ValueError('Source changed since preparation')
        from scripts.runs import ministral3_figure5 as fig
        legacy = fig.load_manifest(m['legacy_manifest'], verify_files=False)  # bytes verified above
        if legacy['experiment_argv'] != m['experiment_argv'] or digest(m['legacy_manifest']) != m['legacy_manifest_sha256']:
            raise ValueError('Side manifest drifted from the legacy Figure 5 manifest')
        check_base(legacy)
    return m


def pool(instances_config):
    from scripts.runs import experiments as exp
    clients, costs, metadata = exp.load_instances(Path(instances_config))
    if len(clients) != len(MODELS) or {c.model_id for c in clients.values()} != set(MODELS) or metadata.get('serving_profile') != PROFILE:
        for client in clients.values():
            client.close()
        raise ValueError('Wrong Ministral serving pool')
    return clients, costs, metadata


def audit_estimator(payload, expected_requests, prefill_tps):
    """The point ran hard_prefill_tps on the live prefill-TPS estimator fed by the frozen Ministral
    service metrics, with arrivals from the dedicated-thread absolute schedule."""
    from scripts.runs.measured_audit import require_complete_ttft
    router = payload['router']
    if len(router.get('runs', [])) != 1:
        raise ValueError('Expected exactly one policy per point')
    run = router['runs'][0]
    if run.get('utility') != POLICY or run.get('wait_estimator') != ESTIMATOR:
        raise ValueError('Point did not run hard_prefill_tps with the prefill-TPS estimator')
    if run.get('arrival_timing') != ARRIVAL_TIMING:
        raise ValueError('Arrivals were not generated by the dedicated-thread absolute schedule')
    if payload['config'].get('instance_metadata', {}).get('serving_profile') != PROFILE:
        raise ValueError('Point was not served on the Ministral profile')
    resolved = router.get('prefill_tps_by_instance') or {}
    if len(resolved) != len(MODELS) or sorted(resolved.values()) != sorted(prefill_tps.values()):
        raise ValueError('Estimator prefill TPS does not match the frozen Ministral service metrics')
    audit_run(run, expected_requests)
    require_complete_ttft(run)
    if any(row.get('wait_estimator') != ESTIMATOR for row in run['per_request']):
        raise ValueError('A request was routed without the prefill-TPS estimator')
    return {'utility': POLICY, 'wait_estimator': ESTIMATOR, 'arrival_timing': ARRIVAL_TIMING,
            'prefill_tps_by_instance': dict(resolved)}


def provenance():
    try:
        gpus = subprocess.check_output(['nvidia-smi', '-L'], text=True).strip().splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        gpus = [f'unavailable: {error}']
    return {'provider': PROVIDER, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'host': socket.gethostname(), 'gpus': gpus}


def audit_point(payload, point, cell, manifest, legacy):
    """Per-cell audit as the cloud worker performs it, plus the Figure 5 consumer contract and the estimator path."""
    from scripts.cloud.worker import audit_cell
    from scripts.runs import ministral3_figure5 as fig
    entry = {**provenance(), 'cell': cell, 'serving_profile': PROFILE, 'audited_at': time.time()}
    entry.update(audit_cell(payload, cell))
    entry.update(audit_estimator(payload, cell['requests'], manifest['calibration_reuse']['prefill_tps_by_model']))
    fig.audit_points([Path(point)], {**legacy, 'loads': {**legacy['loads'], 'qps_values': [cell['qps']]}}, [POLICY])
    return entry


def smoke_gate(smoke_dir, manifest_path, manifest):
    gate = read(Path(smoke_dir)/'audit.json')
    point = Path(smoke_dir)/'smoke/point.json'
    if (gate.get('status') != 'PASS_GPU_SMOKE' or gate.get('policy') != POLICY or gate.get('provider') != PROVIDER
            or gate.get('manifest_sha256') != digest(manifest_path) or gate.get('source_sha256') != source_hashes()
            or gate.get('point_sha256') != digest(point)):
        raise ValueError('Missing/stale Ministral hard_prefill_tps GPU smoke')
    audit_estimator(read(point), SMOKE_REQUESTS, manifest['calibration_reuse']['prefill_tps_by_model'])
    return gate


async def run(mode, manifest_path, manifest, instances_config, wait_logs, output, smoke_dir=None):
    from scripts.runs import experiments as exp
    from scripts.runs import ministral3_figure5 as fig
    from scripts.runs.experiments_sweep import _parse_experiment_args
    from scripts.cloud.worker import run_point
    from sfs_core.shared.shared_experiment_helpers import warm_up_instances
    output = Path(output).resolve()
    if mode == 'sweep':
        smoke_gate(smoke_dir, manifest_path, manifest)
    legacy = fig.load_manifest(manifest['legacy_manifest'], verify_files=False)
    output.mkdir(parents=True, exist_ok=False)
    write(output/'run_started.json', {**provenance(), 'mode': mode, 'policy': POLICY, 'manifest_sha256': digest(manifest_path),
        'source_sha256': manifest['source_sha256'], 'instances_sha256': digest(instances_config),
        'calibration_reuse': manifest['calibration_reuse'], 'wait_logs': list(wait_logs)})
    clients, costs, metadata = pool(instances_config)
    try:
        await warm_up_instances(list(clients.values()))
        await wait_drained(clients, timeout_s=120)
        args = _parse_experiment_args(manifest['experiment_argv'])
        args.utilities, args.per_request_wait_log = [POLICY], list(wait_logs)
        if mode == 'smoke':
            requests = smoke_requests(load_calibration(manifest['calibration_requests']))
            args.num_requests, args.request_rate_qps = SMOKE_REQUESTS, SMOKE_QPS
            payload = await run_point(FAMILY, args, requests, clients, costs, metadata, output/'smoke', data_role='calibration')
            estimator = audit_estimator(payload, SMOKE_REQUESTS, manifest['calibration_reuse']['prefill_tps_by_model'])
            write(output/'audit.json', {**provenance(), 'status': 'PASS_GPU_SMOKE', 'policy': POLICY, 'requests': SMOKE_REQUESTS,
                'qps': SMOKE_QPS, 'data_role': 'calibration', 'evaluation_started': False, **estimator,
                'manifest_sha256': digest(manifest_path), 'source_sha256': source_hashes(),
                'point_sha256': digest(output/'smoke/point.json')})
            return
        _, requests, fingerprint = evaluation_requests(manifest['experiment_argv'])
        results, failures = {}, []
        for cell in cells():
            write(output/'status.json', {'state': 'EVALUATING', 'completed': len(results), 'failed': failures, 'current_cell': cell})
            args.request_rate_qps, args.num_requests = cell['qps'], cell['requests']
            folder = output/'cells'/cell['id']
            payload = await run_point(FAMILY, args, requests, clients, costs, metadata, folder)
            point = folder/'point.json'
            entry = {'cell': cell, 'point': str(point), 'point_sha256': digest(point), 'request_identity_slo_sha256': fingerprint}
            try:
                entry.update(audit_point(payload, point, cell, manifest, legacy))
            except ValueError as error:
                entry.update(provenance(), status='FAIL_CELL', error=f'{type(error).__name__}: {error}')
                failures.append(cell['id'])
            write(folder/'audit.json', entry)
            results[cell['id']] = entry
        status = 'PASS_ALL_CELLS' if not failures else 'FAILED_CELLS'
        write(output/'cells_audit.json', {**provenance(), 'status': status, 'policy': POLICY, 'cells': results,
            'failed': failures, 'manifest_sha256': digest(manifest_path), 'source_sha256': source_hashes()})
        write(output/'status.json', {'state': status, 'completed': len(results), 'failed': failures})
        if failures:
            raise ValueError(f'hard_prefill_tps cell audit failed: {failures}')
    except BaseException as error:
        write(output/'failure.json', {'error': f'{type(error).__name__}: {error}', 'time': time.time()})
        raise
    finally:
        for client in clients.values():
            client.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'validate', 'smoke', 'sweep'])
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--legacy-manifest', type=Path)
    p.add_argument('--calibration-requests', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--instances-config', type=Path)
    p.add_argument('--wait-log', action='append', default=[])
    p.add_argument('--smoke-dir', type=Path)
    o = p.parse_args()
    if o.mode == 'prepare':
        if not o.legacy_manifest or not o.calibration_requests:
            p.error('Preparation requires the legacy manifest and the existing calibration request file')
        if o.manifest.name != 'manifest.json':
            p.error('Preparation writes <output>/manifest.json; pass that path as --manifest')
        prepare(o.legacy_manifest, o.calibration_requests, o.manifest.parent)
        print(json.dumps({'status': STATUS, 'manifest': str(o.manifest), 'cells': [c['id'] for c in cells()]}))
        return
    manifest = validate(o.manifest)
    if o.mode == 'validate':
        print(json.dumps({'status': 'PASS_CPU_PREFLIGHT', 'provider': PROVIDER, 'policy': POLICY, 'qps_values': list(QPS),
            'requests_per_cell': REQUESTS, 'request_identity_slo_sha256': manifest['request_identity_slo_sha256'],
            'manifest_sha256': digest(o.manifest), 'gpu_executed': False}))
        return
    if not o.output or not o.instances_config or len(o.wait_log) != len(MODELS):
        p.error('GPU modes require --output, the existing pool config and all three wait logs')
    if o.mode == 'sweep' and not o.smoke_dir:
        p.error('Sweep requires the matching smoke directory')
    asyncio.run(run(o.mode, o.manifest, manifest, o.instances_config, o.wait_log, o.output, o.smoke_dir))
    print(json.dumps({'status': 'PASS_GPU_SMOKE' if o.mode == 'smoke' else 'PASS_ALL_CELLS', 'output': str(o.output)}))


if __name__ == '__main__':
    main()

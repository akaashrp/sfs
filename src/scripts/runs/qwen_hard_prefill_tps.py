"""Deferred Bridges side jobs: SFS with the prefill-TPS wait estimator.

Additive entry point for the baseline-campaign fallback cells
(``hard_prefill_tps`` at 6, 7, 8 and 8.3 QPS, 16,000 requests each) and the
matching one-predictor ablations. It reuses the canonical Qwen manifest, the
measured Bridges calibration, the existing pool launchers and the variant
helpers without editing the protected Qwen sources. The scoped sweep
replacement mirrors ``ministral3_reliable``; every gate of the reused path
(CPU report, reusable calibration, GPU smoke, frozen variant contract) still
runs. Results are labelled ``provider=bridges``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.prep.paper_ablation_data import sha256, write_json
from scripts.runs import qwen_baselines as qwen
from scripts.runs import qwen_predictor_variants as variants

POLICY = 'hard_prefill_tps'
QPS = (6., 7., 8., 8.3)
SMOKE_POLICIES = (*qwen.POLICIES, POLICY)
PROVIDER = 'bridges'
ARMS = ('canonical', *variants.ARMS)
REQUESTS = 16000


def cell_id(arm, rate):
    # Canonical IDs match scripts/cloud/baseline-campaign-20260916.json fallback_cells.
    return f'qwen-{POLICY}-{rate:g}-rerun' if arm == 'canonical' else f'qwen-{arm}-{POLICY}-{rate:g}'


def cells(arm):
    if arm not in ARMS: raise ValueError('Unknown hard_prefill_tps arm')
    return [{'id': cell_id(arm, r), 'family': 'qwen', 'variant': arm, 'policy': POLICY, 'qps': r, 'requests': REQUESTS} for r in QPS]


def preflight(manifest_path, output):
    """Same 16,000 request identities and SLOs as the canonical Vast/Bridges reference."""
    from scripts.runs import experiments as exp
    from scripts.runs.experiments_sweep import _parse_experiment_args
    from scripts.cloud.canonical_control import REFERENCE_REQUESTS, request_fingerprint
    manifest = qwen.validate_manifest(manifest_path)
    requests, _, _ = exp._build_request_set(_parse_experiment_args(manifest['experiment_argv']))
    fingerprint = request_fingerprint(requests)
    if len(requests) != REQUESTS or fingerprint != REFERENCE_REQUESTS:
        raise ValueError('hard_prefill_tps workload or SLOs differ from the canonical reference')
    report = {'status': 'PASS_CPU_PREFLIGHT', 'provider': PROVIDER, 'policy': POLICY, 'qps_values': list(QPS),
              'requests': len(requests), 'request_identity_slo_sha256': fingerprint,
              'reference': 'scripts.cloud.canonical_control.REFERENCE_REQUESTS (Bridges job 46116020 / Vast controls)',
              'manifest_sha256': sha256(manifest_path), 'gpu_executed': False}
    write_json(Path(output), report)
    return report


def run_sweep(manifest, stage, output, instances_path, predictor, timing_gate=None, policies=None, qps_values=None):
    """Scoped replacement for the canonical sweep: exactly the side policy and grid."""
    policies, rates = tuple(policies or (POLICY,)), tuple(qps_values or QPS)
    if policies != (POLICY,) or timing_gate is not None:
        raise ValueError('Side sweep runs exactly hard_prefill_tps without a bootstrap timing gate')
    if len(set(rates)) != len(rates) or not set(rates).issubset(QPS):
        raise ValueError('Invalid hard_prefill_tps QPS subset')
    stage, output = Path(stage), Path(output)
    timing = stage/'timing_models/methodology_calibration.json'
    argv = [sys.executable, '-m', 'scripts.runs.experiments_sweep', '--sweep', 'qps',
        '--qps-values', *map(str, rates), '--qps-utilities', *policies, '--output-dir', str(output/'outputs'),
        '--output-prefix', 'qwen_'+POLICY, *manifest['experiment_argv'], '--instances-config', str(instances_path),
        '--service-metrics-json', str(stage/'model_metrics.json'), '--methodology-calibration-json', str(timing),
        '--routebalance-predictor-path', str(predictor)]
    for model in qwen.MODELS: argv += ['--per-request-wait-log', str(output/f'wait_{model}.log')]
    from scripts.runs.latency_validation import source_hashes
    sources = source_hashes(manifest['sfs_root'])
    write_json(output/'sweep_started.json', {'provider': PROVIDER, 'source_sha256': sources, 'policies': policies,
        'qps_values': rates, 'requests_per_cell': REQUESTS, 'manifest_sha256': sha256(manifest['manifest_path']),
        'stage': str(stage), 'metrics_sha256': sha256(stage/'model_metrics.json'), 'timing_sha256': sha256(timing)})
    write_json(output/'driver_argv.json', argv)
    with (output/'driver.log').open('x') as log:
        subprocess.run(argv, cwd=Path(manifest['sfs_root'])/'src', stdout=log, stderr=subprocess.STDOUT, check=True)
    if source_hashes(manifest['sfs_root']) != sources:
        raise ValueError('Sources changed during hard_prefill_tps evaluation; preserve outputs for review')
    write_json(output/'sweep_completed.json', {'status': 'COMPLETE_UNCOLLATED', 'provider': PROVIDER, 'source_sha256': sources,
        'manifest_sha256': sha256(manifest['manifest_path']), 'timing_sha256': sha256(timing)})


def canonical(mode, options):
    """Smoke: all nine canonical policies plus hard_prefill_tps on the reused measured calibration.
    Sweep: hard_prefill_tps at the side grid, gated on that smoke."""
    if mode not in ('smoke', 'sweep'): raise ValueError('Canonical side modes are smoke and sweep')
    manifest = qwen.validate_manifest(options.manifest)
    ns = SimpleNamespace(mode=mode, predictor=options.predictor, output_root=options.output_root,
        cpu_validation=options.cpu_validation, stage_dir=options.stage_dir,
        policies=list(SMOKE_POLICIES) if mode == 'smoke' else [POLICY], qps_values=list(QPS))
    with patch.object(qwen, 'run_sweep', run_sweep):
        qwen.start_and_run(ns, manifest)


def variant_prepare(options):
    """Freeze the standard one-predictor variant (unchanged helper), then the side contract over it."""
    base, side = Path(options.base_manifest), Path(options.manifest)
    if side.exists(): raise ValueError('Preserve prior side variant manifest')
    variants.prepare(options.root, options.arm, options.canonical, options.stage, options.cpu_report, base, options.flash_root)
    m = variants.validate(base)
    write_json(side, {'status': 'PASS_CPU_VARIANT_SIDE', 'provider': PROVIDER, 'arm': m['arm'], 'policy': POLICY,
        'qps_values': list(QPS), 'requests_per_cell': REQUESTS, 'base_manifest': str(base.resolve()), 'base_sha256': sha256(base),
        'stage': m['stage'], 'paths': m['paths'], 'side_source_sha256': sha256(Path(__file__)), 'cells': cells(m['arm']),
        'gpu_smoke_passed': False})


def variant_validate(path):
    m = json.loads(Path(path).read_text())
    if (m.get('status') != 'PASS_CPU_VARIANT_SIDE' or m.get('policy') != POLICY or m.get('qps_values') != list(QPS)
            or m.get('arm') not in variants.ARMS or m.get('requests_per_cell') != REQUESTS
            or m.get('side_source_sha256') != sha256(Path(__file__)) or m.get('base_sha256') != sha256(m['base_manifest'])):
        raise ValueError('Invalid frozen hard_prefill_tps variant contract')
    base = variants.validate(m['base_manifest'])
    if base['arm'] != m['arm'] or base['paths'] != m['paths'] or base['stage'] != m['stage']:
        raise ValueError('Side manifest differs from its validated base variant')
    canonical_manifest = qwen.validate_manifest(base['canonical_manifest'])
    predictor = Path(base['sfs_root'])/'experiments/paper_ablation_20260907/qwen_routebalance_predictor'
    qwen.validate_smoke(Path(m['stage']), canonical_manifest, predictor, policies=[POLICY])
    return base, m


async def variant_smoke(base, side, manifest_path, instances_path, output):
    from scripts.runs import experiments as exp
    from scripts.runs.experiments_sweep import _parse_experiment_args
    from sfs_core.shared.shared_experiment_helpers import warm_up_instances
    canonical_manifest = json.loads(Path(base['canonical_manifest']).read_text())
    args = _parse_experiment_args(base['experiment_argv']+['--service-metrics-json', str(Path(base['stage'])/'model_metrics.json')])
    args.utilities, args.num_requests, args.request_rate_qps = [POLICY], 192, 2.
    args.per_request_wait_log = [str(output/f'wait_{model}.log') for model in qwen.MODELS]
    requests = qwen.smoke_requests([exp.ExperimentRequest(**r) for r in qwen.rows(canonical_manifest['calibration_requests'])])
    instances, costs, metadata = exp.load_instances(instances_path)
    try:
        await warm_up_instances(list(instances.values())); await qwen.wait_drained(instances)
        result = await asyncio.wait_for(exp.run_router_experiment(args=args, requests=requests, instances=instances,
            instance_costs=costs, instance_metadata=metadata, response_map_base_path=output/'responses.log',
            request_log_base_path=output/'router_waits.log'), timeout=1800)
        write_json(output/f'smoke_{POLICY}.json', result); qwen.audit_run(result['runs'][0], 192); await qwen.wait_drained(instances)
        variant_validate(manifest_path)
        write_json(output/'smoke_audit.json', {'status': 'PASS_GPU_VARIANT_SMOKE', 'provider': PROVIDER, 'arm': side['arm'],
            'policy': POLICY, 'requests': 192, 'evaluation_started': False, 'manifest_sha256': sha256(manifest_path),
            'result_sha256': sha256(output/f'smoke_{POLICY}.json'), 'paths': side['paths']})
    finally:
        for client in instances.values(): client.close()


def variant_validate_smoke(side, manifest_path, stage):
    stage = Path(stage); audit = json.loads((stage/'smoke_audit.json').read_text())
    if (audit.get('status') != 'PASS_GPU_VARIANT_SMOKE' or audit.get('policy') != POLICY or audit.get('arm') != side['arm']
            or audit.get('paths') != side['paths'] or audit.get('manifest_sha256') != sha256(manifest_path)
            or audit.get('result_sha256') != sha256(stage/f'smoke_{POLICY}.json')):
        raise ValueError('Missing or stale hard_prefill_tps variant GPU smoke')
    qwen.audit_run(json.loads((stage/f'smoke_{POLICY}.json').read_text())['runs'][0], 192)


def variant_run(path, mode, output, smoke_stage=None):
    base, side = variant_validate(path); output = Path(output)
    if mode == 'sweep': variant_validate_smoke(side, path, smoke_stage)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'run_started.json', {'provider': PROVIDER, 'arm': side['arm'], 'policy': POLICY, 'mode': mode,
        'manifest_sha256': sha256(path), 'paths': side['paths'], 'qps_values': list(QPS)})
    with variants.pool(base, output) as instances:
        if mode == 'smoke':
            asyncio.run(variant_smoke(base, side, path, instances, output))
        else:
            argv = [sys.executable, '-m', 'scripts.runs.experiments_sweep', '--sweep', 'qps', '--qps-values', *map(str, QPS),
                '--qps-utilities', POLICY, '--output-dir', str(output/'outputs'), '--output-prefix', f'{side["arm"]}_{POLICY}',
                *base['experiment_argv'], '--instances-config', str(instances),
                '--service-metrics-json', str(Path(base['stage'])/'model_metrics.json')]
            for model in qwen.MODELS: argv += ['--per-request-wait-log', str(output/f'wait_{model}.log')]
            write_json(output/'driver_argv.json', argv)
            with (output/'driver.log').open('x') as log:
                subprocess.run(argv, cwd=Path(base['sfs_root'])/'src', stdout=log, stderr=subprocess.STDOUT, check=True)
            variant_validate(path)
            write_json(output/'sweep_completed.json', {'status': 'COMPLETE_UNCOLLATED', 'provider': PROVIDER,
                'arm': side['arm'], 'policy': POLICY, 'manifest_sha256': sha256(path)})


def audit(arm, output, manifest_path, root):
    """Per-cell audits as the cloud worker performs them, labelled provider=bridges."""
    from scripts.cloud.worker import audit_cell
    from scripts.runs.latency_validation import source_hashes
    output = Path(output)
    points = sorted(p for p in (output/'outputs').glob('*_point*.json') if re.search(r'_point\d+\.json$', p.name))
    expected = {c['qps']: c for c in cells(arm)}
    try:
        gpus = subprocess.check_output(['nvidia-smi', '-L'], text=True).strip().splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        gpus = [f'unavailable: {error}']
    results, failures = {}, []
    for path in points:
        payload = json.loads(path.read_text()); rate = payload['config']['request_rate_qps']
        cell = expected.get(rate)
        if cell is None or cell['id'] in results:
            raise ValueError(f'Unexpected or duplicate hard_prefill_tps point: {path}')
        entry = {'provider': PROVIDER, 'cell': cell, 'point': str(path), 'point_sha256': sha256(path),
                 'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'host': socket.gethostname(), 'gpus': gpus,
                 'serving_profile': qwen.PROFILE, 'utility': payload['router']['runs'][0].get('utility'),
                 'wait_estimator': payload['router']['runs'][0].get('wait_estimator'), 'audited_at': time.time()}
        try:
            entry.update(audit_cell(payload, cell))
        except Exception as error:
            entry.update(status='FAIL_CELL', error=f'{type(error).__name__}: {error}'); failures.append(cell['id'])
        (output/'audits').mkdir(exist_ok=True)
        write_json(output/'audits'/f'{cell["id"]}.json', entry); results[cell['id']] = entry
    missing = [c['id'] for c in cells(arm) if c['id'] not in results]
    status = 'PASS_ALL_CELLS' if not failures and not missing else 'FAILED_CELLS'
    write_json(output/'cells_audit.json', {'status': status, 'provider': PROVIDER, 'arm': arm, 'policy': POLICY,
        'cells': results, 'failed': failures, 'missing': missing, 'source_sha256': source_hashes(root),
        'manifest_sha256': sha256(manifest_path)})
    if status != 'PASS_ALL_CELLS':
        raise ValueError(f'hard_prefill_tps cell audit failed: failed={failures} missing={missing}')
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['preflight', 'canonical-smoke', 'canonical-sweep', 'variant-prepare',
                                    'variant-smoke', 'variant-sweep', 'audit'])
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--predictor', type=Path); p.add_argument('--cpu-validation', type=Path)
    p.add_argument('--stage-dir', type=Path); p.add_argument('--output-root', type=Path)
    p.add_argument('--arm', choices=ARMS); p.add_argument('--root', type=Path); p.add_argument('--canonical', type=Path)
    p.add_argument('--stage', type=Path); p.add_argument('--cpu-report', type=Path); p.add_argument('--flash-root', type=Path)
    p.add_argument('--base-manifest', type=Path); p.add_argument('--smoke-stage', type=Path); p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.mode == 'preflight':
        print(json.dumps(preflight(a.manifest, a.output)))
    elif a.mode.startswith('canonical-'):
        if a.predictor is None or a.output_root is None or a.stage_dir is None: p.error('Canonical modes require predictor, stage and output paths')
        canonical(a.mode.removeprefix('canonical-'), a)
    elif a.mode == 'variant-prepare':
        if a.arm in (None, 'canonical') or a.base_manifest is None: p.error('Variant preparation requires a predictor arm and base manifest path')
        variant_prepare(a); print('PASS_CPU_VARIANT_SIDE')
    elif a.mode.startswith('variant-'):
        if a.output is None: p.error('Variant GPU modes require --output')
        variant_run(a.manifest, a.mode.removeprefix('variant-'), a.output, a.smoke_stage)
    else:
        if a.arm is None or a.output is None or a.root is None: p.error('Audit requires --arm, --output and --root')
        audit(a.arm, a.output, a.manifest, a.root); print('PASS_ALL_CELLS')


if __name__ == '__main__': main()

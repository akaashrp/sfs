#!/usr/bin/env python3
"""One Ministral SFS cell at an off-grid arrival rate, to locate where the policy stops meeting SLOs.

SFS leads every Ministral baseline at 6.0125 and 7.8625 QPS and collapses at 8.7875, where its
feasibility filter finds no feasible candidate and its fallback equalizes lateness across the pool
(scripts/cloud/reports/sfs-saturation-fallback-20260918). This runner measures a single rate between
those two so the transition can be located before deciding whether to change the fallback.

It is a diagnostic, never a campaign cell: the rate is outside the authorized grid, so the run writes
only under its own output directory, is labelled `rate_probe`, and never touches the completion ledger
that `collate` reads. Everything else is the campaign's own machinery — the frozen 8,000-request
Ministral workload with its fingerprint check, the same pool, the same SFS smoke and audits.
"""
import argparse
import asyncio
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ministral_sfs import REFERENCE, preflight, report            # noqa: E402

from scripts.cloud.canonical_control import gates                  # noqa: E402
from scripts.cloud.common import digest, expand, read, source_hashes, write  # noqa: E402
from scripts.cloud.pool import pool                                # noqa: E402
from scripts.cloud.worker import audit_cell, parse, run_point      # noqa: E402

GRID = (6.0125, 7.8625, 8.7875)


def probe_cell(qps):
    """A cell identity that cannot collide with a campaign cell, on or off the grid."""
    return {'id': f'probe-ministral-hard-{qps:g}', 'family': 'ministral', 'variant': 'canonical',
            'policy': 'hard', 'qps': float(qps), 'requests': 8000}


async def workload(options, definition, argv, requests, output, source):
    from scripts.runs import experiments as exp
    from scripts.runs.ministral3_methodology_stage import smoke_requests, audit_run, wait_drained
    from scripts.runs.measured_audit import require_complete_ttft
    from sfs_core.shared.shared_experiment_helpers import warm_up_instances
    bundle, state = Path(options.bundle), Path(options.state)
    runner_hash, bundle_hash = digest(Path(__file__)), digest(bundle/'bundle.json')
    calibration = [exp.ExperimentRequest(**__import__('json').loads(line)) for line in
                   Path(expand(definition['calibration_requests'], bundle, {})).read_text().splitlines()]
    cell = probe_cell(options.qps)

    def status(value, **extra):
        write(output/'status.json', {'state': value, 'pid': os.getpid(), 'time': time.time(), **extra})

    with pool('ministral', definition, read(options.models), bundle, output, options.gpus.split(','),
              state, bundle/'ministral/length') as (instances, machine, processes):
        clients, costs, metadata = exp.load_instances(instances)

        async def run():
            await warm_up_instances(list(clients.values()))
            await wait_drained(clients, timeout_s=120)
            status('SMOKE')
            args = parse(argv)
            args.utilities, args.num_requests, args.request_rate_qps = ['hard'], 192, max(GRID)
            args.per_request_wait_log = [str(output/f'wait_{m}.log') for m in definition['models']]
            payload = await run_point('ministral', args, smoke_requests(calibration), clients, costs,
                                      metadata, output/'smoke', data_role='calibration')
            audit_run(payload['router']['runs'][0], 192)
            require_complete_ttft(payload['router']['runs'][0])

            status('RUNNING_8000', qps=cell['qps'])
            args = parse(argv)
            args.utilities, args.num_requests, args.request_rate_qps = ['hard'], 8000, cell['qps']
            args.per_request_wait_log = [str(output/f'wait_{m}.log') for m in definition['models']]
            folder = output/'cells'/cell['id']
            payload = await run_point('ministral', args, requests, clients, costs, metadata, folder)
            audit = audit_cell(payload, cell)
            result = report(bundle, payload, folder)
            if source_hashes() != source or digest(Path(__file__)) != runner_hash:
                raise ValueError('Runtime source changed')
            write(folder/'audit.json', {**audit, 'cell': cell, 'point': str(folder/'point.json'),
                  'point_sha256': digest(folder/'point.json'), 'bundle_sha256': bundle_hash,
                  'runner_sha256': runner_hash, 'source_sha256': source, 'hardware': machine,
                  'request_identity_slo_sha256': REFERENCE, 'result': result, 'data_role': 'rate_probe',
                  'scope': 'Off-grid SFS rate probe; diagnostic only, never a reportable campaign cell'})
            write(output/'result_summary.json', {'status': 'PASS', 'data_role': 'rate_probe', **result})
            status('COMPLETE', qps=cell['qps'])

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
    for name in ('bundle', 'state', 'models', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--gpus', required=True)
    p.add_argument('--qps', type=float, required=True,
                   help='Arrival rate for the probe; must be off the authorized grid, which only the '
                        'campaign overlay may measure')
    options = p.parse_args()
    if options.qps in GRID:
        p.error('An on-grid rate belongs to the campaign overlay, not to a probe')
    if not 0 < options.qps <= 12:
        p.error('Implausible arrival rate')
    bundle, state = Path(options.bundle).resolve(), Path(options.state).resolve()
    output = Path(options.output).resolve()
    options.bundle, options.state = str(bundle), str(state)
    _, definition, argv, requests = preflight(bundle)
    source = gates(state.parent/'setup', bundle)
    output.mkdir(parents=True, exist_ok=False)
    asyncio.run(workload(options, definition, argv, requests, output, source))


if __name__ == '__main__':
    main()

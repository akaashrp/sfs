"""Record the Bridges Ministral hard_prefill_tps side jobs under
bridges_side_jobs.hard_prefill_tps.ministral in the cloud experiment ledger.

Read-modify-write with an exclusive lock on the ledger; only that nested key is
replaced (it previously held the NOT_SUBMITTED reason), everything else is
preserved byte-for-byte in content. The ledger stays uncommitted in the cloud
worktree.
"""
import datetime
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[2]
sys.path.insert(0, str(ROOT/'src'))
from scripts.runs.ministral3_hard_prefill_tps import cells, QPS, REQUESTS, POLICY, ESTIMATOR, ARRIVAL_TIMING  # noqa: E402

LEDGER = Path('/ocean/projects/cis250162p/aparthas/sfs_cloud_20260914/scripts/cloud/experiment-ledger-20260916.json')
KEY, ENTRY, SUB = 'bridges_side_jobs', 'hard_prefill_tps', 'ministral'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def main():
    submission = json.loads((EXP/'submission.json').read_text())
    manifest = json.loads((EXP/'side/manifest.json').read_text())
    gate = json.loads((EXP/'tests/regression/gate.json').read_text())
    preflight = json.loads((EXP/'preflight/sfs-ptps-ministral-smoke_cpu.json').read_text())
    jobs = submission['jobs']
    sweep_job = jobs['sfs-ptps-ministral-sweep']['job_id']
    entry = {
        'status': 'SUBMITTED_PENDING_LOW_PRIORITY',
        'recorded_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'purpose': 'Ministral prefill-throughput-estimator baseline: hard_prefill_tps at 0.65/0.85/0.95 of K_M=9.25 '
                   '(6.0125/7.8625/8.7875 QPS), 8,000 requests per cell, on the frozen Figure 5 workload and measured calibration',
        'provider': 'bridges', 'family': 'ministral', 'policy': POLICY, 'wait_estimator': ESTIMATOR, 'arrival_timing': ARRIVAL_TIMING,
        'worktree': str(ROOT), 'branch': git('branch', '--show-current'), 'sfs_commit': git('rev-parse', 'HEAD'),
        'vllm_commit': subprocess.check_output(['git', '-C', str(ROOT/'vllm'), 'rev-parse', 'HEAD'], text=True).strip(),
        'runner': {'path': str(ROOT/'src/scripts/runs/ministral3_hard_prefill_tps.py'),
                   'sha256': sha(ROOT/'src/scripts/runs/ministral3_hard_prefill_tps.py'),
                   'test': str(ROOT/'src/scripts/runs/tests/test_ministral3_hard_prefill_tps.py'),
                   'test_sha256': sha(ROOT/'src/scripts/runs/tests/test_ministral3_hard_prefill_tps.py')},
        'estimator_path': 'hard_prefill_tps is not a methodology-scheduler policy (only lmdeploy_proxy/mooncake_prefill/routebalance are); '
                          'like canonical hard it runs on the generic wait-time scheduler with prefill_tps_ttft reading '
                          'score_proxy.prefill_tps per model from the frozen Ministral service metrics. Points run through '
                          'scripts.cloud.worker.run_point(family=ministral) -> ministral3_reliable.run_router_experiment; '
                          'the MinistralMethodologyScheduler substitution is inert for this policy. Arrivals: '
                          'absolute_schedule_thread (commit 80e2687).',
        'requests_per_cell': REQUESTS, 'qps_values': list(QPS),
        'request_identity_slo_sha256': manifest['request_identity_slo_sha256'],
        'reference': manifest['reference'],
        'side_manifest': {'path': str(EXP/'side/manifest.json'), 'sha256': sha(EXP/'side/manifest.json'),
                          'legacy_manifest': manifest['legacy_manifest'], 'legacy_manifest_sha256': manifest['legacy_manifest_sha256'],
                          'legacy_32_cells_unchanged': True, 'legacy_policies_unchanged': manifest['legacy_policies']},
        'calibration_reuse': manifest['calibration_reuse'],
        'serving': 'Frozen ministral3_router_common.sh pool: ministral3-3b/8b/14b, TP 1/1/1 on three H100-80, audited profile, '
                   'served aliases ministral3-{3,8,14}b-instruct; SFS_ROOT=this worktree so the vLLM commit check binds to 28bbf92',
        'gates': {'cpu_regression': {'path': str(EXP/'tests/regression/gate.json'), 'sha256': sha(EXP/'tests/regression/gate.json'),
                                     'status': gate['status'], 'tests': gate['tests']},
                  'side_tests': {'path': str(EXP/'tests/side_tests.xml'), 'sha256': sha(EXP/'tests/side_tests.xml')},
                  'cpu_preflight': {'path': str(EXP/'preflight/sfs-ptps-ministral-smoke_cpu.json'), 'status': preflight['status']},
                  'rehearsal': {'path': str(EXP/'rehearsal.json'), 'sha256': sha(EXP/'rehearsal.json')},
                  'gpu_gates_in_order': ['ministral3_figure5 validate (frozen 32-cell manifest, service/predictor dirs)',
                                         'side manifest validate (frozen inputs, source, side module, legacy policy tuple)',
                                         'smoke: hard_prefill_tps, 192 calibration requests at 2 QPS; audit_run + complete TTFT + '
                                         'estimator path (prefill_tps_ttft, prefill TPS == service metrics, thread arrivals)',
                                         'sweep gated on that smoke audit; per-cell scripts.cloud.worker.audit_cell + '
                                         'ministral3_figure5.audit_points + estimator audit, labelled provider=bridges']},
        'jobs': [{'name': name, **{k: v for k, v in job.items() if k != 'command'}} for name, job in jobs.items()],
        'cells': [{**cell, 'provider': 'bridges', 'status': 'SUBMITTED_PENDING', 'sweep_job_id': sweep_job} for cell in cells()],
        'cell_ids': [c['id'] for c in cells()],
        'priority': 'Low. --nice=400 (after the Qwen chain at 0/100/200/300); sweep afterok smoke with --kill-on-invalid-dep=yes',
        'results_root': str(EXP), 'submission_record': str(EXP/'submission.json'),
        'previous_reason_superseded': 'The additive runner keeps ministral3_methodology_stage.POLICIES and the frozen Figure 5 '
                                      'manifest unchanged; it freezes a separate three-cell side manifest instead',
    }
    with LEDGER.open('r+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        current = json.loads(stream.read())
        container = current.get(KEY)
        if not isinstance(container, dict) or not isinstance(container.get(ENTRY), dict):
            raise ValueError('Expected the bridges_side_jobs.hard_prefill_tps ledger entry')
        previous = container[ENTRY].get(SUB)
        if isinstance(previous, dict) and previous.get('status') != 'NOT_SUBMITTED':
            raise ValueError('Ministral hard_prefill_tps side jobs already recorded')
        entry['superseded_record'] = previous
        container[ENTRY][SUB] = entry
        current['updated_epoch'] = datetime.datetime.now(datetime.timezone.utc).timestamp()
        temporary = LEDGER.with_suffix('.json.tmp-ministral-ptps')
        temporary.write_text(json.dumps(current, indent=2)+'\n')
        os.replace(temporary, LEDGER)
    after = json.loads(LEDGER.read_text())
    assert after[KEY][ENTRY][SUB]['jobs'][1]['job_id'] == sweep_job
    print(json.dumps({'ledger': str(LEDGER), 'key': f'{KEY}.{ENTRY}.{SUB}', 'jobs': {j['name']: j['job_id'] for j in entry['jobs']},
                      'cells': entry['cell_ids']}))


if __name__ == '__main__':
    main()

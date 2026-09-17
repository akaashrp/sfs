"""Record the Bridges FCFS-unchunked hard_prefill_tps side jobs in the cloud experiment ledger.

Read-modify-write under bridges_side_jobs.hard_prefill_tps.fcfs_unchunked only;
sibling keys edited by other agents are preserved byte-for-byte in content.
"""
import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
LEDGER = Path('/ocean/projects/cis250162p/aparthas/sfs_cloud_20260914/scripts/cloud/experiment-ledger-20260916.json')
KEY, ENTRY, SUB = 'bridges_side_jobs', 'hard_prefill_tps', 'fcfs_unchunked'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*argv):
    return subprocess.check_output(['git', '-C', str(ROOT), *argv], text=True).strip()


def main():
    submission = json.loads((EXP/'submission.json').read_text())
    meta = json.loads((EXP/'jobs/scripts.json').read_text())
    rehearsal = json.loads((EXP/'rehearsal.json').read_text())
    preflight = json.loads((EXP/'preflight/sfs-fcfsb-qualify_cpu.json').read_text())
    jobs = submission['jobs']
    entry = {
        'purpose': 'Prefill-throughput-estimator baseline hard_prefill_tps under the Qwen FCFS unchunked serving configuration (qwen-fcfs-unchunked-65536), '
                   'qualified on Bridges hardware first (32B TP=2 smoke, calibrate, coefficient fit, qualify, reviews, coordinator release), then 6/7/8/8.3 QPS x 16,000 requests',
        'provider': 'bridges', 'status': 'SUBMITTED_PENDING_QUALIFICATION_THEN_RELEASE', 'recorded_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'worktree': str(ROOT), 'branch': git('branch', '--show-current'), 'sfs_commit': git('rev-parse', 'HEAD'), 'base_commit': '2a8c061',
        'vllm_commit': preflight['vllm_commit'],
        'vllm_provisioning': 'git worktree of the sfs_model_family vllm submodule repo at 28bbf92 plus symlinks to its compiled extensions built at b70f4dbb4 '
                             '(no csrc/CMake changes between b70f4dbb4 and 28bbf92; same provisioning as sfs_ptps_20260917)',
        'models_provisioning': 'models.json from scripts.cloud.prepare models (HF_HUB_OFFLINE) against a private hub cache whose snapshot directories symlink the pinned '
                               'Qwen snapshots already under /ocean/projects/cis250162p/aparthas/.cache/huggingface/hub; nothing downloaded',
        'storage_root': meta['storage'], 'storage_root_alias': '/ocean/projects/cis250162p/aparthas/sfs_artifacts/fcfs_bridges_20260917 (symlink)',
        'storage_root_deviation': 'Short root keeps the worker pool IPC socket path at 106 bytes (serving_ipc.validate_ipc_paths limit 107)',
        'bundle': str(Path(meta['storage'])/'bundle'), 'bundle_sha256': preflight['bundle_sha256'], 'models': preflight['models'],
        'campaign': meta['campaign'], 'campaign_sha256': meta['campaign_sha256'],
        'validator_change': 'scripts.cloud.fcfs.campaign accepts an explicit policy_allowance (policies outside the nine plus authorization text) on a side overlay; '
                            'the nine-policy grid check is unchanged for the main overlay. scripts.cloud.worker gained --hardware-match model (run mode only) because the '
                            'qualification and run jobs are separate Slurm allocations; cell entries record both hardware fingerprints.',
        'policy': 'hard_prefill_tps', 'family': 'qwen', 'configuration_id': 'qwen-fcfs-unchunked-65536', 'requests_per_cell': 16000, 'qps_values': [6.0, 7.0, 8.0, 8.3],
        'cells': [{'id': c, 'policy': 'hard_prefill_tps', 'qps': q, 'requests': 16000, 'provider': 'bridges', 'status': 'SUBMITTED_PENDING',
                   'run_job_id': jobs['sfs-fcfsb-run']['job_id']} for c, q in zip(meta['cells'], [6.0, 7.0, 8.0, 8.3])],
        'jobs': [{'name': name, **{k: v for k, v in job.items() if k != 'command'}} for name, job in jobs.items()],
        'gates': {'rehearsal': {'path': str(EXP/'rehearsal.json'), 'sha256': sha(EXP/'rehearsal.json'),
                                'exit_codes': {n: r['exit_code'] for n, r in rehearsal.items()}},
                  'cpu_preflight': {'path': str(EXP/'preflight/sfs-fcfsb-qualify_cpu.json'), 'source_digest': preflight['source_digest']},
                  'gpu_gates_in_order': ['job A: CPU gates (test.sh, prepare cpu/serving, FCFS chat admission audit)', 'job A: 32B TP=2 bounded FCFS smoke on two GPUs',
                                         'job A: worker calibrate (four GPUs), coefficients fit (CPU), worker qualify with fitted coefficients, coefficients validate, review_baseline_timing',
                                         'coordinator: review, then scripts.cloud.control release writes release.json', 'job B: waits <= 90 min for release.json, then worker run on the four cells with per-cell audits'],
                  'release_rule': 'Job B exits 3 without running any cell if release.json does not appear within 90 minutes'},
        'pointer': str(EXP/'qualification-pointer.json'), 'results_root': str(EXP), 'submission_record': str(EXP/'submission.json'),
        'priority': 'After the queued Qwen hard_prefill_tps chain (46206357-46206364): --nice=400, afterok dependency A->B, --kill-on-invalid-dep=yes',
    }
    current = json.loads(LEDGER.read_text())
    before = json.loads(json.dumps(current))
    container = current.setdefault(KEY, {})
    if not isinstance(container, dict) or not isinstance(container.get(ENTRY), dict):
        raise ValueError('Expected bridges_side_jobs.hard_prefill_tps to be an object')
    if SUB in container[ENTRY]:
        raise ValueError('fcfs_unchunked ledger entry already exists')
    container[ENTRY][SUB] = entry
    current['updated_epoch'] = datetime.datetime.now(datetime.timezone.utc).timestamp()
    temporary = LEDGER.with_suffix('.json.tmp-fcfsb')
    temporary.write_text(json.dumps(current, indent=2)+'\n')
    os.replace(temporary, LEDGER)
    after = json.loads(LEDGER.read_text())
    strip = lambda d: {k: v for k, v in d.items() if k not in (KEY, 'updated_epoch')}
    assert strip(after) == strip(before)
    assert {k: v for k, v in after[KEY][ENTRY].items() if k != SUB} == before[KEY][ENTRY]
    assert {k: v for k, v in after[KEY].items() if k != ENTRY} == {k: v for k, v in before[KEY].items() if k != ENTRY}
    print(json.dumps({'ledger': str(LEDGER), 'key': f'{KEY}.{ENTRY}.{SUB}', 'jobs': {j['name']: j['job_id'] for j in entry['jobs']}, 'cells': meta['cells']}))


if __name__ == '__main__':
    main()

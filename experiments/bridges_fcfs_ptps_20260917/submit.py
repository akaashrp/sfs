"""Submit job A (qualify) and job B (run, afterok A) with the campaign's Slurm conventions.

Account/QOS/partition follow the queued Qwen hard_prefill_tps chain
(cis260115p / gpu / GPU-shared, 4 x h100-80, --export=NONE); --nice places both
jobs after that chain's largest nice value (300). Each Slurm-stored script is
byte-verified against the frozen launcher.
"""
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

EXP = Path(__file__).resolve().parent
ACCOUNT, QOS, NICE = 'cis260115p', 'gpu', 400


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sbatch(script, extra):
    command = ['sbatch', '--parsable', f'--account={ACCOUNT}', f'--qos={QOS}', '--export=NONE', *extra, str(script)]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    job = result.stdout.strip().split(';')[0]
    stored = subprocess.run(['scontrol', 'write', 'batch_script', job, '-'], capture_output=True, text=True, check=True).stdout
    return job, command, hashlib.sha256(stored.encode()).hexdigest() == sha(script)


def main():
    meta = json.loads((EXP/'jobs/scripts.json').read_text()); scripts = meta['scripts']
    for name, entry in scripts.items():
        if sha(entry['path']) != entry['sha256']:
            raise ValueError(f'Launcher changed after rendering: {name}')
    if (EXP/'submission.json').exists():
        raise ValueError('Preserve the existing submission record')
    rehearsal = json.loads((EXP/'rehearsal.json').read_text())
    if set(rehearsal) != set(scripts) or any(r['exit_code'] != 0 or r['sbatch_test_only']['exit_code'] != 0
                                            or r['script_sha256'] != scripts[n]['sha256'] for n, r in rehearsal.items()):
        raise ValueError('Every launcher must pass its CPU-only rehearsal on the exact frozen bytes')
    jobs = {}

    def submit(name, dependency=None):
        extra = [f'--nice={NICE}']
        if dependency:
            extra += [f'--dependency=afterok:{dependency}', '--kill-on-invalid-dep=yes']
        job, command, verified = sbatch(scripts[name]['path'], extra)
        jobs[name] = {'job_id': job, 'dependency': dependency, 'nice': NICE, 'account': ACCOUNT, 'qos': QOS, 'partition': 'GPU-shared', 'gpus': 4,
                      'command': command, 'script': scripts[name]['path'], 'script_sha256': scripts[name]['sha256'], 'stored_script_verified': verified,
                      'submitted_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      'cells': meta['cells'] if name.endswith('-run') else [], 'evaluation_cells': len(meta['cells']) if name.endswith('-run') else 0}
        print(name, job, 'dep=', dependency, 'nice=', NICE, 'stored_verified=', verified, flush=True)
        return job

    qualify = submit('sfs-fcfsb-qualify')
    submit('sfs-fcfsb-run', qualify)
    (EXP/'submission.json').write_text(json.dumps({'submitted_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'provider': 'bridges', 'policy': 'hard_prefill_tps', 'configuration_id': 'qwen-fcfs-unchunked-65536', 'qps_values': [6.0, 7.0, 8.0, 8.3],
        'requests_per_cell': 16000, 'campaign': meta['campaign'], 'campaign_sha256': meta['campaign_sha256'], 'jobs': jobs}, indent=2)+'\n')


if __name__ == '__main__':
    main()

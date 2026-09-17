"""Submit the two frozen Ministral hard_prefill_tps launchers: smoke, then sweep (afterok smoke).

Both carry --nice above every job of the Qwen prefill-TPS chain (0/100/200/300)
so they queue after it. Each Slurm-stored script is byte-verified.
"""
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP.parents[2]/'src'))
from scripts.runs.ministral3_hard_prefill_tps import cells, QPS, REQUESTS  # noqa: E402

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
    frozen = json.loads((EXP/'scripts.json').read_text())
    scripts = frozen['scripts']
    for name, entry in scripts.items():
        if sha(entry['path']) != entry['sha256']:
            raise ValueError(f'Launcher changed after rendering: {name}')
    if sha(EXP/'side/manifest.json') != frozen['side_manifest_sha256']:
        raise ValueError('Side manifest changed after rendering')
    if (EXP/'submission.json').exists():
        raise ValueError('Preserve the existing submission record')
    rehearsal = json.loads((EXP/'rehearsal.json').read_text())
    if any(r['exit_code'] != 0 or r['script_sha256'] != scripts[n]['sha256'] for n, r in rehearsal.items()) or set(rehearsal) != set(scripts):
        raise ValueError('Every launcher must pass its CPU-only rehearsal on the exact frozen bytes')
    gate = json.loads((EXP/'tests/regression/gate.json').read_text())
    if gate.get('status') != 'PASS_CPU_REGRESSION':
        raise ValueError('Run the CPU regression gate first')
    jobs = {}

    def submit(name, dependency=None):
        extra = [f'--nice={NICE}']
        if dependency:
            extra += [f'--dependency=afterok:{dependency}', '--kill-on-invalid-dep=yes']
        job, command, verified = sbatch(scripts[name]['path'], extra)
        sweep = name.endswith('sweep')
        jobs[name] = {'job_id': job, 'arm': 'canonical', 'stage': name.rsplit('-', 1)[-1], 'dependency': dependency, 'nice': NICE,
                      'account': ACCOUNT, 'qos': QOS, 'partition': 'GPU-shared', 'gpus': 3, 'command': command,
                      'script': scripts[name]['path'], 'script_sha256': scripts[name]['sha256'], 'stored_script_verified': verified,
                      'submitted_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      'cells': [c['id'] for c in cells()] if sweep else [], 'evaluation_cells': len(cells()) if sweep else 0}
        print(name, job, 'dep=', dependency, 'nice=', NICE, 'stored_verified=', verified, flush=True)
        return job

    smoke = submit('sfs-ptps-ministral-smoke')
    submit('sfs-ptps-ministral-sweep', smoke)
    (EXP/'submission.json').write_text(json.dumps({'submitted_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'provider': 'bridges', 'family': 'ministral', 'policy': 'hard_prefill_tps', 'qps_values': list(QPS),
        'requests_per_cell': REQUESTS, 'jobs': jobs}, indent=2)+'\n')


if __name__ == '__main__':
    main()

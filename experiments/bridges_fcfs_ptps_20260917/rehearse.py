"""CPU-only rehearsal of both frozen launchers in an empty environment (SFS_PREFLIGHT_ONLY=1).

Each launcher runs through its frozen-source check, environment script and
CPU preflight, then exits before any GPU command. `sbatch --test-only` also
validates the resource request with the campaign's account/QOS.
"""
import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
ACCOUNT, QOS = 'cis260115p', 'gpu'


def run(name, path):
    scratch = EXP/'rehearsal_scratch'/name
    scratch.mkdir(parents=True, exist_ok=True)
    env = {k: os.environ[k] for k in ['PATH', 'HOME', 'USER', 'LOGNAME'] if k in os.environ}
    env.update(SFS_PREFLIGHT_ONLY='1', SBATCH_EXPORT='NONE', TMPDIR=str(scratch))
    with (EXP/'rehearsal_logs'/f'{name}.log').open('w') as log:
        result = subprocess.run(['bash', path], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    dry = subprocess.run(['sbatch', '--test-only', f'--account={ACCOUNT}', f'--qos={QOS}', '--export=NONE', path], capture_output=True, text=True)
    return name, {'exit_code': result.returncode, 'script_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                  'environment_keys': sorted(env), 'gpu_executed': False, 'preflight': str(EXP/'preflight'/f'{name}_cpu.json'),
                  'sbatch_test_only': {'exit_code': dry.returncode, 'stderr': dry.stderr.strip(), 'stdout': dry.stdout.strip()},
                  'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}


def main():
    scripts = json.loads((EXP/'jobs/scripts.json').read_text())['scripts']
    (EXP/'rehearsal_logs').mkdir(exist_ok=True)
    records = {}
    for name, entry in scripts.items():
        name, record = run(name, entry['path'])
        records[name] = record
        print(name, json.dumps(record), flush=True)
    (EXP/'rehearsal.json').write_text(json.dumps(records, indent=2)+'\n')
    if any(r['exit_code'] != 0 or r['sbatch_test_only']['exit_code'] != 0 for r in records.values()):
        raise SystemExit('A launcher rehearsal failed; see rehearsal_logs')


if __name__ == '__main__':
    main()

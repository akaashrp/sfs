"""CPU-only rehearsal of both frozen Ministral launchers in an empty environment (SFS_PREFLIGHT_ONLY=1)."""
import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[2]


def run(name, path):
    scratch = EXP/'rehearsal_scratch'/name
    scratch.mkdir(parents=True, exist_ok=True)
    env = {k: os.environ[k] for k in ['PATH', 'HOME', 'USER', 'LOGNAME'] if k in os.environ}
    env.update(SFS_PREFLIGHT_ONLY='1', SBATCH_EXPORT='NONE', TMPDIR=str(scratch))
    with (EXP/'rehearsal_logs'/f'{name}.log').open('w') as log:
        result = subprocess.run(['bash', path], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    return name, {'exit_code': result.returncode, 'script_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                  'environment_keys': sorted(env), 'gpu_executed': False,
                  'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}


def main():
    scripts = json.loads((EXP/'scripts.json').read_text())['scripts']
    (EXP/'rehearsal_logs').mkdir(exist_ok=True)
    records = {}
    for name, entry in scripts.items():
        name, record = run(name, entry['path'])
        records[name] = record
        print(name, json.dumps(record), flush=True)
    (EXP/'rehearsal.json').write_text(json.dumps(records, indent=2)+'\n')
    if any(r['exit_code'] != 0 for r in records.values()):
        raise SystemExit('A launcher rehearsal failed; see rehearsal_logs')


if __name__ == '__main__':
    main()

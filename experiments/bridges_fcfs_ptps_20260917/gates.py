"""Produce (or reuse when still valid) the CPU gates the cloud worker demands.

Reuse follows exactly the conditions scripts.cloud.worker.execute enforces:
the regression gate is bound to the current source hashes, the destination CPU
input/serving gates to source and bundle, and the FCFS chat admission audit to
bundle and profile. Anything missing or stale is regenerated here.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.cloud.common import ROOT, digest, read, source_hashes
from scripts.cloud.fcfs.config import SETTINGS


def run(argv, log):
    print('RUN', ' '.join(map(str, argv)), '->', log, flush=True)
    with Path(log).open('a') as stream:
        subprocess.run([str(v) for v in argv], cwd=ROOT/'src', stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--storage', type=Path, required=True)
    a = p.parse_args()
    storage = a.storage.resolve(); setup = storage/'setup'; bundle = storage/'bundle'
    source = source_hashes(); bundle_sha = digest(bundle/'bundle.json'); stamp = time.strftime('%Y%m%d_%H%M%S')

    def loaded(name):
        return read(setup/name) if (setup/name).exists() else {}

    tests = loaded('tests/gate.json')
    if not (tests.get('status') == 'PASS_CPU_REGRESSION' and tests.get('source_sha256') == source
            and (setup/'tests/results.xml').exists() and tests.get('xml_sha256') == digest(setup/'tests/results.xml')):
        if (setup/'tests').exists():
            shutil.move(setup/'tests', setup/f'tests-stale-{stamp}')
        run(['bash', ROOT/'scripts/cloud/test.sh', setup/'tests'], setup/f'test-sh-{stamp}.log')
    else:
        print('REUSE source-bound regression gate', tests.get('tests'), 'tests', flush=True)
    for name, mode, status in (('cpu-inputs.json', 'cpu', 'PASS_CPU_INPUTS'), ('cpu-serving.json', 'serving', 'PASS_CPU_SERVING')):
        gate = loaded(name)
        if gate.get('status') == status and gate.get('source_sha256') == source and gate.get('bundle_sha256') == bundle_sha:
            print('REUSE', name, flush=True); continue
        if (setup/name).exists():
            shutil.move(setup/name, setup/f'{name}.stale-{stamp}')
        run([sys.executable, '-m', 'scripts.cloud.prepare', mode, '--bundle', bundle, '--output', setup/name], setup/f'prepare-{mode}-{stamp}.log')
    inputs = loaded('fcfs-inputs.json')
    if inputs.get('status') == 'PASS_ACTUAL_CHAT_INPUTS' and inputs.get('profile') == SETTINGS and inputs.get('bundle_sha256') == bundle_sha:
        print('REUSE FCFS chat admission audit', flush=True)
    else:
        output = storage/'state'/f'inputs-{stamp}'
        run([sys.executable, '-m', 'scripts.cloud.fcfs.inputs', '--bundle', bundle, '--models', storage/'models.json', '--output', output], setup/f'fcfs-inputs-{stamp}.log')
        shutil.copy2(output/'audit.json', setup/'fcfs-inputs.json')
    for name in ('tests/gate.json', 'cpu-inputs.json', 'cpu-serving.json', 'fcfs-inputs.json'):
        print('GATE', name, read(setup/name).get('status'), flush=True)


if __name__ == '__main__':
    main()

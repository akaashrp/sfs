"""SSH-friendly detached submission, exact cell accounting, and reviewed release."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from scripts.cloud.common import ROOT, read, write, digest, locks


def submit(state, argv):
    state = Path(state).resolve()
    jobs = state/'jobs'; jobs.mkdir(parents=True, exist_ok=True)
    name = time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
    output = state/'runs'/name
    command = [sys.executable, '-m', 'scripts.cloud.worker', *argv, '--state', str(state), '--output', str(output)]
    log = jobs/(name+'.log')
    with log.open('x') as stream:
        process = subprocess.Popen(command, cwd=ROOT/'src', env=os.environ.copy(), stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    record = {'pid': process.pid, 'command': command, 'output': str(output), 'log': str(log), 'submitted': time.time()}
    write(jobs/(name+'.json'), record)
    print(record)


def status(state, bundle, campaign=None):
    state = Path(state)
    manifest = read(Path(bundle)/'bundle.json')
    if campaign:
        from scripts.cloud.campaigns import apply_any_campaign
        manifest = apply_any_campaign(manifest, read(campaign))
    cells = manifest['cells']
    completed, invalid = [], []
    for path in (state/'completed').glob('*.json'):
        entry = read(path)
        try:
            if digest(entry['point']) != entry['point_sha256']: raise ValueError()
            completed.append(entry['cell']['id'])
        except (OSError, ValueError): invalid.append(path.stem)
    jobs = []
    for path in sorted((state/'jobs').glob('*.json')):
        record = read(path); folder = Path(record['output'])
        report = read(folder/'status.json') if (folder/'status.json').exists() else {'state':'STARTING_OR_FAILED_BEFORE_START'}
        heartbeat = read(folder/'heartbeat.json') if (folder/'heartbeat.json').exists() else {}
        alive = Path(f"/proc/{record['pid']}").exists()
        cmdline = Path(f"/proc/{record['pid']}/cmdline")
        if alive and (not cmdline.exists() or 'scripts.cloud.worker' not in cmdline.read_bytes().decode(errors='replace')): alive=False
        if report['state'] == 'RUNNING' and not alive: report['state'] = 'LOST_PROCESS'
        jobs.append({'job': path.stem, **report, 'alive': alive,
            'heartbeat_age_s': time.time()-heartbeat['time'] if heartbeat else None,
            'output': str(folder), 'log': record['log']})
    ids = {c['id'] for c in cells}
    result = {'expected':len(cells), 'completed':len(set(completed) & ids), 'invalid':invalid,
              'remaining':[c['id'] for c in cells if c['id'] not in completed], 'jobs':jobs,
              'completed_outside_campaign':sorted(set(completed) - ids),
              'campaign':str(campaign) if campaign else None, 'campaign_kind':manifest.get('kind'),
              'campaign_sha256':digest(campaign) if campaign else None}
    print(__import__('json').dumps(result, indent=2))


def release(qualification, timing_review, load_review):
    q = Path(qualification).resolve(); report = read(q/'qualification.json')
    if report['status'] != 'GPU_MEASURED_REVIEW_REQUIRED': raise ValueError('Qualification did not finish')
    for name, expected in report['files'].items():
        if digest(q/name) != expected: raise ValueError(f'Changed qualification evidence: {name}')
    if len(timing_review.strip()) < 30 or len(load_review.strip()) < 30:
        raise ValueError('Record substantive timing residual and offered-load reviews')
    if (q/'release.json').exists(): raise ValueError('Preserve the existing release')
    write(q/'release.json', {'status':'RELEASED', 'qualification_sha256':digest(q/'qualification.json'),
        'timing_review':timing_review, 'load_review':load_review, 'reviewed_at':time.time()})


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['submit','status','release'])
    p.add_argument('--state'); p.add_argument('--bundle'); p.add_argument('--qualification')
    p.add_argument('--campaign', help='Overlay whose cells status accounts for (baseline, sfs_score or predictor_variants)')
    p.add_argument('--timing-review'); p.add_argument('--load-review')
    options, argv = p.parse_known_args()
    if argv[:1] == ['--']: argv=argv[1:]
    if options.mode == 'submit': submit(options.state, argv)
    elif options.mode == 'status': status(options.state, options.bundle, options.campaign)
    else: release(options.qualification, options.timing_review or '', options.load_review or '')

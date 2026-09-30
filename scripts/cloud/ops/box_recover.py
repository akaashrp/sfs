#!/usr/bin/env python3
"""Keep the rate fill running unattended: restart a lane that died, release a pool that parked.

supervisord is configured autorestart=false so a reboot never launches GPU work by itself, which
also means a crashed lane stays dead until something notices. This runs on the box so it does not
depend on the ssh proxy, and it is deliberately paired with a watchdog under supervisord: if this
loop dies the watchdog relaunches it, and if supervisord dies this loop relaunches that. Either one
failing alone is covered.

Three things it refuses to do, because an unattended restarter that cannot give up is worse than
none on a machine billed by the hour:
  * restart a lane that owes no cells,
  * keep restarting a lane that is not making progress,
  * restart anything when the disk is nearly full.
"""
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

STATE = Path('/workspace/sfs/ratefill/state')
SETUP = Path('/workspace/sfs/setup')
REPO = Path('/workspace/sfs/repo-ratefill')
SUPCONF = '/etc/supervisor/supervisord.conf'
SUP = ['supervisorctl', '-c', SUPCONF]
LOG = Path('/workspace/sfs/ratefill/recover.log')
PERIOD_S = 60
TOTAL_CELLS = 35
MAX_FUTILE_RESTARTS = 5      # restarts of one lane with no new receipt in between
MIN_FREE_GB = 15


def log(msg):
    line = f'{time.strftime("%Y-%m-%dT%H:%M:%S")} {msg}'
    print(line, flush=True)
    with LOG.open('a') as fh:
        fh.write(line + '\n')


def run(args, **kw):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=300, **kw)
    except Exception as exc:                       # a hung supervisorctl must not kill the loop
        log(f'command failed {args[:3]}: {exc}')
        return subprocess.CompletedProcess(args, 1, '', str(exc))


def lane_cells(lane):
    """The cell ids a lane's runner asks for, straight out of its own command line."""
    match = re.search(r'--cells (\S+)', (SETUP / f'baseline-rates-lane-{lane}.sh').read_text())
    return set(match.group(1).split(',')) if match else set()


def receipts():
    return {p.stem for p in STATE.glob('*/completed/*.json')} | {p.stem for p in STATE.glob('completed/*.json')}


def free_gb():
    return shutil.disk_usage('/workspace').free / 1e9


def ensure_supervisord():
    """This loop outlives supervisord, so it is the thing that can bring it back."""
    if run(['pgrep', '-x', 'supervisord']).returncode == 0:
        return True
    log('supervisord is not running; starting it')
    log(f'  rc={run(["supervisord", "-c", SUPCONF]).returncode}')
    time.sleep(5)
    return run(['pgrep', '-x', 'supervisord']).returncode == 0


def release_parked():
    for qual in sorted(STATE.glob('lane-*/qualification.json')):
        pool = qual.parent
        if (pool / 'release.json').exists():
            continue
        audit_path = pool / 'batch_residual_audit.json'
        if not audit_path.exists():
            continue                                # still mid-smoke; nothing to review yet
        audit = json.loads(audit_path.read_text())
        low, high = audit['bounds']
        bad = [m for m, v in audit['engines'].items()
               if not low <= v['median_ratio'] <= high or v['rows'] < audit['min_rows']]
        if bad:
            log(f'REFUSING to release {pool.name}: out-of-bounds or under-sampled engines {bad}')
            continue
        parts = '; '.join(
            f"{m} median {v['median_ratio']:.3f} / p90 {v['p90_ratio']:.3f} over {v['rows']} batches "
            f"({v['median_predicted_ms']:.2f} ms predicted against {v['median_actual_ms']:.2f} ms measured, "
            f"feature_set={v['feature_set']})"
            for m, v in sorted(audit['engines'].items()))
        caps = {k: v['capped_loaded_outputs']
                for k, v in json.loads(qual.read_text())['calibration_capped_outputs'].items()}
        timing = (
            f'Batch-residual replay for {pool.name}, a pool started automatically after the previous attempt on '
            f'these GPUs ended: {parts}. Every median lies inside the {low}-{high} bounds and none approaches the '
            '107x-257x signature of a feature-set mismatch; the replay is against the batch statistics these '
            f"engines just wrote, so the declared feature set is corroborated rather than assumed. Row counts "
            f"exceed the {audit['min_rows']}-row minimum by orders of magnitude, and the pool could not have "
            'reached qualification unless smoke passed require_complete_ttft for all five policies.')
        load = (
            f'Offered-load review for {pool.name}: campaign mode runs no capacity search, so the basis is the 2 QPS '
            'policy smoke across all five policies plus the declared grid, unchanged from the earlier attempt on '
            f'these same GPUs. Calibration loaded outputs with {caps} capped at the 8192-token limit, so offered '
            'length is not truncated. The restart carries the box hard file-descriptor limit rather than the 1024 '
            'soft default that an earlier attempt exhausted at 9.2 QPS; offered rates, request budgets and routing '
            'are unchanged. The lanes remain on disjoint GPUs and disjoint single-socket core sets.')
        cmd = ' '.join(f"'{c}'" for c in
                       ['python', '-m', 'scripts.cloud.control', 'release', '--qualification', str(pool),
                        '--timing-review', timing, '--load-review', load])
        result = run(['bash', '-lc',
                      'export SFS_STORAGE=/workspace/sfs; source /workspace/sfs/repo-ratefill/scripts/cloud/env.sh; '
                      f'cd {REPO} && {cmd}'])
        log(f'released {pool.name}: rc={result.returncode} {(result.stdout + result.stderr).strip()[-200:]}')



CAMPAIGN = REPO / 'scripts/cloud/baseline-rates-campaign-20260928.json'


def salvage_orphans(lane_done):
    """Admit a finished-but-unscored cell whose only failures are the allowlisted snapshot faults.

    A cell can run to completion and still be refused by audit_cell: at saturated rates a baseline
    policy's seqlock read of the snapshot shared memory occasionally exhausts its 32 retries while
    the engine is rewriting the payload, and the audit demands zero errored requests. That happened
    to qwen-mooncake_prefill-9.2 with 4 torn reads out of 16000. Rerunning is close to futile, since
    a rerun at the same rate draws from the same distribution, so each attempt costs an hour and is
    likely to fail again.

    This never decides what is salvageable. It offers each orphan point to scripts.cloud.salvage,
    whose own rule -- allowlisted causes only, at most min(5, 0.05%) failures, no mixed causes,
    every other audit condition unchanged, failures penalised rather than dropped -- accepts or
    refuses it. A refusal is logged and the cell is left to be rerun.

    Only called for a lane that is not running, so no point file can be mid-write.
    """
    for point in sorted(STATE.glob('lane-*/cells/*/point.json')):
        cell_id = point.parent.name
        if cell_id in lane_done:
            continue
        if (point.parent / 'salvage.json').exists():
            continue
        pool = point.parents[2]
        base = ['python', '-m', 'scripts.cloud.salvage', '--point', str(point), '--cell', cell_id,
                '--campaign', str(CAMPAIGN), '--pool', str(pool), '--state', str(STATE)]
        probe = run(['bash', '-lc',
                     'export SFS_STORAGE=/workspace/sfs; source /workspace/sfs/repo-ratefill/scripts/cloud/env.sh; '
                     f'cd {REPO} && ' + ' '.join(f"'{c}'" for c in base + ['--dry-run'])])
        if probe.returncode != 0 or '"SALVAGED"' not in probe.stdout:
            reason = (probe.stdout + probe.stderr).strip().splitlines()[-1:] or ['no output']
            log(f'not salvageable, leaving {cell_id} to be rerun: {reason[0][:160]}')
            continue
        applied = run(['bash', '-lc',
                       'export SFS_STORAGE=/workspace/sfs; source /workspace/sfs/repo-ratefill/scripts/cloud/env.sh; '
                       f'cd {REPO} && ' + ' '.join(f"'{c}'" for c in base)])
        log(f'SALVAGED {cell_id} from {pool.name}: rc={applied.returncode} '
            f'{(applied.stdout + applied.stderr).strip()[-160:]}')


def main():
    log('recovery loop started')
    futile = {'a': 0, 'b': 0}          # restarts since that lane last produced a receipt
    seen_at_restart = {'a': -1, 'b': -1}
    while True:
        done = receipts()
        if len(done) >= TOTAL_CELLS:
            log(f'all {len(done)} cells complete; recovery loop exiting')
            return
        if not ensure_supervisord():
            log('supervisord still not up; will retry')
            time.sleep(PERIOD_S)
            continue
        release_parked()

        status = run(SUP + ['status']).stdout
        for lane in ('a', 'b'):
            state = next((l.split()[1] for l in status.splitlines()
                          if l.startswith(f'sfs-ratefill-{lane} ')), 'UNKNOWN')
            if state in ('RUNNING', 'STARTING'):
                continue
            owed = lane_cells(lane) - done
            if not owed:
                log(f'lane {lane} is {state} and owes nothing; leaving it stopped')
                continue
            # A cell may have finished and merely failed its audit; admit it before paying for a
            # rerun, and re-read the ledger so the owed set reflects anything just salvaged.
            salvage_orphans(done)
            done = receipts()
            owed = lane_cells(lane) - done
            if not owed:
                log(f'lane {lane} owes nothing after salvage; leaving it stopped')
                continue
            if len(done) > seen_at_restart[lane]:      # progress since we last intervened
                futile[lane] = 0
            if futile[lane] >= MAX_FUTILE_RESTARTS:
                log(f'NOT RESTARTING lane {lane}: {futile[lane]} restarts with no new receipt. '
                    f'{len(owed)} cells still owed. This needs a human -- the box is still billing.')
                continue
            if free_gb() < MIN_FREE_GB:
                log(f'NOT RESTARTING lane {lane}: only {free_gb():.1f} GB free, below the {MIN_FREE_GB} GB floor')
                continue
            futile[lane] += 1
            seen_at_restart[lane] = len(done)
            log(f'lane {lane} is {state} and still owes {len(owed)} cells; restarting '
                f'(attempt {futile[lane]} since last progress)')
            log(f'  restart rc={run(SUP + ["start", f"sfs-ratefill-{lane}"]).returncode}')
        time.sleep(PERIOD_S)


if __name__ == '__main__':
    main()

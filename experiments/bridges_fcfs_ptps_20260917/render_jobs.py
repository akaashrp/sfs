"""Render the two frozen Bridges launchers (job A qualify, job B run) and record their hashes.

Job A (4xH100): CPU gates (reused when still source/bundle-bound), bounded 32B
TP=2 FCFS smoke on two GPUs, worker calibrate on four GPUs, CPU coefficient
fit, worker qualify with fitted coefficients, coefficient and timing-head
reviews. It ends by writing a pointer to its qualification directory.
Job B (4xH100, afterok A): waits up to 90 minutes for the coordinator's
release.json, then runs the four authorized hard_prefill_tps cells.
"""
import hashlib
import json
from pathlib import Path

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
STORAGE = Path('/ocean/projects/cis250162p/aparthas/sfs_fb17')
CAMPAIGN = ROOT/'scripts/cloud/fcfs/campaign-ptps-bridges-20260917.json'
CONFIG_ID = 'qwen-fcfs-unchunked-65536'
CELLS = [f'{CONFIG_ID}-hard_prefill_tps-{q}' for q in ('6', '7', '8', '8.3')]
FROZEN = ('scripts/cloud/fcfs/campaign-ptps-bridges-20260917.json', 'src/scripts/cloud/fcfs/campaign.py', 'src/scripts/cloud/worker.py',
          'experiments/bridges_fcfs_ptps_20260917/env.sh', 'experiments/bridges_fcfs_ptps_20260917/preflight.py', 'experiments/bridges_fcfs_ptps_20260917/gates.py')
RELEASE_WAIT_S = 90*60


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def header(name, hours):
    return f"""#!/usr/bin/env bash
#SBATCH --job-name={name}
#SBATCH --output={EXP}/logs/%x_%j.out
#SBATCH --error={EXP}/logs/%x_%j.err
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100-80:4
#SBATCH --time={hours}

set -euo pipefail
# Frozen side source: the job refuses to run if the overlay, validator, worker or launcher helpers changed after submission.
sha256sum --status -c <<'FROZEN'
{chr(10).join(f'{sha(ROOT/p)}  {ROOT/p}' for p in FROZEN)}
FROZEN
export SFS_ROOT={ROOT}
export SFS_STORAGE={STORAGE}
export EXP={EXP}
export CAMPAIGN={CAMPAIGN}
export CELLS={','.join(CELLS)}
export SLURM_EXPORT_ENV=ALL
source "$EXP/env.sh"
mkdir -p "$EXP/preflight" "$EXP/logs"
JOB="${{SLURM_JOB_ID:-cpu}}"
"""


def job_a():
    return header('sfs-fcfsb-qualify', '03:30:00') + f"""python "$EXP/preflight.py" --stage qualify --storage "$SFS_STORAGE" --campaign "$CAMPAIGN" --cells "$CELLS" \\
  --output "$EXP/preflight/sfs-fcfsb-qualify_${{JOB}}.json"
if [[ "${{SFS_PREFLIGHT_ONLY:-0}}" == 1 ]]; then exit 0; fi
# Bridges exposes the allocated GPUs with nvidia-smi indices 0..3 inside the job; the worker binds by those indices.
GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)
[[ $(tr ',' '\\n' <<<"$GPUS" | wc -l) -eq 4 ]] || {{ echo "Expected four allocated GPUs, got: $GPUS"; exit 2; }}
SMOKE_GPUS=$(cut -d, -f1,2 <<<"$GPUS")
echo "job=$JOB host=$(hostname) gpus=$GPUS smoke_gpus=$SMOKE_GPUS cuda_visible=${{CUDA_VISIBLE_DEVICES:-unset}}"
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv
# 1. CPU gates: source-bound regression tests, destination CPU input/serving gates, FCFS chat admission audit.
python "$EXP/gates.py" --storage "$SFS_STORAGE"
# 2. Bounded 32B TP=2 FCFS smoke (startup, memory, observed whole-prompt scheduling) on two GPUs.
S32B="$SFS_STORAGE/state/b${{JOB}}"
python -m scripts.cloud.fcfs.gpu_smoke --bundle "$SFS_STORAGE/bundle" --models "$SFS_STORAGE/models.json" --output "$S32B" --gpus "$SMOKE_GPUS" --indices 2
python -c "import json,sys;d=json.load(open('$S32B/gpu-smoke.json'));assert d['status']=='PASS_BOUNDED_GPU_FCFS_SMOKE' and d['full_prefill_checks']>0,d['status'];print('32B smoke',d['status'],'full_prefill_checks',d['full_prefill_checks'])"
# 3. Pass A: destination FCFS calibration traces with placeholder coefficients (four GPUs).
CAL="$SFS_STORAGE/state/calibrate-${{JOB}}"
python -m scripts.cloud.worker calibrate --profile fcfs --family qwen --bundle "$SFS_STORAGE/bundle" --models "$SFS_STORAGE/models.json" \\
  --state "$SFS_STORAGE/state" --output "$CAL" --campaign "$CAMPAIGN" --gpus "$GPUS"
# 4. CPU refit of the SFS batch coefficients for this configuration (R^2 >= 0.95, nonnegative).
COEF="$SFS_STORAGE/coefficients-${{JOB}}.json"
python -m scripts.cloud.fcfs.coefficients fit --calibration "$CAL" --output "$COEF"
# 5. Pass B: qualification with fitted coefficients: fresh calibration, timing heads, allowed-policy smoke, load probes at 6 and 8.3 QPS.
Q="$SFS_STORAGE/state/qualify-${{JOB}}"
python -m scripts.cloud.worker qualify --profile fcfs --family qwen --bundle "$SFS_STORAGE/bundle" --models "$SFS_STORAGE/models.json" \\
  --state "$SFS_STORAGE/state" --output "$Q" --campaign "$CAMPAIGN" --coefficients "$COEF" --gpus "$GPUS"
# 6. CPU reviews for the coordinator: independent coefficient residuals and timing-head residuals.
python -m scripts.cloud.fcfs.coefficients validate --coefficients "$COEF" --calibration "$Q" --output "$Q/coefficient-review.json"
python "$SFS_ROOT/ops/cloud/review_baseline_timing.py" --qualification "$Q" --output "$Q/timing-review.json"
python - <<PY
import json,hashlib,time
q=json.load(open('$Q/qualification.json'));assert q['status']=='GPU_MEASURED_REVIEW_REQUIRED',q['status']
pointer={{'qualification':'$Q','coefficients':'$COEF','calibration':'$CAL','smoke_32b':'$S32B','job_id':'$JOB','host':'$(hostname)',
  'qualification_sha256':hashlib.sha256(open('$Q/qualification.json','rb').read()).hexdigest(),'campaign_sha256':q['campaign_sha256'],
  'coefficients_sha256':q['coefficients_sha256'],'load_probes':q['load_probes'],'written':time.time(),
  'next':'Review $Q/coefficient-review.json, $Q/timing-review.json, smoke/hard_prefill_tps/point.json and load probes, then: python -m scripts.cloud.control release --qualification $Q --timing-review ... --load-review ...'}}
open('$EXP/qualification-pointer.json','w').write(json.dumps(pointer,indent=2)+'\\n');print(json.dumps(pointer,indent=2))
PY
"""


def job_b():
    return header('sfs-fcfsb-run', '05:30:00') + f"""python "$EXP/preflight.py" --stage run --storage "$SFS_STORAGE" --campaign "$CAMPAIGN" --cells "$CELLS" \\
  --pointer "$EXP/qualification-pointer.json" --output "$EXP/preflight/sfs-fcfsb-run_${{JOB}}.json"
if [[ "${{SFS_PREFLIGHT_ONLY:-0}}" == 1 ]]; then exit 0; fi
GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)
[[ $(tr ',' '\\n' <<<"$GPUS" | wc -l) -eq 4 ]] || {{ echo "Expected four allocated GPUs, got: $GPUS"; exit 2; }}
echo "job=$JOB host=$(hostname) gpus=$GPUS cuda_visible=${{CUDA_VISIBLE_DEVICES:-unset}}"
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv
Q=$(python -c "import json;print(json.load(open('$EXP/qualification-pointer.json'))['qualification'])")
COEF=$(python -c "import json;print(json.load(open('$EXP/qualification-pointer.json'))['coefficients'])")
# Wait up to 90 minutes for the coordinator's reviewed release (scripts.cloud.control release); never run cells without it.
deadline=$((SECONDS+{RELEASE_WAIT_S}))
until [[ -f "$Q/release.json" ]]; do
  if (( SECONDS >= deadline )); then echo "No release.json under $Q after {RELEASE_WAIT_S} s; exiting without running cells"; exit 3; fi
  sleep 30
done
echo "release.json present; starting the four authorized cells"
RUN="$SFS_STORAGE/state/run-${{JOB}}"
python -m scripts.cloud.worker run --profile fcfs --family qwen --bundle "$SFS_STORAGE/bundle" --models "$SFS_STORAGE/models.json" \\
  --state "$SFS_STORAGE/state" --output "$RUN" --campaign "$CAMPAIGN" --coefficients "$COEF" --qualification "$Q" \\
  --gpus "$GPUS" --hardware-match model --cells "$CELLS"
python - <<PY
import json,glob,os
entries={{os.path.basename(p)[:-5]:json.load(open(p)) for p in sorted(glob.glob('$SFS_STORAGE/state/completed/{CONFIG_ID}-hard_prefill_tps-*.json'))}}
summary={{k:{{'status':v['status'],'requests':v['requests'],'realized_qps':v['realized_qps'],'point':v['point'],'hardware_match':v.get('hardware_match'),'host':v['hardware']['host']}} for k,v in entries.items()}}
json.dump({{'job_id':'$JOB','run':'$RUN','cells':summary}},open('$EXP/results-${{JOB}}.json','w'),indent=2);print(json.dumps(summary,indent=2))
PY
"""


def main():
    (EXP/'jobs').mkdir(exist_ok=True); (EXP/'logs').mkdir(exist_ok=True)
    scripts = {}
    for name, text in (('sfs-fcfsb-qualify', job_a()), ('sfs-fcfsb-run', job_b())):
        path = EXP/'jobs'/f'{name}.sbatch'
        path.write_text(text)
        scripts[name] = {'path': str(path), 'sha256': sha(path)}
    (EXP/'jobs/scripts.json').write_text(json.dumps({'scripts': scripts, 'cells': CELLS, 'campaign': str(CAMPAIGN), 'campaign_sha256': sha(CAMPAIGN),
        'frozen': {p: sha(ROOT/p) for p in FROZEN}, 'storage': str(STORAGE)}, indent=2)+'\n')
    print(json.dumps(scripts, indent=2))


if __name__ == '__main__':
    main()

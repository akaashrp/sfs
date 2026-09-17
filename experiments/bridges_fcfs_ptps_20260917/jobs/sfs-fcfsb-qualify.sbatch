#!/usr/bin/env bash
#SBATCH --job-name=sfs-fcfsb-qualify
#SBATCH --output=/ocean/projects/cis250162p/aparthas/sfs_fcfsb_20260917/experiments/bridges_fcfs_ptps_20260917/logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250162p/aparthas/sfs_fcfsb_20260917/experiments/bridges_fcfs_ptps_20260917/logs/%x_%j.err
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100-80:4
#SBATCH --time=03:30:00

set -euo pipefail
# Frozen side source: the job refuses to run if the overlay, validator, worker or launcher helpers changed after submission.
sha256sum --status -c <<'FROZEN'
cedac71c76d8934511e1d6122a0f1c911ce150444935b1a3ee029952db30191f  /ocean/projects/cis250162p/aparthas/sfs_fcfsb_20260917/scripts/cloud/fcfs/campaign-ptps-bridges-20260917.json
2e52485cee59660143523290e14b41ff2419e6388703c02ee40cafc0f09d3c7a  /ocean/projects/cis250162p/aparthas/sfs_fcfsb_20260917/src/scripts/cloud/fcfs/campaign.py
36dd57be6b341c78670a4dd03bab754df721e86f661401db8482169a6632e7e2  /ocean/projects/cis250162p/aparthas/sfs_fcfsb_20260917/src/scripts/cloud/worker.py
6280eb2e352772cee861e85503e8c7e63db6b2f79b1eb6c1c8707f8ec9768e08  /ocean/projects/cis250162p/aparthas/sfs_fcfsb_20260917/experiments/bridges_fcfs_ptps_20260917/env.sh
6ed50b03451ba0091279f30447ab573b87ed27a76b2918cff3d9ee5f0fb2bf35  /ocean/projects/cis250162p/aparthas/sfs_fcfsb_20260917/experiments/bridges_fcfs_ptps_20260917/preflight.py
fdf8a1eddabc41d07683833bfd4c43386d9bcb937b7015924d2c1947e6801985  /ocean/projects/cis250162p/aparthas/sfs_fcfsb_20260917/experiments/bridges_fcfs_ptps_20260917/gates.py
FROZEN
export SFS_ROOT=/ocean/projects/cis250162p/aparthas/sfs_fcfsb_20260917
export SFS_STORAGE=/ocean/projects/cis250162p/aparthas/sfs_fb17
export EXP=/ocean/projects/cis250162p/aparthas/sfs_fcfsb_20260917/experiments/bridges_fcfs_ptps_20260917
export CAMPAIGN=/ocean/projects/cis250162p/aparthas/sfs_fcfsb_20260917/scripts/cloud/fcfs/campaign-ptps-bridges-20260917.json
export CELLS=qwen-fcfs-unchunked-65536-hard_prefill_tps-6,qwen-fcfs-unchunked-65536-hard_prefill_tps-7,qwen-fcfs-unchunked-65536-hard_prefill_tps-8,qwen-fcfs-unchunked-65536-hard_prefill_tps-8.3
export SLURM_EXPORT_ENV=ALL
source "$EXP/env.sh"
mkdir -p "$EXP/preflight" "$EXP/logs"
JOB="${SLURM_JOB_ID:-cpu}"
python "$EXP/preflight.py" --stage qualify --storage "$SFS_STORAGE" --campaign "$CAMPAIGN" --cells "$CELLS" \
  --output "$EXP/preflight/sfs-fcfsb-qualify_${JOB}.json"
if [[ "${SFS_PREFLIGHT_ONLY:-0}" == 1 ]]; then exit 0; fi
# Bridges exposes the allocated GPUs with nvidia-smi indices 0..3 inside the job; the worker binds by those indices.
GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)
[[ $(tr ',' '\n' <<<"$GPUS" | wc -l) -eq 4 ]] || { echo "Expected four allocated GPUs, got: $GPUS"; exit 2; }
SMOKE_GPUS=$(cut -d, -f1,2 <<<"$GPUS")
echo "job=$JOB host=$(hostname) gpus=$GPUS smoke_gpus=$SMOKE_GPUS cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv
# 1. CPU gates: source-bound regression tests, destination CPU input/serving gates, FCFS chat admission audit.
python "$EXP/gates.py" --storage "$SFS_STORAGE"
# 2. Bounded 32B TP=2 FCFS smoke (startup, memory, observed whole-prompt scheduling) on two GPUs.
S32B="$SFS_STORAGE/state/b${JOB}"
python -m scripts.cloud.fcfs.gpu_smoke --bundle "$SFS_STORAGE/bundle" --models "$SFS_STORAGE/models.json" --output "$S32B" --gpus "$SMOKE_GPUS" --indices 2
python -c "import json,sys;d=json.load(open('$S32B/gpu-smoke.json'));assert d['status']=='PASS_BOUNDED_GPU_FCFS_SMOKE' and d['full_prefill_checks']>0,d['status'];print('32B smoke',d['status'],'full_prefill_checks',d['full_prefill_checks'])"
# 3. Pass A: destination FCFS calibration traces with placeholder coefficients (four GPUs).
CAL="$SFS_STORAGE/state/calibrate-${JOB}"
python -m scripts.cloud.worker calibrate --profile fcfs --family qwen --bundle "$SFS_STORAGE/bundle" --models "$SFS_STORAGE/models.json" \
  --state "$SFS_STORAGE/state" --output "$CAL" --campaign "$CAMPAIGN" --gpus "$GPUS"
# 4. CPU refit of the SFS batch coefficients for this configuration (R^2 >= 0.95, nonnegative).
COEF="$SFS_STORAGE/coefficients-${JOB}.json"
python -m scripts.cloud.fcfs.coefficients fit --calibration "$CAL" --output "$COEF"
# 5. Pass B: qualification with fitted coefficients: fresh calibration, timing heads, allowed-policy smoke, load probes at 6 and 8.3 QPS.
Q="$SFS_STORAGE/state/qualify-${JOB}"
python -m scripts.cloud.worker qualify --profile fcfs --family qwen --bundle "$SFS_STORAGE/bundle" --models "$SFS_STORAGE/models.json" \
  --state "$SFS_STORAGE/state" --output "$Q" --campaign "$CAMPAIGN" --coefficients "$COEF" --gpus "$GPUS"
# 6. CPU reviews for the coordinator: independent coefficient residuals and timing-head residuals.
python -m scripts.cloud.fcfs.coefficients validate --coefficients "$COEF" --calibration "$Q" --output "$Q/coefficient-review.json"
python "$SFS_ROOT/ops/cloud/review_baseline_timing.py" --qualification "$Q" --output "$Q/timing-review.json"
python - <<PY
import json,hashlib,time
q=json.load(open('$Q/qualification.json'));assert q['status']=='GPU_MEASURED_REVIEW_REQUIRED',q['status']
pointer={'qualification':'$Q','coefficients':'$COEF','calibration':'$CAL','smoke_32b':'$S32B','job_id':'$JOB','host':'$(hostname)',
  'qualification_sha256':hashlib.sha256(open('$Q/qualification.json','rb').read()).hexdigest(),'campaign_sha256':q['campaign_sha256'],
  'coefficients_sha256':q['coefficients_sha256'],'load_probes':q['load_probes'],'written':time.time(),
  'next':'Review $Q/coefficient-review.json, $Q/timing-review.json, smoke/hard_prefill_tps/point.json and load probes, then: python -m scripts.cloud.control release --qualification $Q --timing-review ... --load-review ...'}
open('$EXP/qualification-pointer.json','w').write(json.dumps(pointer,indent=2)+'\n');print(json.dumps(pointer,indent=2))
PY

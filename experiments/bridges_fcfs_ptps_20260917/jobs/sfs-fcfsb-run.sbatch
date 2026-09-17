#!/usr/bin/env bash
#SBATCH --job-name=sfs-fcfsb-run
#SBATCH --output=/ocean/projects/cis250162p/aparthas/sfs_fcfsb_20260917/experiments/bridges_fcfs_ptps_20260917/logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250162p/aparthas/sfs_fcfsb_20260917/experiments/bridges_fcfs_ptps_20260917/logs/%x_%j.err
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100-80:4
#SBATCH --time=05:30:00

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
python "$EXP/preflight.py" --stage run --storage "$SFS_STORAGE" --campaign "$CAMPAIGN" --cells "$CELLS" \
  --pointer "$EXP/qualification-pointer.json" --output "$EXP/preflight/sfs-fcfsb-run_${JOB}.json"
if [[ "${SFS_PREFLIGHT_ONLY:-0}" == 1 ]]; then exit 0; fi
GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)
[[ $(tr ',' '\n' <<<"$GPUS" | wc -l) -eq 4 ]] || { echo "Expected four allocated GPUs, got: $GPUS"; exit 2; }
echo "job=$JOB host=$(hostname) gpus=$GPUS cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv
Q=$(python -c "import json;print(json.load(open('$EXP/qualification-pointer.json'))['qualification'])")
COEF=$(python -c "import json;print(json.load(open('$EXP/qualification-pointer.json'))['coefficients'])")
# Wait up to 90 minutes for the coordinator's reviewed release (scripts.cloud.control release); never run cells without it.
deadline=$((SECONDS+5400))
until [[ -f "$Q/release.json" ]]; do
  if (( SECONDS >= deadline )); then echo "No release.json under $Q after 5400 s; exiting without running cells"; exit 3; fi
  sleep 30
done
echo "release.json present; starting the four authorized cells"
RUN="$SFS_STORAGE/state/run-${JOB}"
python -m scripts.cloud.worker run --profile fcfs --family qwen --bundle "$SFS_STORAGE/bundle" --models "$SFS_STORAGE/models.json" \
  --state "$SFS_STORAGE/state" --output "$RUN" --campaign "$CAMPAIGN" --coefficients "$COEF" --qualification "$Q" \
  --gpus "$GPUS" --hardware-match model --cells "$CELLS"
python - <<PY
import json,glob,os
entries={os.path.basename(p)[:-5]:json.load(open(p)) for p in sorted(glob.glob('$SFS_STORAGE/state/completed/qwen-fcfs-unchunked-65536-hard_prefill_tps-*.json'))}
summary={k:{'status':v['status'],'requests':v['requests'],'realized_qps':v['realized_qps'],'point':v['point'],'hardware_match':v.get('hardware_match'),'host':v['hardware']['host']} for k,v in entries.items()}
json.dump({'job_id':'$JOB','run':'$RUN','cells':summary},open('$EXP/results-${JOB}.json','w'),indent=2);print(json.dumps(summary,indent=2))
PY

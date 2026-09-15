#!/bin/bash
#SBATCH --job-name=sfs-ministral-recovery
#SBATCH --output=experiments/campaign_recovery_20260915/recovery_%j.out
#SBATCH --error=experiments/campaign_recovery_20260915/recovery_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=GPU-shared
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100-80:3

set -euo pipefail
: "${SFS_RECOVERY_CODE:?Set the isolated repaired code checkout}"
: "${SFS_RECOVERY_PLAN:?Set the frozen recovery plan}"
: "${SFS_RECOVERY_PLAN_SHA256:?Set the frozen recovery plan checksum}"
printf '%s  %s\n' "$SFS_RECOVERY_PLAN_SHA256" "$SFS_RECOVERY_PLAN" | sha256sum --status -c
export SFS_ROOT=/ocean/projects/cis250162p/aparthas/sfs_model_family
export SERVICE_RUN_DIR="$SFS_ROOT/experiments/ministral3_h100-80_service_metrics_45088929_20260905_084501"
export PREDICTOR_RUN_DIR="$SFS_ROOT/experiments/ministral3_paper/predictors/run_45089699"
source /opt/packages/anaconda3/etc/profile.d/conda.sh
conda activate vllm
export PYTHONPATH="$SFS_RECOVERY_CODE/vllm:$SFS_RECOVERY_CODE/src:$SFS_ROOT/experiments/methodology_baselines/deps"
export TMPDIR="$SFS_ROOT/experiments/campaign_recovery_20260915/scratch"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$TMPDIR"
cd "$SFS_RECOVERY_CODE/src"
python -m scripts.runs.ministral3_recovery validate --plan "$SFS_RECOVERY_PLAN"
if [[ "${SFS_PREFLIGHT_ONLY:-0}" == 1 ]]; then exit 0; fi

# Reuse the original frozen serving path and launch settings. Only the router
# entry point and result consumer are replaced; Qwen's checkout stays intact.
source "$SFS_ROOT/src/slurm/runs/ministral3_router_common.sh"
ministral3_resolve_sfs_root
ministral3_resolve_service_run
MINISTRAL_RUN_ROOT="$SFS_ROOT/experiments/campaign_recovery_20260915/run_${SLURM_JOB_ID:?}"
[[ ! -e "$MINISTRAL_RUN_ROOT" ]] || { echo 'Preserve the existing run directory' >&2; exit 1; }
mkdir -p "$MINISTRAL_RUN_ROOT"
ministral3_start_router_pool
export PYTHONPATH="$SFS_RECOVERY_CODE/vllm:$SFS_RECOVERY_CODE/src:$SFS_ROOT/experiments/methodology_baselines/deps"
cd "$SFS_RECOVERY_CODE/src"
python -m scripts.runs.ministral3_recovery run --plan "$SFS_RECOVERY_PLAN" \
  --instances "$MINISTRAL_INSTANCES_CONFIG" --output "$MINISTRAL_RUN_ROOT/recovery" \
  --wait-log "$MINISTRAL_WAIT_LOG_3B" --wait-log "$MINISTRAL_WAIT_LOG_8B" --wait-log "$MINISTRAL_WAIT_LOG_14B"

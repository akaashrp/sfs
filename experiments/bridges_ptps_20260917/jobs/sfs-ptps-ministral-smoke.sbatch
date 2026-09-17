#!/usr/bin/env bash
#SBATCH --job-name=sfs-ptps-ministral-smoke
#SBATCH --output=/ocean/projects/cis250162p/aparthas/sfs_ptps_20260917/experiments/bridges_ptps_20260917/ministral/logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis250162p/aparthas/sfs_ptps_20260917/experiments/bridges_ptps_20260917/ministral/logs/%x_%j.err
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100-80:3
#SBATCH --time=02:30:00

set -euo pipefail
# Frozen side source and CPU-prepared side manifest: the job refuses to run if either changed after submission.
printf '%s\n' 'f7f6c92452141d9f9368a40ca91ce752a3a0a31f7fe08aee3274f50e97e7f306  /ocean/projects/cis250162p/aparthas/sfs_ptps_20260917/src/scripts/runs/ministral3_hard_prefill_tps.py' | sha256sum --status -c
printf '%s\n' '3353dc34a863fb62242c7a969ab5ada912574a2344f8cada6d66019dd1d53e15  /ocean/projects/cis250162p/aparthas/sfs_ptps_20260917/experiments/bridges_ptps_20260917/ministral/side/manifest.json' | sha256sum --status -c
export SFS_ROOT=/ocean/projects/cis250162p/aparthas/sfs_ptps_20260917
export EXP=/ocean/projects/cis250162p/aparthas/sfs_ptps_20260917/experiments/bridges_ptps_20260917/ministral
export SIDE_MANIFEST=/ocean/projects/cis250162p/aparthas/sfs_ptps_20260917/experiments/bridges_ptps_20260917/ministral/side/manifest.json
export SERVICE_RUN_DIR=/ocean/projects/cis250162p/aparthas/sfs_model_family/experiments/ministral3_h100-80_service_metrics_45088929_20260905_084501
export PREDICTOR_RUN_DIR=/ocean/projects/cis250162p/aparthas/sfs_model_family/experiments/ministral3_paper/predictors/run_45089699
export USER="${USER:-$(id -un)}"
export SLURM_EXPORT_ENV=ALL
source /opt/packages/anaconda3/etc/profile.d/conda.sh
conda activate vllm
export PYTHONPATH="$SFS_ROOT/vllm:$SFS_ROOT/src:$SFS_ROOT/experiments/methodology_baselines/deps"
export TMPDIR="$EXP/scratch"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$TMPDIR" "$EXP/preflight"
cd "$SFS_ROOT/src"
python -c "import vllm.vllm_flash_attn.layers.rotary, vllm.model_executor.layers.rotary_embedding, vllm.v1.engine, vllm.v1.engine._scheduler_sim"
python -m scripts.runs.ministral3_figure5 validate --manifest /ocean/projects/cis250162p/aparthas/sfs_model_family/experiments/campaign_cold_start_fix_20260912/ministral_figure5_manifest.json \
  --service-run-dir "$SERVICE_RUN_DIR" --predictor-run-dir "$PREDICTOR_RUN_DIR"
python -m scripts.runs.ministral3_hard_prefill_tps validate --manifest "$SIDE_MANIFEST" \
  >"$EXP/preflight/sfs-ptps-ministral-smoke_${SLURM_JOB_ID:-cpu}.json"
if [[ "${SFS_PREFLIGHT_ONLY:-0}" == 1 ]]; then exit 0; fi

# Frozen Ministral serving path: three models, TP 1/1/1 on three H100s, audited profile and served aliases.
source "$SFS_ROOT/src/slurm/runs/ministral3_router_common.sh"
ministral3_resolve_sfs_root
ministral3_resolve_service_run
MINISTRAL_RUN_ROOT="$EXP/smoke_pool_${SLURM_JOB_ID:?}"
[[ ! -e "$MINISTRAL_RUN_ROOT" ]] || { echo 'Preserve the existing run directory' >&2; exit 1; }
mkdir -p "$MINISTRAL_RUN_ROOT"
ministral3_start_router_pool
export PYTHONPATH="$SFS_ROOT/vllm:$SFS_ROOT/src:$SFS_ROOT/experiments/methodology_baselines/deps"
cd "$SFS_ROOT/src"
python -m scripts.runs.ministral3_hard_prefill_tps smoke --manifest "$SIDE_MANIFEST" \
  --instances-config "$MINISTRAL_INSTANCES_CONFIG" --output "$EXP/smoke" \
  --wait-log "$MINISTRAL_WAIT_LOG_3B" --wait-log "$MINISTRAL_WAIT_LOG_8B" --wait-log "$MINISTRAL_WAIT_LOG_14B"

#!/usr/bin/env bash
#SBATCH --job-name=ministral3-probes-smoke-scout
#SBATCH --output=experiments/ministral3_methodology_stage_%j.out
#SBATCH --error=experiments/ministral3_methodology_stage_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=GPU-shared
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100-80:3

set -euo pipefail
: "${SFS_ROOT:?Set the active worktree}"
: "${METHODOLOGY_PREPARED_DIR:?Set the CPU-prepared calibration request artifact}"
: "${METHODOLOGY_SERVICE_MANIFEST:?Set the audited completed service manifest}"
: "${ROUTEBALANCE_PREDICTOR_DIR:?Set the native predictor artifact}"
: "${PREDICTOR_RUN_DIR:?Set the shared backend SFS predictor artifact}"
source "$SFS_ROOT/src/slurm/runs/ministral3_router_common.sh"
ministral3_resolve_sfs_root
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm
export PYTHONPATH="$SFS_ROOT/experiments/methodology_baselines/deps:$SFS_ROOT/src:${PYTHONPATH:-}"
export TMPDIR="$SFS_ROOT/experiments/methodology_baselines/scratch"
mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
METHODOLOGY_PREPARED_DIR="$(realpath "$METHODOLOGY_PREPARED_DIR")"
METHODOLOGY_SERVICE_MANIFEST="$(realpath "$METHODOLOGY_SERVICE_MANIFEST")"
ROUTEBALANCE_PREDICTOR_DIR="$(realpath "$ROUTEBALANCE_PREDICTOR_DIR")"
PREDICTOR_RUN_DIR="$(realpath "$PREDICTOR_RUN_DIR")"
# Running Python from the repository root exposes the outer vllm directory
# as a namespace package ahead of the installed editable finder. Check and
# execute from src, preserving the shared environment's actual editable path.
cd "$SFS_ROOT/src"
resume_args=()
if [[ -n "${RESUME_STAGE_DIR:-}" ]]; then
  resume_args+=(--resume-from "$(realpath "$RESUME_STAGE_DIR")")
fi
python -m scripts.runs.ministral3_methodology_stage --validate-only \
  --prepared-dir "$METHODOLOGY_PREPARED_DIR" \
  --service-manifest "$METHODOLOGY_SERVICE_MANIFEST" \
  --routebalance-predictor "$ROUTEBALANCE_PREDICTOR_DIR" "${resume_args[@]}"
if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
  exit 0
fi
MINISTRAL_RUN_ROOT="${RUN_ROOT_OVERRIDE:-$SFS_ROOT/experiments/ministral3_paper/methodology_stage/run_${SLURM_JOB_ID:?}}"
if [[ -e "$MINISTRAL_RUN_ROOT" ]]; then
  echo "[ERROR] Refusing to overwrite $MINISTRAL_RUN_ROOT" >&2
  exit 1
fi
mkdir -p "$MINISTRAL_RUN_ROOT"
MINISTRAL_RUN_ROOT="$(realpath "$MINISTRAL_RUN_ROOT")"
# Archive code actually used, including new untracked modules. Raw calibration
# and predictor payloads remain in place and are referenced by checked hashes.
git -C "$SFS_ROOT" rev-parse HEAD >"$MINISTRAL_RUN_ROOT/sfs_head.txt"
git -C "$SFS_ROOT/vllm" rev-parse HEAD >"$MINISTRAL_RUN_ROOT/vllm_head.txt"
git -C "$SFS_ROOT" diff HEAD --binary >"$MINISTRAL_RUN_ROOT/sfs_changes.patch"
git -C "$SFS_ROOT/vllm" diff HEAD --binary >"$MINISTRAL_RUN_ROOT/vllm_changes.patch"
tar -C "$SFS_ROOT" --exclude='__pycache__' --exclude='*.pyc' \
  -czf "$MINISTRAL_RUN_ROOT/source_snapshot.tar.gz" \
  src scripts/setup requirements-methodology-cpu.txt \
  vllm/vllm/v1/core/sched/scheduler.py vllm/vllm/v1/core/sched/state_snapshot.py \
  vllm/vllm/v1/engine/async_llm.py
ministral3_start_router_pool
python -m scripts.runs.ministral3_methodology_stage \
  --prepared-dir "$METHODOLOGY_PREPARED_DIR" \
  --service-manifest "$METHODOLOGY_SERVICE_MANIFEST" \
  --routebalance-predictor "$ROUTEBALANCE_PREDICTOR_DIR" \
  --instances-config "$MINISTRAL_INSTANCES_CONFIG" \
  --service-metrics-json "$MINISTRAL_SERVICE_METRICS" \
  --wait-log "$MINISTRAL_WAIT_LOG_3B" \
  --wait-log "$MINISTRAL_WAIT_LOG_8B" \
  --wait-log "$MINISTRAL_WAIT_LOG_14B" \
  --output-root "$MINISTRAL_RUN_ROOT" \
  --duration-s 360 --max-trials "${SCOUT_MAX_TRIALS:-16}" "${resume_args[@]}" \
  >"$MINISTRAL_RUN_ROOT/stage_driver.log" 2>&1

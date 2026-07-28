#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFS_ROOT="$SCRIPT_DIR"
while [[ "$SFS_ROOT" != "/" && ! -d "$SFS_ROOT/src/slurm" ]]; do
  SFS_ROOT="$(dirname "$SFS_ROOT")"
done
if [[ "$SFS_ROOT" == "/" ]]; then
  echo "[ERROR] Could not resolve SFS root from $SCRIPT_DIR" >&2
  exit 1
fi

SBATCH_SCRIPT="$SFS_ROOT/src/slurm/runs/router/v100_rebuttal_qps.sbatch"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-v100_score_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$SFS_ROOT/experiments/$EXPERIMENT_TAG}"
CALIBRATION_OUTPUT_PATH="${CALIBRATION_OUTPUT_PATH:-$RUN_ROOT/calibration.json}"
CALIBRATION_TIME="${CALIBRATION_TIME:-12:00:00}"
SWEEP_TIME="${SWEEP_TIME:-48:00:00}"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif (( $# > 0 )); then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

SCHEDULER_SIM_SO=(
  "$SFS_ROOT"/vllm/vllm/v1/engine/_scheduler_sim.cpython-*.so
)
if (( ${#SCHEDULER_SIM_SO[@]} != 1 )) || [[ ! -f "${SCHEDULER_SIM_SO[0]}" ]]; then
  echo "[ERROR] Missing worktree-local scheduler simulator extension." >&2
  echo "[ERROR] Build it with scripts/setup/compile_vllm_scheduler_sim.sh first." >&2
  exit 1
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "SFS_ROOT=$SFS_ROOT"
  echo "EXPERIMENT_TAG=$EXPERIMENT_TAG"
  echo "RUN_ROOT=$RUN_ROOT"
  echo "CALIBRATION_OUTPUT_PATH=$CALIBRATION_OUTPUT_PATH"
  echo "calibration: sbatch --time=$CALIBRATION_TIME $SBATCH_SCRIPT"
  echo "sweep: sbatch --time=$SWEEP_TIME --dependency=afterok:<calibration_job> $SBATCH_SCRIPT"
  exit 0
fi

mkdir -p "$RUN_ROOT"
cd "$SFS_ROOT"

CALIBRATION_JOB_ID="$(
  sbatch --parsable \
    --time="$CALIBRATION_TIME" \
    --export="ALL,RUN_MODE=calibrate,EXPERIMENT_TAG=$EXPERIMENT_TAG,RUN_ROOT=$RUN_ROOT,CALIBRATION_OUTPUT_PATH=$CALIBRATION_OUTPUT_PATH,SFS_ROOT=$SFS_ROOT" \
    "$SBATCH_SCRIPT"
)"
CALIBRATION_JOB_ID="${CALIBRATION_JOB_ID%%;*}"

SWEEP_JOB_ID="$(
  sbatch --parsable \
    --time="$SWEEP_TIME" \
    --dependency="afterok:$CALIBRATION_JOB_ID" \
    --export="ALL,RUN_MODE=sweep,EXPERIMENT_TAG=$EXPERIMENT_TAG,RUN_ROOT=$RUN_ROOT,CALIBRATION_OUTPUT_PATH=$CALIBRATION_OUTPUT_PATH,SFS_ROOT=$SFS_ROOT" \
    "$SBATCH_SCRIPT"
)"
SWEEP_JOB_ID="${SWEEP_JOB_ID%%;*}"

echo "Submitted calibration job: $CALIBRATION_JOB_ID"
echo "Submitted dependent QPS sweep: $SWEEP_JOB_ID"
echo "Run root: $RUN_ROOT"

#!/bin/bash
#SBATCH --job-name=batch-fit-heldout
#SBATCH --output=experiments/batch_fit_%j.out
#SBATCH --error=experiments/batch_fit_%j.err
#SBATCH --time=02:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -p RM-shared
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=BEGIN,FAIL,END

set -euo pipefail

SBATCH_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFS_ROOT="$SBATCH_SCRIPT_DIR"
while [[ "$SFS_ROOT" != "/" && ! -d "$SFS_ROOT/src/slurm" ]]; do
  SFS_ROOT="$(dirname "$SFS_ROOT")"
done
if [[ "$SFS_ROOT" == "/" ]]; then
  echo "[ERROR] Could not resolve SFS repo root from script location: $SBATCH_SCRIPT_DIR" >&2
  exit 1
fi
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SFS_ROOT/.." && pwd)}"
BASE=${BASE:-$SFS_ROOT/experiments/batch_stats}
BASE_TAG=${BASE_TAG:-$(basename "$BASE")}
SCRIPT="python -m scripts.runs.experiments"
OUT=$SFS_ROOT/experiments

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm
export PYTHONPATH="$SFS_ROOT/src:${PYTHONPATH:-}"

run_batch_fit() {
  local model_tag="$1"
  local train_csv="$2"
  local test_csv="$3"

  local out_dir="$OUT/batch_fit_heldout_${model_tag}_${BASE_TAG}"
  mkdir -p "$out_dir"

  $SCRIPT \
    --experiment batch_fit \
    --batch-train-csv "$train_csv" \
    --batch-test-csv "$test_csv" \
    --batch-fit-output-dir "$out_dir" \
    --output-path "$out_dir/batch_fit_heldout_${model_tag}_${BASE_TAG}.json"
}

run_batch_fit \
  "qwen3_0p6b" \
  "$BASE/batch_stats_qwen3_0p6b_train.csv" \
  "$BASE/batch_stats_qwen3_0p6b_test.csv"

run_batch_fit \
  "qwen3_8b" \
  "$BASE/batch_stats_qwen3_8b_train.csv" \
  "$BASE/batch_stats_qwen3_8b_test.csv"

run_batch_fit \
  "qwen3_32b" \
  "$BASE/batch_stats_qwen3_32b_train.csv" \
  "$BASE/batch_stats_qwen3_32b_test.csv"

echo "Batch-fit runs complete."

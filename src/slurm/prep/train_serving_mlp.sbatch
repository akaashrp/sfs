#!/usr/bin/env bash
#SBATCH --job-name=sfs-mlp-export
#SBATCH --output=experiments/mlp_export_%j.out
#SBATCH --error=experiments/mlp_export_%j.err
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000M
#SBATCH --time=02:00:00
set -euo pipefail
: "${SFS_ROOT:?Set the active SFS worktree}"
: "${PREPARED_DATA:?Set the audited Qwen data directory}"
: "${OUTPUT_ROOT:?Set a fresh output directory}"
source /opt/packages/anaconda3/etc/profile.d/conda.sh
conda activate vllm
cd "$SFS_ROOT"
export PYTHONPATH="$SFS_ROOT/vllm:$SFS_ROOT/src:${PYTHONPATH:-}"
export TMPDIR="$SFS_ROOT/experiments/mlp_serving_20260911/scratch/${SLURM_JOB_ID:-local}"
mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m scripts.prep.train_serving_mlp --prepared-dir "$PREPARED_DATA" \
  --predictor-root "$SFS_ROOT/src/assets/predictors" --output-root "$OUTPUT_ROOT"

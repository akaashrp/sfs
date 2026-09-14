#!/usr/bin/env bash
#SBATCH --job-name=qwen-baseline-figures
#SBATCH --output=experiments/qwen_baseline_figures_%j.out
#SBATCH --error=experiments/qwen_baseline_figures_%j.err
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=30000M
#SBATCH --time=04:00:00
set -euo pipefail
: "${SFS_ROOT:?Set the active worktree}"
: "${QWEN_MANIFEST:?Set the canonical Qwen manifest}"
: "${QWEN_RAW_ROOT:?Set the completed four-baseline sweep output}"
: "${OUTPUT_ROOT:?Set a new derived output directory}"
source "$SFS_ROOT/src/slurm/prep/paper_ablation_env.sh"
export MPLBACKEND=Agg
python -m scripts.runs.qwen_baselines validate --manifest "$QWEN_MANIFEST"
if [[ "${VALIDATE_ONLY:-0}" == 1 ]]; then exit 0; fi
python -m scripts.reporting.collate_qwen_baselines --manifest "$QWEN_MANIFEST" \
  --raw-root "$QWEN_RAW_ROOT" --output-root "$OUTPUT_ROOT"

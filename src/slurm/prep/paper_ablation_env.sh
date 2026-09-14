#!/usr/bin/env bash
# Source from CPU validation and launchers to exercise the same environment.
: "${SFS_ROOT:?Set the active SFS worktree}"
source /opt/packages/anaconda3/etc/profile.d/conda.sh
conda activate vllm
export PYTHONPATH="$SFS_ROOT/experiments/methodology_baselines/deps:$SFS_ROOT/src:${PYTHONPATH:-}"
export TMPDIR="$SFS_ROOT/experiments/methodology_baselines/scratch"
mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$SFS_ROOT/src"

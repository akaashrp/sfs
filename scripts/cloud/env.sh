#!/usr/bin/env bash
# Source after bootstrap, including for every SSH command.
SFS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${SFS_STORAGE:?Set SFS_STORAGE to the provisioned persistent storage directory}"
export SFS_ROOT SFS_STORAGE
source "$SFS_STORAGE/miniforge/etc/profile.d/conda.sh"
conda activate vllm
export PYTHONPATH="$SFS_ROOT/vllm:$SFS_ROOT/src"
export HF_HOME="$SFS_STORAGE/hf" HF_HUB_CACHE="$SFS_STORAGE/hf/hub"
export SFS_INPUTS="$SFS_STORAGE/inputs"
export TMPDIR="$SFS_STORAGE/scratch" TMP="$SFS_STORAGE/scratch" TEMP="$SFS_STORAGE/scratch"
export TMUX_TMPDIR="$SFS_STORAGE/scratch"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$TMPDIR"

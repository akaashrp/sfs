#!/usr/bin/env bash
# Bridges equivalent of scripts/cloud/env.sh: the cloud worker is host-agnostic
# but expects SFS_STORAGE-style directories. Source from every launcher and
# from the CPU rehearsal so both exercise the same environment.
: "${SFS_ROOT:?Set the FCFS Bridges worktree}"
: "${SFS_STORAGE:?Set the short Bridges storage root (IPC socket paths are limited to 107 bytes)}"
export SFS_ROOT SFS_STORAGE
_sfs_restore_nounset=false
case $- in *u*) _sfs_restore_nounset=true; set +u ;; esac
source /opt/packages/anaconda3/etc/profile.d/conda.sh
conda activate vllm
if $_sfs_restore_nounset; then set -u; fi
unset _sfs_restore_nounset
[[ "$CONDA_PREFIX" == /jet/home/aparthas/.conda/envs/vllm ]]
# Nested vLLM worktree (submodule commit 28bbf92) with the compiled extensions
# symlinked from sfs_model_family; src for the SFS harness.
# Bridges keeps the CPU-only baseline dependencies (faiss-cpu, xgboost-cpu) in the
# dedicated target under sfs_model_family (symlinked here), not in the conda env.
export PYTHONPATH="$SFS_ROOT/vllm:$SFS_ROOT/src:$SFS_ROOT/experiments/methodology_baselines/deps"
export HF_HOME="$SFS_STORAGE/hf" HF_HUB_CACHE="$SFS_STORAGE/hf/hub"
export SFS_INPUTS="$SFS_STORAGE/inputs"
export TMPDIR="$SFS_STORAGE/scratch" TMP="$SFS_STORAGE/scratch" TEMP="$SFS_STORAGE/scratch"
export TORCHINDUCTOR_CACHE_DIR="$SFS_STORAGE/torchinductor-cache"
export TRITON_CACHE_DIR="$SFS_STORAGE/triton-cache"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SFS_TEST_GO="$SFS_STORAGE/tools/go/bin/go"
mkdir -p "$TMPDIR"
cd "$SFS_ROOT/src"

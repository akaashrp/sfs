#!/usr/bin/env bash
# Environment for the rate-fill campaign: miniforge, the pinned env, torch, then vLLM built from
# the pinned fork commit by box_build_vllm.sh.
#
# It does NOT install the archived wheel. That wheel is 0.11.0rc2.dev266+g4dbdf4a29, twenty commits
# behind the submodule pin, and it predates both the C++ scheduler simulator and
# `Add flag-gated conditional remaining-length targets for running requests` -- the fill rule. It
# has no vllm.v1.core.sched.remaining_length at all, so a campaign run against it would either
# fail at import or, worse, run without the rule while being labelled as fill cells. September's
# own pip freeze records an editable source install, not a wheel.
set -euo pipefail
export SFS_STORAGE=/workspace/sfs
export SFS_ROOT=/workspace/sfs/repo-ratefill
mkdir -p "$SFS_STORAGE/scratch" "$SFS_STORAGE/setup"
export TMPDIR="$SFS_STORAGE/scratch"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export PIP_CACHE_DIR="$SFS_STORAGE/pip-cache"
export CONDA_ENVS_PATH="$SFS_STORAGE/miniforge/envs" CONDA_PKGS_DIRS="$SFS_STORAGE/miniforge/pkgs"

# Transfers run in parallel with this build; wait for the inputs it needs. The runtime tarball is
# still wanted for the relocated SFS_INPUTS tree it carries (symlinks into the bundle), not for its
# wheel.
for f in "$SFS_ROOT/scripts/cloud/requirements-lock.txt" "$SFS_STORAGE/setup/vast-runtime-artifacts.tar.gz"; do
  while [[ ! -f "$f" ]]; do echo "waiting for transfers: $f"; sleep 15; done
done

if [[ ! -f "$SFS_STORAGE/miniforge/etc/profile.d/conda.sh" ]]; then
  echo "=== miniforge"
  installer=Miniforge3-25.3.1-0-Linux-x86_64.sh
  base=https://github.com/conda-forge/miniforge/releases/download/25.3.1-0
  curl -fsSL --retry 3 "$base/$installer" -o "$SFS_STORAGE/setup/$installer"
  bash "$SFS_STORAGE/setup/$installer" -b -p "$SFS_STORAGE/miniforge"
fi
set +u
source "$SFS_STORAGE/miniforge/etc/profile.d/conda.sh"
if [[ ! -d "$SFS_STORAGE/miniforge/envs/vllm" ]]; then
  echo "=== conda env"
  conda create -y -p "$SFS_STORAGE/miniforge/envs/vllm" -c conda-forge python=3.12.11 pip=25.2 \
    'setuptools>=77,<80' cmake ninja ccache git rsync tmux go libnuma libgomp gcc_linux-64=13 gxx_linux-64=13
fi
conda activate vllm
[[ "$CONDA_PREFIX" == "$SFS_STORAGE/miniforge/envs/vllm" ]]
if [[ ! -x "$CONDA_PREFIX/bin/nvcc" ]]; then
  echo "=== cuda toolkit"
  conda install -y -p "$SFS_STORAGE/miniforge/envs/vllm" -c nvidia/label/cuda-12.9.1 cuda-toolkit=12.9.1
fi
set -u
export PATH="$CONDA_PREFIX/bin:$PATH"
echo "=== torch"
python -m pip install -q torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
echo "=== requirements"
python -m pip install -q -r "$SFS_ROOT/scripts/cloud/requirements-lock.txt"
echo "=== relocated inputs tree from the runtime tarball"
cd "$SFS_STORAGE"
tar -xzf setup/vast-runtime-artifacts.tar.gz

echo "=== vllm from source at the pinned commit"
# box_build_vllm.sh touches environment-ready once vllm, remaining_length and _scheduler_sim all
# import, so the rest of the chain stays gated on a working runtime rather than on this script.
bash "$SFS_STORAGE/setup/box_build_vllm.sh"
echo ENV_PREP_DONE

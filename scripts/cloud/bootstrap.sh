#!/usr/bin/env bash
set -euo pipefail
SFS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${SFS_STORAGE:?Set an existing mounted persistent storage directory, for example /workspace/sfs}"
: "${SFS_BUNDLE:?Set the transferred and extracted input bundle directory}"
case "$SFS_STORAGE" in /tmp|/tmp/*|/var/tmp|/var/tmp/*) echo 'Use persistent workspace storage' >&2; exit 1;; esac
mkdir -p "$SFS_STORAGE/scratch" "$SFS_STORAGE/setup"
export SFS_STORAGE
export TMPDIR="$SFS_STORAGE/scratch"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export PIP_CACHE_DIR="$SFS_STORAGE/pip-cache"
command -v git >/dev/null
command -v curl >/dev/null
command -v nvidia-smi >/dev/null
[[ "$(uname -m)" == x86_64 ]] || { echo 'Requires x86_64' >&2; exit 1; }
available_kb=$(df -Pk "$SFS_STORAGE" | awk 'END {print $4}')
(( available_kb >= 300 * 1024 * 1024 )) || { echo 'Need at least 300 GiB free before bootstrap; provision 1 TB for headroom' >&2; exit 1; }
git -C "$SFS_ROOT" submodule update --init --recursive
if [[ ! -f "$SFS_STORAGE/miniforge/etc/profile.d/conda.sh" ]]; then
  installer=Miniforge3-25.3.1-0-Linux-x86_64.sh
  base=https://github.com/conda-forge/miniforge/releases/download/25.3.1-0
  curl -fL --retry 3 "$base/$installer" -o "$SFS_STORAGE/setup/$installer"
  curl -fL --retry 3 "$base/$installer.sha256" -o "$SFS_STORAGE/setup/$installer.sha256"
  (cd "$SFS_STORAGE/setup" && sha256sum -c "$installer.sha256")
  bash "$SFS_STORAGE/setup/$installer" -b -p "$SFS_STORAGE/miniforge"
fi
source "$SFS_STORAGE/miniforge/etc/profile.d/conda.sh"
if [[ ! -d "$SFS_STORAGE/miniforge/envs/vllm" ]]; then
  conda create -y -n vllm -c conda-forge python=3.12.11 pip=25.2 'setuptools>=77,<80' cmake ninja ccache git rsync tmux go libnuma libgomp gcc_linux-64=13 gxx_linux-64=13
fi
conda activate vllm
# Keep provider drivers/system CUDA intact (including Thunder's CUDA 13 image).
conda install -y -n vllm -c nvidia/label/cuda-12.9.1 cuda-toolkit
export CUDA_HOME="$CONDA_PREFIX" CUDA_PATH="$CONDA_PREFIX" CUDACXX="$CONDA_PREFIX/bin/nvcc"
export PATH="$CONDA_PREFIX/bin:$PATH" LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
python -m pip install -r "$SFS_ROOT/scripts/cloud/requirements-lock.txt"
export SFS_ROOT SFS_BUNDLE
python - <<'PY'
import os, pathlib, shutil, zipfile
bundle=pathlib.Path(os.environ['SFS_BUNDLE'])
import sys
sys.path.insert(0,str(pathlib.Path(os.environ['SFS_ROOT'])/'src'))
from scripts.cloud.common import validate_bundle
validate_bundle(bundle)
with zipfile.ZipFile(bundle/'runtime/vllm.whl') as z:
    meta=z.read(next(p for p in z.namelist() if p.endswith('.dist-info/METADATA'))).decode()
    version=next(line.split(': ',1)[1] for line in meta.splitlines() if line.startswith('Version: '))
    wheel=z.read(next(p for p in z.namelist() if p.endswith('.dist-info/WHEEL'))).decode()
    tag=next(line.split(': ',1)[1] for line in wheel.splitlines() if line.startswith('Tag: '))
    path=pathlib.Path(os.environ['SFS_STORAGE'])/'setup'/f'vllm-{version}-{tag}.whl'
    if not path.exists(): shutil.copyfile(bundle/'runtime/vllm.whl',path)
    (path.parent/'wheel-path.txt').write_text(str(path))
PY
python -m pip install --no-deps "$(cat "$SFS_STORAGE/setup/wheel-path.txt")"
export VLLM_TARGET_DEVICE=cuda VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_LOCATION="$SFS_BUNDLE/runtime/vllm.whl"
export TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=8
python -m pip install --no-deps --no-build-isolation -e "$SFS_ROOT/vllm" --verbose >"$SFS_STORAGE/setup/vllm-build.log" 2>&1
source "$SFS_ROOT/scripts/cloud/env.sh"
export SFS_TEST_GO="$CONDA_PREFIX/bin/go"
bash "$SFS_ROOT/scripts/cloud/test.sh" "$SFS_STORAGE/setup/tests"
python -m scripts.cloud.prepare encoder --bundle "$SFS_BUNDLE" --cache "$HF_HUB_CACHE"
python -m scripts.cloud.prepare cpu --bundle "$SFS_BUNDLE" --output "$SFS_STORAGE/setup/cpu-inputs.json"
python -m scripts.cloud.prepare serving --bundle "$SFS_BUNDLE" --output "$SFS_STORAGE/setup/cpu-serving.json"
python -m pip check >"$SFS_STORAGE/setup/pip-check.txt"
python -m pip freeze >"$SFS_STORAGE/setup/installed.txt"
python -m scripts.cloud.diagnose --state "$SFS_STORAGE/state" --output "$SFS_STORAGE/setup/diagnostics.json"
echo "Bootstrap complete. Activate with: source $SFS_ROOT/scripts/cloud/env.sh"

#!/usr/bin/env bash
# Build vLLM from the pinned fork commit, the way the September box had it.
#
# The archived wheel is 0.11.0rc2.dev266+g4dbdf4a29 -- twenty commits behind the pin, predating
# both the C++ scheduler simulator and the flag-gated conditional remaining-length targets, i.e.
# the fill rule itself. September's own pip freeze records no wheel at all, only
#   -e git+https://github.com/akaashrp/vllm.git@28bbf9226...#egg=vllm
# so the campaign ran an editable install from source. The pin adds only Python over that commit
# (7 files, no csrc/CMake/setup.py), so one build here reproduces those binaries exactly.
#
# Hopper-only arch: sm90 kernels do not depend on which other architectures are also compiled, and
# it turns a multi-hour build into a short one on a box billed by the hour.
set -euo pipefail
export SFS_STORAGE=/workspace/sfs
PIN=30ec5b4b5418ce429e5cac2693e8e34fe0e82520
SRC=/workspace/sfs/repo-ratefill/vllm

export TMPDIR=/workspace/sfs/scratch
export CONDA_ENVS_PATH="$SFS_STORAGE/miniforge/envs" CONDA_PKGS_DIRS="$SFS_STORAGE/miniforge/pkgs"
set +u; source "$SFS_STORAGE/miniforge/etc/profile.d/conda.sh"; conda activate vllm; set -u
export PATH="$CONDA_PREFIX/bin:$PATH"
export CUDA_HOME="$CONDA_PREFIX/targets/x86_64-linux"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
# The conda toolkit keeps its headers under targets/x86_64-linux/include, but nvcc resolves through
# PATH to $CONDA_PREFIX/bin/nvcc, so the legacy FindCUDA that torch's Caffe2Config pulls in infers
# the root as $CONDA_PREFIX and then fails with "missing: CUDA_INCLUDE_DIRS". Name the real root.
export CUDAToolkit_ROOT="$CUDA_HOME"
# setup.py derives -DCMAKE_CUDA_COMPILER from CUDA_HOME, which lands on the symlinked nvcc under
# targets/; nvcc resolves its own nvvm relative to wherever it is invoked from, and there is no
# targets/x86_64-linux/nvvm, so compiler identification fails. CMAKE_ARGS is appended last, so
# naming the real nvcc here overrides it while the includes stay at the toolkit root.
export CMAKE_ARGS="-DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME -DCUDAToolkit_ROOT=$CUDA_HOME -DCUDA_INCLUDE_DIRS=$CUDA_HOME/include -DCUDA_CUDART_LIBRARY=$CONDA_PREFIX/lib/libcudart.so -DCMAKE_CUDA_COMPILER=$CONDA_PREFIX/bin/nvcc"

echo "=== checkout $PIN"
if [ ! -d "$SRC/.git" ]; then
  rm -rf "$SRC"
  git clone -q https://github.com/akaashrp/vllm.git "$SRC"
fi
git -C "$SRC" fetch -q origin
git -C "$SRC" checkout -f -q "$PIN"
echo "vllm source at $(git -C "$SRC" rev-parse --short HEAD)"
test -f "$SRC/vllm/v1/core/sched/remaining_length.py" && echo "remaining_length.py present in source"

echo "=== build deps"
python -m pip install -q -r "$SRC/requirements/build.txt"

echo "=== drop the stale wheel so the editable install is the only vllm"
python -m pip uninstall -y vllm >/dev/null 2>&1 || true

echo "=== build (Hopper only)"
export TORCH_CUDA_ARCH_LIST="9.0a"
export VLLM_TARGET_DEVICE=cuda
export MAX_JOBS=96
export NVCC_THREADS=4
export CCACHE_DIR=/workspace/sfs/ccache
export CMAKE_BUILD_TYPE=Release
cd "$SRC"
# A wheel, not `pip install -e .`. The editable path compiles fine and then dies copying
#   vllm/_moe_C.cpython-312-x86_64-linux-gnu.so: doesn't exist or not a regular file
# because CMake emits the limited-API name (_moe_C.abi3.so, WITH_SOABI) while the editable copy step
# expects the interpreter-specific one. The wheel target applies abi3 naming consistently, and it is
# the mode this repo's own archived wheel was built with (cp38-abi3).
WHEELHOUSE=/workspace/sfs/setup/wheelhouse
mkdir -p "$WHEELHOUSE"
time python -m pip wheel . --no-build-isolation --no-deps -w "$WHEELHOUSE"
BUILT=$(find "$WHEELHOUSE" -name 'vllm-*.whl' | head -1)
[[ -n "$BUILT" ]] || { echo 'no wheel produced' >&2; exit 1; }
echo "built $BUILT"
python -m pip install -q --force-reinstall --no-deps "$BUILT"

# env.sh puts $SFS_ROOT/vllm ahead of site-packages, so the source tree shadows the installed
# package. Give it the compiled extensions the wheel carries and it becomes the in-place build the
# September box ran, rather than a tree that shadows the wheel with nothing.
python "$SFS_STORAGE/setup/place_extensions.py" "$BUILT" "$SRC"

echo "=== verify (under the PYTHONPATH the campaign uses, so the shadowing path is what gets tested)"
export PYTHONPATH="$SRC:/workspace/sfs/repo-ratefill/src"
cd /workspace/sfs
python - <<'PY'
import vllm, torch
print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)
print('vllm file', vllm.__file__)
from vllm.v1.core.sched.remaining_length import MODES, CONDITIONINGS
print('remaining_length MODES', MODES)
print('remaining_length CONDITIONINGS', CONDITIONINGS)
import vllm.v1.engine._scheduler_sim as sim
print('scheduler_sim', sim.__file__)
import vllm._C, vllm._moe_C
print('cuda extensions import ok')
PY
touch "$SFS_STORAGE/setup/environment-ready"
echo BUILD_DONE

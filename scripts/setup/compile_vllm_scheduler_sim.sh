#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFS_ROOT="$SCRIPT_DIR"
while [[ "$SFS_ROOT" != "/" && ! -d "$SFS_ROOT/src/slurm" ]]; do
  SFS_ROOT="$(dirname "$SFS_ROOT")"
done
if [[ "$SFS_ROOT" == "/" ]]; then
  echo "[ERROR] Could not resolve SFS repo root from script location: $SCRIPT_DIR" >&2
  exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SFS_ROOT/.." && pwd)}"
VLLM_DIR="${VLLM_DIR:-$SFS_ROOT/vllm}"
VLLM_WHL="${VLLM_WHL:-$PROJECT_ROOT/vllm.whl}"
CONDA_ENV="${CONDA_ENV:-vllm}"
BUILD_TMP_ROOT="${BUILD_TMP_ROOT:-$SFS_ROOT/.build-tmp}"

if [[ ! -d "$VLLM_DIR" ]]; then
  echo "[ERROR] vLLM directory not found: $VLLM_DIR" >&2
  exit 1
fi

if [[ ! -f "$VLLM_WHL" ]]; then
  echo "[WARN] Precompiled wheel path does not exist: $VLLM_WHL" >&2
fi

if [[ -z "${HOSTNAME:-}" ]]; then
  host_shortname="$(hostname -s 2>/dev/null || true)"
  export HOSTNAME="${host_shortname:-unknown-host}"
fi

if ! command -v module >/dev/null 2>&1; then
  echo "[ERROR] Environment modules are unavailable. Run on a host where 'module load' is supported." >&2
  exit 1
fi

echo "[INFO] Loading toolchain modules (cuda/12.6.1, gcc/13.3.1-p20240614)..."
module load cuda/12.6.1
module load gcc/13.3.1-p20240614

if command -v nvidia-smi >/dev/null 2>&1; then
  DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 || true)"
  if [[ -n "$DRIVER_VERSION" ]]; then
    echo "[INFO] Detected NVIDIA driver version: $DRIVER_VERSION"
    DRIVER_MAJOR="${DRIVER_VERSION%%.*}"
    if [[ "$DRIVER_MAJOR" =~ ^[0-9]+$ ]] && (( DRIVER_MAJOR < 560 )); then
      echo "[WARN] Driver branch appears older than R560. CUDA 12.6.1 setups typically require an R560+ compatible driver." >&2
    fi
  else
    echo "[WARN] Could not query NVIDIA driver version via nvidia-smi." >&2
  fi
else
  echo "[WARN] nvidia-smi not found; cannot print NVIDIA driver version." >&2
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export VLLM_TARGET_DEVICE=cuda
mkdir -p "$BUILD_TMP_ROOT"
export TMPDIR="$BUILD_TMP_ROOT"

cd "$VLLM_DIR"

echo "[INFO] Installing local vLLM editable package (builds native extensions including _scheduler_sim)..."
CCACHE_NOHASHDIR="true" VLLM_CPU_DISABLE_AVX512=true \
VLLM_PRECOMPILED_WHEEL_LOCATION="$VLLM_WHL" VLLM_USE_PRECOMPILED=1 \
  pip install --no-build-isolation -e . --verbose

echo "[INFO] Running scheduler simulation timing test..."
VLLM_RUN_SCHEDULER_SIM_TIMING=1 \
  pytest tests/v1/engine/test_scheduler_simulator_native.py -k timing -s

echo "[INFO] Done. vLLM scheduler simulation extension is compiled and timing test completed."

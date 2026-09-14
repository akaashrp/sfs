#!/usr/bin/env bash
# Install additional CPU wheels locally without changing the shared vllm env.
set -euo pipefail
SFS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm
METHODOLOGY_DEPS_DIR="${METHODOLOGY_DEPS_DIR:-$SFS_ROOT/experiments/methodology_baselines/deps}"
mkdir -p "$METHODOLOGY_DEPS_DIR" "$SFS_ROOT/experiments/methodology_baselines/scratch"
export TMPDIR="$SFS_ROOT/experiments/methodology_baselines/scratch"
python -m pip install --disable-pip-version-check --no-deps --only-binary=:all: \
  --target "$METHODOLOGY_DEPS_DIR" --upgrade \
  -r "$SFS_ROOT/requirements-methodology-cpu.txt"
PYTHONPATH="$METHODOLOGY_DEPS_DIR:${PYTHONPATH:-}" python - <<'PY'
import faiss
import numpy
import scipy
import xgboost
print({"faiss": faiss.__version__, "xgboost": xgboost.__version__,
       "numpy": numpy.__version__, "scipy": scipy.__version__})
PY
printf 'CPU dependency target: %s\n' "$METHODOLOGY_DEPS_DIR"

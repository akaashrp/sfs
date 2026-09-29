#!/bin/bash
# Source-bound CPU gate for the baseline rate fill. Writes gate.json under the campaign's own
# setup root; the worker refuses to run if that gate's recorded source hash is not the running
# source, so this reruns after any checkout edit. Reserves the top cores so a claimed lane keeps
# its own CPUs.
set -euo pipefail
export SFS_STORAGE=/workspace/sfs
export SFS_BUNDLE=/workspace/sfs/bundle
REPO="${SFS_REPO:?set SFS_REPO to the checkout directory name under /workspace/sfs}"
source "/workspace/sfs/$REPO/scripts/cloud/env.sh"
export SFS_TEST_GO="${SFS_TEST_GO:-$CONDA_PREFIX/bin/go}"
cd "/workspace/sfs/$REPO"
S=/workspace/sfs/ratefill/setup
rm -rf "$S"; mkdir -p "$S"
CORES=$(nproc)
GATE_CPUS="${GATE_CPUS:-$((CORES - 12))-$((CORES - 1))}"
taskset -c "$GATE_CPUS" bash scripts/cloud/test.sh "$S/tests"
taskset -c "$GATE_CPUS" python -m pytest -q -p no:cacheprovider src/scripts/cloud/tests
taskset -c "$GATE_CPUS" python -m scripts.cloud.prepare cpu --bundle "$SFS_BUNDLE" --output "$S/cpu-inputs.json"
taskset -c "$GATE_CPUS" python -m scripts.cloud.prepare serving --bundle "$SFS_BUNDLE" --output "$S/cpu-serving.json"
touch /workspace/sfs/ratefill/gates-ready
echo RATEFILL_GATES_DONE

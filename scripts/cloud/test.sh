#!/usr/bin/env bash
set -euo pipefail
SFS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${SFS_TEST_GO:?Set the Go compiler for mandatory upstream selector differential tests}"
test_output="${1:?Set a new workspace test output directory}"
mkdir -p "$test_output"
export TMPDIR="$test_output/scratch"
mkdir -p "$TMPDIR"
cd "$SFS_ROOT"
python -m scripts.cloud.test_gate start "$test_output"
python -m pytest -q --basetemp="$test_output/pytest" --junitxml="$test_output/results.xml" \
 src/scripts/cloud/tests \
 src/sfs_core/routing/tests/test_latency_history.py \
 src/sfs_core/routing/tests/test_latency_scheduler.py \
 src/sfs_core/routing/tests/test_latency_stream.py \
 src/scripts/runs/tests/test_latency_baseline.py \
 src/scripts/runs/tests/test_latency_collation.py \
 src/scripts/runs/tests/test_qwen_stage.py \
 src/scripts/runs/tests/test_qwen_job_preparation.py \
 src/scripts/runs/tests/test_qwen_predictor_variants.py \
 src/scripts/runs/tests/test_qwen_prefill_bootstrap.py \
 src/scripts/runs/tests/test_ministral3_figure5.py \
 src/scripts/runs/tests/test_serving_ipc.py \
 src/scripts/runs/tests/test_instance_metadata.py
python -m scripts.cloud.test_gate finish "$test_output"

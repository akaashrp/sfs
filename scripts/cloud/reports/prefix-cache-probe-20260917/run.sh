#!/bin/bash
# Supervisor entry for sfs-prefix-probe: bounded GPU-7 prefix-cache probe (waits inside python for the GPU window).
set -euo pipefail
export SFS_STORAGE=/workspace/sfs
source /workspace/sfs/repo-v3/scripts/cloud/env.sh
export PYTHONPATH=/workspace/sfs/repo-v3/vllm:/workspace/sfs/repo-v3/src
export CUDA_VISIBLE_DEVICES=7
cd /workspace/sfs/repo-v3
OUT=/workspace/sfs/state/diagnostics/prefix-cache-probe-20260917
git log --oneline -1
exec taskset -c 84-95 python -u "$OUT/prefix_cache_probe.py" --gpu 7 --cpus 84-95 \
  --bundle /workspace/sfs/bundle --models /workspace/sfs/models.json --state /workspace/sfs/state \
  --output "$OUT/run" --wait-program sfs-fcfs-run-lane-b --concurrency 128 --smoke-per-bucket 128 --gpu-budget-s 840

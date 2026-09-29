#!/bin/bash
# Baseline rate fill on a 4-GPU box: one pool, all 35 cells, ~18.7 h of routing.
# Rates run in this order so that a run cut short still leaves whole columns of the figure
# complete: 8.6, 9.0, 5, 4, 3, 8.9, 9.2. No lane claim -- a 4-GPU box has a single lane.
set -uo pipefail
export SFS_STORAGE=/workspace/sfs
if [ ! -f /workspace/sfs/bootstrap-complete ]; then
  echo "bootstrap-complete marker missing; not starting GPU work" >&2
  exit 0
fi
source "$SFS_STORAGE/$SFS_REPO/scripts/cloud/env.sh"
cd "$SFS_ROOT"
C="$SFS_ROOT/scripts/cloud/baseline-rates-campaign-20260928.json"
R=/workspace/sfs/ratefill
mkdir -p "$R/state"
bash /workspace/sfs/setup/refresh_gates.sh
GPUS="${GPUS:-0,1,2,3}"
CPUS="${CPUS:-0-$(($(nproc) - 1))}"
echo "baseline rate fill single lane gpus=$GPUS cpus=$CPUS"
taskset -c "$CPUS" python -m scripts.cloud.worker campaign --campaign "$C" --bundle "$SFS_STORAGE/bundle" --models "$SFS_STORAGE/models.json" --state "$R/state" --output "$R/state/single-20260928" --family qwen --variant canonical --gpus "$GPUS" --cpus "$CPUS" --cells qwen-lmdeploy_proxy-8.6,qwen-mooncake_prefill-8.6,qwen-routebalance-8.6,qwen-score-8.6,qwen-vllm_sr_latency-8.6,qwen-lmdeploy_proxy-9,qwen-mooncake_prefill-9,qwen-routebalance-9,qwen-score-9,qwen-vllm_sr_latency-9,qwen-lmdeploy_proxy-5,qwen-mooncake_prefill-5,qwen-routebalance-5,qwen-score-5,qwen-vllm_sr_latency-5,qwen-lmdeploy_proxy-4,qwen-mooncake_prefill-4,qwen-routebalance-4,qwen-score-4,qwen-vllm_sr_latency-4,qwen-lmdeploy_proxy-3,qwen-mooncake_prefill-3,qwen-routebalance-3,qwen-score-3,qwen-vllm_sr_latency-3,qwen-lmdeploy_proxy-8.9,qwen-mooncake_prefill-8.9,qwen-routebalance-8.9,qwen-score-8.9,qwen-vllm_sr_latency-8.9,qwen-lmdeploy_proxy-9.2,qwen-mooncake_prefill-9.2,qwen-routebalance-9.2,qwen-score-9.2,qwen-vllm_sr_latency-9.2
exit $?

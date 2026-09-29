#!/bin/bash
# Baseline rate fill, lane b of two on an 8-GPU box (5/8.6/8.9/9 QPS, 20 cells, ~9.8 h of routing).
# Copy to /workspace/sfs/setup/ during deployment; run under supervisor, never with exec (an exec
# would discard the claim-release trap and leak the lane).
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
mkdir -p "$R/state" "$R/claims"
bash /workspace/sfs/setup/refresh_gates.sh
read -r LANE GPUS CPUS < <(CLAIM_ROOT="$R/claims" bash /workspace/sfs/setup/claim_lane.sh)
trap 'rmdir "$R/claims/$LANE" 2>/dev/null || true' EXIT
echo "baseline rate fill lane b claimed $LANE gpus=$GPUS cpus=$CPUS"
taskset -c "$CPUS" python -m scripts.cloud.worker campaign --campaign "$C" --bundle "$SFS_STORAGE/bundle" --models "$SFS_STORAGE/models.json" --state "$R/state" --output "$R/state/lane-b-20260928" --family qwen --variant canonical --gpus "$GPUS" --cpus "$CPUS" --cells qwen-lmdeploy_proxy-5,qwen-mooncake_prefill-5,qwen-routebalance-5,qwen-score-5,qwen-vllm_sr_latency-5,qwen-lmdeploy_proxy-8.6,qwen-mooncake_prefill-8.6,qwen-routebalance-8.6,qwen-score-8.6,qwen-vllm_sr_latency-8.6,qwen-lmdeploy_proxy-8.9,qwen-mooncake_prefill-8.9,qwen-routebalance-8.9,qwen-score-8.9,qwen-vllm_sr_latency-8.9,qwen-lmdeploy_proxy-9,qwen-mooncake_prefill-9,qwen-routebalance-9,qwen-score-9,qwen-vllm_sr_latency-9
RC=$?
rmdir "$R/claims/$LANE" 2>/dev/null || true
exit $RC

#!/bin/bash
# Baseline rate fill, lane a of two on an 8-GPU box (3/4/9.2 QPS, 15 cells, ~8.9 h of routing).
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
echo "baseline rate fill lane a claimed $LANE gpus=$GPUS cpus=$CPUS"
taskset -c "$CPUS" python -m scripts.cloud.worker campaign --campaign "$C" --bundle "$SFS_STORAGE/bundle" --models "$SFS_STORAGE/models.json" --state "$R/state" --output "$R/state/lane-a-20260928" --family qwen --variant canonical --gpus "$GPUS" --cpus "$CPUS" --cells qwen-lmdeploy_proxy-3,qwen-mooncake_prefill-3,qwen-routebalance-3,qwen-score-3,qwen-vllm_sr_latency-3,qwen-lmdeploy_proxy-4,qwen-mooncake_prefill-4,qwen-routebalance-4,qwen-score-4,qwen-vllm_sr_latency-4,qwen-lmdeploy_proxy-9.2,qwen-mooncake_prefill-9.2,qwen-routebalance-9.2,qwen-score-9.2,qwen-vllm_sr_latency-9.2
RC=$?
rmdir "$R/claims/$LANE" 2>/dev/null || true
exit $RC

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
# Saturated cells hold far more concurrent connections than the unsaturated ones: lane A died at
# 9.2 QPS with "OSError: [Errno 24] Too many open files" while writing its own heartbeat, after 3
# and 4 QPS had passed cleanly. supervisord sets no minfds, so everything inherited a 1024 soft
# limit against a 1048576 hard limit. Raise it here, where it needs no supervisord restart and so
# cannot disturb a lane that is already running.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
echo "open-file limit for this lane: $(ulimit -Sn)"

source "$SFS_STORAGE/$SFS_REPO/scripts/cloud/env.sh"
cd "$SFS_ROOT"
C="$SFS_ROOT/scripts/cloud/baseline-rates-campaign-20260928.json"
R=/workspace/sfs/ratefill
mkdir -p "$R/state" "$R/claims"
bash /workspace/sfs/setup/refresh_gates.sh
read -r LANE GPUS CPUS < <(CLAIM_ROOT="$R/claims" bash /workspace/sfs/setup/claim_lane.sh)
trap 'rmdir "$R/claims/$LANE" 2>/dev/null || true' EXIT
echo "baseline rate fill lane b claimed $LANE gpus=$GPUS cpus=$CPUS"
# taskset takes a range, the worker's --cpus does not: it int()s every comma-separated element, so
# a range reaches os.sched_setaffinity as the literal '0-47' and dies. September passed
# `--cpus "$(seq -s, 0 47)"`. Expand the claimed range to the list form for the flag and keep the
# range for taskset.
BASE_OUTPUT="$R/state/lane-b-20260928"
# A fresh output directory per attempt. The worker creates its output with exist_ok=False, so a
# restart into the same path dies immediately; and the previous directory cannot simply be deleted,
# because each completed-cell receipt records its point path and the worker re-verifies that point's
# digest before skipping the cell. Keeping old attempts intact is what makes the ledger usable.
OUTPUT="$BASE_OUTPUT"
attempt=2
while [ -e "$OUTPUT" ]; do OUTPUT="$BASE_OUTPUT-r$attempt"; attempt=$((attempt + 1)); done
echo "output directory: $OUTPUT"

CPU_LIST=$(seq -s, "${CPUS%%-*}" "${CPUS##*-}")
taskset -c "$CPUS" python -m scripts.cloud.worker campaign --campaign "$C" --bundle "$SFS_STORAGE/bundle" --models "$SFS_STORAGE/models.json" --state "$R/state" --output "$OUTPUT" --family qwen --variant canonical --gpus "$GPUS" --cpus "$CPU_LIST" --cells qwen-lmdeploy_proxy-5,qwen-mooncake_prefill-5,qwen-routebalance-5,qwen-score-5,qwen-vllm_sr_latency-5,qwen-lmdeploy_proxy-8.6,qwen-mooncake_prefill-8.6,qwen-routebalance-8.6,qwen-score-8.6,qwen-vllm_sr_latency-8.6,qwen-lmdeploy_proxy-8.9,qwen-mooncake_prefill-8.9,qwen-routebalance-8.9,qwen-score-8.9,qwen-vllm_sr_latency-8.9,qwen-lmdeploy_proxy-9,qwen-mooncake_prefill-9,qwen-routebalance-9,qwen-score-9,qwen-vllm_sr_latency-9
RC=$?
rmdir "$R/claims/$LANE" 2>/dev/null || true
exit $RC

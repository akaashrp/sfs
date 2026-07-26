#!/usr/bin/env bash
# Run one corrected-SFS seq128 QPS point without modifying the source sbatch.

set -euo pipefail

if (( $# != 2 )); then
  echo "Usage: $(basename "$0") SWEEP_SBATCH QPS" >&2
  exit 2
fi

sweep_script="$1"
qps="$2"
feature_set="${BATCH_TIME_FEATURE_SET:-legacy}"
readiness_predictor_path="${READINESS_PREDICTOR_PATH:-}"

if [[ ! -f "$sweep_script" ]]; then
  echo "[ERROR] Sweep sbatch file not found: $sweep_script" >&2
  exit 1
fi
if [[ -n "${EXPECTED_SWEEP_SHA256:-}" ]]; then
  actual_sweep_sha256="$(sha256sum "$sweep_script" | awk '{ print $1 }')"
  if [[ "$actual_sweep_sha256" != "$EXPECTED_SWEEP_SHA256" ]]; then
    echo "[ERROR] Sweep sbatch changed after submission." >&2
    exit 1
  fi
fi
if [[ ! "$qps" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "[ERROR] Invalid QPS value: $qps" >&2
  exit 1
fi
if [[ "$feature_set" != "legacy" && "$feature_set" != "cross_term" ]]; then
  echo "[ERROR] BATCH_TIME_FEATURE_SET must be legacy or cross_term." >&2
  exit 1
fi
if [[ -z "${SFS_ROOT:-}" ]]; then
  echo "[ERROR] SFS_ROOT must identify the corrected SFS worktree." >&2
  exit 1
fi
if [[
  -n "$readiness_predictor_path"
  && ! -f "$readiness_predictor_path"
]]; then
  echo "[ERROR] Readiness predictor not found: $readiness_predictor_path" >&2
  exit 1
fi

qps_assignment_count="$(
  awk '/^QPS_SWEEP=\(/ { count += 1 } END { print count + 0 }' "$sweep_script"
)"
utility_assignment_count="$(
  awk '/--qps-utilities hard shortest_queue/ { count += 1 } END { print count + 0 }' "$sweep_script"
)"
pythonpath_assignment_count="$(
  awk '/export PYTHONPATH="\$PROJECT_ROOT\/vllm:\$SFS_ROOT\/src:/ { count += 1 } END { print count + 0 }' "$sweep_script"
)"
context_coefficient_count="$(
  awk '/--simulation-sum-sq-coeff=/ { count += 1 } END { print count + 0 }' "$sweep_script"
)"
experiment_assignment_count="$(
  awk '/--experiment router/ { count += 1 } END { print count + 0 }' "$sweep_script"
)"

if [[ "$qps_assignment_count" != "1" ]]; then
  echo "[ERROR] Expected exactly one QPS_SWEEP assignment; found $qps_assignment_count." >&2
  exit 1
fi
if [[ "$utility_assignment_count" != "1" ]]; then
  echo "[ERROR] Expected exactly one hard/shortest-queue utility list; found $utility_assignment_count." >&2
  exit 1
fi
if [[ "$pythonpath_assignment_count" != "1" ]]; then
  echo "[ERROR] Expected exactly one vLLM PYTHONPATH assignment; found $pythonpath_assignment_count." >&2
  exit 1
fi
if [[ "$context_coefficient_count" != "3" ]]; then
  echo "[ERROR] Expected one context coefficient per model; found $context_coefficient_count." >&2
  exit 1
fi
if [[ "$experiment_assignment_count" != "1" ]]; then
  echo "[ERROR] Expected exactly one router experiment argument; found $experiment_assignment_count." >&2
  exit 1
fi

render_sweep() {
  awk \
    -v qps_replacement="QPS_SWEEP=($qps)" \
    -v feature_set="$feature_set" \
    -v readiness_predictor_path="$readiness_predictor_path" '
      /^QPS_SWEEP=\(/ {
        print qps_replacement
        next
      }
      /export PYTHONPATH="\$PROJECT_ROOT\/vllm:\$SFS_ROOT\/src:/ {
        sub(/\$PROJECT_ROOT\/vllm/, "$SFS_ROOT/vllm")
        print
        next
      }
      /--qps-utilities hard shortest_queue/ {
        sub(/hard shortest_queue/, "hard")
        print
        next
      }
      readiness_predictor_path != "" && /--experiment router/ {
        print
        print "  --readiness-predictor-path \"" readiness_predictor_path "\" \\"
        next
      }
      feature_set == "cross_term" && /--simulation-sum-sq-coeff=/ {
        print "  --simulation-batch-time-feature-set cross_term \\"
        sub(/--simulation-sum-sq-coeff=/,
            "--simulation-prefill-x-context-coeff=")
        sub(/SIM_SUM_SQ_COEFF_/, "SIM_PREFILL_X_CONTEXT_COEFF_")
        print
        next
      }
      { print }
    ' "$sweep_script"
}

render_sweep | bash -n
echo "Running corrected seq128 feature_set=$feature_set qps=$qps readiness_predictor=${readiness_predictor_path:-disabled}"
exec bash <(render_sweep)

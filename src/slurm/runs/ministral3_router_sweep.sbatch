#!/bin/bash
#SBATCH --job-name=ministral3-router-sweep
#SBATCH --output=experiments/ministral3_router_sweep_%j.out
#SBATCH --error=experiments/ministral3_router_sweep_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=GPU-shared
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100-80:3
#SBATCH --mail-type=BEGIN,FAIL,END

set -euo pipefail

if [[ -n "${SFS_ROOT:-}" && -f "$SFS_ROOT/src/slurm/runs/ministral3_router_common.sh" ]]; then
  COMMON="$SFS_ROOT/src/slurm/runs/ministral3_router_common.sh"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/src/slurm/runs/ministral3_router_common.sh" ]]; then
  COMMON="$SLURM_SUBMIT_DIR/src/slurm/runs/ministral3_router_common.sh"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  COMMON="$SCRIPT_DIR/ministral3_router_common.sh"
fi
if [[ ! -f "$COMMON" ]]; then
  echo "[ERROR] Could not locate ministral3_router_common.sh." >&2
  exit 1
fi
source "$COMMON"
ministral3_resolve_sfs_root

SWEEP_KIND="${SWEEP_KIND:-}"
UTILITY_GROUP="${UTILITY_GROUP:-}"
VALIDATE_ONLY="${VALIDATE_ONLY:-0}"
case "$SWEEP_KIND" in
  qps|delta|qps_arrival) ;;
  *) echo "[ERROR] SWEEP_KIND must be qps, delta, or qps_arrival." >&2; exit 1 ;;
esac
if [[ "$SWEEP_KIND" == "delta" ]]; then
  if [[ "$UTILITY_GROUP" != "combined" ]]; then
    echo "[ERROR] Delta sweeps require UTILITY_GROUP=combined." >&2
    exit 1
  fi
elif [[ "$UTILITY_GROUP" != "snapshot" && "$UTILITY_GROUP" != "baseline" ]]; then
  echo "[ERROR] QPS/arrival sweeps require UTILITY_GROUP=snapshot or baseline." >&2
  exit 1
fi
if [[ "$VALIDATE_ONLY" != "0" && "$VALIDATE_ONLY" != "1" ]]; then
  echo "[ERROR] VALIDATE_ONLY must be 0 or 1." >&2
  exit 1
fi

if [[ "$VALIDATE_ONLY" == "1" ]]; then
  echo "sweep_kind=$SWEEP_KIND utility_group=$UTILITY_GROUP"
  echo "gpu_count=3 tp_3b=1 tp_8b=1 tp_14b=1"
  echo "service_job_id=${SERVICE_JOB_ID:-<required-at-runtime>}"
  echo "predictor_run_dir=${PREDICTOR_RUN_DIR:-<required-at-runtime>}"
  echo "serving_profile=max_model_len:131072 context:65536 prompt_cap:32768 completion_cap:8192 chunked_prefill:true max_num_batched_tokens:32768 max_num_seqs:512"
  exit 0
fi

RUN_ID="${SLURM_JOB_ID:?Router sweep requires a Slurm job ID}"
MINISTRAL_RUN_ROOT="${RUN_ROOT_OVERRIDE:-$SFS_ROOT/experiments/ministral3_paper/figure_sweeps/${SWEEP_KIND}_${UTILITY_GROUP}_run_${RUN_ID}}"
if [[ -e "$MINISTRAL_RUN_ROOT" ]]; then
  echo "[ERROR] Refusing to overwrite sweep output: $MINISTRAL_RUN_ROOT" >&2
  exit 1
fi
mkdir -p "$MINISTRAL_RUN_ROOT/outputs"
ministral3_start_router_pool

if [[ "$UTILITY_GROUP" == "snapshot" ]]; then
  UTILITIES=(hard shortest_queue)
elif [[ "$UTILITY_GROUP" == "baseline" ]]; then
  UTILITIES=(latency_agnostic round_robin)
else
  UTILITIES=()
fi

COMMON_EXPERIMENT_ARGS=(
  --experiment router
  --instances-config "$MINISTRAL_INSTANCES_CONFIG"
  --worker-count 1
  --max-queue-size 1
  --max-completion-tokens 8192
  --temperature 0
  --top-p 1
  --queue-slo-min-ms 1
  --queue-slo-max-ms 5
  --ttft-slo-min-ms 150
  --ttft-slo-max-ms 1120
  --ttft-slo-base-ms 155
  --ttft-slo-per-prompt-token-ms 0.035
  --ttft-slo-jitter-min 0.98
  --ttft-slo-jitter-max 1.02
  --feasible-slo-mode ttft
  --lambda-weight 5e-4
  --system-prompt 'You are a helpful assistant.'
  --chat-template-kwargs-json '{}'
  --tokenizer-id "$MINISTRAL_TOKENIZER_PATH"
  --tokenizer-mode mistral
  --accuracy-model-path "$MINISTRAL_ACCURACY_PREDICTOR"
  --output-length-model-path "$MINISTRAL_OUTPUT_PREDICTOR"
  --service-metrics-json "$MINISTRAL_SERVICE_METRICS"
  --holdout-start-index 2500
  --holdout-context-length 65536
  --per-request-wait-log "$MINISTRAL_WAIT_LOG_3B"
  --per-request-wait-log "$MINISTRAL_WAIT_LOG_8B"
  --per-request-wait-log "$MINISTRAL_WAIT_LOG_14B"
)

SWEEP_OUTPUT_DIR="$MINISTRAL_RUN_ROOT/outputs"
SWEEP_PREFIX="ministral3_${SWEEP_KIND}_${UTILITY_GROUP}"
EXPECTED_POINTS=0
EXPECTED_REQUESTS=0

case "$SWEEP_KIND" in
  qps)
    read -r -a fractions <<< "${QPS_FRACTIONS:-0.40 0.55 0.70 0.82 0.95 1.05 1.10 1.15 1.18 1.20 1.23 1.26 1.30}"
    read -r -a qps_values <<< "$(ministral3_capacity_qps_values "${fractions[@]}")"
    EXPECTED_POINTS="${#qps_values[@]}"
    EXPECTED_REQUESTS=16000
    printf '%s\n' "${qps_values[*]}" >"$MINISTRAL_RUN_ROOT/qps_values.txt"
    python -m scripts.runs.experiments_sweep \
      --sweep qps \
      --qps-values "${qps_values[@]}" \
      --qps-utilities "${UTILITIES[@]}" \
      --output-dir "$SWEEP_OUTPUT_DIR" \
      --output-prefix "$SWEEP_PREFIX" \
      --arrival-process poisson \
      --num-requests "$EXPECTED_REQUESTS" \
      --delta-weight 0 \
      --bucket-dir "$SFS_ROOT/experiments/data/prompts/ministral3/holdout_cache_4000" \
      --holdout-prompts-per-bucket 4000 \
      --holdout-bucket-dir "$MINISTRAL_HOLDOUT_SOURCE_DIR" \
      --holdout-cache-dir "$SFS_ROOT/experiments/data/prompts/ministral3/holdout_cache_4000" \
      "${COMMON_EXPERIMENT_ARGS[@]}" \
      >"$MINISTRAL_RUN_ROOT/driver.log" 2>&1
    ;;
  delta)
    read -r -a delta_values <<< "${DELTA_VALUES:-0 1e-7 3e-7 4e-7 5e-7 6e-7 7e-7 8e-7 9e-7 1e-6 2e-6 3e-6 4e-6 5e-6 8e-6 1e-5 5e-5 1e-4 5e-4 1e-3 1e9}"
    request_rate_qps="$(ministral3_capacity_qps_values "${DELTA_QPS_FRACTION:-1.10}")"
    EXPECTED_POINTS="${#delta_values[@]}"
    EXPECTED_REQUESTS=8000
    echo "$request_rate_qps" >"$MINISTRAL_RUN_ROOT/qps_values.txt"
    python -m scripts.runs.experiments_sweep \
      --sweep delta \
      --delta-values "${delta_values[@]}" \
      --delta-zero-utilities soft latency_agnostic round_robin shortest_queue \
      --delta-nonzero-utilities soft \
      --output-dir "$SWEEP_OUTPUT_DIR" \
      --output-prefix "$SWEEP_PREFIX" \
      --request-rate-qps "$request_rate_qps" \
      --arrival-process poisson \
      --num-requests "$EXPECTED_REQUESTS" \
      --bucket-dir "$SFS_ROOT/experiments/data/prompts/ministral3/holdout_cache_2000" \
      --holdout-prompts-per-bucket 2000 \
      --holdout-bucket-dir "$MINISTRAL_HOLDOUT_SOURCE_DIR" \
      --holdout-cache-dir "$SFS_ROOT/experiments/data/prompts/ministral3/holdout_cache_2000" \
      "${COMMON_EXPERIMENT_ARGS[@]}" \
      >"$MINISTRAL_RUN_ROOT/driver.log" 2>&1
    ;;
  qps_arrival)
    read -r -a fractions <<< "${ARRIVAL_QPS_FRACTIONS:-0.82 0.96 1.10}"
    read -r -a qps_values <<< "$(ministral3_capacity_qps_values "${fractions[@]}")"
    EXPECTED_POINTS=$(( ${#qps_values[@]} * 2 ))
    EXPECTED_REQUESTS=16000
    printf '%s\n' "${qps_values[*]}" >"$MINISTRAL_RUN_ROOT/qps_values.txt"
    python -m scripts.runs.experiments_sweep \
      --sweep qps_arrival \
      --qps-values "${qps_values[@]}" \
      --arrival-sweep-ids mmpp2_r3 mmpp2_r6 \
      --arrival-sweep-processes mmpp2 mmpp2 \
      --arrival-sweep-rate-ratios 3.0 6.0 \
      --arrival-sweep-mmpp2-high-fraction 0.2 \
      --arrival-sweep-mmpp2-correlation-time-s 2.0 \
      --qps-utilities "${UTILITIES[@]}" \
      --output-dir "$SWEEP_OUTPUT_DIR" \
      --output-prefix "$SWEEP_PREFIX" \
      --num-requests "$EXPECTED_REQUESTS" \
      --delta-weight 0 \
      --bucket-dir "$SFS_ROOT/experiments/data/prompts/ministral3/holdout_cache_4000" \
      --holdout-prompts-per-bucket 4000 \
      --holdout-bucket-dir "$MINISTRAL_HOLDOUT_SOURCE_DIR" \
      --holdout-cache-dir "$SFS_ROOT/experiments/data/prompts/ministral3/holdout_cache_4000" \
      "${COMMON_EXPERIMENT_ARGS[@]}" \
      >"$MINISTRAL_RUN_ROOT/driver.log" 2>&1
    ;;
esac

python - "$SWEEP_OUTPUT_DIR" "$SWEEP_KIND" "$UTILITY_GROUP" \
  "$EXPECTED_POINTS" "$EXPECTED_REQUESTS" <<'PY'
import json
import math
from pathlib import Path
import sys

output_dir = Path(sys.argv[1])
sweep_kind = sys.argv[2]
utility_group = sys.argv[3]
expected_points = int(sys.argv[4])
expected_requests = int(sys.argv[5])
json_paths = sorted(
    path for path in output_dir.glob("*.json")
    if not path.name.endswith("_manifest.json")
)
if len(json_paths) != expected_points:
    raise RuntimeError(
        f"Expected {expected_points} sweep point JSONs; found {len(json_paths)}"
    )

expected_by_group = {
    "snapshot": {"hard", "shortest_queue"},
    "baseline": {"latency_agnostic", "round_robin"},
}
audited_runs = 0
for path in json_paths:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload["request_set"]["num_requests"]) != expected_requests:
        raise RuntimeError(f"{path}: incorrect request-set size")
    runs = payload["router"]["runs"]
    utilities = {str(run["utility"]) for run in runs}
    if sweep_kind == "delta":
        delta = float(payload["config"]["delta_weight"])
        expected_utilities = (
            {"soft", "latency_agnostic", "round_robin", "shortest_queue"}
            if delta == 0.0 else {"soft"}
        )
    else:
        expected_utilities = expected_by_group[utility_group]
    if utilities != expected_utilities:
        raise RuntimeError(
            f"{path}: utilities {sorted(utilities)} != {sorted(expected_utilities)}"
        )
    for run in runs:
        summary = run["summary"]
        records = run["per_request"]
        errors = sum(record.get("error") is not None for record in records)
        if (
            len(records) != expected_requests
            or int(summary["total_requests"]) != expected_requests
            or int(summary["succeeded_requests"]) != expected_requests
            or int(summary["failed_requests"]) != 0
            or errors != 0
        ):
            raise RuntimeError(f"{path}:{run['utility']}: request completion audit failed")
        if int(summary["system_entry_e2e_ttft_missing_count"]) != 0:
            raise RuntimeError(f"{path}:{run['utility']}: incomplete TTFT telemetry")
        attainment = float(summary["system_entry_e2e_ttft_slo_attainment_pct"])
        if not math.isfinite(attainment) or not 0.0 <= attainment <= 100.0:
            raise RuntimeError(f"{path}:{run['utility']}: invalid SLO attainment")
        audited_runs += 1

audit = {
    "status": "PASS",
    "sweep_kind": sweep_kind,
    "utility_group": utility_group,
    "sweep_points": len(json_paths),
    "audited_policy_runs": audited_runs,
    "requests_per_policy_run": expected_requests,
    "zero_request_errors": True,
    "complete_system_entry_e2e_ttft": True,
    "point_files": [str(path.resolve()) for path in json_paths],
}
(output_dir.parent / "audit.json").write_text(
    json.dumps(audit, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(audit, indent=2))
PY

find "$MINISTRAL_RUN_ROOT" -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum >"$MINISTRAL_RUN_ROOT/SHA256SUMS"
echo "[DONE] Ministral $SWEEP_KIND/$UTILITY_GROUP sweep: $MINISTRAL_RUN_ROOT"

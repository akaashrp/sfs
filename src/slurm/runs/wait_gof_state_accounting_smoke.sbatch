#!/bin/bash
#SBATCH --job-name=sfs-ttft-smoke
#SBATCH --output=/ocean/projects/cis250162p/aparthas/sfs_experiments/state_accounting_smoke/slurm_%j.out
#SBATCH --error=/ocean/projects/cis250162p/aparthas/sfs_experiments/state_accounting_smoke/slurm_%j.err
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=GPU-shared
#SBATCH --account=cis250162p
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100-80:1

set -euo pipefail

BASE_ROOT=/ocean/projects/cis250162p/aparthas
OLD_SFS_ROOT=$BASE_ROOT/sfs
NEW_SFS_ROOT=$BASE_ROOT/sfs_state_accounting
OLD_VLLM_ROOT=$BASE_ROOT/vllm
NEW_VLLM_ROOT=$NEW_SFS_ROOT/vllm

NUM_REQUESTS="${NUM_REQUESTS:-256}"
REQUEST_RATE_QPS="${REQUEST_RATE_QPS:-7}"
WORKER_COUNT="${WORKER_COUNT:-4}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-8192}"
SEED="${SEED:-69}"
COMPARISON_MODE="${COMPARISON_MODE:-accounting}"
ENABLE_QUEUE_ALIGNED_DIAGNOSTIC="${ENABLE_QUEUE_ALIGNED_DIAGNOSTIC:-1}"

JOB_ID=${SLURM_JOB_ID:-$$}
if [[ "$COMPARISON_MODE" == "cross_term" ]]; then
  DEFAULT_RESULT_BASE=$BASE_ROOT/sfs_experiments/state_accounting_cross_term
else
  DEFAULT_RESULT_BASE=$BASE_ROOT/sfs_experiments/state_accounting_smoke
fi
RESULT_ROOT="${RESULT_ROOT:-$DEFAULT_RESULT_BASE/$JOB_ID}"
SCRATCH=/local/$USER/$JOB_ID
OLD_RUNTIME_ROOT=$SCRATCH/old_vllm_runtime
NEW_RUNTIME_ROOT=$SCRATCH/new_vllm_runtime

MODEL_SNAPSHOT=c1899de289a04d12100db370d81485cdf75e47ca
MODEL_SOURCE=$BASE_ROOT/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/$MODEL_SNAPSHOT
MODEL_DST=$SCRATCH/models--Qwen--Qwen3-0.6B/snapshots/$MODEL_SNAPSHOT

CHAT_TEMPLATE=$OLD_SFS_ROOT/src/assets/templates/chat_template_qwen3.jinja
OUTPUT_PREDICTOR=$OLD_SFS_ROOT/src/assets/predictors/output_length_predictor
ACCURACY_PREDICTOR=$OLD_SFS_ROOT/src/assets/predictors/accuracy_predictor
HOLDOUT_BUCKET_DIR=$BASE_ROOT/vllm_utils/bucketed_prompts_0.6B
HOLDOUT_CACHE_DIR=$SCRATCH/holdout_cache_2000
TOKENIZER_ID=$MODEL_DST

PORT=$((20000 + JOB_ID % 15000))
SNAPSHOT_SHM_SIZE_BYTES=$((8 * 1024 * 1024))
SERVER_PID=""

ORIGINAL_INTERCEPT=0.002582076153059547
ORIGINAL_PREFILL_COEFF=2.0335711505545034e-06
ORIGINAL_DECODE_COEFF=1.6889500804006565e-05
ORIGINAL_SUM_COEFF=3.5429418765562916e-08
ORIGINAL_PREFILL_SQ_COEFF=1.8925400079684374e-10
ORIGINAL_CONTEXT_FEATURE_COEFF=1.5754738957656986e-13

FIT_ROOT="${FIT_ROOT:-$BASE_ROOT/sfs_experiments/state_accounting_cross_term/fit_32k_qwen3_0p6b}"
LEGACY_REFIT_COEFFICIENT_FILE="${LEGACY_REFIT_COEFFICIENT_FILE:-$FIT_ROOT/legacy/heldout_two_part_coefficients.txt}"
CROSS_TERM_COEFFICIENT_FILE="${CROSS_TERM_COEFFICIENT_FILE:-$FIT_ROOT/cross_term/heldout_two_part_coefficients.txt}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm

mkdir -p \
  "$RESULT_ROOT" \
  "$SCRATCH/.cache/vllm" \
  "$SCRATCH/.triton" \
  "$SCRATCH/.nv" \
  "$SCRATCH/matplotlib" \
  "$MODEL_DST"

rsync -aL --delete "$BASE_ROOT/.cache/vllm/" "$SCRATCH/.cache/vllm/" \
  2>/dev/null || true
rsync -aL --delete "$BASE_ROOT/.triton/" "$SCRATCH/.triton/" \
  2>/dev/null || true
rsync -aL --delete "$BASE_ROOT/.nv/" "$SCRATCH/.nv/" \
  2>/dev/null || true
rsync -aL --delete "$MODEL_SOURCE/" "$MODEL_DST/"

export XDG_CACHE_HOME=$SCRATCH/.cache
export TRITON_CACHE_DIR=$SCRATCH/.triton
export CUDA_CACHE_PATH=$SCRATCH/.nv
export MPLCONFIGDIR=$SCRATCH/matplotlib
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_LOGGING_LEVEL=WARNING

cleanup_server() {
  if [[ -n "$SERVER_PID" ]]; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}

cleanup() {
  trap - EXIT INT TERM
  cleanup_server
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_server() {
  local label="$1"
  local log_file="$2"
  local attempt
  for attempt in $(seq 1 300); do
    if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
      echo "$label server ready on port $PORT"
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "$label server exited before readiness; see $log_file" >&2
      return 1
    fi
    sleep 2
  done
  echo "$label server readiness timed out; see $log_file" >&2
  return 1
}

read_coefficient() {
  local coefficient_file="$1"
  local key="$2"
  local value
  value=$(awk -F= -v wanted="$key" '
    $1 == wanted { count += 1; value = $2 }
    END {
      if (count != 1) exit 1
      print value
    }
  ' "$coefficient_file") || {
    echo "Could not read exactly one '$key' from $coefficient_file" >&2
    return 1
  }
  local numeric_re='^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$'
  if [[ ! "$value" =~ $numeric_re ]]; then
    echo "Non-numeric coefficient '$key=$value' in $coefficient_file" >&2
    return 1
  fi
  printf '%s\n' "$value"
}

prepare_old_runtime() {
  mkdir -p "$OLD_RUNTIME_ROOT/vllm"
  rsync -aL \
    --exclude=__pycache__ \
    --exclude='*.pyc' \
    "$OLD_VLLM_ROOT/vllm/" \
    "$OLD_RUNTIME_ROOT/vllm/"
  rsync -a \
    "$NEW_VLLM_ROOT/vllm/v1/metrics/stats.py" \
    "$OLD_RUNTIME_ROOT/vllm/v1/metrics/stats.py"
  rsync -a \
    "$NEW_VLLM_ROOT/vllm/v1/metrics/loggers.py" \
    "$OLD_RUNTIME_ROOT/vllm/v1/metrics/loggers.py"

  PYTHONPATH="$OLD_RUNTIME_ROOT:$OLD_SFS_ROOT/src" python -c \
    'import vllm, sfs_core; from vllm.v1.engine import _scheduler_sim; print("old runtime with exact TTFT logging:", vllm.__file__, sfs_core.__file__, _scheduler_sim.__file__)'
}

prepare_new_runtime() {
  mkdir -p "$NEW_RUNTIME_ROOT/vllm"
  rsync -aL \
    --exclude=__pycache__ \
    --exclude='*.pyc' \
    "$OLD_VLLM_ROOT/vllm/" \
    "$NEW_RUNTIME_ROOT/vllm/"
  rsync -a \
    --exclude=__pycache__ \
    --exclude='*.pyc' \
    "$NEW_VLLM_ROOT/vllm/" \
    "$NEW_RUNTIME_ROOT/vllm/"

  local python_include
  local torch_include
  local extension_suffix
  python_include=$(python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
  torch_include=$(python -c 'from torch.utils.cpp_extension import include_paths; print(include_paths()[0])')
  extension_suffix=$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')

  c++ \
    -O2 \
    -std=c++17 \
    -shared \
    -fPIC \
    -pthread \
    -I"$python_include" \
    -I"$torch_include" \
    "$NEW_VLLM_ROOT/csrc/scheduler_sim/bindings.cpp" \
    -o "$NEW_RUNTIME_ROOT/vllm/v1/engine/_scheduler_sim$extension_suffix"

  PYTHONPATH="$NEW_RUNTIME_ROOT:$NEW_SFS_ROOT/src" python -c \
    'import vllm, sfs_core; from vllm.v1.engine import _scheduler_sim; print("new runtime:", vllm.__file__, sfs_core.__file__, _scheduler_sim.__file__)'
}

prepare_holdout_cache() {
  PYTHONPATH="$NEW_SFS_ROOT/src" python - \
    "$HOLDOUT_BUCKET_DIR" "$HOLDOUT_CACHE_DIR" "$TOKENIZER_ID" \
    "$MAX_COMPLETION_TOKENS" <<'PY'
import sys
from pathlib import Path

from sfs_core.prep.holdout_cache import prepare_holdout_prompt_cache

source_dir, cache_dir, tokenizer_id, max_completion_tokens = sys.argv[1:]
prepare_holdout_prompt_cache(
    source_bucket_dir=Path(source_dir),
    cache_dir=Path(cache_dir),
    tokenizer_id=tokenizer_id,
    holdout_start_index=2500,
    holdout_prompts_per_bucket=2000,
    holdout_context_length=65536,
    max_completion_tokens=int(max_completion_tokens),
    rebuild=True,
)
PY
}

write_instances_config() {
  local output_path="$1"
  local snapshot_name="$2"
  python - "$output_path" "$snapshot_name" "$MODEL_DST" "$PORT" \
    "$SNAPSHOT_SHM_SIZE_BYTES" <<'PY'
import json
import sys
from pathlib import Path

output_path, snapshot_name, model_path, port, shm_size = sys.argv[1:]
payload = {
    "instances": [
        {
            "instance_id": "vllm-0.6b",
            "address": f"http://127.0.0.1:{port}",
            "default_model": model_path,
            "model_id": "qwen3-0.6b",
            "snapshot_shm_name": snapshot_name,
            "snapshot_shm_size_bytes": int(shm_size),
        }
    ]
}
Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

run_case() {
  local label="$1"
  local runtime_root="$2"
  local router_root="$3"
  local feature_set="$4"
  local intercept="$5"
  local prefill_coeff="$6"
  local decode_coeff="$7"
  local sum_coeff="$8"
  local prefill_sq_coeff="$9"
  local context_feature_coeff="${10}"
  local case_dir=$RESULT_ROOT/$label
  local snapshot_name=vllm_snapshot_${JOB_ID}_${label}
  local instances_config=$case_dir/instances.json
  local wait_log=$case_dir/actual_wait_times.log
  local batch_stats=$case_dir/batch_stats.csv
  local server_log=$case_dir/vllm.log
  local driver_log=$case_dir/driver.log
  local result_json=$case_dir/result.json
  local queue_aligned_log=$case_dir/queue_aligned_ttft.jsonl

  mkdir -p "$case_dir"
  : > "$wait_log"
  write_instances_config "$instances_config" "$snapshot_name"

  local -a diagnostic_env=()
  if [[
    "$ENABLE_QUEUE_ALIGNED_DIAGNOSTIC" == "1"
    && "$runtime_root" == "$NEW_RUNTIME_ROOT"
  ]]; then
    diagnostic_env=(
      VLLM_QUEUE_ALIGNED_TTFT_DIAGNOSTIC_PATH="$queue_aligned_log"
    )
  fi

  local timing_feature_args
  if [[ "$feature_set" == "cross_term" ]]; then
    timing_feature_args=(
      --simulation-batch-time-feature-set cross_term
      --simulation-prefill-x-context-coeff "$context_feature_coeff"
    )
  elif [[ "$feature_set" == "legacy" ]]; then
    timing_feature_args=(
      --simulation-sum-sq-coeff="$context_feature_coeff"
    )
  else
    echo "Unsupported batch-time feature set: $feature_set" >&2
    return 1
  fi

  echo "Starting $label server with batch-time feature set $feature_set"
  setsid env \
    PYTHONPATH="$runtime_root:$router_root/src" \
    VLLM_PER_REQUEST_WAIT_LOG_PATH="$wait_log" \
    "${diagnostic_env[@]}" \
    vllm serve \
      --disable-uvicorn-access-log \
      --model "$MODEL_DST" \
      --max-model-len 131072 \
      --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
      --chat-template "$CHAT_TEMPLATE" \
      --enable-chunked-prefill \
      --batch-stats-file "$batch_stats" \
      --port "$PORT" \
      --max-num-seqs 512 \
      --max-num-batched-tokens 32768 \
      --gpu-memory-utilization 0.9 \
      --tensor-parallel-size 1 \
      --no-enable-wait-time-simulation \
      --enable-snapshot-shm-publishing \
      --snapshot-shm-name "$snapshot_name" \
      --snapshot-shm-size-bytes "$SNAPSHOT_SHM_SIZE_BYTES" \
      --snapshot-shm-publish-interval-ms 0 \
      --simulation-intercept "$intercept" \
      --simulation-prefill-coeff "$prefill_coeff" \
      --simulation-decode-coeff "$decode_coeff" \
      --simulation-sum-coeff "$sum_coeff" \
      --simulation-prefill-sq-coeff "$prefill_sq_coeff" \
      "${timing_feature_args[@]}" \
      --output-length-model-path "$OUTPUT_PREDICTOR" \
      --no-enable-prefix-caching \
      >"$server_log" 2>&1 &
  SERVER_PID=$!

  wait_for_server "$label" "$server_log"

  curl -fsS \
    --max-time 300 \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL_DST\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with one word.\"}],\"max_tokens\":1,\"temperature\":0}" \
    "http://127.0.0.1:$PORT/v1/chat/completions" \
    >"$case_dir/warmup.json"
  sleep 1
  : > "$wait_log"

  echo "Starting $label wait_gof run"
  PYTHONPATH="$runtime_root:$router_root/src" python -m scripts.runs.experiments \
    --experiment wait_gof \
    --instances-config "$instances_config" \
    --wait-eval-instance-id vllm-0.6b \
    --request-rate-qps "$REQUEST_RATE_QPS" \
    --arrival-process poisson \
    --decouple-arrivals \
    --seed "$SEED" \
    --wait-estimators live \
    --worker-count "$WORKER_COUNT" \
    --max-queue-size 0 \
    --max-completion-tokens "$MAX_COMPLETION_TOKENS" \
    --temperature 0 \
    --accuracy-model-path "$ACCURACY_PREDICTOR" \
    --output-length-model-path "$OUTPUT_PREDICTOR" \
    --bucket-dir "$HOLDOUT_CACHE_DIR" \
    --holdout-prompts-per-bucket 2000 \
    --holdout-bucket-dir "$HOLDOUT_BUCKET_DIR" \
    --holdout-cache-dir "$HOLDOUT_CACHE_DIR" \
    --holdout-start-index 2500 \
    --holdout-context-length 65536 \
    --tokenizer-id "$TOKENIZER_ID" \
    --num-requests "$NUM_REQUESTS" \
    --max-prompts-per-source 0 \
    --wait-gof-min-matched-pairs "$((NUM_REQUESTS * 9 / 10))" \
    --per-request-wait-log "$wait_log" \
    --output-path "$result_json" \
    >"$driver_log" 2>&1

  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "$label server exited during the driver run; see $server_log" >&2
    return 1
  fi
  cleanup_server
  echo "Completed $label run: $result_json"
}

write_comparison() {
  local baseline_label="$1"
  local candidate_label="$2"
  shift 2
  PYTHONPATH="$NEW_SFS_ROOT/src" python - \
    "$RESULT_ROOT/comparison.json" \
    "$baseline_label" \
    "$candidate_label" \
    "$@" <<'PY'
import ast
import json
import re
import statistics
import sys
from pathlib import Path

from sfs_core.eval.fit_utils import _fit_error_metrics

output_path = Path(sys.argv[1])
baseline_label = sys.argv[2]
candidate_label = sys.argv[3]
case_args = sys.argv[4:]
if len(case_args) == 0 or len(case_args) % 3 != 0:
    raise RuntimeError("Expected repeated label, result path, log path triples.")
request_id_pattern = re.compile(r"request_id=([^\s]+)")
ttft_pattern = re.compile(r"ttft_s=([0-9.+-eE]+)")
queue_pattern = re.compile(r"queue_ms=([0-9.+-eE]+)")
prefill_pattern = re.compile(r"prefill_s=([0-9.+-eE]+)")
queued_ts_pattern = re.compile(r"queued_ts_s=([0-9.+-eE]+)")
first_token_ts_pattern = re.compile(r"first_token_ts_s=([0-9.+-eE]+)")


def read_engine_latencies(path):
    values = {}
    with path.open("r", encoding="utf-8", errors="ignore") as src:
        for line in src:
            request_match = request_id_pattern.search(line)
            if request_match is None:
                continue
            parsed = {}
            for name, pattern, scale in (
                ("frontend_ttft_ms", ttft_pattern, 1000.0),
                ("queue_ms", queue_pattern, 1.0),
                ("prefill_ms", prefill_pattern, 1000.0),
                ("queued_ts_s", queued_ts_pattern, 1.0),
                ("first_token_ts_s", first_token_ts_pattern, 1.0),
            ):
                match = pattern.search(line)
                if match is not None:
                    parsed[name] = float(match.group(1)) * scale
            values[request_match.group(1)] = parsed
    return values


def read_estimate_timestamps(path):
    values = {}
    with path.open("r", encoding="utf-8", errors="ignore") as src:
        for line in src:
            try:
                record = ast.literal_eval(line)
            except (SyntaxError, ValueError):
                continue
            if not isinstance(record, dict):
                continue
            request_id = record.get("request_id")
            payload = record.get("payload")
            if not isinstance(request_id, str) or not isinstance(payload, dict):
                continue
            reports = payload.get("reports")
            if not isinstance(reports, list) or not reports:
                continue
            report = reports[0]
            if not isinstance(report, dict):
                continue
            timestamp = report.get("simulation_timestamp")
            if isinstance(timestamp, (int, float)):
                values[request_id] = float(timestamp)
    return values


def read_queue_aligned_diagnostics(path):
    predictions = {}
    build_latencies = []
    wall_times = []
    errors = []
    if not path.exists():
        return predictions, build_latencies, wall_times, errors
    with path.open("r", encoding="utf-8", errors="ignore") as src:
        for line in src:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            request_id = record.get("request_id")
            if not isinstance(request_id, str):
                continue
            if record.get("status") != "ok":
                errors.append(record)
                continue
            prediction = record.get("estimated_ttft_ms")
            if isinstance(prediction, (int, float)):
                predictions[request_id] = float(prediction)
            build_latency = record.get("snapshot_build_latency_ms")
            if isinstance(build_latency, (int, float)):
                build_latencies.append(float(build_latency))
            wall_time = record.get("diagnostic_wall_time_ms")
            if isinstance(wall_time, (int, float)):
                wall_times.append(float(wall_time))
    return predictions, build_latencies, wall_times, errors


def fit_metrics(pairs):
    actual = [value for value, _ in pairs]
    predicted = [value for _, value in pairs]
    metrics = _fit_error_metrics(actual, predicted)
    metrics["actual_mean_ms"] = sum(actual) / len(actual)
    metrics["predicted_mean_ms"] = sum(predicted) / len(predicted)
    return metrics


def value_summary(values):
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    p90_index = min(len(ordered) - 1, int(0.9 * len(ordered)))
    return {
        "count": len(values),
        "mean_ms": sum(values) / len(values),
        "p50_ms": statistics.median(values),
        "p90_ms": ordered[p90_index],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def load(path, log_path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    run = payload["wait_gof"]["runs"][0]
    items = run["per_request"]
    engine_latencies = read_engine_latencies(log_path)
    estimate_timestamps = read_estimate_timestamps(Path(run["request_log_path"]))
    queue_aligned_path = path.parent / "queue_aligned_ttft.jsonl"
    (
        queue_aligned_predictions,
        queue_aligned_build_latencies,
        queue_aligned_wall_times,
        queue_aligned_errors,
    ) = read_queue_aligned_diagnostics(queue_aligned_path)
    pairs_by_target = {
        "engine_core_queued_to_first_token_ms": [],
        "engine_first_token_from_estimate_ms": [],
        "vllm_frontend_first_token_latency_ms": [],
    }
    queue_aligned_pairs = []
    admission_gaps = []
    core_component_differences = []

    for item in items:
        response_id = item.get("response_id")
        predicted = item.get("wait_time_ms")
        if response_id not in engine_latencies or not isinstance(
            predicted, (int, float)
        ):
            continue
        observed = engine_latencies[response_id]
        queued_ts = observed.get("queued_ts_s")
        first_token_ts = observed.get("first_token_ts_s")
        if isinstance(queued_ts, float) and isinstance(first_token_ts, float):
            core_ms = (first_token_ts - queued_ts) * 1000.0
            pairs_by_target["engine_core_queued_to_first_token_ms"].append(
                (core_ms, float(predicted))
            )
            queue_aligned_prediction = queue_aligned_predictions.get(response_id)
            if queue_aligned_prediction is not None:
                queue_aligned_pairs.append(
                    (core_ms, queue_aligned_prediction)
                )
            queue_ms = observed.get("queue_ms")
            prefill_ms = observed.get("prefill_ms")
            if isinstance(queue_ms, float) and isinstance(prefill_ms, float):
                core_component_differences.append(
                    core_ms - (queue_ms + prefill_ms)
                )

            scheduler_request_id = item.get("scheduler_request_id")
            estimate_ts = estimate_timestamps.get(scheduler_request_id)
            if estimate_ts is not None:
                admission_gaps.append((queued_ts - estimate_ts) * 1000.0)
                pairs_by_target["engine_first_token_from_estimate_ms"].append(
                    ((first_token_ts - estimate_ts) * 1000.0, float(predicted))
                )

        frontend_ttft_ms = observed.get("frontend_ttft_ms")
        if isinstance(frontend_ttft_ms, float):
            pairs_by_target["vllm_frontend_first_token_latency_ms"].append(
                (frontend_ttft_ms, float(predicted))
            )

    primary_pairs = pairs_by_target["engine_core_queued_to_first_token_ms"]
    if len(primary_pairs) < len(items) * 0.9:
        raise RuntimeError(
            f"Only matched {len(primary_pairs)} of {len(items)} requests to "
            f"EngineCore QUEUED/first-token timestamps in {log_path}"
        )
    metrics_by_target = {
        target: fit_metrics(pairs)
        for target, pairs in pairs_by_target.items()
        if pairs
    }
    consistency = value_summary(core_component_differences)
    consistency["definition"] = (
        "(first_token_ts_s - queued_ts_s) * 1000 "
        "- (queue_ms + prefill_s * 1000)"
    )
    result = {
        "metrics_by_target": metrics_by_target,
        "admission_gap_ms": value_summary(admission_gaps),
        "core_timestamp_component_consistency_ms": consistency,
    }
    if queue_aligned_predictions:
        if len(queue_aligned_pairs) < len(items) * 0.9:
            raise RuntimeError(
                "Only matched "
                f"{len(queue_aligned_pairs)} of {len(items)} queue-aligned "
                f"diagnostic predictions in {queue_aligned_path}"
            )
        result["queue_aligned_diagnostic"] = {
            "actual_metric_name": "engine_core_queued_to_first_token_ms",
            "prediction_definition": (
                "native simulation from scheduler state immediately before "
                "EngineCore admission"
            ),
            "metrics": fit_metrics(queue_aligned_pairs),
            "snapshot_build_latency_ms": value_summary(
                queue_aligned_build_latencies
            ),
            "diagnostic_wall_time_ms": value_summary(
                queue_aligned_wall_times
            ),
            "error_count": len(queue_aligned_errors),
            "path": str(queue_aligned_path),
        }
    return result, primary_pairs


def segment_metrics(pairs):
    if not pairs:
        return {"count": 0, "mae": None, "mean_error": None}
    errors = [predicted - actual for actual, predicted in pairs]
    return {
        "count": len(pairs),
        "mae": sum(abs(error) for error in errors) / len(errors),
        "mean_error": sum(errors) / len(errors),
    }


cases = {}
case_pairs = {}
for offset in range(0, len(case_args), 3):
    label = case_args[offset]
    result_path = Path(case_args[offset + 1])
    log_path = Path(case_args[offset + 2])
    cases[label], case_pairs[label] = load(result_path, log_path)

primary_metric_name = "engine_core_queued_to_first_token_ms"
baseline_metrics = cases[baseline_label]["metrics_by_target"][primary_metric_name]
candidate_metrics = cases[candidate_label]["metrics_by_target"][primary_metric_name]


def metric_delta(key):
    baseline = baseline_metrics.get(key)
    candidate = candidate_metrics.get(key)
    if not isinstance(baseline, (int, float)) or not isinstance(
        candidate, (int, float)
    ):
        return None
    return float(candidate) - float(baseline)


def relative_error_change_pct(key):
    baseline = baseline_metrics.get(key)
    delta = metric_delta(key)
    if not isinstance(baseline, (int, float)) or baseline == 0 or delta is None:
        return None
    return 100.0 * float(delta) / float(baseline)


comparison = {
    "primary_actual_metric_name": primary_metric_name,
    "primary_actual_metric_definition": (
        "(first_token_ts_s - queued_ts_s) * 1000"
    ),
    "diagnostic_clock_note": (
        "engine_first_token_from_estimate_ms compares same-host monotonic/"
        "steady-clock timestamps across router and EngineCore processes; it is "
        "diagnostic only"
    ),
    "baseline_label": baseline_label,
    "candidate_label": candidate_label,
    "cases": cases,
    "delta_candidate_minus_baseline": {
        key: metric_delta(key)
        for key in ("mae", "rmse", "mean_error", "mape_pct", "r2", "pearson_r")
    },
    "relative_error_change_pct": {
        key: relative_error_change_pct(key)
        for key in ("mae", "rmse", "mape_pct")
    },
    "segments": {
        label: {
            "first_half": segment_metrics(pairs[: len(pairs) // 2]),
            "second_half": segment_metrics(pairs[len(pairs) // 2 :]),
        }
        for label, pairs in case_pairs.items()
    },
}
Path(output_path).write_text(json.dumps(comparison, indent=2), encoding="utf-8")
print(json.dumps(comparison, indent=2))
PY
}

echo "Smoke configuration: mode=$COMPARISON_MODE requests=$NUM_REQUESTS qps=$REQUEST_RATE_QPS workers=$WORKER_COUNT"
prepare_new_runtime
prepare_holdout_cache

if [[ "$COMPARISON_MODE" == "accounting" ]]; then
  prepare_old_runtime
  run_case \
    old "$OLD_RUNTIME_ROOT" "$OLD_SFS_ROOT" legacy \
    "$ORIGINAL_INTERCEPT" "$ORIGINAL_PREFILL_COEFF" \
    "$ORIGINAL_DECODE_COEFF" "$ORIGINAL_SUM_COEFF" \
    "$ORIGINAL_PREFILL_SQ_COEFF" "$ORIGINAL_CONTEXT_FEATURE_COEFF"
  run_case \
    corrected "$NEW_RUNTIME_ROOT" "$NEW_SFS_ROOT" legacy \
    "$ORIGINAL_INTERCEPT" "$ORIGINAL_PREFILL_COEFF" \
    "$ORIGINAL_DECODE_COEFF" "$ORIGINAL_SUM_COEFF" \
    "$ORIGINAL_PREFILL_SQ_COEFF" "$ORIGINAL_CONTEXT_FEATURE_COEFF"
  write_comparison \
    old corrected \
    old "$RESULT_ROOT/old/result.json" "$RESULT_ROOT/old/actual_wait_times.log" \
    corrected "$RESULT_ROOT/corrected/result.json" "$RESULT_ROOT/corrected/actual_wait_times.log"
elif [[ "$COMPARISON_MODE" == "cross_term" ]]; then
  for coefficient_file in \
    "$LEGACY_REFIT_COEFFICIENT_FILE" \
    "$CROSS_TERM_COEFFICIENT_FILE"; do
    if [[ ! -s "$coefficient_file" ]]; then
      echo "Missing coefficient file: $coefficient_file" >&2
      exit 1
    fi
  done

  legacy_refit_intercept=$(read_coefficient "$LEGACY_REFIT_COEFFICIENT_FILE" intercept)
  legacy_refit_prefill=$(read_coefficient "$LEGACY_REFIT_COEFFICIENT_FILE" beta_p)
  legacy_refit_decode=$(read_coefficient "$LEGACY_REFIT_COEFFICIENT_FILE" beta_d)
  legacy_refit_sum=$(read_coefficient "$LEGACY_REFIT_COEFFICIENT_FILE" beta_s)
  legacy_refit_prefill_sq=$(read_coefficient "$LEGACY_REFIT_COEFFICIENT_FILE" beta_p_sq_sum)
  legacy_refit_context=$(read_coefficient "$LEGACY_REFIT_COEFFICIENT_FILE" beta_s_sq)

  cross_intercept=$(read_coefficient "$CROSS_TERM_COEFFICIENT_FILE" intercept)
  cross_prefill=$(read_coefficient "$CROSS_TERM_COEFFICIENT_FILE" beta_p)
  cross_decode=$(read_coefficient "$CROSS_TERM_COEFFICIENT_FILE" beta_d)
  cross_sum=$(read_coefficient "$CROSS_TERM_COEFFICIENT_FILE" beta_s)
  cross_prefill_sq=$(read_coefficient "$CROSS_TERM_COEFFICIENT_FILE" beta_p_sq_sum)
  cross_context=$(read_coefficient "$CROSS_TERM_COEFFICIENT_FILE" beta_p_x_ctx)

  run_case \
    original_legacy "$NEW_RUNTIME_ROOT" "$NEW_SFS_ROOT" legacy \
    "$ORIGINAL_INTERCEPT" "$ORIGINAL_PREFILL_COEFF" \
    "$ORIGINAL_DECODE_COEFF" "$ORIGINAL_SUM_COEFF" \
    "$ORIGINAL_PREFILL_SQ_COEFF" "$ORIGINAL_CONTEXT_FEATURE_COEFF"
  run_case \
    refit_legacy "$NEW_RUNTIME_ROOT" "$NEW_SFS_ROOT" legacy \
    "$legacy_refit_intercept" "$legacy_refit_prefill" \
    "$legacy_refit_decode" "$legacy_refit_sum" \
    "$legacy_refit_prefill_sq" "$legacy_refit_context"
  run_case \
    refit_cross_term "$NEW_RUNTIME_ROOT" "$NEW_SFS_ROOT" cross_term \
    "$cross_intercept" "$cross_prefill" \
    "$cross_decode" "$cross_sum" \
    "$cross_prefill_sq" "$cross_context"
  write_comparison \
    refit_legacy refit_cross_term \
    original_legacy "$RESULT_ROOT/original_legacy/result.json" "$RESULT_ROOT/original_legacy/actual_wait_times.log" \
    refit_legacy "$RESULT_ROOT/refit_legacy/result.json" "$RESULT_ROOT/refit_legacy/actual_wait_times.log" \
    refit_cross_term "$RESULT_ROOT/refit_cross_term/result.json" "$RESULT_ROOT/refit_cross_term/actual_wait_times.log"
else
  echo "Unsupported COMPARISON_MODE: $COMPARISON_MODE" >&2
  exit 1
fi

echo "TTFT comparison complete: $RESULT_ROOT/comparison.json"

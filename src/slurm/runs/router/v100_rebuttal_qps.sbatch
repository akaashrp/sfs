#!/bin/bash
#SBATCH --job-name=v100-router-rebuttal
#SBATCH --output=experiments/v100_router_rebuttal_%j.out
#SBATCH --error=experiments/v100_router_rebuttal_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=GPU
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:v100-32:8
#SBATCH --mail-type=BEGIN,FAIL,END

set -euo pipefail

SBATCH_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFS_ROOT="${SFS_ROOT:-${SLURM_SUBMIT_DIR:-$SBATCH_SCRIPT_DIR}}"
while [[ "$SFS_ROOT" != "/" && ! -d "$SFS_ROOT/src/slurm" ]]; do
  SFS_ROOT="$(dirname "$SFS_ROOT")"
done
if [[ "$SFS_ROOT" == "/" ]]; then
  echo "[ERROR] Could not resolve SFS root from $SBATCH_SCRIPT_DIR" >&2
  exit 1
fi

RUN_MODE="${RUN_MODE:-}"
if [[ "$RUN_MODE" != "calibrate" && "$RUN_MODE" != "sweep" ]]; then
  echo "[ERROR] RUN_MODE must be calibrate or sweep." >&2
  exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SFS_ROOT/.." && pwd)}"
SHARED_PROMPT_ROOT="${SHARED_PROMPT_ROOT:-$PROJECT_ROOT/vllm_utils}"
CALIBRATION_PROMPT_BUCKET_DIR="${CALIBRATION_PROMPT_BUCKET_DIR:-$SHARED_PROMPT_ROOT/bucketed_prompt_outputs/holdout_cache_4000}"
HOLDOUT_BUCKET_DIR="${HOLDOUT_BUCKET_DIR:-$SHARED_PROMPT_ROOT/bucketed_prompts_0.6B}"
HOLDOUT_CACHE_DIR="${HOLDOUT_CACHE_DIR:-$SFS_ROOT/experiments/data/prompts/holdout_cache_4000_v100}"
HOLDOUT_TOKENIZER_ID="${HOLDOUT_TOKENIZER_ID:-$PROJECT_ROOT/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-v100_score_proxy}"
RUN_ROOT="${RUN_ROOT:-$SFS_ROOT/experiments/$EXPERIMENT_TAG}"
CALIBRATION_OUTPUT_PATH="${CALIBRATION_OUTPUT_PATH:-$RUN_ROOT/calibration.json}"
CALIBRATION_SHA256="${CALIBRATION_SHA256:-}"
CALIBRATION_TRACE_DIR="$RUN_ROOT/calibration_traces"
QPS_OUTPUT_DIR="$RUN_ROOT/qps_sweep"
CHAT_TEMPLATE="$SFS_ROOT/src/assets/templates/chat_template_qwen3.jinja"
OUTPUT_PREDICTOR="$SFS_ROOT/src/assets/predictors/output_length_predictor"
ACCURACY_PREDICTOR="$SFS_ROOT/src/assets/predictors/accuracy_predictor"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-49152}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-8192}"
SIMULATION_BATCH_TIME_FEATURE_SET="${SIMULATION_BATCH_TIME_FEATURE_SET:-cross_term}"
CALIBRATION_NUM_REQUESTS="${CALIBRATION_NUM_REQUESTS:-1000}"
NUM_REQUESTS="${NUM_REQUESTS:-16000}"
INFERENCE_SMOKE_ONLY="${INFERENCE_SMOKE_ONLY:-0}"
SWEEP_STARTUP_GATE="${SWEEP_STARTUP_GATE:-1}"
SMOKE_MAX_COMPLETION_TOKENS="${SMOKE_MAX_COMPLETION_TOKENS:-4}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
WORKER_COUNT="${WORKER_COUNT:-1}"
MAX_QUEUE_SIZE="${MAX_QUEUE_SIZE:-1}"
LAMBDA_WEIGHT="${LAMBDA_WEIGHT:-5e-4}"
AUTO_QPS_FRACTIONS="${AUTO_QPS_FRACTIONS:-0.4 0.6 0.8 1.0 1.1}"
QPS_UTILITIES="${QPS_UTILITIES:-hard hard_prefill_tps hard_score_proxy shortest_queue}"
PREFILL_TPS_OVERRIDES="${PREFILL_TPS_OVERRIDES:-}"

read -r -a QPS_UTILITY_ARGS <<< "$QPS_UTILITIES"
if (( ${#QPS_UTILITY_ARGS[@]} == 0 )); then
  echo "[ERROR] QPS_UTILITIES must contain at least one policy." >&2
  exit 1
fi
for utility in "${QPS_UTILITY_ARGS[@]}"; do
  case "$utility" in
    hard|hard_prefill_tps|hard_score_proxy|shortest_queue) ;;
    *)
      echo "[ERROR] Unsupported V100 rebuttal policy: $utility" >&2
      exit 1
      ;;
  esac
done

PREFILL_TPS_OVERRIDE_ARGS=()
if [[ -n "$PREFILL_TPS_OVERRIDES" ]]; then
  read -r -a PREFILL_TPS_OVERRIDE_VALUES <<< "$PREFILL_TPS_OVERRIDES"
  for override in "${PREFILL_TPS_OVERRIDE_VALUES[@]}"; do
    PREFILL_TPS_OVERRIDE_ARGS+=(--prefill-tps "$override")
  done
fi

if (( MAX_NUM_BATCHED_TOKENS <= MAX_MODEL_LEN )); then
  echo "[ERROR] Unchunked prefill requires MAX_NUM_BATCHED_TOKENS > MAX_MODEL_LEN." >&2
  exit 1
fi
if [[ "$SIMULATION_BATCH_TIME_FEATURE_SET" != "cross_term" ]]; then
  echo "[ERROR] The V100 rebuttal sweep requires cross_term batch features." >&2
  exit 1
fi
if [[ "$INFERENCE_SMOKE_ONLY" != "0" && "$INFERENCE_SMOKE_ONLY" != "1" ]]; then
  echo "[ERROR] INFERENCE_SMOKE_ONLY must be 0 or 1." >&2
  exit 1
fi
if [[ "$SWEEP_STARTUP_GATE" != "0" && "$SWEEP_STARTUP_GATE" != "1" ]]; then
  echo "[ERROR] SWEEP_STARTUP_GATE must be 0 or 1." >&2
  exit 1
fi
if [[ "$ENFORCE_EAGER" != "0" && "$ENFORCE_EAGER" != "1" ]]; then
  echo "[ERROR] ENFORCE_EAGER must be 0 or 1." >&2
  exit 1
fi
if (( SMOKE_MAX_COMPLETION_TOKENS <= 0 )); then
  echo "[ERROR] SMOKE_MAX_COMPLETION_TOKENS must be positive." >&2
  exit 1
fi

require_prompt_bucket_dir() {
  local label="$1"
  local path="$2"
  local first_jsonl
  if [[ ! -d "$path" ]]; then
    echo "[ERROR] Missing $label prompt directory: $path" >&2
    return 1
  fi
  first_jsonl="$(find "$path" -maxdepth 1 -type f -name '*.jsonl' -print -quit)"
  if [[ -z "$first_jsonl" ]]; then
    echo "[ERROR] No JSONL prompt files found in $label directory: $path" >&2
    return 1
  fi
}

if [[ "$RUN_MODE" == "calibrate" ]]; then
  require_prompt_bucket_dir calibration "$CALIBRATION_PROMPT_BUCKET_DIR"
else
  require_prompt_bucket_dir holdout "$HOLDOUT_BUCKET_DIR"
  if [[ ! -d "$HOLDOUT_TOKENIZER_ID" ]]; then
    echo "[ERROR] Missing stable holdout tokenizer snapshot: $HOLDOUT_TOKENIZER_ID" >&2
    exit 1
  fi
  if [[ ! -f "$CALIBRATION_OUTPUT_PATH" ]]; then
    echo "[ERROR] Missing calibration JSON: $CALIBRATION_OUTPUT_PATH" >&2
    exit 1
  fi
  if [[ -z "$CALIBRATION_SHA256" &&
      -f "${CALIBRATION_OUTPUT_PATH}.sha256" ]]; then
    read -r CALIBRATION_SHA256 _ <"${CALIBRATION_OUTPUT_PATH}.sha256"
  fi
  if [[ ! "$CALIBRATION_SHA256" =~ ^[[:xdigit:]]{64}$ ]]; then
    echo "[ERROR] Sweep requires CALIBRATION_SHA256 or a valid sidecar." >&2
    exit 1
  fi
  ACTUAL_CALIBRATION_SHA256="$(sha256sum "$CALIBRATION_OUTPUT_PATH" | cut -d' ' -f1)"
  if [[ "${ACTUAL_CALIBRATION_SHA256,,}" != "${CALIBRATION_SHA256,,}" ]]; then
    echo "[ERROR] Calibration SHA256 mismatch: expected=$CALIBRATION_SHA256 actual=$ACTUAL_CALIBRATION_SHA256" >&2
    exit 1
  fi
fi

mkdir -p "$RUN_ROOT" "$CALIBRATION_TRACE_DIR" "$QPS_OUTPUT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm

export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_LOGGING_LEVEL=WARNING

JOB_LOCAL="/local/$USER/$SLURM_JOB_ID"
CORRECTED_VLLM_RUNTIME="$JOB_LOCAL/score_vllm_runtime"
export CORRECTED_VLLM_RUNTIME SFS_ROOT
mkdir -p \
  "$JOB_LOCAL/.cache/vllm" \
  "$JOB_LOCAL/.triton" \
  "$JOB_LOCAL/.nv" \
  "$JOB_LOCAL/test-tmp" \
  "$CORRECTED_VLLM_RUNTIME/vllm"
rsync -aL "$PROJECT_ROOT/.cache/vllm/" "$JOB_LOCAL/.cache/vllm/" \
  2>/dev/null || true
rsync -aL "$PROJECT_ROOT/.triton/" "$JOB_LOCAL/.triton/" \
  2>/dev/null || true
rsync -aL "$PROJECT_ROOT/.nv/" "$JOB_LOCAL/.nv/" \
  2>/dev/null || true

# Start from the full editable vLLM runtime for its GPU extensions, then overlay
# this worktree's corrected Python and scheduler-simulator extension.
rsync -aL --exclude=__pycache__ --exclude='*.pyc' \
  "$PROJECT_ROOT/vllm/vllm/" "$CORRECTED_VLLM_RUNTIME/vllm/"
rsync -aL --exclude=__pycache__ --exclude='*.pyc' \
  "$SFS_ROOT/vllm/vllm/" "$CORRECTED_VLLM_RUNTIME/vllm/"
export PYTHONPATH="$CORRECTED_VLLM_RUNTIME:$SFS_ROOT/src:${PYTHONPATH:-}"
export XDG_CACHE_HOME="$JOB_LOCAL/.cache"
export TRITON_CACHE_DIR="$JOB_LOCAL/.triton"
export CUDA_CACHE_PATH="$JOB_LOCAL/.nv"
export TMPDIR="$JOB_LOCAL/test-tmp"

cd "$RUN_ROOT"
run_native_preflight() {
  python -c 'import os; from pathlib import Path; import sfs_core, vllm; from vllm.v1.engine import _scheduler_sim; runtime = Path(os.environ["CORRECTED_VLLM_RUNTIME"]).resolve(); sfs_src = (Path(os.environ["SFS_ROOT"]) / "src").resolve(); assert Path(vllm.__file__).resolve().is_relative_to(runtime), vllm.__file__; assert Path(_scheduler_sim.__file__).resolve().is_relative_to(runtime), _scheduler_sim.__file__; assert Path(sfs_core.__file__).resolve().is_relative_to(sfs_src), sfs_core.__file__; print("V100 SCORE runtime preflight:", vllm.__file__, _scheduler_sim.__file__)' &&
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q -p no:cacheprovider \
      --confcutdir="$SFS_ROOT/vllm/tests/v1/engine" \
      "$SFS_ROOT/vllm/tests/v1/engine/test_scheduler_simulator_native.py::test_native_run_simulation_matches_python_drain" \
      "$SFS_ROOT/vllm/tests/v1/engine/test_scheduler_simulator_native.py::test_native_cross_term_prices_processed_prefill_context"
}

# Validate the frozen native runtime, but never rebuild or install packages from
# inside a submitted experiment.
run_native_preflight

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-}"
if (( ${#GPU_IDS[@]} < 8 )); then
  echo "[ERROR] Expected the full eight-GPU V100 node; got ${CUDA_VISIBLE_DEVICES:-<unset>}." >&2
  exit 1
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi --query-gpu=name --format=csv,noheader |
      grep -v 'V100' >/dev/null; then
    echo "[ERROR] Allocation contains a non-V100 GPU." >&2
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader >&2
    exit 1
  fi
fi

GPU_32B="${GPU_IDS[0]},${GPU_IDS[1]},${GPU_IDS[2]},${GPU_IDS[3]}"
GPU_8B="${GPU_IDS[4]}"
GPU_06B="${GPU_IDS[5]}"
echo "[INFO] GPU mapping: 32B=$GPU_32B TP=4; 8B=$GPU_8B TP=1; 0.6B=$GPU_06B TP=1"
echo "[INFO] GPUs ${GPU_IDS[6]} and ${GPU_IDS[7]} remain unused but reserved by the exclusive node allocation."

PORT_BASE=$((20000 + (SLURM_JOB_ID % 15000) * 3))
PORT_32B=$PORT_BASE
PORT_8B=$((PORT_BASE + 1))
PORT_06B=$((PORT_BASE + 2))

stage_model() {
  local cache_name="$1"
  local snapshot="$2"
  local src="$PROJECT_ROOT/.cache/huggingface/hub/$cache_name/snapshots/$snapshot"
  local dst="$JOB_LOCAL/$cache_name/snapshots/$snapshot"
  if [[ ! -d "$src" ]]; then
    echo "[ERROR] Missing model snapshot: $src" >&2
    return 1
  fi
  mkdir -p "$dst"
  rsync -aL --delete "$src/" "$dst/"
  echo "$dst"
}

MODEL_06B="$(stage_model \
  models--Qwen--Qwen3-0.6B \
  c1899de289a04d12100db370d81485cdf75e47ca)"
MODEL_8B="$(stage_model \
  models--Qwen--Qwen3-8B \
  b968826d9c46dd6066d109eabc6255188de91218)"
MODEL_32B="$(stage_model \
  models--Qwen--Qwen3-32B \
  9216db5781bf21249d130ec9da846c4624c16137)"

PIDS=()
cleanup() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${PIDS[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

wait_for_server() {
  local label="$1"
  local port="$2"
  local pid="$3"
  local log_file="$4"
  local attempt
  for attempt in $(seq 1 900); do
    if curl -fsS "http://127.0.0.1:$port/v1/models" >/dev/null; then
      echo "[INFO] $label ready on port $port"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[ERROR] $label exited before readiness; see $log_file" >&2
      return 1
    fi
    sleep 2
  done
  echo "[ERROR] Timed out waiting for $label; see $log_file" >&2
  return 1
}

start_server() {
  local label="$1"
  local model_path="$2"
  local visible_gpus="$3"
  local tp_size="$4"
  local port="$5"
  local wait_log="$6"
  local batch_stats="$7"
  local snapshot_name="$8"
  shift 8
  local simulation_args=("$@")
  local eager_args=()
  local server_log="$RUN_ROOT/vllm_${RUN_MODE}_${label}_${SLURM_JOB_ID}.log"
  if [[ "$ENFORCE_EAGER" == "1" ]]; then
    eager_args+=(--enforce-eager)
  fi

  VLLM_PER_REQUEST_WAIT_LOG_PATH="$wait_log" \
  CUDA_VISIBLE_DEVICES="$visible_gpus" \
    vllm serve --disable-uvicorn-access-log --model "$model_path" \
      --dtype half \
      --max-model-len "$MAX_MODEL_LEN" \
      --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
      --chat-template "$CHAT_TEMPLATE" \
      --no-enable-chunked-prefill \
      --no-enable-prefix-caching \
      --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --tensor-parallel-size "$tp_size" \
      --batch-stats-file "$batch_stats" \
      --port "$port" \
      --no-enable-wait-time-simulation \
      --output-length-model-path "$OUTPUT_PREDICTOR" \
      "${eager_args[@]}" \
      "${simulation_args[@]}" \
      >"$server_log" 2>&1 &
  LAST_SERVER_PID=$!
  PIDS+=("$LAST_SERVER_PID")
  echo "[INFO] Started $label PID=$LAST_SERVER_PID log=$server_log"
}

smoke_request() {
  local label="$1"
  local port="$2"
  local model_path="$3"
  local response_path="$RUN_ROOT/inference_smoke_${RUN_MODE}_${label}_${SLURM_JOB_ID}.json"
  curl -fsS --max-time 300 \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$model_path\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with OK.\"}],\"temperature\":0,\"max_completion_tokens\":$SMOKE_MAX_COMPLETION_TOKENS,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
    "http://127.0.0.1:$port/v1/chat/completions" \
    --output "$response_path"
  python - "$response_path" "$label" <<'PY'
import json
from pathlib import Path
import sys

response_path = Path(sys.argv[1])
label = sys.argv[2]
payload = json.loads(response_path.read_text(encoding="utf-8"))
choices = payload.get("choices")
if not isinstance(choices, list) or not choices:
    raise RuntimeError(f"{label}: response has no choices: {payload}")
usage = payload.get("usage") or {}
completion_tokens = int(usage.get("completion_tokens") or 0)
if completion_tokens <= 0:
    raise RuntimeError(f"{label}: response generated no completion tokens: {payload}")
print(
    f"[SMOKE] {label} generated {completion_tokens} token(s); "
    f"finish_reason={choices[0].get('finish_reason')}"
)
PY
}

simulation_flags() {
  local snapshot_name="$1"
  local model_key="$2"
  printf '%s\n' \
    --enable-snapshot-shm-publishing \
    "--snapshot-shm-name=$snapshot_name" \
    "--snapshot-shm-size-bytes=$((8 * 1024 * 1024))" \
    --snapshot-shm-publish-interval-ms=0
  python -m scripts.runs.service_metrics_config \
    --path "$CALIBRATION_OUTPUT_PATH" \
    --expected-feature-set "$SIMULATION_BATCH_TIME_FEATURE_SET" \
    --require-nonnegative-sfs \
    simulation-args --model "$model_key"
}

{
  echo "slurm_job_id=$SLURM_JOB_ID"
  echo "run_mode=$RUN_MODE"
  echo "sfs_root=$SFS_ROOT"
  echo "sfs_commit=$(git -C "$SFS_ROOT" rev-parse HEAD)"
  echo "vllm_commit=$(git -C "$SFS_ROOT/vllm" rev-parse HEAD)"
  echo "dtype=half"
  echo "attention_backend=$VLLM_ATTENTION_BACKEND"
  echo "enforce_eager=$ENFORCE_EAGER"
  echo "inference_smoke_only=$INFERENCE_SMOKE_ONLY"
  echo "sweep_startup_gate=$SWEEP_STARTUP_GATE"
  echo "chunked_prefill=false"
  echo "prefix_caching=false"
  echo "simulation_batch_time_feature_set=$SIMULATION_BATCH_TIME_FEATURE_SET"
  echo "max_model_len=$MAX_MODEL_LEN"
  echo "max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS"
  echo "max_num_seqs=$MAX_NUM_SEQS"
  echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
  echo "max_completion_tokens=$MAX_COMPLETION_TOKENS"
  echo "qwen_enable_thinking=false"
  echo "calibration_execution=concurrent_all_models"
  echo "calibration_num_requests=$CALIBRATION_NUM_REQUESTS"
  echo "num_requests=$NUM_REQUESTS"
  echo "auto_qps_fractions=$AUTO_QPS_FRACTIONS"
  echo "explicit_qps_values=${QPS_VALUES:-}"
  echo "qps_utilities=${QPS_UTILITY_ARGS[*]}"
  echo "prefill_tps_overrides=${PREFILL_TPS_OVERRIDE_VALUES[*]:-}"
  echo "tp_qwen3_0.6b=1"
  echo "tp_qwen3_8b=1"
  echo "tp_qwen3_32b=4"
  echo "calibration_prompt_bucket_dir=$CALIBRATION_PROMPT_BUCKET_DIR"
  echo "holdout_bucket_dir=$HOLDOUT_BUCKET_DIR"
  echo "holdout_cache_dir=$HOLDOUT_CACHE_DIR"
  echo "holdout_tokenizer_id=$HOLDOUT_TOKENIZER_ID"
  echo "calibration_json=$CALIBRATION_OUTPUT_PATH"
  echo "calibration_sha256=${ACTUAL_CALIBRATION_SHA256:-}"
} | tee "$RUN_ROOT/provenance_${RUN_MODE}_${SLURM_JOB_ID}.txt"

if [[ "$RUN_MODE" == "calibrate" ]]; then
  export MODEL_METRICS_TRACE_DIR="$CALIBRATION_TRACE_DIR"
  WAIT_06B="$CALIBRATION_TRACE_DIR/actual_wait_times_qwen3-0_6b.log"
  WAIT_8B="$CALIBRATION_TRACE_DIR/actual_wait_times_qwen3-8b.log"
  WAIT_32B="$CALIBRATION_TRACE_DIR/actual_wait_times_qwen3-32b.log"
  BATCH_06B="$CALIBRATION_TRACE_DIR/batch_stats_qwen3-0_6b.csv"
  BATCH_8B="$CALIBRATION_TRACE_DIR/batch_stats_qwen3-8b.csv"
  BATCH_32B="$CALIBRATION_TRACE_DIR/batch_stats_qwen3-32b.csv"
  RESPONSE_06B="$CALIBRATION_TRACE_DIR/request_response_mapping_qwen3-0_6b.log"
  RESPONSE_8B="$CALIBRATION_TRACE_DIR/request_response_mapping_qwen3-8b.log"
  RESPONSE_32B="$CALIBRATION_TRACE_DIR/request_response_mapping_qwen3-32b.log"
  PREDICTED_06B="$CALIBRATION_TRACE_DIR/predicted_wait_times_qwen3-0_6b.log"
  PREDICTED_8B="$CALIBRATION_TRACE_DIR/predicted_wait_times_qwen3-8b.log"
  PREDICTED_32B="$CALIBRATION_TRACE_DIR/predicted_wait_times_qwen3-32b.log"
  : >"$WAIT_06B"
  : >"$WAIT_8B"
  : >"$WAIT_32B"
  : >"$BATCH_06B"
  : >"$BATCH_8B"
  : >"$BATCH_32B"
  : >"$RESPONSE_06B"
  : >"$RESPONSE_8B"
  : >"$RESPONSE_32B"
  : >"$PREDICTED_06B"
  : >"$PREDICTED_8B"
  : >"$PREDICTED_32B"

  start_server qwen3-32b "$MODEL_32B" "$GPU_32B" 4 "$PORT_32B" \
    "$WAIT_32B" "$BATCH_32B" unused
  PID_32B=$LAST_SERVER_PID
  start_server qwen3-8b "$MODEL_8B" "$GPU_8B" 1 "$PORT_8B" \
    "$WAIT_8B" "$BATCH_8B" unused
  PID_8B=$LAST_SERVER_PID
  start_server qwen3-0.6b "$MODEL_06B" "$GPU_06B" 1 "$PORT_06B" \
    "$WAIT_06B" "$BATCH_06B" unused
  PID_06B=$LAST_SERVER_PID

  wait_for_server qwen3-32b "$PORT_32B" "$PID_32B" \
    "$RUN_ROOT/vllm_${RUN_MODE}_qwen3-32b_${SLURM_JOB_ID}.log"
  wait_for_server qwen3-8b "$PORT_8B" "$PID_8B" \
    "$RUN_ROOT/vllm_${RUN_MODE}_qwen3-8b_${SLURM_JOB_ID}.log"
  wait_for_server qwen3-0.6b "$PORT_06B" "$PID_06B" \
    "$RUN_ROOT/vllm_${RUN_MODE}_qwen3-0.6b_${SLURM_JOB_ID}.log"

  if [[ "$INFERENCE_SMOKE_ONLY" == "1" ]]; then
    smoke_request qwen3-0.6b "$PORT_06B" "$MODEL_06B"
    smoke_request qwen3-8b "$PORT_8B" "$MODEL_8B"
    smoke_request qwen3-32b "$PORT_32B" "$MODEL_32B"
    echo "[DONE] V100 inference smoke passed for all three models."
    exit 0
  fi

  python -m scripts.runs.compute_model_service_metrics \
    --prompt-bucket-dir "$CALIBRATION_PROMPT_BUCKET_DIR" \
    --num-requests "$CALIBRATION_NUM_REQUESTS" \
    --request-rate-qps "$CALIBRATION_NUM_REQUESTS" \
    --max-completion-tokens "$MAX_COMPLETION_TOKENS" \
    --context-length "$MAX_MODEL_LEN" \
    --batch-fit-feature-set "$SIMULATION_BATCH_TIME_FEATURE_SET" \
    --batch-fit-nonnegative \
    --output-path "$CALIBRATION_OUTPUT_PATH" \
    --model-dir-0-6b "$MODEL_06B" \
    --model-dir-8b "$MODEL_8B" \
    --model-dir-32b "$MODEL_32B" \
    --port-0-6b "$PORT_06B" \
    --port-8b "$PORT_8B" \
    --port-32b "$PORT_32B" \
    >"$RUN_ROOT/calibration_driver_${SLURM_JOB_ID}.log" 2>&1

  python -m scripts.runs.service_metrics_config \
    --path "$CALIBRATION_OUTPUT_PATH" \
    --expected-feature-set "$SIMULATION_BATCH_TIME_FEATURE_SET" \
    --require-nonnegative-sfs \
    validate
  python -m scripts.runs.service_metrics_config \
    --path "$CALIBRATION_OUTPUT_PATH" \
    --expected-feature-set "$SIMULATION_BATCH_TIME_FEATURE_SET" \
    capacity-summary | tee "$RUN_ROOT/capacity_estimate.json"
  sha256sum "$CALIBRATION_OUTPUT_PATH" >"${CALIBRATION_OUTPUT_PATH}.sha256"
  echo "[DONE] V100 calibration: $CALIBRATION_OUTPUT_PATH"
  exit 0
fi

python -m scripts.runs.service_metrics_config \
  --path "$CALIBRATION_OUTPUT_PATH" \
  --expected-feature-set "$SIMULATION_BATCH_TIME_FEATURE_SET" \
  --require-nonnegative-sfs \
  validate

SNAPSHOT_06B="vllm_snapshot_${SLURM_JOB_ID}_qwen3_0p6b"
SNAPSHOT_8B="vllm_snapshot_${SLURM_JOB_ID}_qwen3_8b"
SNAPSHOT_32B="vllm_snapshot_${SLURM_JOB_ID}_qwen3_32b"
mapfile -t FLAGS_06B < <(simulation_flags "$SNAPSHOT_06B" qwen3-0.6b)
mapfile -t FLAGS_8B < <(simulation_flags "$SNAPSHOT_8B" qwen3-8b)
mapfile -t FLAGS_32B < <(simulation_flags "$SNAPSHOT_32B" qwen3-32b)

validate_simulation_flags() {
  local label="$1"
  shift
  python - "$label" "$@" <<'PY'
import math
import sys

from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser

label = sys.argv[1]
flags = sys.argv[2:]
parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
args = parser.parse_args(flags)
if args.simulation_batch_time_feature_set != "cross_term":
    raise RuntimeError(
        f"{label}: unexpected feature set "
        f"{args.simulation_batch_time_feature_set!r}"
    )
if args.simulation_sum_sq_coeff != 0:
    raise RuntimeError(f"{label}: legacy sum-squared coefficient was populated")
values = (
    args.simulation_intercept,
    args.simulation_prefill_coeff,
    args.simulation_prefill_sq_coeff,
    args.simulation_decode_coeff,
    args.simulation_sum_coeff,
    args.simulation_prefill_x_context_coeff,
)
if any(not math.isfinite(value) or value < 0 for value in values):
    raise RuntimeError(f"{label}: invalid parsed coefficients: {values}")
print(f"[PREFLIGHT] {label} exact vLLM simulation argv parsed: {values}")
PY
}
validate_simulation_flags qwen3-0.6b "${FLAGS_06B[@]}"
validate_simulation_flags qwen3-8b "${FLAGS_8B[@]}"
validate_simulation_flags qwen3-32b "${FLAGS_32B[@]}"

WAIT_06B="$RUN_ROOT/actual_wait_times_qwen3-0_6b_qps_${SLURM_JOB_ID}.log"
WAIT_8B="$RUN_ROOT/actual_wait_times_qwen3-8b_qps_${SLURM_JOB_ID}.log"
WAIT_32B="$RUN_ROOT/actual_wait_times_qwen3-32b_qps_${SLURM_JOB_ID}.log"
BATCH_06B="$RUN_ROOT/batch_stats_qwen3-0_6b_qps_${SLURM_JOB_ID}.csv"
BATCH_8B="$RUN_ROOT/batch_stats_qwen3-8b_qps_${SLURM_JOB_ID}.csv"
BATCH_32B="$RUN_ROOT/batch_stats_qwen3-32b_qps_${SLURM_JOB_ID}.csv"
: >"$WAIT_06B"
: >"$WAIT_8B"
: >"$WAIT_32B"

start_server qwen3-32b "$MODEL_32B" "$GPU_32B" 4 "$PORT_32B" \
  "$WAIT_32B" "$BATCH_32B" "$SNAPSHOT_32B" "${FLAGS_32B[@]}"
PID_32B=$LAST_SERVER_PID
start_server qwen3-8b "$MODEL_8B" "$GPU_8B" 1 "$PORT_8B" \
  "$WAIT_8B" "$BATCH_8B" "$SNAPSHOT_8B" "${FLAGS_8B[@]}"
PID_8B=$LAST_SERVER_PID
start_server qwen3-0.6b "$MODEL_06B" "$GPU_06B" 1 "$PORT_06B" \
  "$WAIT_06B" "$BATCH_06B" "$SNAPSHOT_06B" "${FLAGS_06B[@]}"
PID_06B=$LAST_SERVER_PID

wait_for_server qwen3-32b "$PORT_32B" "$PID_32B" \
  "$RUN_ROOT/vllm_${RUN_MODE}_qwen3-32b_${SLURM_JOB_ID}.log"
wait_for_server qwen3-8b "$PORT_8B" "$PID_8B" \
  "$RUN_ROOT/vllm_${RUN_MODE}_qwen3-8b_${SLURM_JOB_ID}.log"
wait_for_server qwen3-0.6b "$PORT_06B" "$PID_06B" \
  "$RUN_ROOT/vllm_${RUN_MODE}_qwen3-0.6b_${SLURM_JOB_ID}.log"

if [[ "$SWEEP_STARTUP_GATE" == "1" ]]; then
  # Keep this gate at the public HTTP boundary. A short-lived Python process
  # must not attach to server-owned snapshot SHM because its resource tracker
  # can unlink the segments when that process exits.
  smoke_request qwen3-0.6b "$PORT_06B" "$MODEL_06B"
  smoke_request qwen3-8b "$PORT_8B" "$MODEL_8B"
  smoke_request qwen3-32b "$PORT_32B" "$MODEL_32B"
  echo "[DONE] In-job inference smoke passed for all three model instances."
fi

INSTANCES_CONFIG="$RUN_ROOT/instances_qps_${SLURM_JOB_ID}.json"
cat >"$INSTANCES_CONFIG" <<EOF
{
  "instances": [
    {
      "instance_id": "vllm-32b",
      "address": "http://localhost:$PORT_32B",
      "default_model": "$MODEL_32B",
      "model_id": "qwen3-32b",
      "snapshot_shm_name": "$SNAPSHOT_32B",
      "snapshot_shm_size_bytes": $((8 * 1024 * 1024))
    },
    {
      "instance_id": "vllm-8b",
      "address": "http://localhost:$PORT_8B",
      "default_model": "$MODEL_8B",
      "model_id": "qwen3-8b",
      "snapshot_shm_name": "$SNAPSHOT_8B",
      "snapshot_shm_size_bytes": $((8 * 1024 * 1024))
    },
    {
      "instance_id": "vllm-0.6b",
      "address": "http://localhost:$PORT_06B",
      "default_model": "$MODEL_06B",
      "model_id": "qwen3-0.6b",
      "snapshot_shm_name": "$SNAPSHOT_06B",
      "snapshot_shm_size_bytes": $((8 * 1024 * 1024))
    }
  ]
}
EOF

if [[ -n "${QPS_VALUES:-}" ]]; then
  read -r -a QPS_SWEEP <<< "$QPS_VALUES"
else
  read -r -a QPS_FRACTIONS <<< "$AUTO_QPS_FRACTIONS"
  QPS_TEXT="$(
    python -m scripts.runs.service_metrics_config \
      --path "$CALIBRATION_OUTPUT_PATH" \
      --expected-feature-set "$SIMULATION_BATCH_TIME_FEATURE_SET" \
      qps-values --fractions "${QPS_FRACTIONS[@]}"
  )"
  read -r -a QPS_SWEEP <<< "$QPS_TEXT"
fi
if (( ${#QPS_SWEEP[@]} == 0 )); then
  echo "[ERROR] QPS sweep is empty." >&2
  exit 1
fi
echo "[INFO] QPS sweep values: ${QPS_SWEEP[*]}"

mkdir -p "$QPS_OUTPUT_DIR"
python -m scripts.runs.experiments_sweep \
  --sweep qps \
  --qps-values "${QPS_SWEEP[@]}" \
  --qps-utilities "${QPS_UTILITY_ARGS[@]}" \
  --output-dir "$QPS_OUTPUT_DIR" \
  --output-prefix "router_qps_qwen3_v100_fp16_unchunked" \
  --experiment router \
  --instances-config "$INSTANCES_CONFIG" \
  --service-metrics-json "$CALIBRATION_OUTPUT_PATH" \
  "${PREFILL_TPS_OVERRIDE_ARGS[@]}" \
  --arrival-process poisson \
  --num-requests "$NUM_REQUESTS" \
  --worker-count "$WORKER_COUNT" \
  --max-queue-size "$MAX_QUEUE_SIZE" \
  --max-completion-tokens "$MAX_COMPLETION_TOKENS" \
  --temperature 0 \
  --queue-slo-min-ms 1 \
  --queue-slo-max-ms 5 \
  --ttft-slo-min-ms 150 \
  --ttft-slo-max-ms 1120 \
  --ttft-slo-base-ms 155 \
  --ttft-slo-per-prompt-token-ms 0.035 \
  --ttft-slo-jitter-min 0.98 \
  --ttft-slo-jitter-max 1.02 \
  --feasible-slo-mode ttft \
  --lambda-weight "$LAMBDA_WEIGHT" \
  --delta-weight 0 \
  --holdout-prompts-per-bucket 4000 \
  --holdout-bucket-dir "$HOLDOUT_BUCKET_DIR" \
  --holdout-cache-dir "$HOLDOUT_CACHE_DIR" \
  --holdout-start-index 2500 \
  --holdout-context-length "$MAX_MODEL_LEN" \
  --tokenizer-id "$HOLDOUT_TOKENIZER_ID" \
  --accuracy-model-path "$ACCURACY_PREDICTOR" \
  --output-length-model-path "$OUTPUT_PREDICTOR" \
  --per-request-wait-log "$WAIT_32B" \
  --per-request-wait-log "$WAIT_8B" \
  --per-request-wait-log "$WAIT_06B" \
  >"$RUN_ROOT/qps_driver_${SLURM_JOB_ID}.log" 2>&1

echo "[DONE] V100 QPS sweep: $QPS_OUTPUT_DIR"

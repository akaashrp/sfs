#!/bin/bash
# Shared Ministral 3 preparation driver. Submit one of the hardware-specific
# wrappers in this directory instead of submitting this file directly.

set -euo pipefail

mkdir_with_retry() {
  local target="$1"
  local attempts="${2:-8}"
  local delay_s="${3:-2}"
  local attempt
  for attempt in $(seq 1 "$attempts"); do
    if mkdir -p "$target"; then
      return 0
    fi
    echo "[WARN] mkdir attempt ${attempt}/${attempts} failed for: $target" >&2
    sleep "$delay_s"
  done
  echo "[ERROR] mkdir failed after ${attempts} attempts: $target" >&2
  return 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFS_ROOT="$SCRIPT_DIR"
while [[ "$SFS_ROOT" != "/" && ! -d "$SFS_ROOT/src/slurm" ]]; do
  SFS_ROOT="$(dirname "$SFS_ROOT")"
done
if [[ "$SFS_ROOT" == "/" ]]; then
  echo "[ERROR] Could not resolve SFS repo root from $SCRIPT_DIR" >&2
  exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SFS_ROOT/.." && pwd)}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$PROJECT_ROOT/.cache/huggingface/hub}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$SFS_ROOT/experiments}"
SHARED_PROMPT_ROOT="${SHARED_PROMPT_ROOT:-$PROJECT_ROOT/vllm_utils}"

# The family is deliberately explicit here. These are the official BF16
# Instruct checkpoints, not the default FP8 repositories, so the same Mistral-
# format weights can be cast to FP16 on V100 as well as H100.
MODEL_REPO_3B="mistralai/Ministral-3-3B-Instruct-2512-BF16"
MODEL_REPO_8B="mistralai/Ministral-3-8B-Instruct-2512-BF16"
MODEL_REPO_14B="mistralai/Ministral-3-14B-Instruct-2512-BF16"
MODEL_CACHE_3B="${MODEL_CACHE_3B:-models--mistralai--Ministral-3-3B-Instruct-2512-BF16}"
MODEL_CACHE_8B="${MODEL_CACHE_8B:-models--mistralai--Ministral-3-8B-Instruct-2512-BF16}"
MODEL_CACHE_14B="${MODEL_CACHE_14B:-models--mistralai--Ministral-3-14B-Instruct-2512-BF16}"
MODEL_SNAPSHOT_3B="b6d637bef2393152b3da2b2fde72eecdee30557e"
MODEL_SNAPSHOT_8B="f6fae9795746f63c9be8344932f01275f3c63734"
MODEL_SNAPSHOT_14B="3cea74c1ebaf5ce5f5a2553de470e2ceab825142"
SERVED_MODEL_3B="${SERVED_MODEL_3B:-ministral3-3b-instruct}"
SERVED_MODEL_8B="${SERVED_MODEL_8B:-ministral3-8b-instruct}"
SERVED_MODEL_14B="${SERVED_MODEL_14B:-ministral3-14b-instruct}"

# The wrapper owns hardware-dependent placement. Device indices themselves are
# resolved from Slurm's CUDA_VISIBLE_DEVICES allocation below.
GPU_PROFILE="${MINISTRAL_GPU_PROFILE:-}"
TP_3B="${MINISTRAL_TP_3B:-}"
TP_8B="${MINISTRAL_TP_8B:-}"
TP_14B="${MINISTRAL_TP_14B:-}"
VLLM_DTYPE="${VLLM_DTYPE:-half}"
TOKENIZER_MODE="${TOKENIZER_MODE:-mistral}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

# Input, prompt-policy, and generation settings. Ministral Instruct does not
# use Qwen's enable_thinking template kwarg, so the explicit policy is empty.
BUCKET_DIR="${BUCKET_DIR:-$SHARED_PROMPT_ROOT/bucketed_prompts_0.6B}"
MAX_PROMPTS_PER_BUCKET="${MAX_PROMPTS_PER_BUCKET:-2500}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a helpful assistant.}"
CHAT_TEMPLATE_KWARGS_JSON="${CHAT_TEMPLATE_KWARGS_JSON:-}"
if [[ -z "$CHAT_TEMPLATE_KWARGS_JSON" ]]; then
  CHAT_TEMPLATE_KWARGS_JSON='{}'
fi
CONTEXT_LENGTH="${CONTEXT_LENGTH:-40960}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-8192}"
PROMPT_TOKEN_LIMIT="${PROMPT_TOKEN_LIMIT:-32768}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-69}"

MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-49152}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
WORKER_COUNT="${WORKER_COUNT:-16}"
MAX_QUEUE_SIZE="${MAX_QUEUE_SIZE:-0}"
RUN_IN_PARALLEL="${RUN_IN_PARALLEL:-1}"
VALIDATE_ONLY="${VALIDATE_ONLY:-0}"

for required_name in GPU_PROFILE TP_3B TP_8B TP_14B; do
  if [[ -z "${!required_name}" ]]; then
    echo "[ERROR] $required_name must be set by a hardware wrapper." >&2
    exit 1
  fi
done
for integer_name in TP_3B TP_8B TP_14B CONTEXT_LENGTH MAX_NUM_BATCHED_TOKENS \
  MAX_NUM_SEQS WORKER_COUNT; do
  if ! [[ "${!integer_name}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] $integer_name must be a positive integer; got ${!integer_name}." >&2
    exit 1
  fi
done
for boolean_name in CHUNKED_PREFILL ENFORCE_EAGER RUN_IN_PARALLEL VALIDATE_ONLY; do
  if [[ "${!boolean_name}" != "0" && "${!boolean_name}" != "1" ]]; then
    echo "[ERROR] $boolean_name must be 0 or 1; got ${!boolean_name}." >&2
    exit 1
  fi
done
if [[ "$TOKENIZER_MODE" != "mistral" ]]; then
  echo "[ERROR] Ministral launchers require TOKENIZER_MODE=mistral." >&2
  exit 1
fi
if [[ "$CHAT_TEMPLATE_KWARGS_JSON" != "{}" ]]; then
  echo "[ERROR] Ministral Instruct requires CHAT_TEMPLATE_KWARGS_JSON='{}'." >&2
  exit 1
fi
if [[ "$CHUNKED_PREFILL" == "0" ]] && (( MAX_NUM_BATCHED_TOKENS <= CONTEXT_LENGTH )); then
  echo "[ERROR] Unchunked prefill requires MAX_NUM_BATCHED_TOKENS > CONTEXT_LENGTH." >&2
  exit 1
fi

REQUIRED_GPU_COUNT=$((TP_3B + TP_8B + TP_14B))
if [[ "$VALIDATE_ONLY" == "1" ]]; then
  echo "profile=$GPU_PROFILE gpu_count=$REQUIRED_GPU_COUNT dtype=$VLLM_DTYPE tokenizer_mode=$TOKENIZER_MODE"
  echo "ministral3-3b repo=$MODEL_REPO_3B snapshot=$MODEL_SNAPSHOT_3B tp=$TP_3B"
  echo "ministral3-8b repo=$MODEL_REPO_8B snapshot=$MODEL_SNAPSHOT_8B tp=$TP_8B"
  echo "ministral3-14b repo=$MODEL_REPO_14B snapshot=$MODEL_SNAPSHOT_14B tp=$TP_14B"
  exit 0
fi

if [[ ! -d "$BUCKET_DIR" ]]; then
  echo "[ERROR] Bucket directory does not exist: $BUCKET_DIR" >&2
  exit 1
fi

resolve_model_path() {
  local cache_name="$1"
  local snapshot="$2"
  local snapshot_path="$MODEL_CACHE_ROOT/$cache_name/snapshots/$snapshot"
  if [[ ! -d "$snapshot_path" ]]; then
    echo "[ERROR] Missing pinned model snapshot: $snapshot_path" >&2
    return 1
  fi
  for required_file in config.json params.json tekken.json consolidated.safetensors; do
    if [[ ! -f "$snapshot_path/$required_file" ]]; then
      echo "[ERROR] Incomplete model snapshot; missing $snapshot_path/$required_file" >&2
      return 1
    fi
  done
  echo "$snapshot_path"
}

MODEL_SOURCE_3B="$(resolve_model_path "$MODEL_CACHE_3B" "$MODEL_SNAPSHOT_3B")"
MODEL_SOURCE_8B="$(resolve_model_path "$MODEL_CACHE_8B" "$MODEL_SNAPSHOT_8B")"
MODEL_SOURCE_14B="$(resolve_model_path "$MODEL_CACHE_14B" "$MODEL_SNAPSHOT_14B")"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm
export PYTHONPATH="$SFS_ROOT/src:${PYTHONPATH:-}"

# The environment is currently editable-installed from another SCORE worktree.
# That is safe only while it resolves to the exact nested vLLM commit recorded
# by this worktree, including the in-flight/stale-snapshot accounting changes.
EXPECTED_VLLM_COMMIT="$(git -C "$SFS_ROOT/vllm" rev-parse HEAD)"
ACTIVE_VLLM_ROOT="$(cd "$SFS_ROOT/src" && python - <<'PY'
from pathlib import Path
import vllm

print(Path(vllm.__file__).resolve().parent.parent)
PY
)"
if ! ACTIVE_VLLM_COMMIT="$(git -C "$ACTIVE_VLLM_ROOT" rev-parse HEAD 2>/dev/null)"; then
  echo "[ERROR] Active vLLM is not an editable Git checkout: $ACTIVE_VLLM_ROOT" >&2
  exit 1
fi
if [[ "$ACTIVE_VLLM_COMMIT" != "$EXPECTED_VLLM_COMMIT" ]]; then
  echo "[ERROR] Active vLLM commit $ACTIVE_VLLM_COMMIT does not match recorded $EXPECTED_VLLM_COMMIT." >&2
  exit 1
fi

RUN_STAMP="${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$EXPERIMENT_DIR/ministral3_${GPU_PROFILE}_bucketed_all_models_${RUN_STAMP}"
JOB_ID="${SLURM_JOB_ID:-$$}"
JOB_LOCAL="/local/$USER/$JOB_ID"
mkdir_with_retry "$RUN_DIR/completions"
mkdir_with_retry "$JOB_LOCAL/.cache/vllm"
mkdir_with_retry "$JOB_LOCAL/.triton"
mkdir_with_retry "$JOB_LOCAL/.nv"

rsync -aL "$PROJECT_ROOT/.cache/vllm/" "$JOB_LOCAL/.cache/vllm/" 2>/dev/null || true
rsync -aL "$PROJECT_ROOT/.triton/" "$JOB_LOCAL/.triton/" 2>/dev/null || true
rsync -aL "$PROJECT_ROOT/.nv/" "$JOB_LOCAL/.nv/" 2>/dev/null || true

export XDG_CACHE_HOME="$JOB_LOCAL/.cache"
export TRITON_CACHE_DIR="$JOB_LOCAL/.triton"
export CUDA_CACHE_PATH="$JOB_LOCAL/.nv"
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:?hardware wrapper must set VLLM_ATTENTION_BACKEND}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_LOGGING_LEVEL=WARNING
export EXPERIMENT_DIR

stage_model() {
  local cache_name="$1"
  local snapshot="$2"
  local src="$3"
  local dst="$JOB_LOCAL/$cache_name/snapshots/$snapshot"
  mkdir_with_retry "$dst"
  # The BF16 repos also carry indexed Hugging Face shards. This fork supports
  # Ministral 3 through its Mistral-format loader, so stage only the equivalent
  # consolidated weights and avoid copying both representations.
  rsync -aL --delete --delete-excluded \
    --exclude='model-*.safetensors' "$src/" "$dst/"
  echo "$dst"
}

MODEL_DST_3B="$(stage_model "$MODEL_CACHE_3B" "$MODEL_SNAPSHOT_3B" "$MODEL_SOURCE_3B")"
MODEL_DST_8B="$(stage_model "$MODEL_CACHE_8B" "$MODEL_SNAPSHOT_8B" "$MODEL_SOURCE_8B")"
MODEL_DST_14B="$(stage_model "$MODEL_CACHE_14B" "$MODEL_SNAPSHOT_14B" "$MODEL_SOURCE_14B")"

default_gpu_ids="$(seq -s, 0 $((REQUIRED_GPU_COUNT - 1)))"
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-$default_gpu_ids}"
if (( ${#GPU_IDS[@]} < REQUIRED_GPU_COUNT )); then
  echo "[ERROR] Need $REQUIRED_GPU_COUNT visible GPUs for $GPU_PROFILE; got ${CUDA_VISIBLE_DEVICES:-<unset>}." >&2
  exit 1
fi

slice_gpu_ids() {
  local start="$1"
  local count="$2"
  local selected=("${GPU_IDS[@]:start:count}")
  local joined
  IFS=,
  joined="${selected[*]}"
  unset IFS
  echo "$joined"
}

GPU_3B="$(slice_gpu_ids 0 "$TP_3B")"
GPU_8B="$(slice_gpu_ids "$TP_3B" "$TP_8B")"
GPU_14B="$(slice_gpu_ids $((TP_3B + TP_8B)) "$TP_14B")"

PORT_BASE=$((20000 + (JOB_ID % 15000) * 3))
PORT_3B="$PORT_BASE"
PORT_8B=$((PORT_BASE + 1))
PORT_14B=$((PORT_BASE + 2))

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

start_server() {
  local label="$1"
  local model_path="$2"
  local served_model="$3"
  local visible_gpus="$4"
  local tp_size="$5"
  local port="$6"
  local server_log="$RUN_DIR/vllm_${label}.log"
  local wait_log="$RUN_DIR/per_request_wait_${label}.log"
  local batch_stats="$RUN_DIR/batch_stats_${label}.csv"
  local chunked_args=(--no-enable-chunked-prefill)
  local eager_args=()

  if [[ "$CHUNKED_PREFILL" == "1" ]]; then
    chunked_args=(--enable-chunked-prefill)
  fi
  if [[ "$ENFORCE_EAGER" == "1" ]]; then
    eager_args=(--enforce-eager)
  fi

  VLLM_PER_REQUEST_WAIT_LOG_PATH="$wait_log" \
  CUDA_VISIBLE_DEVICES="$visible_gpus" \
    vllm serve --disable-uvicorn-access-log --model "$model_path" \
      --served-model-name "$served_model" \
      --tokenizer-mode "$TOKENIZER_MODE" \
      --config-format mistral \
      --load-format mistral \
      --dtype "$VLLM_DTYPE" \
      --max-model-len "$CONTEXT_LENGTH" \
      "${chunked_args[@]}" \
      --no-enable-prefix-caching \
      --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --tensor-parallel-size "$tp_size" \
      --limit-mm-per-prompt '{"image":0}' \
      --mm-processor-cache-gb 0 \
      --batch-stats-file "$batch_stats" \
      --port "$port" \
      --no-enable-wait-time-simulation \
      "${eager_args[@]}" \
      >"$server_log" 2>&1 &
  LAST_SERVER_PID=$!
  PIDS+=("$LAST_SERVER_PID")
  echo "[INFO] Started $label on GPU(s) $visible_gpus TP=$tp_size; PID=$LAST_SERVER_PID log=$server_log"
}

wait_for_server() {
  local label="$1"
  local port="$2"
  local pid="$3"
  local log_file="$4"
  local attempt
  for attempt in $(seq 1 600); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null; then
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

# Running from the superproject root shadows the editable vLLM package with
# the nested repository directory. The src directory keeps both the vLLM CLI
# and SFS module imports unambiguous.
cd "$SFS_ROOT/src"
start_server ministral3-14b "$MODEL_DST_14B" "$SERVED_MODEL_14B" "$GPU_14B" "$TP_14B" "$PORT_14B"
PID_14B="$LAST_SERVER_PID"
start_server ministral3-8b "$MODEL_DST_8B" "$SERVED_MODEL_8B" "$GPU_8B" "$TP_8B" "$PORT_8B"
PID_8B="$LAST_SERVER_PID"
start_server ministral3-3b "$MODEL_DST_3B" "$SERVED_MODEL_3B" "$GPU_3B" "$TP_3B" "$PORT_3B"
PID_3B="$LAST_SERVER_PID"

wait_for_server ministral3-14b "$PORT_14B" "$PID_14B" "$RUN_DIR/vllm_ministral3-14b.log"
wait_for_server ministral3-8b "$PORT_8B" "$PID_8B" "$RUN_DIR/vllm_ministral3-8b.log"
wait_for_server ministral3-3b "$PORT_3B" "$PID_3B" "$RUN_DIR/vllm_ministral3-3b.log"

nvidia-smi --query-gpu=index,name,uuid,memory.total \
  --format=csv,noheader >"$RUN_DIR/gpus.csv" 2>/dev/null || true
{
  echo "slurm_job_id=${SLURM_JOB_ID:-local}"
  echo "sfs_commit=$(git -C "$SFS_ROOT" rev-parse HEAD)"
  echo "recorded_vllm_commit=$EXPECTED_VLLM_COMMIT"
  echo "active_vllm_root=$ACTIVE_VLLM_ROOT"
  echo "active_vllm_commit=$ACTIVE_VLLM_COMMIT"
  echo "gpu_profile=$GPU_PROFILE"
  echo "model_3b=$MODEL_REPO_3B@$MODEL_SNAPSHOT_3B"
  echo "model_8b=$MODEL_REPO_8B@$MODEL_SNAPSHOT_8B"
  echo "model_14b=$MODEL_REPO_14B@$MODEL_SNAPSHOT_14B"
  echo "served_model_3b=$SERVED_MODEL_3B"
  echo "served_model_8b=$SERVED_MODEL_8B"
  echo "served_model_14b=$SERVED_MODEL_14B"
  echo "gpu_mapping_3b=$GPU_3B"
  echo "gpu_mapping_8b=$GPU_8B"
  echo "gpu_mapping_14b=$GPU_14B"
  echo "tensor_parallel_3b=$TP_3B"
  echo "tensor_parallel_8b=$TP_8B"
  echo "tensor_parallel_14b=$TP_14B"
  echo "dtype=$VLLM_DTYPE"
  echo "attention_backend=$VLLM_ATTENTION_BACKEND"
  echo "tokenizer_mode=$TOKENIZER_MODE"
  echo "config_format=mistral"
  echo "load_format=mistral"
  echo "context_length=$CONTEXT_LENGTH"
  echo "rope_scaling=model_native_yarn"
  echo "system_prompt=$SYSTEM_PROMPT"
  echo "chat_template_kwargs_json=$CHAT_TEMPLATE_KWARGS_JSON"
  echo "multimodal_image_limit=0"
  echo "mm_processor_cache_gb=0"
  echo "chunked_prefill=$CHUNKED_PREFILL"
  echo "prefix_caching=false"
  echo "max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS"
  echo "max_num_seqs=$MAX_NUM_SEQS"
  echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
  echo "max_completion_tokens=$MAX_COMPLETION_TOKENS"
  echo "prompt_token_limit=$PROMPT_TOKEN_LIMIT"
  echo "temperature=$TEMPERATURE"
  echo "top_p=$TOP_P"
  echo "seed=$SEED"
  echo "bucket_dir=$BUCKET_DIR"
} >"$RUN_DIR/provenance.txt"

run_model() {
  local label="$1"
  local instance_id="$2"
  local model_path="$3"
  local served_model="$4"
  local port="$5"
  local run_log="$RUN_DIR/run_${label}.log"
  local request_log="$RUN_DIR/predicted_wait_${label}.log"

  python -m scripts.prep.run_qwen_bucketed_prompts \
    --bucket-dir "$BUCKET_DIR" \
    --model-id "$served_model" \
    --instance-address "http://127.0.0.1:${port}" \
    --job-label "$label" \
    --instance-id "$instance_id" \
    --tokenizer-id "$model_path" \
    --tokenizer-mode "$TOKENIZER_MODE" \
    --run-dir "$RUN_DIR/completions" \
    --system-prompt "$SYSTEM_PROMPT" \
    --chat-template-kwargs-json "$CHAT_TEMPLATE_KWARGS_JSON" \
    --context-length "$CONTEXT_LENGTH" \
    --max-completion-tokens "$MAX_COMPLETION_TOKENS" \
    --prompt-token-limit "$PROMPT_TOKEN_LIMIT" \
    --max-prompts-per-bucket "$MAX_PROMPTS_PER_BUCKET" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --seed "$SEED" \
    --worker-count "$WORKER_COUNT" \
    --max-queue-size "$MAX_QUEUE_SIZE" \
    --disable-wait-time-polling \
    --request-log-path "$request_log" \
    >"$run_log" 2>&1
}

if [[ "$RUN_IN_PARALLEL" == "1" ]]; then
  run_model ministral3-3b vllm-3b "$MODEL_DST_3B" "$SERVED_MODEL_3B" "$PORT_3B" &
  RUN_PID_3B=$!
  run_model ministral3-8b vllm-8b "$MODEL_DST_8B" "$SERVED_MODEL_8B" "$PORT_8B" &
  RUN_PID_8B=$!
  run_model ministral3-14b vllm-14b "$MODEL_DST_14B" "$SERVED_MODEL_14B" "$PORT_14B" &
  RUN_PID_14B=$!

  FAILED=0
  for pid in "$RUN_PID_3B" "$RUN_PID_8B" "$RUN_PID_14B"; do
    if ! wait "$pid"; then
      FAILED=1
    fi
  done
  if (( FAILED != 0 )); then
    echo "[ERROR] At least one Ministral generation process failed." >&2
    exit 1
  fi
else
  run_model ministral3-3b vllm-3b "$MODEL_DST_3B" "$SERVED_MODEL_3B" "$PORT_3B"
  run_model ministral3-8b vllm-8b "$MODEL_DST_8B" "$SERVED_MODEL_8B" "$PORT_8B"
  run_model ministral3-14b vllm-14b "$MODEL_DST_14B" "$SERVED_MODEL_14B" "$PORT_14B"
fi

echo "[DONE] Completed all Ministral 3 generation runs."
echo "[DONE] Run directory: $RUN_DIR"
echo "[NEXT] Score aligned outputs with:"
echo "python -m scripts.prep.quality_metrics --outputs-root '$RUN_DIR/completions' --models ministral3-3b ministral3-8b ministral3-14b"
find "$RUN_DIR/completions" -name 'run_summary_*.json' -print | sort

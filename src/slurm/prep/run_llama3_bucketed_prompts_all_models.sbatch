#!/bin/bash
#SBATCH --job-name=llama3-bucketed-all-models
#SBATCH --output=experiments/llama3_bucketed_all_models_%j.out
#SBATCH --error=experiments/llama3_bucketed_all_models_%j.err
#SBATCH --time=04:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=GPU-shared
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100-80:3
#SBATCH --mail-type=BEGIN,FAIL,END

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

SBATCH_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFS_ROOT="$SBATCH_SCRIPT_DIR"
while [[ "$SFS_ROOT" != "/" && ! -d "$SFS_ROOT/src/slurm" ]]; do
  SFS_ROOT="$(dirname "$SFS_ROOT")"
done
if [[ "$SFS_ROOT" == "/" ]]; then
  echo "[ERROR] Could not resolve SFS repo root from $SBATCH_SCRIPT_DIR" >&2
  exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SFS_ROOT/.." && pwd)}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$PROJECT_ROOT/.cache/huggingface/hub}"
EXPERIMENT_DIR="$SFS_ROOT/experiments"
SHARED_PROMPT_ROOT="${SHARED_PROMPT_ROOT:-$PROJECT_ROOT/vllm_utils}"

# This family-specific launcher intentionally keeps model identity and serving
# settings visible here instead of introducing a repository-wide manifest.
MODEL_CACHE_1B="models--meta-llama--Llama-3.2-1B-Instruct"
MODEL_CACHE_3B="models--meta-llama--Llama-3.2-3B-Instruct"
MODEL_CACHE_8B="models--meta-llama--Llama-3.1-8B-Instruct"
MODEL_SNAPSHOT_1B="${MODEL_SNAPSHOT_1B:-}"
MODEL_SNAPSHOT_3B="${MODEL_SNAPSHOT_3B:-}"
MODEL_SNAPSHOT_8B="${MODEL_SNAPSHOT_8B:-}"
SERVED_MODEL_1B="${SERVED_MODEL_1B:-llama3.2-1b-instruct}"
SERVED_MODEL_3B="${SERVED_MODEL_3B:-llama3.2-3b-instruct}"
SERVED_MODEL_8B="${SERVED_MODEL_8B:-llama3.1-8b-instruct}"

# Input, prompt-policy, and generation settings.
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

# Unchunked-prefill settings match the later V100 routing configuration. Each
# selected model also fits on one H100 here, so tensor parallelism remains 1.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-49152}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
WORKER_COUNT="${WORKER_COUNT:-16}"
MAX_QUEUE_SIZE="${MAX_QUEUE_SIZE:-0}"
RUN_IN_PARALLEL="${RUN_IN_PARALLEL:-1}"

if (( MAX_NUM_BATCHED_TOKENS <= CONTEXT_LENGTH )); then
  echo "[ERROR] Unchunked prefill requires MAX_NUM_BATCHED_TOKENS > CONTEXT_LENGTH." >&2
  exit 1
fi
if [[ ! -d "$BUCKET_DIR" ]]; then
  echo "[ERROR] Bucket directory does not exist: $BUCKET_DIR" >&2
  exit 1
fi

resolve_snapshot() {
  local cache_name="$1"
  local requested_snapshot="$2"
  local snapshot_root="$MODEL_CACHE_ROOT/$cache_name/snapshots"
  local snapshots=()
  local snapshot_path

  if [[ ! -d "$snapshot_root" ]]; then
    echo "[ERROR] Missing cached model: $snapshot_root" >&2
    echo "[ERROR] Cache the gated Meta checkpoint before submitting this job." >&2
    return 1
  fi
  if [[ -n "$requested_snapshot" ]]; then
    if [[ ! -d "$snapshot_root/$requested_snapshot" ]]; then
      echo "[ERROR] Requested snapshot is absent: $snapshot_root/$requested_snapshot" >&2
      return 1
    fi
    echo "$requested_snapshot"
    return 0
  fi

  while IFS= read -r snapshot_path; do
    snapshots+=("$(basename "$snapshot_path")")
  done < <(find "$snapshot_root" -mindepth 1 -maxdepth 1 -type d -print | sort)

  if (( ${#snapshots[@]} != 1 )); then
    echo "[ERROR] Expected exactly one cached snapshot under $snapshot_root; found ${#snapshots[@]}." >&2
    echo "[ERROR] Set the corresponding MODEL_SNAPSHOT_* variable explicitly." >&2
    if (( ${#snapshots[@]} > 0 )); then
      printf '  %s\n' "${snapshots[@]}" >&2
    fi
    return 1
  fi
  echo "${snapshots[0]}"
}

MODEL_SNAPSHOT_1B="$(resolve_snapshot "$MODEL_CACHE_1B" "$MODEL_SNAPSHOT_1B")"
MODEL_SNAPSHOT_3B="$(resolve_snapshot "$MODEL_CACHE_3B" "$MODEL_SNAPSHOT_3B")"
MODEL_SNAPSHOT_8B="$(resolve_snapshot "$MODEL_CACHE_8B" "$MODEL_SNAPSHOT_8B")"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm
export PYTHONPATH="$SFS_ROOT/src:${PYTHONPATH:-}"

RUN_STAMP="${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$EXPERIMENT_DIR/llama3_bucketed_all_models_${RUN_STAMP}"
JOB_ID="${SLURM_JOB_ID:-$$}"
JOB_LOCAL="/local/$USER/$JOB_ID"
mkdir_with_retry "$RUN_DIR"
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
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_LOGGING_LEVEL=WARNING
export EXPERIMENT_DIR

stage_model() {
  local cache_name="$1"
  local snapshot="$2"
  local src="$MODEL_CACHE_ROOT/$cache_name/snapshots/$snapshot"
  local dst="$JOB_LOCAL/$cache_name/snapshots/$snapshot"
  mkdir_with_retry "$dst"
  rsync -aL --delete "$src/" "$dst/"
  echo "$dst"
}

MODEL_DST_1B="$(stage_model "$MODEL_CACHE_1B" "$MODEL_SNAPSHOT_1B")"
MODEL_DST_3B="$(stage_model "$MODEL_CACHE_3B" "$MODEL_SNAPSHOT_3B")"
MODEL_DST_8B="$(stage_model "$MODEL_CACHE_8B" "$MODEL_SNAPSHOT_8B")"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2}"
if (( ${#GPU_IDS[@]} < 3 )); then
  echo "[ERROR] Need at least 3 visible GPUs; got ${CUDA_VISIBLE_DEVICES:-<unset>}." >&2
  exit 1
fi
GPU_1B="${GPU_IDS[0]}"
GPU_3B="${GPU_IDS[1]}"
GPU_8B="${GPU_IDS[2]}"

PORT_BASE=$((20000 + (JOB_ID % 15000) * 3))
PORT_1B="$PORT_BASE"
PORT_3B=$((PORT_BASE + 1))
PORT_8B=$((PORT_BASE + 2))

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
  local gpu_id="$4"
  local port="$5"
  local server_log="$RUN_DIR/vllm_${label}.log"
  local wait_log="$RUN_DIR/per_request_wait_${label}.log"
  local batch_stats="$RUN_DIR/batch_stats_${label}.csv"

  VLLM_PER_REQUEST_WAIT_LOG_PATH="$wait_log" \
  CUDA_VISIBLE_DEVICES="$gpu_id" \
    vllm serve --disable-uvicorn-access-log --model "$model_path" \
      --served-model-name "$served_model" \
      --dtype half \
      --max-model-len "$CONTEXT_LENGTH" \
      --no-enable-chunked-prefill \
      --no-enable-prefix-caching \
      --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --tensor-parallel-size 1 \
      --batch-stats-file "$batch_stats" \
      --port "$port" \
      --no-enable-wait-time-simulation \
      >"$server_log" 2>&1 &
  LAST_SERVER_PID=$!
  PIDS+=("$LAST_SERVER_PID")
  echo "[INFO] Started $label on GPU $gpu_id; PID=$LAST_SERVER_PID log=$server_log"
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

cd "$SFS_ROOT"
start_server llama3.2-1b "$MODEL_DST_1B" "$SERVED_MODEL_1B" "$GPU_1B" "$PORT_1B"
PID_1B="$LAST_SERVER_PID"
start_server llama3.2-3b "$MODEL_DST_3B" "$SERVED_MODEL_3B" "$GPU_3B" "$PORT_3B"
PID_3B="$LAST_SERVER_PID"
start_server llama3.1-8b "$MODEL_DST_8B" "$SERVED_MODEL_8B" "$GPU_8B" "$PORT_8B"
PID_8B="$LAST_SERVER_PID"

wait_for_server llama3.2-1b "$PORT_1B" "$PID_1B" "$RUN_DIR/vllm_llama3.2-1b.log"
wait_for_server llama3.2-3b "$PORT_3B" "$PID_3B" "$RUN_DIR/vllm_llama3.2-3b.log"
wait_for_server llama3.1-8b "$PORT_8B" "$PID_8B" "$RUN_DIR/vllm_llama3.1-8b.log"

{
  echo "slurm_job_id=${SLURM_JOB_ID:-local}"
  echo "sfs_commit=$(git -C "$SFS_ROOT" rev-parse HEAD)"
  echo "vllm_commit=$(git -C "$SFS_ROOT/vllm" rev-parse HEAD)"
  echo "model_1b=meta-llama/Llama-3.2-1B-Instruct@$MODEL_SNAPSHOT_1B"
  echo "model_3b=meta-llama/Llama-3.2-3B-Instruct@$MODEL_SNAPSHOT_3B"
  echo "model_8b=meta-llama/Llama-3.1-8B-Instruct@$MODEL_SNAPSHOT_8B"
  echo "served_model_1b=$SERVED_MODEL_1B"
  echo "served_model_3b=$SERVED_MODEL_3B"
  echo "served_model_8b=$SERVED_MODEL_8B"
  echo "dtype=half"
  echo "tensor_parallel_size=1"
  echo "context_length=$CONTEXT_LENGTH"
  echo "rope_scaling=none_model_native"
  echo "chat_template=model_tokenizer_config"
  echo "system_prompt=$SYSTEM_PROMPT"
  echo "chat_template_kwargs_json=$CHAT_TEMPLATE_KWARGS_JSON"
  echo "chunked_prefill=false"
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
  local output_root="$RUN_DIR/completions/$label"
  local run_log="$RUN_DIR/run_${label}.log"
  local request_log="$RUN_DIR/predicted_wait_${label}.log"
  mkdir_with_retry "$output_root"

  python -m scripts.prep.run_qwen_bucketed_prompts \
    --bucket-dir "$BUCKET_DIR" \
    --model-id "$served_model" \
    --instance-address "http://127.0.0.1:${port}" \
    --job-label "$label" \
    --instance-id "$instance_id" \
    --tokenizer-id "$model_path" \
    --output-root "$output_root" \
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
  run_model llama3.2-1b vllm-1b "$MODEL_DST_1B" "$SERVED_MODEL_1B" "$PORT_1B" &
  RUN_PID_1B=$!
  run_model llama3.2-3b vllm-3b "$MODEL_DST_3B" "$SERVED_MODEL_3B" "$PORT_3B" &
  RUN_PID_3B=$!
  run_model llama3.1-8b vllm-8b "$MODEL_DST_8B" "$SERVED_MODEL_8B" "$PORT_8B" &
  RUN_PID_8B=$!

  FAILED=0
  for pid in "$RUN_PID_1B" "$RUN_PID_3B" "$RUN_PID_8B"; do
    if ! wait "$pid"; then
      FAILED=1
    fi
  done
  if (( FAILED != 0 )); then
    echo "[ERROR] At least one Llama generation process failed." >&2
    exit 1
  fi
else
  run_model llama3.2-1b vllm-1b "$MODEL_DST_1B" "$SERVED_MODEL_1B" "$PORT_1B"
  run_model llama3.2-3b vllm-3b "$MODEL_DST_3B" "$SERVED_MODEL_3B" "$PORT_3B"
  run_model llama3.1-8b vllm-8b "$MODEL_DST_8B" "$SERVED_MODEL_8B" "$PORT_8B"
fi

echo "[DONE] Completed all Llama 3.x generation runs."
echo "[DONE] Run directory: $RUN_DIR"
find "$RUN_DIR/completions" -name run_summary.json -print | sort

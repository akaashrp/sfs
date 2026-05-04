#!/bin/bash
#SBATCH --job-name=qwen-bucketed-all-models
#SBATCH --output=experiments/qwen_bucketed_all_models_%j.out
#SBATCH --error=experiments/qwen_bucketed_all_models_%j.err
#SBATCH --time=04:30:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -p GPU-shared
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100-80:4
#SBATCH --mail-type=BEGIN,FAIL,END

set -euo pipefail

mkdir_with_retry() {
  local target="$1"
  local attempts="${2:-8}"
  local delay_s="${3:-2}"
  local i
  for i in $(seq 1 "$attempts"); do
    if mkdir -p "$target"; then
      return 0
    fi
    echo "[WARN] mkdir attempt ${i}/${attempts} failed for: $target" >&2
    sleep "$delay_s"
  done
  echo "[ERROR] mkdir failed after ${attempts} attempts: $target" >&2
  return 1
}

touch_with_retry() {
  local target="$1"
  local attempts="${2:-8}"
  local delay_s="${3:-2}"
  local i
  for i in $(seq 1 "$attempts"); do
    if : > "$target"; then
      return 0
    fi
    echo "[WARN] touch attempt ${i}/${attempts} failed for: $target" >&2
    sleep "$delay_s"
  done
  echo "[ERROR] touch failed after ${attempts} attempts: $target" >&2
  return 1
}

SBATCH_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFS_ROOT="$SBATCH_SCRIPT_DIR"
while [[ "$SFS_ROOT" != "/" && ! -d "$SFS_ROOT/src/slurm" ]]; do
  SFS_ROOT="$(dirname "$SFS_ROOT")"
done
if [[ "$SFS_ROOT" == "/" ]]; then
  echo "[ERROR] Could not resolve SFS repo root from script location: $SBATCH_SCRIPT_DIR" >&2
  exit 1
fi
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SFS_ROOT/.." && pwd)}"
EXPERIMENT_DIR=$SFS_ROOT/experiments
SCRIPT_DIR=$SFS_ROOT/src/scripts/prep
CHAT_TEMPLATE=$SFS_ROOT/src/assets/templates/chat_template_qwen3.jinja

# Input/output settings.
BUCKET_DIR="${BUCKET_DIR:-$SFS_ROOT/experiments/data/prompts/bucketed_prompts/qwen3-0.6b}"
MAX_PROMPTS_PER_BUCKET="${MAX_PROMPTS_PER_BUCKET:-2500}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-65536}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-8192}"
PROMPT_TOKEN_LIMIT="${PROMPT_TOKEN_LIMIT:-32768}"
TOKENIZER_ID="${TOKENIZER_ID:-Qwen/Qwen3-8B}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-69}"

# Scheduler settings for run_qwen_bucketed_prompts.py.
# For strictest parity with router sweeps use: WORKER_COUNT=1 MAX_QUEUE_SIZE=1.
WORKER_COUNT="${WORKER_COUNT:-16}"
MAX_QUEUE_SIZE="${MAX_QUEUE_SIZE:-0}"

# Run one prompt script per model in parallel (1) or sequentially (0).
RUN_IN_PARALLEL="${RUN_IN_PARALLEL:-1}"

RUN_STAMP="${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$EXPERIMENT_DIR/qwen_bucketed_all_models_${RUN_STAMP}"
mkdir_with_retry "$RUN_DIR"

if [[ ! -d "$BUCKET_DIR" ]]; then
  echo "Bucket directory does not exist: $BUCKET_DIR"
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm
export PYTHONPATH="$SFS_ROOT/src:${PYTHONPATH:-}"

PERSIST_VLLM=$PROJECT_ROOT/.cache/vllm
PERSIST_TRITON=$PROJECT_ROOT/.triton
PERSIST_NV=$PROJECT_ROOT/.nv
JOB_ID=${SLURM_JOB_ID:-$$}
PORT_BASE=$((20000 + (JOB_ID % 15000) * 3))
PORT_32B=$PORT_BASE
PORT_8B=$((PORT_BASE + 1))
PORT_06B=$((PORT_BASE + 2))
SCRATCH=/local/$USER/$JOB_ID

mkdir_with_retry "$SCRATCH/.cache/vllm"
mkdir_with_retry "$SCRATCH/.triton"
mkdir_with_retry "$SCRATCH/.nv"
rsync -aL --delete "$PERSIST_VLLM/" "$SCRATCH/.cache/vllm/" 2>/dev/null || true
rsync -aL --delete "$PERSIST_TRITON/" "$SCRATCH/.triton/" 2>/dev/null || true
rsync -aL --delete "$PERSIST_NV/" "$SCRATCH/.nv/" 2>/dev/null || true

export XDG_CACHE_HOME="$SCRATCH/.cache"
export TRITON_CACHE_DIR="$SCRATCH/.triton"
export CUDA_CACHE_PATH="$SCRATCH/.nv"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_LOGGING_LEVEL=WARNING
export EXPERIMENT_DIR

stage_model() {
  local model_id="$1"
  local snapshot="$2"
  local src="$PROJECT_ROOT/.cache/huggingface/hub/${model_id}/snapshots/${snapshot}"
  local dst="$SCRATCH/${model_id}/snapshots/${snapshot}"
  mkdir_with_retry "$dst"
  rsync -aL --delete "$src/" "$dst/"
  echo "$dst"
}

MODEL_DST_06B=$(stage_model "models--Qwen--Qwen3-0.6B" "c1899de289a04d12100db370d81485cdf75e47ca")
MODEL_DST_8B=$(stage_model "models--Qwen--Qwen3-8B" "b968826d9c46dd6066d109eabc6255188de91218")
MODEL_DST_32B=$(stage_model "models--Qwen--Qwen3-32B" "9216db5781bf21249d130ec9da846c4624c16137")

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if (( ${#GPU_IDS[@]} < 4 )); then
  echo "Need at least 4 visible GPUs, found: ${CUDA_VISIBLE_DEVICES:-<unset>}"
  exit 1
fi
GPU_32_A=${GPU_IDS[0]}
GPU_32_B=${GPU_IDS[1]}
GPU_8B=${GPU_IDS[2]}
GPU_06B=${GPU_IDS[3]}

echo "GPU mapping: 32B=${GPU_32_A},${GPU_32_B} 8B=${GPU_8B} 0.6B=${GPU_06B}"

VLLM_LOG_32B="$RUN_DIR/vllm_qwen3_32b.log"
VLLM_LOG_8B="$RUN_DIR/vllm_qwen3_8b.log"
VLLM_LOG_06B="$RUN_DIR/vllm_qwen3_0p6b.log"
BATCH_STATS_32B="$RUN_DIR/batch_stats_qwen3_32b.csv"
BATCH_STATS_8B="$RUN_DIR/batch_stats_qwen3_8b.csv"
BATCH_STATS_06B="$RUN_DIR/batch_stats_qwen3_0p6b.csv"
WAIT_LOG_32B="$RUN_DIR/per_request_wait_qwen3_32b.log"
WAIT_LOG_8B="$RUN_DIR/per_request_wait_qwen3_8b.log"
WAIT_LOG_06B="$RUN_DIR/per_request_wait_qwen3_0p6b.log"
RUN_LOG_32B="$RUN_DIR/run_qwen3-32b.log"
RUN_LOG_8B="$RUN_DIR/run_qwen3-8b.log"
RUN_LOG_06B="$RUN_DIR/run_qwen3-0.6b.log"
PRED_WAIT_LOG_32B="$RUN_DIR/predicted_wait_qwen3-32b.log"
PRED_WAIT_LOG_8B="$RUN_DIR/predicted_wait_qwen3-8b.log"
PRED_WAIT_LOG_06B="$RUN_DIR/predicted_wait_qwen3-0.6b.log"

mkdir_with_retry "$RUN_DIR/completions"
mkdir_with_retry "$RUN_DIR/completions/qwen3-32b"
mkdir_with_retry "$RUN_DIR/completions/qwen3-8b"
mkdir_with_retry "$RUN_DIR/completions/qwen3-0.6b"

touch_with_retry "$BATCH_STATS_32B"
touch_with_retry "$BATCH_STATS_8B"
touch_with_retry "$BATCH_STATS_06B"
touch_with_retry "$WAIT_LOG_32B"
touch_with_retry "$WAIT_LOG_8B"
touch_with_retry "$WAIT_LOG_06B"
touch_with_retry "$RUN_LOG_32B"
touch_with_retry "$RUN_LOG_8B"
touch_with_retry "$RUN_LOG_06B"
touch_with_retry "$PRED_WAIT_LOG_32B"
touch_with_retry "$PRED_WAIT_LOG_8B"
touch_with_retry "$PRED_WAIT_LOG_06B"

SERVER_PIDS=()
cleanup() {
  local pid
  for pid in "${SERVER_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${SERVER_PIDS[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

wait_for_server() {
  local name="$1"
  local port="$2"
  local pid="$3"
  local log_file="$4"
  local i
  for i in $(seq 1 600); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null; then
      echo "${name} ready on port ${port}"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "${name} exited before readiness. Check ${log_file}"
      return 1
    fi
    sleep 2
  done
  echo "Timed out waiting for ${name} on port ${port}. Check ${log_file}"
  return 1
}

cd "$SFS_ROOT"

VLLM_PER_REQUEST_WAIT_LOG_PATH="$WAIT_LOG_32B" \
CUDA_VISIBLE_DEVICES="${GPU_32_A},${GPU_32_B}" vllm serve --disable-uvicorn-access-log --model "$MODEL_DST_32B" \
  --max-model-len 131072 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
  --chat-template "$CHAT_TEMPLATE" \
  --enable-chunked-prefill \
  --batch-stats-file "$BATCH_STATS_32B" \
  --port "$PORT_32B" \
  --max-num-seqs 512 \
  --max-num-batched-tokens 2048 \
  --tensor-parallel-size 2 \
  --no-enable-wait-time-simulation \
  --no-enable-prefix-caching \
  >"$VLLM_LOG_32B" 2>&1 &
PID_32B=$!
SERVER_PIDS+=("$PID_32B")
echo "Started Qwen3-32B PID=$PID_32B log=$VLLM_LOG_32B"

VLLM_PER_REQUEST_WAIT_LOG_PATH="$WAIT_LOG_8B" \
CUDA_VISIBLE_DEVICES="${GPU_8B}" vllm serve --disable-uvicorn-access-log --model "$MODEL_DST_8B" \
  --max-model-len 131072 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
  --chat-template "$CHAT_TEMPLATE" \
  --enable-chunked-prefill \
  --batch-stats-file "$BATCH_STATS_8B" \
  --port "$PORT_8B" \
  --max-num-seqs 512 \
  --max-num-batched-tokens 2048 \
  --tensor-parallel-size 1 \
  --no-enable-wait-time-simulation \
  --no-enable-prefix-caching \
  >"$VLLM_LOG_8B" 2>&1 &
PID_8B=$!
SERVER_PIDS+=("$PID_8B")
echo "Started Qwen3-8B PID=$PID_8B log=$VLLM_LOG_8B"

VLLM_PER_REQUEST_WAIT_LOG_PATH="$WAIT_LOG_06B" \
CUDA_VISIBLE_DEVICES="${GPU_06B}" vllm serve --disable-uvicorn-access-log --model "$MODEL_DST_06B" \
  --max-model-len 131072 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
  --chat-template "$CHAT_TEMPLATE" \
  --enable-chunked-prefill \
  --batch-stats-file "$BATCH_STATS_06B" \
  --port "$PORT_06B" \
  --max-num-seqs 512 \
  --max-num-batched-tokens 2048 \
  --tensor-parallel-size 1 \
  --no-enable-wait-time-simulation \
  --no-enable-prefix-caching \
  >"$VLLM_LOG_06B" 2>&1 &
PID_06B=$!
SERVER_PIDS+=("$PID_06B")
echo "Started Qwen3-0.6B PID=$PID_06B log=$VLLM_LOG_06B"

wait_for_server "Qwen3-32B" "$PORT_32B" "$PID_32B" "$VLLM_LOG_32B"
wait_for_server "Qwen3-8B" "$PORT_8B" "$PID_8B" "$VLLM_LOG_8B"
wait_for_server "Qwen3-0.6B" "$PORT_06B" "$PID_06B" "$VLLM_LOG_06B"

run_model() {
  local label="$1"
  local instance_id="$2"
  local model_path="$3"
  local port="$4"
  local run_log="$RUN_DIR/run_${label}.log"
  local request_log="$RUN_DIR/predicted_wait_${label}.log"
  local output_root="$RUN_DIR/completions/${label}"
  mkdir_with_retry "$output_root"
  touch_with_retry "$run_log"
  touch_with_retry "$request_log"

  python -m scripts.prep.run_qwen_bucketed_prompts \
    --bucket-dir "$BUCKET_DIR" \
    --model-id "$model_path" \
    --instance-address "http://127.0.0.1:${port}" \
    --job-label "$label" \
    --instance-id "$instance_id" \
    --tokenizer-id "$TOKENIZER_ID" \
    --output-root "$output_root" \
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
  run_model "qwen3-32b" "vllm-32b" "$MODEL_DST_32B" "$PORT_32B" &
  RUN_PID_32B=$!
  run_model "qwen3-8b" "vllm-8b" "$MODEL_DST_8B" "$PORT_8B" &
  RUN_PID_8B=$!
  run_model "qwen3-0.6b" "vllm-0.6b" "$MODEL_DST_06B" "$PORT_06B" &
  RUN_PID_06B=$!

  FAILED=0
  for pid in "$RUN_PID_32B" "$RUN_PID_8B" "$RUN_PID_06B"; do
    if ! wait "$pid"; then
      FAILED=1
    fi
  done
  if (( FAILED != 0 )); then
    echo "At least one run_qwen_bucketed_prompts.py invocation failed."
    exit 1
  fi
else
  run_model "qwen3-32b" "vllm-32b" "$MODEL_DST_32B" "$PORT_32B"
  run_model "qwen3-8b" "vllm-8b" "$MODEL_DST_8B" "$PORT_8B"
  run_model "qwen3-0.6b" "vllm-0.6b" "$MODEL_DST_06B" "$PORT_06B"
fi

echo "Completed all model runs."
echo "Run dir: $RUN_DIR"
echo "Completion summaries:"
find "$RUN_DIR/completions" -name run_summary.json -print | sort

#!/bin/bash
#SBATCH --job-name=router-exp-lambda-sweep
#SBATCH --output=experiments/router_lambda_sweep_%j.out
#SBATCH --error=experiments/router_lambda_sweep_%j.err
#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -p GPU-shared
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100-80:4
#SBATCH --mail-type=BEGIN,FAIL,END

set -euo pipefail

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
CHAT_TEMPLATE=$SFS_ROOT/src/assets/templates/chat_template_qwen3.jinja
OUTPUT_PREDICTOR=$SFS_ROOT/src/assets/predictors/output_length_predictor
ACCURACY_PREDICTOR=$SFS_ROOT/src/assets/predictors/accuracy_predictor

# Lambda sweep configuration: fixed utility family, fixed QPS/delta, sweep lambda.
LAMBDA_SWEEP=(1e-4 2e-4 3e-4 4e-4 5e-4 6.5e-4 8e-4 1e-3 2e-3 3e-3 5e-3)
NUM_REQUESTS=16000
REQUEST_RATE_QPS=8
DELTA_WEIGHT=0

WORKER_COUNT=1
MAX_QUEUE_SIZE=1
MAX_COMPLETION_TOKENS=8192
TEMPERATURE=0

HOLDOUT_PROMPTS_PER_BUCKET=4000
HOLDOUT_BUCKET_DIR=$SFS_ROOT/experiments/data/prompts/bucketed_prompts/qwen3-0.6b
HOLDOUT_CACHE_DIR=$SFS_ROOT/experiments/data/prompts/holdout_cache_4000

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

mkdir -p "$SCRATCH/.cache/vllm" "$SCRATCH/.triton" "$SCRATCH/.nv"
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
export OUTPUT_LENGTH_MODEL_PATH="$OUTPUT_PREDICTOR"
export ACCURACY_MODEL_PATH="$ACCURACY_PREDICTOR"

stage_model() {
  local model_id="$1"
  local snapshot="$2"
  local src="$PROJECT_ROOT/.cache/huggingface/hub/${model_id}/snapshots/${snapshot}"
  local dst="$SCRATCH/${model_id}/snapshots/${snapshot}"
  mkdir -p "$dst"
  rsync -aL --delete "$src/" "$dst/"
  echo "$dst"
}

sanitize_tag() {
  local raw="$1"
  raw="${raw//+/}"
  raw="${raw//-/m}"
  raw="${raw//./p}"
  echo "$raw"
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

LOG_SUFFIX="${SLURM_JOB_ID:-local}_$(date +%F_%H%M%S)"
QPS_TAG=$(sanitize_tag "$REQUEST_RATE_QPS")
DELTA_TAG=$(sanitize_tag "$DELTA_WEIGHT")
SNAPSHOT_SHM_SIZE_BYTES=$((8 * 1024 * 1024))
SNAPSHOT_SHM_32B="vllm_snapshot_${JOB_ID}_qwen3_32b"
SNAPSHOT_SHM_8B="vllm_snapshot_${JOB_ID}_qwen3_8b"
SNAPSHOT_SHM_06B="vllm_snapshot_${JOB_ID}_qwen3_0p6b"
VLLM_LOG_06B="$EXPERIMENT_DIR/vllm_qwen3_0p6b_lambda_sweep_${LOG_SUFFIX}.log"
VLLM_LOG_8B="$EXPERIMENT_DIR/vllm_qwen3_8b_lambda_sweep_${LOG_SUFFIX}.log"
VLLM_LOG_32B="$EXPERIMENT_DIR/vllm_qwen3_32b_lambda_sweep_${LOG_SUFFIX}.log"
SWEEP_DRIVER_LOG="$EXPERIMENT_DIR/router_lambda_sweep_driver_${LOG_SUFFIX}.log"
PER_REQUEST_WAIT_LOG_06B="$EXPERIMENT_DIR/actual_wait_times_qwen3-0_6b_lambda_sweep_${LOG_SUFFIX}.log"
PER_REQUEST_WAIT_LOG_8B="$EXPERIMENT_DIR/actual_wait_times_qwen3-8b_lambda_sweep_${LOG_SUFFIX}.log"
PER_REQUEST_WAIT_LOG_32B="$EXPERIMENT_DIR/actual_wait_times_qwen3-32b_lambda_sweep_${LOG_SUFFIX}.log"
BATCH_STATS_06B="$EXPERIMENT_DIR/batch_stats_qwen3-0_6b_lambda_sweep_${LOG_SUFFIX}.csv"
BATCH_STATS_8B="$EXPERIMENT_DIR/batch_stats_qwen3-8b_lambda_sweep_${LOG_SUFFIX}.csv"
BATCH_STATS_32B="$EXPERIMENT_DIR/batch_stats_qwen3-32b_lambda_sweep_${LOG_SUFFIX}.csv"
INSTANCES_CONFIG="$EXPERIMENT_DIR/router_instances_lambda_sweep_${LOG_SUFFIX}.json"
SWEEP_OUTPUT_DIR="$EXPERIMENT_DIR/router_lambda_sweep_${LOG_SUFFIX}"
SWEEP_OUTPUT_PREFIX="router_lambda_qwen3_qps${QPS_TAG}_delta${DELTA_TAG}"

cat >"$INSTANCES_CONFIG" <<EOF
{
  "instances": [
    {
      "instance_id": "vllm-32b",
      "address": "http://localhost:${PORT_32B}",
      "default_model": "$MODEL_DST_32B",
      "model_id": "qwen3-32b",
      "snapshot_shm_name": "$SNAPSHOT_SHM_32B",
      "snapshot_shm_size_bytes": $SNAPSHOT_SHM_SIZE_BYTES
    },
    {
      "instance_id": "vllm-8b",
      "address": "http://localhost:${PORT_8B}",
      "default_model": "$MODEL_DST_8B",
      "model_id": "qwen3-8b",
      "snapshot_shm_name": "$SNAPSHOT_SHM_8B",
      "snapshot_shm_size_bytes": $SNAPSHOT_SHM_SIZE_BYTES
    },
    {
      "instance_id": "vllm-0.6b",
      "address": "http://localhost:${PORT_06B}",
      "default_model": "$MODEL_DST_06B",
      "model_id": "qwen3-0.6b",
      "snapshot_shm_name": "$SNAPSHOT_SHM_06B",
      "snapshot_shm_size_bytes": $SNAPSHOT_SHM_SIZE_BYTES
    }
  ]
}
EOF

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

VLLM_PER_REQUEST_WAIT_LOG_PATH="$PER_REQUEST_WAIT_LOG_32B" \
CUDA_VISIBLE_DEVICES="${GPU_32_A},${GPU_32_B}" vllm serve --disable-uvicorn-access-log --model "$MODEL_DST_32B" \
  --max-model-len 131072 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
  --chat-template "$CHAT_TEMPLATE" \
  --enable-chunked-prefill \
  --batch-stats-file "$BATCH_STATS_32B" \
  --port "$PORT_32B" \
  --max-num-seqs 512 \
  --max-num-batched-tokens 32768 \
  --tensor-parallel-size 2 \
  --no-enable-wait-time-simulation \
  --enable-snapshot-shm-publishing \
  --snapshot-shm-name "$SNAPSHOT_SHM_32B" \
  --snapshot-shm-size-bytes "$SNAPSHOT_SHM_SIZE_BYTES" \
  --snapshot-shm-publish-interval-ms 0 \
  --simulation-intercept 0.014347877296187458 \
  --simulation-prefill-coeff 5.294185118159464e-05 \
  --simulation-decode-coeff 6.304876078328351e-05 \
  --simulation-sum-coeff 3.6863941393981496e-08 \
  --simulation-prefill-sq-coeff 8.774059443201359e-10 \
  --simulation-sum-sq-coeff 4.519593317761812e-13 \
  --output-length-model-path "$OUTPUT_PREDICTOR" \
  --no-enable-prefix-caching \
  >"$VLLM_LOG_32B" 2>&1 &
PID_32B=$!
PIDS+=("$PID_32B")

echo "Started Qwen3-32B PID=$PID_32B log=$VLLM_LOG_32B"

VLLM_PER_REQUEST_WAIT_LOG_PATH="$PER_REQUEST_WAIT_LOG_8B" \
CUDA_VISIBLE_DEVICES="${GPU_8B}" vllm serve --disable-uvicorn-access-log --model "$MODEL_DST_8B" \
  --max-model-len 131072 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
  --chat-template "$CHAT_TEMPLATE" \
  --enable-chunked-prefill \
  --batch-stats-file "$BATCH_STATS_8B" \
  --port "$PORT_8B" \
  --max-num-seqs 512 \
  --max-num-batched-tokens 32768 \
  --tensor-parallel-size 1 \
  --no-enable-wait-time-simulation \
  --enable-snapshot-shm-publishing \
  --snapshot-shm-name "$SNAPSHOT_SHM_8B" \
  --snapshot-shm-size-bytes "$SNAPSHOT_SHM_SIZE_BYTES" \
  --snapshot-shm-publish-interval-ms 0 \
  --simulation-intercept 0.007295929680901453 \
  --simulation-prefill-coeff 2.0789009922784725e-05 \
  --simulation-decode-coeff 2.5632010897991973e-05 \
  --simulation-sum-coeff 4.6490885513702363e-08 \
  --simulation-prefill-sq-coeff 5.119559033569263e-10 \
  --simulation-sum-sq-coeff 2.6851287984524727e-13 \
  --output-length-model-path "$OUTPUT_PREDICTOR" \
  --no-enable-prefix-caching \
  >"$VLLM_LOG_8B" 2>&1 &
PID_8B=$!
PIDS+=("$PID_8B")

echo "Started Qwen3-8B PID=$PID_8B log=$VLLM_LOG_8B"

VLLM_PER_REQUEST_WAIT_LOG_PATH="$PER_REQUEST_WAIT_LOG_06B" \
CUDA_VISIBLE_DEVICES="${GPU_06B}" vllm serve --disable-uvicorn-access-log --model "$MODEL_DST_06B" \
  --max-model-len 131072 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
  --chat-template "$CHAT_TEMPLATE" \
  --enable-chunked-prefill \
  --batch-stats-file "$BATCH_STATS_06B" \
  --port "$PORT_06B" \
  --max-num-seqs 512 \
  --max-num-batched-tokens 32768 \
  --tensor-parallel-size 1 \
  --no-enable-wait-time-simulation \
  --enable-snapshot-shm-publishing \
  --snapshot-shm-name "$SNAPSHOT_SHM_06B" \
  --snapshot-shm-size-bytes "$SNAPSHOT_SHM_SIZE_BYTES" \
  --snapshot-shm-publish-interval-ms 0 \
  --simulation-intercept 0.002582076153059547 \
  --simulation-prefill-coeff 2.0335711505545034e-06 \
  --simulation-decode-coeff 1.6889500804006565e-05 \
  --simulation-sum-coeff 3.5429418765562916e-08 \
  --simulation-prefill-sq-coeff 1.8925400079684374e-10 \
  --simulation-sum-sq-coeff 1.5754738957656986e-13 \
  --output-length-model-path "$OUTPUT_PREDICTOR" \
  --no-enable-prefix-caching \
  >"$VLLM_LOG_06B" 2>&1 &
PID_06B=$!
PIDS+=("$PID_06B")

echo "Started Qwen3-0.6B PID=$PID_06B log=$VLLM_LOG_06B"

wait_for_server "Qwen3-32B" "$PORT_32B" "$PID_32B" "$VLLM_LOG_32B"
wait_for_server "Qwen3-8B" "$PORT_8B" "$PID_8B" "$VLLM_LOG_8B"
wait_for_server "Qwen3-0.6B" "$PORT_06B" "$PID_06B" "$VLLM_LOG_06B"

if (( ${#LAMBDA_SWEEP[@]} == 0 )); then
  echo "LAMBDA_SWEEP is empty; nothing to run."
  exit 1
fi

mkdir -p "$SWEEP_OUTPUT_DIR"
echo "Starting lambda sweep with qps=${REQUEST_RATE_QPS}, delta=${DELTA_WEIGHT}, lambdas=${LAMBDA_SWEEP[*]}"

python -m scripts.runs.experiments_sweep \
  --sweep lambda \
  --lambda-values "${LAMBDA_SWEEP[@]}" \
  --lambda-utilities hard shortest_queue \
  --output-dir "$SWEEP_OUTPUT_DIR" \
  --output-prefix "$SWEEP_OUTPUT_PREFIX" \
  --experiment router \
  --instances-config "$INSTANCES_CONFIG" \
  --request-rate-qps "$REQUEST_RATE_QPS" \
  --arrival-process poisson \
  --num-requests "$NUM_REQUESTS" \
  --worker-count "$WORKER_COUNT" \
  --max-queue-size "$MAX_QUEUE_SIZE" \
  --max-completion-tokens "$MAX_COMPLETION_TOKENS" \
  --temperature "$TEMPERATURE" \
  --queue-slo-min-ms 1 \
  --queue-slo-max-ms 5 \
  --ttft-slo-min-ms 150 \
  --ttft-slo-max-ms 1120 \
  --ttft-slo-base-ms 155 \
  --ttft-slo-per-prompt-token-ms 0.035 \
  --ttft-slo-jitter-min 0.98 \
  --ttft-slo-jitter-max 1.02 \
  --feasible-slo-mode ttft \
  --delta-weight "$DELTA_WEIGHT" \
  --holdout-prompts-per-bucket "$HOLDOUT_PROMPTS_PER_BUCKET" \
  --holdout-bucket-dir "$HOLDOUT_BUCKET_DIR" \
  --holdout-cache-dir "$HOLDOUT_CACHE_DIR" \
  --holdout-start-index 2500 \
  --holdout-context-length 65536 \
  --tokenizer-id Qwen/Qwen3-8B \
  --accuracy-model-path "$ACCURACY_PREDICTOR" \
  --output-length-model-path "$OUTPUT_PREDICTOR" \
  --per-request-wait-log "$PER_REQUEST_WAIT_LOG_32B" \
  --per-request-wait-log "$PER_REQUEST_WAIT_LOG_8B" \
  --per-request-wait-log "$PER_REQUEST_WAIT_LOG_06B" \
  >"$SWEEP_DRIVER_LOG" 2>&1

echo "Lambda sweep complete. output_dir=$SWEEP_OUTPUT_DIR driver_log=$SWEEP_DRIVER_LOG"

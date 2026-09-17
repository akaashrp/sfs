#!/bin/bash
# Calibration extension, Qwen3 family: greedy completions for raw bucket rows
# 6500-8999 (2,500 per bucket, 10,000 per model), disjoint from the original
# calibration rows 0-2499 and the evaluation holdout rows 2500-6499.
# Serving arguments follow scripts.runs.qwen_baselines.server_argv (the
# campaign pool) minus the SFS scheduler-only snapshot/predictor flags; the
# client is the same scripts.prep.run_qwen_bucketed_prompts pipeline that
# produced the original per-bucket outputs.
#SBATCH --job-name=calib-ext-qwen
#SBATCH --output=/ocean/projects/cis250162p/aparthas/sfs_artifacts/calibration_extension_20260917/slurm/qwen_%j.out
#SBATCH --error=/ocean/projects/cis250162p/aparthas/sfs_artifacts/calibration_extension_20260917/slurm/qwen_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=GPU-shared
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100-80:4

set -euo pipefail

mkdir_with_retry() {
  local target="$1" attempt
  for attempt in 1 2 3 4 5 6 7 8; do
    if mkdir -p "$target"; then return 0; fi
    echo "[WARN] mkdir attempt ${attempt}/8 failed for: $target" >&2
    sleep 2
  done
  echo "[ERROR] mkdir failed for: $target" >&2
  return 1
}

SFS_ROOT=/ocean/projects/cis250162p/aparthas/sfs_calib_ext_20260917
PROJECT_ROOT=/ocean/projects/cis250162p/aparthas
MODEL_CACHE_ROOT="$PROJECT_ROOT/.cache/huggingface/hub"
BUNDLE_DIR="$PROJECT_ROOT/sfs_cloud_artifacts_20260916"
OUTPUT_ROOT="$PROJECT_ROOT/sfs_artifacts/calibration_extension_20260917/qwen"
RUN_DIR="$OUTPUT_ROOT/run"
BUCKET_DIR="$PROJECT_ROOT/vllm_utils/bucketed_prompts_0.6B"
CHAT_TEMPLATE="$SFS_ROOT/src/assets/templates/chat_template_qwen3.jinja"
TOKENIZER_ID="$BUNDLE_DIR/tokenizers/qwen3-8b"

# Pinned model revisions (bundle.json models).
MODEL_CACHE_06B="models--Qwen--Qwen3-0.6B"; MODEL_SNAPSHOT_06B="c1899de289a04d12100db370d81485cdf75e47ca"
MODEL_CACHE_8B="models--Qwen--Qwen3-8B";    MODEL_SNAPSHOT_8B="b968826d9c46dd6066d109eabc6255188de91218"
MODEL_CACHE_32B="models--Qwen--Qwen3-32B";  MODEL_SNAPSHOT_32B="9216db5781bf21249d130ec9da846c4624c16137"
SERVED_06B=qwen3-0.6b; SERVED_8B=qwen3-8b; SERVED_32B=qwen3-32b
TP_06B=1; TP_8B=1; TP_32B=2

# Campaign serving profile (bundle.json families.qwen.profile / qwen_baselines.server_argv).
MAX_MODEL_LEN=131072
ROPE_SCALING='{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'
MAX_NUM_BATCHED_TOKENS=32768
MAX_NUM_SEQS=512
GPU_MEMORY_UTILIZATION=0.9

# Campaign request settings (bundle.json families.qwen.experiment_argv).
SYSTEM_PROMPT="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
CHAT_TEMPLATE_KWARGS_JSON='{"enable_thinking":false}'
CONTEXT_LENGTH=65536
MAX_COMPLETION_TOKENS=8192
PROMPT_TOKEN_LIMIT=32768
TEMPERATURE=0.0
TOP_P=1.0
SEED=69

# Index range and client concurrency.
PROMPT_START_INDEX=6500
MAX_PROMPTS_PER_BUCKET=2500
WORKER_COUNT=128
MAX_QUEUE_SIZE=0

if [[ -e "$RUN_DIR" ]]; then
  echo "[ERROR] Refusing to overwrite existing run directory: $RUN_DIR" >&2
  exit 1
fi
for required in "$BUCKET_DIR" "$CHAT_TEMPLATE" "$TOKENIZER_ID/tokenizer.json" "$BUNDLE_DIR/bundle.json"; do
  if [[ ! -e "$required" ]]; then
    echo "[ERROR] Missing required input: $required" >&2
    exit 1
  fi
done

resolve_model_path() {
  local snapshot_path="$MODEL_CACHE_ROOT/$1/snapshots/$2"
  if [[ ! -f "$snapshot_path/config.json" ]] || ! ls "$snapshot_path"/*.safetensors >/dev/null 2>&1; then
    echo "[ERROR] Missing or incomplete pinned model snapshot: $snapshot_path" >&2
    return 1
  fi
  echo "$snapshot_path"
}
MODEL_SOURCE_06B="$(resolve_model_path "$MODEL_CACHE_06B" "$MODEL_SNAPSHOT_06B")"
MODEL_SOURCE_8B="$(resolve_model_path "$MODEL_CACHE_8B" "$MODEL_SNAPSHOT_8B")"
MODEL_SOURCE_32B="$(resolve_model_path "$MODEL_CACHE_32B" "$MODEL_SNAPSHOT_32B")"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm
# Pinned fork source (vllm submodule at the campaign commit) ahead of the editable install.
export PYTHONPATH="$SFS_ROOT/vllm:$SFS_ROOT/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

EXPECTED_VLLM_COMMIT="$(git -C "$SFS_ROOT/vllm" rev-parse HEAD)"
ACTIVE_VLLM_ROOT="$(cd "$SFS_ROOT/src" && python - <<'PY'
from pathlib import Path
import vllm

print(Path(vllm.__file__).resolve().parent.parent)
PY
)"
ACTIVE_VLLM_COMMIT="$(git -C "$ACTIVE_VLLM_ROOT" rev-parse HEAD)"
if [[ "$ACTIVE_VLLM_COMMIT" != "$EXPECTED_VLLM_COMMIT" ]]; then
  echo "[ERROR] Active vLLM commit $ACTIVE_VLLM_COMMIT does not match pinned $EXPECTED_VLLM_COMMIT." >&2
  exit 1
fi

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
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_LOGGING_LEVEL=WARNING

stage_model() {
  local dst="$JOB_LOCAL/$1/snapshots/$2"
  mkdir_with_retry "$dst"
  rsync -aL --delete "$3/" "$dst/"
  echo "$dst"
}
MODEL_DST_06B="$(stage_model "$MODEL_CACHE_06B" "$MODEL_SNAPSHOT_06B" "$MODEL_SOURCE_06B")"
MODEL_DST_8B="$(stage_model "$MODEL_CACHE_8B" "$MODEL_SNAPSHOT_8B" "$MODEL_SOURCE_8B")"
MODEL_DST_32B="$(stage_model "$MODEL_CACHE_32B" "$MODEL_SNAPSHOT_32B" "$MODEL_SOURCE_32B")"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if (( ${#GPU_IDS[@]} < 4 )); then
  echo "[ERROR] Need 4 visible GPUs; got ${CUDA_VISIBLE_DEVICES:-<unset>}." >&2
  exit 1
fi
GPU_32B="${GPU_IDS[0]},${GPU_IDS[1]}"
GPU_8B="${GPU_IDS[2]}"
GPU_06B="${GPU_IDS[3]}"

PORT_BASE=$((20000 + (JOB_ID % 15000) * 3))
PORT_32B="$PORT_BASE"
PORT_8B=$((PORT_BASE + 1))
PORT_06B=$((PORT_BASE + 2))

PIDS=()
cleanup() {
  local pid
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]:-}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

start_server() {
  local label="$1" model_path="$2" served="$3" visible="$4" tp="$5" port="$6"
  VLLM_PER_REQUEST_WAIT_LOG_PATH="$RUN_DIR/per_request_wait_${label}.log" \
  CUDA_VISIBLE_DEVICES="$visible" \
    python -m vllm.entrypoints.openai.api_server \
      --model "$model_path" \
      --served-model-name "$served" \
      --disable-uvicorn-access-log \
      --dtype auto \
      --max-model-len "$MAX_MODEL_LEN" \
      --rope-scaling "$ROPE_SCALING" \
      --chat-template "$CHAT_TEMPLATE" \
      --enable-chunked-prefill \
      --no-enable-prefix-caching \
      --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --tensor-parallel-size "$tp" \
      --batch-stats-file "$RUN_DIR/batch_stats_${label}.csv" \
      --port "$port" \
      --no-enable-wait-time-simulation \
      --host 127.0.0.1 \
      >"$RUN_DIR/vllm_${label}.log" 2>&1 &
  LAST_SERVER_PID=$!
  PIDS+=("$LAST_SERVER_PID")
  echo "[INFO] Started $label on GPU(s) $visible TP=$tp port=$port PID=$LAST_SERVER_PID"
}

wait_for_server() {
  local label="$1" port="$2" pid="$3" attempt
  for attempt in $(seq 1 600); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null; then
      echo "[INFO] $label ready on port $port"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[ERROR] $label exited before readiness; see $RUN_DIR/vllm_${label}.log" >&2
      return 1
    fi
    sleep 2
  done
  echo "[ERROR] Timed out waiting for $label" >&2
  return 1
}

cd "$SFS_ROOT/src"
start_server "$SERVED_32B" "$MODEL_DST_32B" "$SERVED_32B" "$GPU_32B" "$TP_32B" "$PORT_32B"; PID_32B="$LAST_SERVER_PID"
start_server "$SERVED_8B" "$MODEL_DST_8B" "$SERVED_8B" "$GPU_8B" "$TP_8B" "$PORT_8B"; PID_8B="$LAST_SERVER_PID"
start_server "$SERVED_06B" "$MODEL_DST_06B" "$SERVED_06B" "$GPU_06B" "$TP_06B" "$PORT_06B"; PID_06B="$LAST_SERVER_PID"
wait_for_server "$SERVED_32B" "$PORT_32B" "$PID_32B"
wait_for_server "$SERVED_8B" "$PORT_8B" "$PID_8B"
wait_for_server "$SERVED_06B" "$PORT_06B" "$PID_06B"

nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader >"$RUN_DIR/gpus.csv" 2>/dev/null || true
{
  echo "slurm_job_id=${SLURM_JOB_ID:-local}"
  echo "workload=generation"
  echo "sfs_root=$SFS_ROOT"
  echo "sfs_commit=$(git -C "$SFS_ROOT" rev-parse HEAD)"
  echo "recorded_vllm_commit=$EXPECTED_VLLM_COMMIT"
  echo "active_vllm_root=$ACTIVE_VLLM_ROOT"
  echo "active_vllm_commit=$ACTIVE_VLLM_COMMIT"
  echo "gpu_profile=h100-80"
  echo "model_0.6b=Qwen/Qwen3-0.6B@$MODEL_SNAPSHOT_06B"
  echo "model_8b=Qwen/Qwen3-8B@$MODEL_SNAPSHOT_8B"
  echo "model_32b=Qwen/Qwen3-32B@$MODEL_SNAPSHOT_32B"
  echo "served_model_0.6b=$SERVED_06B"
  echo "served_model_8b=$SERVED_8B"
  echo "served_model_32b=$SERVED_32B"
  echo "gpu_mapping_0.6b=$GPU_06B"
  echo "gpu_mapping_8b=$GPU_8B"
  echo "gpu_mapping_32b=$GPU_32B"
  echo "tensor_parallel_0.6b=$TP_06B"
  echo "tensor_parallel_8b=$TP_8B"
  echo "tensor_parallel_32b=$TP_32B"
  echo "dtype=auto"
  echo "attention_backend=$VLLM_ATTENTION_BACKEND"
  echo "tokenizer_mode=auto"
  echo "client_tokenizer_id=$TOKENIZER_ID"
  echo "chat_template=$CHAT_TEMPLATE"
  echo "max_model_len=$MAX_MODEL_LEN"
  echo "rope_scaling=$ROPE_SCALING"
  echo "context_length=$CONTEXT_LENGTH"
  echo "system_prompt=$SYSTEM_PROMPT"
  echo "chat_template_kwargs_json=$CHAT_TEMPLATE_KWARGS_JSON"
  echo "chunked_prefill=1"
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
  echo "prompt_start_index=$PROMPT_START_INDEX"
  echo "max_prompts_per_bucket=$MAX_PROMPTS_PER_BUCKET"
  echo "worker_count=$WORKER_COUNT"
} >"$RUN_DIR/provenance.txt"

run_model() {
  local label="$1" instance_id="$2" served="$3" port="$4"
  python -m scripts.prep.run_qwen_bucketed_prompts \
    --bucket-dir "$BUCKET_DIR" \
    --model-id "$served" \
    --instance-address "http://127.0.0.1:${port}" \
    --job-label "$label" \
    --instance-id "$instance_id" \
    --tokenizer-id "$TOKENIZER_ID" \
    --tokenizer-mode auto \
    --run-dir "$RUN_DIR/completions" \
    --system-prompt "$SYSTEM_PROMPT" \
    --chat-template-kwargs-json "$CHAT_TEMPLATE_KWARGS_JSON" \
    --context-length "$CONTEXT_LENGTH" \
    --max-completion-tokens "$MAX_COMPLETION_TOKENS" \
    --prompt-token-limit "$PROMPT_TOKEN_LIMIT" \
    --max-prompts-per-bucket "$MAX_PROMPTS_PER_BUCKET" \
    --prompt-start-index "$PROMPT_START_INDEX" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --seed "$SEED" \
    --worker-count "$WORKER_COUNT" \
    --max-queue-size "$MAX_QUEUE_SIZE" \
    --disable-wait-time-polling \
    --request-log-path "$RUN_DIR/predicted_wait_${label}.log" \
    >"$RUN_DIR/run_${label}.log" 2>&1
}

run_model "$SERVED_32B" vllm-32b "$SERVED_32B" "$PORT_32B" & RUN_PID_32B=$!
run_model "$SERVED_8B" vllm-8b "$SERVED_8B" "$PORT_8B" & RUN_PID_8B=$!
run_model "$SERVED_06B" vllm-0.6b "$SERVED_06B" "$PORT_06B" & RUN_PID_06B=$!
DRIVER_STATUS=0
for pid in "$RUN_PID_32B" "$RUN_PID_8B" "$RUN_PID_06B"; do
  if ! wait "$pid"; then DRIVER_STATUS=1; fi
done
echo "[INFO] generation exit status: $DRIVER_STATUS"

AUDIT_STATUS=0
python -m scripts.prep.audit_generation_outputs \
  --outputs-root "$RUN_DIR/completions" \
  --models "$SERVED_06B" "$SERVED_8B" "$SERVED_32B" \
  --expected-per-bucket "$MAX_PROMPTS_PER_BUCKET" \
  --output-path "$RUN_DIR/generation_audit.json" || AUDIT_STATUS=$?
echo "[INFO] audit exit status: $AUDIT_STATUS"

SETTINGS_JSON=$(cat <<JSON
{"dtype": "auto (BF16 checkpoints)", "attention_backend": "FLASH_ATTN", "tokenizer_mode": "auto",
 "chat_template": "src/assets/templates/chat_template_qwen3.jinja", "max_model_len": 131072,
 "rope_scaling": $ROPE_SCALING, "context_length": 65536, "prompt_token_limit": 32768,
 "max_completion_tokens": 8192, "chunked_prefill": true, "max_num_batched_tokens": 32768, "max_num_seqs": 512,
 "prefix_caching": false, "gpu_memory_utilization": 0.9, "tensor_parallel_sizes": [1, 1, 2],
 "system_prompt": "$SYSTEM_PROMPT", "chat_template_kwargs": $CHAT_TEMPLATE_KWARGS_JSON,
 "temperature": 0.0, "top_p": 1.0, "seed": 69, "worker_count": 128, "max_queue_size": 0,
 "client_tokenizer": "$TOKENIZER_ID",
 "server_entrypoint": "python -m vllm.entrypoints.openai.api_server (qwen_baselines.server_argv minus SFS snapshot/predictor flags)",
 "original_calibration_rows": "0-2499 (vllm_utils/bucketed_prompt_outputs/qwen3-*/outputs, March 2026, temperature 0.2)",
 "evaluation_holdout_rows": "2500-6499 (bundle qwen/holdout)"}
JSON
)
MANIFEST_STATUS=0
python -m scripts.prep.calibration_extension_manifest \
  --family qwen \
  --output-root "$OUTPUT_ROOT" \
  --run-dir "$RUN_DIR" \
  --bundle "$BUNDLE_DIR/bundle.json" \
  --sfs-root "$SFS_ROOT" \
  --bucket-dir "$BUCKET_DIR" \
  --start-index "$PROMPT_START_INDEX" \
  --count "$MAX_PROMPTS_PER_BUCKET" \
  --models qwen3-0.6b qwen3-8b qwen3-32b \
  --served-names "$SERVED_06B" "$SERVED_8B" "$SERVED_32B" \
  --tensor-parallel 1 1 2 \
  --settings-json "$SETTINGS_JSON" \
  --code-files "$SFS_ROOT/src/scripts/prep/run_qwen_bucketed_prompts.py" \
    "$SFS_ROOT/src/scripts/prep/calibration_extension_manifest.py" \
    "$SFS_ROOT/src/slurm/prep/calibration_extension_qwen.sbatch" \
  --slurm-job-id "${SLURM_JOB_ID:-local}" \
  --driver-exit-status "$DRIVER_STATUS" \
  --purpose "Additional greedy calibration outputs (rows 6500-8999 per bucket) for SFS output-length predictor mean-head retraining per prompt-length band" \
  || MANIFEST_STATUS=$?
echo "[INFO] manifest exit status: $MANIFEST_STATUS"
if (( DRIVER_STATUS != 0 || AUDIT_STATUS != 0 || MANIFEST_STATUS != 0 )); then
  exit 1
fi
echo "[DONE] Qwen calibration extension complete: $OUTPUT_ROOT"

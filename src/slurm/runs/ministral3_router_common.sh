#!/bin/bash
# Shared server setup for Ministral 3 router sweeps. Source this file from a
# Slurm wrapper, set MINISTRAL_RUN_ROOT, then call ministral3_start_router_pool.

ministral3_resolve_sfs_root() {
  if [[ -n "${SFS_ROOT:-}" && -d "$SFS_ROOT/src/slurm" ]]; then
    SFS_ROOT="$(cd "$SFS_ROOT" && pwd)"
  elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/src/slurm" ]]; then
    SFS_ROOT="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
  else
    local common_dir
    common_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SFS_ROOT="$common_dir"
    while [[ "$SFS_ROOT" != "/" && ! -d "$SFS_ROOT/src/slurm" ]]; do
      SFS_ROOT="$(dirname "$SFS_ROOT")"
    done
    if [[ "$SFS_ROOT" == "/" ]]; then
      echo "[ERROR] Could not resolve SFS repo root." >&2
      return 1
    fi
  fi
  export SFS_ROOT
  MINISTRAL_PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SFS_ROOT/.." && pwd)}"
}

ministral3_resolve_service_run() {
  MINISTRAL_SERVICE_RUN_DIR="${SERVICE_RUN_DIR:-}"
  local service_job_id="${SERVICE_JOB_ID:-}"
  if [[ -z "$MINISTRAL_SERVICE_RUN_DIR" && -z "$service_job_id" ]]; then
    echo "[ERROR] Set SERVICE_RUN_DIR or SERVICE_JOB_ID." >&2
    return 1
  fi
  if [[ -z "$MINISTRAL_SERVICE_RUN_DIR" ]]; then
    shopt -s nullglob
    local matches=("$SFS_ROOT"/experiments/ministral3_h100-80_service_metrics_"$service_job_id"_*)
    shopt -u nullglob
    local directories=()
    local path
    for path in "${matches[@]}"; do
      [[ -d "$path" ]] && directories+=("$path")
    done
    if (( ${#directories[@]} != 1 )); then
      echo "[ERROR] Expected one service directory for job $service_job_id; found ${#directories[@]}." >&2
      return 1
    fi
    MINISTRAL_SERVICE_RUN_DIR="${directories[0]}"
  fi
  MINISTRAL_SERVICE_RUN_DIR="$(cd "$MINISTRAL_SERVICE_RUN_DIR" && pwd)"
  MINISTRAL_SERVICE_METRICS="$MINISTRAL_SERVICE_RUN_DIR/model_metrics.json"
}

ministral3_verify_calibration_artifacts() {
  local predictor_run_dir="${PREDICTOR_RUN_DIR:-}"
  if [[ -z "$predictor_run_dir" ]]; then
    echo "[ERROR] Set PREDICTOR_RUN_DIR to an audited predictor run." >&2
    return 1
  fi
  MINISTRAL_PREDICTOR_RUN_DIR="$(cd "$predictor_run_dir" && pwd)"
  MINISTRAL_OUTPUT_PREDICTOR="$MINISTRAL_PREDICTOR_RUN_DIR/output_length_predictor"
  MINISTRAL_ACCURACY_PREDICTOR="$MINISTRAL_PREDICTOR_RUN_DIR/accuracy_predictor"
  MINISTRAL_TOKENIZER_PATH="$MINISTRAL_PROJECT_ROOT/.cache/huggingface/hub/models--mistralai--Ministral-3-8B-Instruct-2512-BF16/snapshots/f6fae9795746f63c9be8344932f01275f3c63734"
  MINISTRAL_HOLDOUT_SOURCE_DIR="$MINISTRAL_PROJECT_ROOT/vllm_utils/bucketed_prompts_0.6B"

  local required_path
  for required_path in \
    "$MINISTRAL_SERVICE_METRICS" \
    "$MINISTRAL_OUTPUT_PREDICTOR/metadata.json" \
    "$MINISTRAL_OUTPUT_PREDICTOR/mean_model.txt" \
    "$MINISTRAL_ACCURACY_PREDICTOR/metadata.json" \
    "$MINISTRAL_ACCURACY_PREDICTOR/accuracy_model.txt" \
    "$MINISTRAL_TOKENIZER_PATH/tekken.json"; do
    if [[ ! -s "$required_path" ]]; then
      echo "[ERROR] Missing required artifact: $required_path" >&2
      return 1
    fi
  done

  local model_keys=(ministral3-3b ministral3-8b ministral3-14b)
  MINISTRAL_MODEL_KEY_ARGS=()
  local model
  for model in "${model_keys[@]}"; do
    MINISTRAL_MODEL_KEY_ARGS+=(--model-key "$model")
  done
  python -m scripts.runs.service_metrics_config \
    --path "$MINISTRAL_SERVICE_METRICS" \
    --expected-feature-set cross_term \
    --require-nonnegative-sfs \
    "${MINISTRAL_MODEL_KEY_ARGS[@]}" \
    validate >"$MINISTRAL_RUN_ROOT/service_metrics_validation.json"
  if [[ -s "$MINISTRAL_SERVICE_METRICS.sha256" ]]; then
    (cd "$MINISTRAL_SERVICE_RUN_DIR" && \
      sha256sum -c "$(basename "$MINISTRAL_SERVICE_METRICS.sha256")") \
      >"$MINISTRAL_RUN_ROOT/service_metrics_checksum.log"
  fi

  (cd "$MINISTRAL_PREDICTOR_RUN_DIR" && sha256sum -c SHA256SUMS) \
    >"$MINISTRAL_RUN_ROOT/predictor_checksums.log"
  local predictor_audit="$MINISTRAL_PREDICTOR_RUN_DIR/predictor_audit.json"
  if [[ ! -s "$predictor_audit" ]]; then
    predictor_audit="$MINISTRAL_PREDICTOR_RUN_DIR/manual_audit.json"
    (cd "$SFS_ROOT" && sha256sum -c "$predictor_audit.sha256") \
      >>"$MINISTRAL_RUN_ROOT/predictor_checksums.log"
  fi
  python - "$predictor_audit" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
if not path.is_file():
    raise RuntimeError(f"Missing predictor audit: {path}")
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "PASS":
    raise RuntimeError(f"Predictor audit did not pass: {path}")
PY
}

ministral3_stage_model() {
  local cache_name="$1"
  local snapshot="$2"
  local source="$MINISTRAL_PROJECT_ROOT/.cache/huggingface/hub/$cache_name/snapshots/$snapshot"
  local destination="$MINISTRAL_JOB_LOCAL/$cache_name/snapshots/$snapshot"
  local required_file
  for required_file in config.json params.json tekken.json consolidated.safetensors; do
    if [[ ! -f "$source/$required_file" ]]; then
      echo "[ERROR] Missing pinned model file: $source/$required_file" >&2
      return 1
    fi
  done
  mkdir -p "$destination"
  rsync -aL --delete --delete-excluded \
    --exclude='model-*.safetensors' "$source/" "$destination/"
  echo "$destination"
}

ministral3_load_simulation_args() {
  local model="$1"
  local destination_name="$2"
  local -n destination="$destination_name"
  mapfile -t destination < <(
    python -m scripts.runs.service_metrics_config \
      --path "$MINISTRAL_SERVICE_METRICS" \
      --expected-feature-set cross_term \
      --require-nonnegative-sfs \
      "${MINISTRAL_MODEL_KEY_ARGS[@]}" \
      simulation-args --model "$model"
  )
  if (( ${#destination[@]} != 7 )); then
    echo "[ERROR] Expected seven simulation arguments for $model; found ${#destination[@]}." >&2
    return 1
  fi
}

ministral3_cleanup_router_pool() {
  local pid
  for pid in "${MINISTRAL_SERVER_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${MINISTRAL_SERVER_PIDS[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
}

ministral3_start_server() {
  local label="$1"
  local model_path="$2"
  local served_model="$3"
  local gpu_id="$4"
  local port="$5"
  local shm_name="$6"
  local wait_log="$7"
  local batch_stats="$8"
  local simulation_name="$9"
  local -n simulation_args="$simulation_name"
  local server_log="$MINISTRAL_RUN_ROOT/vllm_${label}.log"

  VLLM_PER_REQUEST_WAIT_LOG_PATH="$wait_log" \
  CUDA_VISIBLE_DEVICES="$gpu_id" \
    vllm serve --disable-uvicorn-access-log --model "$model_path" \
      --served-model-name "$served_model" \
      --tokenizer-mode mistral \
      --config-format mistral \
      --load-format mistral \
      --dtype auto \
      --max-model-len 131072 \
      --enable-chunked-prefill \
      --no-enable-prefix-caching \
      --max-num-batched-tokens 32768 \
      --max-num-seqs 512 \
      --gpu-memory-utilization 0.90 \
      --tensor-parallel-size 1 \
      --limit-mm-per-prompt '{"image":0}' \
      --mm-processor-cache-gb 0 \
      --batch-stats-file "$batch_stats" \
      --port "$port" \
      --no-enable-wait-time-simulation \
      --enable-snapshot-shm-publishing \
      --snapshot-shm-name "$shm_name" \
      --snapshot-shm-size-bytes "$MINISTRAL_SNAPSHOT_SHM_SIZE_BYTES" \
      --snapshot-shm-publish-interval-ms 0 \
      --output-length-model-path "$MINISTRAL_OUTPUT_PREDICTOR" \
      "${simulation_args[@]}" \
      >"$server_log" 2>&1 &
  MINISTRAL_LAST_SERVER_PID=$!
  MINISTRAL_SERVER_PIDS+=("$MINISTRAL_LAST_SERVER_PID")
}

ministral3_wait_for_server() {
  local label="$1"
  local port="$2"
  local pid="$3"
  local log_file="$4"
  local attempt
  for attempt in $(seq 1 600); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null; then
      echo "[READY] $label on port $port"
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

ministral3_write_instances_config() {
  python - "$MINISTRAL_INSTANCES_CONFIG" "$MINISTRAL_SERVICE_METRICS" \
    "$MINISTRAL_PORT_3B" "$MINISTRAL_PORT_8B" "$MINISTRAL_PORT_14B" \
    "$MINISTRAL_SHM_3B" "$MINISTRAL_SHM_8B" "$MINISTRAL_SHM_14B" \
    "$MINISTRAL_SNAPSHOT_SHM_SIZE_BYTES" <<'PY'
import json
from pathlib import Path
import sys

output_path = Path(sys.argv[1])
metrics_path = Path(sys.argv[2]).resolve()
ports = [int(value) for value in sys.argv[3:6]]
shm_names = sys.argv[6:9]
shm_size = int(sys.argv[9])
metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
metrics = metrics_payload.get("models", metrics_payload)
models = (
    ("ministral3-3b", "vllm-ministral3-3b", "ministral3-3b-instruct", 0.10),
    ("ministral3-8b", "vllm-ministral3-8b", "ministral3-8b-instruct", 0.15),
    ("ministral3-14b", "vllm-ministral3-14b", "ministral3-14b-instruct", 0.20),
)
instances = []
for (model, instance_id, served_model, _), port, shm_name in zip(
    models, ports, shm_names
):
    sfs = metrics[model]["sfs_simulation"]
    instances.append(
        {
            "instance_id": instance_id,
            "address": f"http://127.0.0.1:{port}",
            "default_model": served_model,
            "model_id": model,
            "served_model_name": served_model,
            "snapshot_shm_name": shm_name,
            "snapshot_shm_size_bytes": shm_size,
            "max_num_batched_tokens": 32768,
            "max_num_seqs": 512,
            "chunked_prefill_enabled": True,
            "long_prefill_token_threshold": 0,
            "ttft_batch_model": {
                key: sfs[key]
                for key in (
                    "intercept",
                    "prefill_coeff",
                    "prefill_sq_coeff",
                    "decode_coeff",
                    "sum_coeff",
                    "sum_sq_coeff",
                )
            },
        }
    )
payload = {
    "instances": instances,
    "instance_costs": {
        instance_id: {"prompt": price, "output": price}
        for _, instance_id, _, price in models
    },
    "cost_units": "USD per million tokens",
    "service_metrics_json": str(metrics_path),
    "serving_profile": {
        "dtype": "auto (BF16 checkpoints)",
        "max_model_len": 131072,
        "context_length": 65536,
        "prompt_token_limit": 32768,
        "max_completion_tokens": 8192,
        "chunked_prefill": True,
        "max_num_batched_tokens": 32768,
        "max_num_seqs": 512,
        "prefix_caching": False,
    },
}
output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

ministral3_start_router_pool() {
  if [[ -z "${MINISTRAL_RUN_ROOT:-}" ]]; then
    echo "[ERROR] Set MINISTRAL_RUN_ROOT before starting the router pool." >&2
    return 1
  fi
  mkdir -p "$MINISTRAL_RUN_ROOT"
  ministral3_resolve_sfs_root
  ministral3_resolve_service_run

  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate vllm
  export PYTHONPATH="$SFS_ROOT/src:${PYTHONPATH:-}"

  MINISTRAL_EXPECTED_VLLM_COMMIT="$(git -C "$SFS_ROOT/vllm" rev-parse HEAD)"
  MINISTRAL_ACTIVE_VLLM_ROOT="$(cd "$SFS_ROOT/src" && python - <<'PY'
from pathlib import Path
import vllm

print(Path(vllm.__file__).resolve().parent.parent)
PY
)"
  MINISTRAL_ACTIVE_VLLM_COMMIT="$(git -C "$MINISTRAL_ACTIVE_VLLM_ROOT" rev-parse HEAD)"
  if [[ "$MINISTRAL_ACTIVE_VLLM_COMMIT" != "$MINISTRAL_EXPECTED_VLLM_COMMIT" ]]; then
    echo "[ERROR] Active vLLM commit $MINISTRAL_ACTIVE_VLLM_COMMIT does not match recorded $MINISTRAL_EXPECTED_VLLM_COMMIT." >&2
    return 1
  fi

  ministral3_verify_calibration_artifacts
  ministral3_load_simulation_args ministral3-3b MINISTRAL_SIMULATION_ARGS_3B
  ministral3_load_simulation_args ministral3-8b MINISTRAL_SIMULATION_ARGS_8B
  ministral3_load_simulation_args ministral3-14b MINISTRAL_SIMULATION_ARGS_14B

  local job_id="${SLURM_JOB_ID:?Router pool requires a Slurm job ID}"
  MINISTRAL_JOB_LOCAL="/local/$USER/$job_id"
  mkdir -p \
    "$MINISTRAL_JOB_LOCAL/.cache/vllm" \
    "$MINISTRAL_JOB_LOCAL/.triton" \
    "$MINISTRAL_JOB_LOCAL/.nv"
  rsync -aL "$MINISTRAL_PROJECT_ROOT/.cache/vllm/" "$MINISTRAL_JOB_LOCAL/.cache/vllm/" 2>/dev/null || true
  rsync -aL "$MINISTRAL_PROJECT_ROOT/.triton/" "$MINISTRAL_JOB_LOCAL/.triton/" 2>/dev/null || true
  rsync -aL "$MINISTRAL_PROJECT_ROOT/.nv/" "$MINISTRAL_JOB_LOCAL/.nv/" 2>/dev/null || true
  MINISTRAL_MODEL_3B="$(ministral3_stage_model models--mistralai--Ministral-3-3B-Instruct-2512-BF16 b6d637bef2393152b3da2b2fde72eecdee30557e)"
  MINISTRAL_MODEL_8B="$(ministral3_stage_model models--mistralai--Ministral-3-8B-Instruct-2512-BF16 f6fae9795746f63c9be8344932f01275f3c63734)"
  MINISTRAL_MODEL_14B="$(ministral3_stage_model models--mistralai--Ministral-3-14B-Instruct-2512-BF16 3cea74c1ebaf5ce5f5a2553de470e2ceab825142)"

  export XDG_CACHE_HOME="$MINISTRAL_JOB_LOCAL/.cache"
  export TRITON_CACHE_DIR="$MINISTRAL_JOB_LOCAL/.triton"
  export CUDA_CACHE_PATH="$MINISTRAL_JOB_LOCAL/.nv"
  export VLLM_USE_V1=1
  export VLLM_ATTENTION_BACKEND=FLASH_ATTN
  export VLLM_USE_FLASHINFER_SAMPLER=0
  export VLLM_LOGGING_LEVEL=WARNING

  local default_gpu_ids="0,1,2"
  IFS=',' read -r -a MINISTRAL_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-$default_gpu_ids}"
  if (( ${#MINISTRAL_GPU_IDS[@]} < 3 )); then
    echo "[ERROR] Need three visible H100 GPUs; got ${CUDA_VISIBLE_DEVICES:-<unset>}." >&2
    return 1
  fi

  local port_base=$((20000 + (job_id % 15000) * 3))
  MINISTRAL_PORT_3B="$port_base"
  MINISTRAL_PORT_8B=$((port_base + 1))
  MINISTRAL_PORT_14B=$((port_base + 2))
  MINISTRAL_SNAPSHOT_SHM_SIZE_BYTES=$((8 * 1024 * 1024))
  MINISTRAL_SHM_3B="vllm_snapshot_${job_id}_ministral3_3b"
  MINISTRAL_SHM_8B="vllm_snapshot_${job_id}_ministral3_8b"
  MINISTRAL_SHM_14B="vllm_snapshot_${job_id}_ministral3_14b"
  MINISTRAL_WAIT_LOG_3B="$MINISTRAL_RUN_ROOT/actual_wait_times_ministral3-3b.log"
  MINISTRAL_WAIT_LOG_8B="$MINISTRAL_RUN_ROOT/actual_wait_times_ministral3-8b.log"
  MINISTRAL_WAIT_LOG_14B="$MINISTRAL_RUN_ROOT/actual_wait_times_ministral3-14b.log"
  MINISTRAL_BATCH_STATS_3B="$MINISTRAL_RUN_ROOT/batch_stats_ministral3-3b.csv"
  MINISTRAL_BATCH_STATS_8B="$MINISTRAL_RUN_ROOT/batch_stats_ministral3-8b.csv"
  MINISTRAL_BATCH_STATS_14B="$MINISTRAL_RUN_ROOT/batch_stats_ministral3-14b.csv"
  MINISTRAL_INSTANCES_CONFIG="$MINISTRAL_RUN_ROOT/instances.json"
  : >"$MINISTRAL_WAIT_LOG_3B"
  : >"$MINISTRAL_WAIT_LOG_8B"
  : >"$MINISTRAL_WAIT_LOG_14B"
  ministral3_write_instances_config

  MINISTRAL_SERVER_PIDS=()
  trap ministral3_cleanup_router_pool EXIT INT TERM
  cd "$SFS_ROOT/src"
  ministral3_start_server \
    ministral3-3b "$MINISTRAL_MODEL_3B" ministral3-3b-instruct \
    "${MINISTRAL_GPU_IDS[0]}" "$MINISTRAL_PORT_3B" "$MINISTRAL_SHM_3B" \
    "$MINISTRAL_WAIT_LOG_3B" "$MINISTRAL_BATCH_STATS_3B" \
    MINISTRAL_SIMULATION_ARGS_3B
  local pid_3b="$MINISTRAL_LAST_SERVER_PID"
  ministral3_start_server \
    ministral3-8b "$MINISTRAL_MODEL_8B" ministral3-8b-instruct \
    "${MINISTRAL_GPU_IDS[1]}" "$MINISTRAL_PORT_8B" "$MINISTRAL_SHM_8B" \
    "$MINISTRAL_WAIT_LOG_8B" "$MINISTRAL_BATCH_STATS_8B" \
    MINISTRAL_SIMULATION_ARGS_8B
  local pid_8b="$MINISTRAL_LAST_SERVER_PID"
  ministral3_start_server \
    ministral3-14b "$MINISTRAL_MODEL_14B" ministral3-14b-instruct \
    "${MINISTRAL_GPU_IDS[2]}" "$MINISTRAL_PORT_14B" "$MINISTRAL_SHM_14B" \
    "$MINISTRAL_WAIT_LOG_14B" "$MINISTRAL_BATCH_STATS_14B" \
    MINISTRAL_SIMULATION_ARGS_14B
  local pid_14b="$MINISTRAL_LAST_SERVER_PID"

  ministral3_wait_for_server ministral3-3b "$MINISTRAL_PORT_3B" "$pid_3b" "$MINISTRAL_RUN_ROOT/vllm_ministral3-3b.log"
  ministral3_wait_for_server ministral3-8b "$MINISTRAL_PORT_8B" "$pid_8b" "$MINISTRAL_RUN_ROOT/vllm_ministral3-8b.log"
  ministral3_wait_for_server ministral3-14b "$MINISTRAL_PORT_14B" "$pid_14b" "$MINISTRAL_RUN_ROOT/vllm_ministral3-14b.log"

  nvidia-smi --query-gpu=index,name,uuid,memory.total \
    --format=csv,noheader >"$MINISTRAL_RUN_ROOT/gpus.csv" 2>/dev/null || true
  {
    echo "slurm_job_id=$job_id"
    echo "sfs_commit=$(git -C "$SFS_ROOT" rev-parse HEAD)"
    echo "recorded_vllm_commit=$MINISTRAL_EXPECTED_VLLM_COMMIT"
    echo "active_vllm_root=$MINISTRAL_ACTIVE_VLLM_ROOT"
    echo "active_vllm_commit=$MINISTRAL_ACTIVE_VLLM_COMMIT"
    echo "service_run_dir=$MINISTRAL_SERVICE_RUN_DIR"
    echo "predictor_run_dir=$MINISTRAL_PREDICTOR_RUN_DIR"
    echo "gpu_mapping_3b=${MINISTRAL_GPU_IDS[0]}"
    echo "gpu_mapping_8b=${MINISTRAL_GPU_IDS[1]}"
    echo "gpu_mapping_14b=${MINISTRAL_GPU_IDS[2]}"
    echo "serving_profile=max_model_len:131072,context:65536,prompt_cap:32768,completion_cap:8192,chunked_prefill:true,max_num_batched_tokens:32768,max_num_seqs:512,prefix_caching:false,dtype:auto"
  } >"$MINISTRAL_RUN_ROOT/provenance.txt"
}

ministral3_capacity_qps_values() {
  python -m scripts.runs.service_metrics_config \
    --path "$MINISTRAL_SERVICE_METRICS" \
    --expected-feature-set cross_term \
    --require-nonnegative-sfs \
    "${MINISTRAL_MODEL_KEY_ARGS[@]}" \
    qps-values --fractions "$@"
}

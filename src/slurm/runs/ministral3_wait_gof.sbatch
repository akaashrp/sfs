#!/bin/bash
#SBATCH --job-name=ministral3-wait-gof
#SBATCH --output=experiments/ministral3_wait_gof_%j.out
#SBATCH --error=experiments/ministral3_wait_gof_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=GPU-shared
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h100-80:1
#SBATCH --mail-type=BEGIN,FAIL,END

set -euo pipefail

if [[ -n "${SFS_ROOT:-}" && -d "$SFS_ROOT/src/slurm" ]]; then
  SFS_ROOT="$(cd "$SFS_ROOT" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/src/slurm" ]]; then
  SFS_ROOT="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  SFS_ROOT="$SCRIPT_DIR"
  while [[ "$SFS_ROOT" != "/" && ! -d "$SFS_ROOT/src/slurm" ]]; do
    SFS_ROOT="$(dirname "$SFS_ROOT")"
  done
  if [[ "$SFS_ROOT" == "/" ]]; then
    echo "[ERROR] Could not resolve SFS repo root." >&2
    exit 1
  fi
fi

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SFS_ROOT/.." && pwd)}"
SERVICE_JOB_ID="${SERVICE_JOB_ID:-}"
SERVICE_RUN_DIR="${SERVICE_RUN_DIR:-}"
PREDICTOR_RUN_DIR="${PREDICTOR_RUN_DIR:-}"
VALIDATE_ONLY="${VALIDATE_ONLY:-0}"
NUM_REQUESTS="${NUM_REQUESTS:-16000}"
WAIT_GOF_LOAD_FRACTION="${WAIT_GOF_LOAD_FRACTION:-0.9}"

if [[ "$VALIDATE_ONLY" != "0" && "$VALIDATE_ONLY" != "1" ]]; then
  echo "[ERROR] VALIDATE_ONLY must be 0 or 1." >&2
  exit 1
fi
if [[ -z "$PREDICTOR_RUN_DIR" ]]; then
  echo "[ERROR] Set PREDICTOR_RUN_DIR to an audited predictor run." >&2
  exit 1
fi
if [[ -z "$SERVICE_RUN_DIR" && -z "$SERVICE_JOB_ID" ]]; then
  echo "[ERROR] Set SERVICE_RUN_DIR or SERVICE_JOB_ID." >&2
  exit 1
fi
if [[ "$NUM_REQUESTS" != 16000 ]]; then
  echo "[ERROR] This paper evaluation requires NUM_REQUESTS=16000." >&2
  exit 1
fi

if [[ -z "$SERVICE_RUN_DIR" ]]; then
  shopt -s nullglob
  matches=("$SFS_ROOT"/experiments/ministral3_h100-80_service_metrics_"$SERVICE_JOB_ID"_*)
  shopt -u nullglob
  directories=()
  for path in "${matches[@]}"; do
    [[ -d "$path" ]] && directories+=("$path")
  done
  if (( ${#directories[@]} != 1 )); then
    echo "[ERROR] Expected one service directory for job $SERVICE_JOB_ID; found ${#directories[@]}." >&2
    exit 1
  fi
  SERVICE_RUN_DIR="${directories[0]}"
fi
SERVICE_RUN_DIR="$(cd "$SERVICE_RUN_DIR" && pwd)"
PREDICTOR_RUN_DIR="$(cd "$PREDICTOR_RUN_DIR" && pwd)"
METRICS_JSON="$SERVICE_RUN_DIR/model_metrics.json"
OUTPUT_PREDICTOR="$PREDICTOR_RUN_DIR/output_length_predictor"
ACCURACY_PREDICTOR="$PREDICTOR_RUN_DIR/accuracy_predictor"
HOLDOUT_CACHE_DIR="$SFS_ROOT/experiments/data/prompts/ministral3/holdout_cache_4000"
HOLDOUT_SOURCE_DIR="$PROJECT_ROOT/vllm_utils/bucketed_prompts_0.6B"
TOKENIZER_PATH="$PROJECT_ROOT/.cache/huggingface/hub/models--mistralai--Ministral-3-8B-Instruct-2512-BF16/snapshots/f6fae9795746f63c9be8344932f01275f3c63734"

for required_path in \
  "$METRICS_JSON" \
  "$OUTPUT_PREDICTOR/metadata.json" \
  "$OUTPUT_PREDICTOR/mean_model.txt" \
  "$ACCURACY_PREDICTOR/metadata.json" \
  "$ACCURACY_PREDICTOR/accuracy_model.txt" \
  "$HOLDOUT_CACHE_DIR/manifest.json" \
  "$TOKENIZER_PATH/tekken.json"; do
  if [[ ! -s "$required_path" ]]; then
    echo "[ERROR] Missing required artifact: $required_path" >&2
    exit 1
  fi
done

RUN_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SFS_ROOT/experiments/ministral3_paper/figure2_wait_gof/run_$RUN_ID}"
if [[ "$VALIDATE_ONLY" == "1" ]]; then
  echo "service_run_dir=$SERVICE_RUN_DIR"
  echo "predictor_run_dir=$PREDICTOR_RUN_DIR"
  echo "holdout_cache_dir=$HOLDOUT_CACHE_DIR"
  echo "output_root=$OUTPUT_ROOT"
  echo "model=ministral3-3b estimators=prefill_tps_ttft,live_sfs requests=$NUM_REQUESTS load_fraction=$WAIT_GOF_LOAD_FRACTION"
  exit 0
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "[ERROR] Refusing to overwrite wait-GoF output: $OUTPUT_ROOT" >&2
  exit 1
fi
mkdir -p "$OUTPUT_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm
export PYTHONPATH="$SFS_ROOT/src:${PYTHONPATH:-}"
export SFS_ROOT
export TMPDIR="$SFS_ROOT/experiments/methodology_baselines/scratch"
mkdir -p "$TMPDIR"

EXPECTED_VLLM_COMMIT="$(git -C "$SFS_ROOT/vllm" rev-parse HEAD)"
ACTIVE_VLLM_ROOT="$(cd "$SFS_ROOT/src" && python - <<'PY'
from pathlib import Path
import vllm

print(Path(vllm.__file__).resolve().parent.parent)
PY
)"
ACTIVE_VLLM_COMMIT="$(git -C "$ACTIVE_VLLM_ROOT" rev-parse HEAD)"
if [[ "$ACTIVE_VLLM_ROOT" != "$SFS_ROOT/vllm" ]]; then
  echo "[ERROR] Active editable vLLM root differs from $SFS_ROOT/vllm: $ACTIVE_VLLM_ROOT" >&2
  exit 1
fi
if [[ "$ACTIVE_VLLM_COMMIT" != "$EXPECTED_VLLM_COMMIT" ]]; then
  echo "[ERROR] Active vLLM commit $ACTIVE_VLLM_COMMIT does not match recorded $EXPECTED_VLLM_COMMIT." >&2
  exit 1
fi

MODEL_KEYS=(ministral3-3b ministral3-8b ministral3-14b)
MODEL_KEY_ARGS=()
for model in "${MODEL_KEYS[@]}"; do
  MODEL_KEY_ARGS+=(--model-key "$model")
done
python -m scripts.runs.service_metrics_config \
  --path "$METRICS_JSON" \
  --expected-feature-set cross_term \
  --require-nonnegative-sfs \
  "${MODEL_KEY_ARGS[@]}" \
  validate >"$OUTPUT_ROOT/service_metrics_validation.json"
if [[ -s "$METRICS_JSON.sha256" ]]; then
  (cd "$SERVICE_RUN_DIR" && sha256sum -c "$(basename "$METRICS_JSON.sha256")") \
    >"$OUTPUT_ROOT/service_metrics_checksum.log"
fi

mapfile -t SIMULATION_ARGS < <(
  python -m scripts.runs.service_metrics_config \
    --path "$METRICS_JSON" \
    --expected-feature-set cross_term \
    --require-nonnegative-sfs \
    "${MODEL_KEY_ARGS[@]}" \
    simulation-args --model ministral3-3b
)
if (( ${#SIMULATION_ARGS[@]} != 7 )); then
  echo "[ERROR] Expected seven simulation arguments; found ${#SIMULATION_ARGS[@]}." >&2
  exit 1
fi

REQUEST_RATE_QPS="$(python - "$METRICS_JSON" "$WAIT_GOF_LOAD_FRACTION" <<'PY'
import json
import math
from pathlib import Path
import sys

row = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["ministral3-3b"]
fraction = float(sys.argv[2])
if not math.isfinite(fraction) or not 0 < fraction <= 1.0:
    raise ValueError("WAIT_GOF_LOAD_FRACTION must be in (0, 1]")
print(f"{float(row['service_rate_qps']) * fraction:.12g}")
PY
)"

PREDICTOR_AUDIT="$PREDICTOR_RUN_DIR/predictor_audit.json"
if [[ ! -s "$PREDICTOR_AUDIT" ]]; then
  PREDICTOR_AUDIT="$PREDICTOR_RUN_DIR/manual_audit.json"
fi
(cd "$PREDICTOR_RUN_DIR" && sha256sum -c SHA256SUMS) \
  >"$OUTPUT_ROOT/predictor_checksums.log"
if [[ "$PREDICTOR_AUDIT" == */manual_audit.json ]]; then
  (cd "$SFS_ROOT" && sha256sum -c "$PREDICTOR_AUDIT.sha256") \
    >>"$OUTPUT_ROOT/predictor_checksums.log"
fi
python - "$PREDICTOR_AUDIT" <<'PY'
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

MODEL_CACHE="models--mistralai--Ministral-3-3B-Instruct-2512-BF16"
MODEL_SNAPSHOT="b6d637bef2393152b3da2b2fde72eecdee30557e"
MODEL_SOURCE="$PROJECT_ROOT/.cache/huggingface/hub/$MODEL_CACHE/snapshots/$MODEL_SNAPSHOT"
for required_file in config.json params.json tekken.json consolidated.safetensors; do
  if [[ ! -f "$MODEL_SOURCE/$required_file" ]]; then
    echo "[ERROR] Missing pinned model file: $MODEL_SOURCE/$required_file" >&2
    exit 1
  fi
done

JOB_LOCAL="/local/$USER/$RUN_ID"
source "$SFS_ROOT/src/slurm/runs/ministral3_router_common.sh"
MINISTRAL_JOB_LOCAL="$JOB_LOCAL"
ministral3_configure_ipc
MODEL_DST="$JOB_LOCAL/$MODEL_CACHE/snapshots/$MODEL_SNAPSHOT"
mkdir -p "$MODEL_DST" "$JOB_LOCAL/.cache/vllm" "$JOB_LOCAL/.triton" "$JOB_LOCAL/.nv"
rsync -aL --delete --delete-excluded \
  --exclude='model-*.safetensors' "$MODEL_SOURCE/" "$MODEL_DST/"
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

VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${VISIBLE_GPUS%%,*}"
PORT=$((20000 + (RUN_ID % 15000) * 3))
SERVED_MODEL="ministral3-3b-instruct"
INSTANCE_ID="vllm-ministral3-3b"
SNAPSHOT_SHM_SIZE_BYTES=$((8 * 1024 * 1024))
SNAPSHOT_SHM_NAME="vllm_snapshot_${RUN_ID}_ministral3_3b"
WAIT_LOG="$OUTPUT_ROOT/actual_wait_times_ministral3-3b.log"
BATCH_STATS="$OUTPUT_ROOT/batch_stats_ministral3-3b.csv"
VLLM_LOG="$OUTPUT_ROOT/vllm_ministral3-3b.log"
INSTANCES_CONFIG="$OUTPUT_ROOT/instances.json"
RESULT_JSON="$OUTPUT_ROOT/wait_gof.json"
: >"$WAIT_LOG"

python - "$INSTANCES_CONFIG" "$PORT" "$SERVED_MODEL" "$INSTANCE_ID" \
  "$SNAPSHOT_SHM_NAME" "$SNAPSHOT_SHM_SIZE_BYTES" "$METRICS_JSON" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
port = int(sys.argv[2])
served_model = sys.argv[3]
instance_id = sys.argv[4]
shm_name = sys.argv[5]
shm_size = int(sys.argv[6])
metrics_path = Path(sys.argv[7]).resolve()
row = json.loads(metrics_path.read_text(encoding="utf-8"))["ministral3-3b"]
sfs = row["sfs_simulation"]
payload = {
    "instances": [
        {
            "instance_id": instance_id,
            "address": f"http://127.0.0.1:{port}",
            "default_model": served_model,
            "model_id": "ministral3-3b",
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
    ],
    "instance_costs": {instance_id: {"prompt": 0.10, "output": 0.10}},
    "service_metrics_json": str(metrics_path),
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

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

cd "$SFS_ROOT/src"
VLLM_PER_REQUEST_WAIT_LOG_PATH="$WAIT_LOG" \
CUDA_VISIBLE_DEVICES="$GPU_ID" \
  vllm serve --disable-uvicorn-access-log --model "$MODEL_DST" \
    --served-model-name "$SERVED_MODEL" \
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
    --batch-stats-file "$BATCH_STATS" \
    --port "$PORT" \
    --no-enable-wait-time-simulation \
    --enable-snapshot-shm-publishing \
    --snapshot-shm-name "$SNAPSHOT_SHM_NAME" \
    --snapshot-shm-size-bytes "$SNAPSHOT_SHM_SIZE_BYTES" \
    --snapshot-shm-publish-interval-ms 0 \
    --output-length-model-path "$OUTPUT_PREDICTOR" \
    "${SIMULATION_ARGS[@]}" \
    >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!
PIDS+=("$VLLM_PID")

for attempt in $(seq 1 600); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[ERROR] Ministral 3B server exited before readiness; see $VLLM_LOG" >&2
    exit 1
  fi
  if (( attempt == 600 )); then
    echo "[ERROR] Timed out waiting for Ministral 3B; see $VLLM_LOG" >&2
    exit 1
  fi
  sleep 2
done

python -m scripts.runs.experiments \
  --experiment wait_gof \
  --instances-config "$INSTANCES_CONFIG" \
  --wait-eval-instance-id "$INSTANCE_ID" \
  --request-rate-qps "$REQUEST_RATE_QPS" \
  --arrival-process poisson \
  --wait-estimators prefill_tps_ttft live \
  --service-metrics-json "$METRICS_JSON" \
  --worker-count 1 \
  --max-queue-size 1 \
  --max-completion-tokens 8192 \
  --temperature 0 \
  --top-p 1 \
  --system-prompt 'You are a helpful assistant.' \
  --chat-template-kwargs-json '{}' \
  --accuracy-model-path "$ACCURACY_PREDICTOR" \
  --output-length-model-path "$OUTPUT_PREDICTOR" \
  --bucket-dir "$HOLDOUT_CACHE_DIR" \
  --holdout-prompts-per-bucket 4000 \
  --holdout-bucket-dir "$HOLDOUT_SOURCE_DIR" \
  --holdout-cache-dir "$HOLDOUT_CACHE_DIR" \
  --require-existing-holdout-cache \
  --holdout-start-index 2500 \
  --holdout-context-length 65536 \
  --tokenizer-id "$TOKENIZER_PATH" \
  --tokenizer-mode mistral \
  --num-requests "$NUM_REQUESTS" \
  --max-prompts-per-source 0 \
  --per-request-wait-log "$WAIT_LOG" \
  --wait-gof-min-matched-pairs "$NUM_REQUESTS" \
  --output-path "$RESULT_JSON" \
  >"$OUTPUT_ROOT/driver.log" 2>&1

python - "$RESULT_JSON" "$NUM_REQUESTS" "$REQUEST_RATE_QPS" \
  "$SERVICE_RUN_DIR" "$PREDICTOR_RUN_DIR" <<'PY'
import json
import math
from pathlib import Path
import sys

result_path = Path(sys.argv[1])
expected = int(sys.argv[2])
payload = json.loads(result_path.read_text(encoding="utf-8"))
gof = payload["wait_gof"]
runs = {run["wait_estimator"]: run for run in gof["runs"]}
if set(runs) != {"prefill_tps_ttft", "live"}:
    raise RuntimeError(f"Unexpected estimator set: {sorted(runs)}")
audit_runs = {}
for estimator, run in runs.items():
    records = run["per_request"]
    errors = [record for record in records if record.get("error") is not None]
    fit = run["actual_wait_fit"]
    metrics = fit["prediction_metrics_ms"]
    if len(records) != expected or errors:
        raise RuntimeError(
            f"{estimator}: expected {expected} error-free requests; "
            f"found {len(records)} rows and {len(errors)} errors"
        )
    if int(fit["matched_pairs"]) != expected or int(metrics["count"]) != expected:
        raise RuntimeError(f"{estimator}: incomplete predicted/actual matching")
    for key in ("mae", "rmse", "mean_error"):
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise RuntimeError(f"{estimator}: invalid {key}={value!r}")
    plot_path = Path(run["actual_wait_plot_path"])
    if not plot_path.is_file() or plot_path.stat().st_size == 0:
        raise RuntimeError(f"{estimator}: missing parity plot {plot_path}")
    audit_runs[estimator] = {
        "matched_pairs": expected,
        "prediction_metrics_ms": metrics,
        "plot_path": str(plot_path.resolve()),
    }

audit = {
    "status": "PASS",
    "model": "ministral3-3b",
    "num_requests": expected,
    "request_rate_qps": float(sys.argv[3]),
    "service_run_dir": str(Path(sys.argv[4]).resolve()),
    "predictor_run_dir": str(Path(sys.argv[5]).resolve()),
    "estimators": {
        "prefill_tps_ttft": "paper prefill-throughput baseline",
        "live": "SFS local snapshot simulation",
    },
    "runs": audit_runs,
}
(result_path.parent / "audit.json").write_text(
    json.dumps(audit, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(audit, indent=2))
PY

{
  echo "slurm_job_id=${SLURM_JOB_ID:-local}"
  echo "sfs_commit=$(git -C "$SFS_ROOT" rev-parse HEAD)"
  echo "recorded_vllm_commit=$EXPECTED_VLLM_COMMIT"
  echo "active_vllm_root=$ACTIVE_VLLM_ROOT"
  echo "active_vllm_commit=$ACTIVE_VLLM_COMMIT"
  echo "service_run_dir=$SERVICE_RUN_DIR"
  echo "predictor_run_dir=$PREDICTOR_RUN_DIR"
  echo "request_rate_qps=$REQUEST_RATE_QPS"
  echo "wait_gof_load_fraction=$WAIT_GOF_LOAD_FRACTION"
  echo "num_requests=$NUM_REQUESTS"
  echo "serving_profile=max_model_len:131072,context:65536,prompt_cap:32768,completion_cap:8192,chunked_prefill:true,max_num_batched_tokens:32768,max_num_seqs:512,prefix_caching:false,dtype:auto"
} >"$OUTPUT_ROOT/provenance.txt"

find "$OUTPUT_ROOT" -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum >"$OUTPUT_ROOT/SHA256SUMS"
echo "[DONE] Ministral Figure 2 wait-GoF artifacts: $OUTPUT_ROOT"

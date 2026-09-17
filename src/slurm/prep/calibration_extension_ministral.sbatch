#!/bin/bash
# Calibration extension, Ministral 3 family: greedy completions for raw bucket
# rows 6500-8999 (2,500 per bucket, 10,000 per model), disjoint from the
# original calibration rows 0-2499 and the evaluation holdout rows 2500-6499.
# Reuses the H100 Ministral generation driver with the campaign serving profile.
#SBATCH --job-name=calib-ext-ministral
#SBATCH --output=/ocean/projects/cis250162p/aparthas/sfs_artifacts/calibration_extension_20260917/slurm/ministral_%j.out
#SBATCH --error=/ocean/projects/cis250162p/aparthas/sfs_artifacts/calibration_extension_20260917/slurm/ministral_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=GPU-shared
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100-80:3

set -euo pipefail

SFS_ROOT=/ocean/projects/cis250162p/aparthas/sfs_calib_ext_20260917
PROJECT_ROOT=/ocean/projects/cis250162p/aparthas
OUTPUT_ROOT="$PROJECT_ROOT/sfs_artifacts/calibration_extension_20260917/ministral"
BUNDLE_DIR="$PROJECT_ROOT/sfs_cloud_artifacts_20260916"
DRIVER="$SFS_ROOT/src/slurm/prep/run_ministral3_bucketed_prompts_all_models.sh"

export PROJECT_ROOT
export EXPERIMENT_DIR="$OUTPUT_ROOT"
export RUN_DIR_OVERRIDE="$OUTPUT_ROOT/run"
# Pinned fork source (vllm submodule at the campaign commit) ahead of the
# editable install; the driver verifies the active commit matches it.
export PYTHONPATH="$SFS_ROOT/vllm${PYTHONPATH:+:$PYTHONPATH}"

# Campaign serving profile (bundle.json families.ministral.profile).
export MINISTRAL_GPU_PROFILE=h100-80
export MINISTRAL_TP_3B=1 MINISTRAL_TP_8B=1 MINISTRAL_TP_14B=1
export VLLM_DTYPE=auto
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TOKENIZER_MODE=mistral
export MAX_MODEL_LEN=131072 CONTEXT_LENGTH=65536 PROMPT_TOKEN_LIMIT=32768 MAX_COMPLETION_TOKENS=8192
export CHUNKED_PREFILL=1 MAX_NUM_BATCHED_TOKENS=32768 MAX_NUM_SEQS=512 GPU_MEMORY_UTILIZATION=0.90
export ENFORCE_EAGER=0
export SYSTEM_PROMPT="You are a helpful assistant."
export CHAT_TEMPLATE_KWARGS_JSON='{}'
export TEMPERATURE=0.0 TOP_P=1.0 SEED=69

# Inputs and index range.
export BUCKET_DIR="$PROJECT_ROOT/vllm_utils/bucketed_prompts_0.6B"
export PROMPT_START_INDEX=6500
export MAX_PROMPTS_PER_BUCKET=2500
export WORKER_COUNT=128 MAX_QUEUE_SIZE=0 RUN_IN_PARALLEL=1
export MINISTRAL_WORKLOAD=generation
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

mkdir -p "$OUTPUT_ROOT"
if [[ -e "$RUN_DIR_OVERRIDE" ]]; then
  echo "[ERROR] Refusing to overwrite existing run directory: $RUN_DIR_OVERRIDE" >&2
  exit 1
fi

DRIVER_STATUS=0
bash "$DRIVER" || DRIVER_STATUS=$?
echo "[INFO] driver exit status: $DRIVER_STATUS"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm
export PYTHONPATH="$SFS_ROOT/vllm:$SFS_ROOT/src"
cd "$SFS_ROOT/src"
SETTINGS_JSON=$(cat <<JSON
{"dtype": "auto (BF16 checkpoints)", "attention_backend": "FLASH_ATTN", "tokenizer_mode": "mistral",
 "config_format": "mistral", "load_format": "mistral", "max_model_len": 131072, "context_length": 65536,
 "prompt_token_limit": 32768, "max_completion_tokens": 8192, "chunked_prefill": true,
 "max_num_batched_tokens": 32768, "max_num_seqs": 512, "prefix_caching": false, "gpu_memory_utilization": 0.9,
 "tensor_parallel_sizes": [1, 1, 1], "system_prompt": "You are a helpful assistant.", "chat_template_kwargs": {},
 "temperature": 0.0, "top_p": 1.0, "seed": 69, "worker_count": 128, "max_queue_size": 0,
 "client_tokenizer": "per-model pinned snapshot (mistral mode), as in the original calibration run 44849564",
 "original_calibration_rows": "0-2499 (sfs_artifacts/ministral3_calibration_44849564_judge_45009121_20260902)",
 "evaluation_holdout_rows": "2500-6499 (bundle ministral/holdout uses 2500-4499; qwen/holdout 2500-6499)"}
JSON
)
MANIFEST_STATUS=0
python -m scripts.prep.calibration_extension_manifest \
  --family ministral \
  --output-root "$OUTPUT_ROOT" \
  --run-dir "$RUN_DIR_OVERRIDE" \
  --bundle "$BUNDLE_DIR/bundle.json" \
  --sfs-root "$SFS_ROOT" \
  --bucket-dir "$BUCKET_DIR" \
  --start-index "$PROMPT_START_INDEX" \
  --count "$MAX_PROMPTS_PER_BUCKET" \
  --models ministral3-3b ministral3-8b ministral3-14b \
  --served-names ministral3-3b-instruct ministral3-8b-instruct ministral3-14b-instruct \
  --tensor-parallel 1 1 1 \
  --settings-json "$SETTINGS_JSON" \
  --code-files "$DRIVER" "$SFS_ROOT/src/scripts/prep/run_qwen_bucketed_prompts.py" \
    "$SFS_ROOT/src/scripts/prep/calibration_extension_manifest.py" \
    "$SFS_ROOT/src/slurm/prep/calibration_extension_ministral.sbatch" \
  --slurm-job-id "${SLURM_JOB_ID:-local}" \
  --driver-exit-status "$DRIVER_STATUS" \
  --purpose "Additional greedy calibration outputs (rows 6500-8999 per bucket) for SFS output-length predictor mean-head retraining per prompt-length band and Ministral tail analysis" \
  || MANIFEST_STATUS=$?
echo "[INFO] manifest exit status: $MANIFEST_STATUS"
if (( DRIVER_STATUS != 0 || MANIFEST_STATUS != 0 )); then
  exit 1
fi
echo "[DONE] Ministral calibration extension complete: $OUTPUT_ROOT"

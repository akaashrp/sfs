#!/bin/bash
#SBATCH --job-name=ministral3-judge
#SBATCH --output=experiments/ministral3_judge_%j.out
#SBATCH --error=experiments/ministral3_judge_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=RM-shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --mail-type=BEGIN,FAIL,END

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SFS_ROOT:-}" && -f "$SFS_ROOT/src/scripts/prep/quality_metrics.py" ]]; then
  RESOLVED_SFS_ROOT="$SFS_ROOT"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/src/scripts/prep/quality_metrics.py" ]]; then
  RESOLVED_SFS_ROOT="$SLURM_SUBMIT_DIR"
else
  RESOLVED_SFS_ROOT="$SCRIPT_DIR"
  while [[ "$RESOLVED_SFS_ROOT" != "/" && ! -d "$RESOLVED_SFS_ROOT/src/slurm" ]]; do
    RESOLVED_SFS_ROOT="$(dirname "$RESOLVED_SFS_ROOT")"
  done
fi
if [[ "$RESOLVED_SFS_ROOT" == "/" ]]; then
  echo "[ERROR] Could not resolve SFS repo root from $SCRIPT_DIR" >&2
  exit 1
fi
SFS_ROOT="$RESOLVED_SFS_ROOT"

OUTPUTS_ROOT="${OUTPUTS_ROOT:-${1:-}}"
GEMINI_ENV_FILE="${GEMINI_ENV_FILE:-/jet/home/aparthas/.config/sfs/gemini.env}"
JUDGE_MODEL="${JUDGE_MODEL:-gemini-3.1-pro-preview}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-20}"
JUDGE_RETRIES="${JUDGE_RETRIES:-3}"
INDIVIDUAL_RETRIES="${INDIVIDUAL_RETRIES:-3}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-10}"
VALIDATE_ONLY="${VALIDATE_ONLY:-0}"

if [[ -z "$OUTPUTS_ROOT" ]]; then
  echo "[ERROR] OUTPUTS_ROOT must point to a completions directory." >&2
  exit 1
fi
OUTPUTS_ROOT="$(realpath -m "$OUTPUTS_ROOT")"
if [[ ! -d "$OUTPUTS_ROOT" ]]; then
  echo "[ERROR] OUTPUTS_ROOT does not exist: $OUTPUTS_ROOT" >&2
  exit 1
fi

for integer_name in JUDGE_BATCH_SIZE JUDGE_RETRIES INDIVIDUAL_RETRIES; do
  if ! [[ "${!integer_name}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] $integer_name must be a positive integer; got ${!integer_name}." >&2
    exit 1
  fi
done
if ! [[ "$RETRY_SLEEP_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "[ERROR] RETRY_SLEEP_SECONDS must be non-negative; got $RETRY_SLEEP_SECONDS." >&2
  exit 1
fi

for model_name in ministral3-3b ministral3-8b ministral3-14b; do
  if [[ ! -d "$OUTPUTS_ROOT/$model_name" ]]; then
    echo "[ERROR] Missing model output directory: $OUTPUTS_ROOT/$model_name" >&2
    exit 1
  fi
done

echo "[CONFIG] outputs_root=$OUTPUTS_ROOT"
echo "[CONFIG] judge_model=$JUDGE_MODEL"
echo "[CONFIG] judge_batch_size=$JUDGE_BATCH_SIZE"
echo "[CONFIG] judge_retries=$JUDGE_RETRIES"
echo "[CONFIG] individual_retries=$INDIVIDUAL_RETRIES"
echo "[CONFIG] retry_sleep_seconds=$RETRY_SLEEP_SECONDS"

if [[ "$VALIDATE_ONLY" == "1" ]]; then
  echo "[VALID] Ministral judge launcher configuration is valid."
  exit 0
fi

if [[ ! -f "$GEMINI_ENV_FILE" || ! -r "$GEMINI_ENV_FILE" ]]; then
  echo "[ERROR] Gemini environment file is not readable: $GEMINI_ENV_FILE" >&2
  exit 1
fi
KEY_FILE_MODE="$(stat -c '%a' "$GEMINI_ENV_FILE")"
KEY_FILE_OWNER="$(stat -c '%U' "$GEMINI_ENV_FILE")"
if [[ "$KEY_FILE_OWNER" != "$USER" ]]; then
  echo "[ERROR] Gemini environment file must be owned by $USER; owner is $KEY_FILE_OWNER." >&2
  exit 1
fi
if [[ "$KEY_FILE_MODE" != "600" && "$KEY_FILE_MODE" != "400" ]]; then
  echo "[ERROR] Gemini environment file mode must be 600 or 400; got $KEY_FILE_MODE." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$GEMINI_ENV_FILE"
set +a
if [[ -z "${GEMINI_API_KEY:-}" && -z "${GOOGLE_API_KEY:-}" ]]; then
  echo "[ERROR] $GEMINI_ENV_FILE must define GEMINI_API_KEY or GOOGLE_API_KEY." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm
export PYTHONPATH="$SFS_ROOT/src:${PYTHONPATH:-}"

SUMMARY_PATH="$OUTPUTS_ROOT/judge_run_summary_${SLURM_JOB_ID:-local}.json"
python -m scripts.prep.quality_metrics \
  --outputs-root "$OUTPUTS_ROOT" \
  --models ministral3-3b ministral3-8b ministral3-14b \
  --judge-model "$JUDGE_MODEL" \
  --judge-batch-size "$JUDGE_BATCH_SIZE" \
  --judge-retries "$JUDGE_RETRIES" \
  --individual-retries "$INDIVIDUAL_RETRIES" \
  --retry-sleep-seconds "$RETRY_SLEEP_SECONDS" \
  --summary-path "$SUMMARY_PATH"

echo "[DONE] Completed Ministral judging."
echo "[DONE] Summary: $SUMMARY_PATH"

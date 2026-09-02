#!/bin/bash
#SBATCH --job-name=ministral3-holdouts
#SBATCH --output=experiments/ministral3_holdouts_%j.out
#SBATCH --error=experiments/ministral3_holdouts_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=RM-shared
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=BEGIN,FAIL,END

set -euo pipefail

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
SOURCE_BUCKET_DIR="${SOURCE_BUCKET_DIR:-$PROJECT_ROOT/vllm_utils/bucketed_prompts_0.6B}"
TOKENIZER_ID="${TOKENIZER_ID:-$PROJECT_ROOT/.cache/huggingface/hub/models--mistralai--Ministral-3-8B-Instruct-2512-BF16/snapshots/f6fae9795746f63c9be8344932f01275f3c63734}"
HOLDOUT_ROOT="${HOLDOUT_ROOT:-$SFS_ROOT/experiments/data/prompts/ministral3}"
HOLDOUT_START_INDEX="${HOLDOUT_START_INDEX:-2500}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-65536}"
PROMPT_TOKEN_LIMIT="${PROMPT_TOKEN_LIMIT:-32768}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-8192}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a helpful assistant.}"
CHAT_TEMPLATE_KWARGS_JSON="${CHAT_TEMPLATE_KWARGS_JSON:-{}}"
VALIDATE_ONLY="${VALIDATE_ONLY:-0}"

if [[ ! -d "$SOURCE_BUCKET_DIR" ]]; then
  echo "[ERROR] Missing source bucket directory: $SOURCE_BUCKET_DIR" >&2
  exit 1
fi
if [[ ! -f "$TOKENIZER_ID/tekken.json" ]]; then
  echo "[ERROR] Missing pinned Mistral tokenizer: $TOKENIZER_ID" >&2
  exit 1
fi
for bucket in alpaca govreport-summarization hotpot_qa writingprompts; do
  path="$SOURCE_BUCKET_DIR/${bucket}.jsonl"
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] Missing source bucket: $path" >&2
    exit 1
  fi
  rows="$(wc -l <"$path")"
  if (( rows < HOLDOUT_START_INDEX + 4000 )); then
    echo "[ERROR] $path has $rows rows; need at least $((HOLDOUT_START_INDEX + 4000))." >&2
    exit 1
  fi
done

if [[ "$VALIDATE_ONLY" == "1" ]]; then
  echo "[VALID] source=$SOURCE_BUCKET_DIR start=$HOLDOUT_START_INDEX sizes=2000,4000"
  echo "[VALID] tokenizer=$TOKENIZER_ID mode=mistral"
  echo "[VALID] context=$CONTEXT_LENGTH prompt_limit=$PROMPT_TOKEN_LIMIT completion_limit=$MAX_COMPLETION_TOKENS"
  echo "[VALID] system_prompt=$SYSTEM_PROMPT chat_template_kwargs=$CHAT_TEMPLATE_KWARGS_JSON"
  echo "[VALID] output_root=$HOLDOUT_ROOT"
  exit 0
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm
export PYTHONPATH="$SFS_ROOT/src:${PYTHONPATH:-}"

for count in 2000 4000; do
  python -m scripts.prep.holdout_prompts \
    --source-bucket-dir "$SOURCE_BUCKET_DIR" \
    --cache-dir "$HOLDOUT_ROOT/holdout_cache_${count}" \
    --tokenizer-id "$TOKENIZER_ID" \
    --tokenizer-mode mistral \
    --holdout-start-index "$HOLDOUT_START_INDEX" \
    --holdout-prompts-per-bucket "$count" \
    --holdout-context-length "$CONTEXT_LENGTH" \
    --max-completion-tokens "$MAX_COMPLETION_TOKENS" \
    --prompt-token-limit "$PROMPT_TOKEN_LIMIT" \
    --system-prompt "$SYSTEM_PROMPT" \
    --chat-template-kwargs-json "$CHAT_TEMPLATE_KWARGS_JSON"
done

python - "$HOLDOUT_ROOT" "$CONTEXT_LENGTH" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
context_length = int(sys.argv[2])
bucket_names = ("alpaca", "govreport-summarization", "hotpot_qa", "writingprompts")
ids_by_size = {}
for count in (2000, 4000):
    cache = root / f"holdout_cache_{count}"
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    if set(manifest["bucket_counts"].values()) != {count}:
        raise RuntimeError(f"Unexpected bucket counts in {cache}")
    ids_by_size[count] = {}
    for bucket in bucket_names:
        path = cache / f"{bucket}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len(rows) != count:
            raise RuntimeError(f"{path}: expected {count} rows, found {len(rows)}")
        request_ids = [row["request_id"] for row in rows]
        if len(set(request_ids)) != count:
            raise RuntimeError(f"{path}: duplicate request IDs")
        if any(row["prompt_tokens"] + row["max_completion_tokens"] > context_length for row in rows):
            raise RuntimeError(f"{path}: context budget violation")
        ids_by_size[count][bucket] = set(request_ids)
for bucket in bucket_names:
    if not ids_by_size[2000][bucket].issubset(ids_by_size[4000][bucket]):
        raise RuntimeError(f"{bucket}: 2000-prompt cache is not a subset of 4000")
print("Validated aligned Ministral holdout caches for N=2000 and N=4000")
PY

echo "[DONE] Ministral holdout caches: $HOLDOUT_ROOT"

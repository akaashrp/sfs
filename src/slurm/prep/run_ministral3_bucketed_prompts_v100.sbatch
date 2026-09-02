#!/bin/bash
#SBATCH --job-name=ministral3-bucketed-v100
#SBATCH --output=experiments/ministral3_bucketed_v100_%j.out
#SBATCH --error=experiments/ministral3_bucketed_v100_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=GPU-shared
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:v100-32:4
#SBATCH --mail-type=BEGIN,FAIL,END

set -euo pipefail

if [[ -n "${SFS_ROOT:-}" && -f "$SFS_ROOT/src/slurm/prep/run_ministral3_bucketed_prompts_all_models.sh" ]]; then
  DRIVER="$SFS_ROOT/src/slurm/prep/run_ministral3_bucketed_prompts_all_models.sh"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/src/slurm/prep/run_ministral3_bucketed_prompts_all_models.sh" ]]; then
  DRIVER="$SLURM_SUBMIT_DIR/src/slurm/prep/run_ministral3_bucketed_prompts_all_models.sh"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  DRIVER="$SCRIPT_DIR/run_ministral3_bucketed_prompts_all_models.sh"
fi
if [[ ! -f "$DRIVER" ]]; then
  echo "[ERROR] Could not locate run_ministral3_bucketed_prompts_all_models.sh" >&2
  exit 1
fi
export MINISTRAL_GPU_PROFILE="v100-32"
export MINISTRAL_TP_3B="${MINISTRAL_TP_3B:-1}"
export MINISTRAL_TP_8B="${MINISTRAL_TP_8B:-1}"
export MINISTRAL_TP_14B="${MINISTRAL_TP_14B:-2}"
export VLLM_DTYPE="half"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
export MAX_MODEL_LEN="40960"
export CONTEXT_LENGTH="40960"
export PROMPT_TOKEN_LIMIT="32768"
export MAX_COMPLETION_TOKENS="8192"
export CHUNKED_PREFILL="0"
export MAX_NUM_BATCHED_TOKENS="49152"
export MAX_NUM_SEQS="64"
export ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

exec bash "$DRIVER"

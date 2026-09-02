#!/bin/bash
#SBATCH --job-name=ministral3-service-metrics
#SBATCH --output=experiments/ministral3_service_metrics_%j.out
#SBATCH --error=experiments/ministral3_service_metrics_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=GPU-shared
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100-80:3
#SBATCH --mail-type=BEGIN,FAIL,END

set -euo pipefail

if [[ -n "${SFS_ROOT:-}" && -f "$SFS_ROOT/src/slurm/prep/run_ministral3_bucketed_prompts_all_models.sh" ]]; then
  DRIVER="$SFS_ROOT/src/slurm/prep/run_ministral3_bucketed_prompts_all_models.sh"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/src/slurm/prep/run_ministral3_bucketed_prompts_all_models.sh" ]]; then
  DRIVER="$SLURM_SUBMIT_DIR/src/slurm/prep/run_ministral3_bucketed_prompts_all_models.sh"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  DRIVER="$SCRIPT_DIR/../prep/run_ministral3_bucketed_prompts_all_models.sh"
fi
if [[ ! -f "$DRIVER" ]]; then
  echo "[ERROR] Could not locate the shared Ministral driver." >&2
  exit 1
fi

export MINISTRAL_WORKLOAD="service_metrics"
export MINISTRAL_GPU_PROFILE="h100-80"
export MINISTRAL_TP_3B="${MINISTRAL_TP_3B:-1}"
export MINISTRAL_TP_8B="${MINISTRAL_TP_8B:-1}"
export MINISTRAL_TP_14B="${MINISTRAL_TP_14B:-1}"
export VLLM_DTYPE="${VLLM_DTYPE:-auto}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-65536}"
export PROMPT_TOKEN_LIMIT="${PROMPT_TOKEN_LIMIT:-32768}"
export MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-8192}"
export CHUNKED_PREFILL="${CHUNKED_PREFILL:-1}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

exec bash "$DRIVER"

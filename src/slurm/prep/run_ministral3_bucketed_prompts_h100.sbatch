#!/bin/bash
#SBATCH --job-name=ministral3-bucketed-h100
#SBATCH --output=experiments/ministral3_bucketed_h100_%j.out
#SBATCH --error=experiments/ministral3_bucketed_h100_%j.err
#SBATCH --time=04:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=GPU-shared
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100-80:3
#SBATCH --mail-type=BEGIN,FAIL,END

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MINISTRAL_GPU_PROFILE="h100-80"
export MINISTRAL_TP_3B="${MINISTRAL_TP_3B:-1}"
export MINISTRAL_TP_8B="${MINISTRAL_TP_8B:-1}"
export MINISTRAL_TP_14B="${MINISTRAL_TP_14B:-1}"
export VLLM_DTYPE="${VLLM_DTYPE:-half}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export CHUNKED_PREFILL="${CHUNKED_PREFILL:-0}"

exec bash "$SCRIPT_DIR/run_ministral3_bucketed_prompts_all_models.sh"

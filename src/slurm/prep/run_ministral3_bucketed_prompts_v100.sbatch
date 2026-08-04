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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MINISTRAL_GPU_PROFILE="v100-32"
export MINISTRAL_TP_3B="${MINISTRAL_TP_3B:-1}"
export MINISTRAL_TP_8B="${MINISTRAL_TP_8B:-1}"
export MINISTRAL_TP_14B="${MINISTRAL_TP_14B:-2}"
export VLLM_DTYPE="half"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
export CHUNKED_PREFILL="0"

exec bash "$SCRIPT_DIR/run_ministral3_bucketed_prompts_all_models.sh"

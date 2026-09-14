#!/usr/bin/env bash
#SBATCH --job-name=sfs-predictor-sweep
#SBATCH --output=experiments/qwen_predictor_sweep_%j.out
#SBATCH --error=experiments/qwen_predictor_sweep_%j.err
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100-80:4
#SBATCH --time=12:00:00
set -euo pipefail
: "${SFS_ROOT:?}" "${VARIANT_MANIFEST:?}" "${SMOKE_STAGE:?}" "${OUTPUT_ROOT:?}"
source "$SFS_ROOT/src/slurm/prep/paper_ablation_env.sh"
export PYTHONPATH="$SFS_ROOT/vllm:$PYTHONPATH"
python - "$VARIANT_MANIFEST" "$SMOKE_STAGE" <<'PY'
import sys
from scripts.runs.qwen_predictor_variants import validate,validate_smoke
manifest=validate(sys.argv[1]);validate_smoke(manifest,sys.argv[1],sys.argv[2])
print('PASS_VARIANT_CPU_AND_GPU_GATES')
PY
if [[ "${VALIDATE_ONLY:-0}" == 1 ]]; then exit 0; fi
python -m scripts.runs.qwen_predictor_variants sweep --manifest "$VARIANT_MANIFEST" --smoke-stage "$SMOKE_STAGE" --output "$OUTPUT_ROOT"

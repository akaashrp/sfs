#!/usr/bin/env bash
#SBATCH --job-name=sfs-predictor-smoke
#SBATCH --output=experiments/qwen_predictor_smoke_%j.out
#SBATCH --error=experiments/qwen_predictor_smoke_%j.err
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100-80:4
#SBATCH --time=04:00:00
set -euo pipefail
: "${SFS_ROOT:?}" "${VARIANT_MANIFESTS:?}" "${OUTPUT_ROOT:?}"
source "$SFS_ROOT/src/slurm/prep/paper_ablation_env.sh"
export PYTHONPATH="$SFS_ROOT/vllm:$PYTHONPATH"
read -r -a manifests <<< "$VARIANT_MANIFESTS"
for manifest in "${manifests[@]}"; do
 python -m scripts.runs.qwen_predictor_variants validate --manifest "$manifest"
done
if [[ "${VALIDATE_ONLY:-0}" == 1 ]]; then exit 0; fi
if [[ -e "$OUTPUT_ROOT" ]]; then echo 'Refusing to overwrite predictor smoke output' >&2; exit 1; fi
mkdir -p "$OUTPUT_ROOT"
for manifest in "${manifests[@]}"; do
 arm="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["arm"])' "$manifest")"
 python -m scripts.runs.qwen_predictor_variants smoke --manifest "$manifest" --output "$OUTPUT_ROOT/$arm"
done

#!/usr/bin/env bash
#SBATCH --job-name=qwen-paper-baselines
#SBATCH --output=experiments/qwen_baselines_%j.out
#SBATCH --error=experiments/qwen_baselines_%j.err
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100-80:4
#SBATCH --time=24:00:00
set -euo pipefail
: "${SFS_ROOT:?Set the active worktree}"
: "${QWEN_MANIFEST:?Set the CPU-prepared canonical manifest}"
: "${BASELINE_MODE:?Set calibrate, smoke or sweep}"
: "${ROUTEBALANCE_PREDICTOR_DIR:?Set the native Qwen predictor}"
case "$BASELINE_MODE" in calibrate|smoke|sweep) ;; *) echo "Invalid BASELINE_MODE" >&2; exit 1 ;; esac
source "$SFS_ROOT/src/slurm/prep/paper_ablation_env.sh"
export PYTHONPATH="$SFS_ROOT/vllm:$PYTHONPATH"
extra=()
: "${QWEN_CPU_VALIDATION:?Set the completed source-bound CPU report}"
extra+=(--cpu-validation "$QWEN_CPU_VALIDATION")
policy_args=()
if [[ -n "${QWEN_POLICIES:-}" ]]; then
  read -r -a selected_policies <<< "$QWEN_POLICIES"
  policy_args=(--policies "${selected_policies[@]}")
fi
if [[ -n "${QWEN_QPS:-}" ]]; then
  read -r -a selected_qps <<< "$QWEN_QPS"
  policy_args+=(--qps-values "${selected_qps[@]}")
fi
if [[ "$BASELINE_MODE" == sweep || "$BASELINE_MODE" == smoke ]]; then
  : "${QWEN_STAGE_DIR:?Set the completed Qwen calibration and smoke output}"
  extra+=(--stage-dir "$QWEN_STAGE_DIR")
fi
validation_args=()
if [[ "$BASELINE_MODE" == smoke ]]; then validation_args+=(--reuse-calibration-only); fi
python -m scripts.runs.qwen_baselines validate --manifest "$QWEN_MANIFEST" \
  --predictor "$ROUTEBALANCE_PREDICTOR_DIR" "${extra[@]}" "${policy_args[@]}" "${validation_args[@]}"
if [[ "${VALIDATE_ONLY:-0}" == 1 ]]; then exit 0; fi
: "${OUTPUT_ROOT:?Set a fresh output root}"
python -m scripts.runs.qwen_baselines "$BASELINE_MODE" --manifest "$QWEN_MANIFEST" \
  --predictor "$ROUTEBALANCE_PREDICTOR_DIR" --output-root "$OUTPUT_ROOT" "${extra[@]}" "${policy_args[@]}"

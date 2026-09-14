#!/usr/bin/env bash
#SBATCH --job-name=sfs-paper-cpu
#SBATCH --output=experiments/paper_cpu_%j.out
#SBATCH --error=experiments/paper_cpu_%j.err
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=30000M
#SBATCH --time=16:00:00
set -euo pipefail
: "${SFS_ROOT:?Set the active worktree}"
: "${ABLATION_MODE:?Set judge, estimators, or routebalance_predictor}"
: "${PREPARED_DATA:?Set the audited Qwen data directory}"
: "${OUTPUT_ROOT:?Set a new output directory}"
source "$SFS_ROOT/src/slurm/prep/paper_ablation_env.sh"
python -m scripts.prep.paper_ablation_data --output-root "$PREPARED_DATA" --validate-only
case "$ABLATION_MODE" in
  judge)
    python -m scripts.eval.judge_ablation validate --prepared-dir "$PREPARED_DATA" --output-root "$OUTPUT_ROOT"
    if [[ "${VALIDATE_ONLY:-0}" == 1 ]]; then exit 0; fi
    # Never enable shell tracing around credentials.
    set +x
    set -a
    source /jet/home/aparthas/.config/sfs/gemini.env
    set +a
    python -m scripts.eval.judge_ablation run --prepared-dir "$PREPARED_DATA" \
      --output-root "$OUTPUT_ROOT" --judge-model "${JUDGE_MODEL:-gemini-2.5-flash}"
    ;;
  estimators)
    extra=()
    if [[ "${VALIDATE_ONLY:-0}" == 1 ]]; then extra+=(--validate-only); fi
    python -m scripts.eval.estimator_ablation --prepared-dir "$PREPARED_DATA" \
      --predictor-root "$SFS_ROOT/src/assets/predictors" --output-root "$OUTPUT_ROOT" "${extra[@]}"
    ;;
  routebalance_predictor)
    extra=()
    if [[ "${VALIDATE_ONLY:-0}" == 1 ]]; then extra+=(--validate-only); fi
    python -m scripts.prep.build_routebalance_predictor --completions-root "$PREPARED_DATA/calibration" \
      --models qwen3-0.6b qwen3-8b qwen3-32b --output-dir "$OUTPUT_ROOT" \
      --validation-id-files "$SFS_ROOT/src/assets/predictors/accuracy_predictor/test_example_ids.json" \
        "$SFS_ROOT/src/assets/predictors/output_length_predictor/test_example_ids.json" \
      --cache-dir /ocean/projects/cis250162p/aparthas/.cache/huggingface/hub --cpu-threads 4 "${extra[@]}"
    ;;
  *) echo "Unknown ABLATION_MODE: $ABLATION_MODE" >&2; exit 1 ;;
esac

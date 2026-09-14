#!/usr/bin/env bash
#SBATCH --job-name=qwen-prefill-figure5
#SBATCH --output=experiments/qwen_prefill_bootstrap_%j.out
#SBATCH --error=experiments/qwen_prefill_bootstrap_%j.err
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100-80:4
#SBATCH --time=24:00:00
set -euo pipefail
: "${SFS_ROOT:?Set the validated worktree}"
: "${QWEN_MANIFEST:?Set the canonical Qwen manifest}"
: "${QWEN_STAGE_DIR:?Set the completed calibration and smoke stage}"
: "${ROUTEBALANCE_PREDICTOR_DIR:?Set the native predictor}"
: "${PREFILL_CANDIDATE:?Set the corrected prefill artifact}"
: "${PREFILL_CANDIDATE_AUDIT:?Set the CPU candidate audit}"
: "${PREFILL_TIMING_REVIEW:?Set the recorded timing review}"
: "${BOOTSTRAP_VALIDATION_REPORT:?Set the CPU validation report}"
source "$SFS_ROOT/src/slurm/prep/paper_ablation_env.sh"
export PYTHONPATH="$SFS_ROOT/vllm:$PYTHONPATH"
: "${QWEN_CPU_VALIDATION:?Set the current source-bound CPU validation report}"
python - "$BOOTSTRAP_VALIDATION_REPORT" <<'PY'
import hashlib, json
from pathlib import Path
import sys
report = json.loads(Path(sys.argv[1]).read_text())
if report.get('status') != 'PASS_CPU':
    raise ValueError('Bootstrap CPU validation has not passed')
for name, expected in report['file_sha256'].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
        raise ValueError(f'CPU-validated input changed: {name}')
PY
python -m scripts.runs.qwen_baselines validate --manifest "$QWEN_MANIFEST" \
  --predictor "$ROUTEBALANCE_PREDICTOR_DIR" --stage-dir "$QWEN_STAGE_DIR" \
  --cpu-validation "$QWEN_CPU_VALIDATION"
args=(--manifest "$QWEN_MANIFEST" --stage-dir "$QWEN_STAGE_DIR"
  --predictor "$ROUTEBALANCE_PREDICTOR_DIR" --candidate "$PREFILL_CANDIDATE"
  --candidate-audit "$PREFILL_CANDIDATE_AUDIT" --timing-review "$PREFILL_TIMING_REVIEW"
  --cpu-validation "$QWEN_CPU_VALIDATION")
if [[ -n "${QWEN_POLICIES:-}" ]]; then
  read -r -a policies <<< "$QWEN_POLICIES"
  args+=(--policies "${policies[@]}")
fi
if [[ -n "${QWEN_QPS:-}" ]]; then
  read -r -a rates <<< "$QWEN_QPS"
  args+=(--qps-values "${rates[@]}")
fi
python -m scripts.runs.qwen_prefill_bootstrap validate "${args[@]}"
if [[ "${VALIDATE_ONLY:-0}" == 1 ]]; then exit 0; fi
: "${OUTPUT_ROOT:?Set a fresh output root}"
python -m scripts.runs.qwen_prefill_bootstrap run "${args[@]}" --output-root "$OUTPUT_ROOT"

#!/usr/bin/env bash
#SBATCH --job-name=ministral-prefill-figure5
#SBATCH --output=experiments/ministral3_prefill_bootstrap_%j.out
#SBATCH --error=experiments/ministral3_prefill_bootstrap_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=GPU-shared
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100-80:3

set -euo pipefail
: "${SFS_ROOT:?Set the validated worktree}"
: "${PREFILL_SOURCE_STAGE:?Set the completed measured scout stage}"
: "${PREFILL_CANDIDATE:?Set the CPU-validated timing correction}"
: "${PREFILL_CANDIDATE_AUDIT:?Set its CPU audit}"
: "${PREFILL_TIMING_REVIEW:?Set the recorded residual review}"
: "${FIGURE5_MANIFEST:?Set a new final manifest path}"
: "${PREDICTOR_RUN_DIR:?Set the validated shared predictor run}"
: "${BOOTSTRAP_VALIDATION_REPORT:?Set the CPU bootstrap validation report}"
source "$SFS_ROOT/src/slurm/prep/paper_ablation_env.sh"
export PYTHONPATH="$SFS_ROOT/vllm:$PYTHONPATH"
source "$SFS_ROOT/src/slurm/runs/ministral3_router_common.sh"
ministral3_resolve_sfs_root
python - "$BOOTSTRAP_VALIDATION_REPORT" <<'PY'
import hashlib, json
from pathlib import Path
import sys
r = json.loads(Path(sys.argv[1]).read_text())
if r.get('status') != 'PASS_CPU':
    raise ValueError('Bootstrap CPU validation has not passed')
for name, expected in r['file_sha256'].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
        raise ValueError(f'CPU-validated input changed: {name}')
PY
common_args=(--stage-dir "$PREFILL_SOURCE_STAGE" --candidate "$PREFILL_CANDIDATE"
  --candidate-audit "$PREFILL_CANDIDATE_AUDIT" --timing-review "$PREFILL_TIMING_REVIEW"
  --manifest "$FIGURE5_MANIFEST")
if [[ -n "${PREFILL_SOURCE_REVIEW:-}" ]]; then
  : "${ROUTEBALANCE_PREDICTOR_DIR:?Full policy refresh requires the native predictor}"
  common_args+=(--source-review "$PREFILL_SOURCE_REVIEW" --routebalance-predictor "$ROUTEBALANCE_PREDICTOR_DIR")
fi
if [[ "${BOOTSTRAP_SMOKE_ONLY:-0}" == 1 ]]; then
  common_args+=(--smoke-only)
fi
python -m scripts.runs.ministral3_prefill_bootstrap validate "${common_args[@]}"
if [[ "${VALIDATE_ONLY:-0}" == 1 ]]; then exit 0; fi
MINISTRAL_RUN_ROOT="${RUN_ROOT_OVERRIDE:-$SFS_ROOT/experiments/ministral3_paper/figure_sweeps/prefill_baseline_${SLURM_JOB_ID:?}}"
if [[ -e "$MINISTRAL_RUN_ROOT" ]]; then
  echo "[ERROR] Refusing to overwrite $MINISTRAL_RUN_ROOT" >&2
  exit 1
fi
mkdir -p "$MINISTRAL_RUN_ROOT"
MINISTRAL_RUN_ROOT="$(realpath "$MINISTRAL_RUN_ROOT")"
cp "$BOOTSTRAP_VALIDATION_REPORT" "$MINISTRAL_RUN_ROOT/bootstrap_cpu_validation.json"
tar -C "$SFS_ROOT" --exclude='__pycache__' --exclude='*.pyc' \
  -czf "$MINISTRAL_RUN_ROOT/source_snapshot.tar.gz" src \
  vllm/vllm/v1/engine/async_llm.py vllm/vllm/v1/core/sched/scheduler.py \
  vllm/vllm/v1/core/sched/state_snapshot.py
ministral3_start_router_pool
python -m scripts.runs.ministral3_prefill_bootstrap run "${common_args[@]}" \
  --output-root "$MINISTRAL_RUN_ROOT" --instances-config "$MINISTRAL_INSTANCES_CONFIG" \
  --service-metrics-json "$MINISTRAL_SERVICE_METRICS" \
  --wait-log "$MINISTRAL_WAIT_LOG_3B" --wait-log "$MINISTRAL_WAIT_LOG_8B" \
  --wait-log "$MINISTRAL_WAIT_LOG_14B" >"$MINISTRAL_RUN_ROOT/bootstrap_driver.log" 2>&1

#!/bin/bash
#SBATCH --job-name=ministral3-batch-fit
#SBATCH --output=experiments/ministral3_batch_fit_%j.out
#SBATCH --error=experiments/ministral3_batch_fit_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=RM-shared
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=BEGIN,FAIL,END

set -euo pipefail

if [[ -n "${SFS_ROOT:-}" && -d "$SFS_ROOT/src/slurm" ]]; then
  SFS_ROOT="$(cd "$SFS_ROOT" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/src/slurm" ]]; then
  SFS_ROOT="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  SFS_ROOT="$SCRIPT_DIR"
  while [[ "$SFS_ROOT" != "/" && ! -d "$SFS_ROOT/src/slurm" ]]; do
    SFS_ROOT="$(dirname "$SFS_ROOT")"
  done
  if [[ "$SFS_ROOT" == "/" ]]; then
    echo "[ERROR] Could not resolve SFS repo root." >&2
    exit 1
  fi
fi

SERVICE_JOB_ID="${SERVICE_JOB_ID:-}"
SERVICE_RUN_DIR="${SERVICE_RUN_DIR:-}"
VALIDATE_ONLY="${VALIDATE_ONLY:-0}"
if [[ "$VALIDATE_ONLY" != "0" && "$VALIDATE_ONLY" != "1" ]]; then
  echo "[ERROR] VALIDATE_ONLY must be 0 or 1." >&2
  exit 1
fi
if [[ -z "$SERVICE_RUN_DIR" && -z "$SERVICE_JOB_ID" ]]; then
  echo "[ERROR] Set SERVICE_RUN_DIR or SERVICE_JOB_ID." >&2
  exit 1
fi

if [[ -z "$SERVICE_RUN_DIR" ]]; then
  shopt -s nullglob
  matches=("$SFS_ROOT"/experiments/ministral3_h100-80_service_metrics_"$SERVICE_JOB_ID"_*)
  shopt -u nullglob
  directories=()
  for path in "${matches[@]}"; do
    [[ -d "$path" ]] && directories+=("$path")
  done
  if (( ${#directories[@]} != 1 )); then
    echo "[ERROR] Expected one service directory for job $SERVICE_JOB_ID; found ${#directories[@]}." >&2
    exit 1
  fi
  SERVICE_RUN_DIR="${directories[0]}"
fi
SERVICE_RUN_DIR="$(cd "$SERVICE_RUN_DIR" && pwd)"
METRICS_JSON="$SERVICE_RUN_DIR/model_metrics.json"

MODELS=(ministral3-3b ministral3-8b ministral3-14b)
for model in "${MODELS[@]}"; do
  path="$SERVICE_RUN_DIR/batch_stats_${model}.csv"
  if [[ ! -s "$path" ]]; then
    echo "[ERROR] Missing batch statistics: $path" >&2
    exit 1
  fi
done
if [[ ! -s "$METRICS_JSON" ]]; then
  echo "[ERROR] Missing service metrics: $METRICS_JSON" >&2
  exit 1
fi

RUN_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SFS_ROOT/experiments/ministral3_paper/figure3_batch_fit/run_$RUN_ID}"
if [[ "$VALIDATE_ONLY" == "1" ]]; then
  echo "service_run_dir=$SERVICE_RUN_DIR"
  echo "metrics_json=$METRICS_JSON"
  echo "output_root=$OUTPUT_ROOT"
  echo "models=${MODELS[*]} feature_set=cross_term nonnegative=true split=80/20 seed=69"
  exit 0
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "[ERROR] Refusing to overwrite batch-fit output: $OUTPUT_ROOT" >&2
  exit 1
fi
mkdir -p "$OUTPUT_ROOT/splits"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm
export PYTHONPATH="$SFS_ROOT/src:${PYTHONPATH:-}"

python -m scripts.runs.service_metrics_config \
  --path "$METRICS_JSON" \
  --expected-feature-set cross_term \
  --require-nonnegative-sfs \
  --model-key ministral3-3b \
  --model-key ministral3-8b \
  --model-key ministral3-14b \
  validate >"$OUTPUT_ROOT/service_metrics_validation.json"

python -m scripts.prep.split_csv \
  --root "$SERVICE_RUN_DIR" \
  --glob 'batch_stats_ministral3-*.csv' \
  --seed 69 \
  --train-fraction 0.8 \
  --output-dir "$OUTPUT_ROOT/splits" \
  >"$OUTPUT_ROOT/split.log"

for model in "${MODELS[@]}"; do
  model_output="$OUTPUT_ROOT/$model"
  mkdir -p "$model_output"
  python -m scripts.runs.experiments \
    --experiment batch_fit \
    --batch-train-csv "$OUTPUT_ROOT/splits/batch_stats_${model}_train.csv" \
    --batch-test-csv "$OUTPUT_ROOT/splits/batch_stats_${model}_test.csv" \
    --batch-fit-output-dir "$model_output" \
    --batch-fit-feature-set cross_term \
    --batch-fit-nonnegative \
    --output-path "$model_output/result.json" \
    >"$model_output/run.log" 2>&1
done

python - "$OUTPUT_ROOT" "$SERVICE_RUN_DIR" "$METRICS_JSON" <<'PY'
import json
import os
from pathlib import Path
import subprocess
import sys

output_root = Path(sys.argv[1])
models = ("ministral3-3b", "ministral3-8b", "ministral3-14b")
summary = {}
for model in models:
    result = json.loads((output_root / model / "result.json").read_text(encoding="utf-8"))
    batch_fit = result["batch_fit"]
    if batch_fit["evaluation"] != "heldout":
        raise RuntimeError(f"{model}: batch fit is not held out")
    params = batch_fit["fit_params"]
    if params["feature_set"] != "cross_term" or not params["nonnegative_coefficients"]:
        raise RuntimeError(f"{model}: incorrect fit configuration")
    metrics = batch_fit["test"]["aggregate"]["goodness_of_fit"]["typical"]
    if int(metrics["count"]) <= 0:
        raise RuntimeError(f"{model}: no held-out fit rows")
    summary[model] = metrics

payload = {
    "status": "PASS",
    "service_run_dir": str(Path(sys.argv[2]).resolve()),
    "service_metrics_json": str(Path(sys.argv[3]).resolve()),
    "split_seed": 69,
    "train_fraction": 0.8,
    "feature_set": "cross_term",
    "nonnegative_coefficients": True,
    "heldout_metrics": summary,
    "sfs_commit": subprocess.check_output(
        ["git", "-C", os.environ["SFS_ROOT"], "rev-parse", "HEAD"], text=True
    ).strip(),
}
(output_root / "audit.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
PY

find "$OUTPUT_ROOT" -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum >"$OUTPUT_ROOT/SHA256SUMS"
echo "[DONE] Held-out Ministral Figure 3 artifacts: $OUTPUT_ROOT"

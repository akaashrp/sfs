#!/bin/bash
#SBATCH --job-name=ministral3-predictors
#SBATCH --output=experiments/ministral3_predictors_%j.out
#SBATCH --error=experiments/ministral3_predictors_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=GPU-shared
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:v100-32:1
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

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SFS_ROOT/.." && pwd)}"
export SFS_ROOT
CALIBRATION_COMPLETIONS_ROOT="${CALIBRATION_COMPLETIONS_ROOT:-$PROJECT_ROOT/sfs_artifacts/ministral3_calibration_44849564_judge_45009121_20260902/run/completions}"
MODEL_METADATA="${MODEL_METADATA:-$SFS_ROOT/src/assets/model_metadata/ministral3.json}"
PREDICTOR_ROOT="${PREDICTOR_ROOT:-$SFS_ROOT/experiments/ministral3_paper/predictors}"
RUN_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$PREDICTOR_ROOT/run_${RUN_ID}"
VALIDATE_ONLY="${VALIDATE_ONLY:-0}"

if [[ "$VALIDATE_ONLY" != "0" && "$VALIDATE_ONLY" != "1" ]]; then
  echo "[ERROR] VALIDATE_ONLY must be 0 or 1." >&2
  exit 1
fi
if [[ ! -f "$MODEL_METADATA" ]]; then
  echo "[ERROR] Missing model metadata: $MODEL_METADATA" >&2
  exit 1
fi

MODELS=(ministral3-3b ministral3-8b ministral3-14b)
BUCKETS=(alpaca govreport-summarization hotpot_qa writingprompts)
SCORED_INPUTS=()
for model in "${MODELS[@]}"; do
  for bucket in "${BUCKETS[@]}"; do
    path="$CALIBRATION_COMPLETIONS_ROOT/$model/${bucket}_scored.jsonl"
    if [[ ! -f "$path" ]]; then
      echo "[ERROR] Missing scored calibration input: $path" >&2
      exit 1
    fi
    SCORED_INPUTS+=("$path")
  done
done

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm
export PYTHONPATH="$SFS_ROOT/src:${PYTHONPATH:-}"

python - "$MODEL_METADATA" "${SCORED_INPUTS[@]}" <<'PY'
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

metadata_path = Path(sys.argv[1])
inputs = [Path(value) for value in sys.argv[2:]]
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
expected_served_model_by_dir = {
    "ministral3-3b": "ministral3-3b-instruct",
    "ministral3-8b": "ministral3-8b-instruct",
    "ministral3-14b": "ministral3-14b-instruct",
}
counts = Counter()
prompt_sets = defaultdict(set)
for path in inputs:
    model_label = path.parent.name
    expected_served_model = expected_served_model_by_dir[model_label]
    if model_label not in metadata:
        raise RuntimeError(f"Missing descriptor for model label {model_label}")
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            if record.get("model_label") != model_label:
                raise RuntimeError(
                    f"{path}:{line_number}: model_label={record.get('model_label')!r}; "
                    f"expected {model_label!r}"
                )
            if record.get("model_id") != expected_served_model:
                raise RuntimeError(
                    f"{path}:{line_number}: model_id={record.get('model_id')!r}; "
                    f"expected {expected_served_model!r}"
                )
            quality = record.get("quality")
            if not isinstance(quality, (int, float)):
                raise RuntimeError(f"{path}:{line_number}: missing numeric quality")
            if not isinstance(record.get("prompt_tokens"), int) or record["prompt_tokens"] <= 0:
                raise RuntimeError(f"{path}:{line_number}: invalid prompt_tokens")
            example_id = (record.get("prompt_metadata") or {}).get("example_id")
            if not example_id:
                raise RuntimeError(f"{path}:{line_number}: missing example_id")
            counts[path] += 1
            prompt_sets[path.parent.name].add(example_id)
for path, count in counts.items():
    if count != 2500:
        raise RuntimeError(f"{path}: expected 2500 records, found {count}")
if len(set(map(frozenset, prompt_sets.values()))) != 1:
    raise RuntimeError("The three model directories do not contain aligned prompts")
print(f"Validated {sum(counts.values())} scored records across {len(inputs)} files")
PY

if [[ "$VALIDATE_ONLY" == "1" ]]; then
  echo "[VALID] calibration_root=$CALIBRATION_COMPLETIONS_ROOT"
  echo "[VALID] model_metadata=$MODEL_METADATA"
  echo "[VALID] output_dir=$RUN_DIR"
  exit 0
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "[ERROR] Refusing to overwrite predictor run: $RUN_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR/output_length_predictor" "$RUN_DIR/accuracy_predictor"

python -m sfs_core.predictors.output_length_model \
  --input "${SCORED_INPUTS[@]}" \
  --output-dir "$RUN_DIR/output_length_predictor" \
  --test-fraction 0.1 \
  --split-seed 69 \
  >"$RUN_DIR/output_length_training.log" 2>&1

python -m sfs_core.predictors.accuracy_model \
  --input "${SCORED_INPUTS[@]}" \
  --output-dir "$RUN_DIR/accuracy_predictor" \
  --model-metadata "$MODEL_METADATA" \
  --test-fraction 0.1 \
  --split-seed 69 \
  --no-balance-quality-metrics \
  >"$RUN_DIR/accuracy_training.log" 2>&1

python - "$RUN_DIR" "$MODEL_METADATA" <<'PY'
import json
import math
from pathlib import Path
import sys

run_dir = Path(sys.argv[1])
source_descriptors = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
expected_models = set(source_descriptors)
expected_lookup_keys = expected_models | {"__unknown__"}

accuracy_dir = run_dir / "accuracy_predictor"
length_dir = run_dir / "output_length_predictor"
accuracy_metadata = json.loads(
    (accuracy_dir / "metadata.json").read_text(encoding="utf-8")
)
length_metadata = json.loads(
    (length_dir / "metadata.json").read_text(encoding="utf-8")
)
accuracy_summary = json.loads(
    (accuracy_dir / "training_summary.json").read_text(encoding="utf-8")
)
length_summary = json.loads(
    (length_dir / "training_summary.json").read_text(encoding="utf-8")
)

for name, lookup in (
    ("accuracy", accuracy_metadata["model_id_lookup"]),
    ("output length", length_metadata["model_id_lookup"]),
):
    if set(lookup) != expected_lookup_keys:
        raise RuntimeError(
            f"{name} model lookup keys {sorted(lookup)} do not match "
            f"{sorted(expected_lookup_keys)}"
        )

trained_descriptors = accuracy_metadata["model_descriptors"]
if set(trained_descriptors) != expected_models | {"__default__"}:
    raise RuntimeError(
        "Accuracy descriptor keys do not match the model labels used for training: "
        f"{sorted(trained_descriptors)} versus {sorted(expected_models)}"
    )
for model, source in source_descriptors.items():
    trained = trained_descriptors[model]
    expected_log_params = math.log(float(source["num_params"]))
    if not math.isclose(
        float(trained["log_num_params"]), expected_log_params, rel_tol=0, abs_tol=1e-12
    ):
        raise RuntimeError(f"Incorrect log_num_params descriptor for {model}")
    if int(trained["max_context_length"]) != int(source["max_context_length"]):
        raise RuntimeError(f"Incorrect max_context_length descriptor for {model}")

for name, summary, model_field in (
    ("accuracy", accuracy_summary, "models"),
    ("output length", length_summary, "model_ids"),
):
    if summary["total_examples"] != 30_000:
        raise RuntimeError(f"{name} total_examples is not 30000")
    if summary["train_examples"] != 27_000 or summary["test_examples"] != 3_000:
        raise RuntimeError(f"{name} train/test split is not 27000/3000")
    if set(summary[model_field]) != expected_models:
        raise RuntimeError(f"{name} summary model labels do not match metadata")

accuracy_test_ids = json.loads(
    (accuracy_dir / accuracy_metadata["test_example_ids_file"]).read_text(
        encoding="utf-8"
    )
)
length_test_ids = json.loads(
    (length_dir / length_metadata["test_example_ids_file"]).read_text(
        encoding="utf-8"
    )
)
if accuracy_test_ids != length_test_ids:
    raise RuntimeError("Accuracy and output-length predictors used different test rows")

audit = {
    "status": "PASS",
    "models": sorted(expected_models),
    "total_examples": 30_000,
    "train_examples": 27_000,
    "test_examples": 3_000,
    "test_example_ids_aligned": True,
    "accuracy_descriptor_labels_aligned": True,
}
(run_dir / "predictor_audit.json").write_text(
    json.dumps(audit, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(audit, sort_keys=True))
PY

python - "$RUN_DIR" "$CALIBRATION_COMPLETIONS_ROOT" "$MODEL_METADATA" <<'PY'
import json
import os
from pathlib import Path
import subprocess
import sys

run_dir = Path(sys.argv[1])
payload = {
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "sfs_commit": subprocess.check_output(
        ["git", "-C", os.environ["SFS_ROOT"], "rev-parse", "HEAD"], text=True
    ).strip(),
    "calibration_completions_root": str(Path(sys.argv[2]).resolve()),
    "model_metadata": str(Path(sys.argv[3]).resolve()),
    "test_fraction": 0.1,
    "split_seed": 69,
    "accuracy_quality_balancing": False,
}
(run_dir / "provenance.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
PY

find "$RUN_DIR" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >"$RUN_DIR/SHA256SUMS"
echo "[DONE] Ministral predictor training completed: $RUN_DIR"

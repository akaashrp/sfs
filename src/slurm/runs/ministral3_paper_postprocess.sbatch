#!/bin/bash
#SBATCH --job-name=ministral3-paper-plots
#SBATCH --output=experiments/ministral3_paper_postprocess_%j.out
#SBATCH --error=experiments/ministral3_paper_postprocess_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=RM-shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
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
    echo "[ERROR] Could not resolve SFS repository root." >&2
    exit 1
  fi
fi

QPS_SNAPSHOT_JOB_ID="${QPS_SNAPSHOT_JOB_ID:-}"
QPS_BASELINE_JOB_ID="${QPS_BASELINE_JOB_ID:-}"
DELTA_JOB_ID="${DELTA_JOB_ID:-}"
ARRIVAL_SNAPSHOT_JOB_ID="${ARRIVAL_SNAPSHOT_JOB_ID:-}"
ARRIVAL_BASELINE_JOB_ID="${ARRIVAL_BASELINE_JOB_ID:-}"
JUDGE_JOB_ID="${JUDGE_JOB_ID:-}"
VALIDATE_ONLY="${VALIDATE_ONLY:-0}"

ID_VARIABLES=(
  QPS_SNAPSHOT_JOB_ID
  QPS_BASELINE_JOB_ID
  DELTA_JOB_ID
  ARRIVAL_SNAPSHOT_JOB_ID
  ARRIVAL_BASELINE_JOB_ID
  JUDGE_JOB_ID
)
for variable_name in "${ID_VARIABLES[@]}"; do
  if ! [[ "${!variable_name}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] $variable_name must be a positive Slurm job ID; got '${!variable_name}'." >&2
    exit 1
  fi
done
if [[ "$VALIDATE_ONLY" != "0" && "$VALIDATE_ONLY" != "1" ]]; then
  echo "[ERROR] VALIDATE_ONLY must be 0 or 1." >&2
  exit 1
fi

SWEEP_ROOT="$SFS_ROOT/experiments/ministral3_paper/figure_sweeps"
QPS_SNAPSHOT_RAW="$SWEEP_ROOT/qps_snapshot_run_$QPS_SNAPSHOT_JOB_ID"
QPS_BASELINE_RAW="$SWEEP_ROOT/qps_baseline_run_$QPS_BASELINE_JOB_ID"
DELTA_RAW="$SWEEP_ROOT/delta_combined_run_$DELTA_JOB_ID"
ARRIVAL_SNAPSHOT_RAW="$SWEEP_ROOT/qps_arrival_snapshot_run_$ARRIVAL_SNAPSHOT_JOB_ID"
ARRIVAL_BASELINE_RAW="$SWEEP_ROOT/qps_arrival_baseline_run_$ARRIVAL_BASELINE_JOB_ID"
SCORED_ROOT="${SCORED_ROOT:-$SFS_ROOT/experiments/ministral3_paper/holdout_4000_generation/completions}"
QPS_REQ_MAP="${QPS_REQ_MAP:-$SFS_ROOT/experiments/ministral3_paper/request_maps/req_map_qps_seed69_holdout4000_n16000.csv}"
DELTA_REQ_MAP="${DELTA_REQ_MAP:-$SFS_ROOT/experiments/ministral3_paper/request_maps/req_map_delta_seed69_holdout2000_n8000.csv}"
RUN_ID="${SLURM_JOB_ID:-local}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SFS_ROOT/experiments/ministral3_paper/paper_figures/run_$RUN_ID}"

if [[ "$VALIDATE_ONLY" == "1" ]]; then
  echo "qps_snapshot_raw=$QPS_SNAPSHOT_RAW"
  echo "qps_baseline_raw=$QPS_BASELINE_RAW"
  echo "delta_raw=$DELTA_RAW"
  echo "arrival_snapshot_raw=$ARRIVAL_SNAPSHOT_RAW"
  echo "arrival_baseline_raw=$ARRIVAL_BASELINE_RAW"
  echo "scored_root=$SCORED_ROOT"
  echo "output_root=$OUTPUT_ROOT"
  echo "figures=5,6,7 raw_sweeps_immutable=true"
  exit 0
fi

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "[ERROR] Refusing to overwrite paper-figure output: $OUTPUT_ROOT" >&2
  exit 1
fi
mkdir -p "$OUTPUT_ROOT/checks" "$OUTPUT_ROOT/derived_sweeps"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm
export PYTHONPATH="$SFS_ROOT/src:${PYTHONPATH:-}"

RAW_RUNS=(
  "$QPS_SNAPSHOT_RAW:qps:snapshot"
  "$QPS_BASELINE_RAW:qps:baseline"
  "$DELTA_RAW:delta:combined"
  "$ARRIVAL_SNAPSHOT_RAW:qps_arrival:snapshot"
  "$ARRIVAL_BASELINE_RAW:qps_arrival:baseline"
)
for specification in "${RAW_RUNS[@]}"; do
  IFS=: read -r run_root sweep_kind utility_group <<< "$specification"
  label="${sweep_kind}_${utility_group}"
  for required_path in "$run_root/audit.json" "$run_root/SHA256SUMS" "$run_root/outputs"; do
    if [[ ! -e "$required_path" ]]; then
      echo "[ERROR] Missing raw sweep artifact: $required_path" >&2
      exit 1
    fi
  done
  (cd "$run_root" && sha256sum -c SHA256SUMS) \
    >"$OUTPUT_ROOT/checks/${label}_raw_before.log"
  python - "$run_root/audit.json" "$sweep_kind" "$utility_group" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
expected_kind, expected_group = sys.argv[2:4]
if payload.get("status") != "PASS":
    raise RuntimeError(f"Raw sweep audit did not pass: {path}")
if payload.get("sweep_kind") != expected_kind:
    raise RuntimeError(f"Unexpected sweep kind in {path}")
if payload.get("utility_group") != expected_group:
    raise RuntimeError(f"Unexpected utility group in {path}")
if not payload.get("zero_request_errors"):
    raise RuntimeError(f"Raw sweep has request errors: {path}")
if not payload.get("complete_system_entry_e2e_ttft"):
    raise RuntimeError(f"Raw sweep has incomplete TTFT telemetry: {path}")
PY
done

for required_path in "$QPS_REQ_MAP" "$DELTA_REQ_MAP"; do
  if [[ ! -s "$required_path" ]]; then
    echo "[ERROR] Missing request map: $required_path" >&2
    exit 1
  fi
done
JUDGE_SUMMARY="$SCORED_ROOT/judge_run_summary_$JUDGE_JOB_ID.json"
if [[ ! -s "$JUDGE_SUMMARY" ]]; then
  echo "[ERROR] Missing judge summary: $JUDGE_SUMMARY" >&2
  exit 1
fi

python - "$SCORED_ROOT" "$JUDGE_SUMMARY" "$OUTPUT_ROOT/checks/judge_audit.json" <<'PY'
import json
import math
from pathlib import Path
import sys

root, summary_path, output_path = map(Path, sys.argv[1:])
models = ("ministral3-3b", "ministral3-8b", "ministral3-14b")
buckets = ("alpaca", "govreport-summarization", "hotpot_qa", "writingprompts")
expected_per_file = 4000
ids_by_bucket = {}
files = {}
imputed = 0
for model in models:
    files[model] = {}
    for bucket in buckets:
        path = root / model / f"{bucket}_scored.jsonl"
        if not path.is_file():
            raise RuntimeError(f"Missing scored file: {path}")
        ids = []
        count = 0
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                quality = row.get("quality")
                if not isinstance(quality, (int, float)) or not math.isfinite(float(quality)):
                    raise RuntimeError(f"Invalid quality at {path}:{line_number}")
                if not 0.0 <= float(quality) <= 1.0:
                    raise RuntimeError(f"Out-of-range quality at {path}:{line_number}")
                metadata = row.get("prompt_metadata")
                example_id = metadata.get("example_id") if isinstance(metadata, dict) else None
                if not isinstance(example_id, str) or not example_id:
                    raise RuntimeError(f"Missing example_id at {path}:{line_number}")
                ids.append(example_id)
                count += 1
                imputed += int(row.get("quality_imputed") is True)
        if count != expected_per_file or len(set(ids)) != expected_per_file:
            raise RuntimeError(f"Expected {expected_per_file} unique scores in {path}; found {count}")
        expected_ids = ids_by_bucket.setdefault(bucket, ids)
        if ids != expected_ids:
            raise RuntimeError(f"Scored example ordering differs for {model}/{bucket}")
        files[model][bucket] = count

summary = json.loads(summary_path.read_text(encoding="utf-8"))
if set(summary.get("models", [])) != set(models):
    raise RuntimeError("Judge summary model set is incorrect")
if set(summary.get("buckets", {})) != {f"{bucket}.jsonl" for bucket in buckets}:
    raise RuntimeError("Judge summary bucket set is incorrect")
payload = {
    "status": "PASS",
    "scored_root": str(root.resolve()),
    "judge_summary": str(summary_path.resolve()),
    "models": list(models),
    "buckets": list(buckets),
    "prompt_groups": len(buckets) * expected_per_file,
    "model_scores": len(models) * len(buckets) * expected_per_file,
    "imputed_model_scores": imputed,
    "files": files,
}
output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

declare -A COPY_SOURCES=(
  [qps_snapshot]="$QPS_SNAPSHOT_RAW/outputs"
  [qps_baseline]="$QPS_BASELINE_RAW/outputs"
  [delta_combined]="$DELTA_RAW/outputs"
  [qps_arrival_snapshot]="$ARRIVAL_SNAPSHOT_RAW/outputs"
  [qps_arrival_baseline]="$ARRIVAL_BASELINE_RAW/outputs"
)
for label in qps_snapshot qps_baseline delta_combined qps_arrival_snapshot qps_arrival_baseline; do
  destination="$OUTPUT_ROOT/derived_sweeps/$label"
  mkdir -p "$destination"
  cp -al "${COPY_SOURCES[$label]}/." "$destination/"
done

DERIVED_DIRS=(
  "$OUTPUT_ROOT/derived_sweeps/qps_snapshot"
  "$OUTPUT_ROOT/derived_sweeps/qps_baseline"
  "$OUTPUT_ROOT/derived_sweeps/delta_combined"
  "$OUTPUT_ROOT/derived_sweeps/qps_arrival_snapshot"
  "$OUTPUT_ROOT/derived_sweeps/qps_arrival_baseline"
)
python -m scripts.eval.augment_router_actual_accuracy \
  "${DERIVED_DIRS[@]}" \
  --delta-req-map "$DELTA_REQ_MAP" \
  --qps-req-map "$QPS_REQ_MAP" \
  --scored-root "$SCORED_ROOT" \
  | tee "$OUTPUT_ROOT/checks/accuracy_augmentation.log"

# Atomic replacement in the augmentation tool breaks the copied hardlink. A
# second checksum pass proves the original raw runs were not altered.
for specification in "${RAW_RUNS[@]}"; do
  IFS=: read -r run_root sweep_kind utility_group <<< "$specification"
  label="${sweep_kind}_${utility_group}"
  (cd "$run_root" && sha256sum -c SHA256SUMS) \
    >"$OUTPUT_ROOT/checks/${label}_raw_after.log"
done

FIGURE5_DIR="$OUTPUT_ROOT/figure5_offered_load"
FIGURE6_DIR="$OUTPUT_ROOT/figure6_latency_tradeoff"
FIGURE7_DIR="$OUTPUT_ROOT/figure7_arrival_processes"
mkdir -p "$FIGURE5_DIR" "$FIGURE6_DIR" "$FIGURE7_DIR/poisson_inputs"

python -m scripts.reporting.router_qps_sweep_summary \
  "$OUTPUT_ROOT/derived_sweeps/qps_snapshot" \
  "$OUTPUT_ROOT/derived_sweeps/qps_baseline" \
  --output-dir "$FIGURE5_DIR"

DELTA_QPS="$(tr -d '[:space:]' < "$DELTA_RAW/qps_values.txt")"
python -m scripts.reporting.router_delta_sweep_summary \
  "$OUTPUT_ROOT/derived_sweeps/delta_combined" \
  --output-dir "$FIGURE6_DIR" \
  --request-rate-qps "$DELTA_QPS" \
  --utilities soft latency_agnostic round_robin shortest_queue \
  --duplicate-policy error
python -m scripts.reporting.plot_router_delta_sweep \
  --summary-json "$FIGURE6_DIR/router_delta_sweep_metrics_summary.json" \
  --output-dir "$FIGURE6_DIR" \
  --sweep-utilities soft

python - \
  "$OUTPUT_ROOT/derived_sweeps/qps_snapshot" \
  "$OUTPUT_ROOT/derived_sweeps/qps_baseline" \
  "$ARRIVAL_SNAPSHOT_RAW/qps_values.txt" \
  "$FIGURE7_DIR/poisson_inputs" <<'PY'
import json
import math
import os
from pathlib import Path
import sys

snapshot_dir, baseline_dir, qps_path, output_root = map(Path, sys.argv[1:])
targets = [float(value) for value in qps_path.read_text(encoding="utf-8").split()]
if len(targets) != 3:
    raise RuntimeError(f"Expected three Figure 7 QPS values; found {targets}")
selection = {}
for label, source_dir in (("snapshot", snapshot_dir), ("baseline", baseline_dir)):
    destination_dir = output_root / label
    destination_dir.mkdir(parents=True, exist_ok=False)
    selected = []
    for source in sorted(source_dir.glob("*.json")):
        if source.name.endswith("manifest.json"):
            continue
        payload = json.loads(source.read_text(encoding="utf-8"))
        qps = payload.get("config", {}).get("request_rate_qps")
        if isinstance(qps, (int, float)) and any(
            math.isclose(float(qps), target, rel_tol=1e-12, abs_tol=1e-12)
            for target in targets
        ):
            destination = destination_dir / source.name
            os.link(source, destination)
            selected.append({"path": str(source.resolve()), "qps": float(qps)})
    if len(selected) != len(targets):
        raise RuntimeError(f"Expected {len(targets)} Poisson {label} files; found {len(selected)}")
    selection[label] = selected
(output_root / "selection.json").write_text(
    json.dumps({"target_qps": targets, "selected": selection}, indent=2) + "\n",
    encoding="utf-8",
)
PY

python -m scripts.reporting.router_arrival_sweep_summary \
  "$FIGURE7_DIR/poisson_inputs/snapshot" \
  "$FIGURE7_DIR/poisson_inputs/baseline" \
  "$OUTPUT_ROOT/derived_sweeps/qps_arrival_snapshot" \
  "$OUTPUT_ROOT/derived_sweeps/qps_arrival_baseline" \
  --output-dir "$FIGURE7_DIR"

python - "$OUTPUT_ROOT" "$SFS_ROOT" \
  "$QPS_SNAPSHOT_RAW" "$QPS_BASELINE_RAW" "$DELTA_RAW" \
  "$ARRIVAL_SNAPSHOT_RAW" "$ARRIVAL_BASELINE_RAW" "$SCORED_ROOT" <<'PY'
import json
import math
from pathlib import Path
import subprocess
import sys

output_root = Path(sys.argv[1])
sfs_root = Path(sys.argv[2])
raw_paths = [Path(value).resolve() for value in sys.argv[3:8]]
scored_root = Path(sys.argv[8]).resolve()
derived_root = output_root / "derived_sweeps"
expected_records = {
    "qps_snapshot": 13 * 2 * 16000,
    "qps_baseline": 13 * 2 * 16000,
    "delta_combined": (4 + 20) * 8000,
    "qps_arrival_snapshot": 3 * 2 * 2 * 16000,
    "qps_arrival_baseline": 3 * 2 * 2 * 16000,
}
record_counts = {}
for label, expected in expected_records.items():
    count = 0
    for path in sorted((derived_root / label).glob("*.json")):
        if path.name.endswith("manifest.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for run in payload.get("router", {}).get("runs", []):
            for record in run.get("per_request", []):
                value = record.get("actual_accuracy")
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise RuntimeError(f"Missing actual_accuracy in {path}")
                count += 1
    if count != expected:
        raise RuntimeError(f"{label}: expected {expected} augmented records; found {count}")
    record_counts[label] = count

figure5 = json.loads(
    (output_root / "figure5_offered_load/router_qps_sweep_summary.json").read_text(encoding="utf-8")
)
expected_utilities = {"hard", "shortest_queue", "latency_agnostic", "round_robin"}
if len(figure5.get("qps", {})) != 13 or set(figure5.get("utilities", [])) != expected_utilities:
    raise RuntimeError("Figure 5 summary does not contain the complete 13-point/four-policy matrix")
for qps, entries in figure5["qps"].items():
    for utility in expected_utilities:
        value = entries.get(utility, {}).get("actual_slo_gated_utility_mean")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise RuntimeError(f"Figure 5 missing OnTimeUtility for qps={qps}, utility={utility}")

figure6 = json.loads(
    (output_root / "figure6_latency_tradeoff/router_delta_sweep_metrics_summary.json").read_text(encoding="utf-8")
)
if len(figure6.get("deltas", {})) != 21:
    raise RuntimeError("Figure 6 summary does not contain 21 delta points")
if set(figure6["deltas"]["0"]["utilities"]) != {
    "soft", "latency_agnostic", "round_robin", "shortest_queue"
}:
    raise RuntimeError("Figure 6 delta=0 baselines are incomplete")
for delta, entry in figure6["deltas"].items():
    if delta != "0" and set(entry["utilities"]) != {"soft"}:
        raise RuntimeError(f"Figure 6 delta={delta} has an unexpected utility set")

figure7 = json.loads(
    (output_root / "figure7_arrival_processes/router_arrival_sweep_summary.json").read_text(encoding="utf-8")
)
if figure7.get("arrival_distributions") != ["poisson", "mmpp2_r3", "mmpp2_r6"]:
    raise RuntimeError("Figure 7 arrival-process set/order is incorrect")
if len(figure7.get("qps_keys", [])) != 3 or figure7.get("missing"):
    raise RuntimeError("Figure 7 does not contain a complete three-rate matrix")
for arrival in figure7["arrival_distributions"]:
    for qps in figure7["qps_keys"]:
        for utility in expected_utilities:
            value = figure7["data"][arrival][qps][utility].get(
                "actual_slo_gated_utility_mean"
            )
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise RuntimeError(
                    f"Figure 7 missing OnTimeUtility for {arrival}/{qps}/{utility}"
                )

plots = [
    output_root / "figure5_offered_load/actual_slo_gated_utility_mean_vs_qps.png",
    output_root / "figure6_latency_tradeoff/utility_vs_system_entry_e2e_ttft_ms_slo_attainment.png",
    output_root / "figure6_latency_tradeoff/utility_vs_average_system_entry_e2e_ttft_ms.png",
    output_root / "figure7_arrival_processes/actual_slo_gated_utility_mean_vs_qps.png",
]
for path in plots:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing plot: {path}")

payload = {
    "status": "PASS",
    "sfs_commit": subprocess.check_output(
        ["git", "-C", str(sfs_root), "rev-parse", "HEAD"], text=True
    ).strip(),
    "raw_sweep_roots": [str(path) for path in raw_paths],
    "scored_root": str(scored_root),
    "raw_sweeps_modified": False,
    "augmented_record_counts": record_counts,
    "augmented_records_total": sum(record_counts.values()),
    "figure5_qps_points": 13,
    "figure6_delta_points": 21,
    "figure7_qps_points": 3,
    "figure7_arrival_processes": ["poisson", "mmpp2_r3", "mmpp2_r6"],
    "plots": [str(path.resolve()) for path in plots],
}
(output_root / "audit.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
PY

python - "$OUTPUT_ROOT" "$QPS_SNAPSHOT_JOB_ID" "$QPS_BASELINE_JOB_ID" \
  "$DELTA_JOB_ID" "$ARRIVAL_SNAPSHOT_JOB_ID" "$ARRIVAL_BASELINE_JOB_ID" \
  "$JUDGE_JOB_ID" <<'PY'
import json
from pathlib import Path
import sys

output_root = Path(sys.argv[1])
payload = {
    "figure_5": str((output_root / "figure5_offered_load").resolve()),
    "figure_6": str((output_root / "figure6_latency_tradeoff").resolve()),
    "figure_7": str((output_root / "figure7_arrival_processes").resolve()),
    "source_jobs": {
        "qps_snapshot": int(sys.argv[2]),
        "qps_baseline": int(sys.argv[3]),
        "delta": int(sys.argv[4]),
        "arrival_snapshot": int(sys.argv[5]),
        "arrival_baseline": int(sys.argv[6]),
        "judge": int(sys.argv[7]),
    },
}
(output_root / "figure_index.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
PY

find "$OUTPUT_ROOT" -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum >"$OUTPUT_ROOT/SHA256SUMS"
echo "[DONE] Audited Ministral paper figures: $OUTPUT_ROOT"

#!/usr/bin/env bash
# Score and compare the corrected seq128 QPS-point runs.

set -euo pipefail

if (( $# != 2 )); then
  echo "Usage: $(basename "$0") RESULT_ROOT OLD_SUMMARY_JSON" >&2
  exit 2
fi

result_root="$1"
old_summary="$2"
sfs_root="${SFS_ROOT:-}"
project_root="${PROJECT_ROOT:-$(cd "$sfs_root/.." && pwd)}"
qps_req_map="${QPS_REQ_MAP:-$project_root/vllm_utils/bucketed_prompt_outputs/req_map_qps_seed69_holdout4000_n16000.csv}"
delta_req_map="${DELTA_REQ_MAP:-$project_root/vllm_utils/bucketed_prompt_outputs/req_map_delta_seed69_holdout2000_n8000.csv}"
scored_root="${SCORED_ROOT:-$project_root/vllm_utils/bucketed_prompt_outputs/holdout_4000_scored}"

if [[ -z "$sfs_root" || ! -d "$sfs_root/src" ]]; then
  echo "[ERROR] SFS_ROOT must identify the corrected SFS worktree." >&2
  exit 1
fi
for required in "$old_summary" "$qps_req_map" "$delta_req_map"; do
  if [[ ! -f "$required" ]]; then
    echo "[ERROR] Required file not found: $required" >&2
    exit 1
  fi
done
if [[ ! -d "$scored_root" ]]; then
  echo "[ERROR] Scored-output directory not found: $scored_root" >&2
  exit 1
fi

find_run_dir() {
  local parent="$1"
  local matches=()
  mapfile -t matches < <(
    find "$parent" -mindepth 1 -maxdepth 1 -type d \
      -name 'router_qps_sweep_*' -print | sort
  )
  if (( ${#matches[@]} != 1 )); then
    echo "[ERROR] Expected exactly one sweep result under $parent; found ${#matches[@]}." >&2
    exit 1
  fi
  printf '%s\n' "${matches[0]}"
}

legacy_dir="$(find_run_dir "$result_root/legacy")"
cross_dir="$(find_run_dir "$result_root/cross_term")"
summary_root="$result_root/summary"
mkdir -p "$summary_root/legacy" "$summary_root/cross_term"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm
export PYTHONPATH="$sfs_root/src:${PYTHONPATH:-}"

python -m scripts.eval.augment_router_actual_accuracy \
  "$legacy_dir" "$cross_dir" \
  --delta-req-map "$delta_req_map" \
  --qps-req-map "$qps_req_map" \
  --scored-root "$scored_root"

python -m scripts.reporting.router_qps_sweep_summary \
  "$legacy_dir" \
  --output-dir "$summary_root/legacy"
python -m scripts.reporting.router_qps_sweep_summary \
  "$cross_dir" \
  --output-dir "$summary_root/cross_term"

python - \
  "$old_summary" \
  "$summary_root/legacy/router_qps_sweep_summary.json" \
  "$summary_root/cross_term/router_qps_sweep_summary.json" \
  "$summary_root/comparison.json" <<'PY'
import json
import sys
from pathlib import Path

old_path, legacy_path, cross_path, output_path = map(Path, sys.argv[1:])
old = json.loads(old_path.read_text(encoding="utf-8"))["qps"]["8.75"]
legacy = json.loads(legacy_path.read_text(encoding="utf-8"))["qps"]["8.75"]["hard"]
cross = json.loads(cross_path.read_text(encoding="utf-8"))["qps"]["8.75"]["hard"]

fields = (
    "system_entry_e2e_ttft_ms_slo_attainment_pct",
    "actual_slo_gated_utility_mean",
    "average_system_entry_e2e_ttft_ms",
    "p90_system_entry_e2e_ttft_ms",
    "actual_accuracy_mean",
    "throughput_qps_all",
)
cases = {
    "old_sfs_legacy": old["hard"],
    "corrected_legacy": legacy,
    "corrected_cross_term": cross,
    "shortest_queue": old["shortest_queue"],
    "latency_agnostic": old["latency_agnostic"],
    "round_robin": old["round_robin"],
}
metrics = {
    label: {field: values.get(field) for field in fields}
    for label, values in cases.items()
}

def deltas(candidate: str, baseline: str) -> dict[str, float | None]:
    result = {}
    for field in fields:
        candidate_value = metrics[candidate][field]
        baseline_value = metrics[baseline][field]
        result[field] = (
            float(candidate_value) - float(baseline_value)
            if isinstance(candidate_value, (int, float))
            and isinstance(baseline_value, (int, float))
            else None
        )
    return result

comparison_fields = (
    "system_entry_e2e_ttft_ms_slo_attainment_pct",
    "actual_slo_gated_utility_mean",
)
baseline_labels = ("shortest_queue", "latency_agnostic", "round_robin")
exceeds_baselines = {
    candidate: {
        baseline: {
            field: (
                float(metrics[candidate][field])
                > float(metrics[baseline][field])
            )
            for field in comparison_fields
        }
        for baseline in baseline_labels
    }
    for candidate in ("corrected_legacy", "corrected_cross_term")
}

payload = {
    "qps": 8.75,
    "num_requests_per_case": 16000,
    "metrics": metrics,
    "deltas": {
        "corrected_legacy_minus_old_sfs": deltas(
            "corrected_legacy", "old_sfs_legacy"
        ),
        "corrected_cross_term_minus_old_sfs": deltas(
            "corrected_cross_term", "old_sfs_legacy"
        ),
        "cross_term_minus_corrected_legacy": deltas(
            "corrected_cross_term", "corrected_legacy"
        ),
    },
    "exceeds_baselines": exceeds_baselines,
    "winner_by_slo_attainment": max(
        metrics,
        key=lambda label: metrics[label][
            "system_entry_e2e_ttft_ms_slo_attainment_pct"
        ],
    ),
    "winner_by_ontime_utility": max(
        metrics,
        key=lambda label: metrics[label]["actual_slo_gated_utility_mean"],
    ),
}
output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

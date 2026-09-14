#!/usr/bin/env bash
#SBATCH --job-name=ministral3-methodology-smoke
#SBATCH --output=experiments/ministral3_methodology_smoke_%j.out
#SBATCH --error=experiments/ministral3_methodology_smoke_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=GPU-shared
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100-80:3

# Functional smoke only. Its fixed absolute QPS is not a capacity estimate.
# May also be sourced from a combined smoke/scout allocation to reuse servers:
# set METHODOLOGY_REUSE_POOL=1 and the common MINISTRAL_* pool variables.
set -euo pipefail
: "${SFS_ROOT:?Set SFS_ROOT to the active sfs_model_family worktree}"
: "${METHODOLOGY_CALIBRATION_JSON:?Set the corrected-profile fitted artifact}"
: "${ROUTEBALANCE_PREDICTOR_DIR:?Set the native MiniLM/KNN artifact directory}"
source "$SFS_ROOT/src/slurm/runs/ministral3_router_common.sh"
ministral3_resolve_sfs_root
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm
export PYTHONPATH="${METHODOLOGY_DEPS_DIR:-$SFS_ROOT/experiments/methodology_baselines/deps}:$SFS_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$SFS_ROOT/experiments/methodology_baselines/scratch"
export TMPDIR="$SFS_ROOT/experiments/methodology_baselines/scratch"
# The common pool launcher changes cwd before the experiment driver starts.
METHODOLOGY_CALIBRATION_JSON="$(realpath "$METHODOLOGY_CALIBRATION_JSON")"
ROUTEBALANCE_PREDICTOR_DIR="$(realpath "$ROUTEBALANCE_PREDICTOR_DIR")"
export METHODOLOGY_CALIBRATION_JSON ROUTEBALANCE_PREDICTOR_DIR
python - <<'PY'
import importlib.util
import json
import os
from pathlib import Path
from sfs_core.routing.methodology_calibration import MethodologyCalibration
from sfs_core.routing.routebalance_predictor import RouteBalancePredictor

root = Path(os.environ["SFS_ROOT"]).resolve()
spec = importlib.util.find_spec("vllm")
if spec is None or Path(spec.origin).resolve().parent.parent != root / "vllm":
    raise SystemExit("Active editable vLLM must point to this worktree, including its snapshot changes")
calibration = MethodologyCalibration.load(os.environ["METHODOLOGY_CALIBRATION_JSON"])
calibration.validate_runtime_profile({
    "dtype": "auto (BF16 checkpoints)", "max_model_len": 131072,
    "context_length": 65536, "prompt_token_limit": 32768,
    "max_completion_tokens": 8192, "chunked_prefill": True,
    "max_num_batched_tokens": 32768, "max_num_seqs": 512, "prefix_caching": False,
})
calibration.preload()
predictor = RouteBalancePredictor.load(os.environ["ROUTEBALANCE_PREDICTOR_DIR"])
expected = {"ministral3-3b", "ministral3-8b", "ministral3-14b"}
if set(calibration.speeds) != expected or set(predictor.model_labels) != expected:
    raise SystemExit("Baseline artifacts must cover exactly the three Ministral candidates")
print(json.dumps({"status": "PASS", "calibration": calibration.metadata,
                  "predictor_artifact": os.environ["ROUTEBALANCE_PREDICTOR_DIR"]}))
PY
if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
  exit 0
fi

if [[ "${METHODOLOGY_REUSE_POOL:-0}" != "1" ]]; then
  MINISTRAL_RUN_ROOT="$(realpath -m "${RUN_ROOT_OVERRIDE:-$SFS_ROOT/experiments/ministral3_paper/methodology_smoke/run_${SLURM_JOB_ID:?}}")"
  if [[ -e "$MINISTRAL_RUN_ROOT" ]]; then
    echo "[ERROR] Refusing to overwrite $MINISTRAL_RUN_ROOT" >&2
    exit 1
  fi
  ministral3_start_router_pool
fi
: "${MINISTRAL_RUN_ROOT:?A reused pool must provide its run root}"
: "${MINISTRAL_INSTANCES_CONFIG:?A reused pool must provide its instance config}"
SMOKE_OUTPUT="$MINISTRAL_RUN_ROOT/methodology_smoke.json"
if [[ -e "$SMOKE_OUTPUT" ]]; then
  echo "[ERROR] Refusing to overwrite $SMOKE_OUTPUT" >&2
  exit 1
fi
git -C "$SFS_ROOT" rev-parse HEAD >"$MINISTRAL_RUN_ROOT/methodology_sfs_head.txt"
git -C "$SFS_ROOT" diff HEAD --binary >"$MINISTRAL_RUN_ROOT/methodology_sfs_changes.patch"
git -C "$SFS_ROOT/vllm" rev-parse HEAD >"$MINISTRAL_RUN_ROOT/methodology_vllm_head.txt"
git -C "$SFS_ROOT/vllm" diff HEAD --binary >"$MINISTRAL_RUN_ROOT/methodology_vllm_changes.patch"
python - "$MINISTRAL_RUN_ROOT" <<'PY'
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import vllm

root = Path(os.environ["SFS_ROOT"]).resolve()
source_paths = [
    "src/sfs_core/routing/methodology_calibration.py",
    "src/sfs_core/routing/methodology_policies.py",
    "src/sfs_core/routing/methodology_scheduler.py",
    "src/sfs_core/routing/methodology_snapshot.py",
    "src/sfs_core/routing/routebalance_predictor.py",
    "src/sfs_core/routing/snapshot_shm_client.py",
    "src/sfs_core/routing/wait_time_scheduler.py",
    "src/scripts/prep/build_routebalance_predictor.py",
    "src/scripts/prep/fit_methodology_calibration.py",
    "src/scripts/runs/experiments.py",
    "src/scripts/runs/experiments_sweep.py",
    "src/scripts/runs/service_metrics_config.py",
    "src/slurm/runs/ministral3_router_common.sh",
    "src/slurm/runs/ministral3_methodology_smoke.sbatch",
    "vllm/vllm/v1/core/sched/scheduler.py",
    "vllm/vllm/v1/core/sched/state_snapshot.py",
    "vllm/vllm/v1/core/sched/snapshot_serialization.py",
    "vllm/vllm/v1/engine/snapshot_shm.py",
    "vllm/vllm/v1/engine/async_llm.py",
    "vllm/vllm/v1/engine/output_length_predictor.py",
]
versions = {}
for name in ("numpy", "scipy", "scikit-learn", "xgboost", "xgboost-cpu", "faiss-cpu", "torch",
             "transformers", "tokenizers", "mistral-common", "msgspec", "openai"):
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
calibration_path = Path(os.environ["METHODOLOGY_CALIBRATION_JSON"])
manifest = {
    "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                      for name in source_paths},
    "source_hash_scope": "Includes new untracked sources; tracked changes also recorded against HEAD",
    "python": sys.version, "executable": sys.executable,
    "platform": platform.platform(), "package_versions": versions,
    "vllm_import": str(Path(vllm.__file__).resolve()),
    "calibration_path": str(calibration_path),
    "calibration_sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
    "predictor_artifact": os.environ["ROUTEBALANCE_PREDICTOR_DIR"],
    "shared_server_predictor": "Existing AsyncLLM output predictor remains enabled for every policy; baseline selectors ignore its targets",
}
Path(sys.argv[1], "methodology_source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY

python -m scripts.runs.experiments \
  --experiment router --utilities lmdeploy_proxy mooncake_prefill routebalance \
  --instances-config "$MINISTRAL_INSTANCES_CONFIG" \
  --methodology-calibration-json "$METHODOLOGY_CALIBRATION_JSON" \
  --routebalance-predictor-path "$ROUTEBALANCE_PREDICTOR_DIR" \
  --routebalance-weights 0.3333333333333333 0.3333333333333333 0.3333333333333333 \
  --routebalance-batch-max-size 16 --routebalance-batch-wait-ms 25 \
  --num-requests 192 --request-rate-qps 2 --arrival-process poisson \
  --holdout-prompts-per-bucket 48 --holdout-start-index 0 \
  --holdout-bucket-dir "$MINISTRAL_HOLDOUT_SOURCE_DIR" \
  --holdout-cache-dir "$MINISTRAL_RUN_ROOT/calibration_smoke_prompts" \
  --bucket-dir "$MINISTRAL_RUN_ROOT/calibration_smoke_prompts" \
  --holdout-context-length 65536 \
  --worker-count 1 --max-queue-size 1 --max-completion-tokens 8192 \
  --temperature 0 --top-p 1 \
  --system-prompt 'You are a helpful assistant.' --chat-template-kwargs-json '{}' \
  --tokenizer-id "$MINISTRAL_TOKENIZER_PATH" --tokenizer-mode mistral \
  --accuracy-model-path "$MINISTRAL_ACCURACY_PREDICTOR" \
  --output-length-model-path "$MINISTRAL_OUTPUT_PREDICTOR" \
  --service-metrics-json "$MINISTRAL_SERVICE_METRICS" \
  --queue-slo-min-ms 1 --queue-slo-max-ms 5 \
  --ttft-slo-min-ms 150 --ttft-slo-max-ms 1120 \
  --ttft-slo-base-ms 155 --ttft-slo-per-prompt-token-ms 0.035 \
  --ttft-slo-jitter-min 0.98 --ttft-slo-jitter-max 1.02 --feasible-slo-mode ttft \
  --per-request-wait-log "$MINISTRAL_WAIT_LOG_3B" \
  --per-request-wait-log "$MINISTRAL_WAIT_LOG_8B" \
  --per-request-wait-log "$MINISTRAL_WAIT_LOG_14B" \
  --output-path "$SMOKE_OUTPUT" >"$MINISTRAL_RUN_ROOT/methodology_smoke_driver.log" 2>&1

python - "$SMOKE_OUTPUT" <<'PY'
import json
import math
from collections import Counter
from pathlib import Path
import sys
path = Path(sys.argv[1])
payload = json.loads(path.read_text())
runs = payload["router"]["runs"]
expected = {"lmdeploy_proxy", "mooncake_prefill", "routebalance"}
source = payload["config"]["prompt_source"]
if source.get("holdout_start_index") != 0 or source.get("holdout_prompts_per_bucket") != 48:
    raise SystemExit("Smoke must use calibration indices 0:48, never evaluation holdouts")
expected_buckets = {"alpaca", "govreport-summarization", "hotpot_qa", "writingprompts"}
if len(runs) != 3 or {run["methodology_config"]["policy"] for run in runs} != expected:
    raise SystemExit("Smoke is missing a policy")
for run in runs:
    rows = run["per_request"]
    if len(rows) != 192 or len({row["request_id"] for row in rows}) != 192:
        raise SystemExit("Smoke request coverage failed")
    if Counter(row["bucket"] for row in rows) != Counter({bucket: 48 for bucket in expected_buckets}):
        raise SystemExit("Smoke calibration bucket coverage failed")
    if any(row.get("error") or not row.get("methodology_terms") for row in rows):
        raise SystemExit("Smoke contains request failures or missing decision logs")
    if any(run["methodology_config"]["unfinished_after_drain"].values()):
        raise SystemExit("Smoke leaked outstanding reservations")
    response_ids = [row.get("response_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in response_ids) or len(set(response_ids)) != 192:
        raise SystemExit("Smoke response ID coverage failed")
    for row in rows:
        if row["methodology_terms"].get("engine_request_id") != row["response_id"]:
            raise SystemExit("Smoke engine/response ID mapping failed")
        for key in ("queue_delay_ms", "ttft_ms", "system_entry_to_dispatch_ms",
                    "system_entry_e2e_ttft_ms", "actual_cost"):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise SystemExit(f"Smoke has missing/invalid measured {key}")
        if not isinstance(row.get("system_entry_e2e_ttft_slo_met"), bool):
            raise SystemExit("Smoke SLO coverage failed")
        for key in ("usage_prompt_tokens", "usage_completion_tokens", "usage_total_tokens"):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SystemExit(f"Smoke has missing/invalid {key}")
        if row["usage_prompt_tokens"] + row["usage_completion_tokens"] != row["usage_total_tokens"]:
            raise SystemExit("Smoke usage accounting failed")
        if not 0 < row["prompt_tokens"] <= 32768:
            raise SystemExit("Smoke prompt exceeds the calibrated serving profile")
audit = {"status": "PASS", "scope": "functional smoke; not measured router capacity",
         "policies": sorted(expected), "requests_per_policy": 192, "qps": 2,
         "data_role": "calibration", "bucket_indices": [0, 48],
         "response_usage_and_latency_coverage": "complete"}
path.with_name("methodology_smoke_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit))
PY

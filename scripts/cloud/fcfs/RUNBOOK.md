# Qwen FCFS unchunked configuration: qualification runbook

Configuration `qwen-fcfs-unchunked-65536` (`src/scripts/cloud/fcfs/config.py`):
`scheduling_policy=fcfs`, chunked prefill disabled, `max_num_batched_tokens=65536`,
`max_model_len=65536`, `max_num_seqs=512`, `long_prefill_token_threshold=0`,
`gpu_memory_utilization=0.90`, prefix caching disabled; TP 1/1/2, kernels,
precision, chat template, YaRN/RoPE, predictors, SLOs and the 16,000-request
workload are canonical. Grid: nine policies × 6/7/8/8.3 QPS = 36 cells
(`scripts/cloud/fcfs/campaign-20260916.json`, `full_matrix_authorized: false`).

**Status:** preparation and qualification only. `worker run/campaign --profile fcfs`
refuse the overlay until `full_matrix_authorized` is `true`, which requires explicit
user authorization. `hard` and `score` cells stay `BLOCKED_TOKEN_LENGTH_VALIDATION`
(execution plan: no measured SFS/SCORE cells) and are excluded from executable cells
even after authorization until their status is changed.

## Host rules

- Host `ssh sfs-vast`, `SFS_STORAGE=/workspace/sfs`. Production programs
  `sfs-baselines-qwen-mrb` (GPUs 0-3) and `sfs-baselines-ministral-mrb` (GPUs 4-6)
  own their GPUs through `/dev/shm/sfs-cloud-locks-0`; every FCFS pool takes the
  same locks, so a busy GPU fails fast instead of sharing. GPU 7 is reserved for
  the other debugging task. Do not stop or restart any existing Supervisor program.
- FCFS work runs only from the separate checkout `/workspace/sfs/fcfs-repo`
  (branch `experiments/qwen-fcfs-unchunked-20260916`). `/workspace/sfs/repo` is
  the production source; never modify it.
- FCFS state root is `/workspace/sfs/fcfs` (`setup/` gates, `state/` runs,
  ledger `state/completed/`). The worker resolves gates from
  `--state`'s parent, so this keeps the production gates in `/workspace/sfs/setup`
  untouched and keeps the FCFS completion ledger separate. Prior FCFS evidence
  lives under `/workspace/sfs/state/fcfs/` (inputs audit, GPU7 smokes).
- One Supervisor program per command (`/etc/supervisor/conf.d/sfs-fcfs-*.conf`,
  `autostart=false`, `autorestart=false`, `stopasgroup=true`), logs under
  `/workspace/sfs/setup/`. Start with `supervisorctl reread && supervisorctl update`
  then `supervisorctl start <program>`.

## Evidence already recorded

| Step | Result | Where |
|---|---|---|
| Actual chat admission audit, all three models, evaluation/calibration/warm-up | `PASS_ACTUAL_CHAT_INPUTS`; max formatted prompt 32,801 tokens; headroom 24,543 tokens after the 8,192 allowance | `/workspace/sfs/state/fcfs/inputs-20260916/audit.json` |
| Bounded GPU7 FCFS smoke, 0.6B | `PASS_BOUNDED_GPU_FCFS_SMOKE`, 7 full-prefill checks | `/workspace/sfs/state/fcfs/gpu7-model-0-20260916-attempt2/gpu-smoke.json` |
| Bounded GPU7 FCFS smoke, 8B | `PASS_BOUNDED_GPU_FCFS_SMOKE`, 63 full-prefill checks; 32,801-token prompt served whole | `/workspace/sfs/state/fcfs/gpu7-model-1-20260916-attempt2/gpu-smoke.json` |
| First 0.6B attempt | engine-core startup failure, retained | `/workspace/sfs/state/fcfs/gpu7-model-0-20260916/server_qwen3-0.6b.log` |

Both smokes used canonical coefficients as placeholders (`coefficient_status`
`CANONICAL_PLACEHOLDER_FOR_TRACE_COLLECTION_ONLY`).

## What qualification requires (repo references)

1. Source-bound CPU regression gate for the FCFS checkout (`scripts/cloud/test.sh`,
   `src/scripts/cloud/test_gate.py`; `worker.execute` refuses stale `tests/gate.json`).
2. Destination CPU input and serving gates (`scripts.cloud.prepare cpu|serving`).
3. FCFS chat admission audit copied to `setup/fcfs-inputs.json` (`worker.execute`,
   profile `fcfs`; produced by `scripts.cloud.fcfs.inputs`).
4. Bounded GPU FCFS smoke for every model, including 32B at TP=2
   (`scripts.cloud.fcfs.gpu_smoke`: startup, memory, observed full-prefill scheduling).
5. Destination calibration traces per model under this profile
   (`worker.calibrate`: warm shapes, singleton prefill probes, 512 loaded requests).
6. SFS batch coefficients refitted for this configuration
   (`scripts.cloud.fcfs.coefficients fit`; same estimator as
   `service_metrics_config.derive_nonnegative_calibration`, R² ≥ 0.95, nonnegative).
   Coefficients reach the servers (`--simulation-*`, published in the snapshot header
   read by the native simulator) and `instances.json` (`ttft_batch_model`), so they
   must exist before the qualification pool starts.
7. Timing heads (Mooncake prefill, RouteBalance TPOT) fitted from FCFS traces
   (`fit_methodology_calibration.fit_manifest`, profile-hashed; the router validates
   `serving_profile` at run time).
8. Nine policy smokes (192 calibration prompts, 2 QPS) and shortest-queue load probes
   at 6 and 8.3 QPS (`worker qualify`), `qualification.json` bound to source, bundle,
   hardware, overlay, configuration id and coefficient file hash.
9. Reviews: coefficient residuals on independent post-calibration batches
   (`scripts.cloud.fcfs.coefficients validate`), timing-head residuals
   (`ops/cloud/review_baseline_timing.py`), load-probe classifications, server logs,
   SFS predicted-versus-observed TTFT on the `hard` smoke; then
   `scripts.cloud.control release --timing-review ... --load-review ...`.
10. Provenance: `qualification.json`, `release.json`, coefficient file, audits copied
    to `scripts/cloud/reports/fcfs-20260916/`; ledger entries carry the serving
    profile identity; execution plan `next_serving_configuration` status updated.

## Step 0: CPU-only, can run now (no GPU)

```bash
ssh sfs-vast
export SFS_STORAGE=/workspace/sfs
cd $SFS_STORAGE/fcfs-repo && git fetch origin && git checkout <this branch commit> && git status --short
sha256sum -c <(python3 -c "import json;[print(r['sha256'],' ',r['path']) for r in json.load(open('$SFS_STORAGE/setup/fcfs-binaries.json'))]")
source scripts/cloud/env.sh
mkdir -p $SFS_STORAGE/fcfs/setup $SFS_STORAGE/fcfs/state
cp $SFS_STORAGE/state/fcfs/inputs-20260916/audit.json $SFS_STORAGE/fcfs/setup/fcfs-inputs.json
export SFS_TEST_GO="$CONDA_PREFIX/bin/go"
bash scripts/cloud/test.sh $SFS_STORAGE/fcfs/setup/tests                # ~15 min CPU, includes the new FCFS tests
python -m scripts.cloud.prepare cpu     --bundle $SFS_STORAGE/bundle --output $SFS_STORAGE/fcfs/setup/cpu-inputs.json
python -m scripts.cloud.prepare serving --bundle $SFS_STORAGE/bundle --output $SFS_STORAGE/fcfs/setup/cpu-serving.json
```

Run these under Supervisor programs (`sfs-fcfs-cpu-gates`) with `taskset -c 84-95`
so the production lanes keep their cores. The gates bind the FCFS source hash; any
later source change repeats this step.

## Step 1: 32B TP=2 bounded smoke (2 GPUs, ~10 min)

After GPUs 0-3 free (Qwen baseline lane complete, `supervisorctl status
sfs-baselines-qwen-mrb` shows EXITED and `nvidia-smi` shows 0 MiB on 0-3):

```bash
# /workspace/sfs/setup/fcfs-32b.sh  (program sfs-fcfs-32b)
export SFS_STORAGE=/workspace/sfs; source $SFS_STORAGE/fcfs-repo/scripts/cloud/env.sh
taskset -c 84-91 python -m scripts.cloud.fcfs.gpu_smoke --bundle $SFS_STORAGE/bundle \
  --models $SFS_STORAGE/models.json --output $SFS_STORAGE/state/fcfs/gpus01-model-2-20260916 \
  --gpus 0,1 --indices 2
```

Pass: `gpu-smoke.json` `PASS_BOUNDED_GPU_FCFS_SMOKE`, `full_prefill_checks > 0`,
all eight responses 128 tokens including the 32,801-token prompt, KV-cache size in
`server_qwen3-32b.log` at `max_model_len 65536` with `gpu_memory_utilization 0.9`.
Fail: debug (keep the retained log); do not proceed.

## Step 2: pass A, trace collection (4 GPUs, ~12 min)

```bash
# /workspace/sfs/setup/fcfs-calibrate.sh  (program sfs-fcfs-calibrate)
export SFS_STORAGE=/workspace/sfs; source $SFS_STORAGE/fcfs-repo/scripts/cloud/env.sh
taskset -c 0-47 python -m scripts.cloud.worker calibrate --profile fcfs --family qwen \
  --bundle $SFS_STORAGE/bundle --models $SFS_STORAGE/models.json \
  --state $SFS_STORAGE/fcfs/state --output $SFS_STORAGE/fcfs/state/calibrate-20260916 \
  --campaign $SFS_ROOT/scripts/cloud/fcfs/campaign-20260916.json --gpus 0,1,2,3 --cpus $(seq -s, 0 47)
```

Servers start with placeholder coefficients (recorded in `instances.json`
`coefficient_status`). Output: `calibration_trace_<model>.csv`, `model_metrics.json`,
`timing_models/`, `calibration.json` (`TRACES_COLLECTED_COEFFICIENT_FIT_REQUIRED`).
No smoke or evaluation runs in this mode.

## Step 3: CPU refit of SFS coefficients (minutes)

```bash
python -m scripts.cloud.fcfs.coefficients fit --calibration $SFS_STORAGE/fcfs/state/calibrate-20260916 \
  --output $SFS_STORAGE/fcfs/coefficients-20260916.json
```

Review `models.<model>.fit_prediction_diagnostics` (R² ≥ 0.95 enforced, MAE,
negative rows, per-group bias) and `fit_inlier_rows`. Compare with the canonical
coefficients in `src/scripts/runs/qwen_baselines.py` and record the comparison; a
prefill coefficient far from canonical is expected (full prompts per batch), a decode
coefficient far from canonical is not.

## Step 4: pass B, qualification with fitted coefficients (4 GPUs, ~45-60 min)

```bash
# /workspace/sfs/setup/fcfs-qualify.sh  (program sfs-fcfs-qualify)
export SFS_STORAGE=/workspace/sfs; source $SFS_STORAGE/fcfs-repo/scripts/cloud/env.sh
taskset -c 0-47 python -m scripts.cloud.worker qualify --profile fcfs --family qwen \
  --bundle $SFS_STORAGE/bundle --models $SFS_STORAGE/models.json \
  --state $SFS_STORAGE/fcfs/state --output $SFS_STORAGE/fcfs/state/qualify-20260916 \
  --campaign $SFS_ROOT/scripts/cloud/fcfs/campaign-20260916.json \
  --coefficients $SFS_STORAGE/fcfs/coefficients-20260916.json --gpus 0,1,2,3 --cpus $(seq -s, 0 47)
```

Sequence inside: fresh calibration (independent of pass A) and timing-head fit, nine
192-request policy smokes at 2 QPS (`smoke/<policy>/point.json`), shortest-queue
load probes at 6 and 8.3 QPS (`load_probes/<rate>/`), then `qualification.json`
with `configuration_id`, `serving_profile`, `coefficients_sha256` and
`GPU_MEASURED_REVIEW_REQUIRED`. The pool exits (qualify mode); nothing is evaluated.
Estimate from the canonical Qwen qualification on this host: calibration 7 min,
2-7 min per policy smoke, 7-8 min per load probe.

## Step 5: CPU reviews and release

```bash
Q=$SFS_STORAGE/fcfs/state/qualify-20260916
python -m scripts.cloud.fcfs.coefficients validate --coefficients $SFS_STORAGE/fcfs/coefficients-20260916.json \
  --calibration $Q --output $Q/coefficient-review.json          # independent post-calibration batches
python ops/cloud/review_baseline_timing.py --qualification $Q --output $Q/timing-review.json
python -c "import json;r=json.load(open('$Q/smoke/hard/point.json'))['router']['runs'][0];print(r['summary'])"
# SFS predicted vs observed TTFT: $Q/smoke/hard/predicted_waits_router_hard.log versus per_request in point.json
```

Review checklist (record concrete numbers in the release text):
- coefficient-review: independent `r2_all_rows`, MAE and bias for pure-decode,
  prefill and mixed groups per model, no negative predictions;
- timing-review: TPOT head MAE on post-calibration batches; `timing_models/
  methodology_calibration.json` prefill coverage including `chunk_token_range`
  up to ~32.8k tokens and `partial_prefill_observed == false` (expected: unchunked);
- load probes: `qualification.json` `load_probes` classifications at 6 and 8.3 QPS,
  realized arrival rates within 10%;
- all nine smokes: `succeeded_requests == 192`, `failed_requests == 0`, complete
  measured end-to-end TTFT (`audit_run` and `require_complete_ttft` already enforced);
- simulator consistency: every snapshot config observed during the pool reports
  `chunked_prefill_enabled false`, `max_num_batched_tokens 65536`; predicted waits in
  `smoke/hard/predicted_waits_router_hard.log` versus observed TTFT in `point.json`
  show no systematic under-prediction on long prompts (the GPU7 smokes already
  verified that observed scheduling is whole-prompt, which is what
  `methodology_snapshot.py` and `scheduler_simulator.py` assume when chunking is off);
- server logs free of OOM/preemption warnings under the 0.90 memory fraction.

```bash
python -m scripts.cloud.control release --qualification $Q \
  --timing-review 'Fitted FCFS coefficients: independent R^2 ..., decode MAE ... ms, prefill MAE ... ms; TPOT head MAE ...; prefill coverage ...' \
  --load-review 'Nine 192-request smokes complete; load probes at 6/8.3 QPS classified ...; hardware ...'
```

Then copy `qualification.json`, `release.json`, `coefficient-review.json`,
`timing-review.json`, the coefficient file and `hardware.json` to
`scripts/cloud/reports/fcfs-20260916/`, add 36 ledger entries (status
`QUALIFIED_NOT_LAUNCHED`, `serving_profile` = the FCFS profile, `configuration_id`)
and update `execution-plan-20260916.json` `next_serving_configuration.status`.
Do not set `full_matrix_authorized`.

## Blocked: full matrix (not authorized)

For reference only, after the user authorizes and `full_matrix_authorized` is
committed as `true` (which changes the overlay hash, so `qualification.json`
`campaign_sha256` must be re-established by re-running qualify, or the launch must
use the same overlay bytes the qualification recorded):

```bash
python -m scripts.cloud.worker run --profile fcfs --family qwen --bundle $SFS_STORAGE/bundle \
  --models $SFS_STORAGE/models.json --state $SFS_STORAGE/fcfs/state \
  --output $SFS_STORAGE/fcfs/state/run-<batch> --campaign $SFS_ROOT/scripts/cloud/fcfs/campaign-20260916.json \
  --coefficients $SFS_STORAGE/fcfs/coefficients-20260916.json --qualification $Q \
  --gpus 0,1,2,3 --cpus $(seq -s, 0 47) --cells <comma-separated cell ids>
```

Every measured cell on this host has taken 39-46 min (16,000 requests at 6-8.3 QPS
plus drain): 28 executable cells ≈ 20 h on GPUs 0-3; all 36 ≈ 26 h.

## GPU time summary

| Step | GPUs | Estimate |
|---|---|---|
| 32B TP=2 smoke | 2 | 10 min |
| Pass A calibrate | 4 | 12 min |
| Pass B qualify | 4 | 45-60 min |
| Total qualification | 4 | about 1.5 h wall |
| Full matrix (blocked) | 4 | 20-26 h |

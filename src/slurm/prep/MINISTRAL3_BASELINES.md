# Implemented Ministral routing baselines

Read [the methodology audit](MINISTRAL3_BASELINE_AUDIT.md) for paper links,
preserved mechanisms, and explicit adaptation choices. The primary baseline
IDs are `lmdeploy_proxy`, `mooncake_prefill`, and `routebalance`. The optional
SFS-predictor sensitivity comparison is excluded.

For the September 7 metadata repair, 209 passing CPU tests, held-job status,
and all four prepared experiment tracks, use [the paper-job guide](PAPER_ABLATIONS.md).
The earlier test counts below describe the original baseline-only validation.

## Code and current validation

| Component | Source |
| --- | --- |
| Selector equations and lifetime counts | `src/sfs_core/routing/methodology_policies.py` |
| Async batching, dispatch, reconciliation, logging | `src/sfs_core/routing/methodology_scheduler.py` |
| Observed telemetry and conservative admission evidence | `src/sfs_core/routing/methodology_snapshot.py`, `snapshot_shm_client.py` |
| Native MiniLM/FAISS predictor and CPU builder | `src/sfs_core/routing/routebalance_predictor.py`, `src/scripts/prep/build_routebalance_predictor.py` |
| Prefill/XGBoost TPOT fitting and artifact validation | `src/sfs_core/routing/methodology_calibration.py`, `src/scripts/prep/fit_methodology_calibration.py` |
| Experiment CLI and generic sweep | `src/scripts/runs/experiments.py`, `experiments_sweep.py` |
| Functional smoke, reusable inside a scout allocation | `src/slurm/runs/ministral3_methodology_smoke.sbatch` |
| Completed-service provenance and coverage audit | `src/scripts/prep/prepare_methodology_service.py` |
| Arrival-window classification and adaptive bracket | `src/scripts/runs/capacity_scout.py` |
| Combined probes, eight-policy smoke, and scout | `src/scripts/runs/ministral3_methodology_stage.py`, `src/slurm/runs/ministral3_methodology_stage.sbatch` |
| Frozen Figure 5 manifest, sweep and derived collation | `src/scripts/runs/ministral3_figure5.py` |

There are 152 validated focused CPU cases across the policy/calibration/async
scheduler (33), predictor/builder (49), snapshot/native compatibility (48), and
actual experiment/sweep/collation integration (22) suites. They include native
FAISS and XGBoost execution. They do not establish GPU correctness, measured
service-fit quality, or router capacity. Shell/heredoc checks cover the smoke
launcher separately.

September 6 validation covers 50 CPU cases across startup (3), capacity
accounting (6), SCORE/controller (9), actual experiment integration (24), and
Figure 5 manifest/collation (8). The controller uses simulated serving. The
integration suite includes bounded arrivals and dispatch-failure completion;
the collation suite checks missing/duplicate cells, frozen configuration and
input hashes, actual arrival attainment, exact QPS precision, and judged-quality
joins on derived copies. These supplement/recheck the earlier suites and are
not GPU serving results. The actual launcher preflight also passed with the
prepared request/predictor checksums and editable import.

A legacy model-size alias was corrected so Ministral 8B calibration cannot
alias Qwen 8B. SCORE now receives runtime service metrics before argument
parsing derives its in-memory calibration lookup.

The preflight runs Python from `src`: from the repository root, the outer
`vllm/` directory can shadow the installed editable package as a namespace.
The shared environment is unchanged. Its active editable path was verified as
this worktree's `vllm/vllm/__init__.py`.

The nested vLLM Python patch adds an optional scheduled-token-by-request map to
inflight snapshots. Existing native snapshot parsing remains compatible;
servers must restart with this worktree's Python publisher before baseline
GPU execution. The launcher verifies the editable import path, records source
hashes/patches, and refuses to overwrite outputs.

## CPU environment and artifacts

Run from `/ocean/projects/cis250162p/aparthas/sfs_model_family`:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm
export PYTHONPATH="$PWD/experiments/methodology_baselines/deps:$PWD/src:${PYTHONPATH:-}"
mkdir -p "$PWD/experiments/methodology_baselines/scratch"
export TMPDIR="$PWD/experiments/methodology_baselines/scratch"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
```

Extra wheels are already installed in the worktree-local dependency target:
`faiss-cpu==1.12.0` and `xgboost-cpu==3.0.5`. To recreate that target, run
`bash scripts/setup/setup_methodology_cpu_deps.sh`. It preserves the shared
conda environment and uses workspace scratch. Never use `/tmp` or `/var/tmp`.

The native predictor build uses the archived 10,000 prompt groups / 30,000
scores. The strict source/split audit is
`experiments/methodology_baselines/routebalance_input_audit_v2.json`.
The split has 8,987 training and 1,013 held-out calibration prompts: the 1,000
common saved SFS validation IDs, plus 13 exclusions preventing duplicate-text
leakage. The common 1,000 IDs remain separately identified for possible
calibration-only configuration selection. Imputed labels are retained and
flagged in provenance (27 model-score rows).

The native artifact destination is
`experiments/ministral3_paper/routebalance_predictor/run_20260905_native`.
`metadata.json` is written last as its completion marker. It records the
encoder revision/weight checksum/cache path, source and split hashes,
dependency versions, training/validation counts, and held-out error metrics.
The index is reconstructed from checked numeric arrays at load; there is no
pickle. The fresh offline scheduler-thread smoke writes
`experiments/methodology_baselines/routebalance_native_smoke.json`.

The native build completed on September 5 using four CPU threads and no GPU.
Its held-out calibration errors are diagnostics, not serving or routing results:

| Model | Quality MAE (quality in [0,1]) | Output-length MAE (tokens) |
| --- | --- | --- |
| 3B | 0.20836 | 263.10 |
| 8B | 0.19021 | 271.64 |
| 14B | 0.17047 | 275.51 |

Full per-bucket errors and RMSE are in `metadata.json`. Retain these prediction
limitations in the final analysis; no optional sensitivity run or tuning was
performed based on these diagnostics.

The fresh offline `asyncio.to_thread` prediction check passed; complete evidence
is `experiments/methodology_baselines/routebalance_native_final_audit.json`.
Cold loading/import took 249.57 seconds on the shared filesystem, and the
first three-prompt batch took 6.56 seconds. These are correctness-smoke
observations, not representative serving latency. Measure warmed predictor
overhead and client arrival-rate attainment in the planned shared smoke/scout.

The build command below is for reproduction with a **new output directory**;
reuse the existing completed artifact for experiments. Encoder weights are
already cached under the project Hugging Face cache, so no download flag is
needed.

```bash
python -m scripts.prep.build_routebalance_predictor \
  --completions-root /ocean/projects/cis250162p/aparthas/sfs_artifacts/ministral3_calibration_44849564_judge_45009121_20260902/run/completions \
  --output-dir "$NEW_ROUTEBALANCE_ARTIFACT_DIR" \
  --validation-id-files \
    experiments/ministral3_paper/predictors/run_45089699/accuracy_predictor/test_example_ids.json \
    experiments/ministral3_paper/predictors/run_45089699/output_length_predictor/test_example_ids.json \
  --cache-dir /ocean/projects/cis250162p/aparthas/.cache/huggingface/hub \
  --cpu-threads 4
```

## Completed service calibration and missing coverage

Job `45088929` completed successfully on September 5. Its final-profile
artifact is
`experiments/ministral3_h100-80_service_metrics_45088929_20260905_084501`.
The September 6 audit checked actual launch provenance, the three H100-80GB
GPUs, BF16 model pins, metric checksums, and 10,000 successful requests per
model with zero failures. The reusable manifest is
`experiments/methodology_baselines/service_manifest_45088929_20260906.json`.
The service run's detailed provenance supplies the full serving profile;
its older `instances.json` contains only a subset of those fields.

| Model | Successful requests / whole-run seconds | Pure singleton prefill rows | Decode-active rows |
| --- | --- | --- | --- |
| 3B | 2.977109 QPS | 5 | 363,421 |
| 8B | 2.709571 QPS | 1 | 103,138 |
| 14B | 1.749134 QPS | 1 | 158,240 |

These per-model request rates include drain and are LMDeploy speed inputs.
Neither they nor their sum establish router capacity. The decode traces are
reusable, but all three models lack sufficient isolated prefill coverage.
The combined stage therefore adds 64 length-stratified calibration requests
per model with a one-token completion cap and one outstanding request per
model. It freezes copies of these traces before any smoke/scout traffic.

To audit a different completed service run, choose a new manifest path:

```bash
python -m scripts.prep.prepare_methodology_service \
  --service-run-dir "$COMPLETED_SERVICE_RUN_DIR" \
  --output-path "$NEW_SERVICE_MANIFEST"
```

The old calibration generations remain quality/length labels; they are not
timing evidence under this profile. No production methodology timing fit has
yet completed. The combined stage invokes the following fitter after probes:

```bash
python -m scripts.prep.fit_methodology_calibration \
  --manifest "$SERVICE_CALIBRATION_MANIFEST" \
  --output-dir "$NEW_METHODOLOGY_CALIBRATION_DIR"
```

The fitter requires at least 40 usable pure singleton prefill rows and 40
decode-active rows per model. It rejects unidentifiable prefill fits, checks
chronological held-out errors, and records prompt/batch/context coverage and
mixed/partial-prefill counts. Inspect these diagnostics before freezing.
If coverage is insufficient, collect only the missing calibration probes in
the shared smoke/scout allocation. Do not lower thresholds merely to accept
unrelated/old traces. No synthetic timing artifact is a production substitute.

## Combined GPU stage

On September 6, the same stage was requested through an interactive allocation
as job `45367216`: three H100-80GB GPUs, 32 CPUs, and at most four hours. Slurm
confirmed `QOS=gpuinteract`; the request is pending resources. Normal-QoS retry
`45365762` is held as a fallback to prevent duplicate execution. Check both jobs
before launching anything else or releasing the fallback. The interactive
run's output directory will be
`experiments/ministral3_paper/methodology_stage/run_45367216`, with Slurm output
in `experiments/ministral3_methodology_interactive_45367216.{out,err}`.

PSC's interactive helper uses the job name `Interact`. An initial request with
a descriptive name was assigned normal `gpu` QoS despite `--qos=gpuinteract`;
that pending request (`45366760`) was cancelled without consuming GPU time.
The helper's next request (`45367029`) failed before allocation with a resource
configuration error. The current foreground allocation uses
`salloc --job-name=Interact --cpus-per-task=32` with the same GPU/time limits and
an `srun` command that automatically executes the existing stage launcher.
Keep its allocation client alive while pending/running. Submission details and
observed states are preserved in
`experiments/methodology_baselines/interactive_submission_45367216_20260906.json`.

The first attempt, `45333598`, failed during server startup on September 6
after 10 minutes 15 seconds. The long workspace TMPDIR made vLLM's Unix-domain
IPC socket names exceed Linux's 107-byte pathname limit. No probe, timing fit,
smoke, or scout ran. The common launcher now sets `VLLM_RPC_BASE_PATH` to
`/local/$USER/$SLURM_JOB_ID/ipc` and verifies an actual ZeroMQ bind before model
staging. The fix passed a CPU bind check; the failed run's files are preserved.
The retry keeps the same workload, artifacts, and resource limits.

CPU preparation is complete at
`experiments/methodology_baselines/prepared_20260906`: 10,000 requests from
calibration indices 0–2499, exactly 2,500 per bucket. `metadata.json` records
the source, construction arguments, and checksums. The preparation cache is
`experiments/data/prompts/ministral3/calibration_cache_2500`.

The stage loads the pool once, adds the missing prefill probes, fits timing
heads using existing decode traces, and runs all eight policies on 192
calibration requests each at 2 QPS. The smoke samples 48 requests per bucket
from the prepared pool. Uniform RouteBalance weights, batch size 16, and a
25 ms collection limit are frozen; the optional sensitivity check is excluded.
The stage drains between trials and audits responses, measured usage/timing,
decision logs, and reservation cleanup.

Shortest-queue scouting starts at an absolute 2 QPS, doubles or halves until
both stable and unstable rates are observed, then bisects to a bracket at
most 5% wide relative to its stable endpoint. A trial normally supplies 360
seconds of arrivals, excludes 120 seconds of warm-up, and classifies the last
three complete 60-second arrival windows using completion flow and backlog
growth. Drain contributes no capacity evidence. Events preserve actual arrival
times, ingress/router work, engine snapshots, and completion times. Backlog
is capped at 1,024 outstanding requests; request failures and instrumentation
problems stop the stage. Borderline observations permit one 600-second retry.
At most ten bracket trials run.

If the backlog cap ends an aggressive trial before enough windows exist, its
rate bounds the search only. The next trial moves below that guarded rate;
the truncated trial never counts as a measured unstable endpoint.

`K_M` is explicitly the smallest observed unstable rate in that narrowed
bracket, with both endpoints retained as uncertainty. The stage computes
`[0.65, 0.85, 0.95, 1.05] * K_M` and requires fresh drained confirmation trials
showing the first point stable and the last unstable before writing a passing
`capacity_scout.json`. Inconclusive measurements never produce final QPS values.
This reference applies to shortest queue; other policies may saturate elsewhere.

To repeat the preflight without allocating GPUs:

```bash
export SFS_ROOT="$PWD"
export SERVICE_JOB_ID=45088929
export PREDICTOR_RUN_DIR="$SFS_ROOT/experiments/ministral3_paper/predictors/run_45089699"
export ROUTEBALANCE_PREDICTOR_DIR="$SFS_ROOT/experiments/ministral3_paper/routebalance_predictor/run_20260905_native"
export METHODOLOGY_PREPARED_DIR="$SFS_ROOT/experiments/methodology_baselines/prepared_20260906"
export METHODOLOGY_SERVICE_MANIFEST="$SFS_ROOT/experiments/methodology_baselines/service_manifest_45088929_20260906.json"
VALIDATE_ONLY=1 bash src/slurm/runs/ministral3_methodology_stage.sbatch
```

The older three-policy `ministral3_methodology_smoke.sbatch` remains available
when a fitted timing artifact and pool already exist. A separate smoke
allocation is unnecessary for the current run.

The generic experiment/sweep CLI accepts all three policy IDs with:

```text
--methodology-calibration-json PATH
--routebalance-predictor-path DIRECTORY
--routebalance-weights 0.3333333333333333 0.3333333333333333 0.3333333333333333
--routebalance-batch-max-size 16
--routebalance-batch-wait-ms 25
--methodology-snapshot-max-age-ms 1000
```

Use `--utilities` in `experiments.py` or `--qps-utilities` in
`experiments_sweep.py`. Old default policy populations are preserved.
LMDeploy/Mooncake decision quality/length fields remain absent; derived
actual-quality joins still drive OnTimeUtility. Per-request `methodology_terms`
and run-level `methodology_config` preserve the reconstruction choices.

## Preparation results and remaining evaluation

Holdout generation `45089100` and judging `45089102` completed successfully.
The audit at `experiments/methodology_baselines/holdout_judge_audit_20260906.json`
confirms all 16,000 prompt groups / 48,000 scores and indices 2500–6499 in
every bucket. There are 24 flagged imputed model scores (eight writingprompts
groups). Preserve these flags in analysis.

Figure 3 job `45090868` failed because the batch-fit path unnecessarily tried
to load an absent legacy Qwen prompt directory. This dependency is removed;
the CPU-only repair from existing traces passes at
`experiments/ministral3_paper/figure3_batch_fit/repair_20260906/audit.json`.
Figure 2 job `45091510` failed after invalid warm-up messages were rejected.
Warm-up now puts sampling arguments at request level and propagates failures;
dispatch failures also resolve the harness completion future. Figure 2 has
not been rerun. The eight-policy GPU smoke first checks the shared SFS path.

Old Figure 5 jobs `45091769`, `45091770`, and collation `45197570` were held
on September 6 before they started. Their raw results and launcher defaults
are preserved. After the combined stage, inspect timing residuals, warmed
predictor overhead, actual arrival attainment, smoke audits, and scout windows.
Then freeze a shared manifest for eight policies, four measured loads, and
16,000 requests per cell (32 cells). The manifest tooling and both launcher
paths are implemented; no production manifest or final jobs exist yet.
Augment derived copies only.

## Freeze and execute Figure 5 after the scout passes

Record a JSON review containing `status: "PASS"`, the timing artifact's
`timing_calibration_sha256`, the smoke audit's `smoke_audit_sha256`, and
substantive `prefill_fit_review`, `tpot_fit_review`, and `arrival_review` strings.
This is an analysis record tied to measured artifacts, not a user-approval
request. Inspect the residual and warmed-overhead evidence before creating it.

```bash
python -m scripts.runs.ministral3_figure5 freeze \
  --stage-dir "$SFS_ROOT/experiments/ministral3_paper/methodology_stage/run_45367216" \
  --review-json "$TIMING_AND_ARRIVAL_REVIEW_JSON" \
  --manifest "$NEW_FIGURE5_MANIFEST"
```

Freezing validates the measured bracket and endpoint confirmations, reconstructs
the evaluation ingestion on CPU, checks full prompt identity against the saved
request map, and records input/code/predictor/score hashes and configurations.
The manifest does not derive rates from service-rate sums.

Set `FIGURE5_MANIFEST` to that new file, `SWEEP_KIND=qps`, and
`UTILITY_GROUP=all`; retain the same `SFS_ROOT`, `SERVICE_JOB_ID`, and
`PREDICTOR_RUN_DIR`. `VALIDATE_ONLY=1 bash
src/slurm/runs/ministral3_router_sweep.sbatch` checks the frozen inputs before
allocation. One subsequent sweep submission can run all 32 cells on one loaded
pool. If measured runtime requires splitting, `snapshot` contains SFS, SCORE,
RouteBalance, and shortest queue; `baseline` contains LMDeploy, Mooncake,
latency agnostic, and round robin. Both consume the identical manifest.

For collation, supply the same `FIGURE5_MANIFEST` and `FIGURE5_RAW_RUNS` as
space-separated absolute completed run roots, plus a fresh `OUTPUT_ROOT`, to
`ministral3_paper_postprocess.sbatch`. The measured mode handles Figure 5 only.
It requires every one of the 32 cells exactly once, complete requests and TTFT
telemetry, attained client arrival rates within 10%, and finite joined quality.
It copies point JSONs to derived directories, checks raw hashes before/after,
and audits the resulting OnTimeUtility summary. Labels explicitly identify the
three adaptations. The original legacy modes remain available for historical
reproduction when `FIGURE5_MANIFEST` is absent.

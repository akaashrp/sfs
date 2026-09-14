# September 14 scope extension

The selector now also covers Ministral via the additive `ministral3_latency`
adapter and the cloud worker. Total campaign: 68 cells. Qwen code and
legacy source-bound launch paths remain unchanged. The Qwen-only restriction
in the historical integration record below is superseded.

# vLLM Semantic Router latency-aware baseline

Implementation authorized and completed on September 11, 2026. The selector,
stream feedback, Qwen manifest/launch paths and Figure 5/13 reporting are wired.
Initial CPU evidence is in `experiments/vllm_sr_latency_20260911/`. The queue
refresh is in `experiments/campaign_refresh_20260911/`: 124 current CPU cases
and both launch rehearsals passed. Qwen nine-policy GPU smoke job **45769260**
is pending on cis260115p / gpuinteract. Old job **45681648** was cancelled
after validation and replacement. Final serving measurements remain gated.

CPU acceptance: **69 distinct test cases have passing latest results**,
including unmodified upstream Go differential tests, real OpenAI SSE parsing,
dispatch/cancellation, causal replay/response attribution, nine-policy stage
and subset launch tests, and multi-root quality joins/plotting. The actual
canonical manifest and three-model serving preflight passed. The source-bound
initial report is `experiments/vllm_sr_latency_20260911/cpu_validation.json`;
the current source-bound release report is
`experiments/campaign_refresh_20260911/qwen_cpu_validation.json`;
it explicitly records `gpu_smoke_passed: false`. Historical failed rehearsals
are preserved alongside their corrected passing runs. A full-window selector
CPU benchmark measured about 0.070 ms per decision, excluding network and
disk I/O; this is not a measured GPU serving-overhead result.

## Implementation and operation

- Runtime: `src/sfs_core/routing/latency_history.py`, `latency_stream.py`,
  `latency_history_scheduler.py`, `latency_warmup.py`.
- Pin, license, unmodified Go oracle sources and timing adaptations:
  `src/assets/vllm_sr_latency/README.md` and `upstream.json`.
- Frozen Qwen inputs: `experiments/vllm_sr_latency_20260911/qwen/qwen_manifest.json`.
  Preparation verified 16,000 request identities and all 16 historical reference
  cells at 7/8.3/8.6/8.9. The manifest contains five new and nine total policies.
- The upstream TPOT producer uses **total completion duration / output tokens**,
  including initial wait; the port preserves that definition. TTFT feedback is
  the first parsed SSE chunk (including role-only), not post-run log data.
  The harness uses its system-entry clock and streaming client: these are
  disclosed measurement adaptations, not full proxy-stack equivalence.
- CLI policy: `--utilities vllm_sr_latency --latency-warmup-requests PATH`.
  The Qwen manifest supplies the frozen calibration-only 32-prompt file.
- `qwen_baselines smoke --stage-dir EXISTING_STAGE` reuses measured calibration
  in a fresh output root; it does not repeat service-rate fitting or evaluation.
  It defaults to nine policies; `--policies vllm_sr_latency` isolates this smoke.
- Every GPU entry through `qwen_baselines` requires `--cpu-validation REPORT`.
  Source edits invalidate the report. A distinct source-bound selector GPU
  smoke plus existing serving/profile gates are required for evaluation.
- Slurm wrapper inputs: `QWEN_CPU_VALIDATION`, `BASELINE_MODE=smoke|sweep`,
  `QWEN_POLICIES='vllm_sr_latency'`; existing SFS_ROOT, QWEN_MANIFEST,
  QWEN_STAGE_DIR, ROUTEBALANCE_PREDICTOR_DIR and fresh OUTPUT_ROOT also apply.
  `VALIDATE_ONLY=1` runs preparation checks without GPU execution. Supplying
  `QWEN_QPS`/`--qps-values` restricts a retry to canonical points; review earlier
  complete cells first and use a new output root. No automatic blind retry.
- Sweeps containing Mooncake still require the corrected-prefill bootstrap
  timing gate via `qwen_prefill_bootstrap`, which accepts the same CPU report,
  policy and QPS subset options. The plain sweep refuses to bypass this gate.
- Collation accepts repeated `--raw-root` arguments. It audits exactly-once
  cells and causal selector feedback, retains historical curves and raw hashes,
  then performs quality joins and plotting on derived copies.


## Scope and experiment contract

Port the upstream latency-aware selector into the existing SFS experiment
harness. Policy key: `vllm_sr_latency`; paper label:
**vLLM-SR latency-aware selector adaptation**. Compare routing decisions in the
same serving setup; do not deploy the complete Semantic Router proxy stack.

Initial scope is canonical Qwen Figures 5 and 13: **four new measured cells**
at **7, 8.3, 8.6, 8.9 QPS**, Poisson arrivals, seed 69, **16,000 requests per
cell**, 4,000 per bucket, canonical holdout indices 2500–6499. Preserve request
maps, prompts/templates, budgets, prices, SLOs, model pins and serving settings.
Qwen3 0.6B/8B/32B use TP 1/1/2 on four H100-80GB GPUs in one server pool.
Retain canonical predictors on servers for consistency with other policies;
this selector must not use their predictions to choose a model.

This adds four cells to the documented 60-cell first-priority plan (total **64**), and expands the Qwen new-baseline block from 16 to **20** cells.
The Qwen comparison then has nine policies including the four existing ones.
Warm-up and functional smoke traffic are additional bounded calibration work.
Do not add Ministral, judge/MLP variants, Figure 7, SFS reruns or scheduling
ablations as part of this integration. Figures 2/3 do not need new measurements
for this selector, which fits neither the SFS wait estimator nor batch model.

## 1. Freeze upstream behavior

At implementation time, resolve and record an immutable upstream commit,
source hashes and Apache-2.0 attribution. Inspect the metric producers as well
as the selector/cache to establish timestamp origins, observation timing,
streaming token counting and error behavior. References inspected for this draft:

- [Selector](https://github.com/vllm-project/semantic-router/blob/main/src/semantic-router/pkg/selection/latency_aware.go)
- [Latency history](https://github.com/vllm-project/semantic-router/blob/main/src/semantic-router/pkg/latency/cache.go)
- [Configuration example](https://vllm-sr.ai/docs/tutorials/algorithm/selection/latency-aware/)

Freeze the documented example settings: P95 TTFT and P90 TPOT. These are chosen
settings, not upstream defaults or tuned values. For candidates with both
metrics, minimize the equally weighted mean of TTFT/min_candidate_TTFT and
TPOT/min_candidate_TPOT. Each metric is its configured percentile.
Preserve 1,000-observation windows per model/metric, linearly interpolated
percentiles, and EWMA alpha 0.3 for the 1–2-observation fallback. Preserve
candidate order and first-candidate ties, skip candidates missing required
metrics, and retain first-candidate fallback if none has usable data. Do not
add exploration, history expiration, queue penalties, quality/cost terms or
request-length prediction. Record any necessary divergence explicitly.

## 2. Implement live feedback and selector wiring

Add a small independent selector/history module under `src/sfs_core/routing/`
and wire it into `src/scripts/runs/experiments.py` and the sweep parser.
Reuse the dispatch lifecycle where practical without forcing this policy to
load RouteBalance models, fitted service rates or SFS queue snapshots.

The current `MethodologyScheduler._complete` records response completion and
usage; Figure 5 TTFT also comes from post-run wait-log joins. Post-run joins
cannot feed online routing. Implement a bounded live feedback path through the
request client/scheduler. Audit upstream streaming behavior first. Prefer an
equivalent client-visible observation path; if the canonical non-streaming
transport requires live server telemetry, label and validate that measurement
adaptation rather than silently substituting server execution time for observed
TTFT. Do not replace TPOT with total response latency or token/chunk counts
without establishing their equivalence.

Publish observations only when their requisite timestamps/usage are available,
using the pinned upstream update lifecycle. Ensure exactly-once updates,
correct request-to-model attribution, event ordering and safe concurrent
selection/update. Define one-token/empty outputs, failures, cancellations,
malformed/nonfinite metrics and late callbacks explicitly. Reject invalid
measurements with recorded reasons; disclose guards that differ from upstream.
Keep the existing end-to-end evaluation timing boundaries, including routing
and feedback overhead. Preserve existing policy transport behavior.

Log policy parameters, upstream commit, candidate order, selected model,
scores, metric values/counts, history generation and fallback reasons. Retain
a sequenced feedback event trace for replay; avoid serializing entire history
buffers for every decision. No future outcomes or unchosen-model evaluation
latencies may enter online state.

## 3. Warm-up and isolation

Before each cell, start with empty history and exercise every model using the
same deterministic calibration-only warm-up selection: initially propose
32 prompts per model, balanced across four buckets and representative prompt
lengths (96 completions per cell). Freeze this protocol before evaluation;
adjust only during calibration if it fails to provide usable measurements.
Warm-up samples populate history but are excluded from the 16,000-request
evaluation denominator. Drain outstanding work before evaluation arrivals.
Verify every model has at least three valid observations for each enabled
metric. Do not carry evaluation histories across QPS points or policies.
Reset only after drain; late callbacks must not contaminate the new history.

## 4. CPU validation before GPU allocation

- Differential-test scores, selected models and history updates against the
  pinned Go implementation on identical event fixtures. Cover percentile
  interpolation, window eviction, early EWMA, ties and missing-data fallbacks.
- Exercise real dispatch/feedback APIs with a fake streaming or telemetry
  source: out-of-order completions, cancellation, errors, one-token output,
  duplicate events, concurrent decisions, reset and late callbacks.
- Rehearse CLI parsing, manifest loading, nine-policy registration, canonical
  16k ingestion, online-log serialization, artifact provenance and collation
  using synthetic results. Confirm existing policy behavior is unchanged.
- Use `conda activate vllm`, verify the active nested vLLM import, and keep
  all scratch under the workspace. Record test evidence and source hashes.

## 5. Correct and extend the canonical Qwen pipeline

Update `src/scripts/runs/qwen_baselines.py`, its tests and
`src/slurm/runs/qwen_baselines.sbatch` to support explicit policy subsets and
the common **7/8.3/8.6/8.9** grid. The implementation inspected for this draft
still hard-codes **6/8.3/8.9/9.5** and imports a shared Ministral policy tuple.
Give Qwen its own explicit policy set so adding this selector does not expand
Ministral runs. Version new manifests; preserve old manifests and raw outputs.

The current preparer also requires exactly 16 historical reference cells.
Inventory actual canonical references at the new grid before replacing this
gate: retain every verified historical point and explicitly represent missing
comparisons. Never interpolate a measured baseline value, relabel another QPS
or silently launch SFS control reruns. Report exact-QPS comparisons only where
reference measurements exist; keep the historical full curves for context.

Parameterize smoke requirements by the policies actually being launched.
Preserve existing measured-serving/profile gates and calibration evidence;
do not call the older eight-policy smoke evidence for this new selector.
Freeze new source/parameter manifests only after the CPU rehearsal passes.

## 6. Bounded GPU validation, then evaluation

Reuse a planned four-GPU Qwen allocation where practical. Run warm-up and a
small mixed calibration-only functional smoke, then require: all requests
accounted for, usable live TTFT/TPOT for all models, causal feedback, selector
replay parity, clean drains, unchanged serving profile and recorded overhead.
Inspect routing concentration and stale histories; these may be consequences
of the upstream policy, not bugs to fix using extra exploration.

If the smoke passes, reuse loaded servers for the four 16k cells, resetting
and warming history before each cell. Preserve per-cell output roots and
allow audited partial completion to resume without repeating successful cells.
Recheck both Slurm accounts and batch/interactive QoS at submission; choose
walltime from measured smoke/past timings, and avoid duplicate active attempts.
Update the campaign registry for any newly submitted job. Diagnose failures
and reproduce on CPU where possible before retrying.

Job 45681648 was PENDING at draft time. Recheck its state, submitted script,
old grid and source gates before changing any launch path. Do not mutate a
queued script's expected contract silently. Revalidate replacement preparation
first, then deliberately manage any superseded pending job. The September 11
refresh cancelled that obsolete Qwen job after submitting validated smoke-only
replacement 45769260. The 20 final Qwen baseline cells are still gated.

## 7. Paper integration and acceptance

Extend `src/scripts/reporting/collate_qwen_baselines.py`, shared point audits
and `router_qps_sweep_summary.py` policy ordering/labels. Accept multiple audited
raw roots so a separate selector allocation can join the existing baseline
campaign without rerunning it. Require each new policy/load cell exactly once,
with 16,000 canonical request identities, complete quality/cost joins and
explicit failure accounting. Preserve source hashes and augment derived copies
only. Verify Figures 5/13 rendering includes this selector at all four measured
loads, alongside retained historical and new-baseline curves. Do not synthesize
unmeasured points or treat interpolated lines as measurements.

Complete when the CPU differential/API/launch tests pass, GPU smoke passes,
four final cells pass data/provenance audits, and derived Figures 5/13 include
the new policy. The scope and implementation status are recorded in `CURRENT_EXPERIMENT_PLAN.md`.

September 11 export-boundary recovery: Qwen 45769260 failed before model startup;
Ministral 45769252 was cancelled before allocation. Current smoke replacements
are Qwen 45779337 and Ministral 45779358. See
`experiments/campaign_export_fix_20260911/README.md` for the reproduced failure,
empty-environment validation, and corrected future launch paths.

# Current SFS experiment plan

Last scope decision: September 14, 2026. Cloud/Ministral extension in this isolated branch.

September 14: add vLLM-SR to Ministral (four cells); active total is **68**,
800,000 evaluation requests. See `scripts/cloud/README.md`. Earlier 64-cell
queue records below remain historical; this change submits no new Slurm jobs.

This is the authoritative scope document for the next campaign, superseding
conflicting counts, QPS grids, exclusions and control requirements in
`PAPER_ABLATIONS.md`, `experiments/paper_gap_resolution_20260910/README.md`, and
older preparation manifests. It records user decisions, not launch readiness.
Keep measured-artifact gates and raw results intact.

Authorized addition (September 11): the vLLM-SR latency-aware selector is
implemented in the existing harness; see `VLLM_SR_LATENCY_BASELINE_PLAN.md`.
Its four canonical Qwen cells bring the original 60-run scope to **64**.
The September 11 queue refresh is recorded in
`experiments/campaign_refresh_20260911/README.md` and `jobs.json`.
CPU validation, submitted GPU smoke and completed evaluation remain separate states.

## Experiments to run now

One run means one routing policy/variant at one load and arrival setting; it
does not mean one Slurm allocation. Reuse loaded servers across runs and drain
between runs. Calibration/smoke requests are additional bounded workloads.

| Track | Policies or variants | QPS / arrivals | Requests per run | Remaining runs |
|---|---|---|---:|---:|
| Qwen Figures 5/13: new baselines | Mooncake prefill adaptation, LMDeploy proxy adaptation, RouteBalance adaptation, SCORE, vLLM-SR latency-aware selector adaptation | 7, 8.3, 8.6, 8.9; Poisson | 16,000 | 20 |
| Ministral Figures 5/13 | SFS, SCORE, Mooncake, LMDeploy, RouteBalance, shortest queue, latency agnostic, round robin, vLLM-SR latency-aware selector | Four existing measured loads: 6.0125, 7.8625, 8.7875, 9.7125; Poisson | **8,000** | 36 |
| Qwen judge-training ablation | SFS with Flash-trained quality predictor | 7, 8.3, 8.6, 8.9; Poisson | **16,000** | 4 |
| Qwen predictor ablations | SFS with MLP quality predictor; separately SFS with MLP output-length predictor | 7, 8.3, 8.6, 8.9; Poisson | **16,000** | 8 |
| **Total: first-priority campaign** | | | | **68** |

The active Qwen Figure 5 baseline grid is **7, 8.3, 8.6, 8.9**. Do not retain
the older 6, 8.3, 8.9, 9.5 baseline grid. The deferred Figure 7 extension retains its
original 6, 7, 8 QPS arrival-pattern grid; it is a different experiment.
MMPP-2 retains the canonical high-state fraction .2 and correlation time 2 s.

Qwen model/serving settings remain canonical (0.6B/8B/32B, TP 1/1/2, H100,
existing snapshots, template, prompts, arrival seeds, SLOs and prices). For
Ministral use a fixed balanced 2,000-per-bucket holdout, common across every
policy and point. Build and audit the 8,000-request map before launch; do not
silently truncate old 16,000-request metrics or change the full collected
Figure 2 data. The measured Ministral rates came from the final Bridges serving
profile; a cloud move requires its own measured qualification.

## Controls and implementation version: explicit decision

Use the **current SFS implementation** for new experiments. The judge and
predictor ablations run the three modified SFS variants only; compare with
existing canonical Figure 5 results at identical QPS and 16,000-request budget.
Do not add ordinary-SFS control reruns, shortest-queue control runs, or matched
Pro-refit serving controls to this active matrix.

Historical April SFS controls and current-code variants are not a perfectly
isolated judge/estimator ablation. Record and disclose implementation versions;
matching request counts does not remove that confound. The user elected to
defer a consistent current-SFS paper refresh. Do not claim the old and new
routing paths are identical or silently relabel historical results as current.
If training requires a differing Pro refit, record the discrepancy rather than
automatically scheduling extra control runs. Revisit the control decision only
when needed to support a stronger causal claim.

## Prerequisites still required

- Flash-score the saved Qwen **calibration** responses, using the same rubric,
  recorded randomized aliases and explicit failure/imputation accounting.
  The completed Flash evaluation job does not supply calibration labels.
- Train/export the Flash quality artifact with canonical feature preprocessing,
  split and training recipe; retain provenance and common label eligibility.
- Train/export the selected **MLP** quality and output-length artifacts using
  calibration-only settings. Offline scalar evaluation did not export serving
  models. Validate reload parity and the actual router/server prediction APIs.
  The serving backend and calibration-only exporter are implemented. CPU job
  45750229 completed both artifacts with convergence and exact reload parity;
  source and artifact hashes were reverified. See `MLP_SERVING.md` for paths,
  metrics and the two-arm wiring. GPU smoke and launcher rehearsal remain
  required before serving evaluation.
- **MLP replaces XGBoost for serving**, following the latest user decision.
  Keep mean/median, ridge, MLP, XGBoost and refitted LightGBM in the completed
  offline scalar comparison. Do not add a serving sweep for every estimator.
  MLP tests a neural alternative to the canonical boosted-tree predictors;
  this is architectural diversity, not a claim that MLP has better holdout error.
  Use the already specified offline MLP recipe as the starting fixed recipe:
  StandardScaler, hidden layers (64,32), max_iter=500, batch_size=256,
  random_state=69, early_stopping=False, tol=1e-4. Preserve feature preprocessing,
  length log1p/inverse transform and clipping. Record convergence and validate
  finite predictions, reload parity and inference overhead on calibration data;
  any training revision must be frozen before serving evaluation and cannot
  use final holdout outcomes for tuning.
  The quality arm retains the existing SFS length predictor; the length arm
  retains the existing SFS quality predictor. Two arms times four QPS points
  remains eight runs. Older XGBoost serving specifications are superseded.
- Treat the output-length variant as **cost and simulated-workload prediction**,
  not a strictly cost-only ablation; validate all relevant prediction consumers.
- Qwen preparation now uses the common grid and nine-policy set. The prepared
  manifest is `experiments/vllm_sr_latency_20260911/qwen/qwen_manifest.json`;
  all 16 canonical reference cells were found and actual 16k ingestion passed.
  Prior pending Qwen job 45681648 is deliberately held (old grid/source contract).
  Require new CPU/source validation and GPU smoke before any replacement.
- Ministral Figure 5 now has an explicit version-2 8,000-request contract.
  Reuse the existing balanced `holdout_cache_2000` and
  `req_map_delta_seed69_holdout2000_n8000.csv`; the map name describes its
  original use, while current Poisson arrivals come from the Figure 5 harness.
  Actual ingestion and all 8,000 prompt identities were CPU-verified.
  Original scout/endpoint evidence remains unchanged, including its historical
  16,000-request evaluation declaration. A recorded exact source-diff review
  allows reuse only with fresh smoke for all eight policies. Freeze the new
  evaluation manifest after that smoke passes; no raw metrics are truncated.
- Prepare the judge/predictor launchers; Figure 7 launchers are lower priority.
  CPU-test ingestion,
  prediction/state accounting, logging and collation before GPU smoke; preserve
  corrected-prefill smoke and measured-calibration gates.
- Reuse loaded model pools, preserve raw outputs, and use fresh output roots.
  CPU validation is not proof that GPU execution cannot fail.

## Already collected / reporting work

- Qwen Pro-versus-Flash score distributions and mean per-query disagreement:
  complete. The observed-score comparison excludes four imputed/incomplete
  queries and uses 15,996 common queries from the full 16,000-query input.
- Evaluation-judge rescoring of existing Qwen routing at the four common QPS
  points: complete, including paired block intervals. It is distinct from the
  pending Flash-trained routing ablation.
- Scalar quality/output-length/USD-cost MAE and RMSE: complete. The reported
  common estimator holdout excludes 238 calibration-text duplicates; it has
  15,762 queries. Serving evaluations retain their specified complete budgets.
- Ministral Figure 2: 16,000 requests already collected. Derived reporting uses
  historical vLLM frontend TTFT, preserving raw data and diagnostic boundaries.
- Qwen Figure 2's saved statistical source: Qwen3-0.6B, 4,000 GovReport prompts,
  .87 QPS, Poisson, seed 69. Do not confuse it with the archived 8k launcher.
- Ministral Figure 3: complete full-heldout batch-fit metrics for all three
  models; R² is .9874/.9994/.9946. Preserve full-set rather than inlier-only metrics.
- Integrate and audit Figures 5/13 after serving runs finish; extend Figure 7
  only if its lower-priority runs are executed. Figure 13 uses
  the same Figure 5 outputs and needs no separate serving run. Use the same
  evaluation judge within each comparison; also report both Pro/Flash views
  of judge-training results. Preserve the canonical saved-response quality
  proxy definition and disclose its limits.

Reporting protocol/results:
`experiments/paper_gap_resolution_20260910/RESULTS.md` and its `results/` directory.
Use paired uncertainty estimates with explicit single-trace limitations;
do not fabricate data at missing policy/load cells.

## Later priorities: excluded from the active 64-run total

1. **Scheduling/serving-configuration ablations**: budget **8,000 requests per
   routing cell** for each of two proposed configurations, common across every
   compared policy within a configuration. Match counts/traffic for any direct
   cross-configuration comparison. Exact configurations and routing-policy
   coverage remain to be selected; prior all-eight-policy estimates are scenarios,
   not a fixed launch matrix. Figures 2/3 require appropriate timing collection.
   A new serving engine such as SGLang has not been selected.
2. **Consistent refresh of SFS across the paper**, including determining the
   exact required reruns, grids, request counts, overlapping cells and reusable
   artifacts. This includes the refresh question for Figures 2, 3, 5, 6, 7, 8,
   12, 13 and 14. Earlier estimates of 4, 12 or 35 additional SFS runs are
   provisional discussion, not active jobs or finalized requirements.
3. **New-baseline Figure 7 extension**, after serving-configuration ablations and
   the SFS refresh, if time and cost permit. Four new baselines at 6/7/8 QPS
   under MMPP-2 ratios 3 and 6 require 24 runs. A full Poisson comparison adds
   eight runs at 6/8 QPS; the 7-QPS Poisson cells come from Figure 5. Retain
   16,000 requests per run for integration with historical canonical curves.
   Thus this optional extension adds 32 runs (or 24 for burst-only plots).
   Existing Figure 7 supports robustness relative to its original comparators;
   do not claim performance never degrades or superiority to untested new baselines.

Do not revive new Figure 6 delta sweeps or Figure 12 lambda sweeps now.
Figures 8/14 can reuse suitable telemetry, but auditing exactly what
is needed to refresh them consistently is part of the deferred work.

## Excluded for now

- Gemma/Llama family extensions.
- `hard_prefill_tps` integration or new runs.
- Optional RouteBalance weight sensitivity.
- Additional cost-only and dedicated original-SFS/comparator control arms.

## Documentation versus submitted state

The September 11 live queue review found two old SFS sweeps: 45681645
(Ministral baseline group, 16,000 requests per cell) and 45681648 (Qwen,
obsolete grid/manifest). Both were held before changing launch inputs.
Their replacements begin with bounded GPU validation, and have zero final
serving evaluation cells. See the refresh README for submitted IDs/current
state. The other queued TransMLA jobs belong to a separate workstream.

The revised evaluation launch environments cover 20 Qwen baseline cells and
both 16-cell Ministral groups. They remain gated on the fresh smoke artifacts.
The Flash-quality and two MLP variants account for the remaining 12 cells;
they are not ready for GPU submission. Flash still needs calibration labeling
and training/export; the MLP exports exist, but their variant launchers and
matching GPU smoke remain required. Do not imply that all 64 cells are queued.

Current queue registry:
`experiments/paper_submit_20260909/submission_status.json`.
Complete active matrix and prerequisite states:
`experiments/campaign_refresh_20260911/campaign_matrix.json` and `jobs.json`.

September 11 export-boundary recovery: Qwen 45769260 failed before model startup;
Ministral 45769252 was cancelled before allocation. The first replacements
were Qwen 45779337 and Ministral 45779358. See
`experiments/campaign_export_fix_20260911/README.md` for the reproduced failure,
empty-environment validation, and corrected future launch paths.

September 12 cold-start recovery: Qwen 45779337 loaded all servers but failed
before smoke traffic because it checked snapshots before warm-up. The exact
failure was reproduced on CPU, then fixed; 28 regression checks and both
empty-environment launch rehearsals passed. Current smoke-only jobs are
**Qwen 45827995** and **Ministral 45828000**. Ministral 45779358 was replaced
before allocation to refresh its source-bound validation bundle. Active launch
paths and evidence: `experiments/campaign_cold_start_fix_20260912/README.md`,
`jobs.json`, and `launch/`. Evaluation scope and release gates are unchanged.

September 12 evaluation release: both smoke jobs completed and passed independent
artifact re-audits. The 52 Qwen/Ministral evaluation cells are now submitted in
three grouped jobs: qwen_baselines 45842578, ministral_baseline 45842572, ministral_snapshot 45842584.
See `experiments/campaign_release_20260912/README.md` and `jobs.json` for
validation, resources and current launch paths. The remaining 12 Flash/MLP
cells retain their prerequisite gates. No production source or scope changed.

September 12 predictor prerequisites are now submitted: Flash calibration labeling,
training/export and CPU variant qualification **45843211**; combined MLP
quality/length GPU smoke **45843212**. Both MLP CPU manifests and launch
rehearsals passed. All three variants retain 16000 requests per evaluation cell.
See `experiments/predictor_prereqs_20260912/README.md` for exact next actions,
validation evidence and the three prepared sweep environments.

Flash completion review (September 12): job 45843211 completed with exit 0. All 30000 calibration labels accounted for (24 imputed); imputed labels excluded by the common observed-label training mask. Export reload difference 0; validation MAE 0.153305 and RMSE 0.228761 against Flash labels. CPU serving manifest, exact 16000-request ingestion and all source/artifact hashes passed. One-arm GPU smoke job 45847352 submitted on cis260115p / GPU-shared / gpu (4 H100, 2h, 192 calibration requests), after empty-environment rehearsal and stored-script byte verification. Both accounts and both GPU QoS options were rechecked. Four Flash evaluation points remain unsubmitted until matching GPU audit passes. Evidence: experiments/predictor_prereqs_20260912/flash_completion_review.json and flash_smoke_submission.json.

Queue verification: Interactive promotion rejected: gpuinteract MaxSubmitPU=1 and MaxJobsPU=1; MLP 45843212 occupies the submitted slot. Flash remains queued in gpu batch; retry promotion when the interactive slot clears.

September 13: MLP smoke 45843212 completed successfully (17m16s). Flash smoke 45847352 was promoted to gpuinteract after that slot cleared; promotion verified in Slurm. Both MLP arms passed 192/192 requests with no request failures. Fresh sweep launch qualification and route decisions are recorded in experiments/predictor_release_20260913/.

September 13 MLP release: mlp_quality job 45865299 submitted on cis260115p / GPU-shared / gpu, four H100s, 12h. Four SFS points at 7, 8.3, 8.6, 8.9 QPS with 16000 requests each. Matching smoke completed 192/192 requests, no failures; source/artifact hashes, actual server predictor arguments and TP1/1/2 verified. Frozen launcher passed an empty-environment CPU rehearsal and its Slurm stored script matched before release. Both accounts and QoS options tested; 12h exceeds interactive maximum 8h. Evidence: experiments/predictor_release_20260913/mlp_quality_submission.json.

September 13 MLP release: mlp_length job 45865340 submitted on cis260115p / GPU-shared / gpu, four H100s, 12h. Four SFS points at 7, 8.3, 8.6, 8.9 QPS with 16000 requests each. Matching smoke completed 192/192 requests, no failures; source/artifact hashes, actual server predictor arguments and TP1/1/2 verified. Frozen launcher passed an empty-environment CPU rehearsal and its Slurm stored script matched before release. Both accounts and QoS options tested; 12h exceeds interactive maximum 8h. Evidence: experiments/predictor_release_20260913/mlp_length_submission.json.

Release status: 60 of 64 evaluation points are now submitted. MLP quality 45865299 and MLP length 45865340 each cover four 16000-request points. Only four Flash-quality evaluation points remain unsubmitted, waiting on Flash smoke 45847352 (promoted to interactive QoS). All measured-artifact gates remain enforced.

September 13 Flash release: smoke 45847352 completed in 15m58s with 192/192 requests and zero failures. Matching manifest, source/artifact hashes, actual server predictor arguments and TP1/1/2 verified. Flash-quality sweep 45890304 submitted on cis260115p / GPU-shared / gpu, four H100s, 12h: four SFS points at 7, 8.3, 8.6, 8.9 QPS with 16000 requests each. Only the quality predictor changes; evaluation judge and length predictors stay canonical. Frozen launcher passed an empty-environment rehearsal and the stored Slurm script matched before release. Both accounts and QoS options checked; 12h exceeds interactive maximum 8h, so batch preserves runtime headroom. All 64 evaluation points are now submitted, not completed. All measured-artifact gates remain enforced. Evidence: experiments/flash_release_20260913/flash_quality_submission.json.

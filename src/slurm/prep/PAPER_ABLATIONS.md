# Paper experiment jobs, updated September 8, 2026

> Historical preparation record. For the September 11 active scope, grids,
> request budgets and deferred work, use [CURRENT_EXPERIMENT_PLAN.md](CURRENT_EXPERIMENT_PLAN.md).
> This file and its old manifests do not establish current launch readiness.

Worktree: `sfs_model_family`, branch `experiments/model-family-port`.
The launch bundle and validation evidence live in
`experiments/paper_ablation_20260907/`, with the 16k revision's validation
evidence in `experiments/paper_ablation_20260908_16000/`. This document describes preparation,
not completed GPU results. Backend scheduler/serving-engine ablations and the
optional RouteBalance sensitivity check are excluded.

`jobs.json` records ten job definitions, exact environments, output paths and
dependencies. `launch/` contains CPU-preflighted submission commands and a
release command for existing held job 45365762. `validation_report.json`
records the passing checks for this revision, the original 209-test baseline validation,
reused Flash canaries and current Slurm resource dry runs. `validated_source.sha256` prevents submission after
source drift without a new validation record. None of these commands has been
used to submit or release a GPU or full CPU campaign during this preparation.

| Track | Work and request budget | Dependencies |
|---|---|---|
| Ministral Figure 2 | Existing `ministral3_wait_gof.sbatch`, 16,000 holdout requests per wait estimator; one H100 | Reuse service job 45088929 and predictor 45089699; validate the real 16k cache before launch |
| Ministral Figure 3 | Reuse `figure3_batch_fit/repair_20260906`, including its PASS audit and plots | Already CPU-completed; no replacement GPU job |
| Ministral Figure 5 | Corrected combined probe/smoke/scout stage, then eight policies in two four-policy jobs at four measured loads, **16,000 requests/cell** | Stage 45365762 remains held until launch validation is recorded; timing/scout review and manifest freeze precede evaluation |
| Judge | `paper_ablation_cpu.sbatch`, mode `judge`: Gemini 2.5 Flash on 16,000 saved Qwen groups / 48,000 candidates, paired against existing Pro scores | Audited copied inputs; API model lookup and four grouped canaries; retries, alias logging, distributions and per-query disagreement |
| Accuracy/cost estimators | Same CPU launcher, mode `estimators`: existing SFS, mean, median, Ridge, MLP, XGBoost and refitted LightGBM, using the full 16,000-query holdout | Canonical Qwen features and prices; fixed hyperparameters; common split and duplicate-text exclusion; scalar quality, length and USD-cost MAE/RMSE |
| Canonical Qwen baselines | `qwen_baselines.sbatch`, modes `calibrate` then `sweep`: Mooncake, LMDeploy, RouteBalance and SCORE; **16,000 requests/cell** | Native Qwen MiniLM/FAISS artifact, fresh timing inputs, eight-policy smoke, strict canonical request-map/SLO checks |
| Qwen figure integration | `scripts.reporting.collate_qwen_baselines` | Audits all new cells; copies and joins raw results into a separate derived directory; retains original full curves, adds four measured baseline points |

All relevant policy plots include `hard` (SFS), `score`, `mooncake_prefill`,
`lmdeploy_proxy`, `routebalance`, `shortest_queue`, `latency_agnostic`, and
`round_robin`. Labels explicitly identify paper adaptations. Figure 13 uses
the same QPS outputs. Router-only and judge-only changes do not require
repeating the unchanged canonical Qwen wait/batch-estimator traces.
Every evaluation workload now selects 4,000 prompts per bucket, indices
2500–6499. Calibration, smoke and capacity-scout sample sizes are unchanged;
their purpose is fitting and validation, not full holdout evaluation.

## Qwen controls and calibration

The four baseline loads are 6, 8.3, 8.9 and 9.5 QPS, selected from existing
successful canonical Poisson runs. The preparation manifest records the exact
original policy/point files; later PK-M/G/1 and V100/regime experiments are not
used as substitutes. Original curves retain their other measured points.

Qwen uses 0.6B/8B/32B on four H100-80GB GPUs, TP 1/1/2, the pinned Qwen
snapshots and template, BF16/auto, 131072 model context, 65536 client context,
32768 prompt/batch-token limits, 8192 completion cap, 512 sequences, chunked
prefill, no prefix cache, memory utilization .9, temperature 0/top-p 1,
the canonical Qwen system prompt and `enable_thinking=false`. Predictor
weights match the existing canonical copies byte for byte.

The shared 4000-per-bucket cache was retokenized after the original Qwen runs.
The launch bundle therefore has its own **frozen canonical cache**, recovered
from unchanged larger holdouts and checked against every original request-map
token count. Its legacy routing counts are preserved explicitly; they are not
presented as newly computed tokenizer counts. Serving payloads retain the
canonical non-thinking template. Existing-only/frozen-cache flags prevent
automatic retokenization in preflight and GPU jobs.

Old Qwen service metrics reference missing batch CSVs. The prepared combined
calibration job collects 64 isolated prefill probes plus 512 calibration
responses/model with bounded concurrency, fits the prefill and TPOT models,
then smoke-tests all eight policies with 192 calibration requests/policy.
Those measured per-instance rates are speed inputs, **not router capacity**.
No final evaluation requests are used to tune the baselines.

## CPU/API work

`data_16000_20260908/data_audit.json` verifies 30,000 Qwen calibration scores
and 48,000 holdout scores. Imputation counts are recorded in that audit.
The original 8k preparation is preserved under `data/` and is rejected by
the current evaluation validator. The native Qwen RouteBalance predictor uses
8,987 training and 1,013 validation prompts, excluding cross-split duplicate
text. Its files and validation metrics are saved in `qwen_routebalance_predictor/`.

The scalar estimator comparison reads all 16,000 holdout query groups and
excludes 238 groups whose text duplicates calibration text for every estimator:
15,762 queries / 47,286 candidate outcomes remain in the reported error metrics.
This exclusion does not change serving-run request counts.
Evaluation reports identify the saved SFS predictor separately from refitted
learners. All alternatives use the same deployed feature builders and
prompt-token convention; training targets/clipping are recorded in the report.

Judge scores are paired by `(bucket, example_id, model_label)`. Candidate
ordering is randomized, realized aliases are saved, and imputed pairs are
excluded from observed-score comparisons. Query-level disagreement averages
the three absolute candidate differences, then averages complete queries.
Differences include presentation-order and judge-sampling variance. The rubric
is unchanged; the Gemini payload now uses `config.system_instruction` because
the live Flash endpoint rejected `role=system`. This is an API-format repair.

## Validation and launch boundary

The original metadata failure was caused by `load_instances` discarding
`serving_profile`. The loader now retains top-level provenance, and CPU tests
exercise the production shell renderer, real loader, and stage contract.
The stage performs the same check before any server is loaded. Figure 2 also
uses the short job-local IPC directory that fixed the earlier socket failure.

Validation includes actual 16k Qwen/16k Ministral request construction, all
96,000 candidate score joins, real saved-predictor inference, native predictor
reload/inference, server CLI parsing (only hardware discovery supplied on CPU),
policy/state/error-path tests, collation/diagnostic-file separation, shell
syntax, Slurm resource dry runs, and small live Gemini grouped requests.
The CPU memory request is 30000M/16 cores to satisfy RM-shared's 2000M/core cap.
Figure 2 reserves eight hours: its two 16k arrival periods alone take about
200 minutes at the selected measured rate. The Ministral Figure 5 sweep uses
two 48-hour jobs, one per existing policy group, because GPU-shared's partition
QoS caps each job at 48 hours. Both feed the same collation and frozen manifest.
Judge/estimator CPU reservations are 16 hours to retain headroom after doubling
their workloads. Qwen baseline counts and time limits
are unchanged. These are time limits, not predicted runtimes.

The rehearsal initially rewrote the shared cache's Alpaca file before the
implicit rebuild was detected. Both rehearsal processes were stopped. The file
was restored using the same conversion that exactly reproduces the three
untouched version-2 bucket files; the recovery method, preserved intermediate,
and hashes are in `cache_rehearsal_recovery/recovery_audit.json`. No raw model
responses or existing router-result files were modified.

CPU/API PASS does not establish GPU timing, capacity, or hardware reliability.
Final Ministral evaluation requires the measured scout and timing review;
Qwen evaluation requires the calibrated-model checksums and all-policy GPU
smoke. Do not release a duplicate interactive/normal job concurrently, and
do not submit an evaluation launcher with a missing dependency artifact.

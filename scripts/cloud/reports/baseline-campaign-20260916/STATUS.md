# Cloud campaign status — 18 September 2026

Live operational handoff, refreshed after the Ministral SFS diagnosis, the SCORE
multiplier sweep and the requeue that followed both. It supersedes the 16 September
capture. Refresh it after any restart or new completion.

## Current operational state

- Two pools are evaluating on Vast, both on `repo-v7` (`9b8d040`), both released.
  - GPUs 0-3: `score-qwen-tuned-20260918` — the four Qwen SCORE cells at 6/7/8/8.3
    QPS by 16,000 requests, rerun at SCORE's tuned Lagrange multiplier 5e-2.
  - GPUs 4-6: `sfs-score-ministral-20260918` — the three Ministral SFS cells rerun on
    a corrected batch clock, then the three Ministral SCORE cells at the tuned
    multiplier, 8,000 requests per cell.
- Queued behind them, each chained on its predecessor's COMPLETE status:
  unchunked lane A (qualify + 4 SFS cells, GPUs 0-3) and lane B (qualify + 4 SCORE
  cells, GPUs 4-7) on `fcfs-repo-v7` (`235136c`); then the predictor ablations
  (lane A: mlp_length 9 cells and flash_quality hard/score 6; lane B: mlp_quality 9
  and flash_quality latency-agnostic 3, all at 7/8/8.3); then chunk-8192 (12 cells)
  on GPUs 0-3 and the staleness sweep (4 cells) on GPUs 4-7.
- Built and committed but not yet queued: the constrained-capacity serving
  configuration `qwen-kv-constrained` (12 cells), last by the user's ordering.
- 60 audited cells stand: 32 canonical and 28 unchunked FCFS.

## Two defects found and fixed today

### The Ministral batch-latency feature set (`scripts/cloud/reports/ministral-sfs-gap-20260918`)

`pool.py` copied the six batch-latency coefficients out of the Ministral fit and
dropped the fit's `"feature_set": "cross_term"` declaration, so vLLM fell back to
`legacy` and multiplied the prefill x processed-context coefficient by the sum of
squared context lengths. Measured against 2,090,402 real engine steps, the simulated
per-batch clock ran 107x to 257x fast. SFS's TTFT feasibility filter then found no
feasible candidate on 99.78 / 99.88 / 99.88 percent of decisions (Qwen at 8.3: 0.29
percent), so the policy degenerated to a squared-context balancer and lost to every
baseline on every Ministral cell.

Both families now build engine arguments through `service_metrics_config.build_simulation_args`,
which emits the declared feature set and the option that matches it; an unrecognised
declaration is an error rather than a silent default. Because a mismatch of this kind
still runs and still audits clean, every pool now replays its own coefficients against
the batch statistics its engines just wrote and refuses to evaluate when the median
predicted/actual batch time leaves [0.5, 2.0] (`scripts/cloud/batch_residual.py`).

Confirmed on the live pool: median predicted/actual is now 1.024 / 0.991 / 0.997 for
3B / 8B / 14B over 41,978 batches, and the SFS smoke at 8.7875 QPS attains 100.00
percent of TTFT SLOs with zero violation milliseconds, against 92.19 percent and
2,068.9 ms on the broken pool. SCORE, which does not route on the simulator, is
unchanged (78.65 vs 76.04 percent) — the control that isolates the cause.

Blast radius: the three `ministral-hard-*` cells only. Qwen's coefficients are legacy
fits and its arguments are unchanged apart from the now-explicit flag; the Bridges
Ministral launcher always went through the correct builder.

### SCORE's Lagrange multiplier (`scripts/cloud/reports/score-lambda-sweep-20260917`)

SCORE's lambda and the campaign objective's cost weight were the same number, so the
published method ran at 5e-4, where a second of predicted latency is worth half a
thousandth of a quality point. It met 9.70 percent of TTFT SLOs. A four-level probe
(2,000 requests at 7 QPS, all scored at the canonical 5e-4 so the levels are
comparable) gives 9.70 / 77.35 / 97.30 / 98.25 percent attainment and 0.0571 / 0.3055
/ 0.3056 / 0.2750 Pro OnTimeUtility at 5e-4 / 5e-3 / 5e-2 / 5e-1.

`--score-lambda-weight` now carries SCORE's routing multiplier alone, and 5e-2 is
adopted for every SCORE cell of the campaign. Evaluation stays at the bundle's 5e-4
for every method including SCORE, so one objective scores the whole grid.

## Superseded, not deleted

`state/superseded-20260918/` holds the five retired completed-ledger entries with the
reason for each: `qwen-score-6` and `qwen-score-7` (routed at the untuned multiplier)
and the three `ministral-hard-*` cells (wrong batch-time feature set). Their cell
directories are renamed `...-superseded-<reason>`, and the alert watcher skips those,
skips tuning probes, and skips pools stopped deliberately.

## Headline results that stand

Qwen SFS leads every baseline at every canonical rate: OnTimeUtility 0.5402 / 0.5209
/ 0.5043 / 0.4941 at 6 / 7 / 8 / 8.3 QPS with 96.6 / 95.4 / 94.3 / 93.6 percent TTFT
attainment. The strongest external baselines are Mooncake (0.5103 at 7) and LMDeploy
(0.3961 at 8.3). Ministral SFS is being remeasured; its earlier numbers are void.

## Open items

- The dead analytic TTFT estimator in `experiments.py` hard-codes the legacy batch-time
  form. It has no load-bearing callers, so it is a trap rather than a bug, but it should
  be guarded or deleted at the next protected-source edit.
- `fd253b4` records the overlay-wide SCORE multiplier in `qualification.json`; it lands
  at the next deployment rather than disturbing the running pools.
- The qwen3-0.6b batch fit over-predicts by about 18 percent (median 1.178). It is
  inside the audit bound, conservative in direction, and unchanged across every audited
  Qwen pool; worth revisiting only if SFS is ever tightened on the small engine.

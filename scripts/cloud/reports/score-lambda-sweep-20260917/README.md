# SCORE Lagrange multiplier (2026-09-17)

**Recommendation: run every SCORE cell at `--lambda-weight 0.05`.** The current 0.0005 is not
competitive: it is a corner of the objective where the latency term is negligible, SCORE sends every
request to the two largest engines, and the baseline collapses.

**These four points are a tuning probe, not reportable evidence.** Each is 2,000 requests, carries
`data_role: "tuning_probe"`, has an id prefixed `probe-`, and its receipt is written to
`<state>/completed-probes`, never the canonical completed ledger. They exist only to pick lambda; the
SCORE cells themselves are then rerun in full at 16,000 requests under
`scripts/cloud/sfs-score-campaign-20260917.json`.

Machine-readable data: `score-lambda-sweep.json`, produced by `ops/cloud/score_lambda_report.py`.

## 1. What was varied, and what was not

SCORE (Lakha, Yu, Shahout, ICLR 2025 SLLM workshop, *Faster, Cheaper, Just as Good*) routes to

    d_t = argmax_i  Qhat_i - lambda ( w_C c_i Shat_i + w_L W_i + w_L s_i Shat_i )

and says lambda is what "enforces the cost and latency constraints, while w_C and w_L determine the
relative importance". The paper never states a lambda, and its scales are not ours: quality on 1..5
and outputs up to 512 tokens at 1 request/second over two models, against quality in [0, 1], outputs
up to 8192 tokens and 6-8.3 QPS over three models here. lambda is therefore the one quantity tuned.
`w_C = w_L = 1` is kept, and nothing else in the SCORE implementation, the calibration, the
remaining-length rule or the workload was touched.

Overlay: `scripts/cloud/score-lambda-sweep-20260917.json`
(sha256 `f6fa4af1bc86e8a7aab1077a64b525fd302acdd31aa275aa8b7ed066da1c5bb3`), kind `score_lambda_sweep`,
four canonical Qwen `score` cells at 7.0 QPS x 2,000 requests, one per
lambda in {0.0005 (control), 0.005, 0.05, 0.5}, with the SFS/SCORE overlay's remaining-length block
(`qwen3-0.6b=running_all_prompt_bin_q50`) unchanged.

Run on Vast host `b1a221322e90`, GPUs 0-3 (4x H100 80GB), source commit `785137b`
(source pin `e54c15b97acd...`), bundle `4e36359aaa8d...`, pool
`/workspace/sfs/state/baselines/score-lambda-sweep-20260917`, qualification `3dac91fe9feff1e7...`
released with a review that records the probe status. Calibration: 512 loaded requests per model,
0 failures; SCORE proxy inputs within a few percent of the previous SCORE pool on the same GPUs.
Every probe: 2,000/2,000 requests succeeded, 0 failed, TTFT SLO coverage 100 percent.

The four probes see exactly the same prompts: the holdout pool is mixed and shuffled from the full
per-bucket limit with seed 69 and only then truncated to `num_requests`, so `req-0 .. req-1999` are
the *first 2,000 requests of the canonical 16,000-request sequence*. Checked, not assumed: all 2,000
rows match the canonical `qwen-score-7` point on (bucket, prompt tokens, TTFT SLO) exactly.

## 2. Per-lambda results

| lambda | TTFT SLO attainment | Pro OnTimeUtility | median TTFT | p90 TTFT | mean TTFT | 0.6B / 8B / 32B | mean realized cost | realized QPS |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 0.0005 (current) | 9.70 % | 0.0571 | 35.81 s | 121.76 s | 46.81 s | 0 / 834 / 1166 | 712.9 | 4.00 |
| 0.005 | 77.35 % | 0.3055 | 30.0 ms | 62.16 s | 11.11 s | 1057 / 507 / 436 | 376.6 | 5.13 |
| **0.05** | **97.30 %** | **0.3056** | **20.1 ms** | **83.1 ms** | 111.5 ms | 1691 / 302 / 7 | 237.9 | 6.41 |
| 0.5 | 98.25 % | 0.2750 | 20.2 ms | 72.1 ms | 78.7 ms | 1845 / 150 / 5 | 231.3 | 6.32 |

Attainment is system-entry end-to-end TTFT over all 2,000 requests (the producer's own summary; also
recomputed from the rows, identical). Routing counts are the answering model of every request.
Realized QPS is 2,000 / elapsed; it falls below the offered 7.0 exactly when the backlog grows, which
is the load symptom of the collapse rather than an arrival-rate setting.

### OnTimeUtility: which multiplier scores the points

`scripts.cloud.collate.observed_utilities` scores a point with the multiplier that point was *routed*
under, so scoring each probe with its own lambda would compare four different objectives. Both
numbers are given; the column above is the first:

* **Pro OnTimeUtility (reported)** - the campaign's own metric: mean over routed requests of
  `quality - 0.0005 * actual_cost`, gated by the system-entry TTFT SLO, over the frozen observed-judge
  cohort. Every reportable cell of the campaign (SFS, SCORE and every baseline) is scored at this same
  0.0005, so this is the only column comparable with the campaign and across the four probes.
* **Pro OnTimeUtility at the as-run lambda** - the same expression with each probe's own routing
  multiplier: 0.0571, -0.4991, -9.6209, -100.4396. It is reported for completeness only; the cost term
  is in USD per million tokens, so at lambda = 0.5 it swamps a quality in [0, 1] and says nothing about
  serving quality.

Method, exactly: `observed_utilities` was reused unchanged on the frozen quality index. The 2,000-request
subset does break the campaign's full-cell join, so the frozen `qwen/request_map.csv` was restricted to
the 2,000 routed ids (valid because those ids are the canonical sequence's first 2,000, verified above)
and passed in; the observed-query cohort stayed the frozen full-bundle cohort of 15,996 queries. Each
probe scored 1,999 of its 2,000 requests, the one exclusion being a query outside that cohort. Flash
OnTimeUtility, same definition: 0.0608, 0.3391, 0.3546, 0.3272 - the same ordering.

### Predicted versus actual latency

SCORE's own terms for the engine it selected, against what the request then saw (mean signed error,
positive = SCORE predicted more than happened):

| lambda | predicted total latency - realized latency | mean abs | predicted wait - realized pre-prefill wait | mean abs |
|---:|---:|---:|---:|---:|
| 0.0005 | +24.52 s | 41.39 s | +49.96 s | 52.61 s |
| 0.005 | -4.24 s | 10.27 s | -0.29 s | 4.85 s |
| 0.05 | +4.78 s | 6.78 s | +5.33 s | 5.33 s |
| 0.5 | +5.98 s | 7.69 s | +6.53 s | 6.53 s |

`predicted_total_latency_ms` is `W_i + s_i Shat_i`, the quantity SCORE actually models, compared with
the request's realized end-to-end latency. The wait column compares SCORE's `waiting_time_ms` with
system-entry TTFT less measured prefill; read it with care, because SCORE's wait term includes the
engine's decode backlog, which under chunked prefill does not delay a new request's first token. That
is why the wait term reads ~5 s on the 0.6B at lambda = 0.05 while the measured median TTFT there is
20 ms: the term is pessimistic as a TTFT predictor by construction. No rows were missing the terms.

## 3. Reading

* **0.0005 is a corner solution.** With quality in [0, 1] and the latency terms in milliseconds,
  `lambda * (W_i + s_i Shat_i)` is worth a few thousandths of a quality point, so SCORE always buys the
  larger model: 1166 requests to the 32B, 834 to the 8B, none at all to the 0.6B, mean predicted quality
  0.819 - and 9.7 percent attainment with a 35.8 s median TTFT. The multiplier was chosen for SFS's
  objective, whose latency handling is structurally different; carrying it into SCORE is what produces
  the collapse the campaign recorded (6.5 percent at 6 QPS, 3.8 percent at 7 QPS).
* **0.005 already fixes the median but not the tail.** Attainment 77 percent and a 30 ms median, yet a
  p90 of 62 s: it still sends 436 requests to the 32B and 507 to the 8B, and the backlog those build is
  paid by the requests behind them. Realized throughput 5.13 QPS against 7.0 offered.
* **0.05 is the interior optimum.** Highest Pro OnTimeUtility (0.3056), 97.3 percent attainment, a p90
  of 83 ms rather than tens of seconds, and realized 6.41 QPS - the closest of the four to the offered
  rate. It still routes 302 requests to the 8B and 7 to the 32B, so it is trading model size for
  latency rather than abandoning the large engines.
* **0.5 over-penalizes.** Attainment rises trivially (98.25 vs 97.30) while utility falls to 0.2750:
  92 percent of requests go to the 0.6B and mean predicted quality drops to 0.335. Past 0.05 the
  objective is buying latency it no longer needs with quality it cannot spare.
* 0.005 and 0.05 are within 0.00014 of each other on utility - a tie at this sample size. The tie is
  broken on everything else: attainment (97.3 vs 77.4), p90 (83 ms vs 62 s) and realized load.

## 4. Caveats

1. **The chosen lambda must be used for every SCORE cell.** Adopting 0.05 means rerunning
   `qwen-score-6`, `qwen-score-7`, `qwen-score-8` and `qwen-score-8.3` at 0.05; a grid that mixes
   multipliers across rates is not a baseline.
2. **Probed at one rate, on one family.** The sweep is Qwen at 7 QPS. 7 QPS sits in the middle of the
   6-8.3 grid and the ordering is not marginal, but the Ministral SCORE cells were not probed; using
   0.05 there is an extrapolation, and a Ministral probe of the same shape would settle it.
3. **2,000 requests, not 16,000.** Shorter runs build less backlog, so the absolute attainment of a
   probe is optimistic relative to a full cell (the 16,000-request qwen-score-7 cell at 0.0005 reached
   3.8 percent where this probe reached 9.7). The comparison between probes is sound - same prompts,
   same pool, same calibration, one qualification - but a probe number must not be quoted as a cell
   result.
4. **Not tuned:** `w_C = w_L = 1`, the per-request latency limit, the cost budget, the quality and
   length predictors, the calibration and the remaining-length rule are all untouched. Only lambda moved.

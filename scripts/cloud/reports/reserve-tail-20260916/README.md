# Remaining-decode target tail errors: audit, characterization and held-out candidates

Side investigation on the SFS/SCORE remaining-output-length ("reserve") estimate that feeds the
wait-time simulator. Read-only over saved evidence, plus two bounded single-model diagnostics on
Vast GPU 7 (Qwen3-0.6B, Qwen3-8B). No production runtime, predictor, coefficient, SLO or campaign
state was changed.

Evidence files live in `evidence/` (JSON) and were produced by `evaluate_tail.py`
(controller, offline), `replay_candidates.py` (Vast, native simulator, CPUs 84-95 only) and
`gpu7_analyze.py` (Vast, CPU only). `gpu7_single_model_run.py` is the GPU-7 diagnostic runner.
Raw inputs: the five completed Vast control points
(`controls/{qps8p6,qps8p75,qps8p6-repeat1,qps8-lane-a,qps8-lane-b}`), the 57 saved read-only
snapshots (`diagnostics/{qwen-repeat1,qps8-lane-a,qps8-lane-b}-snapshots`), the 30,000 calibration
outputs, and the GPU-7 runs under `/workspace/sfs/state/diagnostics/reserve-tail-20260916/`
(`gpu7-qwen3-0.6b/`, `gpu7-qwen3-8b/`; the controller mirror of that tree lagged during this work,
so the analysis ran on Vast and the evidence JSON was copied into `evidence/`).

## 1. Re-audit of `reserve-alternatives-20260916`

What that README established, and what it did not:

- Sample size. Its tables come from 15 snapshots of one run: 965 running-decode observations
  from 960 distinct requests, but the 0.6B rows are 167 observations of 163 requests captured at
  five instants of one 31-minute run. Two further snapshot sets (lane A and lane B, 21 files each,
  42 in total, captured before that README was committed) were analysed afterwards
  (`conditional-comparison.json` in each lane directory) but never reported. Pooling all 57
  snapshots gives 2,878 observations (0.6B 462, 8B 954, 32B 1,462); the per-model conclusions
  below are stable across the three runs (`evidence/tail-characterization.json: by_model_run`).
- "Conditional median improves actual token error for all three models" is true for MAE only.
  On the same 0.6B rows the conditional median's median absolute error is 87 tokens but its p90
  absolute error is 5,524 tokens and 35/167 errors exceed 1,000 tokens
  (`qwen-repeat1-snapshots/conditional-median-error-distribution.json`, computed at the time and
  not included in the README). MAE hides that the tail is untouched; see section 3.
- The hindsight replay ("oracle") assigns every running request its eventual length. It is an
  upper bound on what any remaining-length rule can achieve, not a deployable estimator, and the
  replay is a hypothetical probe through the simulator, not measured TTFT. The README says so; the
  numbers were nevertheless read as if a 4x wait-estimate improvement were on the table.
- `conditional-lookup-summary.json` reports `PASS_OFFLINE_LOOKUP`. That gate checks table build
  and lookup latency (65 ns), not predictive quality; its own `limitations` field says so.
- Model-only conditioning was compared against the current rule on snapshots whose reserve was
  already warm (532-845 tokens). It was not compared against prompt-length or prediction
  conditioning, and no held-out evaluation was performed (the calibration set is disjoint, so the
  model-only table is legitimately out-of-sample, but no alternative was tested).
- Not overstated: the one-token-remaining mechanism (`_snapshot_output_target` floors at
  `generated+1` once the prediction plus reserve is exceeded) is real, and its simulator effect
  (premature simulated completion of long-running requests) is correctly described.

## 2. Where the tail comes from (measured)

Sources: `evidence/tail-characterization.json` (57 snapshots joined to eventual lengths) and
`evidence/predictor-bias.json` (80,000 requests from the five Vast controls).

Current rule, remaining-token absolute error on real snapshots:

| Model | n | MAE | median | p90 | p95 | p99 | coverage | exhausted obs | final length capped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.6B | 462 | 2,218 | 1,162 | 5,820 | 6,490 | 7,332 | 29% | 47% | 48% |
| 8B | 954 | 292 | 182 | 572 | 673 | 3,287 | 57% | 11% | 1.5% |
| 32B | 1,462 | 350 | 220 | 619 | 840 | 3,259 | 37% | 16% | 1.8% |

Decomposition of the current rule's error (share of total absolute error / of under-estimated tokens):

- 0.6B: 95% of observations are govreport prompts and 80% have prompts of 8,192 tokens or more;
  requests whose final length hits the 8,192 cap are 48% of observations but 79% of absolute
  error and 86% of under-estimation. These are degenerate long generations of the 0.6B model on
  long summarization prompts (6.4% of the offline holdout govreport outputs and 8.2% online reach
  the cap). Prediction-exhausted observations (target-generated <= 1) are 47% of rows and 55% of
  error. The router additionally has no prediction for 5.6% of 0.6B rows (`predicted_output_tokens
  = 1.0`, the `output_lengths.get(target_id, 1.0)` default), 7% of error.
- 8B and 32B: the p50-p90 error (200-600 tokens) is dominated by predictor bias, not by the
  reserve rule. The length predictor's mean head predicts about 40% of the true length for the two
  long-output buckets on both models (32B writingprompts: predicted p50 438 vs calibration-data p50
  1,047 and online p50 1,042; 8B govreport: 262 vs 768/773). The calibration data the predictor was
  trained on has the same lengths as observed online, so this is predictor under-fit, not a
  serving difference. The warm adaptive reserve (a 0.65-0.85 quantile of past residuals, 550-650
  tokens) is what currently compensates for this bias. The p99 tail (about 3,300 tokens) comes from
  the rare capped generations: 1.5-1.8% of observations carry 19-21% of absolute error, 29-38% of
  under-estimation and 55% of context-weighted under-estimation on 32B.
- Exhausted rows: on 8B/32B, 11-16% of observations carry 28-30% of absolute error and 42-56% of
  under-estimated tokens; on 32B they carry 72% of context-weighted (KV-hold) under-estimation.

Wait-estimate error at the request level (router estimate vs measured engine TTFT, all requests):
the 0.6B instance under 8.6 QPS has TTFT p50 14.9-44 s against estimates p50 0.9-9.7 s (error p50
12-20 s, 63-72% of requests under-estimated by more than 1 s); 8B and 32B errors are 0.6-1.9 s MAE
with p90 0.5-3.6 s. At 8 QPS (lanes) all errors are small (0.6B p99 6-12 s from a few backlog
episodes). The wait-estimate tail is therefore a 0.6B-under-overload phenomenon plus moderate
8B/32B under-estimation; the token tail above is the input side of it, but the run-level TTFT gap
on 0.6B cannot be attributed to the reserve rule alone from these data (see section 5).

## 3. Candidate estimators, held-out

`evaluate_tail.py` fits survival tables (quantile of total length among training lengths
strictly greater than tokens generated so far; a hazard table is the same object) with three
conditioning schemes and back-off, and evaluates on (a) the 57 real snapshots and (b) a synthetic
held-out set where every online request is observed at eight evenly spaced generation points,
token-weighted to mimic the snapshot population. Held-out discipline:

- `cal_*`: fitted on the 30,000 calibration outputs only (prompt sets disjoint from all evaluation
  requests). `cal_model` = the README's model-only table; `cal_prompt` conditions on prompt-length
  bin (<128, 128-512, 512-2k, 2k-8k, 8k-32k, >=32k tokens); `cal_bucket` uses the dataset label
  (an oracle feature, upper bound only).
- `online_*`: fitted on the five pooled Vast controls with 5 prompt-level folds (fold =
  sha256(example id) mod 5); every evaluated observation comes from a prompt absent from its table.
  `online_prompt_pred` conditions on prompt-length bin and router-prediction bin (with p<=1 as its
  own bin), backing off to prompt bin, then model.
- Quantile targets q50/q65/q80; `exhausted_only_*` keeps the current target until the current rule
  says <=1 token remains, then switches.

Real snapshots, remaining-token absolute error (full table in `evidence/candidate-evaluation.json`):

| Model | Rule | MAE | median | p90 | p95 | p99 | cov. | backlog ratio (median snapshot) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0.6B | current | 2,222 | 1,190 | 5,794 | 6,481 | 7,454 | 29% | 0.16 |
| 0.6B | cal_model_q50 (README) | 1,190 | 94 | 5,522 | 7,496 | 7,831 | 64% | 0.84 |
| 0.6B | cal_prompt_q65 | 1,211 | 80 | 5,711 | 7,162 | 7,512 | 78% | 1.03 |
| 0.6B | online_prompt_pred_q65 | 1,195 | 73 | 5,209 | 6,389 | 7,323 | 81% | 1.15 |
| 0.6B | exhausted_only_cal_prompt_q65 | 1,247 | 171 | 5,431 | 6,204 | 7,390 | 75% | 0.87 |
| 8B | current | 315 | 202 | 591 | 753 | 3,287 | 65% | 1.22 |
| 8B | cal_model_q50 (README) | 231 | 172 | 488 | 595 | 951 | 62% | 1.10 |
| 8B | cal_prompt_q50 | 175 | 113 | 360 | 533 | 928 | 55% | 0.98 |
| 8B | online_prompt_pred_q50 | 170 | 110 | 336 | 475 | 992 | 52% | 0.93 |
| 8B | online_bucket_pred_q50 (label oracle) | 169 | 118 | 318 | 421 | 997 | 51% | 0.92 |
| 8B | exhausted_only_cal_prompt_q65 | 263 | 194 | 574 | 672 | 989 | 72% | 1.31 |
| 32B | current | 358 | 229 | 662 | 873 | 3,259 | 48% | 0.89 |
| 32B | cal_model_q50 (README) | 289 | 184 | 649 | 759 | 1,445 | 55% | 1.02 |
| 32B | cal_prompt_q50 | 265 | 163 | 560 | 749 | 1,435 | 53% | 0.99 |
| 32B | online_prompt_pred_q50 | 254 | 150 | 505 | 725 | 1,450 | 51% | 0.96 |
| 32B | online_bucket_pred_q50 (label oracle) | 241 | 144 | 447 | 643 | 1,632 | 51% | 0.90 |
| 32B | exhausted_only_cal_prompt_q65 | 306 | 212 | 611 | 816 | 1,327 | 56% | 1.03 |

Backlog ratio = sum of estimated remaining tokens over sum of actual remaining tokens for the
running set of a snapshot (what the simulator's decode occupancy sees), median over snapshots.

Findings:

- 8B and 32B: the tail is reducible. Prompt-length conditioning (deployable: the engine knows
  `num_prompt_tokens`) cuts p90 by 39%/15% and p99 by 72%/56% relative to the current rule, and
  beats the model-only table at p90 (8B 360 vs 488) because prompt length separates the long-output
  buckets that the predictor under-estimates. Adding the router prediction bin adds little beyond
  prompt length (8B p90 336 vs 360; 32B 505 vs 560); the dataset-label oracle gains a further
  10-15% at p90, so most of the achievable gain is available without labels. The synthetic held-out
  set agrees (8B token-weighted p90 630 -> 392, p99 3,583 -> 1,999; 32B 641 -> 522, 4,607 -> 1,616).
- 0.6B: no point estimate removes the tail. Every candidate leaves p90 at 5,200-5,800 tokens and
  p95/p99 at 6,200-7,800, and several make p95 worse than the current rule (the survivor median
  becomes 8,192 once a long-prompt request passes about 1,500 tokens, so an uncapped finish is
  over-estimated by thousands). The conditional distribution given (0.6B, prompt >= 8k, generated
  > 1.5k) is bimodal: cap or stop soon. Quantile choice only moves error between under and over.
  What conditioning does fix for 0.6B is the aggregate: the running-set backlog ratio moves from
  0.16 (current) to 0.87-1.15, i.e. the simulator's total decode occupancy becomes right on average
  even though individual requests do not. Whether that aggregate correction improves wait estimates
  is the GPU-7 question in section 4.
- `exhausted_only_*` (switch only once the current rule is exhausted) removes the p99 tail on
  8B/32B (3,287 -> 989; 3,259 -> 1,327) while leaving MAE/p90 close to current; it is the smallest
  behavioural change and never touches requests still inside their prediction.
- Rolling floor (`max(target, generated+reserve)`) does not help the tail on any model and inflates
  8B/32B backlog by 16-56%; confirmed with the larger sample.

## 4. GPU-7 diagnostic (measured)

Design. `gpu7_single_model_run.py` runs one canonical engine on GPU 7 (CPUs 84-95) behind the
canonical hard-policy router with a single instance, replays the requests that the `qps8p6-repeat1`
control routed to that model (same arrival order, at a fixed rate), records the router's per-request
wait estimate and the engine's measured TTFT, and captures a read-only scheduler snapshot every 1 s.
With one instance routing is fixed, so the estimate cannot influence the measured TTFT: the run
measures estimator accuracy, not policy outcome. `replay_candidates.py --mode dispatch` then aligns
every request with the last snapshot published before its engine enqueue (staleness p50 0.5 s, p90
0.9 s), substitutes each candidate's targets for the running-decode requests, runs the deployed native
simulator for that request's own prompt and compares the estimate with the request's measured TTFT.
The replayed current rule reproduces the router's logged estimate (median |diff| 96 ms, MAE 0.9 s from
the pending-dispatch gap), which validates the alignment. Raw outputs are under
`/workspace/sfs/state/diagnostics/reserve-tail-20260916/` (`gpu7-qwen3-0.6b/`, `gpu7-qwen3-8b/`,
`replay-*.json`, `analysis/`); `gpu7_analyze.py` produced the `evidence/gpu7-*` files.

### 4.1 Qwen3-0.6B under overload (`evidence/gpu7-qwen3-0.6b-*.json`)

Run: 1,500 requests at 1.25 QPS (realized 1.26; the control's realized 0.6B rate was 1.37), 1,457 s,
1,456 snapshots, 0 failures. Mix: 75% govreport, 47% prompts >= 8,192 tokens, 6.1% capped final
lengths, 5.7% with the `1.0` default prediction. The instance saturates after about 4 minutes:
waiting queue p50 51 (p90 151), KV usage >= 0.95 in 71% of snapshots, measured TTFT p50 49 s and p90
126 s (first 300 s: p50 73 ms), TTFT-SLO attainment 27%. This is more overloaded than the 0.6B
instance in the 8.6 QPS controls (TTFT p50 15-44 s), because the control's router backed off 0.6B
when it was swamped while the diagnostic replays at a constant rate; read the numbers as an overload
stress, not a production replica.

Wait estimate versus measured engine TTFT, 1,499 aligned requests (seconds; error = estimate - TTFT):

| Rule | MAE | median AE | p90 AE | median error | bias | under >1 s | over >1 s | within 20% or 1 s | rank corr. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| router logged (production estimate) | 30.1 | 22.2 | 72.5 | -22.2 | -30.1 | 73% | 0% | 27% | 0.94 |
| current (replayed) | 30.2 | 22.2 | 73.0 | -22.2 | -30.2 | 73% | 0% | 27% | 0.94 |
| oracle (hindsight lengths, simulator floor) | 6.5 | 4.2 | 16.0 | +0.1 | +3.8 | 20% | 44% | 80% | 0.99 |
| cal_model_q50 (README's model-only table) | 7.9 | 6.0 | 19.1 | 0.0 | +0.8 | 32% | 37% | 72% | 0.98 |
| cal_prompt_q50 | 9.0 | 6.4 | 22.2 | +0.2 | +4.7 | 22% | 48% | 67% | 0.98 |
| online_prompt_pred_q50 | 9.8 | 6.8 | 25.1 | +2.5 | +6.7 | 16% | 55% | 65% | 0.98 |
| cal_prompt_q65 | 12.8 | 9.5 | 31.6 | +8.0 | +11.0 | 10% | 62% | 59% | 0.97 |
| online_prompt_pred_q65 | 17.5 | 14.5 | 40.7 | +14.2 | +16.8 | 5% | 67% | 49% | 0.97 |
| cal_prompt_q80 | 21.9 | 21.2 | 46.8 | +21.2 | +21.7 | 2% | 71% | 44% | 0.96 |
| exhausted_only_cal_prompt_q65 | 9.9 | 6.3 | 26.6 | 0.0 | +0.9 | 33% | 37% | 66% | 0.96 |
| exhausted_only_cal_model_q50 | 10.1 | 6.6 | 27.0 | 0.0 | +0.3 | 34% | 36% | 66% | 0.96 |

Measured findings:

- The production estimate under-estimates the wait of 73% of requests by more than 1 s and never
  over-estimates; median error -22 s, MAE 30 s. Once the queue forms (TTFT > 10 s, 1,005 requests)
  the current rule under-estimates 100% of requests (median -21 s in the 10-60 s bin, -58 s above
  60 s).
- Decomposition, measured - current = (oracle - current) + (measured - oracle): means 30.2 = 34.0 +
  (-3.8) s, medians 22.2, 33.9 and -0.1 s. The remaining-length targets account for the entire
  under-estimate. With true lengths the simulator is unbiased in the median and over-estimates
  slightly in the mean; that residual is concentrated in the 10-60 s TTFT bin, where even the oracle
  over-estimates by a median 10 s (85% of that bin). That part is simulator error (admission and
  queue dynamics while the queue is building), not target error, and no remaining-length rule can
  remove it.
- Any q50 conditional target brings the estimate to within about 1.5-3 s MAE of the oracle floor:
  MAE 7.9-9.8 s against 6.5 s, median error 0 to +2.5 s, under-estimated share 16-32% instead of 73%.
  The model-only calibration table (`cal_model_q50`) is marginally the best on this run (MAE 7.9 s,
  bias +0.8 s); prompt conditioning adds nothing on 0.6B at the wait level because the running set is
  almost entirely long govreport prompts, so the prompt-bin table and the model table coincide. The
  differences between the q50 candidates (7.9-9.8 s) are second order next to the 30 -> 8 s change.
- Quantile choice is first order in the other direction: q65 targets over-estimate by a median 8-14 s
  and q80 by 21-26 s (`online_prompt_pred_q80` reaches MAE 28 s, nearly as bad as the current rule).
  The survival quantile of a long-prompt 0.6B request that has passed about 1,500 tokens is the 8,192
  cap, so any quantile above the median pushes most of the running set to the cap.
- `exhausted_only_*` variants (all quantiles give the same numbers, because the switch only touches
  exhausted requests whose survival quantile is the cap) reach MAE 9.9-10.1 s with bias below 1 s,
  but under-estimate 69% of the > 60 s requests (median -6.7 s in that bin): the requests still inside
  prediction + reserve keep their under-sized targets.
- Ordering is not the problem: Spearman rank correlation between estimate and measured TTFT is 0.94
  for the current rule, 0.98 for every q50 candidate, 0.99 for the oracle. The current rule ranks the
  0.6B requests almost as well as the candidates; it is the level that is wrong by 20-60 s.
- Dense-snapshot token error (`gpu7-qwen3-0.6b-dense-snapshot-token-error.json`: 44,305 observations
  of 1,478 requests, thinned to every 30th snapshot = 48 snapshots / 1,477 observations; same held-out
  tables as section 3). The per-request tail is unchanged by every candidate (thinned p90 4.7-5.1k,
  p95 6.3-6.8k tokens against 5.7k / 6.3k for the current rule; 46% of observations exhausted, 48%
  capped final lengths), while the running-set backlog ratio moves from 0.19 (p10-p90 0.11-0.33) to
  0.93-1.13 for the q50/q65 candidates. This reproduces section 3 on an independent sample, and
  together with the wait-estimate table it answers the section 3 question: on 0.6B the aggregate
  occupancy correction is what fixes the wait estimate; per-request accuracy does not improve and
  does not need to.

### 4.2 Hindsight replay over the 57 saved snapshots (`evidence/gpu7-replay-hindsight-57snapshots.json`)

Hypothetical probes (128 / 4,096 / 16,384 prompt tokens) through the native simulator from each saved
snapshot, candidate targets against the all-running-length oracle; 18 instants per model with running
decode (5 `repeat1` at 8.6 QPS, 7 + 6 lanes at 8 QPS), 54 probes per model. Differences from the
oracle in seconds (MAE; sign of the median in brackets):

| Model | current | cal_model_q50 | cal_prompt_q50 | online_prompt_pred_q50 | cal_prompt_q65 | exhausted_only_cal_prompt_q65 |
|---|---:|---:|---:|---:|---:|---:|
| 0.6B | 5.95 (under) | 2.70 (under) | 1.68 (under) | 2.24 (mixed) | 1.98 (over) | 1.65 (over) |
| 8B | 0.68 (under) | 2.11 (over) | 0.89 (over) | 0.84 (over) | 1.63 (over) | 1.06 (over) |
| 32B | 0.68 (under) | 0.34 (over) | 0.57 (over) | 0.42 (over) | 0.96 (over) | 1.02 (over) |

The effect is concentrated in the `repeat1` (8.6 QPS) instants: there the 0.6B current rule is 17.9 s
under the oracle and the candidates 5-8 s from it; on 8B `cal_model_q50` over-estimates by a median
+2.2 s (+3.4 s on the 16k probe) and on 32B by +0.9 s; the 8 QPS lane snapshots show near-zero
differences on 32B and sub-second ones on 8B. At the wait-estimate level, therefore, every candidate is
worse than the current rule on 8B, and only the q50 tables beat it on 32B, even though all of them
reduce the token-level p90/p99 error in section 3. This is hindsight through the simulator, not
measured TTFT.

Interpretation (hypothesis, consistent with both tables): a new request's wait depends on the earliest
completions among the running requests (an order statistic), not on each request's point error. Under
0.6B overload the current rule marks about half the running set as finishing within one token, so the
simulator frees KV and batch slots at once; that is the 20-60 s hole, and any target that keeps those
requests occupied closes it. On 8B/32B few requests are exhausted, and replacing every request's
remaining length by a conditional median delays the earliest simulated completions (the median of each
request exceeds the minimum of the actuals), so the wait is over-estimated while the token MAE falls.

### 4.3 Qwen3-8B at its production rate (`evidence/gpu7-qwen3-8b-*.json`)

Run: 3,000 requests (the first 3,000 the control routed to 8B) at 2.9 QPS (realized 2.90; the control's
realized 8B rate was 2.88), 1,351 s, 1,350 snapshots, 0 failures. Mix: 39% govreport, 36% hotpot_qa, 18%
alpaca, 8% writingprompts; 19% prompts >= 8,192; 0.03% capped final lengths; 0.2% default predictions.

The run did not reproduce the control's 8B regime, and that limits what it can say. The control's own
8B instance, at the same steady state (running 42-45, KV 94-97%, about 1,050 generated tokens/s,
prompt 11-14k tokens/s), kept 0-2 requests waiting throughout (`controls/qps8p6-repeat1/server_qwen3-8b.log`).
The diagnostic instance, fed the same stream at the same rate, fell behind from the first minute: waiting
queue p50 374 (max 762), KV >= 0.95 in 94% of snapshots, measured TTFT p50 213 s and p90 257 s, TTFT-SLO
attainment 3%. Its engine ran slower per step, not busier: prompt throughput 8-11k tokens/s, GPU
utilisation 35-80% against 85-92% in the control, and step interval 0.034-0.046 s against 0.028-0.030 s
with the scheduler/overhead part of the step (interval minus execute) 3-5x the control's
(`batch_stats_qwen3-8b.csv`). The most plausible cause is CPU-side: the diagnostic pins the API server,
engine core, worker, router, client and snapshot capture to the 12 CPUs 84-95, and vLLM's per-step
CPU work grows with a queue of several hundred requests; this is a hypothesis, not measured. The 0.6B run
shares the pinning but its engine was GPU-bound (step interval 0.029 s, execute 0.026 s), so its numbers
are not affected in the same way. Read the 8B run as an overload stress in a regime production does not
reach.

Wait estimate versus measured TTFT, 2,999 aligned requests (seconds):

| Rule | MAE | median AE | p90 AE | median error | bias | under >1 s | over >1 s | within 20% or 1 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| router logged | 28.8 | 26.7 | 57.8 | -26.7 | -28.4 | 86% | 6% | 69% |
| current (replayed) | 28.9 | 27.0 | 57.8 | -27.0 | -28.6 | 86% | 6% | 69% |
| oracle (running lengths) | 29.5 | 29.6 | 59.6 | -29.6 | -29.3 | 85% | 4% | 71% |
| cal_model_q50 | 29.7 | 28.8 | 59.4 | -28.8 | -29.4 | 86% | 5% | 68% |
| cal_prompt_q50 | 30.4 | 29.2 | 60.4 | -29.2 | -30.2 | 88% | 4% | 68% |
| online_prompt_pred_q50 | 30.7 | 30.4 | 61.0 | -30.4 | -30.5 | 89% | 3% | 68% |
| cal_prompt_q80 | 26.7 | 25.2 | 54.9 | -25.2 | -25.4 | 74% | 16% | 75% |
| exhausted_only_cal_prompt_q65 | 27.9 | 26.5 | 56.7 | -26.5 | -27.4 | 83% | 8% | 72% |

Measured findings:

- The running-decode targets are irrelevant to the 8B wait error. Decomposition: measured - current =
  (oracle - current) + (measured - oracle) = 28.6 = (-0.7) + 29.3 s in the mean (medians 27.0, -0.8,
  29.6 s). Every candidate sits within 2.2-2.9 s MAE of the oracle, as does the current rule (2.6 s), and
  the candidates' differences from the current rule (-1.0 to +1.8 s MAE) are noise against a 29 s miss.
  In the 10-60 s TTFT bin (412 requests, the closest thing to a production queue here) the current rule's
  median error is -2.3 s, the oracle's -1.3 s, `cal_prompt_q50` -2.3 s, `cal_prompt_q65` -0.2 s.
- The 29 s miss is simulator error: with every running request given its true length the simulator still
  under-estimates the wait by a median 29.6 s at TTFTs around 200 s (about 14%), which matches an engine
  that runs 20-50% slower per step than the coefficients the simulator was calibrated with (previous
  paragraph). No remaining-length rule can address it.
- Dense-snapshot token error (`gpu7-qwen3-8b-dense-snapshot-token-error.json`, 78,304 observations of
  2,985 requests, thinned 44 snapshots / 2,593 observations; 9.8% exhausted, 0.5% capped): the prompt-
  conditioned tables reduce the per-request tail as in section 3 (p90 597 -> 409-423, p99 1,284 ->
  1,068-1,124 tokens; backlog ratio 1.10 -> 0.93-0.96), and none of that reaches the wait estimate.

### 4.4 Waiting-request targets (both runs, `*_all` variants)

Every variant above, including the oracle, leaves the waiting requests at prediction + reserve, which is
what the router does. Two extra variants also replace the waiting requests' targets: `oracle_all` with
their eventual lengths (everything the simulator could know about lengths) and `cal_prompt_q50_all` with
the calibration prompt-bin median at zero generated tokens (a deployable rule that ignores the router
prediction). Against measured TTFT:

| Run | oracle | oracle_all | cal_prompt_q50 | cal_prompt_q50_all |
|---|---:|---:|---:|---:|
| 0.6B, MAE / median error / bias (s) | 6.5 / +0.1 / +3.8 | 9.3 / -5.9 / -9.2 | 9.0 / +0.2 / +4.7 | 15.3 / -6.4 / -14.7 |
| 8B, MAE / median error / bias (s) | 29.5 / -29.6 / -29.3 | 48.3 / -57.9 / -48.3 | 30.4 / -29.2 / -30.2 | 50.6 / -58.5 / -50.5 |

Giving waiting requests their true lengths lowers the estimate on both runs (waiting-target component
-13.1 s mean on 0.6B, -19.0 s on 8B) and makes it worse: prediction + reserve over-covers waiting requests
(their median final length is 430-500 tokens against a 550-650-token reserve on top of the prediction),
and that over-coverage is currently masking a simulator under-estimate that has nothing to do with
lengths (residual measured - oracle_all: +9.2 s mean on 0.6B, +48 s on 8B). Replacing the prediction with
a prompt-bin median for waiting requests (`cal_prompt_q50_all`) is worse still. In the production
hindsight snapshots the whole waiting-set effect is small: `oracle_all` differs from `oracle` by a median
-1.7 s on 0.6B `repeat1` (waiting median 19) and by under 0.1 s on 8B/32B, so this is an overload-only
phenomenon; but it means the reserve on waiting requests must not be shrunk on the strength of section
2's predictor-bias finding without re-checking the simulator floor.

## 5. What is established, what is not

Established (measured on saved runs, snapshots and the two GPU-7 diagnostics):

- 0.6B under overload: the router's wait estimate is wrong by a median -22 s (73% of requests under-
  estimated by more than 1 s, none over-estimated), and the running-decode remaining-length targets are
  the entire cause: with true lengths for the running set the same simulator is unbiased in the median
  (MAE 6.5 s). The current rule under-estimates aggregate 0.6B decode backlog about 5-6x (backlog ratio
  0.16-0.19) in every overloaded snapshot, driven by 8,192-capped govreport generations that the rule
  marks as one token from finishing.
- Any q50 conditional survival table for running requests (model-only, prompt-bin or prompt+prediction,
  calibration-fitted or held-out online) brings the 0.6B wait estimate to within 1.5-3 s MAE of that
  floor (7.9-9.8 s), by correcting aggregate occupancy (backlog ratio 0.93-1.13); the per-request token
  tail (p90 5k, p95 6-7k tokens) is not reduced by any of them and does not need to be. Quantiles above
  the median over-estimate by 8-26 s and are not usable on 0.6B.
- 8B: the running-decode targets do not matter for the wait estimate. Measured under overload every
  candidate is within noise of the current rule (all within 2.2-2.9 s of the oracle, against a 29 s
  simulator miss); in the production-regime hindsight snapshots the candidates are slightly worse than the
  current rule (0.8-2.1 s vs 0.7 s from the oracle, by over-estimation). The section 3 token-tail gains on
  8B (p90 -30%, p99 -15% on the dense 8B snapshots) do not translate into wait-estimate gains. 32B
  (hindsight only): sub-second either way, q50 tables 0.3-0.6 s vs 0.7 s current.
- The simulator has errors that no length rule touches: on 0.6B about +/-10 s at 50-100 s TTFT
  (over-estimating in the mean while the queue builds, under-estimating once waiting requests are given
  their true lengths); on 8B -30 s at 200 s TTFT when the engine runs slower per step than the
  simulator's coefficients assume. Prediction + reserve on waiting requests over-covers and currently
  masks part of that.
- On 8B/32B the p50-p90 token error is predictor bias (mean head at about 40% of true length on the
  long-output buckets), the warm reserve compensates, and the p99 token tail is rare capped generations
  plus exhaustion (section 2). This remains true; its consequence for wait estimates is small.

Not established:

- That the 0.6B fix improves end-to-end TTFT attainment or the Bridges/Vast gap. Both diagnostics ran a
  single instance, so routing was fixed and the corrected estimate could not act. In the pool the hard
  policy would divert requests away from a 0.6B instance whose estimate rises from about 5 s to 30-100 s,
  and the effect on the other instances and on cost is unmeasured.
- Anything about 8B at production load beyond the hindsight replay: the 8B diagnostic ran in a CPU-
  limited regime the production pool does not reach (section 4.3), and the cause of the slowdown is a
  hypothesis.
- Runtime cost of the table lookup inside the engine's target rule and its interaction with the adaptive
  residual reserve; behaviour on 32B (measured) and on Ministral (no data).

## 6. Recommended next step

1. Change the running-request target for Qwen3-0.6B only, behind a per-model flag: replace
   prediction + reserve by the calibration-fitted conditional survival median (q50) of total length
   given tokens generated so far, floored at generated + 1 and capped at max_tokens. Model-only and
   prompt-bin tables are equivalent on 0.6B (7.9 vs 9.0 s MAE); use the prompt-bin table if one mechanism
   is wanted for all models, but enable it only for 0.6B. Do not use q65/q80, and do not use the
   `exhausted_only` form (it leaves 69% of the > 60 s requests under-estimated). Keep prediction + reserve
   for waiting requests and probes. Judge it in a paired 8.6 QPS full-pool run on measured TTFT
   attainment, the 0.6B routing share and cost, not on estimate error: the expected estimate change (MAE
   30 s to 8-10 s on the overloaded instance) is now measured, the routing consequence is not.
2. Do not change the running-request target on 8B or 32B: the token-tail improvements of section 3 do
   not reach the wait estimate, and on 8B every candidate is slightly worse than the current rule in the
   production regime. The earlier README's model-only table is unsuitable for 8B for this reason (+2.2 s
   median over-estimate in the 8.6 QPS snapshots), not because prompt conditioning is better there.
3. Predictor recalibration for 8B/32B (section 2) is still the right input-side fix, but it must be
   paired with a re-check of the waiting-request reserve: prediction + reserve currently over-covers
   waiting requests and masks a simulator under-estimate (section 4.4); shrinking the reserve after a
   better predictor would expose it.
4. If wait estimates under overload need to be better than about +/-10 s, the next target is the
   simulator itself (queue-building dynamics on 0.6B; step-time drift when the engine is CPU-limited on
   8B), not the length rule. An 8B diagnostic in a GPU-bound regime (router and client off the engine's
   CPUs, or the production pool with dense snapshots) is the way to measure that; the runner supports it.
5. Fill the router's missing-prediction default (`1.0` tokens, 5.7% of 0.6B requests) for whichever
   target is used; the survival table makes it irrelevant for running requests but not for waiting ones.

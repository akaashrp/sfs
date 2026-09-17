# Remaining-decode target tail: Ministral 3 family (3B / 8B / 14B)

Companion to `scripts/cloud/reports/reserve-tail-20260916/README.md` (the Qwen analysis), so that the
pending reserve-change decision (the flag-gated `remaining_length_mode` rule in
`vllm/vllm/v1/core/sched/remaining_length.py`) can cover both families. Controller CPU only, read-only
over saved evidence: no GPU was used, nothing was launched on Vast, and no runtime, predictor, coefficient,
SLO or campaign state was changed.

Scripts: `extract_inputs_ministral.py` builds the compact inputs from the raw mirrors;
`evaluate_tail_ministral.py` imports the Qwen `evaluate_tail.py` (survival tables, back-off, prompt bins,
prediction bins, folds, metrics) and drives it for the Ministral family. Evidence JSON is in `evidence/`.
Section numbering follows the Qwen README where the content corresponds.

Summary: none of the three Ministral models shows the Qwen3-0.6B failure mode (long-prompt generations that
either hit the 8,192 cap or stop, and a 5-6x aggregate backlog under-count). All three look like Qwen3-8B/32B:
the length predictor's mean head predicts about 42% of the true length on govreport (and 68-75% on
writingprompts), the warm reserve compensates only partly, 17-20% of running-decode tokens are observed
after the current rule is exhausted, and in reconstructed running sets the current rule under-counts aggregate
decode backlog by a median 13-26% at the warm low-congestion reserve (centred at the medium-congestion reserve,
27-34% over at the high-congestion one), i.e. its aggregate accuracy tracks the adaptive reserve's congestion
state. The calibration-fitted prompt-bin survival median
(`running_all` / `prompt_bin` / q50) is the only candidate the evidence supports if the rule is enabled on
Ministral; q65 over-estimates, model-only conditioning is worse at p90, and `exhausted_only` gains little.
Whether any of this reaches the wait estimate is not established without GPU evidence (section 7).

## 1. Data inventory and provenance (`evidence/data-inventory.json`)

Calibration outputs (the Ministral equivalent of Qwen's 30,000 calibration generations):

- `/ocean/projects/cis250162p/aparthas/sfs_artifacts/ministral3_calibration_44849564_judge_45009121_20260902/run/completions/<model>/<bucket>.jsonl`,
  the `calibration_completions_root` recorded in the length predictor's provenance
  (`sfs_model_family/experiments/ministral3_paper/predictors/run_45089699/provenance.json`; the bundle's
  `ministral/length` predictor has the identical `training_summary.json`). 2,500 outputs per bucket per model
  (holdout indices 0-2499), 30,000 in total, 0 request errors, `max_completion_tokens` 8,192 (24 govreport
  prompts per model have 8,181). All 1,000 of the predictor's `test_example_ids` are inside this set, so it is
  the predictor's training/test population.
- Overlap check: 0 of these 10,000 prompts appear among the 8,000 online evaluation prompts
  (`bundle/ministral/request_map.csv`, holdout indices 2500-4499; sha256 `5d4fa1fd...`).
- The bundle's `ministral/scores/<model>/<bucket>_scored.jsonl` (4,000 per bucket, listed in `bundle.json` as
  `scored_root`) are not calibration data: they hold holdout indices 2500-6499, and their first 2,000 per bucket
  are the online evaluation prompts themselves. They were excluded; the remaining 2,000 per bucket per model
  (indices 4500-6499, 24,000 rows, disjoint from both the online prompts and the predictor training prompts)
  are kept as a second independent table source, `holdout2`, used only as a robustness check.
- `bundle/ministral/calibration_requests.jsonl` (10,000 rows) is the timing-calibration request set; it carries
  prompts and SLOs but no generations, so it is not used here.

Online evaluation rows (per-request `prompt_tokens`, `usage_completion_tokens`, `predicted_output_tokens`,
timestamps), 23 completed cells, 8,000 requests each, 0 failures, 184,000 rows:

| Host | Cells | Requests | SFS prediction logged |
|---|---|---:|---|
| Vast `ministral-20260916` | shortest_queue 6.0125 / 7.8625 / 8.7875; vllm_sr_latency 6.0125 / 7.8625 | 40,000 | shortest_queue only |
| Vast `ministral-20260916-mrb` | mooncake_prefill 6.0125 / 7.8625 / 8.7875; routebalance 6.0125 / 7.8625 / 8.7875 | 48,000 | none (see below) |
| Bridges (12 audited cells of `bridges-reuse-review-20260916.json`) | lmdeploy_proxy, latency_agnostic, round_robin at 6.0125 / 7.8625 / 8.7875 / 9.7125 | 96,000 | latency_agnostic, round_robin |

Provenance: Vast cells from the controller mirror `sfs_cloud_results_20260916/sfs-vast/baselines/`, except
`mooncake_prefill-7.8625/8.7875` and the three `routebalance` cells, which the mirror had not received (the
7.8625 directory was an empty stub); those five were read from `/workspace/sfs/state/baselines/ministral-20260916-mrb/cells/`
by plain `cat` over ssh (all `COMPLETE`, audit.json present). Bridges cells are the
`campaign_recovery_20260915/reused_baselines/outputs/point_000..011.json` files whose sha256 the audit records
(verified). Bridges and Vast use the same 8,000 prompts (0 mismatches of bucket and prompt tokens per request id),
and for the same (request, model) the SFS prediction is identical on both hosts (8,891 pairs, all equal).

Predictions: the engine's target rule uses the SFS length predictor (`--output-length-model-path
bundle/ministral/length`) whatever the router policy. `shortest_queue`, `latency_agnostic` and `round_robin`
log that prediction (88,000 rows); `routebalance` logs its own predictor's value (0 of 6,102 pairs match the
SFS value) and is treated as unlogged; `mooncake_prefill`, `lmdeploy_proxy`, `vllm_sr_latency` log none. Because
the SFS prediction depends only on (prompt, model), it was imputed for unlogged rows from the logged value of the
same (request, model): 84,862 rows imputed, 11,138 rows (pairs never routed by a logging policy) left without a
prediction and excluded from the current-rule evaluation. The router-vs-imputed split is a check in section 5
(differences under 5%).

Per model:

| Model | online rows (all) | with prediction (prompts) | online capped >= 8,192 | calibration capped | holdout2 capped | online p50 / p90 / p99 | calibration p50 / p90 / p99 | prediction <= 1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3B | 76,275 | 73,141 (7,419) | 0.05% (35) | 0.08% (8) | 0.14% | 799 / 1,481 / 2,203 | 692 / 1,438 / 2,396 | 0.07% |
| 8B | 58,451 | 53,462 (6,747) | 0.10% (56) | 0.10% (10) | 0.07% | 697 / 1,455 / 2,437 | 787 / 1,487 / 2,489 | 0.09% |
| 14B | 49,274 | 46,259 (6,577) | 0.06% (27) | 0.05% (5) | 0.10% | 747 / 1,492 / 2,314 | 840 / 1,539 / 2,278 | 0.14% |

Cap: `--max-completion-tokens 8192` in the bundle's `experiment_argv`, `--max-model-len 131072` on the servers
(`server_argv_ministral3-*.json`), so the per-request cap is 8,192 for every prompt (prompt limit 32,768). The
Ministral cap rate is 0.05-0.14% on every source, against 6.4-8.2% of govreport outputs and 48% of running
observations for Qwen3-0.6B. The router's missing-prediction default (`1.0`) occurs on 0.07-0.14% of rows
(Qwen 0.6B: 5.6%).

Engine snapshots: none exist for Ministral. `/workspace/sfs/state/diagnostics/baseline-snapshot-gpu7{,-attempt2}`
are Ministral-8B single-instance router runs with shm publishing enabled but no saved captures (the first
failed with "Baseline scheduler snapshot is not published yet", the second completed a 100-request
mooncake/routebalance smoke); the `snapshot_stress/` directories under the Ministral baselines are 512-request
router runs; the Bridges `campaign_release_20260912/ministral_snapshot*` files are a code-snapshot sbatch
rehearsal. The evaluation therefore uses (a) the synthetic held-out construction of `evaluate_tail.py`
(every online request observed at 8 evenly spaced generation points, token-weighted) and (b) pseudo-snapshots:
the real running set of each (run, model) every 5 s, reconstructed from `first_token_ts_s` and the measured
decode time, with generated-so-far interpolated linearly (measured composition, approximated progress).

Length reproducibility (`data-inventory.json: length_reproducibility`): the same (request, model) pair repeated
across cells does not reproduce its length at temperature 0. Same host, range across repeats p50 89-157 tokens
(p90 518-595), identical in only 19-24% of pairs; across hosts, |difference of medians| p50 25 tokens. This is
batch-composition non-determinism of the engine, and it is a per-request noise floor of roughly 100 tokens that no
length rule can remove.

## 2. Reserve reconstruction (`evidence/reserve-reconstruction.json`)

There are no Ministral snapshots to read `decode_reserve_tokens` from, so the engine's rule was replayed:
reserve = max(floor, q-quantile of the positive residuals `max(0, L - ceil(p))` of the last 4,096 finished
requests with `p > 0`), floor 32/64/128 at KV usage <= 0.8 / <= 0.9 / > 0.9, quantile 0.50 until 200 samples then
0.65 / 0.75 / 0.85 by congestion (`scheduler.py: _snapshot_decode_reserve_base_tokens`). Fitted per (run, model)
on the run's own finished requests, median over the 23 runs:

| Model | q50 (cold) | q65 (warm, KV <= 0.8) | q75 | q85 (KV > 0.9) | q65 range over runs | positive residuals |
|---|---:|---:|---:|---:|---:|---:|
| 3B | 118 | 410 | 575 | 765 | 340-757 | 57-68% (88% under latency_agnostic) |
| 8B | 86 | 399 | 567 | 747 | 37-444 | 57-68% |
| 14B | 102 | 390 | 590 | 761 | 204-504 | 57-68% |

`reserve_used` = q65 (410 / 399 / 390) for the synthetic evaluation; the Qwen analysis used the median snapshot
reserve (0.6B 308, 8B 614, 32B 565). The reserve is strongly mix-dependent: the 3B instance under
`latency_agnostic` (which sends it the prompts predicted short, 88% of them under-predicted) reaches q65 = 750.
Section 5 reports the current rule at the q50 and q85 reserves as well; section 6 replays each run with its own
reserve. This reconstruction is a hypothesis about the engine's state, not a measurement.

## 3. Length-predictor bias per model and bucket (`evidence/predictor-bias.json`)

Router prediction (mean head) against the online final length, rows with a prediction; calibration p50 in
parentheses shows that the predictor's own training population has the same lengths as observed online:

| Model / bucket | n | pred p50 | online p50 (calibration p50) | pred / online | residual L - p: p50 / p90 / p99 | online capped |
|---|---:|---:|---:|---:|---:|---:|
| 3B alpaca | 15,788 | 725 | 248 (250) | 2.92 | -330 / 604 / 1,974 | 22 |
| 3B govreport | 25,710 | 504 | 1,193 (1,181) | 0.42 | +686 / 1,103 / 1,528 | 0 |
| 3B hotpot_qa | 16,589 | 127 | 89 (88) | 1.42 | -34 / 124 / 731 | 3 |
| 3B writingprompts | 15,054 | 633 | 845 (833) | 0.75 | +212 / 706 / 1,344 | 10 |
| 8B alpaca | 14,182 | 820 | 300 (294) | 2.73 | -363 / 757 / 2,166 | 29 |
| 8B govreport | 11,043 | 498 | 1,168 (1,187) | 0.43 | +671 / 1,030 / 1,375 | 0 |
| 8B hotpot_qa | 15,585 | 118 | 80 (79) | 1.48 | -37 / 111 / 835 | 14 |
| 8B writingprompts | 12,652 | 677 | 961 (958) | 0.70 | +298 / 808 / 1,428 | 13 |
| 14B alpaca | 13,365 | 741 | 288 (295) | 2.57 | -304 / 648 / 1,927 | 8 |
| 14B govreport | 6,456 | 521 | 1,272 (1,277) | 0.41 | +747 / 1,108 / 1,462 | 0 |
| 14B hotpot_qa | 10,949 | 120 | 84 (84) | 1.43 | -33 / 124 / 696 | 0 |
| 14B writingprompts | 15,489 | 677 | 991 (1,008) | 0.68 | +317 / 823 / 1,417 | 19 |

Per model (all buckets): prediction MAE 459 / 423 / 417 tokens (3B / 8B / 14B); residual p50 +213 / +48 / +78,
p90 927 / 842 / 853, p99 1,532 / 1,560 / 1,504; `ceil(p) + reserve_q65` covers the final length of 60% / 68% / 68%
of requests. The pattern is the Qwen 8B/32B one: the mean head predicts about 42% of the true length on govreport
and 68-75% on writingprompts (both long-output buckets), over-predicts alpaca by 2.6-2.9x at the median (alpaca's
online p90 is 1,557-1,754, so its mean is pulled up by a long tail) and hotpot by 1.4-1.5x. Predictor under-fit,
not a serving difference: calibration and online medians agree within 20 tokens on every bucket. Bias is the same
on Vast and Bridges (per-host residual medians in the evidence file).

## 4. Tail of the current rule (`evidence/tail-characterization.json`)

Remaining-token error of `prediction + reserve` on the synthetic held-out set (1,382,896 observations of 172,862
requests; token-weighted = snapshot population; `reserve_used` per model). Qwen rows are the same construction from
`reserve-tail-20260916/evidence/candidate-evaluation.json` (real-snapshot values in brackets):

| Model | MAE | median | p90 | p95 | p99 | coverage | exhausted obs | capped obs | backlog ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ministral 3B | 381 | 288 | 755 | 964 | 1,673 | 31% | 19.9% | 0.5% | 0.74 |
| Ministral 8B | 429 | 299 | 836 | 1,142 | 2,559 | 36% | 18.3% | 1.2% | 0.79 |
| Ministral 14B | 394 | 295 | 788 | 1,011 | 1,840 | 38% | 17.3% | 0.6% | 0.82 |
| Qwen 0.6B (synthetic) [real] | 2,210 [2,218] | [1,162] | 5,631 [5,820] | 6,631 [6,490] | 7,555 [7,332] | 25% [29%] | [47%] | [48%] | [0.16] |
| Qwen 8B (synthetic) [real] | 339 [292] | [182] | 630 [572] | 790 [673] | 3,583 [3,287] | 68% [57%] | [11%] | [1.5%] | [1.22] |
| Qwen 32B (synthetic) [real] | 362 [350] | [220] | 641 [619] | 856 [840] | 4,607 [3,259] | 42% [37%] | [16%] | [1.8%] | [0.89] |

Backlog ratio = sum of estimated over sum of actual remaining tokens, token-weighted (the simulator's decode
occupancy under a stationary population; the Qwen synthetic set has no such field, its real-snapshot value is
shown). The synthetic construction tracks the Qwen real snapshots on every column, which is the basis for using
it here. Request-weighted Ministral values (each request counted once): MAE 426-461, p90 843-926, p99
1,535-1,650, coverage 60-68%.

Where the Ministral error comes from (token-weighted shares, `tail_share`):

- 3B: govreport is 56% of observations, 48% of absolute error, 65% of under-estimated tokens and 94% of
  context-weighted (KV-hold) under-estimation; prompts >= 8,192 tokens are 33% of observations and 76% of
  context-weighted under-estimation. Exhausted observations (target - generated <= 1) are 20% of tokens, 21% of
  absolute error, 29% of under-estimation. Capped final lengths are 0.5% of tokens, 5.4% of absolute error, 7.6%
  of under-estimation (Qwen 0.6B: 48% / 79% / 86%). Errors >= 1,000 tokens are 4.5% of observations and 20% of
  absolute error.
- 8B: govreport 35% / 25% / 37% / 80%; exhausted 18% / 24% / 36%; capped 1.2% / 11.6% / 17.5% (the 8B p99 of
  2,559 and the hotpot-bucket p95 of 1,535 / p99 of 6,655 come from 14 degenerate hotpot generations and 29
  alpaca ones that reach the cap; by request count they are 0.1-0.2% of the bucket, by tokens 6% of the hotpot
  observations).
- 14B: govreport 25% / 22% / 34% / 83%; exhausted 17% / 20% / 31%; capped 0.6% / 6.6% / 10.3%.
- By prompt bin, the current rule's coverage is 1-6% (bias -330 to -394 tokens, 28-32% exhausted) on 8k-32k
  prompts and 5-13% on 2k-8k prompts, against 77-85% coverage (bias +170 to +230 on 3B/14B) on 512-2k prompts
  (hotpot). Prompt length therefore separates the under- and over-predicted buckets, as it did on Qwen 8B/32B.
- The exhausted cohort under the current rule has MAE 398 / 554 / 448, p95 1,067 / 2,437 / 1,341 and p99
  4,303 / 6,655 / 4,607 tokens (3B / 8B / 14B); the capped cohort has MAE about 4,020 on all three.

Reserve sensitivity of the current rule (same synthetic set): with the cold q50 reserve (118 / 86 / 102 tokens)
coverage falls to 15-20% and the backlog ratio to 0.41-0.48; with the high-congestion q85 reserve (765 / 747 / 761)
coverage is 64-69% and the backlog ratio 1.24-1.34. The current rule's aggregate accuracy on Ministral thus swings
from a 2x under-count to a 1.3x over-count with the congestion state of the adaptive reserve.

## 5. Held-out candidate evaluation (`evidence/candidate-evaluation.json`)

Same candidates and discipline as the Qwen section 3. `cal_*`: fitted on the 10,000 calibration outputs per
model only (prompt-bin table support: bins <128 / 512-2k / 2k-8k / 8k-32k = 4,993 / 2,323 / 1,328 / 1,309 rows;
bins 128-512 and >= 32k have 23-24 rows and back off to the model level). `h2_*`: fitted on `holdout2` only.
`online_*`: five prompt-level folds (`sha256(example_id) mod 5`) over the pooled online rows; every evaluated
observation is from a prompt absent from its table. `cal_bucket` uses the dataset label (oracle upper bound).
`exhausted_only_*` keeps the current target until it says <= 1 token remains. Token-weighted:

| Model | Rule | MAE | median | p90 | p95 | p99 | coverage | backlog ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 3B | current | 381 | 288 | 755 | 964 | 1,673 | 31% | 0.74 |
| 3B | rolling_floor | 349 | 280 | 686 | 904 | 1,573 | 52% | 0.96 |
| 3B | cal_model_q50 | 323 | 225 | 664 | 803 | 1,583 | 46% | 0.84 |
| 3B | cal_prompt_q50 | 273 | 183 | 556 | 764 | 1,661 | 50% | 0.86 |
| 3B | cal_prompt_q65 | 290 | 222 | 543 | 722 | 1,489 | 64% | 1.06 |
| 3B | cal_prompt_q80 | 376 | 319 | 690 | 840 | 1,315 | 79% | 1.36 |
| 3B | cal_bucket_q50 (label oracle) | 270 | 182 | 531 | 735 | 1,604 | 50% | 0.86 |
| 3B | online_prompt_pred_q50 | 269 | 181 | 539 | 735 | 1,630 | 50% | 0.87 |
| 3B | h2_prompt_q50 | 272 | 184 | 559 | 762 | 1,626 | 49% | 0.87 |
| 3B | exhausted_only_cal_prompt_q50 | 353 | 260 | 720 | 928 | 1,563 | 42% | 0.84 |
| 3B | exhausted_only_cal_prompt_q65 | 357 | 269 | 714 | 921 | 1,532 | 44% | 0.88 |
| 3B | exhausted_only_cal_model_q50 | 354 | 253 | 721 | 932 | 1,571 | 42% | 0.84 |
| 8B | current | 429 | 299 | 836 | 1,142 | 2,559 | 36% | 0.79 |
| 8B | rolling_floor | 398 | 293 | 781 | 1,084 | 2,317 | 55% | 0.97 |
| 8B | cal_model_q50 | 359 | 226 | 752 | 903 | 2,140 | 51% | 0.90 |
| 8B | cal_prompt_q50 | 322 | 205 | 667 | 943 | 2,169 | 51% | 0.85 |
| 8B | cal_prompt_q65 | 347 | 242 | 655 | 910 | 2,146 | 66% | 1.07 |
| 8B | cal_prompt_q80 | 454 | 357 | 810 | 1,036 | 2,254 | 81% | 1.40 |
| 8B | cal_bucket_q50 (label oracle) | 313 | 194 | 618 | 879 | 2,051 | 52% | 0.86 |
| 8B | online_prompt_pred_q50 | 316 | 192 | 646 | 893 | 2,203 | 50% | 0.85 |
| 8B | h2_prompt_q50 | 331 | 208 | 677 | 979 | 2,559 | 51% | 0.83 |
| 8B | exhausted_only_cal_prompt_q50 | 392 | 268 | 795 | 1,074 | 1,997 | 46% | 0.90 |
| 8B | exhausted_only_cal_prompt_q65 | 398 | 284 | 787 | 1,062 | 2,076 | 49% | 0.94 |
| 8B | exhausted_only_cal_model_q50 | 393 | 266 | 798 | 1,075 | 2,005 | 46% | 0.90 |
| 14B | current | 394 | 295 | 788 | 1,011 | 1,840 | 38% | 0.82 |
| 14B | rolling_floor | 366 | 288 | 729 | 960 | 1,729 | 56% | 0.99 |
| 14B | cal_model_q50 | 346 | 237 | 754 | 881 | 1,688 | 55% | 0.98 |
| 14B | cal_prompt_q50 | 307 | 204 | 659 | 848 | 1,772 | 52% | 0.89 |
| 14B | cal_prompt_q65 | 336 | 255 | 651 | 893 | 1,690 | 67% | 1.12 |
| 14B | cal_prompt_q80 | 442 | 374 | 789 | 1,023 | 1,585 | 82% | 1.44 |
| 14B | cal_bucket_q50 (label oracle) | 299 | 198 | 595 | 834 | 1,810 | 51% | 0.88 |
| 14B | online_prompt_pred_q50 | 297 | 189 | 632 | 840 | 1,769 | 50% | 0.87 |
| 14B | h2_prompt_q50 | 305 | 201 | 649 | 832 | 1,775 | 53% | 0.91 |
| 14B | exhausted_only_cal_prompt_q50 | 370 | 265 | 760 | 987 | 1,691 | 48% | 0.91 |
| 14B | exhausted_only_cal_prompt_q65 | 376 | 290 | 754 | 983 | 1,729 | 50% | 0.95 |
| 14B | exhausted_only_cal_model_q50 | 371 | 267 | 761 | 988 | 1,707 | 47% | 0.91 |

Findings:

- The prompt-bin q50 survival table (the deployable `running_all` / `prompt_bin` / q50 setting) lowers MAE by
  25-28% and p90 by 16-26% on all three models, brings coverage to 50% and the aggregate backlog ratio from
  0.74-0.82 to 0.85-0.89. It captures almost all of the label-oracle gain (p90 556 vs 531 on 3B, 667 vs 618 on
  8B) and matches the held-out online table (which additionally conditions on the prediction bin) within 3% on
  every metric, so the router prediction adds nothing beyond prompt length, as on Qwen 8B/32B. The
  independent `holdout2` table gives the same numbers (within 3%), so the calibration set is not special.
- Model-only conditioning (`cal_model_q50`) is worse than prompt-bin at p90 by 13-20% (664 vs 556; 752 vs 667;
  754 vs 659) for the reason seen on Qwen 8B: prompt length separates the under-predicted long-output buckets
  from hotpot.
- q65 raises coverage to 64-67% but over-counts aggregate backlog (1.06-1.12) and worsens MAE relative to q50;
  q80 over-counts by 36-44% and is worse than the current rule on MAE. Quantile choice is again the first-order
  knob for the aggregate, and q50 is the one that centres it.
- `exhausted_only_*` variants touch only the 17-20% of tokens that are past the current target: MAE -7%, p90
  -4 to -5%, backlog 0.84-0.95. On Ministral the exhausted cohort's p99 stays at 3,400-6,300 tokens under every
  candidate (it is the capped generations), so the p99 relief that `exhausted_only` gave on Qwen 8B/32B
  (3,287 -> 989) does not appear here; the Ministral p99 tail is not dominated by exhaustion.
- p99 is barely reducible on Ministral (1,673 -> 1,489-1,661 on 3B; 2,559 -> 2,140-2,254 on 8B) because it is
  driven by the rare capped generations (0.5-1.2% of tokens) and by alpaca's long uncapped tail, which no point
  estimate can anticipate; the survival median moves to the cap only once half the survivors are capped, which on
  Ministral never happens below g = 2,000 (section 6).
- `rolling_floor` centres the aggregate (0.96-0.99) but leaves MAE and p90 close to the current rule; it is the
  same over-coverage-by-construction that inflated Qwen 8B/32B backlog, so it is not a candidate.
- Checks: Vast-only vs Bridges-only rows, and router-logged vs imputed predictions, agree within 5% on every
  metric for every candidate (`by_host`), so pooling the hosts and imputing the prediction did not bias the result.

## 6. Aggregate backlog in reconstructed running sets and the 3B question (`evidence/pseudo-snapshot-backlog.json`, `evidence/bimodality.json`)

Pseudo-snapshots (real per-instance running sets every 5 s from the request timestamps of all 23 runs, decode
progress interpolated; snapshots with fewer than 5 running requests dropped):

| Model | snapshots (runs) | running p50 / p90 / max | exhausted (median) | current | current, run-fitted reserve q65 / q75 / q85 | cal_model_q50 | cal_prompt_q50 | cal_prompt_q65 | exhausted_only_cal_prompt_q50 | online_prompt_pred_q50 | h2_prompt_q50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3B | 5,913 (23) | 57 / 109 / 351 | 20% | 0.74 (p10 0.52, p90 0.94) | 0.87 / 1.09 / 1.33 | 0.83 | 0.89 (0.76-1.02) | 1.08 | 0.83 | 0.90 | 0.89 |
| 8B | 5,122 (23) | 40 / 81 / 414 | 17% | 0.81 (0.59, 1.12) | 0.78 / 1.01 / 1.27 | 0.93 | 0.88 (0.69-1.09) | 1.11 | 0.92 | 0.88 | 0.87 |
| 14B | 5,585 (23) | 40 / 80 / 483 | 17% | 0.83 (0.60, 1.22) | 0.82 / 1.07 / 1.34 | 1.00 | 0.93 (0.75-1.18) | 1.15 | 0.91 | 0.91 | 0.94 |

Values are median backlog ratios over snapshots (p10-p90 in brackets for the prompt-bin table). `current` uses
the family-level `reserve_used`; the run-fitted columns replay each run with the reserve its own finished
requests would have produced at the low / medium / high congestion quantile. Per run, the run-fitted q65 ratio
spans 0.70-1.04 on 3B, 0.51-0.90 on 8B and 0.69-1.01 on 14B (with the fixed reserve: 0.55-0.86, 0.73-1.01 and
0.71-0.95; the 3B minimum is the four Bridges `latency_agnostic` runs, whose 3B mix is 88% under-predicted);
the run-fitted q75 ratio 0.94-1.20, 0.67-1.16 and 0.90-1.26; q85 1.24-1.39, 1.01-1.46 and 1.17-1.54. The
prompt-bin q50 table spans 0.84-0.94, 0.79-0.94 and 0.87-1.02 across runs and does not depend on the reserve
state. In the busiest decile of snapshots (>= 109 / 81 / 80 running) the fixed-reserve current rule gives
0.76 / 0.88 / 1.26, the run-fitted q65 rule 0.94 / 0.91 / 1.41, and the prompt-bin q50 table 0.91 / 0.92 / 1.18
(the 14B busiest decile is the tail of the Bridges `round_robin` 8.8-9.7 QPS runs with 300-480 running requests,
where linear progress interpolation is at its crudest; read those numbers with that caveat).

Read together: at the warm low-congestion reserve the current rule under-counts Ministral decode backlog by a
median 13-22% (up to 30-50% in individual runs), at the medium-congestion reserve it is centred (1.01-1.09), and
at the high-congestion reserve it over-counts by 27-34%. Its aggregate accuracy on Ministral is therefore a
function of the adaptive reserve's congestion state rather than a fixed bias, and the survival table's value is
that it removes that dependence (0.88-0.93 in every state) while also lowering per-request error.

Does Ministral 3B show the Qwen 0.6B pattern? No. The conditional distribution of the final length among
requests that have already generated g tokens, long prompts (8k-32k), online rows (3B: 13,827; Qwen 0.6B from its
calibration table, 1,293):

| g | Ministral 3B survivors | capped | stop within 250 | remaining p50 / p90 | Qwen 0.6B survivors | capped | stop within 250 | remaining p50 / p90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 13,827 | 0.0% | 0% | 1,273 / 1,690 | 1,293 | 10.3% | 6% | 619 / 8,192 |
| 500 | 13,827 | 0.0% | 1% | 773 / 1,190 | 838 | 15.9% | 39% | 371 / 7,692 |
| 1,000 | 11,840 | 0.0% | 38% | 329 / 719 | 350 | 38.0% | 21% | 1,750 / 7,192 |
| 1,500 | 3,209 | 0.0% | 69% | 165 / 474 | 239 | 55.6% | 10% | 6,692 / 6,692 |
| 2,000 | 275 | 0.0% | 82% | 92 / 325 | 204 | 65.2% | 7% | 6,192 / 6,192 |
| 3,000 | 0 | | | | 164 | 81.1% | 3% | 5,192 / 5,192 |

Ministral 3B's long-prompt generations are unimodal (calibration and online agree: p50 1,266-1,273, p90 1,686-1,690,
0 capped) and finish by about 2,000 tokens; the survival median stays a few hundred tokens ahead of the request
and never jumps to the cap. Qwen 0.6B's are cap-or-stop (56% of survivors at g = 1,500 will run to 8,192). The
8B and 14B long-prompt distributions are the same shape as 3B's (`bimodality.json`). Correspondingly the current
rule's aggregate ratio on 3B is 0.74-0.87 (0.55-0.70 at worst per run), not 0.16-0.19, and the exhausted share is
20%, not 47%. What 3B does share with 0.6B is that govreport dominates its running set (56% of tokens, 76% of KV-hold
under-estimation from >= 8k prompts) and that its instance carries the most requests (3,600-4,600 of 8,000 per cell).

Which flag-gated setting the evidence supports, per model (measured token-level evidence only):

- Ministral 3B: no failure mode that requires a change; `off` is a defensible default. If the rule is enabled,
  `running_all` / `prompt_bin` / q50 with the calibration table (10,000 rows) is the only setting supported: it is
  the best or within 3% of the best deployable candidate on every metric, holds the aggregate at 0.89 in every
  reserve state (the current rule moves between 0.74-0.87 and 1.33 with the congestion quantile, section 6
  table; 0.41-1.34 on the synthetic set), and does not touch waiting requests. Not q65 (aggregate 1.06-1.08, MAE worse), not `model` conditioning (p90 +19%), not
  `exhausted_only` (backlog 0.83, MAE -7%, no p99 relief).
- Ministral 8B: same conclusion (current 0.78-0.81 at the low-congestion reserve, 1.01 at medium);
  `running_all` / `prompt_bin` / q50 if enabled (0.88; MAE -25%, p90 -20%).
- Ministral 14B: same (current 0.82-0.83, 1.07 at medium; prompt-bin q50 0.93; MAE -22%, p90 -16%).
- Across the family the three models behave alike (per-request gains 3B >= 8B > 14B, aggregate behaviour the
  same shape) and none of it is of the Qwen 0.6B magnitude, so the decision on Ministral is a uniform one (all
  three off, or all three `running_all` / `prompt_bin` / q50), not a per-model exception like Qwen 0.6B.

Hypothesis (not measured): on Qwen 8B the GPU-7 diagnostic showed that a backlog-ratio difference of the size seen
here (1.22 vs 0.93-1.10) did not change the wait estimate, because the simulator's own error dominated. If that
holds for Ministral, the token-level gains above do not reach the wait estimate either and `off` costs nothing;
if the Ministral simulator is closer to its floor than Qwen 8B's was, the 13-26% aggregate under-count at the
low-congestion reserve (or the 27-34% over-count at the high-congestion one) would show up in the wait estimate
of the busiest instance. Nothing in the saved data decides this.

## 7. What is not established without GPU evidence

- Any wait-estimate or TTFT effect. There are no Ministral engine snapshots, no logged
  `decode_reserve_tokens`, and no single-instance diagnostic; the wait-time estimator's per-request error against
  measured TTFT (the Qwen section 4 quantity) was not computed for Ministral, and the simulator's floor for
  Ministral is unknown.
- The engine's actual reserve state under load. Section 2 is a replay of the rule on finished requests; the
  real window mixes in-flight requests, congestion-dependent quantiles and the floor, and section 4 shows the
  current rule's aggregate ratio moves between 0.41 and 1.34 across that range.
- Per-request generation progress inside the pseudo-snapshots (linear interpolation over the measured decode
  time; real progress is batch-size dependent) and hence the exact per-snapshot ratios in section 6; the
  cross-run agreement (0.84-1.02 for the prompt-bin table) is the evidence that the conclusion is not sensitive
  to it.
- The routing consequence of any change in the pool (the hard policy diverting requests when an instance's
  estimate rises), its cost, and the interaction with the adaptive reserve on waiting requests (the Qwen
  section 4.4 caveat applies unchanged: waiting requests keep prediction + reserve, which over-covers them and
  masks a simulator under-estimate; nothing here supports shrinking that).
- Anything about `hard` / `score` (the SFS/SCORE policies): no completed Ministral cell of either exists on Vast
  yet; the evaluation rows come from the baselines, whose length and prediction data are policy-independent but
  whose running-set composition per instance is not.

## Files

- `README.md` (this file), `extract_inputs_ministral.py`, `evaluate_tail_ministral.py`, `driver.log`.
- `evidence/data-inventory.json`: counts, cells, provenance checks, cap rates, length reproducibility.
- `evidence/reserve-reconstruction.json`: per (run, model) and per model reserve replay.
- `evidence/predictor-bias.json`: per model / bucket predictor bias with per-host split.
- `evidence/tail-characterization.json`: current-rule error by model, host, bucket, prompt bin, prediction bin,
  exhausted / capped cohorts, tail shares.
- `evidence/candidate-evaluation.json`: all candidates, request- and token-weighted, exhausted and capped cohorts,
  host / prediction-source checks, reserve sensitivity, table support.
- `evidence/pseudo-snapshot-backlog.json`: reconstructed running sets per (run, model) and per model.
- `evidence/bimodality.json`: conditional final-length distributions at fixed generated counts, Ministral (all
  three models, calibration and online, prompt bins 2k-8k and 8k-32k, govreport) and Qwen 0.6B.
- `evidence/qwen-comparison.json`: the Qwen numbers used above, copied from `reserve-tail-20260916/evidence`.

# SCORE-proxy decode window (2026-09-17)

SCORE uses two decode inputs from each pool's calibration: `score_proxy.decode_tps` and
`score_proxy.mean_decode_batch_ms`, both in `model_metrics.json`. `worker.calibrate` computes them from the
frozen loaded-phase trace `calibration_trace_<model>.csv`. The loaded phase sends 512 calibration prompts at
concurrency 128 with `max_completion_tokens` 8192.

Until now the summary averaged every pure-decode iteration in that trace. Sometimes one output runs to the
8,192-token cap after every other sequence has finished. The engine then decodes that one sequence alone for
about 50 s. Those single-sequence iterations become most of the pure-decode rows and pull both inputs down,
even though the engine's speed at each batch size has not changed.

This report checks three candidate rules for which rows count as loaded decode. It tests them against every
calibration trace on Vast: 39 model traces in 13 pools, copied read-only. The machine-readable data is in
`score-proxy-window.json`.

## Chosen rule

**`multi_sequence_pure_decode`**: *SCORE's decode throughput and mean decode-iteration latency are computed
over the pure-decode iterations of the loaded calibration phase that have at least two running sequences.*

The rule is implemented in `sfs_core.shared.trace_theta.estimate_score_proxy_metrics_from_batch_stats` as
`decode_window="multi_sequence_pure_decode"`. The default is still `all_pure_decode`, the old behavior, so
other callers are unchanged. `worker.calibrate` passes the new rule.

Each model's `score_proxy` in `model_metrics.json` now also records:

- `decode_window_rule`
- `decode_rows_excluded_by_window`
- `loaded_outputs`, `capped_loaded_outputs` and `loaded_max_completion_tokens`, counted from the recorded
  calibration responses

`qualification.json` gains `calibration_capped_outputs`, a per-model summary of the same fields for reviewers.
`service_rate_qps` and the methodology TPOT fit are unchanged.

The implementation was run again on all 39 Vast traces. Its results match the (a1) column below exactly, and
its capped-output counts match the table.

## Inventory and rule comparison

Candidate rules:

- **(a1)** Exclude rows with `num_seqs` = 1. This is the chosen rule.
- **(a2)** Exclude rows with `num_seqs` below 5% of concurrency, which is 7.
- **(b)** Cut the trace after the last row with `num_seqs` of at least 25% of concurrency, which is 32.
- **(c)** Cut the trace after the last decode row with `num_seqs` > k, where k is the model's number of
  capped loaded outputs. This is the trace-level form of "after the last completion that did not hit the cap".

How to read the table:

- Each cell shows `decode_tps / mean_decode_batch_ms`.
- The current column adds the number of pure-decode rows in brackets.
- The rule columns add the number of rows each rule excludes in parentheses.
- Bold marks the runs skewed by a single runaway output.

The stored `model_metrics.json` values equal the current column for all 39 traces. Every run has 512 loaded
outputs, so no responses were lost.

| Run | Profile | Model | Capped / 512 | Max output | current | (a1) num_seqs >= 2 **chosen** | (a2) num_seqs >= 7 | (b) 25% tail cut | (c) capped-tail cut |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| fcfs/state/calibrate-20260916 | unchunked FCFS 65k | qwen3-0.6b | 13 | 8192 | 1718 / 15.9 [9379] | 1722 / 16.1 (-143) | 1739 / 16.6 (-544) | 3572 / 25.0 (-7712) | 1802 / 17.1 (-1347) |
| fcfs/state/calibrate-20260916 | unchunked FCFS 65k | qwen3-8b | 0 | 2035 | 2667 / 22.7 [4531] | 2674 / 22.8 (-39) | 2725 / 23.7 (-306) | 2816 / 25.6 (-901) | 2667 / 22.7 (-0) |
| fcfs/state/calibrate-20260916 | unchunked FCFS 65k | **qwen3-32b** | 1 | 8192 | 1354 / 24.1 [10059] | 1596 / 27.5 (-2626) | 1637 / 28.1 (-2986) | 1690 / 28.7 (-3664) | 1596 / 27.5 (-2626) |
| fcfs/state/qualify-20260916 | unchunked FCFS 65k | qwen3-0.6b | 9 | 8192 | 1809 / 13.3 [9340] | 1811 / 13.3 (-54) | 1897 / 14.4 (-1309) | 3503 / 25.3 (-7618) | 2411 / 17.0 (-4717) |
| fcfs/state/qualify-20260916 | unchunked FCFS 65k | qwen3-8b | 0 | 1962 | 2662 / 22.7 [4465] | 2688 / 23.2 (-147) | 2719 / 23.7 (-293) | 2812 / 25.7 (-873) | 2662 / 22.7 (-0) |
| fcfs/state/qualify-20260916 | unchunked FCFS 65k | qwen3-32b | 0 | 1846 | 1601 / 27.3 [7419] | 1605 / 27.3 (-34) | 1654 / 28.0 (-447) | 1712 / 28.6 (-1093) | 1601 / 27.3 (-0) |
| fcfs/state/qualify-20260916-lane-a | unchunked FCFS 65k | qwen3-0.6b | 11 | 8192 | 1721 / 14.4 [9423] | 1737 / 15.1 (-556) | 1791 / 15.9 (-1374) | 3445 / 24.2 (-7606) | 2246 / 18.1 (-4726) |
| fcfs/state/qualify-20260916-lane-a | unchunked FCFS 65k | qwen3-8b | 0 | 2185 | 2648 / 22.1 [4720] | 2654 / 22.2 (-36) | 2715 / 23.3 (-384) | 2820 / 25.4 (-1022) | 2648 / 22.1 (-0) |
| fcfs/state/qualify-20260916-lane-a | unchunked FCFS 65k | qwen3-32b | 0 | 2215 | 1586 / 27.0 [7570] | 1616 / 27.4 (-258) | 1662 / 28.0 (-628) | 1711 / 28.5 (-1154) | 1586 / 27.0 (-0) |
| fcfs/state/qualify-20260916-merged | unchunked FCFS 65k | qwen3-0.6b | 10 | 8192 | 1714 / 14.3 [9445] | 1721 / 14.6 (-242) | 1791 / 15.8 (-1434) | 3713 / 23.5 (-7750) | 2016 / 16.7 (-3301) |
| fcfs/state/qualify-20260916-merged | unchunked FCFS 65k | qwen3-8b | 0 | 2003 | 2699 / 22.3 [4597] | 2708 / 22.5 (-48) | 2740 / 23.0 (-215) | 2850 / 25.3 (-928) | 2699 / 22.3 (-0) |
| fcfs/state/qualify-20260916-merged | unchunked FCFS 65k | qwen3-32b | 0 | 2007 | 1584 / 27.4 [7372] | 1615 / 27.9 (-257) | 1644 / 28.2 (-484) | 1693 / 28.7 (-1011) | 1584 / 27.4 (-0) |
| fcfs/state/qualify-sfs-score-lane-b | unchunked FCFS 65k | qwen3-0.6b | 10 | 8192 | 1836 / 13.8 [8960] | 1841 / 14.0 (-180) | 1877 / 14.9 (-953) | 3474 / 25.9 (-7273) | 2353 / 17.5 (-4346) |
| fcfs/state/qualify-sfs-score-lane-b | unchunked FCFS 65k | qwen3-8b | 0 | 2081 | 2622 / 21.4 [4857] | 2663 / 22.2 (-234) | 2738 / 23.4 (-629) | 2836 / 25.4 (-1251) | 2622 / 21.4 (-0) |
| fcfs/state/qualify-sfs-score-lane-b | unchunked FCFS 65k | **qwen3-32b** | 1 | 8192 | 1318 / 23.4 [10720] | 1620 / 27.4 (-3341) | 1668 / 28.0 (-3738) | 1721 / 28.5 (-4290) | 1620 / 27.4 (-3338) |
| state/baselines/ministral-20260916 | chunked 32k | ministral3-3b | 0 | 3162 | 3798 / 21.2 [4076] | 3798 / 21.2 (-3) | 3847 / 22.8 (-352) | 3935 / 24.4 (-744) | 3798 / 21.2 (-0) |
| state/baselines/ministral-20260916 | chunked 32k | ministral3-8b | 0 | 3367 | 2569 / 22.9 [6227] | 2592 / 23.4 (-195) | 2627 / 24.3 (-503) | 2687 / 25.5 (-1059) | 2569 / 22.9 (-0) |
| state/baselines/ministral-20260916 | chunked 32k | ministral3-14b | 0 | 3368 | 1736 / 24.8 [9028] | 1749 / 25.1 (-170) | 1770 / 25.5 (-447) | 1838 / 26.3 (-1315) | 1736 / 24.8 (-0) |
| state/baselines/ministral-20260916-mrb | chunked 32k | ministral3-3b | 0 | 2884 | 3681 / 21.2 [4230] | 3716 / 22.2 (-232) | 3754 / 23.0 (-428) | 3884 / 25.1 (-961) | 3681 / 21.2 (-0) |
| state/baselines/ministral-20260916-mrb | chunked 32k | ministral3-8b | 0 | 3347 | 2550 / 23.5 [6029] | 2564 / 23.8 (-120) | 2603 / 24.7 (-426) | 2683 / 26.3 (-1083) | 2550 / 23.5 (-0) |
| state/baselines/ministral-20260916-mrb | chunked 32k | ministral3-14b | 0 | 3420 | 1681 / 24.7 [9404] | 1697 / 25.0 (-211) | 1747 / 26.1 (-913) | 1791 / 26.8 (-1635) | 1681 / 24.7 (-0) |
| state/baselines/ministral-20260916-sr | chunked 32k | ministral3-3b | 0 | 3991 | 3567 / 16.9 [5403] | 3651 / 18.4 (-559) | 3827 / 22.9 (-1731) | 3903 / 24.8 (-2171) | 3567 / 16.9 (-0) |
| state/baselines/ministral-20260916-sr | chunked 32k | ministral3-8b | 0 | 3401 | 2554 / 23.2 [6249] | 2573 / 23.6 (-151) | 2605 / 24.3 (-416) | 2688 / 25.9 (-1112) | 2554 / 23.2 (-0) |
| state/baselines/ministral-20260916-sr | chunked 32k | ministral3-14b | 0 | 3401 | 1733 / 25.0 [8942] | 1742 / 25.2 (-112) | 1754 / 25.4 (-256) | 1841 / 26.4 (-1348) | 1733 / 25.0 (-0) |
| state/baselines/ministral-20260916-sr2 | chunked 32k | ministral3-3b | 0 | 2741 | 3665 / 19.4 [4608] | 3772 / 22.2 (-687) | 3808 / 23.3 (-918) | 3896 / 24.9 (-1301) | 3665 / 19.4 (-0) |
| state/baselines/ministral-20260916-sr2 | chunked 32k | ministral3-8b | 0 | 3347 | 2558 / 23.4 [6029] | 2573 / 23.8 (-120) | 2612 / 24.6 (-426) | 2693 / 26.2 (-1083) | 2558 / 23.4 (-0) |
| state/baselines/ministral-20260916-sr2 | chunked 32k | ministral3-14b | 0 | 3398 | 1667 / 25.0 [9310] | 1684 / 25.3 (-225) | 1721 / 26.1 (-717) | 1780 / 26.8 (-1510) | 1667 / 25.0 (-0) |
| state/baselines/qwen-20260916 | chunked 32k | qwen3-0.6b | 8 | 8192 | 1807 / 12.6 [9479] | 1813 / 12.8 (-194) | 1921 / 13.8 (-1484) | 3580 / 23.9 (-7719) | 2407 / 16.3 (-4778) |
| state/baselines/qwen-20260916 | chunked 32k | qwen3-8b | 0 | 2060 | 2758 / 23.3 [4138] | 2769 / 23.5 (-48) | 2808 / 24.1 (-230) | 2894 / 26.2 (-747) | 2758 / 23.3 (-0) |
| state/baselines/qwen-20260916 | chunked 32k | qwen3-32b | 0 | 2108 | 1702 / 28.0 [6732] | 1714 / 28.2 (-93) | 1758 / 28.8 (-426) | 1834 / 29.8 (-1099) | 1702 / 28.0 (-0) |
| state/baselines/qwen-20260916-mrb | chunked 32k | qwen3-0.6b | 10 | 8192 | 1846 / 13.1 [9509] | 1854 / 13.3 (-239) | 1938 / 14.5 (-1491) | 3443 / 24.6 (-7757) | 2256 / 16.0 (-4000) |
| state/baselines/qwen-20260916-mrb | chunked 32k | qwen3-8b | 0 | 1930 | 2697 / 23.2 [4361] | 2704 / 23.3 (-32) | 2796 / 24.8 (-458) | 2882 / 26.5 (-935) | 2697 / 23.2 (-0) |
| state/baselines/qwen-20260916-mrb | chunked 32k | qwen3-32b | 0 | 1910 | 1691 / 28.4 [6554] | 1703 / 28.6 (-84) | 1723 / 28.9 (-235) | 1809 / 29.9 (-990) | 1691 / 28.4 (-0) |
| state/baselines/qwen-20260916-sr2 | chunked 32k | qwen3-0.6b | 13 | 8192 | 1634 / 16.6 [9426] | 1635 / 16.7 (-48) | 1676 / 17.6 (-884) | 3613 / 24.9 (-7733) | 2146 / 19.6 (-4480) |
| state/baselines/qwen-20260916-sr2 | chunked 32k | **qwen3-8b** | 1 | 8192 | 1872 / 13.6 [11096] | 2735 / 23.6 (-6841) | 2806 / 24.9 (-7177) | 2876 / 26.7 (-7623) | 2734 / 23.6 (-6835) |
| state/baselines/qwen-20260916-sr2 | chunked 32k | qwen3-32b | 0 | 2054 | 1686 / 28.4 [6673] | 1715 / 28.8 (-216) | 1742 / 29.2 (-416) | 1811 / 30.1 (-1032) | 1686 / 28.4 (-0) |
| state/baselines/sfs-score-qwen-20260917 | chunked 32k | qwen3-0.6b | 10 | 8192 | 1828 / 13.2 [9421] | 1835 / 13.4 (-205) | 1908 / 14.5 (-1410) | 3606 / 25.6 (-7828) | 2389 / 17.0 (-4800) |
| state/baselines/sfs-score-qwen-20260917 | chunked 32k | **qwen3-8b** | 1 | 8192 | 1870 / 13.6 [11096] | 2733 / 23.7 (-6841) | 2804 / 24.9 (-7177) | 2875 / 26.7 (-7623) | 2732 / 23.6 (-6835) |
| state/baselines/sfs-score-qwen-20260917 | chunked 32k | qwen3-32b | 0 | 2014 | 1685 / 28.5 [6546] | 1697 / 28.7 (-91) | 1734 / 29.2 (-364) | 1795 / 30.1 (-915) | 1685 / 28.5 (-0) |

Spread across runs within each model and serving configuration, shown as max/min (`tps / ms`):

| Model | Profile | Runs | current | (a1) chosen | (a2) | (b) | (c) |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3-0.6b | unchunked FCFS 65k | 5 | 1.071 / 1.198 | 1.069 / 1.207 | 1.091 / 1.149 | 1.078 / 1.105 | 1.338 / 1.085 |
| qwen3-8b | unchunked FCFS 65k | 5 | 1.029 / 1.057 | 1.020 / 1.047 | 1.009 / 1.029 | 1.014 / 1.014 | 1.029 / 1.057 |
| qwen3-32b | unchunked FCFS 65k | 5 | 1.214 / 1.171 | 1.015 / 1.019 | 1.019 / 1.007 | 1.018 / 1.007 | 1.022 / 1.018 |
| ministral3-3b | chunked 32k | 4 | 1.065 / 1.255 | 1.040 / 1.207 | 1.025 / 1.019 | 1.013 / 1.028 | 1.065 / 1.255 |
| ministral3-8b | chunked 32k | 4 | 1.007 / 1.027 | 1.011 / 1.019 | 1.009 / 1.018 | 1.004 / 1.030 | 1.007 / 1.027 |
| ministral3-14b | chunked 32k | 4 | 1.041 / 1.013 | 1.039 / 1.013 | 1.028 / 1.028 | 1.035 / 1.020 | 1.041 / 1.013 |
| qwen3-0.6b | chunked 32k | 4 | 1.130 / 1.315 | 1.134 / 1.299 | 1.156 / 1.279 | 1.049 / 1.073 | 1.122 / 1.227 |
| qwen3-8b | chunked 32k | 4 | 1.474 / 1.716 | 1.024 / 1.014 | 1.004 / 1.032 | 1.007 / 1.020 | 1.022 / 1.018 |
| qwen3-32b | chunked 32k | 4 | 1.010 / 1.018 | 1.010 / 1.024 | 1.020 / 1.014 | 1.022 / 1.010 | 1.010 / 1.018 |

Largest change compared with the current summary:

| Rule | Max change, runs with no capped output (tps / ms) | Max change, qwen3-0.6b (8-13 capped) (tps / ms) |
|---|---:|---:|
| a_floor1 | 2.9% / 13.9% | 0.9% / 5.1% |
| a_floor5pct | 7.3% / 35.5% | 6.3% / 10.6% |
| b_tail25pct | 9.4% / 47.0% | 121.1% / 94.6% |
| c_capped_tail | 0.0% / 0.0% | 33.2% / 29.0% |

## Why rule (a1)

1. **It fixes the affected runs.** Four runs have a single capped output:
   - In the 8B chunked group, the spread falls from 1.47x (tps) and 1.72x (ms) to 1.02x for both.
   - In the 32B unchunked group, it falls from 1.21x (tps) and 1.17x (ms) to 1.02x for both.

   Rule (c) gives the same numbers on these four runs. Rules (a2) and (b) fix them too, but they also move
   every other run.
2. **It barely changes healthy runs.**
   - On runs with no capped output, tps changes by at most 2.9%.
   - Latency moves more than 5% in only two runs, both Ministral-3B: `-sr` goes from 16.9 to 18.4 ms and
     `-sr2` from 19.4 to 22.2 ms. The same thing happens there on a smaller scale: one long uncapped output
     (3,991 and 2,741 tokens) drains alone. Removing those rows narrows the Ministral-3B latency spread.
   - On qwen3-0.6b, tps moves by at most 0.9%. Every 0.6B run has 8-13 capped outputs, but never a single
     one draining alone.

   By comparison:
   - (a2) moves healthy runs by up to 7% in tps and 35% in latency.
   - (b) roughly doubles 0.6B tps to about 3,500, because it drops the whole post-peak drain in every run.
     It measures a different quantity.
   - (c) leaves runs with no capped output unchanged by construction. However, it moves every 0.6B run by
     29-33%, depending on where the tail of about 10 capped sequences happens to fall. This widens the
     unchunked 0.6B tps spread from 1.07x to 1.34x. It also misses a lone long output that is not capped.
3. **It is easy to state.** A loaded decode iteration has at least two running sequences. The rule does not
   depend on the token cap, the configured concurrency, or matching responses to trace rows.

**Caveat:** qwen3-0.6b always produces 8-13 capped outputs, which then drain together at 13 or fewer
sequences. Rule (a1) keeps those iterations. The 0.6B inputs (tps 1,635-1,854) are consistent across runs,
and `capped_loaded_outputs` now makes this visible to reviewers.

## Recomputed SCORE inputs for the affected runs

| Run | Model | Current tps / ms | (a1) tps / ms | Healthy runs, same model and configuration, under (a1) | Rows excluded by (a1) |
|---|---|---:|---:|---|---:|
| state/baselines/qwen-20260916-sr2 | qwen3-8b | 1,872 / 13.6 | **2,735 / 23.6** | qwen-20260916: 2,769 / 23.5; -mrb: 2,704 / 23.3 | 6,841 of 11,096 |
| state/baselines/sfs-score-qwen-20260917 | qwen3-8b | 1,870 / 13.6 | **2,733 / 23.7** | Same as above | 6,841 of 11,096 |
| fcfs/state/qualify-sfs-score-lane-b | qwen3-32b | 1,318 / 23.4 | **1,620 / 27.4** | qualify-20260916: 1,605 / 27.3; -lane-a: 1,616 / 27.4; -merged: 1,615 / 27.9 | 3,341 of 10,720 |
| fcfs/state/calibrate-20260916 | qwen3-32b | 1,354 / 24.1 | **1,596 / 27.5** | Same as above | 2,626 of 10,059 |

## Other calibration outputs in the affected pools (definitions not changed)

### `service_rate_qps` is skewed

It is 512 divided by the total elapsed time of the loaded phase, and that time includes the runaway tail.

| Model and configuration | Affected runs | Healthy runs | Difference |
|---|---|---|---:|
| 8B chunked | 2.303 qps (222 s), 2.290 qps (224 s) | 2.983 qps (172 s), 3.111 qps (165 s) | 23-26% lower |
| 32B unchunked | 1.366 qps (375 s), 1.327 qps (386 s) | 1.455-1.514 qps (338-352 s) | 6-12% lower |

Only the LMDeploy policy uses this rate. On 2026-09-17 the affected pools' `cells/` held no LMDeploy cell: `qwen-20260916-sr2` had `qwen-vllm_sr_latency-{7,8,8.3}` and `sfs-score-qwen-20260917` had `qwen-hard-6`. If an LMDeploy cell later uses either qualification, it gets the lower rate.

### The TPOT fit's function is essentially unskewed, but its held-out diagnostics are skewed

The XGBoost head predicts iteration time from `decode`, `prefill` and `sum_tokens`. At 8-128 decode tokens,
the affected heads predict values within the healthy range for the same configuration. Predictions for 8B
chunked, assuming 2,500 context tokens per sequence:

| Batch size | 1 | 8 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|
| sr2 (ms) | 7.3 | 9.9 | 15.7 | 24.7 | 25.9 |
| Healthy runs (ms) | 9.2-9.5 | 9.3-9.9 | 15.7-16.3 | 23.0-25.3 | 23.5-25.9 |

At batch size 1, the affected heads predict about 1.5-2 ms lower than healthy runs, for both 8B and 32B.

The held-out diagnostics use the last 20% of the trace in time order. In the affected pools that slice is
made up entirely of single-sequence rows:

| Model | Affected pools | Healthy pools |
|---|---|---|
| 8B | 2,249 rows, all in the `decode < 2` bucket; MAE 0.03-0.05 ms | 860-998 rows; MAE 3.2-6.8 ms |
| 32B | 2,037 and 2,168 rows; MAE 0.42 and 0.31 ms | 1,339-1,540 rows; MAE 4.7-6.9 ms |

So in those pools the held-out numbers say nothing about accuracy under load.

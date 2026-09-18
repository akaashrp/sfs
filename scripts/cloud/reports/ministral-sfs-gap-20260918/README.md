# Why SFS (`hard`) loses to every baseline on the Ministral family

Diagnosis only. Read-only over saved Vast state and the controller checkout; no supervisor program was
started or stopped, no checkout or campaign state was modified, no GPU was used, and every CPU command on
Vast ran under `taskset -c 84-95`. Controller worktree `sfs_cloud_20260914`, branch
`experiments/cloud-68-20260914`, tip `bc72345`.

Sources: pool `/workspace/sfs/state/baselines/sfs-score-ministral-20260917` (checkout
`/workspace/sfs/repo-v4` at `c62c6ed`, 3 models TP 1/1/1 on GPUs 4,5,6, remaining-length rule OFF for all
Ministral models, 8,000 requests per cell), pool `sfs-score-qwen2-20260917` for the Qwen comparison, and
pools `ministral-20260916` / `ministral-20260916-mrb` for the Ministral baselines.

---

## Summary

SFS's wait estimate on Ministral over-predicts the per-batch execution time of the serving engine by
**107x to 257x**, at every arrival rate, because the simulator is fed a coefficient that was fitted for one
feature and consumed against a different feature. The consequence is not a mild miscalibration: SFS's hard
TTFT feasibility filter reports **no feasible candidate on 99.78% / 99.88% / 99.88% of decisions** at
6.0125 / 7.8625 / 8.7875 QPS (Qwen at 8.3: **0.29%**), so the policy silently degenerates to
`argmin(predicted wait)` on every Ministral cell. That fallback is not capacity-aware, and it overloads the
14B engine by 1.39x-1.89x its share of aggregate service capacity, which is what costs SFS the 0.01-0.08
utility at 6.0125/7.8625 and produces the collapse at 8.7875.

> **Status note.** While this diagnosis was being written, a concurrent session landed the P0 fix in the
> working tree of this worktree (uncommitted: `src/scripts/cloud/pool.py`, `src/scripts/runs/qwen_baselines.py`,
> `src/scripts/cloud/tests/test_remaining_length_flag.py`). It carries `feature_set` into the pool config,
> refuses an unrecognised or absent one, and routes both families through
> `service_metrics_config.build_simulation_args`. That is exactly the change section 5's P0 asks for; P0
> below is therefore now a validation-and-rerun item, not an implementation item. Nothing in this report was
> produced with that code, and none of the measurements change.

This is a **plumbing defect in the campaign's server launch path**, not a defect of the SFS method, the
batch-latency fit, the length predictor, the reserve rule, the KV/preemption model or the TTFT stop-mode
semantics. The underlying batch-latency model is excellent (R^2 0.994-0.999; median predicted/actual 0.98-1.00
when the correct feature is used). Qwen is structurally immune. The three Ministral `hard` cells are not a
valid measurement of SFS and must be rerun after the fix; the Qwen cells are untouched by the fix and must
not be rerun.

---

## 1. Mechanism

### 1.1 The defect, exactly

`bundle/ministral/bridges_metrics.json` records, per model:

```json
"sfs_simulation": {"feature_set": "cross_term", "intercept": 0.0068052, ..., "sum_sq_coeff": 1.1780830225912282e-09}
```

`feature_set: "cross_term"` means the sixth coefficient is the **prefill x processed-context** coefficient
(`p_x_ctx`), to be multiplied by `sum_i (prefill_tokens_i * processed_ctx_i)`.

`src/scripts/cloud/pool.py:31-32` copies six coefficient keys into `instances.json`'s `ttft_batch_model` and
**drops `feature_set`**:

```python
"ttft_batch_model": {k: rows[model]["sfs_simulation"][k] for k in
    ("intercept", "prefill_coeff", "prefill_sq_coeff", "decode_coeff", "sum_coeff", "sum_sq_coeff")}
```

`src/scripts/cloud/pool.py:110-111` (the non-Qwen branch of `server_argv`) then emits them positionally:

```python
for key, value in row["ttft_batch_model"].items():
    argv += ["--simulation-" + key.replace("_", "-"), str(value)]
```

so the launched server receives `--simulation-sum-sq-coeff 1.1780830225912282e-09` and **no
`--simulation-batch-time-feature-set`**. vLLM's default is `legacy`
(`vllm/vllm/config/scheduler.py:136`), under which the simulator multiplies that coefficient by
`sum_sq_tokens = sum_i (context_len_i)^2` instead:

```
vllm/vllm/v1/engine/scheduler_simulator.py:1060-1061   sum_sq_tokens += request_tokens * request_tokens
vllm/vllm/v1/engine/scheduler_simulator.py:1383-1399   context_feature_value = prefill_x_processed_ctx_sum if cross_term else sum_sq_tokens
vllm/vllm/v1/engine/scheduler_simulator.py:1209-1232   batch_ms = (intercept + p*P + p_sq*P_sq + d*D + s*C + ctx_coeff*context_feature_value) * 1000
```

The correct mapping already exists elsewhere and is not used here:
`src/scripts/runs/service_metrics_config.py:262-286` (`build_simulation_args`) switches the option name on
`feature_set` and always emits `--simulation-batch-time-feature-set`. The vLLM-side validation at
`vllm/vllm/config/scheduler.py:358-379` cannot fire, because the argv names a legal legacy option.

The value reaches the router's simulator through the SHM header, not `instances.json`
(`src/sfs_core/routing/snapshot_shm_client.py:421-441`), so `instances.json` is not the fix point.

Confirmed in `server_argv_ministral3-8b.json` of the pool: `--simulation-sum-sq-coeff
1.1780830225912282e-09`, no feature-set flag.

### 1.2 Magnitude, measured against real engine steps

Every server in both pools writes `batch_stats_<model>.csv` with the exact regression features per executed
step (`prefill`, `prefill_sq_sum`, `decode`, `sum_tokens`, `sum_sq_tokens`, `prefill_x_processed_ctx_sum`)
and the measured `exec` time. Scoring those rows with the coefficients the server was actually launched with,
under both readings of the sixth feature (`evidence/batch-time-model-validation.json`, `batchfit.py`):

| engine | rows | measured step p50 | as shipped (legacy, `sum_sq_tokens`) | ratio | intended (`cross_term`) | ratio |
|---|---:|---:|---:|---:|---:|---:|
| ministral3-3b | 354,689 | 7.93 ms | 1,328.70 ms | **161.9x** | 7.77 ms | 0.996 |
| ministral3-8b | 200,910 | 14.69 ms | 2,102.72 ms | **143.9x** | 14.21 ms | 0.985 |
| ministral3-14b | 157,461 | 21.03 ms | 2,380.48 ms | **106.9x** | 21.02 ms | 1.004 |
| qwen3-0.6b | 949,274 | 2.14 ms | 2.66 ms | 1.25x | — | — |
| qwen3-8b | 237,027 | 26.62 ms | 27.22 ms | 1.02x | — | — |
| qwen3-32b | 191,041 | 30.48 ms | 30.26 ms | 1.04x | — | — |

Restricted to decode-only steps at concurrency >= 50 (the regime that matters under backlog), the Ministral
ratios rise to 256.9x (3B), 208.0x (8B) and 138.1x (14B), while `cross_term` stays at 1.03 / 1.00 / 1.02. The
spurious `sum_sq_coeff * sum_i ctx_i^2` term contributes **more than 99%** of the shipped prediction
(median 1,319 / 2,086 / 2,355 ms of a 1,329 / 2,103 / 2,380 ms prediction). It is not a physical term: decode
step time scales with the total KV read, `sum_i ctx_i`, which the `sum_coeff` term already carries.

The error grows with the square of each running request's context, so it is largest exactly where the router
needs the estimate to be right: a deeply loaded engine holding long govreport contexts.

### 1.3 Walk of one late request (8.7875 cell)

`req-7950`, bucket `govreport-summarization`, 7,519 prompt tokens, routed to `vllm-ministral3-8b`:

| field | value |
|---|---|
| `ttft_slo_ms` | 413.998 |
| `wait_time_ms` = `live_wait_time_ms` = `wait_estimates_ms.live` | **5,748,046.7 ms** |
| `queue_delay_ms` (measured) | 35,869.6 ms |
| `ttft_ms` (measured) | 36,333.9 ms |
| `prefill_ms` (measured) | 464.3 ms |
| `score_candidate_terms` / `score_policy_terms` / `methodology_terms` | `None` (only populated for `hard_score_proxy` / methodology policies) |
| `wait_time_metadata.simulation_mode` | `critical_path_prefill_done` |
| `wait_time_metadata.num_batches` | 1,047 |
| `wait_time_metadata.running_at_snapshot` / `queued_at_snapshot` | 70 / 110 |
| `wait_time_metadata.num_running` / `num_waiting` at stop | 83 / 0 |
| `wait_time_metadata.prefill_backlog_total_tokens` | 292,858 |
| `wait_time_metadata.total_prefill_tokens` / `total_decode_tokens` | 432,201 / 109,761 |

Decomposition: 5,748,046.7 ms / 1,047 batches = **5,490 ms per simulated batch**, against a measured
8B step time of 26.7 ms at concurrency 70-90. The simulated batches themselves are ordinary: on average 413
prefill tokens and 105 decode tokens each. Term by term for a representative decode step of that snapshot
(83 running requests, ~8,000 tokens of context each):

| term | value |
|---|---:|
| intercept | 6.8 ms |
| `decode_coeff * D` (83) | 0.8 ms |
| `sum_coeff * sum ctx` (~664,000) | 33.1 ms |
| **`sum_sq_coeff * sum ctx^2`** (~5.3e9) | **~6,250 ms** |
| correct `cross_term` value for a decode step | 0 ms |

So the whole estimate is the spurious term. Nothing is contributed by prefill backlog beyond a few percent,
nothing by KV pressure or preemption, and nothing by remaining-decode targets.

### 1.4 The simulator's *scheduling* is right; only its clock is wrong

Replacing the simulated per-batch time with the **measured** median step time of that engine at the
concurrency the snapshot reported, and keeping the simulation's own `num_batches`
(`evidence/corrected-estimate-check.json`, `corrected.py`):

| 8.7875 cell, arrival quintile | raw predicted p50 | corrected predicted p50 | measured `queue_delay_ms` p50 | raw / actual | corrected / actual |
|---|---:|---:|---:|---:|---:|
| 1 | 7,130 ms | 26 ms | 0.01 ms | 3.0x | — |
| 2 | 15,412 ms | 34 ms | 2,743 ms | 20.9x | 0.05 |
| 3 | 1,987,161 ms | 8,234 ms | 9,845 ms | 184.7x | **0.77** |
| 4 | 2,885,773 ms | 13,043 ms | 15,541 ms | 198.5x | **0.84** |
| 5 | 4,272,965 ms | 18,742 ms | 23,339 ms | 185.9x | **0.81** |

In the saturated quintiles the corrected estimate lands within 16-23% of the measured queue delay. The
simulator's admission logic, KV accounting, preemption model, backlog reconstruction and pending-dispatch
overlay are therefore all behaving; the single broken input is the per-batch latency.

### 1.5 Hypotheses (a)-(e)

| # | hypothesis | verdict | evidence |
|---|---|---|---|
| a | remaining-decode targets inflated (rule OFF, predictor bias) | **ruled out** | `reserve-tail-ministral-20260917` measures an *under*-count of aggregate decode backlog by a median 13-26% at the warm reserve, i.e. the wrong direction; and at 6.0125 the median simulation runs `num_batches` = **1**, where no decode target can enter the estimate, yet the estimate is already 2,394 ms against a 166 ms SLO. |
| b | KV exhaustion / repeated preemption loops in the simulator | **ruled out** | `num_batches` matches reality to within 16-23% (1.4); a preemption loop would inflate the batch count, not the per-batch time. `total_prefill_tokens` / `prefill_backlog_total_tokens` = 1.48x for `req-7950`, a second-order effect, not 150x. KV pressure on the real engine is real (8B `sum_tokens` peaks at 391,000 against a 391,232-token KV cache) but is modelled, not the source of the error. |
| c | pending-dispatch overlay double counts in-flight requests | **ruled out** | at 6.0125 `queued_at_snapshot` p50 = 0 and the over-prediction is already 2,394 ms; the overlay's id-matching guard (`bindings.cpp:566-570`) and the `observed_pending_request_ids` reconciliation are intact. |
| d | Ministral batch timing coefficients mispredict under deep queues | **ruled in, with a precise cause** | the coefficients are fine (R^2 0.9943-0.9989 on 103k-363k fit rows; median predicted/actual 0.985-1.004 on 713k measured Vast steps under the *intended* feature). They are consumed against the wrong feature (1.1). |
| e | TTFT-feasibility semantics (simulate to prefill-done behind a long backlog) | **ruled out** | at 6.0125 the median simulation is a single batch and still over-predicts 14x the SLO; the stop mode is not what inflates the number. |

### 1.6 What the broken estimate does to routing

The `hard` utility (`src/sfs_core/routing/score_proxy.py:54-83`) returns `-inf` for every candidate that
misses its TTFT SLO, `quality - lambda*cost` among feasible candidates, and falls back to `-wait_ms` (i.e.
argmin predicted wait) **only when no candidate is feasible**. Because the winner is always a feasible
candidate when one exists, a request whose *selected* engine was predicted infeasible proves that no
candidate was feasible. Measured (`evidence/wait-estimate-vs-measured.json`):

| cell | predicted wait p50 | TTFT SLO p50 | measured TTFT p50 | decisions with no feasible candidate |
|---|---:|---:|---:|---:|
| ministral-hard-6.0125 | 2,393.98 ms | 165.60 ms | 25-45 ms | **99.78%** |
| ministral-hard-7.8625 | 5,412.51 ms | 165.60 ms | 31-85 ms | **99.88%** |
| ministral-hard-8.7875 | 1,937,243.85 ms | 165.60 ms | 8,682-11,031 ms | **99.88%** |
| qwen-hard-8.3 | 60.34 ms | 166.01 ms | 26-123 ms | **0.29%** |

So on Ministral the SFS SLO filter never fires and the policy is, in practice, `argmin(c_m * sum ctx^2)` —
a squared-context-mass balancer whose coefficients (3B 8.63e-10, 8B 1.18e-9, 14B 1.48e-9) differ by only
1.7x while the engines' service rates differ by 2.2x. Nothing in that quantity is capacity-aware, and it
systematically favours whichever engine currently holds the least squared context; the 14B, which under this
rule receives mostly short prompts (routed prompt p50 62-65 tokens against 1,220-1,344 on the 3B), keeps its
squared-context mass low and keeps attracting traffic. The result
(`evidence/per-engine-capacity.json`):

| rate | policy | 3B share / cap ratio | 8B share / cap ratio | 14B share / cap ratio |
|---|---|---|---|---|
| 6.0125 | `hard` | 0.327 / 0.71 | 0.315 / 0.96 | 0.358 / **1.70** |
| 6.0125 | shortest queue | 0.466 / 1.01 | 0.310 / 0.94 | 0.224 / 1.07 |
| 7.8625 | `hard` | 0.311 / 0.68 | 0.291 / 0.88 | 0.398 / **1.89** |
| 7.8625 | shortest queue | 0.469 / 1.02 | 0.311 / 0.94 | 0.220 / 1.05 |
| 8.7875 | `hard` | 0.391 / 0.85 | 0.316 / 0.96 | 0.293 / **1.39** |
| 8.7875 | shortest queue | 0.473 / 1.03 | 0.334 / 1.01 | 0.194 / 0.92 |

(capacity share = the engine's share of the calibrated aggregate service rate 3.2368 + 2.3141 + 1.4779 =
7.0289 QPS; that rate is a concurrency-128 calibration figure, not router capacity, and is used here only
for relative shares.)

---

## 2. Why Qwen is unaffected

`bundle/qwen/bridges_metrics.json` has **no `sfs_simulation` block at all**. The Qwen servers are launched
through `scripts/runs/qwen_baselines.py` with the hard-coded coefficients at
`src/scripts/runs/experiments.py:2614-2646`, which are genuine **legacy** (`sum ctx^2`) coefficients:

| | 0.6B | 8B | 32B |
|---|---:|---:|---:|
| Qwen `sum_sq_coeff` (legacy, correct) | 1.575e-13 | 2.685e-13 | 4.520e-13 |
| Ministral sixth coefficient (cross term, mis-consumed) | 8.632e-10 | 1.178e-09 | 1.479e-09 |

The Ministral 8B value is **4,388x** the Qwen 8B value. The `sum ctx^2` term contributes a median of
0.79-1.54 ms to a Qwen step (measured) against 1,319-6,913 ms on Ministral, so on Qwen the term is a genuine,
small, correctly-fitted curvature correction.

Everything else about the two pools is close enough that it cannot explain the gap
(`evidence/coefficient-provenance.json`):

| | Ministral | Qwen |
|---|---|---|
| models | 3 (3B / 8B / 14B) | 3 (0.6B / 8B / 32B) |
| tensor parallel | 1 / 1 / 1 | 1 / 1 / **2** |
| `max_num_seqs` | 512 | 512 |
| `max_num_batched_tokens` | 32,768 | 32,768 |
| `gpu_memory_utilization` | 0.90 | 0.90 |
| prefix caching / chunked prefill | off / on | off / on |
| GPU KV cache (tokens) | 617,984 / 391,232 / 267,488 | 628,512 / 382,304 / 298,416 |
| remaining-length rule | off on all three | `qwen3-0.6b = running_all:0.5:prompt_bin`, 8B/32B off |
| offered rate vs aggregate calibrated service rate | 8.7875 / 7.0289 = **1.25x** | 8.3 / 7.3086 = 1.14x |
| predicted wait p50 vs TTFT SLO p50 | 1,937,244 vs 166 ms | 60 vs 166 ms |
| decisions with no feasible candidate | 99.88% | 0.29% |

The reserve rule being ON for `qwen3-0.6b` only is not the differentiator: the Qwen 8B and 32B engines run
the rule OFF, exactly as all three Ministral engines do, and their predicted/actual step ratios are 1.02 and
1.04. The KV capacities and the load-vs-capacity ratios are comparable. The single structural difference is
the provenance of the sixth coefficient.

**Blast radius.** `hard` is the only campaign policy whose routing value is the native simulation's
`estimated_wait_ms` (`_baseline_runtime_params`, `experiments.py:6410-6418` -> `_wait_estimator_live`,
`:3134`). `score` (`_wait_estimator_score_total_latency`, `:3779`), `hard_score_proxy` (`:3656`),
`hard_prefill_tps` / `soft_prefill_tps` (`:3587`), `hard_pk_mg1` / `soft_pk_mg1` (`:3914`) all consume
calibrated prefill/decode TPS or theta_p, never the simulation coefficients. `shortest_queue` computes the
live estimate but routes on `effective_num_requests` (`_select_shortest_queue_instance`, `:1185`), which is
why it reaches 97.81% on the 3B at 8.7875 while SFS reaches 11.09%. So:

- affected, run: `ministral-hard-6.0125`, `ministral-hard-7.8625`, `ministral-hard-8.7875`;
- **not** affected: `ministral-score-*` (6.0125 in flight at the time of writing, 7.8625/8.7875 pending),
  any pending Ministral `hard_prefill_tps`, and every Ministral baseline cell already recorded in
  `ministral-20260916` / `ministral-20260916-mrb`;
- **not** affected: the entire Qwen campaign, all serving-configuration overlays on Qwen, and the predictor
  ablations on Qwen.
- Unverified: whether the April Bridges Ministral SFS reference was launched through the same `pool.py`
  branch. It should be checked before that reference is cited alongside corrected Vast numbers.

Incidentally, the analytic legacy-only TTFT estimator `_estimate_live_prefill_term_ms`
(`experiments.py:3064`) that applies `sum_sq_coeff * sum_sq_tokens` in Python has **zero load-uses**, as do
`_validate_live_ttft_params` (`:2765`) and `_instance_ttft_params_from_metadata` (`:2749`). The
`ttft_batch_model` block in `instances.json` is therefore write-only and never reaches a routing decision.

---

## 3. Is SFS's routing wrong, or is the SLO unattainable at 8.7875?

Both, in that order. At 8.7875 the offered rate is 1.25x the aggregate calibrated service rate, so 100%
attainment is not available; but a policy that concentrates the shortfall on one engine keeps most requests
inside SLO, and SFS does not do that.

Per-engine evidence at 8.7875 (`evidence/per-engine-capacity.json`):

| policy | engine | routed | engine QPS | TTFT attain | TTFT p50 | TTFT p90 | decode tok/s | prompt tok/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `hard` | 3B | 3,128 | 3.054 | 11.09% | 8,682 ms | 21,566 ms | 2,300 | 11,004 |
| `hard` | 8B | 2,531 | 2.471 | 14.46% | 11,031 ms | 27,861 ms | 1,875 | 6,729 |
| `hard` | 14B | 2,341 | 2.286 | 13.71% | 9,012 ms | 30,058 ms | 1,656 | 4,474 |
| shortest queue | 3B | 3,782 | 3.899 | **97.81%** | 33 ms | 184 ms | 2,746 | 11,365 |
| shortest queue | 8B | 2,669 | 2.752 | **73.77%** | 64 ms | 3,371 ms | 2,044 | 7,289 |
| shortest queue | 14B | 1,549 | 1.597 | 12.72% | 15,873 ms | 32,605 ms | 1,251 | 4,794 |

Shortest queue pushes the 3B to **3.899 QPS** (1.20x its calibrated 3.237) and the 8B to **2.752 QPS**
(1.19x its 2.314) and still holds them at 97.8% and 73.8%, while letting the 14B absorb the entire overload
at 1.597 QPS (1.08x its 1.478) and 12.7%. So the per-engine feasible frontier at this load is roughly:
**3B feasible to at least 3.9 QPS, 8B marginal at 2.75 QPS, 14B infeasible above ~1.6 QPS.** SFS instead runs
all three engines at 2.29-3.05 QPS, i.e. every engine at or past its own frontier, and loses all three.

A counterfactual "SFS with the shortest-queue split" is not measurable from saved data, and no number in
this report should be read as one. What the data does establish is (i) a feasible split exists at this load
and reaches 73.15% aggregate TTFT attainment, (ii) SFS's split gives the 14B 1.39x its capacity share while
under-using the 3B at 0.85x, and (iii) the quantity SFS minimises in these decisions (`c_m * sum ctx^2`)
carries no information about that frontier.

The same pattern, milder, explains the losses at the lower rates: at 7.8625 SFS gives the 14B 3,181
requests (1.89x its capacity share, 2.908 QPS) and the 14B attains 68.78% with TTFT p90 3,252 ms, while
mooncake's 14B at 1.832 QPS attains 98.15% and shortest queue's at 1.656 QPS attains 75.04%. At 6.0125 SFS
gives the 14B 2,866 (1.70x share) and attains 94.63% there, against 99.65% for mooncake's 853.

---

## 4. Does the over-prediction exist at 6.0125 and 7.8625?

Yes, at every rate, and it is the same defect (`evidence/wait-estimate-vs-measured.json`).

| cell | predicted wait p50 | p90 | measured `queue_delay_ms` p50 / p90 | `num_batches` p50 | implied ms per simulated batch p50 | measured step p50 |
|---|---:|---:|---:|---:|---:|---:|
| 6.0125 | 2,393.98 ms | 3,834.87 ms | 0.00 / 0.01 ms | 1 | 2,392 ms | 8-21 ms |
| 7.8625 | 5,412.51 ms | 7,242.54 ms | 0.00 / 0.02 ms | 1 | 5,378 ms | 8-21 ms |
| 8.7875 | 1,937,243.85 ms | 4,294,246.14 ms | 9,011 / 24,970 ms | 309-703 (q3-q5) | 5,949-6,628 ms | 26-28 ms |
| qwen 8.3 | 60.34 ms | 227.34 ms | 0.00 / 0.01 ms | 1 | 60 ms | 2-31 ms |

At 6.0125 and 7.8625 the simulation still ends in a single batch — the queue is genuinely empty — but that
one batch is priced at 2,392 ms and 5,378 ms instead of 8-21 ms, which is already 14x and 32x the median TTFT
SLO of 165.6 ms. Hence the 99.78% / 99.88% infeasible-fallback rates and the anti-capacity split at those
rates too. **The same defect explains the whole Ministral family, not only the 8.7875 collapse.**

---

## 5. Recommendations, in priority order

**P0 - Feature-set plumbing in `src/scripts/cloud/pool.py` (required before any Ministral SFS number is
reported). Already implemented in the working tree; validate it.**
Evidence: sections 1.1-1.2; the correct mapping already existed in
`service_metrics_config.build_simulation_args`. The landed change does the two things this diagnosis calls
for: (i) carries `feature_set` through `config()` into the instance row (`batch_time_feature_set`), raising
on an unrecognised value rather than defaulting; (ii) replaces the positional argv loop in `server_argv`
with `build_simulation_args`, so Ministral now launches with `--simulation-batch-time-feature-set=cross_term`
and `--simulation-prefill-x-context-coeff=<value>`, and refuses an `instances.json` that predates the fix.
The same treatment was applied to the Qwen path with `FEATURE_SET = "legacy"`, which preserves Qwen's
existing argv exactly apart from the newly explicit feature-set flag - worth confirming byte-for-byte before
the next Qwen pool starts, since that is the only way this change could touch Qwen at all.
Cost to validate: one paired `ministral-hard-8.7875` cell, ~40 min on 3 GPUs; a CPU-only pre-check is
cheaper still — replay `batchfit.py` against the first few thousand rows of the new `batch_stats` CSV and
require the median predicted/actual step ratio inside [0.8, 1.25], which the current build fails at 144x.
**Does not require rerunning any Qwen cell:** Qwen never traverses this code path
(`pool.py:19-21` dispatches Qwen to `qwen_baselines.pool_config` / `server_argv`), its coefficients are
legacy and measured accurate to 1.02-1.05x, and its SHM header is byte-identical before and after.

**P1 - Rerun the three Ministral `hard` cells after P0; do not publish the current three.**
Evidence: section 1.6 - the SLO filter was dead on 99.8%+ of decisions, so these cells measure
`argmin(c_m * sum ctx^2)`, not SFS.
Cost: 3 cells x ~40 min = ~2 h on GPUs 4,5,6, to be scheduled after the in-flight Ministral SCORE cells.
The Ministral SCORE cells do **not** need rerunning (section 2, blast radius).

**P2 - Make the class of defect impossible.**
(a) Done by the landed P0 change: `build_simulation_args` is now the only argv producer for both families.
(b) `feature_set` now reaches `instances.json` as `batch_time_feature_set`; still worth asserting at router
start that the SHM header's coefficients and the manifest's feature set agree, since the router reads the
header and not `instances.json` (`snapshot_shm_client.py:421-441`).
(c) Turn the calibration JSON's existing `"requires_runtime_residual_audit": true` into an actual
qualification gate: after warm-up, score the first N rows of `batch_stats_<model>.csv` with the launched
coefficients and fail the pool if the median predicted/actual step ratio falls outside [0.5, 2.0]. That gate
alone would have caught this at 107-257x, on any family, on the first smoke run.
(d) Consider making `--simulation-batch-time-feature-set` mandatory (no default) in the fork.
Cost to validate: CPU only, no GPU; runs inside the existing smoke phase.

**P3 - Do not enable the remaining-length rule for Ministral on the strength of this failure.**
Evidence: section 1.5(a) - the rule is not implicated, and the offline evidence in
`reserve-tail-ministral-20260917` (an under-count of 13-26%, none of the Qwen3-0.6B failure mode, cap rate
0.05-0.14%) still argues against it. Revisit only if a corrected paired run at 8.7875 shows the estimate
still biased; that decision costs one further paired cell (~40 min) and, again, touches no Qwen cell.

**P4 - Remove or guard the dead analytic estimator.**
`_estimate_live_prefill_term_ms` (`experiments.py:3064`), `_validate_live_ttft_params` (`:2765`) and
`_instance_ttft_params_from_metadata` (`:2749`) have no callers, and the first hard-codes the legacy form
with no feature-set awareness. It is a loaded gun for the next person who wires it up. Delete it, or raise
loudly if it is ever called with a non-zero `sum_sq_coeff` and no feature-set provenance.
Cost: CPU only.

**P5 - Reporting.** Until P1 lands, the Ministral family has no valid SFS measurement. The honest interim
statement is that the Ministral SFS cells were invalidated by a launch-path defect in the batch-latency
feature mapping, with the defect, its magnitude and its containment to `hard`-on-Ministral all measured.
Nothing in the Qwen results, the serving-configuration overlays, or the predictor ablations is affected.

---

## Files

| file | content |
|---|---|
| `evidence/coefficient-provenance.json` | the defect, its code sites, both families' coefficients and engine configuration, and the per-policy exposure analysis |
| `evidence/batch-time-model-validation.json` | 2.09M measured engine steps scored under both readings of the sixth feature, per model, by batch composition and concurrency |
| `evidence/wait-estimate-vs-measured.json` | per-request predicted wait vs measured queue delay / TTFT, by arrival quintile and engine, for the three Ministral `hard` cells and `qwen-hard-8.3`, including the infeasible-fallback rate |
| `evidence/corrected-estimate-check.json` | `num_batches` x measured step time vs measured queue delay, plus the measured step-time-by-concurrency table per engine |
| `evidence/per-engine-capacity.json` | per-engine routed counts, QPS, attainment, TTFT percentiles and token throughput for `hard`, shortest queue, mooncake and RouteBalance at all three rates, with capacity shares |
| `batchfit.py`, `cellstats.py`, `corrected.py`, `capacity.py` | the analysis scripts, run read-only on Vast under `taskset -c 84-95` |

---

## Resolved after the report: the Bridges Ministral runs were never affected

The report left open whether the April Bridges Ministral SFS reference carries the same defect. It does
not. `src/slurm/runs/ministral3_router_common.sh:135-150` builds every Ministral server's simulation
arguments through `scripts.runs.service_metrics_config simulation-args`, i.e. the same
`build_simulation_args` the fix now routes the cloud pools through, under
`--expected-feature-set cross_term`; it then refuses to launch unless exactly seven arguments come back,
which is the feature-set flag plus the six coefficients. The defect was confined to the second launch
path, `scripts/cloud/pool.py`, which reimplemented that construction and dropped the declaration.

So Bridges Ministral numbers stay citable alongside the corrected cloud reruns, and the blast radius is
exactly the three `ministral-hard-*` cells run on Vast.

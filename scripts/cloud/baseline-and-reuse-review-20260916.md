# Baseline equivalence and Bridges reuse review

Main cloud jobs remain paused. The canonical Qwen controls are independent and
must remain uninterrupted. Qwen's planned grid is 7, 8, 8.6, 8.75 QPS.
Ministral's committed points are 6.0125, 7.8625, 8.7875; a common fourth point
is a follow-up decision. Existing SLOs are retained.

## Estimator similarity does not establish routing equivalence

`src/scripts/runs/experiments.py::_wait_estimator_prefill_tps_ttft` computes
`(prefill_backlog_tokens + incoming_prompt_tokens) / fixed_prefill_tps`.
The `hard_prefill_tps` configuration retains SFS's utility selector, quality
predictor, token cost and TTFT feasibility logic. It changes the estimator.

LMDeploy's pinned proxy selector minimizes unfinished requests divided by
configured speed, with random ties. Its count spans forwarding through terminal
completion, including decode. It does not count remaining prefill tokens or
optimize quality/cost. See the [pinned upstream implementation](https://github.com/InternLM/lmdeploy/blob/a76d91dbc7f812c340c4a3a7b2a60afe80ebfc6d/lmdeploy/serve/proxy/proxy.py).
Locally this is `methodology_policies.py::select_lmdeploy` and
`RequestLifetimeLedger`. It resembles a throughput-weighted shortest queue,
but unweighted shortest queue is equivalent only with equal speeds and the
same counter semantics.

Mooncake estimates request-specific prefill execution from offline profiles
and sums queued prefill times; prompt length and cached-prefix length matter.
The complete system also handles transfer, decode placement and rejection.
See [paper section 6.1](https://arxiv.org/html/2407.00079v4#S6.SS1).
Our cache-disabled prefill-placement adaptation minimizes fitted remaining
prefill work plus incoming work. Its polynomial fit is not constant TPS.
Only if the fit were purely linear with zero intercept and identical backlog
accounting would its estimate reduce to token-backlog/TPS. SFS's selector
would still differ. Do not relabel `hard_prefill_tps` as Mooncake.

The current Mooncake main tree, commit
`97abb51488208b97b1f19a775df70a03e861241b`, includes Conductor prefix-indexing
and KV-event code and an architecture document. The nontruncated recursive
tree inspection did not identify an implemented paper prefill-duration head
or complete request-placement scheduler in that component. This is a bounded
inspection, not a claim that no relevant implementation exists on any branch.

[RouteBalance section 4.2](https://arxiv.org/html/2606.17949v1#S4.SS2)
uses XGBoost TPOT heads, a decode-work/length latency formula, MiniLM/KNN quality
and length estimates, and batched multi-objective routing. Our separate
RouteBalance adaptation already covers this learned-latency approach. An
extra estimator swap would be a RouteBalance ablation, not its default baseline.

The [llm-d article](https://llm-d.ai/blog/predicted-latency-based-scheduling-for-llms)
describes online XGBoost TTFT/TPOT prediction from request and server state,
with latency/SLO-aware selection and cache affinity. It does not document an
MLP implementation. Our MLP quality and output-length arms predict different
targets and cannot be presented as llm-d latency prediction.

[Preble sections 2.1 and 4.2](https://arxiv.org/html/2407.00023v1)
describe profiled prefill/decode time functions and cache/load-aware scheduling.
"Per-layer latency from offline profiles" is too narrow a description of its
scheduler. Neither Preble nor llm-d is reproduced by citing an existing SFS arm.

Recommendation: retain LMDeploy and the explicitly scoped Mooncake adaptation
as distinct routing comparisons, retain `hard_prefill_tps` as an estimator
ablation, and retain RouteBalance. If reducing scope, omit a comparison
explicitly rather than claim another policy reproduces it.

## Existing Ministral measurements

The twelve valid cells from Bridges job 45842572 cover LMDeploy, latency
agnostic and round robin at all four original rates. Each has 8,000 successful
requests, zero failures, and complete joined quality. Their original and
salvaged point hashes were rechecked, and TTFT attainment and OnTimeUtility
were recomputed from all 96,000 derived request rows. See
`bridges-reuse-review-20260916.json` for provenance and exact values.

| QPS | LMDeploy | Latency agnostic | Round robin |
|---|---|---|---|
| 6.0125 | 98.54% / 0.5008 | 73.19% / 0.4926 | 68.16% / 0.3486 |
| 7.8625 | 97.88% / 0.4981 | 72.28% / 0.4905 | 59.54% / 0.3018 |
| 8.7875 | 80.06% / 0.4092 | 72.00% / 0.4895 | 37.40% / 0.1856 |
| 9.7125 | 11.49% / 0.0600 | 72.14% / 0.4904 | 37.05% / 0.1836 |

Entries are system-entry TTFT SLO attainment / OnTimeUtility with saved
Ministral Pro quality and the original cost penalty. The 105% cells are valid
measurements, but LMDeploy shows a large overload collapse. Successful request
completion is not SLO attainment or proof of queue stability.

The four old Mooncake cells had 1, 7, 19 and 43 stale-snapshot failures and are
excluded from reuse. Recovery job 46077428 was still PENDING during this review;
no newly completed recovery results were found. SFS, shortest queue, SCORE,
RouteBalance and vLLM-SR are not supplied by these twelve cells.

Keeping the SLOs requires no rerun. Both deployments use three H100 80GB GPUs
with Ministral TP 1/1/1; the cloud pool reserves a fourth GPU but leaves it
unused for Ministral. Nevertheless host conditions and calibration may differ.
Preserve the Bridges results, prefer completing that comparison there, or use
limited overlapping policy/load measurements to assess a combined comparison
before pooling Vast and Bridges results. The Qwen controls do not establish
Ministral cross-host equivalence. No automatic reruns or job cancellations
are authorized by this review.

## Does latency-agnostic performance justify different SLOs?

Recomputing attainment on the saved latency-agnostic request rows gives:

| QPS | Original SLO | 2x SLO | 5x SLO | 10x SLO |
|---|---:|---:|---:|---:|
| 6.0125 | 73.188% | 73.275% | 73.500% | 73.638% |
| 7.8625 | 72.275% | 72.338% | 72.375% | 72.487% |
| 8.7875 | 72.000% | 72.088% | 72.188% | 72.225% |
| 9.7125 | 72.138% | 72.162% | 72.188% | 72.200% |

These are retrospective threshold sensitivities on identical measured rows,
not fresh runs under modified SLOs. GovReport has 2,000 requests at each load;
1,908 are routed to 3B, 91 to 8B, and one to 14B. Its median TTFT grows from
151.65 seconds to 618.77 seconds, against a median SLO of 455.99 milliseconds.
Alpaca and WritingPrompts median TTFTs remain approximately 14-17 milliseconds
with roughly 159-160 millisecond median SLOs. GovReport attainment is already
11.3% at the lowest load and 7.45% at the highest. The bucket mix is fixed.

This is consistent with a policy that overloads one model while most requests
on other models continue meeting their deadlines. The 65% label references
the historical shortest-queue capacity scout, not latency-agnostic capacity.
Binary attainment therefore conceals substantial worsening within the
already-failing group. Keep the current SLOs for the committed experiment;
report latency quantiles or bucket breakdowns alongside attainment. If needed,
a lower-load calibration probe can locate latency-agnostic saturation without
changing the committed grid. Revisit SLOs only for an independently motivated
latency target or calibration finding. SFS uses SLOs for decisions, so a changed
SLO experiment cannot in general be obtained by rescoring its old routing.

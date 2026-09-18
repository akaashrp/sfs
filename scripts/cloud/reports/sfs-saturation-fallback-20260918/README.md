# SFS at saturation: the infeasible-candidate fallback equalizes lateness

Measured on the corrected Ministral pool `sfs-score-ministral-20260918` (repo-v7, `9b8d040`, batch-time
residuals 1.02 / 0.99 / 1.00), so nothing here is contaminated by the feature-set defect documented in
`../ministral-sfs-gap-20260918`. That defect explained 6.0125 and 7.8625 QPS, where SFS now leads every
baseline. It does not explain 8.7875, which is the subject of this note.

## What was measured

Pro OnTimeUtility and TTFT SLO attainment at 8.7875 QPS (0.95 x K_M), 8,000 requests:

| policy | OnTimeUtility | attainment |
|---|---:|---:|
| shortest_queue | 0.3711 | 73.15% |
| routebalance | 0.1836 | 30.49% |
| **hard (SFS)** | **0.0629** | **11.01%** |
| mooncake_prefill | 0.0597 | 10.72% |
| vllm_sr_latency | 0.0407 | 8.31% |

Before the feature-set fix SFS scored 0.0632 / 12.88% on this cell, so the corrected batch clock changed
nothing here. The estimator is now accurate; the behaviour is the policy's own.

## Where the time goes

Per-request decomposition of the SFS cell, by arrival quarter (p50):

| quarter | system-entry TTFT | engine TTFT | engine queue | entry to dispatch |
|---|---:|---:|---:|---:|
| 1 | 2.08 s | 2.08 s | 1.62 s | 0.01 s |
| 2 | 15.58 s | 15.57 s | 15.20 s | 0.01 s |
| 3 | 23.48 s | 23.46 s | 23.13 s | 0.01 s |
| 4 | 26.59 s | 26.58 s | 26.21 s | 0.01 s |

The router is not the bottleneck: dispatch takes 10 ms throughout, total simulation latency per decision
is 0.1 ms rising to 6.0 ms, and the router sustained 8.77 decisions per second against an offered 8.7875.
All of the delay is engine queueing.

## The mechanism

Per-engine outcome, SFS against shortest queue on the same offered load:

| p50 TTFT | 3B | 8B | 14B |
|---|---:|---:|---:|
| SFS | 20.72 s (6.5% met, 2840 routed) | 20.80 s (8.4%, 2421) | 17.43 s (17.9%, 2739) |
| shortest_queue | 0.04 s (97.6% met, 3782 routed) | 0.07 s (73.6%, 2669) | 15.88 s (12.7%, 1549) |

SFS equalizes *wait time* across the three engines; shortest queue equalizes *queue length*. For
heterogeneous engines those are very different: a fast engine drains its queue quickly, so equal counts
leave it with a short time-queue, while equal waits drag it down to the slow engine's latency.

`sfs_core/routing/score_proxy.py:77-83` is the rule. With at least one feasible candidate, SFS maximizes
quality minus lambda times cost over the feasible set. With none, it returns `-wait_ms`: argmin predicted
wait. That is a reasonable tie-break under light load and actively harmful at saturation, because a missed
TTFT SLO scores zero whatever its quality, so spreading lateness evenly converts "two thirds on time" into
"nothing on time". SFS's own numbers show it: 44% attainment in the first quarter, then 0.0% in every
quarter after, against shortest queue's steady 84 / 69 / 70 / 70%.

## Scope

Ministral 8.7875 is the only genuinely saturated cell in the campaign. Qwen at 8.3 QPS attains 93.6% and
finds no feasible candidate on 0.29% of decisions, so it never enters this branch; the Ministral cell hits
it on essentially every decision once the pool backs up.

## Options (user decision pending, 18 September 2026)

1. Report as a limitation of the method at saturation. No code, no reruns.
2. Change the fallback to concentrate damage: when no candidate can meet the SLO the request is already
   lost, so route it to the engine that is already most backed up, keeping the other queues short so later
   arrivals stay feasible. Small change in `hard_slo_candidate_value`, but it is part of SFS's definition
   and every SFS cell that ever reaches the branch must be rerun (on Qwen that is 0.29% of decisions, so
   the effect should be nil — which still has to be measured, not assumed).
3. Explicit overload admission control (shed or defer). Most defensible in principle, largest change, and
   it makes SFS a different class of system than the baselines it is compared against.

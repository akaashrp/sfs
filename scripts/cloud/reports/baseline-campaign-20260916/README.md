# Missing baseline campaign, 16 September 2026

The active cloud manifest is `scripts/cloud/baseline-campaign-20260916.json`.
The older 68-cell `campaign.json` and artifact bundle remain frozen for provenance;
they are not the active launch list. The worker applies and validates the explicit
baseline overlay without changing any prompt, predictor, tokenizer, price, SLO,
model revision or serving setting in that bundle.

| Pool | GPUs | Requests/cell | QPS | Missing policies |
|---|---|---:|---|---|
| Qwen | 0-3 | 16000 | 6, 7, 8, 8.3 | LMDeploy, vLLM-SR, Mooncake, RouteBalance |
| Ministral | 4-6 | 8000 | 6.0125, 7.8625, 8.7875 | shortest queue, vLLM-SR, Mooncake, RouteBalance |
| Debug | 7 | bounded probes only | — | long-prefill snapshot smoke |

This is 28 new cells / 352000 evaluation requests. SCORE and SFS remain paused.
Ministral 105% (9.7125 QPS) is excluded. Preserve existing 105% results.
Reuse Ministral LMDeploy, round robin and latency agnostic from audited Bridges
cells. Historical Qwen references already contain round robin, latency agnostic
and shortest queue on the new grid; do not automatically rerun them.
Comparisons that combine hosts must retain host provenance; prior SFS controls
show that host/runtime comparisons cannot simply be assumed equivalent.

Seven additional `hard_prefill_tps` reruns are recorded in `fallback_cells`, using
the same family grids and budgets. They are deliberately absent from executable
primary cells. Submit them only when their model pool has no higher-priority
runnable work and would otherwise be idle. GPU7 remains reserved for debugging.

## Telemetry repair

Both families now use the same latest-coherent-publication scheduler. Snapshot
age is logged, but is not a dispatch eligibility test. The Ministral-only retry
loop and ten-second freshness deadline are removed. The historical age keyword
is accepted for call compatibility and has no routing effect.

Keep snapshot transport seqlock/schema/header checks, monotone versions, the
request-lifetime ledger, completed-ID tombstones, unobserved local reservations,
and per-assignment local updates. An old batch retains its work; elapsed time is
never treated as completion. LMDeploy's request-lifetime `N/speed` rule and the
SFS publisher/native simulator/output target expression are unchanged.

Mooncake remains the cache-disabled, coupled-server prefill placement adaptation:
fitted unfinished prefill work plus incoming prefill work, deduplicating observed
and locally dispatched requests. This follows the placement structure of
[Mooncake section 6.1](https://arxiv.org/html/2407.00079v4#S6.SS1).

RouteBalance retains MiniLM/KNN estimates, the calibrated TPOT head, normalized
quality/cost/latency selection, bounded batches and LPT order. A repeated busy
publication cannot establish *new* free-slot capacity: use the queued-latency
branch until there is a new publication (or an exclusively idle server), while
still applying sequence, KV, prefill and local-dispatch checks. This conservative
capacity proxy is an explicit adaptation of the free-slot branch; it neither
waits for another batch nor rejects the request. Telemetry remains once per
routing batch with local state updates, consistent with
[RouteBalance section 4](https://arxiv.org/html/2606.17949v1#S4.SS2).

Infrastructure monitoring is separate from policy selection: the worker checks
owned serving processes, retains partial artifacts on failures, and a read-only
30-second monitor records GPU use, snapshot versions/age, batch-log progress,
controller heartbeat, active cells and audits. An unchanged snapshot by itself
is not declared a dead GPU. No Slurm jobs are cancelled.

## Execution gates

Supervisor owns `sfs-baselines-qwen` and `sfs-baselines-ministral`. They verify
source-bound CPU gates, fit destination service/prefill/TPOT calibration from
calibration prompts, and run matching policy smokes. Ministral additionally
runs 512-request Mooncake/RouteBalance stress at 8.7875 QPS. Evaluation is held
until the controller reviews the recorded calibration and smoke evidence and
writes a hash-bound release. The pool remains loaded during this review.
Every measured cell must pass request identity/count, errors, end-to-end TTFT,
arrival-rate and source-hash checks before entering the completion ledger.

The debug probe uses one Ministral 8B on GPU7 and eight long calibration prompts
per policy, with 32 output tokens and a 180-second request-stage timeout. Its
single-model result is a plumbing smoke, not a model-selection or quality result.

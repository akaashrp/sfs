# Ministral methodology baseline audit — updated 2026-09-07

The three routing policies, native RouteBalance predictor builder, service-fit
builder, snapshot adapter, CLI/sweep integration, and reusable smoke launcher
are implemented and CPU-tested. Baseline-specific timing fitting, GPU smoke,
capacity scouting, and the reduced final sweep remain pending. The corrected
service run and holdout generation/judging have completed. The combined
interactive stage `45367216` failed because the instance loader discarded
serving-profile metadata. This is fixed and covered by the production-renderer
CPU regression. Job `45365762` remains held; no GPU work was released during
the September 7 preparation. The expanded paper-job suite has 209 passing CPU
tests. Final sweep jobs await stage results. See [the current four-track job
bundle guide](PAPER_ABLATIONS.md) and [implementation guide](MINISTRAL3_BASELINES.md).

Changes are currently uncommitted on `experiments/model-family-port`, based
on SFS `133d100` and nested vLLM
`e5233a4abbad78c793b694ffe643a7675633b22c`. The nested change is Python-only;
the snapshot's optional per-request scheduled-token map does not change the
native parser ABI. Existing publishers must restart to produce this map.

## Decision and source contracts

Use named routing adaptations on the common three-model Ministral deployment.
LMDeploy can retain its selector and request lifecycle exactly; Mooncake
remains a scoped prefill placement method; RouteBalance retains its predictor
architecture, scoring, batching, and local work updates. These runs do not
reproduce the original systems' complete deployment/performance claims.

| Policy ID | Figure label | Preserved behavior | Deployment/reconstruction choices |
| --- | --- | --- | --- |
| `lmdeploy_proxy` | LMDeploy proxy adaptation | `argmin(N/speed)`, random exact ties, forwarding-to-completion counts | Broaden same-model eligibility to three model candidates; fixed fresh per-model rates |
| `mooncake_prefill` | Mooncake prefill adaptation | Sum queued fitted remaining prefill time plus incoming prefill time | Coupled servers; no cache reuse, transfer, disaggregation, or rejection |
| `routebalance` | RouteBalance adaptation | MiniLM/KNN predictions, XGBoost TPOT, normalized score, batching/LPT, immediate work updates | Paper-based reconstruction of unspecified feature schema, trigger, and admission test |

[Mooncake v4 §6.1/Algorithm 1](https://arxiv.org/html/2407.00079v4#S6.SS1)
also selects decode instances, manages cache transfers, and rejects requests
exceeding TTFT/TBT SLOs. Full fidelity would require a materially different
deployment and experiment, with disaggregation/replication and rejection
accounting. The current cache-disabled, coupled-server comparison deliberately
retains its prefill placement objective. Decode-heavy saturation is a possible
limitation of that objective, not a reason to silently add an SFS gate.

[LMDeploy pinned source, commit a76d91d](https://github.com/InternLM/lmdeploy/blob/a76d91dbc7f812c340c4a3a7b2a60afe80ebfc6d/lmdeploy/serve/proxy/proxy.py#L288-L303)
shuffles candidates and minimizes `unfinished / speed`.
Its [counter lifecycle](https://github.com/InternLM/lmdeploy/blob/a76d91dbc7f812c340c4a3a7b2a60afe80ebfc6d/lmdeploy/serve/proxy/proxy.py#L419-L447)
spans forwarding through response completion. All idle candidates tie even
when speeds differ. Neither `(N+1)/speed`, sampled engine queue counts, nor
SFS's unobserved-dispatch ledger implements this lifecycle.

[RouteBalance v1 §4](https://arxiv.org/html/2606.17949v1#S4) specifies

`S = wq*Q + wc*(1-C/max_j C_j) + wl*(1-T/max_j T_j)`

with nonnegative weights summing to one, raw quality `Q in [0,1]`,
per-request candidate maxima, and input-plus-output cost. It batches requests,
sorts by descending `max_model(predicted_output_length)`, and updates local
work after each assignment. Its latency is `T = TPOT*(d/b + L)`, dropping
`d/b` if a decode slot is free. The paper uses MiniLM with distance-weighted
KNN and learned per-tier TPOT heads. Its
[linked official repository](https://github.com/AKafakA/route-balance) returned
404 during this audit. The complete trigger, TPOT feature schema, and free-slot
test are not specified, so our implementation records these choices.

## Implemented accounting and estimators

`RequestLifetimeLedger` increments atomically before network submission and
releases once on success, failure, or cancellation, including cancellation
before a submission coroutine starts. Observation in vLLM never releases a
proxy count. Router IDs are reconciled to actual `chatcmpl-*`/`cmpl-*-0`
engine IDs; completed IDs prevent stale observations from resurrecting work.

The baseline snapshot reader uses the shared-memory seqlock and Python
MessagePack path without running SFS simulation. It subtracts scheduled
per-request work from planned computed counts, retaining executing prefill in
the work estimate. Decode batch size counts actual decode sequences.
SFS's predicted-output target/reserve rule is ignored. The common backend
still runs its existing output predictor for all policies to support the
shared publisher; this is common serving overhead, not baseline routing input.

Mooncake sums nonnegative per-request remaining prefill estimates; it does not
fit aggregate queued tokens or count decode-only requests as pending prefill.
Offline singleton prefill fitting uses
`intercept + a*p + b*(p*p + 2*p*processed_context)`, summed over remaining
chunks. The tied context coefficient is an explicit causal-quadratic work
assumption. Full-prefill probes can identify the fit; lack of partial-prefill
observations is reported as unvalidated extrapolation. Pure singleton rows,
chronological validation, prompt-length coverage, and fit residuals are
required. Queued TTFT and mixed-iteration scalar prefill TPS are not isolated
prefill execution time. Add a small length-stratified calibration probe only
if the corrected service traces lack usable coverage.

RouteBalance uses pinned `sentence-transformers/all-MiniLM-L6-v2` revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, CPU masked mean pooling/L2
normalization, and FAISS KNN with `k=10`. Inverse Euclidean distance,
exact-neighbor averaging, and 256-wordpiece right truncation are explicit
reconstruction choices. The existing 30,000 model scores provide labels;
there is no SFS predictor substitution or evaluation lookup. Saved SFS
validation IDs define the held-out calibration split, with duplicate prompt
text excluded from training. Inference remains on the timed request path.

The TPOT head is XGBoost with scheduled decode tokens, scheduled prefill
tokens, and aggregate context tokens as inputs, and iteration execution
milliseconds as target. Mixed iterations are included. Native JSON heads are
checksummed and warmed before timed arrivals. There is no static fallback;
`1/decode_tps` is not TPOT. The latency formula retains the paper's
decode-based form, with no implicit SFS wait/prefill correction. Audit
residuals by prompt length before final evaluation.

The collector flushes immediately if any candidate has zero router-owned
outstanding requests; otherwise it collects at most 16 requests for at most
25 ms from the oldest waiting request, and flushes at end of arrivals.
Idle status changes collection timing, not the scoring rule. Predictions
batch across prompts, ties in LPT preserve input order, and each assignment
updates outstanding work before the next assignment. One TPOT prediction is
cached per model per scheduling batch.

The free-slot branch uses a named conservative capacity proxy: fresh
telemetry, no waiting/unobserved local requests or unfinished/inflight prefill,
sequence and token-budget room, context limits, and enough KV blocks for the
incoming predicted context plus conservative resident growth. It supports
the current TP1/single-KV-group configuration. Busy decode-only instances may
qualify. Reasons and all evidence are logged; `num_running < 512` alone
never proves admission.

Event-driven publication can leave an old empty snapshot on a drained
exclusive server. Such an empty snapshot is accepted with no local work,
plus bounded grace during a new dispatch-to-engine handoff. Its original age
is retained, so this exception does not establish a fresh admission slot.
Stale active snapshots and version regressions fail the trial.

Use uniform `(wq,wc,wl)=(1/3,1/3,1/3)` initially. The optional sensitivity
check is excluded. Only a material calibration finding can justify a
predeclared alternative; never exceed two calibration weight settings, and
freeze one setting before evaluation. Weight selection must use saved
validation IDs disjoint from predictor training, never the 2500+ holdouts.

## Measurement and validation boundary

Decision logs retain collection wait, batch size/position, prediction and
telemetry time, selection time, snapshot age, candidate terms, tie candidates,
seed, before/after counts, engine IDs, and artifact provenance. Original
arrival/system-entry timestamps survive batching and dispatch reordering.
Costs use input-plus-output tokens; existing price-weighted values are USD
times one million, which cancels in the candidate cost ratio.

Figure 5 OnTimeUtility continues to use actual joined quality and
`system_entry_e2e_ttft_slo_met`, charging ingress/router/collection delay.
The composed TTFT combines router delay and engine timing; it omits
dispatch-to-engine admission delay and is not client-measured TTFT. Historical
semantics are preserved. A supplementary frontend-composed TTFT or common
streaming measurement is separate follow-up work.

CPU coverage includes selector equations, exact ties, work reconciliation,
inflight progress, admission evidence, async batch flush/cancellation,
failure cleanup, native FAISS/XGBoost fit/load, calibration split leakage,
CLI and sweep serialization, native snapshot compatibility, and actual-quality
collation with absent LMDeploy/Mooncake predicted quality.

Real corrected-profile service fits and GPU execution remain pending.
Artifact checksum/profile equality validates integrity and declared launch
fields; it does not prove measured fit quality or complete-stack compatibility.
GPU type, source/model pins, and runtime provenance must also be checked.
The smoke launcher verifies the active editable vLLM path and records source
provenance; it does not install or rebuild native code.

## Remaining experiment plan

Reuse corrected service calibration and holdout preparation/judging. In one
additional three-H100 allocation, load the pool once, top up only missing
calibration coverage, smoke the baselines, then scout shortest-queue capacity
on calibration traffic, draining between trials. The smoke helper can reuse an
already loaded pool. Its fixed 2 QPS is a functional check, not capacity.

Scout with an adaptive absolute-rate bracket (initial candidates 2, 4, 8,
16 QPS), extending/refining as needed. Use throughput during arrivals and
sustained backlog growth after warm-up; record arrival shortfall, ingress,
router outstanding work, engine backlog, and drain separately. Mark
inconclusive trials rather than treating arrival-and-drain throughput as
stability. Persist the stable/unstable bracket, uncertainty, and the reference
`K_M` selection rule.

Neither inherited Qwen 7.30 QPS nor a sum of fresh standalone service rates
is measured Ministral router capacity. The existing helper's sum remains a
historical statistic, not the revised sweep's input. Four shared rates near
`0.65, 0.85, 0.95, 1.05 * K_M` remain proposals until the scout verifies the
intended light-load through mild-overload range.

The final manifest tooling defines eight policies (SFS, SCORE, these
three, shortest queue, latency agnostic, round robin), four measured rates,
16,000 requests per cell, seeds, holdout/request-map hashes, and source pins.
Both launcher and collation now have a measured-manifest mode and reject
missing/duplicate cells. A production manifest still awaits passing scout and
timing/arrival review evidence. Preserve raw results and augment derived copies only.
Figures 6 and 7 are outside this ablation.

On September 6, service `45088929`, holdout generation `45089100`, and judging
`45089102` were confirmed complete. Figures 2/3 (`45091510`, `45090868`) failed
on startup paths now fixed and CPU-tested; Figure 3 was successfully repaired
from existing traces without another GPU allocation. Figure 2 awaits a rerun.
The obsolete Figure 5 sweeps `45091769`/`45091770` and collation `45197570`
were held before execution. Their defaults and all raw artifacts are preserved.

The completed service traces have only 5/1/1 pure singleton prefill rows for
3B/8B/14B, so 64 isolated calibration probes per model are necessary. Job
`45367216` combines these probes, fitted timing heads, all eight smoke policies,
and the scout on one loaded pool. The controller requires an observed bracket
at most 5% wide and confirmation of the proposed light/overload endpoints.
It records `K_M` as the smallest observed unstable endpoint, not an exact
universal capacity. No final QPS values or baseline GPU results exist yet.
No other worktrees were changed.

The original stage job `45333598` failed before server readiness because the
workspace scratch path produced IPC socket names longer than 107 bytes. The
launcher now uses a short job-local `VLLM_RPC_BASE_PATH` and tests binding
before staging models. A real CPU ZeroMQ bind passed before the retry was
submitted; no baseline GPU measurements were produced by the failed attempt.

At the user's request, the same three-H100, 32-CPU, four-hour stage is now queued
through `salloc`/`srun` with the site helper's `Interact` job name. Slurm confirms
`gpuinteract` for job `45367216`; the normal-QoS retry is held to prevent duplicate
runs. Two preliminary interactive requests were cancelled/failed before any GPU
allocation. The CPU artifact/import preflight passed again; interactive queue
acceptance does not establish server readiness or produce baseline results.

# Cloud campaign status — 16 September 2026

This is a live operational handoff, captured after inspecting the Vast state,
the completion ledger, the result collator, the local Git branches, and the
Bridges queue. It distinguishes audited cells from raw but invalid artifacts.
Refresh it after any restart or new completion.

## Current operational state

- No evaluation pool is currently running on Vast. The Qwen and Ministral
  campaign workers both stopped at their arrival-rate audit; all eight GPUs are
  idle. The read-only monitor and collator remain running.
- The primary cloud campaign is a 28-cell overlay:
  - Qwen: LMDeploy, vLLM-SR, Mooncake, and RouteBalance at 6, 7, 8, and 8.3
    QPS, with 16,000 requests per cell.
  - Ministral: shortest queue, vLLM-SR, Mooncake, and RouteBalance at 6.0125,
    7.8625, and 8.7875 QPS, with 8,000 requests per cell.
  - SFS and SCORE are deliberately outside this launch list while the
    remaining-output-length investigation is unresolved.
- Mooncake and RouteBalance use the repaired telemetry behavior: snapshot age
  is logged but no longer causes a one-second rejection; Ministral no longer
  waits up to ten seconds for fresh telemetry. Both methods passed their GPU
  smoke/stress gates, including a busy-snapshot probe older than one second.
  Their full cells have not yet begun.
- The active Vast checkout is `da1c78b`; the local/cloud preparation branch is
  pushed through `7380c13`. The FCFS preparation worktree is pushed through
  `7764e49`.

## Valid newly completed cloud cells

Every row below has the required request count, zero request failures, a
passing cell audit, and a completion-ledger entry. `Arrival QPS` is recomputed
from the recorded system-entry offsets; `OnTimeUtility` is the Pro-scored,
actual SLO-gated utility in the collated result.

| Family | Policy | Target QPS | Arrival QPS | Requests | TTFT SLO attainment | OnTimeUtility |
|---|---|---:|---:|---:|---:|---:|
| Qwen | LMDeploy | 6 | 5.965 | 16,000 | 97.80% | 0.3819 |
| Qwen | LMDeploy | 7 | 6.946 | 16,000 | 97.19% | 0.3895 |
| Qwen | LMDeploy | 8 | 7.937 | 16,000 | 91.04% | 0.3897 |
| Qwen | LMDeploy | 8.3 | 8.233 | 16,000 | 95.94% | 0.3961 |
| Qwen | vLLM-SR | 6 | 5.524 | 16,000 | 38.68% | 0.1107 |
| Ministral | shortest queue | 6.0125 | 5.963 | 8,000 | 97.74% | 0.4994 |
| Ministral | shortest queue | 7.8625 | 7.785 | 8,000 | 92.94% | 0.4795 |
| Ministral | shortest queue | 8.7875 | 8.675 | 8,000 | 73.31% | 0.3711 |
| Ministral | vLLM-SR | 6.0125 | 5.700 | 8,000 | 25.25% | 0.1244 |

The raw points and per-cell audits live under
`/workspace/sfs/state/baselines/{qwen,ministral}-20260916/`; the derived
summary lives under `/workspace/sfs/state/baseline-results/`. The distinction
between arrival rate and achieved service throughput must be retained when
interpreting these measurements.

## Preserved but non-reportable partial cells

Both cells completed all requests, but the audit correctly rejected them
because the recorded offered arrival rate was more than 10% below the requested
rate. They have no completion-ledger entry and must not enter a figure or a
cross-policy comparison.

| Cell | Requests completed | Requested QPS | Recorded arrival QPS | Status |
|---|---:|---:|---:|---|
| Qwen vLLM-SR, 7 QPS | 16,000 | 7.000 | 5.648 | invalid: arrival-rate audit failed |
| Ministral vLLM-SR, 7.8625 QPS | 8,000 | 7.8625 | 6.260 | invalid: arrival-rate audit failed |

The immediate stopping condition is known. The cause of the under-delivery in
the arrival process has not yet been diagnosed, so the 10% gate should not be
weakened merely to resume the campaign.

## Remaining primary cloud work

There are 19 actions remaining before the 28-cell primary overlay is complete:
17 cells never began and the two invalid cells above require fresh reruns after
the arrival issue is repaired and revalidated.

| Family | Rerun after repair | Never started |
|---|---|---|
| Qwen | vLLM-SR at 7 | vLLM-SR at 8 and 8.3; Mooncake at 6/7/8/8.3; RouteBalance at 6/7/8/8.3 |
| Ministral | vLLM-SR at 7.8625 | vLLM-SR at 8.7875; Mooncake at 6.0125/7.8625/8.7875; RouteBalance at 6.0125/7.8625/8.7875 |

The dispatcher is resume/skip-completed aware. It should retain the nine valid
cells above, preserve the two rejected artifact directories, and launch only
the repaired reruns and the never-started cells.

## Reuse rather than duplicate work

- Qwen already has audited historical round-robin, latency-agnostic, and
  shortest-queue references at 6, 7, 8, and 8.3 QPS, each with 16,000 requests.
  These are reference artifacts, not cloud jobs to repeat automatically.
- Bridges has 12 valid Ministral cells for LMDeploy, round robin, and
  latency-agnostic at 6.0125, 7.8625, 8.7875, and 9.7125 QPS, all with 8,000
  successful requests. The first three rates can be reused for the current
  grid; the valid 9.7125 QPS result remains archived and is not part of the
  newly committed cloud grid.
- Pending Bridges jobs remain untouched: `45842578` (`sfs-qwen_baselines`) and
  `46077428` (`sfs-ministral-recovery`) are both still `PENDING` due to
  priority. Do not cancel them automatically.

## Deferred and held work

- Seven `hard_prefill_tps` fallback cells are prepared but explicitly held:
  Qwen at 6/7/8/8.3 and Ministral at 6.0125/7.8625/8.7875. They are lower
  priority than the primary baseline completion and FCFS preparation.
- Flash-quality, MLP-quality, and MLP-output-length variants remain deferred
  until the canonical baseline comparison is settled.
- The conditional remaining-output-length lookup is an investigation only.
  No prediction change has been deployed to SFS or SCORE, and no SFS/SCORE
  measurement should be silently substituted with it.

## FCFS unchunked serving configuration preparation

The isolated branch `experiments/qwen-fcfs-unchunked-20260916` prepares the
second Qwen serving configuration requested for later evaluation:

- FCFS; chunked prefill and prefix caching disabled; 65,536 model/batch-token
  limits; 512 sequences; threshold 0; 0.90 GPU memory utilization.
- The actual OpenAI chat preprocessing audit passed for all three Qwen models,
  evaluation/calibration/warm-up inputs, and the preserved 8,192-token output
  allowance. The largest formatted prompt is 32,801 tokens, leaving 24,543
  tokens of context headroom under the 65,536-token limit.
- Bounded GPU startup and loaded no-chunking smoke passed for Qwen 0.6B and 8B
  on GPU 7. The Qwen 32B TP=2 smoke, all-model configuration-specific
  calibration/refitting, scheduler-simulator validation, and loaded
  predicted-versus-observed timing checks remain to be done.
- The 36-cell FCFS matrix is prepared conceptually but has not been launched;
  it remains blocked pending those qualification artifacts and a later explicit
  full-matrix launch decision. The second additional serving configuration is
  intentionally unspecified.

## Recommended next sequence

1. Diagnose the arrival-rate under-delivery using the two retained raw vLLM-SR
   points. Repair the producer or measurement only with evidence, then run a
   short bounded arrival validation at the affected targets.
2. Resume the primary overlay with strict skip-completed behavior. Re-run only
   the two rejected vLLM-SR points and execute the 17 cells that never started.
   Continue to retain raw result and host provenance.
3. Once the base campaign is stable, complete FCFS qualification: Qwen 32B
   TP=2 smoke, independent calibration traces, batch-model refits, simulator
   consistency, and representative loaded wait/TTFT validation. Do not launch
   the FCFS full matrix as part of preparation.
4. Revisit the held `hard_prefill_tps` fallbacks only when the higher-priority
   primary and FCFS work no longer needs the pool.
5. Periodically check the two pending Bridges jobs and reconcile any finished
   outputs against the ledger without cancelling or duplicating valid work.

## Key artifacts

- Active primary overlay: `scripts/cloud/baseline-campaign-20260916.json`
- Prior methodology/reuse review:
  `scripts/cloud/baseline-and-reuse-review-20260916.md`
- Cloud execution plan and ledger:
  `scripts/cloud/execution-plan-20260916.json` and
  `scripts/cloud/experiment-ledger-20260916.json`
- Mooncake/RouteBalance GPU probe and timing reviews:
  `scripts/cloud/reports/baseline-campaign-20260916/gpu7-probe.json`,
  `qwen-timing-review.json`, and `ministral-timing-review.json`
- FCFS implementation branch:
  `/ocean/projects/cis250162p/aparthas/sfs_fcfs_20260916`

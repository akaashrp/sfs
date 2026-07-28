# V100 latency-aware rebuttal baselines

`hard_score_proxy` is a SCORE-inspired TTFT baseline, not an exact
reproduction of SCORE. For candidate instance `j`, it estimates

```text
TTFT_j =
  (effective_prefill_backlog_j + new_prompt_tokens) / prefill_tps_j
  + effective_decode_backlog_j / decode_tps_j
  + mean_decode_batch_latency_j
```

The first two terms are converted to milliseconds. Effective backlogs come
from the same scheduler snapshot used by SFS and include router dispatches
that are not yet visible in that snapshot. Pending decode work uses the
output-length predictor and scheduler reserve; it never uses realized or
oracle output length. The estimator uses the lightweight snapshot-summary
path and never invokes the SFS simulator.

All hard baselines share exactly the same feasibility and value rule:

1. Keep candidates whose estimated TTFT meets the request TTFT SLO.
2. Among feasible candidates, maximize predicted quality minus
   `lambda * predicted_cost`.
3. If none is feasible, choose minimum estimated TTFT.

Thus `hard`, `hard_prefill_tps`, and `hard_score_proxy` differ only in their
latency estimate. They use the same prompts, arrivals, SLOs, quality
predictor, output-length predictor, cost model, and random seed.

Each `hard_score_proxy` request stores a compact `score_candidate_terms`
record for every candidate. It includes the effective prefill/decode
backlogs, the three latency terms, predicted TTFT, SLO feasibility,
quality/cost inputs, the hard candidate value, and the selected instance.
Calibration constants stay in the run-level calibration JSON instead of
being duplicated in every request.

## Required calibration

The calibration job records, per model and under the exact V100 serving
configuration:

- prefill TPS from matched per-request prompt tokens and prefill time;
- decode TPS from pure-decode batch-stat rows;
- mean execution latency of pure-decode batches;
- SFS batch-time regression coefficients.

Mixed prefill/decode rows are excluded from decode calibration. Missing,
non-finite, mismatched-feature-set, or incomplete metrics fail the sweep
closed. Warm-up and prior-file rows are excluded consistently from the
prefill, decode, and SFS-regression fits.

The calibration also measures successful requests per second for each model
under a standalone saturated stream. The automatic QPS scale is the sum of
those three rates and is written to `capacity_estimate.json`. This is a
workload-specific normalization, not a claim of exact concurrent cluster
capacity: it does not measure CPU/interconnect contention when all three
servers are busy simultaneously.

## V100 configuration

- one exclusive eight-GPU `GPU` partition node;
- Qwen3-0.6B: TP1 on one V100-32;
- Qwen3-8B: TP1 on one V100-32;
- Qwen3-32B: TP4 on four V100-32 GPUs;
- two GPUs reserved but unused;
- FP16, XFormers, V1, prefix caching off, chunked prefills off;
- `cross_term` batch-time regression and simulation features;
- max model length 40,960, max batched tokens 49,152, max sequences 64.

The V1 change in this worktree preserves an explicit
`--no-enable-chunked-prefill`; upstream behavior in this snapshot otherwise
forces chunking on for generation models.

## Launch

First build this worktree's native scheduler-simulator extension:

```bash
scripts/setup/compile_vllm_scheduler_sim.sh
```

The batch job then imports the staged extension and runs two native
simulation smoke tests, including the `cross_term` path, on the allocated
V100 node before staging models. This catches a host-ISA mismatch before the
expensive part of the run. If that preflight fails, the job rebuilds the
extension on the allocated node, restages it, and reruns the preflight.

Preview without submitting:

```bash
src/slurm/runs/router/submit_v100_rebuttal_qps.sh --dry-run
```

Submit calibration and its `afterok`-dependent sweep:

```bash
src/slurm/runs/router/submit_v100_rebuttal_qps.sh
```

By default the sweep derives QPS points at 40%, 60%, 80%, 100%, and 110% of
the calibrated aggregate service rate. Set `QPS_VALUES` to an explicit
space-separated list to override this.

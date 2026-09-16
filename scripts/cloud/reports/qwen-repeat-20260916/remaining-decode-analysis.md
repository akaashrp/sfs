# Remaining decode underestimation in the completed 8.6 repeat

A read-only join of five coherent scheduler snapshots to the completed request records finds running requests whose snapshot target minus generated tokens is one while their eventual generated output is thousands of tokens longer. At snapshot 1789548919290306853, the 0.6B model has 29 running and 78 waiting requests. Sixteen running requests have a predicted remaining length of one and at least 128 actual tokens remaining. Their contexts total 361,510 of 625,825 running context tokens (57.8%). Request hard-req-12933 has 5,561 actual tokens remaining despite a snapshot remaining target of one. These are hindsight measurements, not online information available to the router.

## Source mechanism

In `vllm/v1/core/sched/scheduler.py`, `_snapshot_output_target` computes (subject to generation/context caps):

```
target = max(ceil(predicted_total) + reserve, generated_so_far + 1)
```

`_compute_adaptive_decode_reserve` currently returns the global reserve and does not use its predicted-target or generated-so-far arguments. Once a long generation exceeds prediction plus reserve, repeated snapshots can continue to place completion just one token away. `_record_decode_tail_residual` updates history on completion, so a currently outstanding long generation has not yet contributed its final residual.

Native `SimRequestState::decode_remaining` subtracts generated tokens from the supplied snapshot target. `ApplyBatchResultNative` frees simulated KV blocks when this remaining count reaches zero. `materialize_projected_output_tokens` can also finish such requests as it projects the in-flight output. The actual engine continues until its real generation stopping condition. The observed one-token value exists before native simulation; this evidence does not indicate a corrupt binary or a failure to count already emitted tokens.

This can make future memory availability and queue delay too optimistic. Its magnitude in the complete workload, and the initial source of the Bridges/Vast difference, remain unproven. No production changes have been made. The simultaneous QPS 8 controls retain the existing behavior.

## Historical difference

Git commit `58bcdc931d4a4bba8ee5c059724b4cd00a55d60a` dated May 2, 2026 changed the minimum target from `generated_so_far + adaptive_reserve` to `generated_so_far + 1`, removed the per-request overshoot contribution to the reserve, and reduced residual quantiles. July's refactoring moved the target logic into a helper. This is a relevant source-history difference for the April comparison; it does not establish the precise deployed source of the archived April job. Recent Bridges and Vast share the current rule.

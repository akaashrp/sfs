# Two concurrent canonical Qwen controls at 8 QPS

Explicitly requested September 16 to compare GPU pools at the same lower load. Each runs 16,000 requests after the existing 192-request smoke, with canonical predictors, SLOs and coefficients unchanged. GPUs 0-3 use CPUs 0-47; GPUs 4-7 use CPUs 48-95. Outputs are separate at `/workspace/sfs/state/controls/qps8-lane-a` and `qps8-lane-b` and mirrored to the controller.

The destination runtime remains commit 09b04788440ed7bc5a278b6e73c00eed414daf77. `ops/cloud/qwen_qps8_control.py` calls its existing `execute` function with QPS 8, bypassing only the historical CLI rate enumeration (8.6, 8.75). It preserves all runtime gates, workload fingerprint checks, serving, smoke, measurement and reporting behavior. CPU preflight passed for all 16,000 canonical request identities and SLOs. Each lane must pass its own smoke before measurement.

The same offered load on both lanes does not guarantee identical routing decisions or generated outputs. No production snapshot or simulator changes are included in this diagnostic. Main baseline campaigns remain paused.

Both lanes completed 16,000 successful requests with zero errors and passed independent controller audits of raw counts, per-request TTFT gates and saved Pro/Flash utilities. Lane A: TTFT attainment 92.1375%, Pro OnTimeUtility 0.4982115415. Lane B: 92.90625%, Pro OnTimeUtility 0.5013760386. All GPUs returned to idle after normal serving cleanup. The lane agreement at 8 QPS does not resolve the 8.6-QPS or Bridges comparison. No reserve or predictor changes were included.

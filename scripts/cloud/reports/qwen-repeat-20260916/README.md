# Canonical Qwen 8.6 repeat, 2026-09-16

Completed all 16,000 requests with zero failures. The independent controller audit recomputed each TTFT gate, both utilities on 15,996 observed queries, and the raw point SHA256; all match the destination report.

| Run | TTFT attainment | Pro OnTimeUtility |
|---|---:|---:|
| Bridges 46116020 | 81.16875% | 0.4543130698492278 |
| Earlier Vast, with concurrent second pool | 49.9625% | 0.27815865541318135 |
| Vast repeat, one active pool | 63.7125% | 0.35614945124710856 |

The repeat used identical source, predictors, SLOs, smoke sequence, GPU IDs 0–3 and CPU affinity 0–47. GPUs 4–7 remained idle. Thus it tests recurrence with one active pool, not an exact repetition of simultaneous two-pool operation.

Arrival-ordered groups of 4,000 requests attained 94.15%, 51.425%, 87.825%, and 21.45%. Backlog spikes and recovery occurred around the middle; substantial degradation recurred in the final quarter. The previous run's monotonic collapse to zero final-quarter attainment was not reproduced exactly. The broader degradation is reproducible without a second active pool, but its cause and any additional two-pool penalty remain unresolved.

The model processes shut down normally after successful reporting; all eight GPUs are idle. Main experiments remain paused. No further run was started and no queued Bridges job was changed.

Raw output on the volume: `/workspace/sfs/state/controls/qps8p6-repeat1`.
Controller backup: `/ocean/projects/cis250162p/aparthas/sfs_cloud_results_20260916/sfs-vast/controls/qps8p6-repeat1`.
Read-only coherent routing snapshots at five points during this run are under `state/diagnostics/qwen-repeat1-snapshots`, mirrored by the controller. They use read-only mmap and the publisher's seqlock contract; the observer does not create, register, modify or unlink shared memory. These captures are diagnostic evidence, not a change to routing behavior.

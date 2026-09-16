# Offline alternatives to the one-token remaining-length floor

## Available evidence

Fifteen coherent full snapshots were saved from the completed Vast 8.6 QPS / 16,000-request repeat: five times for three engines. Original Vast 8.6 and recent Bridges control retain request/batch logs, but no complete snapshot captures were found for either. These snapshots cover 965 running-decode observations, with repeated requests across snapshots, not 965 independent samples. The raw files remain under `/workspace/sfs/state/diagnostics/qwen-repeat1-snapshots` and the controller mirror.

All comparisons hold the original state, coefficients, timing and waiting/prefill targets fixed and replace only running-decode targets. The 128/4096/16384-token probes are hypothetical. Replay errors below compare against using eventual lengths for *all* currently running decode requests; this differs from the earlier table that corrected only selected one-token overruns. Neither replay is measured ground-truth latency.

## Rules compared

- Current target, read directly from the snapshot.
- Rolling floor: `max(original_target, generated + reserve)`, equivalent to `max(predicted + reserve, generated + reserve)` for positive predictions before caps.
- Half/double reserve sensitivity controls; these are not fitted recommendations.
- Exhausted-only floor: keep original target unless only one predicted token remains, then use `generated + reserve`.
- Conditional mean and median total length among calibration outputs for the same model with length greater than tokens already generated. This estimates remaining work given that the request is still alive, without using the current reserve formula.
- Conditional mean/median only for original one-token overruns; other requests retain the current target.

Calibration uses 10,000 outputs per model, 30,000 total, drawn from the original calibration indices 0-2499 in each of four buckets. The preprocessing verifies all indices, errors, completion caps, summary-array agreement and zero query-ID overlap with the 16,000 evaluation queries. No evaluation output is used to construct these length distributions. Eventual evaluation outputs are used solely for scoring and the hindsight replay. Model-only conditioning is a deliberately simple candidate: it does not condition on prompt features, initial prediction, dataset labels, or partial generated text. Conditioning support minima are 151/17/12 calibration outputs for 0.6B/8B/32B; sparse long-tail support remains a limitation.

## Results

Mean absolute remaining-length error (tokens):

| Model | Current | Rolling full reserve | Conditional mean | Conditional median |
|---|---:|---:|---:|---:|
| 0.6B | 2237 | 2059 | 1260 | 1206 |
| 8B | 266 | 370 | 226 | 196 |
| 32B | 328 | 415 | 275 | 263 |

Mean absolute hypothetical wait-estimate difference from the all-running-length hindsight replay (seconds):

| Model | Current | Rolling full reserve | Conditional mean | Conditional median |
|---|---:|---:|---:|---:|
| 0.6B | 17.90 | 9.58 | 4.36 | 7.60 |
| 8B | 0.92 | 9.43 | 3.86 | 1.82 |
| 32B | 2.43 | 14.23 | 3.57 | 1.21 |

Blindly maintaining a full reserve beyond every running request overstates future KV occupancy on the larger models. Conditional remaining-length estimates merit controlled evaluation: the median improves actual token error for all three models, but it does not uniformly improve even the hypothetical wait metric. Mean remaining work and median remaining length optimize different quantities, and neither guarantees accurate simulated queue delay.

Recommendation: pursue conditioning on observed generation progress rather than patching the same fixed reserve everywhere. Keep current production behavior until a candidate has been checked against dispatch-aligned real TTFT and a controlled run; this artifact does not establish a globally optimal rule or expected SLO improvement. A richer conditional predictor or conditional length distribution can incorporate the original prediction and prompt features, but needs separate calibration and runtime-cost validation. The existing snapshots omit the engine's raw predicted mean, and router-reported means can differ because engine admission predicts separately; using the stored target avoids assuming they match.

## Baseline scope

SFS simulator-based variants consume these targets. SCORE and the SCORE-proxy estimator also consume `decode_backlog_total_tokens` derived from the same targets, so they are not exempt. Round robin, latency agnostic, shortest queue, LMDeploy, Mooncake, RouteBalance and vLLM-SR do not use this remaining-decode estimate to select instances. `hard_prefill_tps` uses prefill backlog only. Actual engine stop/free decisions still follow emitted tokens and real stopping conditions. Snapshot prediction disagreement can force telemetry republication, but this is not premature actual KV reclamation. This finding alone does not invalidate otherwise audited baseline results; unrelated failed runs remain failed.

No serving source, active job, SLO, predictor artifact or runtime coefficient was changed. The read-only analysis runs against saved files through the existing native simulator; it allocates no GPU work.

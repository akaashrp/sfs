# vLLM-SR latency-aware selector adaptation

Upstream: https://github.com/vllm-project/semantic-router
Pin: `544ce4f15eb5d04c1efaa597660c4a8ed042e22c` (Apache-2.0; LICENSE included).
`upstream.json` records SHA-256 for unmodified selector, history and metric
producer sources. Python port: `sfs_core/routing/latency_history.py`.

Frozen settings: P95 TTFT + P90 TPOT, equally weighted relative-to-best ratios,
1,000 latest observations per model/metric, linear interpolation, EWMA alpha
0.3 for fewer than three samples. Exact ties retain candidate order. Missing
metrics exclude candidates; no complete candidate falls back to the first.
No exploration, TTL, SFS simulation, quality prediction or cost term is added.

## Measurement adaptation

The pinned upstream streaming producer updates TTFT on its first body chunk,
measured from ProcessingStartTime, and computes TPOT as total completion time
(from StartTime) divided by actual completion tokens. **It does not subtract
TTFT or divide by tokens minus one.** Its implementation differs from the
usual decode-only interpretation of TPOT.

Our harness uses its system-entry monotonic timestamp for both start origins,
since it has no separate Envoy/ext_proc ingress stage. The first OpenAI SSE
data event (including role-only) supplies TTFT immediately; total stream
duration and actual terminal usage supply TPOT on successful completion. The
client consumes the raw SSE body (`latency_stream.RawChunks`) instead of the
SDK's per-chunk model construction: it JSON-decodes only the first chunk and
chunks that may carry a new response ID/model, a non-null finish_reason or a
usage object, recognizing the remaining compact vLLM content chunks by
substring checks, so per-chunk event-loop load stays small under thousands of
concurrent streams. Parsed SSE chunk delivery can still lag raw HTTP body
arrival; report this as a selector adaptation, not a full-stack reproduction.
Generated text is neither decoded nor retained in memory by this client.
Streaming is enabled only for this policy. Final paper
TTFT/SLO metrics retain the existing server-log joins and end-to-end boundaries;
feedback metric definitions are recorded separately.

Explicit guards: reject nonfinite/nonpositive observations, duplicate updates,
wrong response model/ID, incomplete streams, invalid usage and old-generation
callbacks. A failed stream may already have supplied TTFT, just as upstream
observes first-body latency before knowing the terminal outcome. TPOT is not
updated on failed/incomplete streams; zero-output completions have no TPOT.
One-token successful completions use total duration/1. Never substitute chunk
counts, predicted tokens, offline response outcomes or server-only execution
latency. These guards are documented divergences on invalid inputs.

Before every evaluation cell, a new history uses the same 32 balanced canonical
calibration prompts on each of the three models (96 warm-up completions).
Warm-up completes before evaluation arrivals; actual tokens cap at 8192.
No history is shared between runs. Event traces allow deterministic replay.

## Validation and release

`test_latency_history.py` compiles the unmodified Go selector/cache in an offline
module with logging/type stubs, then compares decisions and scores. Go is a
CPU-test dependency only. Set `SFS_TEST_GO` to the compiler; set TMPDIR and
pytest basetemp to workspace scratch. Activate conda vllm and put the active
nested vLLM plus src on PYTHONPATH. Other tests cover real OpenAI SSE parsing,
request lifecycle, cancellation and failure. A source-bound CPU report is
required before Qwen GPU launch; a separate source-bound 192-request GPU smoke
is required before the four final 16k cells. CPU test success alone never
releases evaluation.

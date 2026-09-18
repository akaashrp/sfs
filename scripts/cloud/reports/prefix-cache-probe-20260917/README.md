# Prefix-cache hit-rate probe (Qwen3-0.6B, frozen evaluation prompts) — STAGED, not yet run

Purpose: measure vLLM's prefix-cache token hit rate on the campaign's frozen 16,000 Qwen evaluation
requests, to decide whether a prefix-caching serving-configuration ablation can show anything.

Staged on Vast as supervisor program `sfs-prefix-probe` (2026-09-17). It waits inside the Python process
until `supervisorctl status sfs-fcfs-run-lane-b` is not RUNNING and GPU 7 reads 0 MiB (re-checked after
20 s), then serves one canonical Qwen3-0.6B engine on GPU 7 (CPUs 84-95) with ONLY
`--no-enable-prefix-caching` -> `--enable-prefix-caching` (plus `--enable-prompt-tokens-details` so usage
reports cached tokens), streams 512 calibration smoke prompts and then the 16,000 frozen requests in
frozen arrival order at concurrency 128 with `max_completion_tokens=1`, and exits leaving GPU 7 empty.
Hard GPU budget: 840 s from server launch (dispatch stops at budget-45 s; skipped requests are counted).

Files (Vast, `/workspace/sfs/state/diagnostics/prefix-cache-probe-20260917/`):
- `prefix_cache_probe.py`, `run.sh`, `sfs-prefix-probe.conf` (copies here)
- `sfs-prefix-probe.log` — supervisor stdout
- `dry-run/` — CPU-only validation (preflight PASS_CPU_CONTROL, attribution.json, server argv)
- `run/status.json` — WAITING_GPU (with poll count / blocker state) -> STARTING_SERVER -> RUNNING(phase) -> ANALYZING -> COMPLETE
- `run/analysis.json`, `run/README.md` — results (mirror here when complete)
- `run/records.jsonl` (per request: prompt_tokens, cached_tokens, bucket, phase), `run/timeline.jsonl`
  (5 s /metrics + GPU-7 memory/PID samples), `run/attribution.json`, `run/server_qwen3-0.6b.log`

Metrics produced: /metrics `vllm:prefix_cache_queries_total` / `vllm:prefix_cache_hits_total` deltas per
phase (warmup, isolation, smoke, eval); per-request `usage.prompt_tokens_details.cached_tokens`; overall
token hit rate, hit rate of cacheable tokens, per-bucket hit rates, first-request-of-bucket vs later;
decomposition into template blocks (16 tokens), dataset-header blocks (govreport/writingprompts 32,
hotpot/alpaca 16) and beyond-header hits (duplicates/content sharing); cold-cache isolation phase; the
server log's periodic "Prefix cache hit rate" lines.

Dry-run attribution (tokenizer level): shared system-prompt/template prefix = 24 tokens (16 block-aligned);
per-bucket common prefixes: govreport 32 (32), writingprompts 34 (32), hotpot_qa 26 (16), alpaca 24 (16);
duplicate prompts within the eval stream: writingprompts 197, govreport 5.

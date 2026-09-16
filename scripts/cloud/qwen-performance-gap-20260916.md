# Qwen Vast versus Bridges: diagnostic audit, 2026-09-16

The two complete Vast controls are preserved, but are not performance-equivalent to the Bridges control. New measured campaign cells are paused. Short diagnostic GPU work is authorized and saved under a separate `state/diagnostics` tree.

## Complete original results

| Run | Requests | Failed | TTFT SLO attainment | Pro OnTimeUtility |
|---|---:|---:|---:|---:|
| Bridges job 46116020, 8.6 QPS | 16000 | 0 | 81.16875% | 0.4543130698492278 |
| Vast GPUs 0–3, 8.6 QPS | 16000 | 0 | 49.9625% | 0.27815865541318135 |
| Vast GPUs 4–7, 8.75 QPS | 16000 | 0 | 43.31875% | 0.24153421719990933 |

TTFT uses all 16,000 requests; both judges' utility uses the common 15,996 observed-score queries, excluding four imputed queries. The experiment ledger records source paths, raw SHA256 and both judge scores. Completion and metric integrity do not establish a fair cross-host comparison.

## Findings supported by the original traces

- Canonical request identities/order, SLOs, arrival schedule, routing predictor artifacts and six batch-model coefficients per model match. For the 9,768 requests routed to the same model in Bridges and Vast at 8.6 QPS, prompt token counts, predicted output lengths and predicted qualities match exactly.
- Serving is TP 1/1/2, BF16, prefix caching off, chunked prefill, 32768 token budget and 512 sequences on both. KV token capacity differs by only 32–48 tokens per model. Both use FlashAttention and the PyTorch greedy sampler.
- Mean router delay is 6.41 ms on Bridges versus 11.23 ms on Vast at 8.6. Mean engine TTFT is 968.6 ms versus 17809.9 ms. The large gap is inside serving queues, not a router/network-delay accounting difference.
- Arrival-ordered quarters at 8.6 have attainment 82.60, 87.175, 90.675, 64.225% on Bridges versus 92.775, 79.40, 27.675, 0% on Vast. Vast initially performs well and subsequently develops backlog.
- In Vast's second quarter, 0.6B median engine TTFT is 686 ms; in the third it is about 50 seconds. Long-prompt traffic then increasingly reaches 32B, and short traffic shifts toward 8B/0.6B. This describes the observed feedback, not its initiating cause.
- Generated lengths differ for 7,064 of the 9,768 same-route requests. Vast averages 23.78 more completion tokens on that matched subset. Greedy settings alone do not establish output identity across different batching. Neither numerical causes nor the contribution of this difference to overload are established.

`reports/qwen-gap-20260916/request-comparison.json` and `shape-and-time-comparison.json` preserve the derived comparisons. The latter is reproducible with `ops/cloud/analyze_qwen_gap.py` against the original output directories.

## GPU topology, CPU placement and short contention test

- Vast reports NV18 connectivity for every GPU pair, including the actual 32B pairs 2–3 and 6–7. Read and write peer-to-peer capability reports OK for all pairs. All GPUs report CPU affinity 0–95 and NUMA 0; the two pool affinity sets are disjoint physical CPU IDs (0–47 and 48–95), with no SMT siblings exposed for those CPUs.
- Original serving logs show the same custom-allreduce P2P initialization path on both hosts; they do not include a detailed NCCL transport trace. The earlier NCCL correctness gates passed for 2–3 and 6–7; these gates alone are not bandwidth measurements.
- The cgroup reports 92.16 CPU cores of quota and zero cumulative quota throttling. This does not exclude ordinary CPU scheduling or memory contention.
- Original decode traces grouped by 8-sequence and 25,000-context-token bins show Vast/Bridges weighted execution-time ratios 1.012/1.010/1.011 for 0.6B/8B/32B. Step-interval ratios are 1.018/1.021/1.018. Bins approximate shapes; these are not exact tensor-shape matches. CPU scheduler times are somewhat higher on Vast, but small relative to each full step.
- A bounded synthetic diagnostic loaded both pools, then ran pool A alone and both pools together. Each active model received 32 simultaneous identical 4096-word prompts with exactly 256 generated tokens (`ignore_eos`). All 288 measured requests completed with the required token count. From coordinator start through cleanup, the diagnostic took about 104 seconds.
- Pool A throughput changed from 3144 to 3173 tokens/s (0.6B), 1169 to 1156 (8B), and 626.85 to 625.92 (32B). In the concurrent phase pool B's 32B rate was 627.27 tokens/s. No large pair asymmetry or immediate two-pool slowdown was observed. This does not test sustained near-capacity routing or long-prompt KV saturation.

Diagnostic source: `ops/cloud/qwen_runtime_probe.py`. Raw evidence: `/workspace/sfs/state/diagnostics/qwen-pool-contention-20260916`, mirrored at `/ocean/projects/cis250162p/aparthas/sfs_cloud_results_20260916/sfs-vast/diagnostics/qwen-pool-contention-20260916`. The initial launcher exited before any GPU work because its upload failed; after upload the diagnostic completed successfully. Both attempts' launcher messages are preserved.

## Confirmed launch-state confound; performance contribution unresolved

Vast's `canonical_control.execute` runs its 192-request SFS smoke on the same model processes subsequently used for measurement. The Bridges `control.py` validates a separately completed smoke and starts fresh serving processes for the measured control.

This matters because `vllm/v1/core/sched/scheduler.py` stores up to 4096 online positive output-length residuals in `_decode_tail_residual_samples`; completion appends samples. At 200 samples the residual quantile starts contributing to `_snapshot_decode_reserve_base_tokens`, changing SFS's simulated remaining decode work. Draining the queue does not clear this history. Vast smoke routed 8/94/90 requests to 0.6B/8B/32B at 8.6, and 8/93/91 at 8.75, before the measured run.

Thus the serving argv and predictor files match, but the initial learned state does not. This is a real confound, not proof that it explains a 31-point attainment gap. Future fresh-control comparisons need smoke in a separate model-process lifetime, followed by the same warmup used by the Bridges control. No historical Qwen code, result or frozen active runtime has been changed during this diagnosis.

A separate bounded diagnostic (`ops/cloud/qwen_state_probe.py`) compared the identical first 1024 canonical requests at 8.6 QPS on fresh engines versus engines that first ran the original smoke. It is deliberately a prefix diagnostic, not a replacement control or a capacity estimate. The corrected attempt completed: fresh engines attained 93.75%, smoked engines 95.21484375%, each with 1024 successes and zero failures. The short prefix did not reproduce the severe full-run degradation, so the launch-state difference cannot currently be called its cause. Total wall time including pool load, smoke, prefix and cleanup was about 420 seconds. All GPUs returned to idle. The original failed attempt is retained: the wrapper read a snapshot before the fresh engine's first warmup. The corrected wrapper lets the canonical router perform its normal warmup before reading telemetry. The smoked lane also emitted HTTP client finalizer warnings about its already-closed smoke event loop; the measured prefix itself completed without failed requests. Results are exploratory and do not establish a steady-state comparison.

## Next decision

Keep all main experiments paused until this comparison is interpretable. Do not automatically rerun either 16k control, change SLOs, cancel pending Bridges work, or relabel the completed results. Ministral SFS remains next among measured runs: 8000 requests at 6.0125, 7.8625 and 8.7875 QPS, unchanged SLOs, after this investigation and a completed-result inventory check.

## Kernel and library audit

The explicitly disabled library path is FlashInfer sampling: both serving launchers set `VLLM_USE_FLASHINFER_SAMPLER=0`. Bridges logs say FlashInfer is installed but disabled; Vast logs say unavailable. Both select the PyTorch implementation. Both launchers select `VLLM_ATTENTION_BACKEND=FLASH_ATTN`; no quantization or eager-only fallback appears in either engine configuration.

Live package metadata agrees for PyTorch 2.8.0+cu129, Triton 3.4.0, NCCL 2.27.3, cuBLAS 12.9.1.4, CUDA runtime 12.9.79, cuDNN 9.10.2.21, Transformers 4.56.1, Tokenizers 0.22.0, xformers 0.0.32.post1, LightGBM 4.6.0, scikit-learn 1.7.2, OpenAI 1.107.3, NumPy 2.2.6 and msgspec 0.19.0. Python is 3.12.11 on both, from different Conda distributions. These are current environment inventories, reinforced by completed-run logs for vLLM/backend/NCCL; a complete historical `/proc/maps` inventory was not saved for Bridges.

The main vLLM CUDA extension and both bundled FlashAttention CUDA extensions are byte-identical:

| Binary | SHA256 on both hosts |
|---|---|
| `_C.abi3.so` | `37b54afd7aee99ed9d482f0b98760c8cf96fd397ca323cfae70534c244b63b93` |
| `_vllm_fa2_C.abi3.so` | `a1bd03e4848c7bb0673e9a91e27505cab9e42d68b31c573d8f597689f1871fc2` |
| `_vllm_fa3_C.abi3.so` | `1975ea2476bb8f17d6c1cbf4da8c1b28a1c272135a4b7d98c3c2c9b2772e4f8f` |

Vast's live worker mappings load cuBLAS, CUDA runtime and NCCL from the matching pip package directories, plus these same vLLM CUDA extensions. The host NVIDIA driver is provider-owned (`580.95.05`), so this is not a claim of a byte-identical whole operating-system environment. Both FA2 and FA3 binaries are loaded by the interface; their presence alone is not proof of which function executes. The H100 selection code defaults to FA3 when supported; Vast workers have no `VLLM_FLASH_ATTN_VERSION` override. After the probe completed, calling the deployed selector returned attention version 3 and GPU capability (9, 0). Historical logs only identify the FlashAttention backend, not its exact internal version.

vLLM package labels differ (`b70f4dbb4` metadata on Bridges, `28bbf9226` on Vast). The cloud commit packages the required family/predictor/snapshot changes; equality of checked runtime source contents and CUDA hashes is more informative than those editable-build version strings. The scheduler simulator C++ source hash also matches (`d135b72b2a90b6e4b0a2a78f7e222dca29d35f02bf251e8359af6729c76b1596`); that CPU extension was rebuilt on Vast and is not claimed byte-identical.

## Earlier canonical reference

The manifest-selected April 6 Qwen 8.6 point has 90.63125% end-to-end TTFT attainment, versus the recent Bridges control's 81.16875%. After normalizing the old model-path aliases, 11,049 requests chose the same model between those two runs. Prompt token counts, prompt usage, quality predictions and output-length predictions match exactly for all 11,049. Actual completion lengths differ for 8,166, with mean current-minus-old delta -2.17 tokens. The original-to-current difference therefore also needs runtime/history analysis; it is not explained by retraining or relocating the canonical predictors. The older trace lacks `system_entry_offset_s`; any older request-quarter calculation uses request-ID order, not independently verified arrival timestamps.

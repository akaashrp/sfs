# Plan: predictor/judge ablations for all Qwen methods (6/7/8/8.3 QPS)

Repo: `/ocean/projects/cis250162p/aparthas/sfs_cloud_20260914` (branch `experiments/cloud-68-20260914`, HEAD `7380c13`). Bundle: `/ocean/projects/cis250162p/aparthas/sfs_cloud_artifacts_20260916`.

## 0. Bottom line

- Only four policies consume either predictor at routing time: `hard`, `hard_prefill_tps`, `score`, `latency_agnostic`. The other six (`round_robin`, `shortest_queue`, `lmdeploy_proxy`, `mooncake_prefill`, `routebalance`, `vllm_sr_latency`) are pure no-ops for all three variants; their canonical cells should be reused and, for `flash_quality`, simply reported under the Flash judge (which collation already computes for every Qwen cell).
- `flash_quality` is both a different routing-time quality predictor (a LightGBM model trained on Flash labels, swapped via `--accuracy-model-path`) and a different primary judge at collation. Only the four consuming policies need fresh runs.
- `mlp_length` changes every vLLM server's argv, but the engine uses the prediction only to build the published SHM snapshot (`num_output_target_tokens`, decode backlog, snapshot-emission projection); it never alters batch composition or KV admission. Snapshot consumers that read those fields are `hard`, `hard_prefill_tps` (pending overlay), and `score`; `shortest_queue` reads only request counts; the methodology snapshot parser ignores the field.
- Net fresh work: 36 cells now (3 variants x {hard, score, latency_agnostic} x 4 rates) plus 11-12 canonical comparator cells (hard/score/latency_agnostic at 6/7/8/8.3 on cloud); +16 later for `hard_prefill_tps`. Removed 72 no-op cells.

## 1. Which policies consume which predictor (with references)

### 1.1 How a variant enters the system

- Router argv: `src/scripts/cloud/worker.py:18-22` — `arguments()` replaces `--output-length-model-path` (mlp_length) or `--accuracy-model-path` (mlp_quality, flash_quality) with `manifest['variants'][variant]`.
- Servers: `worker.py:171` extracts the router's `--output-length-model-path` value and passes it to `pool(...)`; `src/scripts/cloud/pool.py:33-36` sets `--output-length-model-path` on every Qwen server (`qwen_baselines.server_argv` + `set_option`). No server flag carries the quality predictor. So: mlp_length reaches the router and all three servers for every policy; quality variants reach the router only.
- Router scheduler selection: `src/scripts/runs/experiments.py:4440-4443` and `4461-4502`. `vllm_sr_latency` → `LatencyHistoryScheduler`; `{lmdeploy_proxy, mooncake_prefill, routebalance}` → `MethodologyScheduler`; everything else → `CollectingWaitTimeScheduler(accuracy_model_path=..., output_length_model_path=...)`.

### 1.2 Policies that never load the router predictors

- `src/sfs_core/routing/methodology_scheduler.py:42-43`: `kwargs.update(accuracy_model_path=None, output_length_model_path=None, enable_wait_time_polling=False, readiness_predictor_path=None)`. RouteBalance's quality/length come from its own MiniLM/KNN artifact (`bundle/qwen/routebalance`, `methodology_scheduler.py:296-325`); Mooncake uses prompt tokens and prefill work only (`:279-292`); LMDeploy uses unfinished counts (`:275-276`) and reads no snapshots (`:236`).
- `src/sfs_core/routing/latency_history_scheduler.py:12-13`: same `None` override for `vllm_sr_latency`.
- Methodology snapshot parsing (`src/sfs_core/routing/methodology_snapshot.py:250-272`) reads `num_prompt_tokens`, `num_computed_tokens`, `num_output_processed_tokens`, `status`, `kv_block_counts` — never `num_output_target_tokens`. So the server-side length predictor is invisible to Mooncake/RouteBalance too.

### 1.3 Policies on the shared WaitTimeScheduler

Predictions are computed unconditionally for every request (`experiments.py:1365-1369`; `wait_time_scheduler.py:585-589`) and logged as `predicted_accuracy` / `predicted_output_tokens` (`experiments.py:1595-1604`), so the per-request logs will differ for every policy under a variant. What matters is consumption:

- Utility functions (`experiments.py:2400-2482`): `latency_agnostic` = `accuracy - λ·_predicted_cost(output_lengths)`; `score` = published SCORE terms with `predicted_quality` and `output_rate * predicted_output_tokens`; `hard`/`hard_prefill_tps` = `hard_slo_candidate_value(predicted_quality, predicted_cost(output_lengths))`.
- Pending-dispatch overlay: `experiments.py:1494-1500` reserves `predicted_output_tokens=output_lengths.get(target_id, 1.0)`; the native simulator uses it (`vllm/csrc/scheduler_sim/bindings.cpp:358-372`, `:889`) when estimating waits for polled policies.
- Route strategy / estimator table (`experiments.py:6221-6285`): `round_robin` → cursor only (`:1085-1091`), zero estimator, skips wait build; `latency_agnostic` → zero estimator, skips wait build; `shortest_queue` → live polling but selection uses only `num_requests` (`:1177-1218`, `:2847-2872`; native count at `bindings.cpp:2426-2433` is a request count, independent of targets); `hard` → live snapshot simulation; `hard_prefill_tps` → prefill-backlog estimator (`:3553-3581`, prompt tokens only) plus router-side cost term; `score` → total-latency estimator that needs router `output_lengths` (`:3731-3760`) and snapshot `decode_backlog_total_tokens` (`:3776-3812`).

### 1.4 Server-side effect of mlp_length

- `vllm/vllm/v1/engine/async_llm.py:351-389`: prediction attached to every request (mandatory because `--enable-snapshot-shm-publishing` is on).
- `vllm/vllm/v1/request.py:165-175` → `request.output_length_prediction`.
- `vllm/vllm/v1/core/sched/scheduler.py:1527-1572` `_snapshot_output_target` → `num_output_target_tokens` and `decode_backlog_total` in the published snapshot (`:1636-1673`); `:1414-1430` decode-tail residual statistics; `:1058-1075` projected-finished set, used only to force `_emit_snapshot` (`:1273-1276`). No use in `schedule()`'s batching/KV decisions. Generation is unchanged; only published telemetry (and negligibly, predictor CPU cost/emission timing) differs.

### 1.5 Matrix

| Policy | flash_quality | mlp_quality | mlp_length | Verdict |
|---|---|---|---|---|
| hard | routing changes | routing changes | routing (cost term, pending overlay) + snapshot targets | fresh run |
| hard_prefill_tps | routing changes | routing changes | routing (cost term, pending overlay) | fresh run (later) |
| score | routing changes | routing changes | routing (cost/latency terms) + snapshot decode backlog | fresh run |
| latency_agnostic | routing changes | routing changes | routing (cost term only) | fresh run |
| shortest_queue | logged only | logged only | logged only; snapshot targets unused (count only) | no-op: reuse canonical |
| round_robin | logged only | logged only | logged only | no-op: reuse canonical |
| lmdeploy_proxy | not loaded | not loaded | not loaded; no snapshots | no-op |
| mooncake_prefill | not loaded | not loaded | snapshot field ignored | no-op |
| routebalance | not loaded (own predictor) | not loaded | snapshot field ignored | no-op |
| vllm_sr_latency | not loaded | not loaded | not loaded; no snapshots | no-op |

A rerun of a no-op cell measures run-to-run noise only.

## 2. What flash_quality really is

- Bundle contents: `variants/flash_quality/accuracy_model.txt` (LightGBM), `metadata.json` (`predictor_backend: lightgbm`, `train_examples: 26448`, `test_examples: 2961` vs canonical 26961/3000), `test_example_ids.json`. `provenance/training_audit.json`: `judge_model: gemini-2.5-flash`, trained on common observed Pro/Flash labels, 513 imputed rows excluded.
- So the arm has two parts: (a) a Flash-label-trained routing predictor, applied via `--accuracy-model-path` (`worker.py:20-22`); (b) the Flash judge as primary at collation (`src/scripts/cloud/collate.py:14-21` `evaluation_judge`). Part (b) is already computed for every Qwen cell regardless of variant: `observed_utilities` returns `ontimeutility: {pro, flash}` on the common 15,996 cohort (`collate.py:39-58`).
- Consequence: fresh runs only for `hard`, `hard_prefill_tps`, `score`, `latency_agnostic`; for the six no-op policies, "flash_quality" = the canonical cell's Flash-judge number, already present in `observed_judge_summary.json`. A relabeling/matrix step, not a re-collation.
- Comparators: the canonical `hard`/`score`/`latency_agnostic` cells at 6/7/8/8.3 do not yet exist on cloud (execution plan pauses SFS/SCORE; latency_agnostic/RR/SQ references are Bridges-only, April sources, see `scripts/cloud/reports/baseline-campaign-20260916/qwen-reference-inventory.json`). Cross-hardware comparison is disallowed by the README, so those canonical cells must run on cloud too (two Vast `hard` @ 8 controls exist and can serve as the 8 QPS comparator).

## 3. Minimal code changes

### 3a. Variant overlay (new module; do not touch `apply_campaign`)

`apply_campaign` (`src/scripts/cloud/baseline_campaign.py:9-26`) hard-codes the canonical grid and rejects non-canonical variants; `test_baseline_campaign.py` pins that behaviour. `validate_bundle` (`common.py:70-79`) requires 68 cells but only reads `bundle.json`; overlays replace `cells` in memory only, so the frozen bundle is unaffected.

Add `src/scripts/cloud/variant_campaign.py`:
- `CONSUMING_POLICIES = ('hard', 'hard_prefill_tps', 'score', 'latency_agnostic')`; `NOOP_POLICIES = (...)` with a `VARIANT_SENSITIVITY` dict documenting the reason (section 1).
- `apply_variant_campaign(bundle, campaign)`: require `campaign['kind'] == 'predictor_variants'`, `family == 'qwen'`, every cell `variant in bundle['variants']`, `policy in CONSUMING_POLICIES` (reject no-op policies in `cells`; they belong in `noop_cells` with `reuse` pointers), `qps in RATES['qwen']`, `requests == 16000`, unique ids of the form `qwen-{variant}-{policy}-{qps:g}` (the frozen ids `qwen-mlp_quality-7` omit the policy; new ids must include it). Return a deep copy with `cells`, `families.qwen.policies` (set of policies in cells), `families.qwen.qps`, `requests_total`; `files` untouched.
- Commit `scripts/cloud/variant-campaign-20260916.json` (36 cells now; `hard_prefill_tps` block added later). `source_hashes()` (`common.py:60-66`) does not hash arbitrary JSON under `scripts/cloud`, but `campaign_sha256` is already recorded in `qualification.json` (`worker.py:265`) and enforced by `validate_release` (`worker.py:334`), so the overlay is bound to the qualification for free.
- `worker.main` (`worker.py:359-361`): dispatch on `campaign.get('kind')` — absent/`baseline` → `apply_campaign`, `predictor_variants` → `apply_variant_campaign`.

### 3b. Worker: explicit policy list for non-canonical variants

- Replace `worker.py:206` (`policies = definition['policies'] if options.variant == 'canonical' else ['hard']`) with a helper `policies_for(manifest, family, variant, definition)` = sorted set of policies among `manifest['cells']` for that family/variant, falling back to `definition['policies']`. With the frozen bundle this yields `['hard']` for variants and the existing lists for canonical (behaviour unchanged); with the overlay it yields the campaign's policies. The smoke loop (`worker.py:208-216`) then smokes every policy the pool will run.
- Cell filtering (`worker.py:271-276`) and `--cells` need no change. `--variant` choices (`worker.py:346`) already cover the three arms.
- `score` needs `score_proxy` (decode TPS, mean decode-batch ms) and `hard_prefill_tps` needs `prefill_tps` from `--service-metrics-json`; `calibrate` already writes both (`worker.py:139-141`).

### 3c. Qualification

- `validate_release` (`worker.py:327-337`) binds `report['variant'] == options.variant`, so each variant already needs its own qualification. Recommendation: keep it that way and run `campaign` mode once per variant (fresh calibration + per-policy smoke + review). Rationale: for `mlp_length` the server argv differs, so calibration must be fresh; for the quality variants the servers are byte-identical and calibration could be reused, but adding a reuse path costs code and audit complexity to save roughly 10-15 minutes per pool. Do not add reuse code.
- One pool per variant; never two variants in one 4-GPU slot (README).

### 3d. Collation, ledger, provenance

- `collate.py:76-77` builds `expected` from the 68 bundle cells and rejects anything else ("Unexpected campaign cell"). Add `--campaign` to `collate()` using the same kind-dispatch, record `campaign_sha256` in `audit.json`, and replace the literal `PASS_68_CELLS` with `f'PASS_{len(expected)}_CELLS'`.
- Derived layout `derived/{family}/{variant}/{cid}.json` (`collate.py:99`) already tolerates several policies per variant because `aggregate_jsons` groups by qps x utility.
- Add `variant_matrix.json` to the collate output: one row per (variant, policy, qps) with `source: measured|reused_canonical`, `cell_id`, `primary_judge` (from `evaluation_judge`), `ontimeutility {pro, flash}`, `ttft_slo_attainment_pct`, `hardware`, `source_sha256`, `qualification_sha256`, `campaign_sha256`. Reused rows are copied from the canonical cells of the baseline collation (`--reuse-root` pointing at its `observed_judge_summary.json`), never from Bridges points without `provider: bridges` in the row.
- `control.status` (`control.py:27-48`) reads bundle cells only; add `--campaign` so `remaining`/`completed` reflect the overlay. Completion entries (`worker.py:302-306`) already carry `cell` (variant+policy) and `qualification_sha256`; add `campaign_sha256`.
- Append entries with `variant` to `scripts/cloud/experiment-ledger-20260916.json` (identity fields already include it).

## 4. Ordering and GPU-time estimate

Fresh cells after removing no-ops: 3 variants x 3 policies x 4 rates = 36 (16,000 requests each). The frozen variant `hard` cells are at 7/8/8.6/8.75; 7 and 8 coincide with the new grid but none have completed on cloud per `STATUS.md`, so assume all 36. Canonical comparators on cloud: `hard`, `score`, `latency_agnostic` at 6/7/8/8.3 = 12, minus the two completed `hard` @ 8 controls = 11. Later: `hard_prefill_tps` = 12 variant + 4 canonical (already in the fallback backlog).

At 40-48 min per cell on a 4-GPU pool: 36 cells = 24-29 h of pool time; +11 canonical = 7-9 h; per-variant qualification (calibration, 3 policy smokes, review) about 25 min each. On two slots, roughly 16-19 h wall for the first tranche; `hard_prefill_tps` later adds about 12 h.

Suggested order:
1. Prerequisites: resolve the arrival-rate under-delivery that stopped both campaign workers (the dirty `_ArrivalSchedule` change in `src/scripts/runs/experiments.py` looks like that fix; it is a protected-source file, so `protected-source.json`, the CPU gate and `source_hashes` must be refreshed after review); lift the SFS/SCORE pause from `execution-plan-20260916.json`.
2. Slot A: canonical `hard`/`score`/`latency_agnostic` at 6/7/8/8.3 (11 cells). Slot B: qualify and run `mlp_length` first (the only variant that changes server behaviour), policies in order hard → score → latency_agnostic, rates ascending.
3. Then `mlp_quality`, then `flash_quality` (both are router-only changes; either slot).
4. `hard_prefill_tps` for all arms after the backlog is authorized.

## 5. Tests to add (all run inside `scripts/cloud/test.sh` because they live in `src/scripts/cloud/tests`; the gate needs at least 70 tests, none skipped)

- `test_variant_campaign.py`: accepts the committed overlay; rejects no-op policies in `cells`, a Ministral cell, an unknown variant, an off-grid rate, a non-16000 budget, duplicate ids; the input bundle and its `files`/68 cells are untouched; `requests_total` correct; kind dispatch in worker/collate/control.
- Worker policy derivation: `policies_for` on the frozen `bundle.json` returns `['hard']` for each variant and the five NEW_POLICIES for canonical Qwen; on the overlay returns the campaign policies.
- `arguments()` per variant x policy: exactly one flag differs from canonical (pattern from `src/scripts/runs/tests/test_qwen_predictor_variants.py`); `pool.server_argv` receives the variant path only for `mlp_length` (extend `test_mlp_length_reaches_every_qwen_server`), identical to canonical for quality variants.
- No-op classification pinned to code: `MethodologyScheduler` and `LatencyHistoryScheduler` constructed with predictor paths still have no predictors loaded; `_baseline_runtime_params('round_robin')`/`('latency_agnostic')` skip wait building with the zero estimator; `methodology_snapshot` parsing is invariant to `num_output_target_tokens`; `_resolve_effective_num_requests` depends only on `num_requests`.
- Collate: `--campaign` acceptance and status label; `variant_matrix.json` built from a synthetic mix of measured and reused rows with correct judge labelling and provenance fields; overlay cell mismatch still rejected.
- `control.status --campaign` reports overlay cells.

### Critical Files for Implementation
- /ocean/projects/cis250162p/aparthas/sfs_cloud_20260914/src/scripts/cloud/worker.py
- /ocean/projects/cis250162p/aparthas/sfs_cloud_20260914/src/scripts/cloud/baseline_campaign.py (pattern for the new variant_campaign.py)
- /ocean/projects/cis250162p/aparthas/sfs_cloud_20260914/src/scripts/cloud/collate.py
- /ocean/projects/cis250162p/aparthas/sfs_cloud_20260914/src/scripts/cloud/control.py
- /ocean/projects/cis250162p/aparthas/sfs_cloud_20260914/src/scripts/cloud/tests/test_baseline_campaign.py (test pattern; plus test_portable_campaign.py)
## Decisions recorded 16 September 2026 (user)

- No-op policies (round robin, shortest queue, LMDeploy, Mooncake, RouteBalance, vLLM-SR) are NOT rerun for any variant; their canonical cells are reused and labeled `reused_canonical` in the variant matrix.
- Canonical SFS (`hard`) IS rerun on Vast at 6/7/8/8.3 as the on-host comparator for the flash/MLP ablations (8 QPS may reuse the two audited lane controls). Canonical latency-agnostic is NOT rerun; the April Bridges reference stays the main-grid entry and latency-agnostic is dropped from the ablation matrix.
- SCORE has no result at these rates anywhere; canonical SCORE at 6/7/8/8.3 is required for the main grid and as the ablation comparator.
- Resulting fresh work: 3 variants x {SFS, SCORE} x 4 rates = 24 ablation cells; canonical SFS 6/7/8.3 = 3 cells; canonical SCORE 6/7/8/8.3 = 4 cells. `hard_prefill_tps` stays at the back of the queue.
- The overlay module must therefore also express canonical SFS/SCORE cells on the 6/7/8/8.3 grid (the frozen bundle has SCORE only at 7/8/8.6/8.75 and no canonical Qwen `hard` cells).

## Decision update, 17 September 2026 (user)

- Latency-agnostic IS ablated (flash_quality, mlp_quality, mlp_length at 6/7/8/8.3); its comparator is the April Bridges canonical latency-agnostic reference, accepted by the user as comparable (no Vast canonical rerun). Label host/source provenance in the matrix.
- Ablation tranche therefore = 3 variants x {SFS, SCORE, latency-agnostic} x 4 rates = 36 cells, plus canonical SFS 6/7/8.3 and SCORE 6/7/8/8.3 on Vast (7 cells).
- Prefill-TPS estimator (hard_prefill_tps) deferred.
- SFS and SCORE are also required under the FCFS unchunked configuration (8 cells, currently BLOCKED in the FCFS overlay) and for Ministral (6 cells); all SFS/SCORE work waits for the reserve-change decision.

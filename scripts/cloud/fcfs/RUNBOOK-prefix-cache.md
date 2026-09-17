# Qwen prefix-caching serving configuration (`qwen-prefix-cache`): qualification and reduced-grid runbook

Nothing here has been run. Configuration `prefix_cache` (`src/scripts/cloud/serving/profiles.py`):
canonical Qwen serving with vLLM automatic prefix caching ON (`--no-enable-prefix-caching`
swapped for `--enable-prefix-caching`); chunked prefill on, `max_num_batched_tokens` 32768,
`max_model_len` 131072, `max_num_seqs` 512, TP 1/1/2, kernels, precision, chat template,
YaRN/RoPE, predictors, SLOs and the 16,000-request workload canonical. The only other argv
addition is `--enable-prompt-tokens-details`, so every response usage carries
`prompt_tokens_details.cached_tokens` (observability, no scheduling effect); instances.json rows
carry `prefix_caching_enabled: true`.

Overlay: `scripts/cloud/fcfs/campaign-prefix-cache-20260917.json` (kind `serving_config`, sha256
`41faa6f87d31e2b8a0210fbdac1170e2f9bc7936c90aed4fe4f2639477b405ac`), reduced grid
`policies = ["hard", "mooncake_prefill", "lmdeploy_proxy"]` x 6/7/8/8.3 QPS x 16,000 requests =
12 cells `qwen-prefix-cache-<policy>-<qps>`, `full_matrix_authorized: true` ("User authorized on
17 September 2026: reduced grids (SFS + two strongest baselines) for the prefix-caching and
fine-chunk serving configurations"), policy rule "SFS plus the two highest-utility external
baselines on the canonical Vast grid; revisit when canonical SCORE lands", and the same
`remaining_length` block as the canonical and FCFS SFS/SCORE overlays (0.6B rule
`running_all:0.5:prompt_bin`, 8B/32B on the current rule). Cells derive from `policies` when
the overlay is applied: to change the grid edit only that list; the overlay hash changes, so the
lane re-qualifies against the edited file before `worker run` accepts it.

Coefficient policy `canonical`: per-token step costs are unchanged by prefix caching (batch
stats count computed tokens, cached tokens are never scheduled), so the canonical SFS batch
coefficients are retained; `worker qualify/run --profile prefix_cache` refuse `--coefficients`
and `calibrate`, and `qualification.json` records `coefficient_policy canonical`,
`coefficients_sha256 null`. The overlay records that the SFS batch simulator
(`get_computed_blocks` returns no cached blocks), the Mooncake prefill estimator and the
RouteBalance TPOT head are cache-unaware by design: waiting requests are costed as full-prompt
prefills although the engine will serve cached prefixes from the KV cache, while running
requests are read from the snapshot after the hit. Measuring the routers under that
estimator/engine mismatch is the point of the ablation; no estimator is changed.

Before launching, read the GPU-7 hit-rate probe if it has completed
(`/workspace/sfs/state/diagnostics/prefix-cache-probe-20260917/run/analysis.json`, staged from the
cloud branch as `sfs-prefix-probe`; the dry-run attribution found a 24-token shared template
prefix, 16-34-token dataset headers and 202 duplicate prompts): the user decides whether the
measured hit rate justifies the 12 cells.

Source: branch `experiments/qwen-fcfs-unchunked-20260916` at `454d98c902a74088e5e4b7a1c8ab2f3c2262e144` (vllm submodule
`30ec5b4b5418ce429e5cac2693e8e34fe0e82520`, no csrc/CMake/setup change since `28bbf92`, so the
extensions compiled in `/workspace/sfs/fcfs-repo` stay valid). The runbook commits that follow
only add markdown and do not change the source hash the gates bind.

Host rules (unchanged): `ssh sfs-vast`, `SFS_STORAGE=/workspace/sfs`; do not touch
`/workspace/sfs/repo`, `repo-v2`, `repo-v3`, `reserve-repo`, `fcfs-repo` or `fcfs-repo-v2`,
their state roots or the FCFS gates in `/workspace/sfs/fcfs/setup`; do not stop any running
Supervisor program. This configuration uses a NEW state root
`/workspace/sfs/serving/prefix-cache` (`setup/` gates, `state/` runs, ledger
`state/completed/`); the worker resolves gates from `--state`'s parent. Every pool takes the
GPU-UUID locks in `/dev/shm/sfs-cloud-locks-0`, so a busy GPU fails fast instead of sharing.
One Supervisor program per command (`/etc/supervisor/conf.d/sfs-serving-*.conf`,
`autostart=false`, `autorestart=false`), logs under `/workspace/sfs/setup/`.

## 1. Checkout `/workspace/sfs/fcfs-repo-v3` (shared with the chunk8192 configuration)

Same as `RUNBOOK-chunk8192.md` step 1 (one checkout serves both configurations; each has its
own state root). Verify the overlay hash and create the state root:

```bash
ssh sfs-vast 'cd /workspace/sfs/fcfs-repo-v3 && git rev-parse HEAD && git submodule status && sha256sum scripts/cloud/fcfs/campaign-prefix-cache-20260917.json'
ssh sfs-vast 'mkdir -p /workspace/sfs/serving/prefix-cache/setup /workspace/sfs/serving/prefix-cache/state'
```

Expected: `454d98c902a74088e5e4b7a1c8ab2f3c2262e144`, ` 30ec5b4b5...` without `+`/`-`, overlay sha256 `41faa6f8...`.

## 2. Gates into `/workspace/sfs/serving/prefix-cache/setup` (CPU only, about 20 min)

Identical to the chunk8192 gates with its own setup directory (the gates bind source and bundle,
not the profile; they are duplicated so each state root is self-contained). If the chunk8192
gates already passed on this same commit, copy them instead:
`cp -a /workspace/sfs/serving/chunk8192/setup/. /workspace/sfs/serving/prefix-cache/setup/`.

```bash
ssh sfs-vast 'cat > /workspace/sfs/setup/serving-prefix-cache-gates.sh' <<'EOF2'
#!/usr/bin/env bash
set -euo pipefail
export SFS_STORAGE=/workspace/sfs
source /workspace/sfs/fcfs-repo-v3/scripts/cloud/env.sh
export SFS_TEST_GO="$CONDA_PREFIX/bin/go"
cd /workspace/sfs/fcfs-repo-v3
S=/workspace/sfs/serving/prefix-cache/setup
rm -rf $S/tests
taskset -c 84-95 bash scripts/cloud/test.sh $S/tests
taskset -c 84-95 python -m pytest -q -p no:cacheprovider src/sfs_core/routing/tests/test_remaining_length_fallback.py src/scripts/runs/tests/test_arrival_schedule.py
(cd vllm && taskset -c 84-95 python -m pytest -q -p no:cacheprovider tests/v1/core/test_scheduler_remaining_length.py tests/v1/core/test_snapshot_serialization.py)
taskset -c 84-95 python -m scripts.cloud.prepare cpu     --bundle /workspace/sfs/bundle --output $S/cpu-inputs.json
taskset -c 84-95 python -m scripts.cloud.prepare serving --bundle /workspace/sfs/bundle --output $S/cpu-serving.json
python3 -c "import json;g=json.load(open('$S/tests/gate.json'));print(g['status'],g['tests'])"
cd src && python -m scripts.cloud.control status --state /workspace/sfs/serving/prefix-cache/state --bundle /workspace/sfs/bundle \  --campaign /workspace/sfs/fcfs-repo-v3/scripts/cloud/fcfs/campaign-prefix-cache-20260917.json
EOF2
ssh sfs-vast 'cat > /etc/supervisor/conf.d/sfs-serving-prefix-cache-gates.conf' <<'EOF2'
[program:sfs-serving-prefix-cache-gates]
command=/bin/bash /workspace/sfs/setup/serving-prefix-cache-gates.sh
directory=/workspace/sfs/fcfs-repo-v3
autostart=false
autorestart=false
startsecs=0
stdout_logfile=/workspace/sfs/setup/sfs-serving-prefix-cache-gates.log
redirect_stderr=true
EOF2
ssh sfs-vast 'supervisorctl reread && supervisorctl update && supervisorctl start sfs-serving-prefix-cache-gates'
```

Expected: `PASS_CPU_REGRESSION` with at least 316 tests, the extra tests passed, and a status
listing with `expected: 12`, `campaign_kind: serving_config` and the twelve `qwen-prefix-cache-*`
ids remaining.

## 3. Qualification (4 GPUs, about 35-50 min; no calibrate pass, no coefficient refit)

```bash
ssh sfs-vast 'cat > /workspace/sfs/setup/serving-prefix-cache-qualify.sh' <<'EOF2'
#!/usr/bin/env bash
set -euo pipefail
export SFS_STORAGE=/workspace/sfs
source /workspace/sfs/fcfs-repo-v3/scripts/cloud/env.sh
cd /workspace/sfs/fcfs-repo-v3/src
exec taskset -c 48-95 python -m scripts.cloud.worker qualify --profile prefix_cache --family qwen --variant canonical \  --bundle /workspace/sfs/bundle --models /workspace/sfs/models.json \  --state /workspace/sfs/serving/prefix-cache/state --output /workspace/sfs/serving/prefix-cache/state/qualify-20260917 \  --campaign /workspace/sfs/fcfs-repo-v3/scripts/cloud/fcfs/campaign-prefix-cache-20260917.json \  --gpus 4,5,6,7 --cpus $(seq -s, 48 95)
EOF2
ssh sfs-vast 'cat > /etc/supervisor/conf.d/sfs-serving-prefix-cache-qualify.conf' <<'EOF2'
[program:sfs-serving-prefix-cache-qualify]
command=/bin/bash /workspace/sfs/setup/serving-prefix-cache-qualify.sh
directory=/workspace/sfs/fcfs-repo-v3
autostart=false
autorestart=false
startsecs=0
stdout_logfile=/workspace/sfs/setup/sfs-serving-prefix-cache-qualify.log
redirect_stderr=true
EOF2
ssh sfs-vast 'supervisorctl reread && supervisorctl update && supervisorctl start sfs-serving-prefix-cache-qualify'
```

(Written for lane B, GPUs 4-7 / CPUs 48-95, so it can run beside the chunk8192 lane; for GPUs
0-3 change `--gpus`, `--cpus` and `taskset` to `0,1,2,3` / `0-47`. The four GPUs must read 0 MiB
first.) Inside: calibration with the canonical coefficients (warm shapes, singleton prefill
probes, 512 loaded requests) and timing-head fit, the `hard`, `mooncake_prefill` and
`lmdeploy_proxy` smokes (192 calibration prompts at 2 QPS each), shortest-queue load probes at 6
and 8.3 QPS, then `qualification.json` (`GPU_MEASURED_REVIEW_REQUIRED`) and exit. Note that the
singleton prefill probes repeat the warm-shape prompts, so under prefix caching they are cache
hits; the prefill coverage of the timing heads comes from the 512-request loaded phase (checked in
step 4). Check the bound identity and the cache evidence:

```bash
ssh sfs-vast 'O=/workspace/sfs/serving/prefix-cache/state/qualify-20260917; cat $O/status.json; echo;
  python3 -c "import json;q=json.load(open(\"$O/qualification.json\"));print(q[\"configuration_id\"],q[\"coefficient_policy\"],q[\"coefficients_sha256\"],q[\"campaign_kind\"],q[\"policy_smoke\"]);print(q[\"campaign_sha256\"]);print(q[\"remaining_length_rule\"]);print(q[\"serving_profile\"][\"prefix_caching\"],q[\"serving_coefficients\"]);print(q[\"load_probes\"])";
  python3 -c "import json;b=json.load(open(\"$O/instances.json\"));print(b[\"configuration_id\"],b[\"coefficient_status\"],[r.get(\"prefix_caching_enabled\") for r in b[\"instances\"]]);r=b[\"remaining_length\"];print(r[\"rule\"],{m:x[\"rule\"] for m,x in r[\"models\"].items()})";
  for f in $O/server_argv_*.json; do echo "$(basename $f): prefix on $(grep -c -- "\"--enable-prefix-caching\"" $f || true) prefix off $(grep -c -- "--no-enable-prefix-caching" $f || true) details $(grep -c -- "--enable-prompt-tokens-details" $f || true) rule flags $(grep -c -- "--remaining-length-mode" $f || true)"; done;
  grep -h "Prefix cache hit rate" $O/server_qwen3-*.log | tail -3;
  python3 -c "
import json,collections
r=json.load(open(\"$O/calibration_responses.json\"))[\"responses\"];by=collections.defaultdict(lambda:[0,0,0])
for x in r:
    u=x[\"usage\"];d=(u.get(\"prompt_tokens_details\") or {}).get(\"cached_tokens\") or 0;m=by[x[\"model\"]];m[0]+=u[\"prompt_tokens\"];m[1]+=d;m[2]+=d>0
for m,(p,c,h) in by.items():print(m,\"prompt tokens\",p,\"cached\",c,\"hit rate %.1f%%\"%(100*c/p),\"responses with a hit\",h,\"/\",sum(1 for x in r if x[\"model\"]==m))"'
```

Expected: `qwen-prefix-cache canonical None serving_config ['hard', 'mooncake_prefill',
'lmdeploy_proxy']`, the overlay sha256 `41faa6f8...`, rule `qwen3-0.6b=running_all_prompt_bin_q50`
with 8B/32B `current`, `serving_profile.prefix_caching true`, `coefficient_status
CANONICAL_RETAINED_BY_CONFIGURATION_POLICY`, rows `[True, True, True]`, every argv `prefix on 1
prefix off 0 details 1`, rule flags `1` only for `server_argv_qwen3-0.6b.json`, the server logs'
"Prefix cache hit rate" lines above 0%, and per-model cached-token counts > 0 in the
calibration responses (the repeated singleton probes are full hits; the loaded phase shows the
template/header hits). Record the per-model hit rate in the release text.

## 4. Reviews and release (CPU)

```bash
ssh sfs-vast 'export SFS_STORAGE=/workspace/sfs; source /workspace/sfs/fcfs-repo-v3/scripts/cloud/env.sh; cd /workspace/sfs/fcfs-repo-v3;
  Q=/workspace/sfs/serving/prefix-cache/state/qualify-20260917;
  python ops/cloud/review_baseline_timing.py --qualification $Q --output $Q/timing-review.json;
  python3 -c "import json;c=json.load(open(\"$Q/timing_models/methodology_calibration.json\"));print({k:c.get(k) for k in c if \"partial_prefill\" in k or \"chunk_token_range\" in k})";
  for P in hard mooncake_prefill lmdeploy_proxy; do python3 -c "import json;r=json.load(open(\"$Q/smoke/$P/point.json\"))[\"router\"][\"runs\"][0];print(\"$P\",r[\"summary\"],r.get(\"remaining_length_rule\"))"; done;
  grep -ciE "out of memory|preempt|cuda error" $Q/server_qwen3-*.log || true'
```

Checklist (record concrete numbers in the release text): TPOT-head MAE on independent smoke
rows and the Mooncake prefill head residuals (`timing-review.json`); `chunk_token_range`
coverage in `methodology_calibration.json` still reaches the long prompts from the loaded phase
(if the coverage collapsed because of cache hits, stop and report; do not widen the
calibration silently); the three smokes `succeeded_requests == 192`, `failed_requests == 0`,
`remaining_length_rule` equal to the pool rule; load probes at 6 and 8.3 QPS stable with
realized rates within 10%; SFS predicted-versus-observed TTFT on
`smoke/hard/predicted_waits_router_hard.log`: systematic over-prediction on cache-hit prompts
is expected (the estimators are cache-unaware) and is recorded, not corrected; the per-model
cache hit rates from step 3; server logs free of OOM/preemption. There is no coefficient
review: `coefficients_sha256` is null by policy.

```bash
ssh sfs-vast 'export SFS_STORAGE=/workspace/sfs; source /workspace/sfs/fcfs-repo-v3/scripts/cloud/env.sh; cd /workspace/sfs/fcfs-repo-v3/src;
  python -m scripts.cloud.control release --qualification /workspace/sfs/serving/prefix-cache/state/qualify-20260917 \    --timing-review "<TPOT head MAE, prefill head residuals, chunk coverage, canonical coefficients retained, SFS over-prediction on cache hits noted>" \    --load-review "<hard/mooncake/lmdeploy smokes 192/192 with attainment, realized 2 QPS, load probes 6/8.3 stable, per-model cache hit rates, 0.6B rule flags placement, hardware GPUs 4-7>"'
```

## 5. Run the 12 cells (`worker run`, same four GPUs, about 8-9 h)

`validate_release` compares `hardware.json` of the qualification with the live `nvidia-smi`
fingerprint exactly, so the run uses the same four GPUs as the qualification; it also re-checks
the overlay hash, configuration id, null coefficient hash, rule and source per cell.

```bash
ssh sfs-vast 'cat > /workspace/sfs/setup/serving-prefix-cache-run.sh' <<'EOF2'
#!/usr/bin/env bash
set -euo pipefail
export SFS_STORAGE=/workspace/sfs
source /workspace/sfs/fcfs-repo-v3/scripts/cloud/env.sh
cd /workspace/sfs/fcfs-repo-v3/src
exec taskset -c 48-95 python -m scripts.cloud.worker run --profile prefix_cache --family qwen --variant canonical \  --bundle /workspace/sfs/bundle --models /workspace/sfs/models.json \  --state /workspace/sfs/serving/prefix-cache/state --output /workspace/sfs/serving/prefix-cache/state/run-20260917 \  --campaign /workspace/sfs/fcfs-repo-v3/scripts/cloud/fcfs/campaign-prefix-cache-20260917.json \  --qualification /workspace/sfs/serving/prefix-cache/state/qualify-20260917 \  --gpus 4,5,6,7 --cpus $(seq -s, 48 95) \  --cells qwen-prefix-cache-hard-6,qwen-prefix-cache-hard-7,qwen-prefix-cache-hard-8,qwen-prefix-cache-hard-8.3,qwen-prefix-cache-mooncake_prefill-6,qwen-prefix-cache-mooncake_prefill-7,qwen-prefix-cache-mooncake_prefill-8,qwen-prefix-cache-mooncake_prefill-8.3,qwen-prefix-cache-lmdeploy_proxy-6,qwen-prefix-cache-lmdeploy_proxy-7,qwen-prefix-cache-lmdeploy_proxy-8,qwen-prefix-cache-lmdeploy_proxy-8.3
EOF2
ssh sfs-vast 'cat > /etc/supervisor/conf.d/sfs-serving-prefix-cache-run.conf' <<'EOF2'
[program:sfs-serving-prefix-cache-run]
command=/bin/bash /workspace/sfs/setup/serving-prefix-cache-run.sh
directory=/workspace/sfs/fcfs-repo-v3
autostart=false
autorestart=false
startsecs=0
stdout_logfile=/workspace/sfs/setup/sfs-serving-prefix-cache-run.log
redirect_stderr=true
EOF2
ssh sfs-vast 'supervisorctl reread && supervisorctl update && supervisorctl start sfs-serving-prefix-cache-run'
```

The run pool starts fresh servers (empty prefix cache), re-smokes the three policies (192
calibration prompts each, which warms the template/header blocks exactly as a canonical run
warms its kernels), then measures the cells in overlay order. Within a cell the cache is warm
from the previous cell; that is the deployed steady state and is not reset. A cell is skipped
only if its ledger entry exists for this bundle under the current source, so a restarted
program resumes where it stopped. Every entry carries `configuration_id`, `coefficient_policy`,
`coefficients_sha256` (null), `campaign_sha256`, `campaign_kind`, `remaining_length_rule` and
the table sha256. Measured Qwen cells on this host took 39-46 min; cache hits can only shorten
prefill, so budget 40-45 min per cell (about 8-9 h for 12).

Progress, mirroring and collation:

```bash
ssh sfs-vast 'export SFS_STORAGE=/workspace/sfs; source /workspace/sfs/fcfs-repo-v3/scripts/cloud/env.sh; cd /workspace/sfs/fcfs-repo-v3/src;
  python -m scripts.cloud.control status --state /workspace/sfs/serving/prefix-cache/state --bundle /workspace/sfs/bundle \    --campaign /workspace/sfs/fcfs-repo-v3/scripts/cloud/fcfs/campaign-prefix-cache-20260917.json; cat /workspace/sfs/serving/prefix-cache/state/run-20260917/phase.json'
python scripts/cloud/sync_results.py --host sfs-vast --remote-state /workspace/sfs/serving/prefix-cache \  --destination /ocean/projects/cis250162p/aparthas/sfs_cloud_results_20260916/serving/prefix-cache --watch
python -m scripts.cloud.collate --bundle "$SFS_BUNDLE" --raw-root <mirror>/serving/prefix-cache/state/run-20260917 \  --output <results>/serving-prefix-cache-20260917 --campaign scripts/cloud/fcfs/campaign-prefix-cache-20260917.json --allow-partial
```

`audit.json` reports `PASS_12_CELLS` when complete. Afterwards copy `qualification.json`,
`release.json`, `timing-review.json` and `hardware.json` to
`scripts/cloud/reports/serving-prefix-cache-20260917/`, record the per-cell cache hit rates from
the points' usage (`prompt_tokens_details.cached_tokens` per response) beside the twelve ledger
entries with the configuration id and rule, and flag any SFS OnTimeUtility below a baseline at
the same rate (under a cache-unaware estimator that outcome is plausible and is the finding, not
a bug to fix silently).

## GPU time summary

| Step | GPUs | Estimate |
|---|---|---|
| Gates (CPU) | 0 | 20 min (or copy the chunk8192 gates from the same commit) |
| Qualify (calibration, 3 smokes, 2 probes) | 4 | 35-50 min |
| Reviews and release (CPU) | 0 | 15 min |
| Run 12 cells | 4 | 8-9 h (40-45 min per cell plus 10 min start and smokes) |
| Total | 4 | about 1 h qualification + 8-9 h measurement |

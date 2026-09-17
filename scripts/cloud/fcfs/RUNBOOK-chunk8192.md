# Qwen fine-chunk serving configuration (`qwen-chunk8192`): qualification and reduced-grid runbook

Nothing here has been run. Configuration `chunk8192` (`src/scripts/cloud/serving/profiles.py`):
canonical Qwen serving with `max_num_batched_tokens` 8192 instead of 32768 (chunked prefill on,
`long_prefill_token_threshold` 0 so prompts are chunked at 8192, `max_model_len` 131072,
`max_num_seqs` 512, prefix caching off, TP 1/1/2, kernels, precision, chat template, YaRN/RoPE,
predictors, SLOs and the 16,000-request workload canonical). Server argv delta over the
canonical Qwen argv: `--max-num-batched-tokens 8192`; instances.json rows carry
`max_num_batched_tokens: 8192` so the router's Python TTFT estimator chunks at 8192 too.

Overlay: `scripts/cloud/fcfs/campaign-chunk8192-20260917.json` (kind `serving_config`, sha256
`126f20bec633181ce05410dac412b2c91c26dc94cb94cb64e525cfc05d29403a`), reduced grid
`policies = ["hard", "mooncake_prefill", "lmdeploy_proxy"]` x 6/7/8/8.3 QPS x 16,000 requests =
12 cells `qwen-chunk8192-<policy>-<qps>`, `full_matrix_authorized: true` ("User authorized on
17 September 2026: reduced grids (SFS + two strongest baselines) for the prefix-caching and
fine-chunk serving configurations"), policy rule "SFS plus the two highest-utility external
baselines on the canonical Vast grid; revisit when canonical SCORE lands", and the same
`remaining_length` block as the canonical and FCFS SFS/SCORE overlays (0.6B rule
`running_all:0.5:prompt_bin`, 8B/32B on the current rule). Cells derive from `policies` when
the overlay is applied: to change the grid edit only that list; the overlay hash changes, so the
lane re-qualifies against the edited file before `worker run` accepts it.

Coefficient policy `refit`: the 8192-token step changes the batch timing regime, so the SFS batch
coefficients are fitted from destination traces of this configuration
(`scripts.cloud.serving.coefficients fit --profile chunk8192`) and `worker qualify/run --profile
chunk8192` require `--coefficients`; the file hash, `configuration_id`, `coefficient_policy`,
overlay hash, rule and table sha256 are bound to `qualification.json` and every completed-ledger
entry, and `validate_release` refuses a run whose configuration, coefficient file, overlay, rule,
source, bundle or hardware differs from its qualification.

Source: branch `experiments/qwen-fcfs-unchunked-20260916` at `454d98c902a74088e5e4b7a1c8ab2f3c2262e144` (vllm submodule
`30ec5b4b5418ce429e5cac2693e8e34fe0e82520`, no csrc/CMake/setup change since `28bbf92`, so the
extensions compiled in `/workspace/sfs/fcfs-repo` stay valid). The runbook commits that follow
only add markdown and do not change the source hash the gates bind.

Host rules (unchanged): `ssh sfs-vast`, `SFS_STORAGE=/workspace/sfs`; do not touch
`/workspace/sfs/repo`, `repo-v2`, `repo-v3`, `reserve-repo`, `fcfs-repo` or `fcfs-repo-v2`,
their state roots or the FCFS gates in `/workspace/sfs/fcfs/setup`; do not stop any running
Supervisor program. This configuration uses a NEW state root `/workspace/sfs/serving/chunk8192`
(`setup/` gates, `state/` runs, ledger `state/completed/`) so nothing under `/workspace/sfs/fcfs`
or `/workspace/sfs/state` is disturbed; the worker resolves gates from `--state`'s parent. Every
pool takes the GPU-UUID locks in `/dev/shm/sfs-cloud-locks-0`, so a busy GPU fails fast instead
of sharing. One Supervisor program per command (`/etc/supervisor/conf.d/sfs-serving-*.conf`,
`autostart=false`, `autorestart=false`), logs under `/workspace/sfs/setup/`.

## 1. Checkout `/workspace/sfs/fcfs-repo-v3` (copy keeps the compiled extensions)

```bash
ssh sfs-vast 'test -d /workspace/sfs/fcfs-repo-v3 || cp -a /workspace/sfs/fcfs-repo-v2 /workspace/sfs/fcfs-repo-v3'
ssh sfs-vast 'cd /workspace/sfs/fcfs-repo-v3 && git fetch origin experiments/qwen-fcfs-unchunked-20260916 \  && git checkout -q 454d98c902a74088e5e4b7a1c8ab2f3c2262e144 \  && git submodule update --init vllm && git submodule status \  && git status --short | grep -v "^??"; \  git -C vllm diff --stat 28bbf92 HEAD -- csrc CMakeLists.txt cmake setup.py; \  ls -la vllm/vllm/_C.abi3.so vllm/vllm/v1/engine/_scheduler_sim*.so; \  sha256sum scripts/cloud/fcfs/campaign-chunk8192-20260917.json scripts/cloud/remaining-length-tables-20260917/*'
ssh sfs-vast 'mkdir -p /workspace/sfs/serving/chunk8192/setup /workspace/sfs/serving/chunk8192/state'
```

Expected: `git submodule status` shows ` 30ec5b4b5...` with no `+`/`-` prefix, no tracked
modifications, an empty build-file diff, the extension files present, overlay sha256
`126f20be...` and the table hashes of `RUNBOOK-sfs-score.md` step 1 (`2eac8a79`, `50a14ca7`,
`74e3a4ef`, `2392950a`).

## 2. Gates into `/workspace/sfs/serving/chunk8192/setup` (CPU only, about 20 min)

No FCFS admission audit is needed (the context length is canonical); the worker refuses to
start without `tests/gate.json`, `cpu-inputs.json` and `cpu-serving.json` bound to this
source and bundle.

```bash
ssh sfs-vast 'cat > /workspace/sfs/setup/serving-chunk8192-gates.sh' <<'EOF2'
#!/usr/bin/env bash
set -euo pipefail
export SFS_STORAGE=/workspace/sfs
source /workspace/sfs/fcfs-repo-v3/scripts/cloud/env.sh
export SFS_TEST_GO="$CONDA_PREFIX/bin/go"
cd /workspace/sfs/fcfs-repo-v3
S=/workspace/sfs/serving/chunk8192/setup
rm -rf $S/tests
taskset -c 84-95 bash scripts/cloud/test.sh $S/tests
taskset -c 84-95 python -m pytest -q -p no:cacheprovider src/sfs_core/routing/tests/test_remaining_length_fallback.py src/scripts/runs/tests/test_arrival_schedule.py
(cd vllm && taskset -c 84-95 python -m pytest -q -p no:cacheprovider tests/v1/core/test_scheduler_remaining_length.py tests/v1/core/test_snapshot_serialization.py)
taskset -c 84-95 python -m scripts.cloud.prepare cpu     --bundle /workspace/sfs/bundle --output $S/cpu-inputs.json
taskset -c 84-95 python -m scripts.cloud.prepare serving --bundle /workspace/sfs/bundle --output $S/cpu-serving.json
python3 -c "import json;g=json.load(open('$S/tests/gate.json'));print(g['status'],g['tests'])"
cd src && python -m scripts.cloud.control status --state /workspace/sfs/serving/chunk8192/state --bundle /workspace/sfs/bundle \  --campaign /workspace/sfs/fcfs-repo-v3/scripts/cloud/fcfs/campaign-chunk8192-20260917.json
EOF2
ssh sfs-vast 'cat > /etc/supervisor/conf.d/sfs-serving-chunk8192-gates.conf' <<'EOF2'
[program:sfs-serving-chunk8192-gates]
command=/bin/bash /workspace/sfs/setup/serving-chunk8192-gates.sh
directory=/workspace/sfs/fcfs-repo-v3
autostart=false
autorestart=false
startsecs=0
stdout_logfile=/workspace/sfs/setup/sfs-serving-chunk8192-gates.log
redirect_stderr=true
EOF2
ssh sfs-vast 'supervisorctl reread && supervisorctl update && supervisorctl start sfs-serving-chunk8192-gates'
```

Expected at the end of the log: `PASS_CPU_REGRESSION` with at least 316 tests (263 on the
previous source plus the 53 serving-configuration tests; none skipped), the extra tests passed,
and a status listing with `expected: 12`, `campaign_kind: serving_config` and the twelve
`qwen-chunk8192-*` ids remaining.

## 3. 32B TP=2 bounded smoke (2 free GPUs, about 10 min)

Startup, memory and observed chunk-bound scheduling for the largest model; the snapshots must
report `max_num_batched_tokens 8192`, `chunked_prefill_enabled true`, and every scheduled step
must stay within 8192 tokens while the 32,801-token calibration prompt is prefilled in chunks.
Pick two GPUs that read 0 MiB (`nvidia-smi --query-gpu=index,memory.used --format=csv`).

```bash
ssh sfs-vast 'cat > /workspace/sfs/setup/serving-chunk8192-smoke32b.sh' <<'EOF2'
#!/usr/bin/env bash
set -euo pipefail
export SFS_STORAGE=/workspace/sfs
source /workspace/sfs/fcfs-repo-v3/scripts/cloud/env.sh
cd /workspace/sfs/fcfs-repo-v3/src
exec taskset -c 84-91 python -m scripts.cloud.serving.gpu_smoke --profile chunk8192 --bundle /workspace/sfs/bundle \  --models /workspace/sfs/models.json --output /workspace/sfs/serving/chunk8192/state/smoke-32b-20260917 \  --gpus 2,3 --indices 2
EOF2
ssh sfs-vast 'cat > /etc/supervisor/conf.d/sfs-serving-chunk8192-smoke32b.conf' <<'EOF2'
[program:sfs-serving-chunk8192-smoke32b]
command=/bin/bash /workspace/sfs/setup/serving-chunk8192-smoke32b.sh
directory=/workspace/sfs/fcfs-repo-v3
autostart=false
autorestart=false
startsecs=0
stdout_logfile=/workspace/sfs/setup/sfs-serving-chunk8192-smoke32b.log
redirect_stderr=true
EOF2
ssh sfs-vast 'supervisorctl reread && supervisorctl update && supervisorctl start sfs-serving-chunk8192-smoke32b'
ssh sfs-vast 'python3 -c "import json;s=json.load(open(\"/workspace/sfs/serving/chunk8192/state/smoke-32b-20260917/gpu-smoke.json\"));print(s[\"status\"],s[\"configuration_id\"],s[\"evidence_rule\"],s[\"evidence_checks\"],s[\"maximum_running\"],[r[\"usage\"][\"completion_tokens\"] for r in s[\"responses\"]])"'
```

Pass: `PASS_BOUNDED_GPU_SMOKE qwen-chunk8192 chunk_bound <n>0> ...` with all eight responses at
128 tokens, and `server_qwen3-32b.log` free of OOM/preemption at `gpu_memory_utilization 0.9`
with `max_num_batched_tokens=8192` in its engine config line. (Edit `--gpus` to the free pair;
`--indices 0,1,2 --gpus a,b,c,d` smokes all three models on four GPUs in one go, about 12 min.)

## 4. Pass A, trace collection (4 GPUs, about 12 min)

```bash
ssh sfs-vast 'cat > /workspace/sfs/setup/serving-chunk8192-calibrate.sh' <<'EOF2'
#!/usr/bin/env bash
set -euo pipefail
export SFS_STORAGE=/workspace/sfs
source /workspace/sfs/fcfs-repo-v3/scripts/cloud/env.sh
cd /workspace/sfs/fcfs-repo-v3/src
exec taskset -c 0-47 python -m scripts.cloud.worker calibrate --profile chunk8192 --family qwen --variant canonical \  --bundle /workspace/sfs/bundle --models /workspace/sfs/models.json \  --state /workspace/sfs/serving/chunk8192/state --output /workspace/sfs/serving/chunk8192/state/calibrate-20260917 \  --campaign /workspace/sfs/fcfs-repo-v3/scripts/cloud/fcfs/campaign-chunk8192-20260917.json \  --gpus 0,1,2,3 --cpus $(seq -s, 0 47)
EOF2
ssh sfs-vast 'cat > /etc/supervisor/conf.d/sfs-serving-chunk8192-calibrate.conf' <<'EOF2'
[program:sfs-serving-chunk8192-calibrate]
command=/bin/bash /workspace/sfs/setup/serving-chunk8192-calibrate.sh
directory=/workspace/sfs/fcfs-repo-v3
autostart=false
autorestart=false
startsecs=0
stdout_logfile=/workspace/sfs/setup/sfs-serving-chunk8192-calibrate.log
redirect_stderr=true
EOF2
ssh sfs-vast 'supervisorctl reread && supervisorctl update && supervisorctl start sfs-serving-chunk8192-calibrate'
```

Servers start with placeholder canonical coefficients (`instances.json` `coefficient_status
CANONICAL_PLACEHOLDER_FOR_TRACE_COLLECTION_ONLY`, `configuration_id qwen-chunk8192`). Output:
`calibration_trace_<model>.csv`, `model_metrics.json`, `timing_models/`, `calibration.json`
(`TRACES_COLLECTED_COEFFICIENT_FIT_REQUIRED`). Nothing else runs in this mode. (Lane B, GPUs 4-7 /
CPUs 48-95, works the same way: change `--gpus`, `--cpus` and `taskset`.)

## 5. CPU refit of the SFS coefficients (minutes)

```bash
ssh sfs-vast 'export SFS_STORAGE=/workspace/sfs; source /workspace/sfs/fcfs-repo-v3/scripts/cloud/env.sh; cd /workspace/sfs/fcfs-repo-v3/src;
  taskset -c 84-95 python -m scripts.cloud.serving.coefficients fit --profile chunk8192 \    --calibration /workspace/sfs/serving/chunk8192/state/calibrate-20260917 \    --output /workspace/sfs/serving/chunk8192/coefficients-20260917.json;
  sha256sum /workspace/sfs/serving/chunk8192/coefficients-20260917.json;
  python3 -c "import json;c=json.load(open(\"/workspace/sfs/serving/chunk8192/coefficients-20260917.json\"));print(c[\"configuration_id\"],c[\"status\"]);[print(m,{k:round(v,9) for k,v in r.items() if k.endswith(\"coeff\") or k==\"intercept\"},r[\"fit_rows\"],r[\"fit_inlier_rows\"],round(r[\"fit_prediction_diagnostics\"][\"r2_all_rows\"],4),r[\"fit_prediction_diagnostics\"][\"negative_nonempty_rows\"]) for m,r in c[\"models\"].items()]"'
```

R^2 >= 0.95 and nonnegativity are enforced. Record the comparison with the canonical
coefficients (`src/scripts/runs/qwen_baselines.py`) and the FCFS refit
(`/workspace/sfs/fcfs/coefficients-20260916.json`): a different prefill/sum coefficient is
expected (8192-token steps), a decode coefficient far from canonical is not.

## 6. Pass B, qualification with the fitted coefficients (4 GPUs, about 35-50 min)

```bash
ssh sfs-vast 'cat > /workspace/sfs/setup/serving-chunk8192-qualify.sh' <<'EOF2'
#!/usr/bin/env bash
set -euo pipefail
export SFS_STORAGE=/workspace/sfs
source /workspace/sfs/fcfs-repo-v3/scripts/cloud/env.sh
cd /workspace/sfs/fcfs-repo-v3/src
exec taskset -c 0-47 python -m scripts.cloud.worker qualify --profile chunk8192 --family qwen --variant canonical \  --bundle /workspace/sfs/bundle --models /workspace/sfs/models.json \  --state /workspace/sfs/serving/chunk8192/state --output /workspace/sfs/serving/chunk8192/state/qualify-20260917 \  --campaign /workspace/sfs/fcfs-repo-v3/scripts/cloud/fcfs/campaign-chunk8192-20260917.json \  --coefficients /workspace/sfs/serving/chunk8192/coefficients-20260917.json \  --gpus 0,1,2,3 --cpus $(seq -s, 0 47)
EOF2
ssh sfs-vast 'cat > /etc/supervisor/conf.d/sfs-serving-chunk8192-qualify.conf' <<'EOF2'
[program:sfs-serving-chunk8192-qualify]
command=/bin/bash /workspace/sfs/setup/serving-chunk8192-qualify.sh
directory=/workspace/sfs/fcfs-repo-v3
autostart=false
autorestart=false
startsecs=0
stdout_logfile=/workspace/sfs/setup/sfs-serving-chunk8192-qualify.log
redirect_stderr=true
EOF2
ssh sfs-vast 'supervisorctl reread && supervisorctl update && supervisorctl start sfs-serving-chunk8192-qualify'
```

Inside: fresh calibration (independent of pass A) and timing-head fit, the `hard`,
`mooncake_prefill` and `lmdeploy_proxy` smokes (192 calibration prompts at 2 QPS each),
shortest-queue load probes at 6 and 8.3 QPS, then `qualification.json`
(`GPU_MEASURED_REVIEW_REQUIRED`) and exit. Estimate from the FCFS qualification on this host:
calibration 7 min, 2-7 min per smoke, 7-8 min per load probe. Check the bound identity:

```bash
ssh sfs-vast 'O=/workspace/sfs/serving/chunk8192/state/qualify-20260917; cat $O/status.json; echo;
  python3 -c "import json;q=json.load(open(\"$O/qualification.json\"));print(q[\"configuration_id\"],q[\"coefficient_policy\"],q[\"campaign_kind\"],q[\"policy_smoke\"]);print(q[\"coefficients_sha256\"]);print(q[\"campaign_sha256\"]);print(q[\"remaining_length_rule\"]);print(q[\"serving_profile\"][\"max_num_batched_tokens\"],q[\"serving_coefficients\"]);print(q[\"load_probes\"])";
  python3 -c "import json;b=json.load(open(\"$O/instances.json\"));print(b[\"configuration_id\"],b[\"coefficient_status\"],[r[\"max_num_batched_tokens\"] for r in b[\"instances\"]]);r=b[\"remaining_length\"];print(r[\"rule\"],{m:x[\"rule\"] for m,x in r[\"models\"].items()})";
  for f in $O/server_argv_*.json; do echo "$(basename $f): batched $(python3 -c "import json;a=json.load(open(\"$f\"));print(a[a.index(\"--max-num-batched-tokens\")+1])") rule flags $(grep -c -- "--remaining-length-mode" $f || true) chunked $(grep -c -- "\"--enable-chunked-prefill\"" $f || true)"; done'
```

Expected: `qwen-chunk8192 refit serving_config ['hard', 'mooncake_prefill', 'lmdeploy_proxy']`,
the coefficient file sha256 from step 5, the overlay sha256 `126f20be...`, rule
`qwen3-0.6b=running_all_prompt_bin_q50` with 8B/32B `current`, `coefficient_status
FITTED_FOR_CONFIGURATION`, rows `[8192, 8192, 8192]`, every argv `batched 8192` and
`chunked 1`, rule flags `1` only for `server_argv_qwen3-0.6b.json`, `smoke/<policy>/point.json`
for the three policies, load probes at 6 and 8.3 QPS classified stable.

## 7. Reviews and release (CPU)

```bash
ssh sfs-vast 'export SFS_STORAGE=/workspace/sfs; source /workspace/sfs/fcfs-repo-v3/scripts/cloud/env.sh; cd /workspace/sfs/fcfs-repo-v3;
  Q=/workspace/sfs/serving/chunk8192/state/qualify-20260917;
  (cd src && python -m scripts.cloud.serving.coefficients validate --profile chunk8192 --coefficients /workspace/sfs/serving/chunk8192/coefficients-20260917.json --calibration $Q --output $Q/coefficient-review.json);
  python ops/cloud/review_baseline_timing.py --qualification $Q --output $Q/timing-review.json;
  python3 -c "import json;c=json.load(open(\"$Q/timing_models/methodology_calibration.json\"));print({k:c.get(k) for k in c if \"partial_prefill\" in k or \"chunk_token_range\" in k})";
  for P in hard mooncake_prefill lmdeploy_proxy; do python3 -c "import json;r=json.load(open(\"$Q/smoke/$P/point.json\"))[\"router\"][\"runs\"][0];print(\"$P\",r[\"summary\"],r.get(\"remaining_length_rule\"))"; done;
  grep -ciE "out of memory|preempt|cuda error" $Q/server_qwen3-*.log || true'
```

Checklist (record concrete numbers in the release text): independent post-calibration R^2 per
model >= 0.95 with no negative predictions and per-group (pure decode, prefill, mixed) MAE/bias;
TPOT-head MAE on independent smoke rows (`timing-review.json`); `partial_prefill_observed ==
true` and `chunk_token_range` up to 8192 in `methodology_calibration.json` (expected for this
configuration; the FCFS expectation was false); the three smokes `succeeded_requests == 192`,
`failed_requests == 0`, `remaining_length_rule` in each point equal to the pool rule; load probes
at 6 and 8.3 QPS stable with realized rates within 10%; SFS predicted-versus-observed TTFT on
`smoke/hard/predicted_waits_router_hard.log` with no systematic under-prediction on long
prompts (a 32k prompt now needs four steps); server logs free of OOM/preemption.

```bash
ssh sfs-vast 'export SFS_STORAGE=/workspace/sfs; source /workspace/sfs/fcfs-repo-v3/scripts/cloud/env.sh; cd /workspace/sfs/fcfs-repo-v3/src;
  python -m scripts.cloud.control release --qualification /workspace/sfs/serving/chunk8192/state/qualify-20260917 \    --timing-review "<independent R2 per model, decode/prefill/mixed MAE, TPOT head MAE, chunk coverage to 8192, partial prefill observed>" \    --load-review "<hard/mooncake/lmdeploy smokes 192/192 with attainment, realized 2 QPS, load probes 6/8.3 stable, 0.6B rule flags placement, hardware GPUs 0-3>"'
```

## 8. Run the 12 cells (`worker run`, same four GPUs, about 8-10 h)

`validate_release` compares `hardware.json` of the qualification with the live `nvidia-smi`
fingerprint exactly (host, boot id, GPU UUIDs, topology), so the run uses the same four GPUs as
the qualification; it also re-checks the overlay hash, coefficient file hash, configuration id,
rule and source per cell.

```bash
ssh sfs-vast 'cat > /workspace/sfs/setup/serving-chunk8192-run.sh' <<'EOF2'
#!/usr/bin/env bash
set -euo pipefail
export SFS_STORAGE=/workspace/sfs
source /workspace/sfs/fcfs-repo-v3/scripts/cloud/env.sh
cd /workspace/sfs/fcfs-repo-v3/src
exec taskset -c 0-47 python -m scripts.cloud.worker run --profile chunk8192 --family qwen --variant canonical \  --bundle /workspace/sfs/bundle --models /workspace/sfs/models.json \  --state /workspace/sfs/serving/chunk8192/state --output /workspace/sfs/serving/chunk8192/state/run-20260917 \  --campaign /workspace/sfs/fcfs-repo-v3/scripts/cloud/fcfs/campaign-chunk8192-20260917.json \  --coefficients /workspace/sfs/serving/chunk8192/coefficients-20260917.json \  --qualification /workspace/sfs/serving/chunk8192/state/qualify-20260917 \  --gpus 0,1,2,3 --cpus $(seq -s, 0 47) \  --cells qwen-chunk8192-hard-6,qwen-chunk8192-hard-7,qwen-chunk8192-hard-8,qwen-chunk8192-hard-8.3,qwen-chunk8192-mooncake_prefill-6,qwen-chunk8192-mooncake_prefill-7,qwen-chunk8192-mooncake_prefill-8,qwen-chunk8192-mooncake_prefill-8.3,qwen-chunk8192-lmdeploy_proxy-6,qwen-chunk8192-lmdeploy_proxy-7,qwen-chunk8192-lmdeploy_proxy-8,qwen-chunk8192-lmdeploy_proxy-8.3
EOF2
ssh sfs-vast 'cat > /etc/supervisor/conf.d/sfs-serving-chunk8192-run.conf' <<'EOF2'
[program:sfs-serving-chunk8192-run]
command=/bin/bash /workspace/sfs/setup/serving-chunk8192-run.sh
directory=/workspace/sfs/fcfs-repo-v3
autostart=false
autorestart=false
startsecs=0
stdout_logfile=/workspace/sfs/setup/sfs-serving-chunk8192-run.log
redirect_stderr=true
EOF2
ssh sfs-vast 'supervisorctl reread && supervisorctl update && supervisorctl start sfs-serving-chunk8192-run'
```

The run pool re-smokes the three policies, then measures the cells in overlay order (policy
major). A cell is skipped only if its ledger entry exists for this bundle under the current
source, so a restarted program resumes where it stopped. Every entry carries `configuration_id`,
`coefficient_policy`, `coefficients_sha256`, `campaign_sha256`, `campaign_kind`,
`remaining_length_rule` and the table sha256. Measured Qwen cells on this host took 39-46 min
under the canonical and FCFS configurations; the 8192-token step lengthens prefill-heavy phases,
so budget 45-50 min per cell (about 9-10 h for 12).

Progress, mirroring and collation:

```bash
ssh sfs-vast 'export SFS_STORAGE=/workspace/sfs; source /workspace/sfs/fcfs-repo-v3/scripts/cloud/env.sh; cd /workspace/sfs/fcfs-repo-v3/src;
  python -m scripts.cloud.control status --state /workspace/sfs/serving/chunk8192/state --bundle /workspace/sfs/bundle \    --campaign /workspace/sfs/fcfs-repo-v3/scripts/cloud/fcfs/campaign-chunk8192-20260917.json; cat /workspace/sfs/serving/chunk8192/state/run-20260917/phase.json'
python scripts/cloud/sync_results.py --host sfs-vast --remote-state /workspace/sfs/serving/chunk8192 \  --destination /ocean/projects/cis250162p/aparthas/sfs_cloud_results_20260916/serving/chunk8192 --watch
python -m scripts.cloud.collate --bundle "$SFS_BUNDLE" --raw-root <mirror>/serving/chunk8192/state/run-20260917 \  --output <results>/serving-chunk8192-20260917 --campaign scripts/cloud/fcfs/campaign-chunk8192-20260917.json --allow-partial
```

`audit.json` reports `PASS_12_CELLS` when complete and refuses a cell completed under another
overlay hash or rule. Afterwards copy `qualification.json`, `release.json`,
`coefficient-review.json`, `timing-review.json`, `hardware.json`, the coefficient file and
`smoke-32b-20260917/gpu-smoke.json` to `scripts/cloud/reports/serving-chunk8192-20260917/` and
record the twelve cells in the experiment ledger with the configuration id, coefficient hash and
rule. Flag any SFS OnTimeUtility below a baseline at the same rate.

## GPU time summary

| Step | GPUs | Estimate |
|---|---|---|
| Gates (CPU) | 0 | 20 min |
| 32B TP=2 bounded smoke | 2 | 10 min |
| Pass A calibrate | 4 | 12 min |
| Coefficient fit (CPU) | 0 | minutes |
| Pass B qualify (3 smokes, 2 probes) | 4 | 35-50 min |
| Reviews and release (CPU) | 0 | 15 min |
| Run 12 cells | 4 | 9-10 h (45-50 min per cell plus 10 min start and smokes) |
| Total | 4 | about 1.5 h qualification + 9-10 h measurement |

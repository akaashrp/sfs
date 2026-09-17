# SFS/SCORE and predictor-variant launch on Vast (17 September 2026)

Nothing here has been run. Source: `experiments/cloud-68-20260914` at `110798073d864fb9231a8878898dbf3d29145f84`
(vllm submodule `30ec5b4b5418ce429e5cac2693e8e34fe0e82520`, the fork's
`experiments/sfs-cloud-20260914` tip; no csrc/CMake/setup change since `28bbf92`, so the
extensions compiled in `/workspace/sfs/repo-v2` stay valid). Do not touch `/workspace/sfs/repo`
(da1c78b pools) or `/workspace/sfs/repo-v2` (bedeec4/80e2687 baseline workers).

What runs, in order:

1. `scripts/cloud/sfs-score-campaign-20260917.json` (kind `sfs_score`, 14 cells, 176,000 requests):
   Qwen `hard`/`score` at 6/7/8/8.3 QPS x 16,000 (ids `qwen-hard-6` ... `qwen-score-8.3`);
   Ministral `hard`/`score` at 6.0125/7.8625/8.7875 QPS x 8,000 (`ministral-hard-6.0125` ...).
2. `scripts/cloud/predictor-variants-campaign-20260917.json` (kind `predictor_variants`, 36 cells,
   576,000 requests): `{flash_quality, mlp_quality, mlp_length} x {hard, score, latency_agnostic} x
   {6, 7, 8, 8.3}` x 16,000 (ids `qwen-<variant>-<policy>-<qps>`), one pool per variant.

Both overlays carry the same `remaining_length` block: in every Qwen pool the `qwen3-0.6b` engine
runs `--remaining-length-mode running_all --remaining-length-table qwen3-0.6b.json
--remaining-length-quantile 0.5 --remaining-length-conditioning prompt_bin` and the router attaches
that table for its missing-prediction fill on that instance; `qwen3-8b`, `qwen3-32b` and every
Ministral engine keep the current rule (the Ministral pool refuses any rule). Table hashes and the
rule are bound into `qualification.json`, every completed-ledger entry and `instances.json`;
`validate_release` refuses a pool whose rule differs from its qualification.

## 1. Checkout `/workspace/sfs/repo-v3` (copy keeps the compiled extensions)

```bash
ssh sfs-vast 'test -d /workspace/sfs/repo-v3 || cp -a /workspace/sfs/repo-v2 /workspace/sfs/repo-v3'
ssh sfs-vast 'cd /workspace/sfs/repo-v3 && git fetch origin experiments/cloud-68-20260914 \
  && git checkout -q 110798073d864fb9231a8878898dbf3d29145f84 && git -C vllm fetch origin experiments/sfs-cloud-20260914 \
  && git -C vllm checkout -q 30ec5b4b5418ce429e5cac2693e8e34fe0e82520 \
  && git submodule status && git status --short | grep -v "^??" ; \
  git -C vllm diff --stat 28bbf92 HEAD -- csrc CMakeLists.txt cmake setup.py; \
  ls -la vllm/vllm/_C.abi3.so vllm/vllm/v1/engine/_scheduler_sim*.so'
```

Expected: `git submodule status` shows ` 30ec5b4b5...` with no `+`/`-` prefix, no tracked
modifications, an empty build-file diff, and the extension files present. Tables come with the
checkout; verify them (these are the sha256s the overlays bind):

```bash
ssh sfs-vast 'sha256sum /workspace/sfs/repo-v3/scripts/cloud/remaining-length-tables-20260917/*'
# 2eac8a79de5650982d998c7ded87f8cf48e19f3a53d1005afb734d7c4c491fbb  manifest.json
# 50a14ca76681895a622cbf2d69ba362e37792b32b76c5967b7b9b67f3f6ed87d  qwen3-0.6b.json
# 74e3a4ef30c8be24e01983b2609c4d76c8641422f26def2a899ec47ec88cf599  qwen3-32b.json
# 2392950a25c04003e945124554fd33654da1c552ccdecf7e59d20770f9c04ce6  qwen3-8b.json
```

If the checkout somehow lacks them (partial clone), stream them from the controller:

```bash
cd /ocean/projects/cis250162p/aparthas/sfs_cloud_20260914
tar -cf - scripts/cloud/remaining-length-tables-20260917 | ssh sfs-vast 'tar -xf - -C /workspace/sfs/repo-v3'
```

## 2. Gates (CPU only; separate state root so nothing under `/workspace/sfs/setup` or `/workspace/sfs/state` changes)

The worker reads its gates from `<state>/../setup`; use `/workspace/sfs/state-v3/{setup,state}`.
The GPU-UUID locks in `/dev/shm` are shared across checkouts, so pools cannot overlap with
`repo`/`repo-v2` pools whichever state root they use.

```bash
ssh sfs-vast 'export SFS_STORAGE=/workspace/sfs; source /workspace/sfs/repo-v3/scripts/cloud/env.sh;
  export SFS_TEST_GO="$CONDA_PREFIX/bin/go"; cd /workspace/sfs/repo-v3;
  mkdir -p /workspace/sfs/state-v3/setup /workspace/sfs/state-v3/state /workspace/sfs/state-v3/runs;
  taskset -c 84-95 bash scripts/cloud/test.sh /workspace/sfs/state-v3/setup/tests &&
  taskset -c 84-95 python -m pytest -q src/sfs_core/routing/tests/test_remaining_length_fallback.py src/scripts/runs/tests/test_arrival_schedule.py &&
  (cd vllm && taskset -c 84-95 python -m pytest -q tests/v1/core/test_scheduler_remaining_length.py tests/v1/core/test_snapshot_serialization.py) &&
  taskset -c 84-95 python -m scripts.cloud.prepare cpu --bundle /workspace/sfs/bundle --output /workspace/sfs/state-v3/setup/cpu-inputs.json &&
  taskset -c 84-95 python -m scripts.cloud.prepare serving --bundle /workspace/sfs/bundle --output /workspace/sfs/state-v3/setup/cpu-serving.json &&
  python -m scripts.cloud.control status --state /workspace/sfs/state-v3/state --bundle /workspace/sfs/bundle \
    --campaign /workspace/sfs/repo-v3/scripts/cloud/sfs-score-campaign-20260917.json'
```

`test.sh` must report at least 70 tests, none skipped (the controller run of the same list plus
the reserve tests passed 239 on the controller with this source; none skipped). The status call also validates the overlay
against the tables on the host (it fails on a hash mismatch) and should list 14 remaining cells.

## 3. Launch the SFS/SCORE overlay (one Supervisor program per pool)

Slot A: Qwen on GPUs 0-3, CPUs 0-47. Slot B: Ministral on GPUs 4,5,6, CPUs 48-95 (GPU 7 stays
free). Before starting, `nvidia-smi --query-gpu=index,memory.used --format=csv` must read 0 MiB on
the slot's GPUs and no other `sfs-*` program may be RUNNING on them.

```bash
ssh sfs-vast 'cat > /workspace/sfs/setup/sfs-score-qwen.sh' <<'EOF2'
#!/usr/bin/env bash
set -euo pipefail
export SFS_STORAGE=/workspace/sfs
source /workspace/sfs/repo-v3/scripts/cloud/env.sh
cd /workspace/sfs/repo-v3/src
exec taskset -c 0-47 python -m scripts.cloud.worker campaign \
  --campaign /workspace/sfs/repo-v3/scripts/cloud/sfs-score-campaign-20260917.json \
  --bundle /workspace/sfs/bundle --models /workspace/sfs/models.json \
  --state /workspace/sfs/state-v3/state --output /workspace/sfs/state-v3/runs/sfs-score-qwen \
  --family qwen --variant canonical --gpus 0,1,2,3 \
  --cells qwen-hard-6,qwen-hard-7,qwen-hard-8,qwen-hard-8.3,qwen-score-6,qwen-score-7,qwen-score-8,qwen-score-8.3
EOF2
ssh sfs-vast 'cat > /workspace/sfs/setup/sfs-score-ministral.sh' <<'EOF2'
#!/usr/bin/env bash
set -euo pipefail
export SFS_STORAGE=/workspace/sfs
source /workspace/sfs/repo-v3/scripts/cloud/env.sh
cd /workspace/sfs/repo-v3/src
exec taskset -c 48-95 python -m scripts.cloud.worker campaign \
  --campaign /workspace/sfs/repo-v3/scripts/cloud/sfs-score-campaign-20260917.json \
  --bundle /workspace/sfs/bundle --models /workspace/sfs/models.json \
  --state /workspace/sfs/state-v3/state --output /workspace/sfs/state-v3/runs/sfs-score-ministral \
  --family ministral --variant canonical --gpus 4,5,6 \
  --cells ministral-hard-6.0125,ministral-hard-7.8625,ministral-hard-8.7875,ministral-score-6.0125,ministral-score-7.8625,ministral-score-8.7875
EOF2
for name in sfs-score-qwen sfs-score-ministral; do
ssh sfs-vast "cat > /etc/supervisor/conf.d/$name.conf" <<EOF2
[program:$name]
command=/bin/bash /workspace/sfs/setup/$name.sh
directory=/workspace/sfs/repo-v3
autostart=false
autorestart=false
startsecs=0
stdout_logfile=/workspace/sfs/setup/$name.log
redirect_stderr=true
EOF2
done
ssh sfs-vast 'supervisorctl reread && supervisorctl update && supervisorctl start sfs-score-qwen sfs-score-ministral'
```

The output directory must not exist beforehand (the worker creates it). `--cells` lists every
cell of the family so the worker runs them all; drop ids to split a family across attempts. The
worker skips `qwen-hard-8` only if `/workspace/sfs/state-v3/state/completed/qwen-hard-8.json`
exists with an accepted source pin; nothing writes that entry automatically (the two 8 QPS lane
controls ran with the current rule and the pre-thread arrival producer, see the overlay's `reuse`
note), so by default the cell is measured like the others.

Each pool calibrates, smokes `hard` then `score` (192 calibration prompts; Qwen at 2 QPS,
Ministral at 8.7875 QPS), writes `qualification.json` (`GPU_MEASURED_REVIEW_REQUIRED`) and then
waits for `release.json`. Check the flag placement before releasing:

```bash
ssh sfs-vast 'for O in /workspace/sfs/state-v3/runs/sfs-score-qwen /workspace/sfs/state-v3/runs/sfs-score-ministral; do
  echo "== $O"; cat $O/phase.json;
  python3 -c "import json;q=json.load(open(\"$O/qualification.json\"));print(q[\"campaign_kind\"],q[\"policy_smoke\"],q[\"remaining_length_rule\"]);print(q[\"remaining_length\"])";
  python3 -c "import json;b=json.load(open(\"$O/instances.json\"))[\"remaining_length\"];print(b[\"rule\"],{m:r[\"rule\"] for m,r in b[\"models\"].items()})";
  for f in $O/server_argv_*.json; do echo -n "$(basename $f): "; grep -c -- "--remaining-length-mode" $f || true; done; done'
```

Expected for Qwen: `policy_smoke == ["hard", "score"]`, rule
`qwen3-0.6b=running_all_prompt_bin_q50` with `{'qwen3-0.6b': 'running_all_prompt_bin_q50',
'qwen3-8b': 'current', 'qwen3-32b': 'current'}`, table sha256
`50a14ca7...` under `remaining_length.models.qwen3-0.6b`, `--remaining-length-mode` count 1 for
`server_argv_qwen3-0.6b.json` and 0 for the other two, `smoke/hard/point.json` and
`smoke/score/point.json` present. Expected for Ministral: rule `current` everywhere and no
`--remaining-length-*` flags. Then review calibration residuals, smoke audits and server logs as
the README requires and release (per pool; the qualification directory is the worker output):

```bash
ssh sfs-vast 'export SFS_STORAGE=/workspace/sfs; source /workspace/sfs/repo-v3/scripts/cloud/env.sh; cd /workspace/sfs/repo-v3/src;
  python -m scripts.cloud.control release --qualification /workspace/sfs/state-v3/runs/sfs-score-qwen \
    --timing-review "<concrete assessment of the destination timing residuals and coverage>" \
    --load-review "<concrete assessment of the smoke arrivals, backlog and hardware comparability>"'
```

The worker then measures the cells in overlay order; every completed entry carries
`campaign_sha256`, `campaign_kind`, `remaining_length_rule` and the per-model table sha256s, and
the router must have recorded the same rule in the point (`router.runs[0].remaining_length_rule`).
Progress:

```bash
ssh sfs-vast 'export SFS_STORAGE=/workspace/sfs; source /workspace/sfs/repo-v3/scripts/cloud/env.sh; cd /workspace/sfs/repo-v3/src;
  python -m scripts.cloud.control status --state /workspace/sfs/state-v3/state --bundle /workspace/sfs/bundle \
    --campaign /workspace/sfs/repo-v3/scripts/cloud/sfs-score-campaign-20260917.json'
```

Mirror the new state root with a second watcher on the controller so raw points reach
`sfs_cloud_results_20260916` (the existing watcher only pulls `/workspace/sfs/state`):

```bash
python scripts/cloud/sync_results.py --host sfs-vast --remote-state /workspace/sfs/state-v3 \
  --destination /ocean/projects/cis250162p/aparthas/sfs_cloud_results_20260916/state-v3 --watch
```

Collate on the controller from the mirrored tree (or on Vast):

```bash
python -m scripts.cloud.collate --bundle "$SFS_BUNDLE" --raw-root <mirror>/state-v3/runs \
  --output <results>/sfs-score-20260917 --campaign scripts/cloud/sfs-score-campaign-20260917.json --allow-partial
```

`audit.json` reports `PASS_14_CELLS` when complete and refuses a cell completed under another
overlay hash or rule.

## 4. Predictor-variant overlay, one pool per variant

Order: `mlp_length` first (the only variant that changes the servers), then `mlp_quality`, then
`flash_quality`; policies within a pool run hard, score, latency_agnostic at ascending rates.
Slot A takes the next variant when the Qwen SFS/SCORE pool finishes; slot B (GPUs 4-7, CPUs
48-95) takes another variant once the Ministral pool finishes. Never two variants in one slot.
Each variant is qualified separately (`--variant` is per pool: fresh calibration, three policy
smokes, review, release).

```bash
V=mlp_length   # then mlp_quality, flash_quality
ssh sfs-vast "cat > /workspace/sfs/setup/sfs-variant-$V.sh" <<EOF2
#!/usr/bin/env bash
set -euo pipefail
export SFS_STORAGE=/workspace/sfs
source /workspace/sfs/repo-v3/scripts/cloud/env.sh
cd /workspace/sfs/repo-v3/src
exec taskset -c 0-47 python -m scripts.cloud.worker campaign \\
  --campaign /workspace/sfs/repo-v3/scripts/cloud/predictor-variants-campaign-20260917.json \\
  --bundle /workspace/sfs/bundle --models /workspace/sfs/models.json \\
  --state /workspace/sfs/state-v3/state --output /workspace/sfs/state-v3/runs/variant-$V \\
  --family qwen --variant $V --gpus 0,1,2,3 \\
  --cells qwen-$V-hard-6,qwen-$V-hard-7,qwen-$V-hard-8,qwen-$V-hard-8.3,qwen-$V-score-6,qwen-$V-score-7,qwen-$V-score-8,qwen-$V-score-8.3,qwen-$V-latency_agnostic-6,qwen-$V-latency_agnostic-7,qwen-$V-latency_agnostic-8,qwen-$V-latency_agnostic-8.3
EOF2
ssh sfs-vast "cat > /etc/supervisor/conf.d/sfs-variant-$V.conf" <<EOF2
[program:sfs-variant-$V]
command=/bin/bash /workspace/sfs/setup/sfs-variant-$V.sh
directory=/workspace/sfs/repo-v3
autostart=false
autorestart=false
startsecs=0
stdout_logfile=/workspace/sfs/setup/sfs-variant-$V.log
redirect_stderr=true
EOF2
ssh sfs-vast "supervisorctl reread && supervisorctl update && supervisorctl start sfs-variant-$V"
```

For slot B use `taskset -c 48-95` and `--gpus 4,5,6,7`. Verify before release exactly as in
section 3 (`policy_smoke == ["hard", "score", "latency_agnostic"]`; the 0.6B rule is on in every
Qwen pool, including for the latency_agnostic cells, and is recorded per cell); for `mlp_length`
every `server_argv_*.json` must carry `--output-length-model-path .../variants/mlp_length`, for the
quality variants the server argv must equal the canonical pool's. Release with `control release
--qualification /workspace/sfs/state-v3/runs/variant-$V`. Status with `--campaign
.../predictor-variants-campaign-20260917.json` (36 expected cells).

Collate the variant tranche on the controller, where the April Bridges points are readable, so the
latency_agnostic comparator and the round-robin/shortest-queue reuse rows get their OnTimeUtility
recomputed on the observed judge cohort (on Vast those rows keep the inventory attainment only):

```bash
python -m scripts.cloud.collate --bundle "$SFS_BUNDLE" --raw-root <mirror>/state-v3/runs \
  --output <results>/variants-20260917 --campaign scripts/cloud/predictor-variants-campaign-20260917.json \
  --reuse-root <results>/sfs-score-20260917 --reuse-root <results>/baseline-20260916 \
  --bridges-reference scripts/cloud/reports/baseline-campaign-20260916/qwen-reference-inventory.json --allow-partial
```

`variant_matrix.json` then holds one row per (variant, policy, qps): `measured` ablation cells,
`reused_canonical` rows for the six no-op policies (provider `vast` from the reused collations,
`bridges` for round robin / shortest queue) and the canonical comparators (hard/score from the
SFS/SCORE collation, latency_agnostic from Bridges, labelled `provider=bridges` with the point
path and sha256); `flash_quality` rows and their comparators report the Flash judge as primary.

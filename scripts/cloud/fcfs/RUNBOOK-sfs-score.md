# FCFS unchunked SFS/SCORE lanes on Vast (17 September 2026)

Nothing here has been run. Overlay: `scripts/cloud/fcfs/campaign-sfs-score-20260917.json`
(kind `fcfs_sfs_score`, sha256 `e2781d05f57fc5ab6e174425857f2638ec2a465dd38f668de1b933dd38df4490`,
8 cells x 16,000 requests): `qwen-fcfs-unchunked-65536-{hard,score}-{6,7,8,8.3}` on the
`qwen-fcfs-unchunked-65536` configuration with the 0.6B remaining-length rule
(`qwen3-0.6b=running_all_prompt_bin_q50`, table sha256 `50a14ca7...`; 8B/32B on the current
rule). Authorization recorded in the overlay: "User authorized on 17 September 2026: unchunked
SFS and SCORE cells run before the predictor ablations, with the 0.6B remaining-length rule".

Source: branch `experiments/qwen-fcfs-unchunked-20260916` at `e8441a90da4cd4524ac447d74e1dcef34abb094b` (merge of
`experiments/cloud-68-20260914` eef1791 plus the FCFS SFS/SCORE overlay; vllm submodule
`30ec5b4b5418ce429e5cac2693e8e34fe0e82520`, no csrc/CMake/setup change since `28bbf92`, so the
extensions compiled in `/workspace/sfs/fcfs-repo` stay valid). The runbook commit that follows
only adds this file and does not change the source hash the gates bind.

Order on the host (user, 2026-09-17): canonical SFS/SCORE (`repo-v3`, `state-v3`) first, then
these two lanes, then the predictor ablations. Do not touch `/workspace/sfs/repo`, `repo-v2`,
`repo-v3` or `reserve-repo`, and do not stop any running Supervisor program. The 28-cell FCFS
matrix lanes keep running from `/workspace/sfs/fcfs-repo` (a707597); this work uses a second
FCFS checkout and the same FCFS state root, so the completion ledger stays in one place.

What differs from the matrix lanes: the overlay carries `remaining_length`, the worker under
`--profile fcfs` passes the rule to the pool, and only `server_argv_qwen3-0.6b.json` gains
`--remaining-length-mode running_all --remaining-length-table .../qwen3-0.6b.json
--remaining-length-quantile 0.5 --remaining-length-conditioning prompt_bin`. `qualification.json`
and every completed-ledger entry record `configuration_id`, `coefficients_sha256`,
`campaign_sha256`, `campaign_kind`, `remaining_length_rule` and the per-model table sha256;
`validate_release` refuses a run whose configuration, coefficient file, overlay, rule, source,
bundle or hardware differs from its lane's qualification.

## 1. Checkout `/workspace/sfs/fcfs-repo-v2` (copy keeps the compiled extensions)

```bash
ssh sfs-vast 'test -d /workspace/sfs/fcfs-repo-v2 || cp -a /workspace/sfs/fcfs-repo /workspace/sfs/fcfs-repo-v2'
ssh sfs-vast 'cd /workspace/sfs/fcfs-repo-v2 && git fetch origin experiments/qwen-fcfs-unchunked-20260916 \
  && git checkout -q e8441a90da4cd4524ac447d74e1dcef34abb094b \
  && git -C vllm fetch origin experiments/sfs-cloud-20260914 \
  && git submodule update --init vllm && git submodule status \
  && git status --short | grep -v "^??"; \
  git -C vllm diff --stat 28bbf92 HEAD -- csrc CMakeLists.txt cmake setup.py; \
  ls -la vllm/vllm/_C.abi3.so vllm/vllm/v1/engine/_scheduler_sim*.so; \
  sha256sum scripts/cloud/fcfs/campaign-sfs-score-20260917.json scripts/cloud/remaining-length-tables-20260917/*'
```

Expected: `git submodule status` shows ` 30ec5b4b5...` with no `+`/`-` prefix, no tracked
modifications, an empty build-file diff, the extension files present, overlay sha256
`e2781d05...`, and the tables:

```
2eac8a79de5650982d998c7ded87f8cf48e19f3a53d1005afb734d7c4c491fbb  manifest.json
50a14ca76681895a622cbf2d69ba362e37792b32b76c5967b7b9b67f3f6ed87d  qwen3-0.6b.json
74e3a4ef30c8be24e01983b2609c4d76c8641422f26def2a899ec47ec88cf599  qwen3-32b.json
2392950a25c04003e945124554fd33654da1c552ccdecf7e59d20770f9c04ce6  qwen3-8b.json
```

The fitted FCFS coefficients stay `/workspace/sfs/fcfs/coefficients-20260916.json` (sha256
`8d4ded024e2f4bd13a789c20d8cf332f1c0fa37c10af0256174b96065247c8a2`, the file both matrix
qualifications bound); verify before anything else:

```bash
ssh sfs-vast 'sha256sum /workspace/sfs/fcfs/coefficients-20260916.json'
```

## 2. Gates into `/workspace/sfs/fcfs/setup` (CPU only; back up the a707597 gates first)

The worker reads gates from `<state>/../setup`, so the FCFS state root `/workspace/sfs/fcfs/state`
resolves `/workspace/sfs/fcfs/setup`. The matrix lanes checked those gates when their pools
started and re-check only their own qualification directory per cell, so replacing the gate
files does not disturb them; the backup lets a matrix lane be resumed from `fcfs-repo` later
(its source hash differs from `fcfs-repo-v2`, so it would need the a707597 gates back).

```bash
ssh sfs-vast 'cp -a /workspace/sfs/fcfs/setup /workspace/sfs/fcfs/setup-a707597-$(date +%Y%m%d%H%M) && ls -d /workspace/sfs/fcfs/setup-a707597-*'
ssh sfs-vast 'cat > /workspace/sfs/setup/fcfs-sfs-score-gates.sh' <<'EOF2'
#!/usr/bin/env bash
set -euo pipefail
export SFS_STORAGE=/workspace/sfs
source /workspace/sfs/fcfs-repo-v2/scripts/cloud/env.sh
export SFS_TEST_GO="$CONDA_PREFIX/bin/go"
cd /workspace/sfs/fcfs-repo-v2
rm -rf /workspace/sfs/fcfs/setup/tests
taskset -c 84-95 bash scripts/cloud/test.sh /workspace/sfs/fcfs/setup/tests
taskset -c 84-95 python -m pytest -q -p no:cacheprovider src/sfs_core/routing/tests/test_remaining_length_fallback.py src/scripts/runs/tests/test_arrival_schedule.py
(cd vllm && taskset -c 84-95 python -m pytest -q -p no:cacheprovider tests/v1/core/test_scheduler_remaining_length.py tests/v1/core/test_snapshot_serialization.py)
taskset -c 84-95 python -m scripts.cloud.prepare cpu     --bundle /workspace/sfs/bundle --output /workspace/sfs/fcfs/setup/cpu-inputs.json
taskset -c 84-95 python -m scripts.cloud.prepare serving --bundle /workspace/sfs/bundle --output /workspace/sfs/fcfs/setup/cpu-serving.json
python3 -c "import json;g=json.load(open('/workspace/sfs/fcfs/setup/tests/gate.json'));print(g['status'],g['tests'])"
python3 -c "import json;a=json.load(open('/workspace/sfs/fcfs/setup/fcfs-inputs.json'));print(a['status'])"
python -m scripts.cloud.control status --state /workspace/sfs/fcfs/state --bundle /workspace/sfs/bundle \
  --campaign /workspace/sfs/fcfs-repo-v2/scripts/cloud/fcfs/campaign-sfs-score-20260917.json
EOF2
ssh sfs-vast 'cat > /etc/supervisor/conf.d/sfs-fcfs-sfs-score-gates.conf' <<'EOF2'
[program:sfs-fcfs-sfs-score-gates]
command=/bin/bash /workspace/sfs/setup/fcfs-sfs-score-gates.sh
directory=/workspace/sfs/fcfs-repo-v2
autostart=false
autorestart=false
startsecs=0
stdout_logfile=/workspace/sfs/setup/sfs-fcfs-sfs-score-gates.log
redirect_stderr=true
EOF2
ssh sfs-vast 'supervisorctl reread && supervisorctl update && supervisorctl start sfs-fcfs-sfs-score-gates'
```

`prepare cpu`/`serving` overwrite the two gate files in place (the outputs are bound to the
new source hash). `fcfs-inputs.json` (the actual chat admission audit) binds the profile and the
bundle, not the source, and is kept. Expected at the end of the log: `PASS_CPU_REGRESSION` with at
least 263 tests (the controller ran 263 on this source, none skipped), 13 + 8 extra tests passed,
`PASS_ACTUAL_CHAT_INPUTS`, and a status listing with `expected: 8`, `campaign_kind:
fcfs_sfs_score` and all eight ids remaining. (`control status` now inspects FCFS overlays; the
28 matrix cells appear under `completed_outside_campaign` as they finish.)

## 3. Qualify per lane (4 GPUs each, about 45-60 min per lane)

Lane A is GPUs 0-3 / CPUs 0-47, lane B is GPUs 4-7 / CPUs 48-95. Before starting a lane,
`nvidia-smi --query-gpu=index,memory.used --format=csv` must read 0 MiB on its four GPUs and no
`sfs-*` program may be RUNNING on them (`supervisorctl status`); the GPU-UUID locks in
`/dev/shm/sfs-cloud-locks-0` make an overlap fail fast rather than share. The output directory
must not exist beforehand. GPU 7 is in lane B here (the debugging reservation applied to the
matrix lanes only); if the user still wants GPU 7 free, lane B cannot run and both halves go
through lane A in sequence.

```bash
for lane in a b; do
  if [ $lane = a ]; then gpus=0,1,2,3; cpus=0-47; else gpus=4,5,6,7; cpus=48-95; fi
ssh sfs-vast "cat > /workspace/sfs/setup/fcfs-sfs-score-qualify-$lane.sh" <<EOF2
#!/usr/bin/env bash
set -euo pipefail
export SFS_STORAGE=/workspace/sfs
source /workspace/sfs/fcfs-repo-v2/scripts/cloud/env.sh
cd /workspace/sfs/fcfs-repo-v2/src
exec taskset -c $cpus python -m scripts.cloud.worker qualify --profile fcfs --family qwen --variant canonical \\
  --bundle /workspace/sfs/bundle --models /workspace/sfs/models.json \\
  --state /workspace/sfs/fcfs/state --output /workspace/sfs/fcfs/state/qualify-sfs-score-lane-$lane \\
  --campaign /workspace/sfs/fcfs-repo-v2/scripts/cloud/fcfs/campaign-sfs-score-20260917.json \\
  --coefficients /workspace/sfs/fcfs/coefficients-20260916.json \\
  --gpus $gpus --cpus \$(seq -s, ${cpus/-/ })
EOF2
ssh sfs-vast "cat > /etc/supervisor/conf.d/sfs-fcfs-sfs-score-qualify-$lane.conf" <<EOF2
[program:sfs-fcfs-sfs-score-qualify-$lane]
command=/bin/bash /workspace/sfs/setup/fcfs-sfs-score-qualify-$lane.sh
directory=/workspace/sfs/fcfs-repo-v2
autostart=false
autorestart=false
startsecs=0
stdout_logfile=/workspace/sfs/setup/sfs-fcfs-sfs-score-qualify-$lane.log
redirect_stderr=true
EOF2
done
ssh sfs-vast 'supervisorctl reread && supervisorctl update'
ssh sfs-vast 'supervisorctl start sfs-fcfs-sfs-score-qualify-a'    # when GPUs 0-3 are free
ssh sfs-vast 'supervisorctl start sfs-fcfs-sfs-score-qualify-b'    # when GPUs 4-7 are free
```

(`--cpus` is the explicit comma list the worker pins with `sched_setaffinity`; `${cpus/-/ }`
expands on the controller to `0 47` or `48 95`, so the generated script carries
`$(seq -s, 0 47)`. Check the script with `cat` before starting the program.)

Inside each qualification: fresh calibration with the fitted coefficients, timing-head fit, the
`hard` and `score` smokes (192 calibration prompts at 2 QPS), shortest-queue load probes at 6 and
8.3 QPS, then `qualification.json` (`GPU_MEASURED_REVIEW_REQUIRED`) and exit. Check the rule
placement and the bound identity per lane before reviewing:

```bash
ssh sfs-vast 'for L in a b; do O=/workspace/sfs/fcfs/state/qualify-sfs-score-lane-$L; test -d $O || continue; echo "== $O"; cat $O/status.json; echo;
  python3 -c "import json;q=json.load(open(\"$O/qualification.json\"));print(q[\"configuration_id\"],q[\"campaign_kind\"],q[\"policy_smoke\"]);print(q[\"coefficients_sha256\"]);print(q[\"campaign_sha256\"]);print(q[\"remaining_length_rule\"]);print(q[\"remaining_length\"]);print(q[\"load_probes\"])";
  python3 -c "import json;b=json.load(open(\"$O/instances.json\"));print(b[\"configuration_id\"],b[\"coefficient_status\"]);r=b[\"remaining_length\"];print(r[\"rule\"],{m:x[\"rule\"] for m,x in r[\"models\"].items()});print(r[\"models\"][\"qwen3-0.6b\"][\"table\"][\"sha256\"])";
  for f in $O/server_argv_*.json; do echo -n "$(basename $f): rule flags $(grep -c -- "--remaining-length-mode" $f || true), unchunked $(grep -c -- "--no-enable-chunked-prefill" $f || true), fcfs $(grep -c -- "--scheduling-policy" $f || true)"; echo; done; done'
```

Expected per lane: `configuration_id qwen-fcfs-unchunked-65536`, `campaign_kind fcfs_sfs_score`,
`policy_smoke ["hard", "score"]`, `coefficients_sha256 8d4ded02...`, `campaign_sha256 e2781d05...`,
rule `qwen3-0.6b=running_all_prompt_bin_q50` with `{'qwen3-0.6b': 'running_all_prompt_bin_q50',
'qwen3-8b': 'current', 'qwen3-32b': 'current'}`, table sha256 `50a14ca7...`,
`coefficient_status FITTED_FOR_CONFIGURATION`, rule flags `1` for `server_argv_qwen3-0.6b.json`
and `0` for the other two, unchunked and fcfs `1` for all three, `smoke/hard/point.json` and
`smoke/score/point.json` present, load probes at 6 and 8.3 QPS classified stable.

## 4. Review and release per lane

Same reviews as the matrix qualification (`scripts/cloud/fcfs/RUNBOOK.md` step 5), on this
lane's directory:

```bash
ssh sfs-vast 'export SFS_STORAGE=/workspace/sfs; source /workspace/sfs/fcfs-repo-v2/scripts/cloud/env.sh; cd /workspace/sfs/fcfs-repo-v2;
  for L in a b; do Q=/workspace/sfs/fcfs/state/qualify-sfs-score-lane-$L; test -f $Q/qualification.json || continue;
    python -m scripts.cloud.fcfs.coefficients validate --coefficients /workspace/sfs/fcfs/coefficients-20260916.json --calibration $Q --output $Q/coefficient-review.json;
    python ops/cloud/review_baseline_timing.py --qualification $Q --output $Q/timing-review.json;
    for P in hard score; do python3 -c "import json;r=json.load(open(\"$Q/smoke/$P/point.json\"))[\"router\"][\"runs\"][0];print(\"$L $P\",r[\"summary\"],r.get(\"remaining_length_rule\"))"; done;
    grep -ciE "out of memory|preempt|cuda error" $Q/server_qwen3-*.log || true; done'
```

Checklist: independent post-calibration R2 per model at least 0.95 with no negative predictions;
TPOT-head MAE on independent smoke rows; `partial_prefill_observed == false` in
`timing_models/methodology_calibration.json`; both smokes `succeeded_requests == 192`,
`failed_requests == 0`, and `remaining_length_rule` in the point equal to the pool rule; SFS
predicted-versus-observed TTFT on `smoke/hard/predicted_waits_router_hard.log` with no systematic
under-prediction on long prompts; server logs free of OOM/preemption. Then release each lane with
concrete numbers (the qualification directory is the worker output):

```bash
ssh sfs-vast 'export SFS_STORAGE=/workspace/sfs; source /workspace/sfs/fcfs-repo-v2/scripts/cloud/env.sh; cd /workspace/sfs/fcfs-repo-v2/src;
  python -m scripts.cloud.control release --qualification /workspace/sfs/fcfs/state/qualify-sfs-score-lane-a \
    --timing-review "<independent R2 per model, decode/prefill MAE, TPOT head MAE, prefill coverage>" \
    --load-review "<hard/score smoke 192/192 with attainment, realized 2 QPS, load probes 6/8.3 stable, 0.6B rule flags placement, hardware GPUs 0-3>"'
# lane B: --qualification /workspace/sfs/fcfs/state/qualify-sfs-score-lane-b with its own numbers (GPUs 4-7)
```

## 5. Run per lane (`worker run`, hardware match exact)

Lane A runs the four `hard` cells on GPUs 0-3; lane B the four `score` cells on GPUs 4-7. That
split keeps one policy per lane so a lane's calibration/smoke evidence matches every cell it
measures, and the two policies finish at about the same time (each measured FCFS cell took 39-46
min; four cells per lane is about 3 h). `validate_release` compares `hardware.json` of the
qualification with the live `nvidia-smi` fingerprint (host, boot id, GPU UUIDs, topology)
exactly, so a lane must run on the same four GPUs as its qualification and the host must not
reboot in between; if either changes, re-qualify that lane. It also re-checks the overlay hash,
coefficient file hash, configuration id, rule and source per cell.

```bash
for lane in a b; do
  if [ $lane = a ]; then gpus=0,1,2,3; cpus=0-47; policy=hard; else gpus=4,5,6,7; cpus=48-95; policy=score; fi
  cells=qwen-fcfs-unchunked-65536-$policy-6,qwen-fcfs-unchunked-65536-$policy-7,qwen-fcfs-unchunked-65536-$policy-8,qwen-fcfs-unchunked-65536-$policy-8.3
ssh sfs-vast "cat > /workspace/sfs/setup/fcfs-sfs-score-run-$lane.sh" <<EOF2
#!/usr/bin/env bash
set -euo pipefail
export SFS_STORAGE=/workspace/sfs
source /workspace/sfs/fcfs-repo-v2/scripts/cloud/env.sh
cd /workspace/sfs/fcfs-repo-v2/src
exec taskset -c $cpus python -m scripts.cloud.worker run --profile fcfs --family qwen --variant canonical \\
  --bundle /workspace/sfs/bundle --models /workspace/sfs/models.json \\
  --state /workspace/sfs/fcfs/state --output /workspace/sfs/fcfs/state/run-sfs-score-lane-$lane \\
  --campaign /workspace/sfs/fcfs-repo-v2/scripts/cloud/fcfs/campaign-sfs-score-20260917.json \\
  --coefficients /workspace/sfs/fcfs/coefficients-20260916.json \\
  --qualification /workspace/sfs/fcfs/state/qualify-sfs-score-lane-$lane \\
  --gpus $gpus --cpus \$(seq -s, ${cpus/-/ }) \\
  --cells $cells
EOF2
ssh sfs-vast "cat > /etc/supervisor/conf.d/sfs-fcfs-sfs-score-run-$lane.conf" <<EOF2
[program:sfs-fcfs-sfs-score-run-$lane]
command=/bin/bash /workspace/sfs/setup/fcfs-sfs-score-run-$lane.sh
directory=/workspace/sfs/fcfs-repo-v2
autostart=false
autorestart=false
startsecs=0
stdout_logfile=/workspace/sfs/setup/sfs-fcfs-sfs-score-run-$lane.log
redirect_stderr=true
EOF2
done
ssh sfs-vast 'supervisorctl reread && supervisorctl update'
ssh sfs-vast 'supervisorctl start sfs-fcfs-sfs-score-run-a'    # after lane A is released and GPUs 0-3 are free again
ssh sfs-vast 'supervisorctl start sfs-fcfs-sfs-score-run-b'    # after lane B is released and GPUs 4-7 are free again
```

A run pool starts new servers with the lane's qualified calibration (no new calibration),
re-smokes `hard` and `score` (192 calibration prompts each) on the same host, then measures
the cells in overlay order. Every completed entry
(`/workspace/sfs/fcfs/state/completed/<cell id>.json`) carries `configuration_id`,
`coefficients_sha256`, `campaign_sha256`, `campaign_kind`, `remaining_length_rule` and the table
sha256s, and the router must have recorded the same rule in the point. A cell is skipped only if
its ledger entry exists for this bundle under the current source (the overlay accepts no prior
source digests), so a restarted lane resumes where it stopped. If one lane must take all eight
cells, give it both policies in `--cells` (the lane's qualification smoked both).

Progress and mirroring:

```bash
ssh sfs-vast 'export SFS_STORAGE=/workspace/sfs; source /workspace/sfs/fcfs-repo-v2/scripts/cloud/env.sh; cd /workspace/sfs/fcfs-repo-v2/src;
  python -m scripts.cloud.control status --state /workspace/sfs/fcfs/state --bundle /workspace/sfs/bundle \
    --campaign /workspace/sfs/fcfs-repo-v2/scripts/cloud/fcfs/campaign-sfs-score-20260917.json;
  for L in a b; do cat /workspace/sfs/fcfs/state/run-sfs-score-lane-$L/phase.json 2>/dev/null; echo; done'
# Controller: if no watcher already mirrors /workspace/sfs/fcfs, start one (check `pgrep -af sync_results`).
python scripts/cloud/sync_results.py --host sfs-vast --remote-state /workspace/sfs/fcfs \
  --destination /ocean/projects/cis250162p/aparthas/sfs_cloud_results_20260916/fcfs --watch
```

Collate on the controller from the two run directories only (the matrix lanes' `cells/` under
the same state root belong to the matrix overlay and would be rejected as unexpected cells):

```bash
python -m scripts.cloud.collate --bundle "$SFS_BUNDLE" \
  --raw-root <mirror>/fcfs/state/run-sfs-score-lane-a --raw-root <mirror>/fcfs/state/run-sfs-score-lane-b \
  --output <results>/fcfs-sfs-score-20260917 --campaign scripts/cloud/fcfs/campaign-sfs-score-20260917.json --allow-partial
```

`audit.json` reports `PASS_8_CELLS` when complete and refuses a cell completed under another
overlay hash or rule. Afterwards copy each lane's `qualification.json`, `release.json`,
`coefficient-review.json`, `timing-review.json` and `hardware.json` to
`scripts/cloud/reports/fcfs-sfs-score-20260917/lane-{a,b}/` and record the eight cells in the
experiment ledger with the configuration id and the rule.

## GPU time summary

| Step | GPUs | Estimate |
|---|---|---|
| Gates (CPU) | 0 | 20 min |
| Qualify lane A / lane B | 4 + 4 | 45-60 min each, in parallel when both slots are free |
| Run lane A (4 hard cells) / lane B (4 score cells) | 4 + 4 | about 3 h each (server start and the two smokes add about 10 min) |

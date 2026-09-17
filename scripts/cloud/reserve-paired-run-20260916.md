# Paired 8.6 QPS control with the per-model remaining-length rule (flag ON)

Readiness note for the paired run recommended in
`scripts/cloud/reports/reserve-tail-20260916/README.md` section 6. Nothing here has been
launched: GPUs 0-3 are busy with campaign cells and this run must wait until the Qwen pool
frees. The run is a standard `scripts.cloud.canonical_control` 8.6 QPS control (same 16,000
requests, SLOs, predictors, coefficients, hard policy) with one difference: the Qwen3-0.6B engine
starts with `--remaining-length-mode running_all --remaining-length-table qwen3-0.6b.json
--remaining-length-quantile 0.5 --remaining-length-conditioning prompt_bin`; the 8B and 32B
engines are unchanged (mode `off`, byte-identical argv); the router attaches the 0.6B table for its
missing-prediction fallback on that instance only.

## 1. What the flag changes (and what it does not)

Engine (`vllm/vllm/v1/core/sched/scheduler.py::_snapshot_output_target`, per engine = per model):

- `--remaining-length-mode off` (default): prediction + adaptive reserve, unchanged.
- `running_all`: every RUNNING request whose prefill is complete gets
  `Q_q(total length | conditioning, total > generated)` as its target (the README's
  `cal_prompt_q50` / `cal_model_q50` candidates).
- `exhausted_only`: the same, but only once prediction + reserve leaves <= 1 token (the README's
  `exhausted_only_*` candidates; available, not the default).
- `--remaining-length-conditioning prompt_bin` looks the prompt-length bin up first
  (`<128, 128-512, 512-2k, 2k-8k, 8k-32k, >=32k` tokens) and backs off to the model level when a cell
  has fewer than 8 survivors; `model` uses the model level only. `--remaining-length-quantile` is
  the survival quantile (0.5 for the paired run).
- Every target is floored at `generated + 1` and capped at `min(max_tokens, max_model_len - prompt)`;
  when no level has support the current target is kept. Waiting requests, running prefills, probes,
  the adaptive-reserve statistics, the predictor and the simulator are untouched.
- Provenance: `remaining_length_rule` (e.g. `running_all_prompt_bin_q50`),
  `remaining_length_quantile` and `remaining_length_table_sha256` travel in the `config` map of
  every published snapshot payload (absent when the mode is `off`, so the off-path payload is
  byte-identical; the fixed 112-byte SHM header is parsed by the native reader and cannot change
  without C++). They also appear in `instances.json` (`remaining_length` block, per model), in
  `provenance.json`, in the point's `config.instance_metadata.remaining_length` and in each run's
  `remaining_length_rule` (`qwen3-0.6b=running_all_prompt_bin_q50` for this run, `current` otherwise).

Router (`src/sfs_core/routing/wait_time_scheduler.py::_reserve_pending_dispatch`): for an instance
whose engine has a rule, a missing prediction (the `1.0` default; 5.6-5.7% of 0.6B rows) makes the
pending-dispatch overlay use the table's unconditional prompt-bin median (cap-aware) instead of one
token. Real predictions and instances without a rule are untouched; with everything off the default
stays `1.0`.

Tables: `scripts.prep.remaining_length_tables` fits one JSON table per model from the 30,000
calibration outputs only (prompt indices 0-2499 of the four buckets; the build asserts zero
(bucket, example_id) overlap with the 16,000 evaluation ids of `qwen/request_map.csv`). Bins,
support minimum (8) and quantile interpolation are those of `evaluate_tail.py`. The built artifact
and its sha256s are in section 3 (all three models are built; only the 0.6B table is used here).

## 2. How section 4 of the README shaped this design

- 4.1 (0.6B, measured on GPU 7): the current rule's wait-estimate MAE is 30.2 s with 73% of
  requests under-estimated by more than 1 s; the hindsight oracle is 6.5 s; every q50 conditional
  table lands within 1.5-3 s of that floor (model-only 7.9 s, prompt-bin 9.0 s), q65/q80 over-estimate
  by 8-26 s, and the `exhausted_only` forms reach about 10 s but leave 69% of the > 60 s requests
  under-estimated. Hence `running_all` at q = 0.5 on 0.6B, not exhausted-only q65: the default mode
  and quantile of this change follow that table. Prompt-bin and model-only are equivalent on 0.6B
  (the running set is almost all long govreport prompts, so the two tables coincide); prompt-bin is
  kept because it costs nothing extra (same file, same O(log n) lookup) and is the single mechanism
  the README asks for.
- 4.3 (8B at its production rate) and 4.2 (57-snapshot hindsight replay): the running-set targets
  are irrelevant to the 8B wait error (target component -0.7 s; every candidate within noise of the
  current rule, and in the hindsight replay every 8B candidate is worse than current through
  over-estimation); 32B is sub-second either way. Hence 8B and 32B stay `off`, which is why the rule
  is per model rather than pool-wide.
- 4.4: replacing waiting-request targets makes both runs worse (prediction + reserve over-covers
  waiting requests and masks a simulator under-estimate), so waiting requests keep prediction +
  reserve in every mode.
- 4.1 also shows 5.7% of the replayed 0.6B requests carried the `1.0` default prediction, which is
  why the router fallback (README recommendation 5) is part of the same flag rather than a separate
  arm; it matters for waiting requests, where the survival table does not apply.
- 4.1's decomposition (measured - oracle median -0.1 s, residual concentrated in the 10-60 s bin)
  bounds what any target rule can achieve, and 4.3's 29 s simulator miss under 8B overload is out of
  scope: the run is a test of the routing feedback loop that section 5 lists as "not established",
  judged on measured attainment, the 0.6B routing share and cost, not on estimate error.

## 3. Table artifact

Built on the controller from
`/ocean/projects/cis250162p/aparthas/vllm_utils/bucketed_prompt_outputs` against
`/ocean/projects/cis250162p/aparthas/sfs_cloud_artifacts_20260916/qwen/request_map.csv`
(sha256 `97c146f6328a776cf23ee6fde23dcdced3720e5f0916c4a5412c375e8d77a2bb`, identical to
`req_map_qps_seed69_holdout4000_n16000.csv`), 10,000 outputs per model, overlap 0:

```bash
cd /ocean/projects/cis250162p/aparthas/sfs_reserve_20260916
source ~/.bashrc; conda activate vllm; export PYTHONPATH=$PWD/src:$PWD/vllm
python -m scripts.prep.remaining_length_tables \
  --calibration /ocean/projects/cis250162p/aparthas/vllm_utils/bucketed_prompt_outputs \
  --request-map /ocean/projects/cis250162p/aparthas/sfs_cloud_artifacts_20260916/qwen/request_map.csv \
  --output .scratch/reserve-tail-20260916/tables
```

| file | sha256 | records | support per prompt bin (<128 / 128-512 / 512-2k / 2k-8k / 8k-32k / >=32k / all) |
|---|---|---:|---|
| `qwen3-0.6b.json` | `50a14ca76681895a622cbf2d69ba362e37792b32b76c5967b7b9b67f3f6ed87d` | 10,000 | 4990 / 26 / 2313 / 1357 / 1293 / 21 / 10000 |
| `qwen3-8b.json` | `2392950a25c04003e945124554fd33654da1c552ccdecf7e59d20770f9c04ce6` | 10,000 | same |
| `qwen3-32b.json` | `74e3a4ef30c8be24e01983b2609c4d76c8641422f26def2a899ec47ec88cf599` | 10,000 | same |
| `manifest.json` | `2eac8a79de5650982d998c7ded87f8cf48e19f3a53d1005afb734d7c4c491fbb` | | |

The build is deterministic (byte-identical on rebuild); `manifest.json` records the sha256 of each
table, the request-map sha256 and the per-bin support. The 128-512 and >=32k bins (26 and 21
outputs) back off to the model level as soon as their survivors drop below 8, exactly as
`evaluate_tail.py`'s `cal_prompt_*` did. 0.6B: 151/10,000 calibration outputs hit the 8,192 cap
(133 of them in the 8k-32k prompt bin, median total 619 there). The files are already on Vast at
`/workspace/sfs/state/diagnostics/reserve-tail-20260916/tables/` with matching sha256s.

## 4. Vast preparation (CPU only; nothing under /workspace/sfs/repo* is modified)

```bash
# 1. Own checkout at bedeec4 + this diff (repo-v2 carries the compiled vllm extension in-tree).
ssh sfs-vast 'test -d /workspace/sfs/reserve-repo || cp -a /workspace/sfs/repo-v2 /workspace/sfs/reserve-repo'
cd /ocean/projects/cis250162p/aparthas/sfs_reserve_20260916
{ git diff --name-only bedeec4; git ls-files --others --exclude-standard src scripts/cloud/reserve-paired-run-20260916.md;
  git -C vllm diff --name-only | sed 's#^#vllm/#'; git -C vllm ls-files --others --exclude-standard | sed 's#^#vllm/#'; } \
  | grep -v '^vllm$' | sort -u | tar -cf - -T - | ssh sfs-vast 'tar -xf - -C /workspace/sfs/reserve-repo'
# 2. Tables (already streamed; re-verify).
ssh sfs-vast 'sha256sum /workspace/sfs/state/diagnostics/reserve-tail-20260916/tables/*.json'   # must match section 3
# 3. Setup gates for the new source (canonical_control.gates() binds them to source_hashes()).
#    Separate state root so nothing under /workspace/sfs/state or /workspace/sfs/setup is touched.
ssh sfs-vast 'export SFS_STORAGE=/workspace/sfs; source /workspace/sfs/reserve-repo/scripts/cloud/env.sh;
  export SFS_TEST_GO="$CONDA_PREFIX/bin/go"; cd /workspace/sfs/reserve-repo;
  mkdir -p /workspace/sfs/reserve-state/setup /workspace/sfs/reserve-state/state;
  taskset -c 84-95 bash scripts/cloud/test.sh /workspace/sfs/reserve-state/setup/tests &&
  taskset -c 84-95 python -m pytest -q src/scripts/cloud/tests/test_remaining_length_flag.py src/sfs_core/routing/tests/test_remaining_length_fallback.py &&
  taskset -c 84-95 python -m scripts.cloud.prepare cpu --bundle /workspace/sfs/bundle --output /workspace/sfs/reserve-state/setup/cpu-inputs.json &&
  taskset -c 84-95 python -m scripts.cloud.prepare serving --bundle /workspace/sfs/bundle --output /workspace/sfs/reserve-state/setup/cpu-serving.json &&
  python -m scripts.cloud.canonical_control preflight --bundle /workspace/sfs/bundle \
    --state /workspace/sfs/reserve-state/state --output /workspace/sfs/reserve-state/setup/control-preflight.json --qps 8.6 --gpus 0,1,2,3'
```

## 5. Launch (only once GPUs 0-3 are free; the GPU-UUID locks in /dev/shm refuse an overlap)

```bash
ssh sfs-vast 'nvidia-smi --query-gpu=index,memory.used --format=csv -i 0,1,2,3; supervisorctl status | grep RUNNING'
# all four must read 0 MiB and no sfs-baselines-qwen* program may be RUNNING on 0-3
ssh sfs-vast 'cat > /workspace/sfs/setup/sfs-reserve-paired-8p6.sh' <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
export SFS_STORAGE=/workspace/sfs
source /workspace/sfs/reserve-repo/scripts/cloud/env.sh
cd /workspace/sfs/reserve-repo
exec taskset -c 0-47 python -m scripts.cloud.canonical_control run \
  --bundle /workspace/sfs/bundle --models /workspace/sfs/models.json \
  --state /workspace/sfs/reserve-state/state \
  --output /workspace/sfs/state/diagnostics/reserve-tail-20260916/paired-qps8p6-on \
  --qps 8.6 --gpus 0,1,2,3 \
  --remaining-length-tables /workspace/sfs/state/diagnostics/reserve-tail-20260916/tables \
  --remaining-length-rules qwen3-0.6b=running_all:0.5:prompt_bin
EOF
# (--remaining-length-rules is repeatable; models not named stay off. Omitting it applies the same
#  default, PAIRED_RUN_RULES in canonical_control.py; write it out anyway so the log is explicit.)
ssh sfs-vast 'cat > /etc/supervisor/conf.d/sfs-reserve-paired-8p6.conf' <<'EOF'
[program:sfs-reserve-paired-8p6]
command=/bin/bash /workspace/sfs/setup/sfs-reserve-paired-8p6.sh
directory=/workspace/sfs/reserve-repo
autostart=false
autorestart=false
startsecs=0
stdout_logfile=/workspace/sfs/setup/sfs-reserve-paired-8p6.log
redirect_stderr=true
EOF
ssh sfs-vast 'supervisorctl reread && supervisorctl update && supervisorctl start sfs-reserve-paired-8p6'
```

Gate before the 16,000 requests: `canonical_control run` itself executes the 192-request
calibration-prompt smoke (`smoke()`: 192 requests at 2 QPS, hard policy, `audit_run` +
`require_complete_ttft`) on the same servers with the flag on, and aborts before the control if it
fails. Confirm it passed and that the flag is on where intended before letting the run continue:

```bash
ssh sfs-vast 'O=/workspace/sfs/state/diagnostics/reserve-tail-20260916/paired-qps8p6-on;
  cat $O/status.json; python3 -c "import json;b=json.load(open(\"$O/instances.json\"))[\"remaining_length\"];print(b[\"rule\"]);print({m:r[\"rule\"] for m,r in b[\"models\"].items()})";
  for m in qwen3-0.6b qwen3-8b qwen3-32b; do echo -n "$m: "; grep -c -- "--remaining-length-mode" $O/server_argv_$m.json; done;
  ls $O/smoke/point.json'
```

Expected: `status.state` = `RUNNING_16000` after the smoke; rule `qwen3-0.6b=running_all_prompt_bin_q50`
with `{'qwen3-0.6b': 'running_all_prompt_bin_q50', 'qwen3-8b': 'current', 'qwen3-32b': 'current'}`;
`--remaining-length-mode` count 1 for 0.6B and 0 for 8B/32B; `smoke/point.json` present. Optional
in-run check: a read-only snapshot capture of the 0.6B SHM segment (as `gpu7_single_model_run.py`
does) decoded with `decode_scheduler_state_snapshot` must show
`config.remaining_length_rule == "running_all_prompt_bin_q50"` and
`config.remaining_length_table_sha256 == 50a14ca7...` while the 8B/32B segments carry no
`remaining_length_*` keys.

## 6. OFF comparators and acceptance

OFF comparators (same 16,000 requests, same SLOs, same predictors/coefficients, source bedeec4
lineage accepted by audit; `result_summary.json` of each):

| control | output | TTFT SLO attainment (system entry, /16,000) | Pro OnTimeUtility | Flash |
|---|---|---:|---:|---:|
| qps8p6 | `/workspace/sfs/state/controls/qps8p6` | 49.96% | 0.2782 | 0.2977 |
| qps8p6-repeat1 | `/workspace/sfs/state/controls/qps8p6-repeat1` | 63.71% | 0.3561 | 0.3812 |

Both pass `audit_cell` (16,000 complete requests, realized rate within 10% of 8.6 QPS, complete
TTFT). Their spread (13.8 points of attainment, 0.078 utility) is the run-to-run noise floor of
this configuration and is the reference against which the ON run is read.

Acceptance metric: `result_summary.json` of the ON run, produced by the same `report()`
(`ttft_slo_attainment_pct` over the 16,000 requests using measured system-entry TTFT against each
request's TTFT SLO, and `primary_ontimeutility` = Pro OnTimeUtility on the 15,996-query observed
judge cohort). The ON run must first pass the same `audit_cell` (status PASS, realized 8.6 QPS
within 10%, 16,000 complete TTFTs).

Reading rule (decided before the data): the flag is a candidate for the campaign only if ON
attainment exceeds the better OFF control (63.71%) and Pro OnTimeUtility exceeds 0.356; a result
inside the OFF spread (49.96-63.71%) is inconclusive and requires a second paired run (one more
ON, or ON with `qwen3-0.6b=running_all:0.5:model`) before any conclusion; a result below 49.96%
rejects the flag. Also report the 0.6B routing share and cost (`instance_route_counts`,
per-request `response_model`, `cost_source_counts`) and the per-model TTFT split, because section 4
predicts the estimate change on 0.6B only and the routing consequence (fewer requests sent to a
swamped 0.6B) is what the run is meant to measure.

## 7. Bookkeeping

- Output tree: `/workspace/sfs/state/diagnostics/reserve-tail-20260916/paired-qps8p6-on/`
  (`provenance.json` carries `remaining_length`, `instances.json` the per-model rules and table
  sha256s, `server_argv_*.json` the engine flags, `outputs/*_point01.json` the run with
  `router.runs[0].remaining_length_rule`).
- Mirror to the controller with `sync_results.py` as for the controls; the result is labelled
  `remaining_length_rule = qwen3-0.6b=running_all_prompt_bin_q50`, never merged with `current` rows.
- Do not reuse `/workspace/sfs/state` as `--state` for this run: its `../setup` gates are bound to
  the bedeec4 source and `gates()` would refuse the modified checkout.
- GPU-7 smoke of the flag (bounded, 0.6B, CPUs 84-95): see the report of this change; raw outputs
  under `/workspace/sfs/state/diagnostics/reserve-tail-20260916/flag-smoke-*`.

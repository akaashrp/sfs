# SFS cloud handoff: active controls and paused main campaign

**September 16 execution plan:** main jobs are paused for baseline and Bridges-result
reuse review. Qwen stays at 7 / 8 / 8.6 / 8.75 QPS. Ministral commits to
6.0125 / 7.8625 / 8.7875 QPS; the fourth point is an unscheduled follow-up.
This makes 47 first-priority cells plus 12 deferred predictor/judge cells before
any reuse savings. See `execution-plan-20260916.json` and
`baseline-and-reuse-review-20260916.md`. The 68-cell inventory below describes
the preserved artifact used by the running controls, **not permission to run
all of it**. Reconcile the final main execution bundle after review; leave the
active controls and their checkout/bundle untouched.

This checkout adds the vLLM-SR latency-aware selector to Ministral at
6.0125 / 7.8625 / 8.7875 / 9.7125 QPS, 8,000 requests each. Qwen's existing
implementation, policy tuple, launchers, grids, and source-bound Bridges
artifacts are preserved. All new cloud code lives under `scripts/cloud` and
`src/scripts/cloud`; the Ministral alias adapter is `src/scripts/runs/ministral3_latency.py`.
The active Bridges checkout and submitted jobs are separate and unchanged.
The Qwen cloud rates are **7, 8, 8.6, and 8.75 QPS** for both baselines and
predictor/judge variants; the Ministral rates above remain unchanged.

| Track | Cells | Requests/cell |
|---|---:|---:|
| Qwen: Mooncake, LMDeploy, RouteBalance, SCORE, vLLM-SR × four loads | 20 | 16,000 |
| Ministral: original eight policies plus vLLM-SR × four loads | 36 | 8,000 |
| Qwen: Flash-quality, MLP-quality, MLP-length × four loads | 12 | 16,000 |
| Total | 68 | 800,000 total |

All arrivals remain Poisson, seed 69. Figures 5/13 share results. Calibration,
smoke, and load-probe traffic are separate and excluded from those counts.

## What to rent

September 15 recovery update: the real producer emits
`system_entry_e2e_ttft_slo_missing_count`; the corrected consumer also checks
every measured TTFT value. The Ministral-only scheduler waits for fresh busy
snapshots while retaining the 1,000 ms age limit, with a 10 s stall timeout and
25 ms polling. Waiting is included in end-to-end TTFT and recorded in each
decision. It releases the routing lock between retries so completions can drain.
The original shared scheduler and Qwen launch paths remain unchanged.
Ministral cloud smoke now uses the maximum evaluation load and adds 512-request
calibration stress for Mooncake and RouteBalance. These are required GPU gates,
not evidence that the new path has already run on a GPU.

The additive Bridges recovery entry point is
`scripts.runs.ministral3_recovery` and its launcher is
`scripts/cloud/ministral-recovery.sbatch`. It reuses the 12 audited non-Mooncake
cells from job 45842572, reruns the four failed Mooncake cells, and runs the
16 formerly pending snapshot-policy cells. It audits each cell immediately,
preserves its raw checkpoint, and collates all 32 after completion. It uses the
original immutable manifest and serving checkout plus a separately checksummed
repair plan. Qwen jobs keep their existing sources and manifests.

- Two **4 × H100 80GB** instances or one **8 × H100 80GB** instance, Linux x86_64.
  Prefer the same H100 SXM configuration on both hosts; inspect `nvidia-smi topo -m`.
  A Qwen pool always stays within one host: TP 1/1/2. No cross-host NCCL is needed.
- Prefer at least 32 assigned CPU cores and 256 GB RAM per four-GPU slot
  (64 cores and 512 GB RAM for eight GPUs). CPU affinity and arrival attainment
  are verified. Do not rent MIG slices or a GPU virtualization mode that lacks
  normal NCCL/shared-memory behavior.
- **Provision 1 TB of storage per instance** for comfortable setup, model, cache,
  and log headroom. Bootstrap requires at least 300 GiB free. The transferable
  input bundle is a few GB; all six BF16 checkpoints are roughly 140 GB before
  environment/build caches and outputs. Models are downloaded once per host,
  and pools use those paths directly rather than copying weights per job.
- Shared memory: configure at least **16 GiB `/dev/shm`** when the provider exposes
  a container option; preflight rejects less than 1 GiB free. Use a CUDA-enabled
  Linux image with git/curl and an NVIDIA driver compatible with CUDA 12.9
  (R575 or newer recommended). Bootstrap creates an isolated conda environment
  and does not replace the host driver or system CUDA. Docker-in-Docker is unnecessary.

Storage is provider-specific; the mounted path must be checked after creation.

| Provider | Steps before deployment |
|---|---|
| Prime Intellect | For independent persistence, create a persistent disk in the **same provider and datacenter** as the GPU offer; select a compatible offer with **Add Shared Filesystem**, attach the disk, and note its actual mount path. Set `SFS_STORAGE` underneath that mount. Instance-local disk is not a substitute for a persistent disk. |
| Thunder Compute | Check the actual instance quote for included disk capacity and expansion charges; do not assume 100 GB per GPU is included. Put `SFS_STORAGE` on persistent storage and expand to the planned capacity if needed. Create a snapshot and wait for READY before deleting/replacing an instance; maintain off-host result copies regardless. |
| Vast | Set disk size at creation; container disk cannot subsequently grow and is lost on destroy. To survive instance deletion, create/attach a **volume on the same physical host** and put `SFS_STORAGE` under its mount. A Vast volume cannot migrate to another host. Maintain off-host copies. |

Sources checked September 14, 2026:
[Prime disk attachment](https://docs.primeintellect.ai/tutorials-storage/use-persistent-storage-with-instances),
[Prime disk lifecycle](https://docs.primeintellect.ai/cli-reference/managing-disks),
[Thunder specifications](https://www.thundercompute.com/docs/technical-specs),
[Thunder snapshots](https://www.thundercompute.com/docs/cli/operations/snapshots),
[Vast storage](https://docs.vast.ai/guides/instances/storage/types).
Check offer-specific capacity before purchasing. The September 16 destination is
Vast instance 51183839 in Taiwan: eight H100 SXM 80 GB GPUs, a 1,000 GB local
volume mounted at `/workspace`, and a separate 64 GB container disk. Use
`SFS_STORAGE=/workspace/sfs`; `/workspace` was verified as a writable host volume.
The controller aliases `sfs-vast` and `sfs-cloud-a` both reach this instance.

## SSH and transfer

Add your public SSH key in the provider dashboard. On the controller machine,
add `sfs-cloud-a` and, for two hosts, `sfs-cloud-b` entries using
`ssh-config.example`. Use the provider's exact user and SSH port. Keep private
keys on the controller; no Gemini/API credentials are needed for this campaign.
If the SSH configuration is only on your laptop, the agent running on Bridges
cannot use it: the aliases/key must also be available to the controller that
will execute SSH. Check the host fingerprint on the first connection.

The prepared artifact location and SHA256 are recorded in `release.json` and
the handoff response. Commands below run from this checkout on the controller:

```bash
python scripts/cloud/remote.py check --host sfs-cloud-a --storage /workspace/sfs
python scripts/cloud/remote.py deploy --host sfs-cloud-a --storage /workspace/sfs \
  --archive /path/to/sfs-cloud-inputs.tar.gz --archive-sha256 THE_RECORDED_SHA256
python scripts/cloud/remote.py bootstrap --host sfs-cloud-a --storage /workspace/sfs
python scripts/cloud/remote.py models --host sfs-cloud-a --storage /workspace/sfs
```

Substitute the actual persistent mount path, e.g. `/home/USER/sfs` on Thunder.
Repeat for host B if using two hosts. GitHub code is cloned at the release's
exact SFS and vLLM commits. The checksummed data/predictor bundle is copied
separately; bulky datasets, checkpoints, and credentials are not committed to Git.
All six model revisions and the encoder revision are pinned in `bundle.json`.

Run bootstrap inside an SSH session protected by your local terminal/session
manager if desired. Worker jobs themselves detach and survive SSH disconnects.
If bootstrap fails, inspect `setup/vllm-build.log`, resolve that concrete error,
and rerun bootstrap; it reuses the environment/download cache. Do not upgrade
vLLM or Torch to a provider template's latest package.

## Destination checks and launches

On the instance, every command starts with:

```bash
export SFS_STORAGE=/workspace/sfs
export SFS_BUNDLE="$SFS_STORAGE/bundle"
source "$SFS_STORAGE/repo/scripts/cloud/env.sh"
```

Bootstrap runs CPU workload/predictor checks, selector/Qwen regression tests,
and native-extension/GPU/IPC checks. Before Qwen, verify its TP pair:

```bash
python -m scripts.cloud.diagnose --state "$SFS_STORAGE/state" \
  --output "$SFS_STORAGE/setup/nccl-23.json" --nccl --gpus 2,3
```

On an eight-GPU host also check `--gpus 6,7`. The two slots are GPUs `0,1,2,3`
and `4,5,6,7`. Use **one shared state directory per host** so cell ownership
and completion checks cover both slots. Split `os.sched_getaffinity(0)` into
two disjoint CPU lists and pass `--cpus` for both slots. On two hosts each
slot uses `0,1,2,3` and that host's full CPU affinity. GPU UUID locks prevent
overlapping pools, including pools launched from different output roots.

An initial balanced assignment is in `hosts.example.json`: slot A runs Qwen
baselines + MLP quality (384,000 requests); slot B runs Ministral + MLP length
+ Flash quality (416,000). These are scheduling groups, not new experimental
conditions. Do not concurrently run two variants within one four-GPU slot.

Start calibration/qualification (example for Ministral on slot B):

```bash
python -m scripts.cloud.control submit --state "$SFS_STORAGE/state" -- \
  qualify --bundle "$SFS_BUNDLE" --models "$SFS_STORAGE/models.json" \
  --family ministral --variant canonical --gpus 4,5,6,7 --cpus CPU_LIST_B
```

For Qwen use `--family qwen`, `--gpus 0,1,2,3`, `--cpus CPU_LIST_A`.
The returned record contains the detached PID, log, and unique run directory.
Qualification uses calibration-only prompts, warmed singleton prefill probes,
loaded decode/service measurements, policy smoke, and two sustained
shortest-queue load probes. Every vLLM-SR cell starts with 32 calibration
prompts × three models and empty latency history; each point drains completely.
Ministral's served aliases are checked before conversion to canonical routing keys.

Qualification produces `qualification.json` with **GPU_MEASURED_REVIEW_REQUIRED**.
The agent must inspect the timing-fit residuals, coverage, load-probe classifications,
server logs, and retained SFS batch coefficients before recording the review:

```bash
python -m scripts.cloud.control release --qualification /path/to/qualification-run \
  --timing-review 'Concrete assessment of the measured timing residuals and coverage...' \
  --load-review 'Concrete assessment of actual arrivals, backlog and hardware comparability...'
```

This is an evidence review the agent performs, not a request for another user
approval. A failed/inconclusive qualification is debugging work; do not fabricate
a PASS, reuse Bridges smoke as cloud evidence, or silently change the QPS grid.
Cloud timing models are fitted from destination calibration. Canonical SFS batch
coefficients and serving settings are retained and their destination residuals
must be acceptable for the intended comparison. Different GPU topology/performance
must remain visible in provenance; cloud points are not silently merged into
historical Bridges curves as though hardware were unchanged.

Then launch the same family/variant with the matching qualification:

```bash
python -m scripts.cloud.control submit --state "$SFS_STORAGE/state" -- \
  run --bundle "$SFS_BUNDLE" --models "$SFS_STORAGE/models.json" \
  --family ministral --variant canonical --gpus 4,5,6,7 --cpus CPU_LIST_B \
  --qualification /path/to/qualification-run
```

Repeat qualification and run for each Qwen predictor arm: `mlp_quality`,
`mlp_length`, `flash_quality`. Each variant is independently gated; the MLP
length artifact reaches both the router and all serving models. Quality variants
retain the canonical length predictor. No ordinary Qwen SFS control reruns are added.

`--cells ID,ID` optionally restricts evaluation to exact cells from `bundle.json`.
Re-running a batch skips only completed cells with matching bundle/point hashes.
Failed partial attempts remain in their unique directories. After reboot, new
hardware, source changes, or changed inputs, rerun CPU checks and qualification.
Do not run a completed cell on a second host: coordinate with the global matrix
and review Bridges results before assigning cloud work.

## Monitoring, backup, and results

```bash
python -m scripts.cloud.control status --state "$SFS_STORAGE/state" --bundle "$SFS_BUNDLE"
```

The status lists exact completed and remaining cell IDs, failed/lost jobs,
heartbeat ages, and logs. `active_cell.json` identifies the current cell. A stale
heartbeat needs investigation; server readiness may take several minutes before
the first heartbeat. Worker failures stop the batch rather than silently retrying.
Every successful cell has complete request/response IDs, zero request failures,
end-to-end TTFT, a ≤10% arrival-rate discrepancy, and a checksummed result.

On the controller, keep an off-host pull running for each instance:

```bash
python scripts/cloud/sync_results.py --host sfs-cloud-a \
  --remote-state /workspace/sfs/state \
  --destination /ocean/projects/cis250162p/aparthas/sfs_cloud_results_20260914 --watch
```

This uses only the controller's SSH key. It copies logs, qualification evidence,
raw points, and completion records every 60 seconds. Inspect
`controller_sync_status.json` for backup health; a running GPU job does not prove
backup success. An interruption can lose progress within the current cell; cells
with verified off-host copies survive instance loss. Stop the watcher only after
a final successful sync. Verify point hashes before terminating any instance.
Code never automatically destroys paid instances or deletes source results.

Collate both hosts' recovered result roots, preserving raw points:

```bash
python -m scripts.cloud.collate --bundle /path/to/extracted-bundle \
  --raw-root /path/to/recovered/sfs-cloud-a --raw-root /path/to/recovered/sfs-cloud-b \
  --output /path/to/new/derived-report
```

Full collation rejects missing or duplicate cells; `--allow-partial` explicitly
labels an interim report. It joins the frozen judge scores and produces Figure
5/13 summaries and full per-cell hardware/source provenance on derived copies.
The default `--judge auto` uses Flash for `flash_quality` and Pro for all other
arms. Explicit `--judge pro` or `--judge flash` produces a controlled judge
comparison on the same serving results; Ministral always uses its own Pro scores.
`observed_judge_summary.json` is the primary Qwen quality report: it reports both
judges on the same 15,996 complete observed query groups and identifies the
matching primary judge. Four imputed groups are excluded from both utility
denominators; TTFT attainment still covers all 16,000 requests. The legacy
`figure5_13_summary.json` retains all requests and the frozen imputed scores,
and is identified as such in the audit. No judge API calls or reruns are needed.
Use existing plotting code on the derived directories, keeping variant and
hardware identities explicit. Historical baseline reuse remains a separate
comparison with the implementation/hardware caveats recorded above.

## September 16 refresh and background downloads

The active Bridges shared router and predictor source matches this release;
the September 15 Ministral completeness audit, bounded freshness waiting,
and Ministral vLLM-SR adapter are already included. Bridges job 46116020 is
the reference for the additive canonical Qwen cloud controls at 8.6 and 8.75 QPS
described below. These controls are separate from the 68-cell matrix. No
submitted Bridges scripts or source files are edited.

Pinned model downloads can start before the full serving environment exists:

```bash
python -m scripts.cloud.model_downloads --manifest /path/to/bundle.json \
  --cache /workspace/sfs/hf/hub --output /workspace/sfs/models.json --workers 2
```

This requires the locked Hugging Face Hub dependency, but no Torch/vLLM imports.
It uses the same cache paths and allow-list as `prepare models`, hashes completed
files, records `models.json.progress.json`, and publishes `models.json` only when
all requested checkpoints are complete. Ministral downloads only the consolidated
BF16 checkpoint. The Taiwan prefetch worker is managed by Supervisor:

```bash
ssh sfs-vast 'supervisorctl status sfs-model-downloads; tail -20 /workspace/sfs/setup/model-downloads.log'
```

Its small download-only conda environment is separate from the pinned serving
environment. Finish downloads and setup before measuring latency or load probes.

## Bridges-only selector extension

`python -m scripts.runs.ministral3_latency prepare` accepts the existing
32-cell Ministral manifest and calibration requests and writes a separate
four-cell selector manifest. Its `smoke` and `sweep` modes use an explicitly
supplied existing three-model pool and three wait logs. It does not rewrite
the legacy manifest, change the shared policy tuple, or submit/cancel Slurm jobs.
For cloud use the destination-qualified worker above.

## Canonical Qwen controls, September 16

The two requested controls are additive to the frozen 68-cell campaign. They use
`scripts.cloud.canonical_control`, the unchanged Qwen pool and sweep producer,
canonical quality/length predictors, and the original batch coefficients. The
preflight checks all 16,000 request identities and SLOs against the reference used
by Bridges job 46116020. Every pool must pass a 192-request canonical SFS smoke
using calibration prompts before the measured run. Destination CPU regression,
input and serving gates are required. Complete all model downloads and GPU/NCCL
preflight before launch. These controls do not refit the SFS coefficients.

On the Taiwan 8-H100 host, use disjoint CPU allocations (0–47 and 48–95) with:

```bash
export SFS_STORAGE=/workspace/sfs
source /workspace/sfs/repo/scripts/cloud/env.sh
# Launch each command under its own Supervisor program, with separate logs.
taskset -c 0-47 python -m scripts.cloud.canonical_control run \
  --bundle /workspace/sfs/bundle --models /workspace/sfs/models.json \
  --state /workspace/sfs/state --output /workspace/sfs/state/controls/qps8p6 \
  --qps 8.6 --gpus 0,1,2,3
taskset -c 48-95 python -m scripts.cloud.canonical_control run \
  --bundle /workspace/sfs/bundle --models /workspace/sfs/models.json \
  --state /workspace/sfs/state --output /workspace/sfs/state/controls/qps8p75 \
  --qps 8.75 --gpus 4,5,6,7
```

Raw points, per-request responses, server/driver logs, hardware and source
provenance stay in each output directory. `result_summary.json` reports TTFT
attainment over all 16,000 requests and both Pro/Flash OnTimeUtility on the common
15,996 observed scored queries; Pro is primary for these canonical controls.
Historical Bridges results remain a separate hardware comparison.

The bootstrap explicitly places Conda environments and package caches on
`SFS_STORAGE`. On NVIDIA's Conda CUDA 12.9 packaging, `CUDA_HOME` points to
`$CONDA_PREFIX/targets/x86_64-linux` for headers and libraries, while `CUDACXX`
points to `$CONDA_PREFIX/bin/nvcc` beside its required `nvcc.profile`. Source
`env.sh` for every session so these paths and the pinned environment remain
consistent. The bootstrap tolerates optional unset variables in vendor activation
hooks and preserves Python 3.12.11 when installing the CUDA toolkit.

For an already transferred September 14 bundle, apply the explicit cloud rate
update before destination verification (the September 16 archive already includes it):

```bash
python -m scripts.cloud.schedule --bundle "$SFS_BUNDLE" \
  --audit "$SFS_STORAGE/setup/rate-update.json"
```

This updates only Qwen rates and cell IDs, preserving every artifact checksum and
all Ministral settings. It retains the original bundle manifest as a backup.

## Second Qwen serving configuration (FCFS, unchunked): preparation only

`qwen-fcfs-unchunked-65536` is prepared but not launchable: overlay
`scripts/cloud/fcfs/campaign-20260916.json` (36 cells, `full_matrix_authorized: false`),
worker profile `--profile fcfs` (`calibrate`/`qualify` only until authorization),
coefficient refit `scripts.cloud.fcfs.coefficients`, and the step-by-step
qualification runbook `scripts/cloud/fcfs/RUNBOOK.md`. The canonical chunked
configuration, its gates and completed cells are unchanged.

## Further Qwen serving configurations (reduced grids)

`src/scripts/cloud/serving/profiles.py` is the table of serving configurations behind
worker `--profile` (`fcfs`, `chunk8192`, `prefix_cache`): configuration id, scheduler settings,
server argv delta over the canonical Qwen argv, instances.json row overrides, coefficient policy
(`refit` = SFS batch coefficients fitted from destination traces of that configuration and bound
with `--coefficients`; `canonical` = canonical coefficients retained, `--coefficients` refused)
and the bounded GPU smoke rule. `scripts.cloud.serving.campaign --profile <name>` writes the
reduced-grid overlays `scripts/cloud/fcfs/campaign-chunk8192-20260917.json` and
`campaign-prefix-cache-20260917.json` (kind `serving_config`: SFS plus the two strongest external
baselines at 6/7/8/8.3 QPS; cells derive from the single editable `policies` list, so an edit
changes the overlay hash and re-qualification follows). Generic tooling:
`scripts.cloud.serving.coefficients fit|validate --profile`, `scripts.cloud.serving.gpu_smoke
--profile`; runbooks `scripts/cloud/fcfs/RUNBOOK-chunk8192.md` and `RUNBOOK-prefix-cache.md`.

# SFS cloud handoff: 68 evaluation cells

This checkout adds the vLLM-SR latency-aware selector to Ministral at
6.0125 / 7.8625 / 8.7875 / 9.7125 QPS, 8,000 requests each. Qwen's existing
implementation, policy tuple, launchers, grids, and source-bound Bridges
artifacts are preserved. All new cloud code lives under `scripts/cloud` and
`src/scripts/cloud`; the Ministral alias adapter is `src/scripts/runs/ministral3_latency.py`.
The active Bridges checkout and submitted jobs are separate and unchanged.

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
| Thunder Compute | The home directory is on persistent disk. Current docs include 100 GB per GPU (400 GB for four, 800 GB for eight); expand if needed. Put `SFS_STORAGE` under the home directory. Create a snapshot and wait for READY before deleting/replacing an instance; maintain off-host result copies regardless. |
| Vast | Set disk size at creation; container disk cannot subsequently grow and is lost on destroy. To survive instance deletion, create/attach a **volume on the same physical host** and put `SFS_STORAGE` under its mount. A Vast volume cannot migrate to another host. Maintain off-host copies. |

Sources checked September 14, 2026:
[Prime disk attachment](https://docs.primeintellect.ai/tutorials-storage/use-persistent-storage-with-instances),
[Prime disk lifecycle](https://docs.primeintellect.ai/cli-reference/managing-disks),
[Thunder specifications](https://www.thundercompute.com/docs/technical-specs),
[Thunder snapshots](https://www.thundercompute.com/docs/cli/operations/snapshots),
[Vast storage](https://docs.vast.ai/guides/instances/storage/types).
Check offer-specific capacity before purchasing; no instance has been provisioned by this preparation.

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
Run a second collation with `--judge flash` into another directory for the Qwen
judge-sensitivity view; the original Flash holdout imputation/comparison audit
is retained in the bundle and report. The two evaluation-judge views use the
same serving results and do not add experiments.
Use existing plotting code on the derived directories, keeping variant and
hardware identities explicit. Historical baseline reuse remains a separate
comparison with the implementation/hardware caveats recorded above.

## Bridges-only selector extension

`python -m scripts.runs.ministral3_latency prepare` accepts the existing
32-cell Ministral manifest and calibration requests and writes a separate
four-cell selector manifest. Its `smoke` and `sweep` modes use an explicitly
supplied existing three-model pool and three wait logs. It does not rewrite
the legacy manifest, change the shared policy tuple, or submit/cancel Slurm jobs.
For cloud use the destination-qualified worker above.

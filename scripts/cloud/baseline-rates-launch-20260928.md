# Baseline rate fill — launch runbook

Measures the five canonical-pool policies whose Figure 5 rows stop at 8.3 QPS — Mooncake,
LMDeploy, RouteBalance, SCORE, vLLM-SR — at **3, 4, 5, 8.6, 8.9, 9.0, 9.2 QPS**.
35 cells, 440,000 requests, ~18.7 h of routing.

| rate | requests | reason |
|---|---|---|
| 3, 4, 5 | 8,000 | nothing saturates; the queue is stationary, so the short budget measures the same steady state |
| 8.6, 8.9, 9.0, 9.2 | 16,000 | saturated; run length changes the result, so these match the existing cells at those rates |

Overlay: `scripts/cloud/baseline-rates-campaign-20260928.json` (kind `baseline_rates`,
module `src/scripts/cloud/baseline_rates_campaign.py`, tests
`src/scripts/cloud/tests/test_baseline_rates_campaign.py`). The overlay carries the canonical
0.6B remaining-length rule and SCORE's tuned multiplier (0.05) with its sweep evidence, so these
cells are parameterised exactly like the 6/7/8/8.3 cells they extend.

## Allocation plans

**8 GPUs — two lanes, ~10 h wall clock.** Set `LANE_CORES` to the per-lane core count the
cells being extended used — 48 — whenever the box has more cores than the September one (96). The
claim helper otherwise splits the box in half, so a 192-core rental would hand each lane 96 cores
and measure these rates under less CPU contention than the 6/7/8/8.3 cells they join. Lane A then
takes cores 0–47 and lane B 96–143, one socket each, with the gate on the top 12.

| lane | rates | cells | routing |
|---|---|---|---|
| A | 3, 4, 9.2 | 15 | ~8.9 h |
| B | 5, 8.6, 8.9, 9.0 | 20 | ~9.8 h |

**4 GPUs — one lane, ~19 h wall clock.** `baseline-rates-single.sh` runs all 35 cells in the
order 8.6, 9.0, 5, 4, 3, 8.9, 9.2 — rate by rate, all five policies each — so a run cut short
still leaves whole columns of the figure complete rather than a partial row everywhere.

## Rental requirements

- **GPUs:** 4 or 8 H100. Four run one lane; eight run two.
- **Disk: at least 200 GB.** This campaign needs only the Qwen family — Qwen3-0.6B, 8B and 32B
  in BF16, about 83 GB of weights under `/workspace/sfs/hf/hub/` — plus the vLLM build and conda
  environment (~20 GB) and the campaign state (35 points at ~50 MB, with logs, under 10 GB).
  The September box additionally held the three Ministral checkpoints, ~133 GB of weights in all.

## Deployment

1. Rent, then bootstrap as usual (`.scratch/claude-watchers/vast_deploy_v7.sh`): conda env,
   vLLM build, bundle, models, supervisord.
2. Check out this branch as `/workspace/sfs/<repo-dir>` and export `SFS_REPO=<repo-dir>` for
   every program below — the helpers take the checkout name as an input rather than hardcoding it.
3. Copy `scripts/cloud/runners/*.sh` to `/workspace/sfs/setup/` and append
   `scripts/cloud/runners/baseline-rates-supervisor.conf` to the supervisord config, then
   `supervisorctl reread && supervisorctl update`.
4. `mkdir -p /workspace/sfs/ratefill/{state,claims}`.

## Running

```
supervisorctl start sfs-ratefill-gates      # CPU gate; wait for RATEFILL_GATES_DONE
supervisorctl start sfs-ratefill-a          # 8-GPU box: both lanes
supervisorctl start sfs-ratefill-b
# or, on a 4-GPU box:
supervisorctl start sfs-ratefill-single
```

Each runner refreshes the gate, claims a lane (two-lane case) and releases the claim on exit.
They deliberately do not `exec` the worker: an `exec` discards the release trap and leaks the
lane, which idled GPUs for hours in the September campaign.

## Checks while running

- `nvidia-smi` — both lanes should hold memory; an idle lane with a live program means a leaked claim
  (`rmdir /workspace/sfs/ratefill/claims/<lane>`).
- `/workspace/sfs/ratefill/lane-*.log` — one `point.json` per cell under
  `/workspace/sfs/ratefill/state/lane-*/cells/<id>/`.
- A cell that fails on an infrastructure fault is salvaged, not rerun; the salvage record sits
  beside the point and its failed requests score zero in the denominator.

## After the runs

1. Archive `/workspace/sfs/ratefill` the same way as the September campaign, and verify the
   tarball before destroying the instance.
2. Rebuild the plotting JSONs: `python sfs_work/paper_build/build_results.py` — add the new state
   root to `POOL_ROOTS` first. The new cells carry the same ids as the canonical grid
   (`qwen-<policy>-<rate>`), so they slot into `figure5_otu_vs_qps.json` automatically.
3. Re-render: `python sfs_paper_results/figures/make_figures.py`, then set `FIG5_RATES` to the
   grid you publish (3, 4, 5, 6, 7, 8, 8.3, 8.6, 8.9, 9.0, 9.2).

## Interruptible rentals

The campaign is resumable by construction: the worker writes a receipt per finished cell to
`<state>/completed/<cell-id>.json` and skips any cell that has one, after checking the bundle
digest and the source pin. An interruption costs at most the cell in flight — ~44 min at 3 QPS,
~31 min at 8.6.

Use `baseline-rates-supervisor-interruptible.conf` instead of the lane programs. It autostarts
the single-lane runner and the mirror, so the campaign resumes by itself when the instance comes
back. Two safeguards make that safe to leave unattended:

- every runner exits immediately unless `/workspace/sfs/bootstrap-complete` exists, so an
  autostarted runner on a fresh, unbootstrapped instance does nothing rather than failing in a loop;
- the runner is idempotent against the ledger, so a restart re-runs only unfinished cells.

**The disk is not independent of the instance.** Mirror the ledger, points and audits off the
box, and on a replacement instance restore them before the runner starts; without the mirror,
losing the disk means redoing every cell.

Two ways to do it, depending on what the far end has:

- `sfs-ratefill-sync` on the box, with `BRIDGES_DEST` set, pushes with rsync — it needs rsync at
  **both** ends.
- `sfs_work/bridges_pull_ratefill.sh` pulls from Bridges instead, over ssh and tar only. Bridges
  has neither rsync nor scp installed, so this is the one that works there. It compares name and
  size each pass and fetches only new or grown files, so a pass costs almost nothing once the set
  is current. `restore-ratefill.sh` needs rsync too; from a pull mirror, push the tree back with
  `tar -czf - -C <mirror> state | ssh <box> 'tar -xzf - -C /workspace/sfs/ratefill'`.

If an interruption lands mid-rate, the five policies of that rate end up split across two
machines. At 3, 4 and 5 QPS this is immaterial. At 8.6 and above it is a within-column mix; if
you want to remove it, delete that rate's receipts from `<state>/completed/` before restarting so
the whole group re-runs together — five cells, about 2.5 h.

## Cost

~19 lane-hours of routing plus seven pool startups. On two lanes that is ~10 h wall clock; on
one lane, ~19 h. Budget $150–250 at recent 4–8 GPU rates.

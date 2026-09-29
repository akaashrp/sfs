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

**8 GPUs — two lanes, ~10 h wall clock.**

| lane | rates | cells | routing |
|---|---|---|---|
| A | 3, 4, 9.2 | 15 | ~8.9 h |
| B | 5, 8.6, 8.9, 9.0 | 20 | ~9.8 h |

**4 GPUs — one lane, ~19 h wall clock.** `baseline-rates-single.sh` runs all 35 cells in the
order 8.6, 9.0, 5, 4, 3, 8.9, 9.2 — rate by rate, all five policies each — so a run cut short
still leaves whole columns of the figure complete rather than a partial row everywhere.

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
digest and the source pin. Restarting the same runner therefore re-runs only unfinished cells.
What is lost to an interruption is the cell in flight — at most ~44 min (3 QPS) or ~31 min
(8.6 QPS).

Three things have to be true for that to hold:

1. **The state directory must survive.** The instance's disk is not independent of the instance;
   a destroyed instance takes `/workspace/sfs/ratefill` with it. Start `sfs-ratefill-sync` with
   `BRIDGES_DEST` set, which mirrors the completed-cell ledger, the points and the audits to
   durable storage every five minutes. Restore it to the same path on the replacement instance
   before starting a runner, and the campaign picks up where it stopped.
2. **Restart has to happen.** The lane programs are `autostart=false` so a reboot never launches
   GPU work unattended. On an interruptible rental either flip the runner you are using to
   `autostart=true` — it is idempotent against the ledger — or restart it by hand after each
   interruption.
3. **Watch for a machine change.** If the instance is destroyed rather than stopped and you
   re-rent elsewhere, cells after the interruption are measured on different hardware. That does
   not matter for 3, 4 and 5 QPS, where nothing saturates, but it does at 8.6 and above, which is
   exactly where deployment differences show up.

The sensible split, if the price difference is worth it: run **3, 4, 5 on an interruptible**
instance, and the **8.6–9.2 block on an on-demand** one so that saturated band is measured on a
single machine. Those two groups are independent — the single-lane script's rate order already
completes one rate before starting the next, so a rate group never straddles a boundary unless an
interruption lands mid-rate.

## Cost

~19 lane-hours of routing plus seven pool startups. On two lanes that is ~10 h wall clock; on
one lane, ~19 h. Budget $150–250 at recent 4–8 GPU rates.

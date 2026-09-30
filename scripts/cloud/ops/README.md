# Rate-fill operations tooling

What actually ran the baseline rate fill on a rented box, kept here because most of it was written
against failures that will recur on the next rental and none of it is obvious from the runbook.

## On the box, in order

| script | role |
|---|---|
| `box_env_prep.sh` | miniforge, the pinned env, torch, then the source build |
| `box_build_vllm.sh` | builds vLLM from the pinned fork commit; **not** the archived wheel, which predates the fill rule |
| `place_extensions.py` | copies the built wheel's `.so` into the source tree that shadows it on `PYTHONPATH` |
| `box_models.sh` | the three Qwen checkpoints at the September revisions |
| `box_install.sh` | supervisor config, `LANE_CORES`, `bootstrap-complete` |
| `box_autopilot.sh` | starts both lanes |
| `box_recover.py` | restarts a dead lane, releases a parked pool, salvages a finished-but-unscored cell |
| `recover_watchdog.sh` | keeps `box_recover.py` alive; it in turn restarts supervisord |

## From Bridges

| script | role |
|---|---|
| `bridges_pull_ratefill.sh` | off-box mirror over ssh+tar (Bridges has no rsync or scp) |
| `mirror_keepalive.sh` | keeps that mirror running |
| `watch_box.sh` | one line per state change: receipts, parked pools, lane faults |
| `archive_ratefill.sh` | pull, verify the tarball, stage into `allpoints/ratefill/state` |
| `pull_setup.sh` | pulls the built wheel and logs, which the campaign archive does not cover |
| `finalize.sh` | waits for the last cell, then archive → rebuild → re-render → email |
| `finish_figures.sh` | sets the published rate grid and re-renders |

## Things that cost time, so that they do not again

* **The archived wheel cannot run this campaign.** It is twenty commits behind the submodule pin and
  predates the conditional remaining-length targets. Build from the pin.
* **`nproc` honours `OMP_NUM_THREADS`**, which `env.sh` sets to 4. Use `getconf _NPROCESSORS_ONLN`
  for anything sizing itself to the box, or a 192-core machine reads as 4 and the lanes overlap.
* **supervisord sets no `minfds`**, so everything inherits a 1024 soft limit. A saturated cell holds
  over a thousand concurrent request sockets. Raise it in the runner, where it needs no supervisord
  restart and so cannot disturb a lane that is already running.
* **Never copy a runner script over itself while it is executing.** bash re-reads by byte offset and
  runs whatever now sits there.
* **A cell can finish and still be refused**, when a snapshot-reading baseline hits a torn read at a
  saturated rate. Offer it to `scripts.cloud.salvage` before paying for a rerun; the rerun draws from
  the same distribution and will probably fail too.
* **`salvage.py` calls `apply_any_campaign(..., inspect=True)`**, which that function does not accept,
  so `--bundle` is unusable for every campaign kind. Left unfixed during the run because `src/` is
  hashed into the release the pools validate against before every cell.

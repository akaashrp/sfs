#!/bin/bash
# Print "<lane> <gpus> <cpus>" for a four-GPU lane this caller now owns, waiting until one is free.
# A lane is free when no foreign program that owns it is RUNNING, its GPUs are idle (a few MiB, not
# exactly zero), and nobody else has claimed it. The claim is a directory, so two waiters can never take the same lane.
# Generalised from the September campaign: claim root, lane owners and the CPU split are inputs,
# because a new rental may not have 96 cores or the same program names.
set -euo pipefail
CLAIMS="${CLAIM_ROOT:-/workspace/sfs/ratefill/claims}"
# Programs from OTHER campaigns that hold these GPUs, so a lane is not claimed out from under one.
# They default to empty and must never name the caller's own program: the two lane runners are
# started together, so defaulting these to sfs-ratefill-a/b made each runner see its own program
# RUNNING, mark its lane busy, and spin forever -- both lanes deadlocked with eight GPUs idle.
# Exclusion between the two lane runners is the mkdir claim below, not this check.
OWNERS_A="${LANE_A_OWNERS:-}"
OWNERS_B="${LANE_B_OWNERS:-}"
mkdir -p "$CLAIMS"
CORES=$(nproc)
HALF=$(( CORES / 2 ))
# Cores per lane. It defaults to half the box, but a rental with more cores than the campaign it
# extends must not hand its lanes more CPU than the cells already measured had: the September
# cells ran four GPUs against 48 cores, and giving them 96 here would change the very contention
# these points are compared on. Each lane still starts at a socket boundary.
LANE_CORES="${LANE_CORES:-$HALF}"
while true; do
  for lane in A B; do
    if [ "$lane" = A ]; then gpus=0,1,2,3; cpus="0-$((LANE_CORES - 1))"; owners="$OWNERS_A";
    else gpus=4,5,6,7; cpus="$HALF-$((HALF + LANE_CORES - 1))"; owners="$OWNERS_B"; fi
    busy=no
    for owner in $owners; do
      supervisorctl status "$owner" 2>/dev/null | grep -qE "RUNNING|STARTING" && busy=yes
    done
    [ "$busy" = yes ] && continue
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpus" 2>/dev/null | sort -n | tail -1)
    [ -z "$mem" ] && continue
    # An idle H100 reports a few MiB, not 0, so an exact-zero test never frees a lane on this
    # hardware. A live pool holds tens of GB, so any small floor separates the two cleanly.
    [ "$mem" -gt "${GPU_FREE_MB:-64}" ] && continue
    if mkdir "$CLAIMS/$lane" 2>/dev/null; then
      echo "$lane $gpus $cpus"
      exit 0
    fi
  done
  sleep 45
done

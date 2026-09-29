#!/bin/bash
# Print "<lane> <gpus> <cpus>" for a four-GPU lane this caller now owns, waiting until one is free.
# A lane is free when no program that owns it is RUNNING, its GPUs hold no memory, and nobody else
# has claimed it. The claim is a directory, so two waiters can never take the same lane.
# Generalised from the September campaign: claim root, lane owners and the CPU split are inputs,
# because a new rental may not have 96 cores or the same program names.
set -euo pipefail
CLAIMS="${CLAIM_ROOT:-/workspace/sfs/ratefill/claims}"
OWNERS_A="${LANE_A_OWNERS:-sfs-ratefill-a}"
OWNERS_B="${LANE_B_OWNERS:-sfs-ratefill-b}"
mkdir -p "$CLAIMS"
CORES=$(nproc)
HALF=$(( CORES / 2 ))
while true; do
  for lane in A B; do
    if [ "$lane" = A ]; then gpus=0,1,2,3; cpus="0-$((HALF - 1))"; owners="$OWNERS_A";
    else gpus=4,5,6,7; cpus="$HALF-$((CORES - 1))"; owners="$OWNERS_B"; fi
    busy=no
    for owner in $owners; do
      supervisorctl status "$owner" 2>/dev/null | grep -qE "RUNNING|STARTING" && busy=yes
    done
    [ "$busy" = yes ] && continue
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpus" 2>/dev/null | sort -n | tail -1)
    [ -z "$mem" ] && continue
    [ "$mem" -gt 0 ] && continue
    if mkdir "$CLAIMS/$lane" 2>/dev/null; then
      echo "$lane $gpus $cpus"
      exit 0
    fi
  done
  sleep 45
done

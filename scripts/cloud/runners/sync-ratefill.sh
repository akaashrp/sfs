#!/bin/bash
# Mirror finished cells off the instance while the campaign runs. An interruptible rental can lose
# its disk, and the completed-cell ledger is what makes a restart resume rather than redo, so both
# the ledger and the points are pushed to durable storage every few minutes.
#   BRIDGES_DEST  user@data.bridges2.psc.edu:/ocean/projects/cis250162p/aparthas/sfs_ratefill_incoming
#   SYNC_PERIOD_S interval between passes (default 300)
set -uo pipefail
DEST="${BRIDGES_DEST:?set BRIDGES_DEST to the rsync destination}"
PERIOD="${SYNC_PERIOD_S:-300}"
SRC=/workspace/sfs/ratefill
while true; do
  if [ -d "$SRC/state" ]; then
    # Points are ~50 MB each; --partial keeps a cut-off transfer resumable, --inplace avoids
    # doubling peak disk on the receiving side.
    rsync -az --partial --inplace --prune-empty-dirs \
      --include='*/' \
      --include='completed/***' \
      --include='*/cells/*/point.json' \
      --include='*/cells/*/audit.json' \
      --include='*/cells/*/salvage.json' \
      --include='*/instances.json' --include='*/hardware.json' --include='*/status.json' \
      --exclude='*' \
      "$SRC/state/" "$DEST/state/" \
      && echo "$(date -Is) synced $(find $SRC/state -name 'point.json' | wc -l) points, $(find $SRC/state -path '*completed/*.json' | wc -l) receipts" \
      || echo "$(date -Is) sync failed (will retry)"
  fi
  sleep "$PERIOD"
done

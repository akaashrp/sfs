#!/bin/bash
# Put a mirrored campaign back on a replacement instance so the worker resumes instead of redoing.
# Pulls the completed-cell ledger and the points from durable storage into /workspace/sfs/ratefill.
#   BRIDGES_DEST  same value the sync used
set -euo pipefail
DEST="${BRIDGES_DEST:?set BRIDGES_DEST to the rsync source used by sync-ratefill.sh}"
mkdir -p /workspace/sfs/ratefill/state
rsync -az --partial "$DEST/state/" /workspace/sfs/ratefill/state/
echo "restored $(find /workspace/sfs/ratefill/state -path '*completed/*.json' | wc -l) completed-cell receipts"
echo "the runner will skip those cells and continue with the rest"

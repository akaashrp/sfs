#!/usr/bin/env bash
# Pull the finished campaign off the box and stage it where the results builder already looks.
#
# Runs from Bridges, which has no rsync or scp, so the transfer is a tar stream over ssh. The
# builder reads sfs_work/allpoints/ratefill/state, so the tree lands there directly; a dated
# tarball is kept beside it because the instance is destroyed once this verifies.
set -euo pipefail
HOST="${HOST:-sfs-ratefill}"
STAMP="$(date +%Y%m%d-%H%M)"
ARCHIVE_DIR=/ocean/projects/cis250162p/aparthas/sfs_ratefill_archive_20260929
STAGE=/ocean/projects/cis250162p/aparthas/sfs_work/allpoints/ratefill
mkdir -p "$ARCHIVE_DIR" "$STAGE"

echo "== what the box holds"
ssh "$HOST" 'echo "points:   $(find /workspace/sfs/ratefill/state -name point.json | wc -l)"
echo "receipts: $(find /workspace/sfs/ratefill/state -path "*completed/*.json" | wc -l)"
echo "audits:   $(find /workspace/sfs/ratefill/state -name audit.json | wc -l)"
echo "salvages: $(find /workspace/sfs/ratefill/state -name salvage.json | wc -l)"
echo "size:     $(du -sh /workspace/sfs/ratefill | cut -f1)"'

echo "== pulling the whole campaign directory"
TARBALL="$ARCHIVE_DIR/ratefill-$STAMP.tar.gz"
ssh "$HOST" 'tar -czf - -C /workspace/sfs ratefill' > "$TARBALL"
echo "wrote $TARBALL ($(du -h "$TARBALL" | cut -f1))"

echo "== verifying the tarball before anything is destroyed"
tar -tzf "$TARBALL" > "$ARCHIVE_DIR/ratefill-$STAMP.list"
echo "entries: $(wc -l < "$ARCHIVE_DIR/ratefill-$STAMP.list")"
grep -c 'point.json$'  "$ARCHIVE_DIR/ratefill-$STAMP.list" | sed 's/^/points in tarball: /'

echo "== staging for build_results.py"
tmp=$(mktemp -d /ocean/projects/cis250162p/aparthas/.scratch/ratefill-unpack.XXXXXX)
tar -xzf "$TARBALL" -C "$tmp"
mkdir -p "$STAGE"
# The mirrored copy was staged here during the run for a preview; replace it wholesale with the
# archive, which is the authoritative copy. Copying onto an existing state/ would nest it.
rm -rf "$STAGE/state"
cp -a "$tmp/ratefill/state" "$STAGE/"
rm -rf "$tmp"
echo "staged: $(find "$STAGE/state" -name point.json | wc -l) points under $STAGE/state"
echo "next: python sfs_work/paper_build/build_results.py   (ratefill/state is already in POOL_ROOTS)"
echo ARCHIVE_DONE

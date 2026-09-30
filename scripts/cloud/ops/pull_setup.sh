set -uo pipefail
DEST=/ocean/projects/cis250162p/aparthas/sfs_ratefill_archive_20260929
OUT=$DEST/setup-20260930.tar.gz
# Everything from the box's setup dir except what is large and re-obtainable: the Miniforge
# installer, the bundle inputs and the September runtime tarball all already exist on Bridges.
# The built wheel does NOT -- it is the exact binary these results were produced with, and
# rebuilding it costs ~37 min of GPU time.
ssh sfs-ratefill 'tar -czf - -C /workspace/sfs \
  --exclude="setup/Miniforge3-*.sh" \
  --exclude="setup/bundle-inputs.tar.gz" \
  --exclude="setup/vast-runtime-artifacts.tar.gz" \
  --exclude="setup/__pycache__" \
  setup' > "$OUT"
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
gzip -t "$OUT" && echo "gzip stream OK"
tar -tzf "$OUT" > "$DEST/setup-20260930.list"
echo "entries: $(wc -l < "$DEST/setup-20260930.list")"
grep -c '\.whl$' "$DEST/setup-20260930.list" | sed 's/^/wheels captured: /'
grep -E '\.whl$|\.log$' "$DEST/setup-20260930.list" | head -8 | sed 's/^/  /'
echo SETUP_PULL_DONE

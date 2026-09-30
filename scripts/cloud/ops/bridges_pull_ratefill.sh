#!/usr/bin/env bash
# Mirror the rate-fill campaign from the rented box to Bridges, pulling rather than pushing.
#
# The repo's sync-ratefill.sh pushes with rsync; Bridges has neither rsync nor scp installed, so
# the mirror runs from this side over ssh and tar alone. Each pass lists the campaign's artefacts
# on the box, compares name and size against what is already here, and fetches only what is new
# or has grown -- points are ~50 MB each, so re-sending the whole set every pass would waste the
# shared ssh proxy the campaign itself depends on.
#
#   HOST          ssh target for the box (default sfs-ratefill)
#   DEST          local mirror root (default /ocean/.../sfs_ratefill_incoming)
#   PERIOD_S      seconds between passes (default 900)
#   ONCE=1        single pass, then exit
set -uo pipefail
HOST="${HOST:-sfs-ratefill}"
DEST="${DEST:-/ocean/projects/cis250162p/aparthas/sfs_ratefill_incoming}"
PERIOD_S="${PERIOD_S:-900}"
SRC=/workspace/sfs/ratefill/state
mkdir -p "$DEST/state"

pass() {
  local remote_list want fetched=0
  remote_list=$(ssh -o BatchMode=yes "$HOST" "cd $SRC 2>/dev/null && find . \
      \( -name point.json -o -name audit.json -o -name salvage.json \
      -o -name instances.json -o -name hardware.json -o -name status.json \
      -o -path '*/completed/*.json' \) -type f -printf '%s %p\n'" 2>/dev/null)
  if [[ -z "$remote_list" ]]; then
    echo "$(date -Is) nothing to mirror yet"
    return 0
  fi
  # Keep a file only if we lack it locally or our copy is a different size.
  want=$(while read -r size path; do
           local_path="$DEST/state/${path#./}"
           if [[ ! -f "$local_path" ]] || [[ "$(stat -c %s "$local_path")" != "$size" ]]; then
             printf '%s\n' "$path"
           fi
         done <<< "$remote_list")
  if [[ -z "$want" ]]; then
    echo "$(date -Is) mirror current ($(wc -l <<< "$remote_list") files)"
    return 0
  fi
  fetched=$(wc -l <<< "$want")
  printf '%s\n' "$want" \
    | ssh -o BatchMode=yes "$HOST" "cd $SRC && tar -czf - -T -" \
    | tar -xzf - -C "$DEST/state" \
    && echo "$(date -Is) mirrored $fetched new/changed file(s); receipts here: $(find "$DEST/state" -path '*completed/*.json' | wc -l)" \
    || echo "$(date -Is) mirror pass failed (will retry)"
}

if [[ "${ONCE:-0}" == 1 ]]; then pass; exit 0; fi
echo "mirroring $HOST:$SRC -> $DEST/state every ${PERIOD_S}s"
while true; do pass; sleep "$PERIOD_S"; done

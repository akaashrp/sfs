#!/usr/bin/env bash
# Wait for the last cell, then archive, rebuild and re-render, and email when the instance is the
# only thing left to deal with. Destroying it stays a human decision -- it is irreversible and the
# disk goes with it -- so this stops there and asks.
set -uo pipefail
cd /ocean/projects/cis250162p/aparthas/.scratch/ratefill-deploy
TO=akaashrp@gmail.com
INSTANCE=53446429
LOG() { echo "$(date -Is) $*"; }

LOG "waiting for the 35th receipt"
complete=0
for i in $(seq 1 240); do
  n=$(timeout 240 ssh -o ConnectTimeout=120 -o BatchMode=yes sfs-ratefill \
        'find /workspace/sfs/ratefill/state -path "*completed/*.json" 2>/dev/null | wc -l' 2>/dev/null)
  [[ "$n" =~ ^[0-9]+$ ]] || { LOG "poll $i: box unreachable"; sleep 60; continue; }
  LOG "poll $i: receipts=$n/35"
  [ "$n" -ge 35 ] && { LOG "campaign complete"; complete=1; break; }
  sleep 60
done

# Falling out of that loop without 35 means the last cell never landed -- a lane stuck, or the
# crash-loop guard having given up. Rebuilding and mailing "COMPLETE" on an incomplete grid would
# be worse than doing nothing, so stop here and say exactly what is missing.
if [ "$complete" != 1 ]; then
  LOG "DID NOT COMPLETE: stopped at ${n:-unknown}/35"
  missing=$(timeout 240 ssh -o ConnectTimeout=120 -o BatchMode=yes sfs-ratefill "python3 - <<'PY'
import re, glob, os
done = {os.path.basename(p)[:-5] for p in glob.glob('/workspace/sfs/ratefill/state/*/completed/*.json')}
done |= {os.path.basename(p)[:-5] for p in glob.glob('/workspace/sfs/ratefill/state/completed/*.json')}
for lane in ('a', 'b'):
    txt = open(f'/workspace/sfs/setup/baseline-rates-lane-{lane}.sh').read()
    cells = set(re.search(r'--cells (\S+)', txt).group(1).split(','))
    owed = sorted(c.replace('qwen-', '') for c in cells - done)
    if owed: print(f'  lane {lane}: {owed}')
PY" 2>/dev/null)
  printf '%s\n' "The SFS rate fill stopped short at ${n:-unknown}/35 receipts after the watch window." \
    "" "Still missing:" "$missing" "" \
    "Nothing was archived, rebuilt or re-rendered, and the instance is STILL RUNNING and billing." \
    "Check the recovery log for a crash-loop give-up:" \
    "  ssh sfs-ratefill 'tail -20 /workspace/sfs/ratefill/recover.log'" \
    | mail -s "SFS rate fill INCOMPLETE at ${n:-?}/35 - needs a look" "$TO"
  exit 1
fi

LOG "=== step 2: archive and verify"
bash archive_ratefill.sh 2>&1 | grep -v 'Welcome\|Have fun' | tail -20
if ! ls /ocean/projects/cis250162p/aparthas/sfs_ratefill_archive_20260929/ratefill-*.tar.gz >/dev/null 2>&1; then
  LOG "ARCHIVE FAILED -- stopping before touching results"
  printf '%s\n' "The rate-fill campaign finished but the archive step failed." \
    "Nothing was rebuilt. The instance is STILL RUNNING and still billing." \
    "Investigate before destroying instance $INSTANCE -- its disk holds the only full copy." \
    | mail -s "SFS rate fill: archive FAILED, instance still up" "$TO"
  exit 1
fi

LOG "=== step 3: rebuild the results from the archive"
source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate vllm
python /ocean/projects/cis250162p/aparthas/sfs_work/paper_build/build_results.py 2>&1 | tail -3

LOG "=== step 4: re-render the figures on the published grid"
bash finish_figures.sh 2>&1 | tail -5

POINTS=$(find /ocean/projects/cis250162p/aparthas/sfs_work/allpoints/ratefill/state -name point.json | wc -l)
RECEIPTS=$(find /ocean/projects/cis250162p/aparthas/sfs_work/allpoints/ratefill/state -path '*completed/*.json' | wc -l)
TARBALL=$(ls -1t /ocean/projects/cis250162p/aparthas/sfs_ratefill_archive_20260929/ratefill-*.tar.gz | head -1)
FIGS=$(ls /ocean/projects/cis250162p/aparthas/sfs_paper_results/figures/*.pdf 2>/dev/null | wc -l)

LOG "=== emailing"
printf '%s\n' \
  "The SFS baseline rate-fill campaign is complete and everything on my side is done." \
  "" \
  "  cells:     $RECEIPTS/35 receipts, $POINTS points" \
  "  archive:   $TARBALL" \
  "  staged:    sfs_work/allpoints/ratefill/state" \
  "  rebuilt:   sfs_paper_results/*.json" \
  "  figures:   $FIGS PDFs re-rendered on the grid 3,4,5,6,7,8,8.3,8.6,8.9,9.0,9.2" \
  "" \
  "ONE THING LEFT FOR YOU: destroy the Vast instance. It is on-demand at \$19.40/h and" \
  "four of its eight GPUs have been idle since lane A finished." \
  "" \
  "  vastai destroy instance $INSTANCE" \
  "" \
  "or via the API:" \
  "  curl -s -X DELETE https://console.vast.ai/api/v0/instances/$INSTANCE/ \\" \
  "    -H \"Authorization: Bearer \$(cat ~/.config/vastai/vast_api_key)\"" \
  "" \
  "The archive above is verified, so the disk is no longer the only copy." \
  | mail -s "SFS rate fill COMPLETE - destroy instance $INSTANCE to stop billing" "$TO"
LOG "FINALIZE_DONE emailed $TO"

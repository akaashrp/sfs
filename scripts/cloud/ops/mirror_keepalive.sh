#!/usr/bin/env bash
# Keep the off-box mirror running. It was a bare nohup, so anything that killed it -- a dropped
# session, an ssh failure cascade -- left the campaign with no copy off the rented disk and nothing
# to notice. The mirror is only insurance, but insurance that silently stops is worse than none.
cd /ocean/projects/cis250162p/aparthas/.scratch/ratefill-deploy
while true; do
  if ! pgrep -f "bridges_pull_ratefil[l]" >/dev/null; then
    echo "$(date -Is) mirror absent; relaunching"
    nohup env HOST=sfs-ratefill PERIOD_S=900 \
      bash /ocean/projects/cis250162p/aparthas/sfs_work/bridges_pull_ratefill.sh >> mirror.log 2>&1 &
  fi
  sleep 120
done

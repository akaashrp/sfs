#!/bin/bash
# The worker refuses a CPU gate whose recorded source hash is not the running source, so every
# runner refreshes the gate before claiming a lane. Editing the checkout invalidates it.
set -euo pipefail
export SFS_STORAGE=/workspace/sfs
REPO="${SFS_REPO:?set SFS_REPO to the checkout directory name under /workspace/sfs}"
GATE="${GATE_JSON:-/workspace/sfs/ratefill/setup/tests/gate.json}"
PROGRAM="${GATE_PROGRAM:-sfs-ratefill-gates}"
source "/workspace/sfs/$REPO/scripts/cloud/env.sh"
cd "/workspace/sfs/$REPO"
CURRENT=$(python -c "
import json,sys
sys.path.insert(0,'src')
from scripts.cloud.common import source_hashes
print(json.dumps(source_hashes(), sort_keys=True))" 2>/dev/null || echo '')
RECORDED=$(python -c "
import json
try: print(json.dumps(json.load(open('$GATE'))['source_sha256'], sort_keys=True))
except Exception: print('')" 2>/dev/null)
if [ -n "$CURRENT" ] && [ "$CURRENT" = "$RECORDED" ]; then echo "gate current"; exit 0; fi
echo "gate stale; rerunning"
supervisorctl start "$PROGRAM" >/dev/null 2>&1 || true
for i in $(seq 1 90); do
  supervisorctl status "$PROGRAM" | grep -q RUNNING || break
  sleep 20
done
echo "gate refreshed"

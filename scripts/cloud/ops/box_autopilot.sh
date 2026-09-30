#!/usr/bin/env bash
# Start both lanes as soon as the box is bootstrapped. Each runner refreshes the source-bound CPU
# gate itself, claims a lane, and releases it on exit, so nothing here needs to sequence them.
set -uo pipefail
SUP="supervisorctl -c /etc/supervisor/supervisord.conf"
while [[ ! -f /workspace/sfs/bootstrap-complete ]]; do sleep 30; done
echo "bootstrap complete at $(date -Is); starting lanes"
$SUP start sfs-ratefill-a
$SUP start sfs-ratefill-b
$SUP status
echo AUTOPILOT_STARTED_LANES

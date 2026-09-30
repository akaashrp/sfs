#!/usr/bin/env bash
# Install the campaign runners under supervisor once the environment, models and bundle are in
# place, then mark the box bootstrapped so the guarded runners will start.
set -euo pipefail
export SFS_STORAGE=/workspace/sfs
R=/workspace/sfs/repo-ratefill
for marker in setup/environment-ready setup/models-ready; do
  while [[ ! -f "$SFS_STORAGE/$marker" ]]; do sleep 20; done
done
while [[ ! -f "$SFS_STORAGE/bundle/bundle.json" ]]; do sleep 20; done

mkdir -p /workspace/sfs/setup /workspace/sfs/ratefill/{state,claims} /etc/supervisor/conf.d
cp "$R"/scripts/cloud/runners/*.sh /workspace/sfs/setup/
chmod +x /workspace/sfs/setup/*.sh

# Supervisor programs need SFS_REPO; the helpers take the checkout name as an input.
# LANE_CORES holds each lane to the core count the cells being extended were measured with, so a
# box with more cores than the September one does not quietly change the contention.
# BRIDGES_DEST, when set, lets the mirror program run.
export LANE_CORES="${LANE_CORES:-48}"
export BRIDGES_DEST="${BRIDGES_DEST:-}"
python3 - <<'PY'
import os
from pathlib import Path
src = Path('/workspace/sfs/repo-ratefill/scripts/cloud/runners/baseline-rates-supervisor.conf')
env = ['SFS_REPO="repo-ratefill"', 'HOME="/root"', f'LANE_CORES="{os.environ["LANE_CORES"]}"']
dest = os.environ.get('BRIDGES_DEST', '')
if dest:
    env.append(f'BRIDGES_DEST="{dest}"')
out = []
for line in src.read_text().splitlines():
    out.append(line)
    if line.startswith('command='):
        out.append('environment=' + ','.join(env))
Path('/etc/supervisor/conf.d/sfs-ratefill.conf').write_text('\n'.join(out) + '\n')
print('wrote /etc/supervisor/conf.d/sfs-ratefill.conf with LANE_CORES=' + os.environ['LANE_CORES']
      + (' and a mirror destination' if dest else ' and no mirror destination'))
PY

if ! pgrep -x supervisord >/dev/null; then
  cat > /etc/supervisor/supervisord.conf <<'CONF'
[unix_http_server]
file=/var/run/supervisor.sock
chmod=0700
[supervisord]
logfile=/workspace/sfs/setup/supervisord.log
pidfile=/var/run/supervisord.pid
nodaemon=false
[rpcinterface:supervisor]
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface
[supervisorctl]
serverurl=unix:///var/run/supervisor.sock
[include]
files = /etc/supervisor/conf.d/*.conf
CONF
  supervisord -c /etc/supervisor/supervisord.conf
  sleep 3
fi
supervisorctl -c /etc/supervisor/supervisord.conf reread || true
supervisorctl -c /etc/supervisor/supervisord.conf update || true
touch /workspace/sfs/bootstrap-complete
supervisorctl -c /etc/supervisor/supervisord.conf status || true
echo INSTALL_DONE

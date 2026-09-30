#!/bin/bash
# Keep the recovery loop alive. It is launched detached, so it survives supervisord dying; this
# watchdog covers the other direction, the loop itself dying. Between them, either one failing
# alone is recoverable without a human.
while true; do
  if ! pgrep -f "box_recove[r].py" >/dev/null; then
    echo "$(date -Is) recovery loop absent; relaunching"
    setsid nohup python3 /workspace/sfs/setup/box_recover.py >> /workspace/sfs/setup/box_recover.log 2>&1 < /dev/null &
  fi
  sleep 60
done

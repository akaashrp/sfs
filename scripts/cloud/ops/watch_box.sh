#!/usr/bin/env bash
# Emit one line whenever the box's bootstrap/campaign state changes. Polls over the ssh proxy, so
# it must tolerate a refused connection without dying, and must report failures as loudly as
# progress: a silent monitor and a crashed stage look identical otherwise.
prev=""
for i in $(seq 1 2000); do
  s=$(timeout 180 ssh -o ConnectTimeout=90 -o BatchMode=yes sfs-ratefill '
    for m in environment-ready models-ready; do [ -f /workspace/sfs/setup/$m ] && printf "%s " "$m"; done
    [ -f /workspace/sfs/bootstrap-complete ] && printf "bootstrap-complete "
    [ -f /workspace/sfs/ratefill/gates-ready ] && printf "gates-ready "
    printf "points=%s receipts=%s " \
      "$(find /workspace/sfs/ratefill/state -name point.json 2>/dev/null | wc -l)" \
      "$(find /workspace/sfs/ratefill/state -path "*completed/*.json" 2>/dev/null | wc -l)"
    printf "gpumem=%s " "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd, -)"
    sup=$(supervisorctl -c /etc/supervisor/supervisord.conf status 2>/dev/null | awk "{print \$1\"=\"\$2}" | paste -sd, -)
    printf "sup=[%s] " "${sup:-none}"
    # A pool that has qualified but has no release is parked with its GPUs warm and idle. That cost
    # 25 minutes the first time because nothing watched for it.
    for q in /workspace/sfs/ratefill/state/lane-*/qualification.json; do
      [ -f "$q" ] || continue
      [ -f "$(dirname "$q")/release.json" ] || printf "AWAITING_RELEASE:%s " "$(basename "$(dirname "$q")")"
    done
    # finish_bootstrap drives the gate and the lane start. Without this line a failed gate looks
    # exactly like a gate still running: everything stays STOPPED and the watcher says nothing.
    if [ -f /workspace/sfs/setup/finish_bootstrap.log ]; then
      printf "finish=%s " "$(grep -oE "waiting for environment-ready|environment-ready seen|models step|install step|gate \(running|gate passed|GATE FAILED|LANES_STARTED" /workspace/sfs/setup/finish_bootstrap.log | tail -1 | tr " " "_")"
      grep -q "GATE FAILED" /workspace/sfs/setup/finish_bootstrap.log && printf "FAIL:gate "
    fi
    for l in a b; do
      [ -f /workspace/sfs/ratefill/lane-$l.log ] && printf "lane%s=%s " "$l" \
        "$(grep -oE "claimed [AB]|gate current|gate refreshed|cell [^ ]+|Traceback" /workspace/sfs/ratefill/lane-$l.log | tail -1 | tr " " "_")"
    done
    if [ -f /workspace/sfs/setup/box_build_vllm.log ]; then
      printf "build=%s " "$(grep -oE "=== (checkout|build deps|drop the stale wheel|build \(Hopper only\)|verify)|Building wheel|BUILD_DONE" /workspace/sfs/setup/box_build_vllm.log | tail -1 | tr " " "_")"
    fi
    for f in box_build_vllm box_models box_install; do
      grep -qE "Traceback|^ERROR|No space left|error:|Failed to build" /workspace/sfs/setup/$f.log 2>/dev/null && printf "FAIL:%s " "$f"
    done
    # Only faults from the CURRENT run: supervisor appends, so the logs still hold the tracebacks
    # from the attempts that failed before the claim fixes. An unscoped grep would flag those
    # forever, which is the same as not watching at all.
    for l in a b; do
      f=/workspace/sfs/ratefill/lane-$l.log
      [ -f "$f" ] || continue
      n=$(awk "/claimed [AB] gpus=/{keep=1; buf=\"\"} keep{buf=buf \$0 ORS} END{printf \"%s\", buf}" "$f" \
          | grep -cE "Traceback|No space left|CUDA out of memory|RuntimeError")
      [ "$n" -gt 0 ] && printf "LANEFAIL:%s=%s " "$l" "$n"
    done
  ' 2>/dev/null)
  if [ -z "$s" ]; then s="control channel unavailable"; fi
  # Fire on receipts, lane state, parked pools and faults -- not on the points counter. Every smoke
  # point and every cell bumps points, which over the rest of this campaign is dozens of events that
  # say nothing actionable.
  key=$(sed -E 's/points=[0-9]+ //; s/gpumem=[0-9,]+ //' <<< "$s")
  if [ "$key" != "$prev" ]; then echo "$(date +%H:%M) $s"; prev="$key"; fi
  sleep 60
done

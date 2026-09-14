#!/usr/bin/env python3
"""Pull cloud logs/results to durable controller storage, without remote credentials."""
import argparse
import json
from pathlib import Path
import re
import os
import shutil
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',required=True);p.add_argument('--remote-state',required=True)
    p.add_argument('--destination',required=True);p.add_argument('--watch',action='store_true')
    a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',a.host):p.error('Use an SSH config alias')
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+',a.remote_state):p.error('Use a simple absolute remote state path')
    dest=Path(a.destination).resolve()/a.host;dest.mkdir(parents=True,exist_ok=True)
    binary=os.environ.get('SFS_RSYNC') or shutil.which('rsync') or str(Path(__file__).resolve().parents[2]/'.tools/bin/rsync')
    if not Path(binary).is_file():p.error('Install rsync on the controller or set SFS_RSYNC to its absolute path')
    while True:
        command=[binary,'-a','--partial','--protect-args','--exclude=ipc/','--exclude=cache/',
                 '--rsync-path',str(Path(a.remote_state).parent/'miniforge/envs/vllm/bin/rsync'),
                 '-e','ssh -o BatchMode=yes -o ConnectTimeout=20',f'{a.host}:{a.remote_state}/',str(dest)+'/']
        result=subprocess.run(command)
        (dest/'controller_sync_status.json').write_text(json.dumps({'time':time.time(),'returncode':result.returncode})+'\n')
        if not a.watch:raise SystemExit(result.returncode)
        time.sleep(60)


if __name__=='__main__':main()

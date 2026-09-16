#!/usr/bin/env python3
"""Deploy the frozen release over SSH; rent instances separately in the provider UI."""
import argparse
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


def ssh(host, script):
    subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20',host,
                    'bash -lc '+shlex.quote(script)],check=True)


def transfer(host, source, destination):
    """Use an SSH stream when the controller does not provide scp."""
    if shutil.which('scp'):
        subprocess.run(['scp', str(source), f'{host}:{destination}'], check=True)
    else:
        with Path(source).open('rb') as stream:
            subprocess.run(['ssh', '-o', 'BatchMode=yes', host,
                            'cat > ' + shlex.quote(destination)], stdin=stream, check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['check','deploy','bootstrap','models','status'])
    p.add_argument('--host',required=True);p.add_argument('--storage',required=True)
    p.add_argument('--archive');p.add_argument('--archive-sha256');p.add_argument('--family',default='all',choices=['all','qwen','ministral'])
    a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',a.host):p.error('Use an SSH config alias')
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+',a.storage):p.error('Use an absolute storage path without spaces')
    root=Path(__file__).resolve().parents[2]
    release=json.loads((root/'scripts/cloud/release.json').read_text())
    storage=shlex.quote(a.storage)
    prefix=f'export SFS_STORAGE={storage}; export SFS_BUNDLE="$SFS_STORAGE/bundle"; '
    activated=prefix+'source "$SFS_STORAGE/repo/scripts/cloud/env.sh"; '
    if a.mode=='check':
        ssh(a.host,f'mkdir -p {storage}; df -h {storage} /dev/shm; nvidia-smi; nvidia-smi topo -m; command -v git; command -v curl')
    elif a.mode=='deploy':
        if not a.archive or not a.archive_sha256:p.error('Deployment needs the frozen archive and SHA256')
        if not re.fullmatch('[a-f0-9]{64}',a.archive_sha256):p.error('Invalid archive SHA256')
        ssh(a.host,f'mkdir -p {storage}/transfer')
        transfer(a.host, Path(a.archive).resolve(), f'{a.storage}/transfer/inputs.tar.gz')
        script=prefix+f'''set -euo pipefail
cd "$SFS_STORAGE/transfer"
echo '{a.archive_sha256}  inputs.tar.gz' | sha256sum -c -
if [[ ! -d "$SFS_STORAGE/repo" ]]; then
 git clone --branch {shlex.quote(release['sfs_branch'])} https://github.com/akaashrp/sfs.git "$SFS_STORAGE/repo"
fi
cd "$SFS_STORAGE/repo"
git fetch origin {shlex.quote(release['sfs_branch'])}
git checkout --detach {shlex.quote(release['sfs_commit'])}
git submodule update --init --recursive
[[ "$(git -C vllm rev-parse HEAD)" == {shlex.quote(release['vllm_commit'])} ]]
[[ ! -e "$SFS_STORAGE/bundle" ]] || {{ echo 'Bundle already exists; preserve it and use bootstrap/status' >&2; exit 1; }}
mkdir "$SFS_STORAGE/bundle"
tar -xzf "$SFS_STORAGE/transfer/inputs.tar.gz" -C "$SFS_STORAGE/bundle" --no-same-owner
'''
        ssh(a.host,script)
    elif a.mode=='bootstrap':ssh(a.host,prefix+'bash "$SFS_STORAGE/repo/scripts/cloud/bootstrap.sh"')
    elif a.mode=='models':
        ssh(a.host,activated+f'export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0; python -m scripts.cloud.prepare models --bundle "$SFS_BUNDLE" --cache "$HF_HUB_CACHE" --output "$SFS_STORAGE/models.json" --family {a.family}')
    else:ssh(a.host,activated+'python -m scripts.cloud.control status --state "$SFS_STORAGE/state" --bundle "$SFS_BUNDLE"')


if __name__=='__main__':main()

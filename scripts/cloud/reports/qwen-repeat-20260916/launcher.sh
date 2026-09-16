#!/bin/bash
set -euo pipefail
export SFS_STORAGE=/workspace/sfs
[[ -f "$SFS_STORAGE/setup/bootstrap-complete" ]]
[[ -f "$SFS_STORAGE/models.json" ]]
source "$SFS_STORAGE/repo/scripts/cloud/env.sh"
exec taskset -c 0-47 python -m scripts.cloud.canonical_control run --bundle "$SFS_STORAGE/bundle" --models "$SFS_STORAGE/models.json" --state "$SFS_STORAGE/state" --output "$SFS_STORAGE/state/controls/qps8p6-repeat1" --qps 8.6 --gpus 0,1,2,3

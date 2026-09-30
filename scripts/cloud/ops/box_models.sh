#!/usr/bin/env bash
# The three Qwen checkpoints at the exact revisions the September cells used, so the new rates
# are measured against identical weights. Waits for the environment, then writes models.json.
set -euo pipefail
export SFS_STORAGE=/workspace/sfs
while [[ ! -f "$SFS_STORAGE/setup/environment-ready" ]]; do sleep 20; done
export SFS_ROOT=/workspace/sfs/repo-ratefill
set +u; source "$SFS_STORAGE/miniforge/etc/profile.d/conda.sh"; conda activate vllm; set -u
export HF_HOME="$SFS_STORAGE/hf" HF_HUB_CACHE="$SFS_STORAGE/hf/hub"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HUB_ENABLE_HF_TRANSFER=1
python - <<'PY'
import json, os
from huggingface_hub import snapshot_download
WANT = {
    'qwen3-0.6b': ('Qwen/Qwen3-0.6B', 'c1899de289a04d12100db370d81485cdf75e47ca'),
    'qwen3-8b':   ('Qwen/Qwen3-8B',   'b968826d9c46dd6066d109eabc6255188de91218'),
    'qwen3-32b':  ('Qwen/Qwen3-32B',  '9216db5781bf21249d130ec9da846c4624c16137'),
}
paths = {}
for label, (repo, rev) in WANT.items():
    print('downloading', label, repo, rev, flush=True)
    paths[label] = snapshot_download(repo_id=repo, revision=rev, max_workers=16)
    print('done', label, paths[label], flush=True)
with open('/workspace/sfs/models.json', 'w') as fh:
    json.dump(paths, fh, indent=2)
print(json.dumps(paths, indent=2))
PY
touch "$SFS_STORAGE/setup/models-ready"
echo MODELS_DONE

#!/usr/bin/env bash
#SBATCH --job-name=sfs-flash-calibration
#SBATCH --output=experiments/flash_calibration_%j.out
#SBATCH --error=experiments/flash_calibration_%j.err
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16000M
#SBATCH --time=16:00:00
set -euo pipefail
: "${SFS_ROOT:?}" "${FLASH_ROOT:?}" "${SOURCE_CHECKSUMS:?}" "${CANONICAL_MANIFEST:?}" "${QWEN_STAGE:?}" "${QWEN_CPU_REPORT:?}" "${FLASH_VARIANT_MANIFEST:?}"
source "$SFS_ROOT/src/slurm/prep/paper_ablation_env.sh"
export PYTHONPATH="$SFS_ROOT/vllm:$PYTHONPATH"
sha256sum --status -c "$SOURCE_CHECKSUMS"
python - "$FLASH_ROOT" <<'PY'
from pathlib import Path
import json,sys
from scripts.prep.flash_calibration import verify_files
p=Path(sys.argv[1]);m=json.loads((p/'prepared.json').read_text());verify_files(m['file_sha256'])
a=json.loads((p/'canary_audit.json').read_text())
assert a['status']=='PASS_API_CANARY' and len(a['buckets'])==4
assert all(v['candidates']==3 and v['imputed']==0 for v in a['buckets'].values())
assert not (p/'training_audit.json').exists(),'Training already completed; do not overwrite'
print('PASS_CPU_AND_API_CANARY')
PY
if [[ "${VALIDATE_ONLY:-0}" == 1 ]]; then exit 0; fi
set +x
set -a
source /jet/home/aparthas/.config/sfs/gemini.env
set +a
python -m scripts.prep.flash_calibration run --output "$FLASH_ROOT"
python -m scripts.prep.flash_calibration train --output "$FLASH_ROOT" --predictor-root "$SFS_ROOT/src/assets/predictors"
python -m scripts.runs.qwen_predictor_variants prepare --root "$SFS_ROOT" --arm flash_quality \
 --canonical "$CANONICAL_MANIFEST" --stage "$QWEN_STAGE" --cpu-report "$QWEN_CPU_REPORT" \
 --flash-root "$FLASH_ROOT" --manifest "$FLASH_VARIANT_MANIFEST"
sha256sum --status -c "$SOURCE_CHECKSUMS"

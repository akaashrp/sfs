"""Render the two frozen Bridges launchers for the Ministral hard_prefill_tps cells.

Frozen-launcher precedent (experiments/bridges_ptps_20260917/render_jobs.py):
literal settings are baked in, the additive side module and its CPU-prepared
side manifest are hash-pinned, and each script exits after its CPU gates when
SFS_PREFLIGHT_ONLY=1 so it can be rehearsed on a login node before sbatch.
The pool is served exactly as the recovery/Figure 5 jobs serve it: the frozen
ministral3_router_common.sh path (three models, TP 1/1/1 on three H100s, the
audited profile and served aliases). SFS_ROOT for that launcher is this
worktree so its vLLM commit check binds to the worktree's vLLM (28bbf92), the
same provisioning the Qwen chain and the cancelled recovery checkout use.
"""
import hashlib
import json
from pathlib import Path

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[2]
JOBS = EXP.parent/'jobs'
MF = Path('/ocean/projects/cis250162p/aparthas/sfs_model_family')
LEGACY = MF/'experiments/campaign_cold_start_fix_20260912/ministral_figure5_manifest.json'
SERVICE = MF/'experiments/ministral3_h100-80_service_metrics_45088929_20260905_084501'
PREDICTORS = MF/'experiments/ministral3_paper/predictors/run_45089699'
SIDE = ROOT/'src/scripts/runs/ministral3_hard_prefill_tps.py'
MANIFEST = EXP/'side/manifest.json'
WALLTIME = {'smoke': '02:30:00', 'sweep': '05:00:00'}
WAIT_LOGS = '--wait-log "$MINISTRAL_WAIT_LOG_3B" --wait-log "$MINISTRAL_WAIT_LOG_8B" --wait-log "$MINISTRAL_WAIT_LOG_14B"'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def script(stage):
    name = f'sfs-ptps-ministral-{stage}'
    body = f"""#!/usr/bin/env bash
#SBATCH --job-name={name}
#SBATCH --output={EXP}/logs/%x_%j.out
#SBATCH --error={EXP}/logs/%x_%j.err
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100-80:3
#SBATCH --time={WALLTIME[stage]}

set -euo pipefail
# Frozen side source and CPU-prepared side manifest: the job refuses to run if either changed after submission.
printf '%s\\n' '{sha(SIDE)}  {SIDE}' | sha256sum --status -c
printf '%s\\n' '{sha(MANIFEST)}  {MANIFEST}' | sha256sum --status -c
export SFS_ROOT={ROOT}
export EXP={EXP}
export SIDE_MANIFEST={MANIFEST}
export SERVICE_RUN_DIR={SERVICE}
export PREDICTOR_RUN_DIR={PREDICTORS}
export USER="${{USER:-$(id -un)}}"
export SLURM_EXPORT_ENV=ALL
source /opt/packages/anaconda3/etc/profile.d/conda.sh
conda activate vllm
export PYTHONPATH="$SFS_ROOT/vllm:$SFS_ROOT/src:$SFS_ROOT/experiments/methodology_baselines/deps"
export TMPDIR="$EXP/scratch"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$TMPDIR" "$EXP/preflight"
cd "$SFS_ROOT/src"
python -c "import vllm.vllm_flash_attn.layers.rotary, vllm.model_executor.layers.rotary_embedding, vllm.v1.engine, vllm.v1.engine._scheduler_sim"
python -m scripts.runs.ministral3_figure5 validate --manifest {LEGACY} \\
  --service-run-dir "$SERVICE_RUN_DIR" --predictor-run-dir "$PREDICTOR_RUN_DIR"
python -m scripts.runs.ministral3_hard_prefill_tps validate --manifest "$SIDE_MANIFEST" \\
  >"$EXP/preflight/{name}_${{SLURM_JOB_ID:-cpu}}.json"
if [[ "${{SFS_PREFLIGHT_ONLY:-0}}" == 1 ]]; then exit 0; fi

# Frozen Ministral serving path: three models, TP 1/1/1 on three H100s, audited profile and served aliases.
source "$SFS_ROOT/src/slurm/runs/ministral3_router_common.sh"
ministral3_resolve_sfs_root
ministral3_resolve_service_run
MINISTRAL_RUN_ROOT="$EXP/{stage}_pool_${{SLURM_JOB_ID:?}}"
[[ ! -e "$MINISTRAL_RUN_ROOT" ]] || {{ echo 'Preserve the existing run directory' >&2; exit 1; }}
mkdir -p "$MINISTRAL_RUN_ROOT"
ministral3_start_router_pool
export PYTHONPATH="$SFS_ROOT/vllm:$SFS_ROOT/src:$SFS_ROOT/experiments/methodology_baselines/deps"
cd "$SFS_ROOT/src"
"""
    if stage == 'smoke':
        body += f"""python -m scripts.runs.ministral3_hard_prefill_tps smoke --manifest "$SIDE_MANIFEST" \\
  --instances-config "$MINISTRAL_INSTANCES_CONFIG" --output "$EXP/smoke" \\
  {WAIT_LOGS}
"""
    else:
        body += f"""python -m scripts.runs.ministral3_hard_prefill_tps sweep --manifest "$SIDE_MANIFEST" \\
  --instances-config "$MINISTRAL_INSTANCES_CONFIG" --smoke-dir "$EXP/smoke" --output "$EXP/sweep" \\
  {WAIT_LOGS}
"""
    return name, body


def main():
    (EXP/'logs').mkdir(exist_ok=True)
    record = {}
    for stage in ('smoke', 'sweep'):
        name, body = script(stage)
        path = JOBS/f'{name}.sbatch'
        path.write_text(body)
        record[name] = {'path': str(path), 'sha256': sha(path)}
    (EXP/'scripts.json').write_text(json.dumps({'side_module_sha256': sha(SIDE), 'side_manifest_sha256': sha(MANIFEST),
                                                'scripts': record}, indent=2)+'\n')
    print(json.dumps(record, indent=1))


if __name__ == '__main__':
    main()

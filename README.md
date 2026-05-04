# SFS Experiment Workflow

This repository contains the experiment pipeline for serving framework simulation (SFS).

## Core Data-Split Policy (Critical)

Experiments are **not** run on the initial calibration prompts.

Define:
- `X` = calibration prefix size per bucket (default in current scripts: `X=2500`)
- `N` = holdout size per bucket (examples used here: `N=2000` or `N=4000`)

For each bucket:
1. Calibration/training pool uses indices `[0, X-1]`.
2. Holdout pool starts at index `X` and uses the next `N` prompts.
3. Router experiments run on this holdout pool only.

`holdout_prompts.py` preserves original `prompt_index` and builds request IDs with that index, so holdout requests remain disjoint from calibration/training data by construction.

## Repository Layout

| Path | Purpose |
|---|---|
| `src/scripts/prep` | Data prep, scoring, holdout cache, mapping scripts. |
| `src/scripts/runs` | Main experiment drivers (`experiments.py`, `experiments_sweep.py`). |
| `src/scripts/eval` | Post-run augmentation/diagnostics. |
| `src/scripts/reporting` | Sweep summary + plotting scripts. |
| `src/sfs_core` | Shared library code (routing, predictors, regression, helpers). |
| `src/slurm/prep` | SLURM wrapper for bucketed prompt generation. |
| `src/slurm/runs` | SLURM wrappers for calibration + sweeps. |
| `src/assets` | Static assets (template + predictor artifacts). |
| `experiments/data/prompts` | Generated prompt artifacts (bucket pools + holdout caches). |
| `experiments/` | Runtime outputs. |

## Environment Setup

```bash
cd /path/to/sfs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

For judge-based quality scoring, set `GOOGLE_API_KEY` or `GEMINI_API_KEY`.

## Standard Workflow

### 1) Aggregate prompts into bucket files

Create the full bucketed prompt pool once.

```bash
python -m scripts.prep.aggregate_buckets \
  --out_dir experiments/data/prompts/bucketed_prompts/qwen3-0.6b \
  --tokenizer Qwen/Qwen3-0.6B \
  --max_per_dataset 16000 \
  --max_per_bucket 16000 \
  --seed 69
```

### 2) Generate calibration outputs on first `X` prompts per bucket

Use the generation wrapper with `MAX_PROMPTS_PER_BUCKET=X`.

```bash
MAX_PROMPTS_PER_BUCKET=2500 sbatch src/slurm/prep/run_qwen_bucketed_prompts_all_models.sbatch
```

This writes timestamped outputs under:
`experiments/qwen_bucketed_all_models_<stamp>/completions/...`

Keep this `X` aligned with:
- `--holdout-start-index X` used later in sweeps.
- service-metric calibration assumptions (current `compute_model_service_metrics.py` defaults are set for `X=2500` and 4 buckets).

### 3) Normalize generation outputs to scoring layout

Scoring expects per-model JSONLs in:
`experiments/bucketed_prompt_outputs/<model>/*.jsonl`

Copy/symlink the calibration-generation JSONLs into that layout.

### 4) Score calibration outputs

```bash
python -m scripts.prep.quality_metrics \
  --outputs-root experiments/bucketed_prompt_outputs \
  --models qwen3-0.6b qwen3-8b qwen3-32b
```

### 5) Train prediction models on calibration data

Output-length predictor:

```bash
python -m sfs_core.predictors.output_length_model \
  --input \
    experiments/bucketed_prompt_outputs/qwen3-0.6b/scored \
    experiments/bucketed_prompt_outputs/qwen3-8b/scored \
    experiments/bucketed_prompt_outputs/qwen3-32b/scored \
  --output-dir src/assets/predictors/output_length_predictor \
  --test-fraction 0.1
```

Accuracy predictor:

```bash
python -m sfs_core.predictors.accuracy_model \
  --input \
    experiments/bucketed_prompt_outputs/qwen3-0.6b/scored \
    experiments/bucketed_prompt_outputs/qwen3-8b/scored \
    experiments/bucketed_prompt_outputs/qwen3-32b/scored \
  --output-dir src/assets/predictors/accuracy_predictor \
  --model-metadata experiments/bucketed_prompt_outputs/model_metadata.json \
  --test-fraction 0.1
```

### 6) Calibrate per-model serving priors/coefficients

#### 6a) Service metrics (used by wait estimators)

```bash
sbatch src/slurm/runs/service_metrics.sbatch
```

This produces `experiments/model_metrics*.json` with per-model:
- `service_rate_qps`
- `prefill_theta`

Use these values to update priors used by experiments:
- `prefill_tps_ttft` estimator/utility path: update `DEFAULT_PREFILL_TPS` or pass `--prefill-tps key=value`.
- `pk_mg1` estimator/utility path: update service-rate priors (currently resolved from defaults in `experiments.py`).

#### 6b) Batch-fit coefficient calibration per model

1. Ensure per-model batch stats CSVs exist under a directory like:
   `experiments/batch_stats/<batch_stats_dir>/`
2. Split train/test:

```bash
export BATCH_STATS_DIR=experiments/batch_stats/<batch_stats_dir>

python -m scripts.prep.split_csv \
  --root "$BATCH_STATS_DIR" \
  --seed 69
```

3. Fit held-out two-part batch models:

```bash
BASE=$PWD/$BATCH_STATS_DIR sbatch src/slurm/runs/batch_stats.sbatch
```

Use resulting coefficient artifacts to update vLLM simulation flags and/or instance TTFT parameter blocks used in runs.

### 7) Build holdout caches from the remainder (`start_index = X`)

You can prebuild caches explicitly (recommended for reproducibility) or let `experiments.py` build them on first run.

Example explicit build for two holdout sizes:

```bash
for N in 2000 4000; do
  python -m scripts.prep.holdout_prompts \
    --source-bucket-dir experiments/data/prompts/bucketed_prompts/qwen3-0.6b \
    --cache-dir experiments/data/prompts/holdout_cache_${N} \
    --tokenizer-id Qwen/Qwen3-8B \
    --holdout-start-index 2500 \
    --holdout-prompts-per-bucket ${N} \
    --holdout-context-length 65536 \
    --max-completion-tokens 8192 \
    --prompt-token-limit 32768
 done
```

### 8) Generate and score holdout outputs for accuracy augmentation

Router JSON augmentation needs per-model quality labels on the same holdout prompt set.

For each holdout size `N` used in sweeps:
1. Run model generation on `holdout_cache_${N}` (same models as routing candidates).
2. Normalize outputs into a dedicated root such as:
   `experiments/bucketed_prompt_outputs/holdout_${N}_scored/<model>/...`
3. Run `quality_metrics.py` on that root.

Example generation wrapper invocation for `N=4000`:

```bash
BUCKET_DIR=$PWD/experiments/data/prompts/holdout_cache_4000 \
MAX_PROMPTS_PER_BUCKET=4000 \
sbatch src/slurm/prep/run_qwen_bucketed_prompts_all_models.sbatch
```

Then score the normalized holdout outputs with `scripts.prep.quality_metrics`.

### 9) Run experiments on holdout caches only

All provided sweep scripts pass holdout settings (`--holdout-prompts-per-bucket`, `--holdout-start-index`, `--holdout-cache-dir`) and run on holdout caches, not on calibration prompts.

## Sweep Scripts

Snapshot-enabled sweeps:
- `src/slurm/runs/router/router_experiments_qps_sweep.sbatch`
- `src/slurm/runs/router/router_experiments_lambda_sweep.sbatch`
- `src/slurm/runs/router/router_experiments_qps_arrival_sweep.sbatch`
- `src/slurm/runs/router/router_experiments_delta_sweep.sbatch`

No-snapshot sweeps:
- `src/slurm/runs/router/router_experiments_qps_sweep_no_snapshot.sbatch`
- `src/slurm/runs/router/router_experiments_lambda_sweep_no_snapshot.sbatch`
- `src/slurm/runs/router/router_experiments_qps_arrival_sweep_no_snapshot.sbatch`

Current defaults:
- QPS/lambda/arrival sweeps: `N=4000`, `start_index=2500`.
- Delta sweep: `N=2000`, `start_index=2500`.

### 10) Build request maps, augment actual accuracy, summarize, plot

Build request maps for default holdout sizes:

```bash
python -m scripts.prep.map_holdout_request_ids \
  --preset all \
  --output-dir experiments
```

`--preset all` currently emits mappings for the provided defaults (`N=2000` and `N=4000`).  
If you run different holdout sizes, update the mapping job definitions in `scripts.prep.map_holdout_request_ids`.

Augment router JSONs with actual accuracy:

```bash
python -m scripts.eval.augment_router_actual_accuracy \
  <router_run_dir_1> <router_run_dir_2> \
  --delta-req-map experiments/req_map_delta_seed69_holdout2000_n8000.csv \
  --qps-req-map experiments/req_map_qps_seed69_holdout4000_n16000.csv \
  --scored-root experiments/bucketed_prompt_outputs/holdout_4000_scored
```

Then run summary/plot scripts in `src/scripts/reporting`.

## Adding a New Model

1. Add model serving stanza to run scripts or `--instances-config`.
2. Set `instance_id`, `model_id`, model path, ports, GPU mapping, snapshot SHM settings.
3. Include new model outputs in `experiments/bucketed_prompt_outputs/<new_model>/...`.
4. Update `experiments/bucketed_prompt_outputs/model_metadata.json` for accuracy model training.
5. Retrain output-length and accuracy predictors including the new model.
6. Calibrate service metrics for the new model (`service_metrics.sbatch`) and apply priors.
7. Calibrate batch-fit coefficients for the new model and update serve/config coefficients.
8. Update per-instance costs and any alias mappings used by utilities.
9. Rebuild holdout caches if needed and rerun sweeps + reporting.

## Updating vLLM Config

When changing scheduler/serve config (for example `max_num_batched_tokens`, `max_num_seqs`, chunked prefill, TP size, snapshot publish behavior, simulation coefficients):

1. Update `vllm serve` flags in relevant SLURM scripts.
2. Re-run calibration on the calibration pool:
   - service metrics
   - batch-fit coefficient calibration
   - predictor training if behavior/data changed materially
3. Rebuild holdout caches if `X`, context limits, or completion caps changed.
4. Re-run snapshot and no-snapshot sweeps.
5. Regenerate request maps/augmentation/summaries/plots.

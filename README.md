# SFS Experiment Workflow

This repository contains the end-to-end SFS pipeline for prompt bucketing, calibration, holdout construction, router sweeps, augmentation, and plotting.

## Requirements

### System and Toolchain

- Linux `x86_64`.
- SLURM cluster (all provided run wrappers are `sbatch` scripts).
- NVIDIA GPU nodes for generation/routing experiments.
  - Allocation shape: 4 GPUs from one H100 GPU node (a full h100-80 node contains 8 NVIDIA H100-80GB SXM5 GPUs).
  - GPU memory: 80 GB per GPU, 320 GB total across the 4 GPUs reserved per experiment.
  - CPU: 2 Intel Xeon "Sapphire Rapids" 8470 CPUs, 52 cores per CPU, 2.0-3.8 GHz.
  - System RAM: 2,048 GB per node using 128 GB DDR5-4800 DIMMs.
  - Intra-node GPU interconnect: NVLink.
  - Node-local storage: 4 NVMe SSDs, 7.68 TB total.
  - Network: InfiniBand 900GB.
- Environment modules:
  - `cuda/12.6.1`
  - `gcc/13.3.1-p20240614`
- NVIDIA driver compatible with CUDA `12.6.1` (`R560+` recommended; the build script prints the detected version via `nvidia-smi` when available).
- Conda or Mamba.

Record the runtime driver in experiment logs with:
`nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1`

### Compute Estimate

Estimated H100 GPU-hours are computed as wall-clock hours multiplied by the number of H100 GPUs allocated. CPU-only post-processing jobs are negligible in wall-clock time.

| Figure(s) | Experiment | Worker allocation | Estimated wall-clock time | Estimated H100 GPU-hours |
|---|---|---|---:|---:|
| Figure 2 | Waiting-time fit for prefill TPS and SFS estimators | 1 H100 | ~1 h | ~1 |
| Figure 3 | Batch time plot | CPU only | <1 min | 0 |
| Figure 4 | Observed TTFT for sequence generation | 1 H100 | ~2 h | ~2 |
| Figures 5 and 13 | QPS sweep over 12 QPS values for SFS, shortest queue, latency agnostic, and round robin | 4 H100s | ~64 h | ~256 |
| Figure 6 | Delta sweep over 15 SFS delta values plus shortest queue, latency agnostic, and round robin | 4 H100s | ~16 h | ~64 |
| Figure 7 | Arrival sweep over 3 QPS values and 3 arrival processes for SFS, shortest queue, latency agnostic, and round robin | 4 H100s | ~36 h | ~144 |
| Figure 8 | Simulation-latency distribution | CPU only | <1 min | 0 |
| Figure 12 | Lambda sweep over 8 values for hard and latency agnostic policies | 4 H100s | ~16 h | ~64 |
| Figure 14 | Routing composition across model instances | CPU only | <1 min | 0 |

The experiments presented require ~135 wall-clock hours and ~531 H100 GPU-hours excluding preparation/scoring and preliminary experiments. Figure 13 reuses the same QPS sweep outputs as Figure 5 and does not require an additional run.

Preparation and scoring adds ~6 hours on 4 H100s for generating outputs for calibration and holdout prompts (~24 H100 GPU-hours), ~16 hours for Gemini-based judging of calibration and holdout prompts, ~30 minutes on a single H100 for training the accuracy and output predictors (~0.5 H100 GPU-hours), and ~2 hours on 4 H100s for calibration of batch fit and service rates (~8 H100 GPU-hours). The total time estimator for the experiments along with preparation and scoring is ~159.5 wall-clock hours and ~563.5 H100 GPU-hours. Including preliminary experiments, the total time estimate is ~220 wall-clock hours and ~770 H100 GPU-hours.

### Python Environment (Conda/Mamba)

```bash
cd /path/to/sfs
mamba env create -f env.yml
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

For judge-based quality scoring, set `GOOGLE_API_KEY` or `GEMINI_API_KEY`.

### Compile vLLM Scheduler Simulation Extension

The router's snapshot-SHM wait-time simulation depends on the native `_scheduler_sim` extension in `vllm`.

Run this on a host where CUDA modules are available:

```bash
cd /path/to/sfs
bash scripts/setup/compile_vllm_scheduler_sim.sh
```

This script:
- Resolves `SFS_ROOT` and `PROJECT_ROOT` from script location.
- Loads `cuda/12.6.1` and `gcc/13.3.1-p20240614`.
- Activates conda env `vllm`.
- Runs editable install in `sfs/vllm` with the same flags used in our setup.
- Executes `tests/v1/engine/test_scheduler_simulator_native.py -k timing`.

## Third-Party Assets (Version, URL, License)

| Asset | Version Used | URL | License |
|---|---|---|---|
| vLLM (vendored as submodule) | commit `4dbdf4a2944849a540c97542c4839b87fcd2988c` | https://github.com/vllm-project/vllm | Apache License 2.0 |
| Qwen3 model family (`Qwen/Qwen3-0.6B`, `Qwen/Qwen3-8B`, `Qwen/Qwen3-32B`) | HF snapshots `c1899de289a04d12100db370d81485cdf75e47ca`, `b968826d9c46dd6066d109eabc6255188de91218`, `9216db5781bf21249d130ec9da846c4624c16137` | https://github.com/QwenLM/Qwen3 | Apache License 2.0 |
| Alpaca dataset | HF revision `dce01c9b08f87459cf36a430d809084718273017` | https://huggingface.co/datasets/tatsu-lab/alpaca/tree/dce01c9b08f87459cf36a430d809084718273017 | CC BY-NC 4.0 |
| WritingPrompts dataset | HF revision `35f0aa359452ba8147b34d925684fccee26679cc` | https://huggingface.co/datasets/euclaise/writingprompts/tree/35f0aa359452ba8147b34d925684fccee26679cc | MIT |
| HotpotQA dataset (`distractor` config) | HF revision `1908d6afbbead072334abe2965f91bd2709910ab` | https://huggingface.co/datasets/hotpotqa/hotpot_qa/tree/1908d6afbbead072334abe2965f91bd2709910ab | CC BY-SA 4.0 |
| GovReport summarization dataset (`ccdv/govreport-summarization`) | HF revision `4e21184e01ae8017e2c036e180fe5e541fef60a0` | https://huggingface.co/datasets/ccdv/govreport-summarization/tree/4e21184e01ae8017e2c036e180fe5e541fef60a0 | CC BY 4.0 |

- Apache License 2.0: permits use, modification, distribution, and commercial use, with license/notice preservation, change notices, patent terms, and no warranty.
- CC BY-NC 4.0: permits sharing and adaptation with attribution, but only for noncommercial purposes and without additional legal or technical restrictions.
- MIT: permits use, modification, distribution, sublicensing, and sale, with copyright/license notice preservation and no warranty.
- CC BY-SA 4.0: permits sharing and adaptation, including commercial use, with attribution; adaptations must be distributed under the same license and without additional restrictions.
- CC BY 4.0: permits sharing and adaptation, including commercial use, with attribution and without additional legal or technical restrictions.

## Repository Layout

| Path | Purpose |
|---|---|
| `src/scripts/prep` | Data prep, scoring, holdout cache, mapping scripts. |
| `src/scripts/runs` | Main experiment drivers (`experiments.py`, `experiments_sweep.py`). |
| `src/scripts/eval` | Post-run augmentation/diagnostics. |
| `src/scripts/reporting` | Sweep summary + plotting scripts. |
| `src/sfs_core` | Shared library code (routing, predictors, regression, helpers). |
| `scripts/setup` | Reproducible environment/build wrappers (including vLLM scheduler simulation build). |
| `src/slurm/prep` | SLURM wrapper for bucketed prompt generation. |
| `src/slurm/runs` | SLURM wrappers for calibration + sweeps. |
| `src/assets` | Static assets (template + predictor artifacts). |
| `experiments/data/prompts` | Generated prompt artifacts (bucket pools + holdout caches). |
| `experiments/` | Runtime outputs. |

## Data-Split Policy

Experiments are **not** run on the initial calibration prompts.

Define:
- `X`: calibration prefix size per bucket (default in current scripts: `X=2500`)
- `N`: holdout size per bucket (examples used here: `N=2000` or `N=4000`)

For each bucket:
1. Calibration/training pool uses indices `[0, X-1]`.
2. Holdout pool starts at index `X` and uses the next `N` prompts.
3. Router experiments run on this holdout pool only.

`holdout_prompts.py` preserves original `prompt_index` and builds request IDs with that index, so holdout requests remain disjoint from calibration/training data by construction.

## Reproducing Results

All experimental results reported are generated via scripts and commands in this repository.

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

Note: GEMINI-based judging is not fully deterministic. Google recommends default inference settings, and the effective sampling configuration may introduce small score variance across reruns.

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

The output-length predictor used in the experiments (along with the specific hyperparameters used for training) is provided in `src/assets/predictors/output_length_predictor`.

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

The accuracy predictor used in the experiments (along with the specific hyperparameters used for training) is provided in `sfs/src/assets/predictors/accuracy_predictor`.

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

### 9) Run experiments

All provided sweep scripts pass holdout settings (`--holdout-prompts-per-bucket`, `--holdout-start-index`, `--holdout-cache-dir`) and run on holdout caches, not on calibration prompts.

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

## Adapting to New Model or vLLM Config

For a new model:
1. Add model serving stanza to run scripts or `--instances-config`.
2. Set `instance_id`, `model_id`, model path, ports, GPU mapping, snapshot SHM settings.
3. Include new model outputs in `experiments/bucketed_prompt_outputs/<new_model>/...`.
4. Update `experiments/bucketed_prompt_outputs/model_metadata.json` for accuracy model training.
5. Retrain output-length and accuracy predictors including the new model.
6. Calibrate service metrics for the new model (`service_metrics.sbatch`) and apply priors.
7. Calibrate batch-fit coefficients for the new model and update serve/config coefficients.

For a new vLLM runtime config:
1. Update vLLM serve flags in the relevant `src/slurm/prep` and `src/slurm/runs` scripts.
2. Re-run step 6 (service metrics + batch-fit coefficient calibration) for that config.
3. Regenerate outputs and re-run scoring if generation behavior changes.
4. Re-run sweeps and post-processing (steps 9-10) to keep comparisons consistent.

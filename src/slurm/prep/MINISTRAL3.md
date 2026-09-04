# Ministral 3 calibration generations

The family-specific preparation launchers generate aligned calibration outputs
for the official BF16 Ministral 3 Instruct variants:

| Output label | Hugging Face repository | Pinned revision |
|---|---|---|
| `ministral3-3b` | `mistralai/Ministral-3-3B-Instruct-2512-BF16` | `b6d637bef2393152b3da2b2fde72eecdee30557e` |
| `ministral3-8b` | `mistralai/Ministral-3-8B-Instruct-2512-BF16` | `f6fae9795746f63c9be8344932f01275f3c63734` |
| `ministral3-14b` | `mistralai/Ministral-3-14B-Instruct-2512-BF16` | `3cea74c1ebaf5ce5f5a2553de470e2ceab825142` |

The revisions are pinned in the shared driver. It refuses to silently select a
different cached revision.

## Cache the checkpoints

Run these commands from the SFS worktree, not a compute job. This custom vLLM
fork serves Ministral 3 through the official Mistral-format configuration and
consolidated weights. The equivalent indexed Hugging Face shards are excluded
to avoid caching both representations.

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm

SFS_ROOT="$(git rev-parse --show-toplevel)"
MODEL_CACHE_ROOT="$(cd "$SFS_ROOT/.." && pwd)/.cache/huggingface/hub"
hf download mistralai/Ministral-3-3B-Instruct-2512-BF16 \
  --revision b6d637bef2393152b3da2b2fde72eecdee30557e \
  --cache-dir "$MODEL_CACHE_ROOT" --exclude 'model-*.safetensors'
hf download mistralai/Ministral-3-8B-Instruct-2512-BF16 \
  --revision f6fae9795746f63c9be8344932f01275f3c63734 \
  --cache-dir "$MODEL_CACHE_ROOT" --exclude 'model-*.safetensors'
hf download mistralai/Ministral-3-14B-Instruct-2512-BF16 \
  --revision 3cea74c1ebaf5ce5f5a2553de470e2ceab825142 \
  --cache-dir "$MODEL_CACHE_ROOT" --exclude 'model-*.safetensors'
```

## Validate and submit

The validation mode checks the profile and model mapping without inspecting
the cache, activating Python, or touching a GPU:

```bash
VALIDATE_ONLY=1 bash src/slurm/prep/run_ministral3_bucketed_prompts_h100.sbatch
VALIDATE_ONLY=1 bash src/slurm/prep/run_ministral3_bucketed_prompts_v100.sbatch
```

Submit exactly one hardware wrapper:

```bash
MAX_PROMPTS_PER_BUCKET=2500 \
  sbatch src/slurm/prep/run_ministral3_bucketed_prompts_h100.sbatch

# Or, for the slower four-GPU V100 profile:
MAX_PROMPTS_PER_BUCKET=2500 \
  sbatch src/slurm/prep/run_ministral3_bucketed_prompts_v100.sbatch
```

The H100 profile uses one GPU per model and mirrors the main Qwen H100 serving
envelope: automatic dtype selection (BF16 for these pinned checkpoints), a
131,072-token vLLM model limit, a 65,536-token client context budget, chunked
prefill, 32,768 batched tokens, 512 sequences, prefix caching disabled, and
CUDA graphs enabled. The client independently caps prompt text at 32,768 tokens
and completion length at 8,192 tokens. Keeping `MAX_MODEL_LEN` distinct from
`CONTEXT_LENGTH` is intentional: the former sizes the server while the latter
controls request construction.

The V100 profile uses one GPU for 3B, one for 8B, and tensor parallelism across
two GPUs for 14B. It explicitly preserves the previously validated compatibility
profile: FP16, a 40,960-token model and client limit, unchunked prefill, 49,152
batched tokens, and 64 sequences. The text-only experiment rejects image inputs
and disables the multimodal processor cache. The fork's Mistral-format Pixtral
wrapper still loads each checkpoint's 0.4B vision component; the hardware
profiles budget for that overhead.

The launchers use vLLM's Mistral tokenizer backend for both serving and
client-side token accounting, the neutral system prompt `You are a helpful
assistant.`, and no Qwen-specific chat-template arguments. Each job writes
three aligned model directories under its `completions` directory and prints
the exact grouped judge-scoring command when generation finishes.

For dependency chains, set `RUN_DIR_OVERRIDE` to a new, deterministic path.
The driver refuses to overwrite an existing override directory. It exits zero
only after a strict audit confirms the expected record count, no per-request
errors or empty completions, and identical example IDs across all three models;
this makes an `afterok` judge dependency safe.

The accuracy-augmentation reader accepts the judge launcher's direct layout
(`<root>/<model>/*_scored.jsonl`) as well as the legacy Qwen
`<root>/<model>/scored/` layout.

These launchers prepare calibration generations only. A complete Ministral
family experiment still needs fresh judge scores, accuracy/output-length
predictors, serving calibration, family-specific costs, and routing artifacts.

## Train the family predictors

The audited calibration archive can be used directly; the training launcher
passes the twelve scored files explicitly so raw and scored JSONLs cannot be
mixed accidentally:

```bash
VALIDATE_ONLY=1 bash src/slurm/prep/train_ministral3_predictors.sbatch
sbatch src/slurm/prep/train_ministral3_predictors.sbatch
```

Each job writes a distinct run under
`experiments/ministral3_paper/predictors/`. It uses the same 90/10 grouped
split, seed 69, and unbalanced 30,000-example training population as the saved
Qwen predictors. Model descriptors live in
`src/assets/model_metadata/ministral3.json` and use the checkpoints' native
262,144-token context capability. Descriptor keys intentionally match the
`model_label` values consumed by predictor feature extraction; served IDs are
audited separately.

For cost-aware routing, the reference cost rates are the standard global
Mistral API prices accessed on 2026-09-02: 3B is $0.10/$0.10, 8B is
$0.15/$0.15, and 14B is $0.20/$0.20 per million input/output tokens. Runtime
instance JSON records these rates so they are not confused with Qwen defaults.
Sources: <https://docs.mistral.ai/models/ministral-3-3b-25-12> and
<https://docs.mistral.ai/inference/pricing>.

## Calibrate the corrected H100 service profile

```bash
VALIDATE_ONLY=1 bash src/slurm/runs/ministral3_service_metrics.sbatch
sbatch src/slurm/runs/ministral3_service_metrics.sbatch
```

The service job starts all three models concurrently under exactly the H100
profile documented above. It replays the 10,000 audited calibration prompts per
model at saturation, fits the paper's `cross_term` batch-time equation with
nonnegative physical coefficients, validates complete trace coverage, and
writes service rates, prefill/decode measurements, coefficients, batch traces,
and a capacity summary to a job-specific directory. It uses the scored 3B
records only as an aligned prompt/token-count view; quality labels do not enter
serving calibration. Old record-level completion caps are recomputed under the
65,536-token client context budget.

## Prepare the disjoint sweep prompts

This CPU-only job builds both paper-sized holdout caches from source rows after
the 2,500 calibration examples in every bucket:

```bash
VALIDATE_ONLY=1 bash src/slurm/prep/prepare_ministral3_holdouts.sbatch
sbatch src/slurm/prep/prepare_ministral3_holdouts.sbatch
```

It writes `holdout_cache_2000` for the delta sweep and
`holdout_cache_4000` for the QPS and arrival-process sweeps under
`experiments/data/prompts/ministral3/`, using the pinned 8B Mistral tokenizer,
the neutral system prompt, an empty chat-template policy, a 65,536-token client
context, a 32,768-token prompt cap, and an 8,192-token completion cap.

Build the corresponding deterministic router request maps with:

```bash
python -m scripts.prep.map_holdout_request_ids \
  --preset all \
  --qps-cache-dir experiments/data/prompts/ministral3/holdout_cache_4000 \
  --delta-cache-dir experiments/data/prompts/ministral3/holdout_cache_2000 \
  --output-dir experiments/ministral3_paper/request_maps
```

## Judge the generated outputs

Keep the Gemini key outside the repository in a user-only environment file.
Enter the key at the hidden prompt; do not paste it into a command or chat:

```bash
install -d -m 700 /jet/home/aparthas/.config/sfs
(
  umask 077
  IFS= read -rsp "Gemini API key: " SFS_GEMINI_KEY
  printf '\n'
  printf 'export GEMINI_API_KEY=%q\n' "$SFS_GEMINI_KEY" \
    > /jet/home/aparthas/.config/sfs/gemini.env
)
chmod 600 /jet/home/aparthas/.config/sfs/gemini.env
```

Validate the CPU-only judge launcher without reading the key or calling Gemini:

```bash
COMPLETIONS_ROOT="$PWD/experiments/<generation-run>/completions"
VALIDATE_ONLY=1 bash src/slurm/prep/run_ministral3_judge.sbatch \
  "$COMPLETIONS_ROOT"
```

Then submit the resumable CPU job:

```bash
sbatch src/slurm/prep/run_ministral3_judge.sbatch "$COMPLETIONS_ROOT"
```

The judge matches the Qwen grouped-scoring protocol: one Gemini request scores
the three shuffled model responses for each aligned prompt, using
`gemini-3.1-pro-preview`. It processes 20 prompt groups concurrently by
default. Only failed requests are retried, up to three concurrent attempts with
exponential backoff; retry-wave concurrency steps down from 20 to 10 to 5.
Requests still failing are retried individually after all concurrency batches,
up to three more attempts. Persistent failures receive the mean of the
successful scores for the same model and bucket, with the same
`judge_default_bucket_mean`, `quality_imputed`, and
`quality_imputed_reason` fields used by the saved Qwen results. Existing scored
records are reused, so resubmitting the job resumes rather than starting over.

Override `JUDGE_BATCH_SIZE`, `JUDGE_RETRIES`, `INDIVIDUAL_RETRIES`,
`RETRY_SLEEP_SECONDS`, `JUDGE_MODEL`, or `GEMINI_ENV_FILE` only when needed.
The launcher writes a job-specific `judge_run_summary_*.json` under the
completions root and exits successfully only after every common prompt has a
score for every model.

Before the first GPU run, build/install the nested vLLM at this worktree's
recorded gitlink so that the Python-side Ministral scaling backport and the
existing native scheduler-simulation extension come from one checkout:

```bash
bash scripts/setup/compile_vllm_scheduler_sim.sh
```

The launcher verifies the active editable vLLM commit and fails before starting
servers if a different SCORE worktree is still installed.

## Reproduce the paper-style figures

The Ministral workflow has separate, dependency-safe jobs for the principal
paper comparisons:

- `ministral3_wait_gof.sbatch` reproduces the Figure 2 wait-estimator
  comparison on the 3B model at 90% of its calibrated standalone capacity.
- `ministral3_batch_fit.sbatch` performs an 80/20 held-out batch-time fit for
  all three models and emits the Figure 3 parity plots.
- `ministral3_router_sweep.sbatch` runs the Figure 5 offered-load sweep, the
  Figure 6 delta sweep, and the Figure 7 Poisson/MMPP-2 comparison. QPS values
  are expressed as fractions of the measured three-server capacity so the load
  regimes remain comparable after changing model family.

Run the router wrapper twice for Figure 5 (`SWEEP_KIND=qps`, snapshot and
baseline utility groups), once for Figure 6 (`SWEEP_KIND=delta`, combined), and
twice for Figure 7 (`SWEEP_KIND=qps_arrival`, snapshot and baseline). Figure 7
uses capacity fractions 0.82, 0.95, and 1.10, which are also present in the
Poisson Figure 5 sweep.

For each ablation, submit the default Figure-5-only collation after judging and
the two QPS sweeps succeed:

```bash
sbatch --dependency=afterok:<judge>:<qps-sfs>:<qps-base> \
  --export=ALL,COLLATION_SCOPE=figure5,QPS_SNAPSHOT_JOB_ID=<qps-sfs>,QPS_BASELINE_JOB_ID=<qps-base>,JUDGE_JOB_ID=<judge> \
  src/slurm/runs/ministral3_paper_postprocess.sbatch
```

Only for a canonical full-suite run, submit the collation job after all five
sweeps and select the explicit `full` scope:

```bash
sbatch --dependency=afterok:<judge>:<qps-sfs>:<qps-base>:<delta>:<arrival-sfs>:<arrival-base> \
  --export=ALL,COLLATION_SCOPE=full,QPS_SNAPSHOT_JOB_ID=<qps-sfs>,QPS_BASELINE_JOB_ID=<qps-base>,DELTA_JOB_ID=<delta>,ARRIVAL_SNAPSHOT_JOB_ID=<arrival-sfs>,ARRIVAL_BASELINE_JOB_ID=<arrival-base>,JUDGE_JOB_ID=<judge> \
  src/slurm/runs/ministral3_paper_postprocess.sbatch
```

The collation job verifies every raw checksum and sweep audit, verifies the
48,000 held-out judge scores, hard-links the router JSONs into a derived tree,
and augments only those copies with realized quality. The default scope creates
and audits Figure 5 without requiring Figure 6 or 7 jobs; `full` creates all
three figures. A second raw checksum pass proves that postprocessing did not
mutate the original sweep outputs.

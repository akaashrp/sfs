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

The H100 profile uses one GPU per model. The V100 profile uses one GPU for 3B,
one for 8B, and tensor parallelism across two GPUs for 14B. Both default to
FP16 and unchunked prefill, disable prefix caching, and set the batched-token
limit above the 40,960-token model limit. The V100 wrapper enforces FP16 and
unchunked prefill; the H100 wrapper leaves those two settings overridable. The
text-only experiment rejects image inputs and disables the multimodal processor
cache. The fork's Mistral-format Pixtral wrapper still loads each checkpoint's
0.4B vision component; the hardware profiles budget for that overhead.

The launchers use vLLM's Mistral tokenizer backend for both serving and
client-side token accounting, the neutral system prompt `You are a helpful
assistant.`, and no Qwen-specific chat-template arguments. Each job writes
three aligned model directories under its `completions` directory and prints
the exact grouped judge-scoring command when generation finishes.

These launchers prepare calibration generations only. A complete Ministral
family experiment still needs fresh judge scores, accuracy/output-length
predictors, serving calibration, family-specific costs, and routing artifacts.

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

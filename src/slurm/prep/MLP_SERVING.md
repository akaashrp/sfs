# MLP predictor serving integration

Status on September 11, 2026: 16 CPU tests passed across the MLP and existing
serving-prediction contract suites (`experiments/mlp_serving_20260911/cpu_tests.xml`).
Full calibration training/export completed as CPU job **45750229** (exit 0,
11m06s), with output root `experiments/mlp_serving_20260911/trained`.
Both models converged, with exact export/reload prediction parity. Quality
calibration-validation MAE/RMSE: 0.173419/0.248056; output-length MAE/RMSE:
234.829/596.713 tokens. Batch-of-three CPU feature/inference time was
0.656 ms for quality and 0.690 ms for length. These are calibration-validation
results, not final holdout or GPU serving results. Source and exported-weight
hashes were reverified in `experiments/mlp_serving_20260911/completion_review.json`.
Its submission and source hashes are recorded in
`experiments/mlp_serving_20260911/submission.json` and the campaign registry.
The trained-artifact CPU audit passed. GPU smoke and its launcher rehearsal
remain pending; no GPU job was submitted for this integration.

The existing `--accuracy-model-path` and `--output-length-model-path` options
accept exported MLP artifact directories. No new routing policy or vLLM
scheduler is needed. Metadata selects `predictor_backend: numpy_mlp`; artifacts
without this field retain their existing LightGBM behavior.

Training uses the fixed offline MLP recipe (StandardScaler, ReLU 64/32 hidden
units, seed 69, max_iter 500, batch_size 256, no early stopping). It reads only
canonical calibration responses, reserves the original predictor validation
IDs, and excludes matching validation prompt texts from training. It never
reads final serving holdouts. Exported metadata retains the existing feature
preprocessing, model labels and ordering. Numeric NPZ weights carry a SHA-256
checksum, dimension/target checks and finite-value checks; serving requires
NumPy, not scikit-learn or pickle deserialization.

## CPU training and export

Activate `conda activate vllm`, use the active nested checkout on `PYTHONPATH`,
and set `TMPDIR` to workspace scratch. Run the following on a CPU allocation
(the wrapper is `src/slurm/prep/train_serving_mlp.sbatch`):

```bash
python -m scripts.prep.train_serving_mlp \
  --prepared-dir experiments/paper_ablation_20260907/data_16000_20260908 \
  --predictor-root src/assets/predictors \
  --output-root experiments/mlp_serving_20260911/trained
```

The output root must be new. The audit records source hashes, convergence,
calibration validation MAE/RMSE, export/reload parity, and batch-of-three CPU
feature/inference time. Review any convergence warning before freezing models.
`serving_variants.json` records the exact artifact paths for the two arms.

## Serving wiring

| Arm | Router accuracy path | Router length path | Length path on every vLLM server |
|---|---|---|---|
| MLP quality | exported `accuracy_predictor` | canonical `output_length_predictor` | canonical `output_length_predictor` |
| MLP output length | canonical `accuracy_predictor` | exported `output_length_predictor` | the same exported `output_length_predictor` |

Pass the first two paths to the router's existing predictor options. Pass the
third to every vLLM server's `--output-length-model-path` option. Restart the
server pool when switching length artifacts: existing resident requests must
not carry predictions from a different artifact. Record the selected paths and
hashes with each run. Quality predictions clip to [0,1]; length trains in
log1p(tokens), inverts and clips to [1,8192], with the router also respecting
the request completion cap. The length arm changes both routing cost/work
estimates and resident-request simulation inputs.

`src/scripts/runs/tests/test_mlp_serving.py` checks numerical parity with
scikit-learn, shared prompt contexts, both production consumer APIs, invalid
artifacts, clipping and existing LightGBM behavior. CPU API checks do not run a
GPU scheduler. Before an evaluation sweep, require a GPU smoke with matching
router/server artifacts and the existing measured-serving gates; check finite
predictions, snapshot attachment, request completion and timing overhead.
The eight evaluation cells and their budget remain in
`CURRENT_EXPERIMENT_PLAN.md`. Existing launch/source-hash audits predate this
integration and must be refreshed before releasing predictor sweeps.

September 12 update: both full CPU serving/ingestion checks and launcher rehearsal
passed. A shared four-H100 allocation **45843212** is submitted for separate
192-request smokes with a pool restart between quality and length arms. Evidence
and the two future 16000-request/four-QPS sweep launch environments are in
`experiments/predictor_prereqs_20260912/`. Evaluation still requires GPU smoke audit.

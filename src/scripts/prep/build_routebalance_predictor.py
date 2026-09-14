#!/usr/bin/env python3
"""Build RouteBalance's native MiniLM/FAISS estimator from scored calibration.

Example (activate conda vllm; put optional faiss-cpu target on PYTHONPATH):
  python -m scripts.prep.build_routebalance_predictor \
    --completions-root /path/to/archived/run/completions \
    --output-dir experiments/ministral3_paper/routebalance_predictor/run_DATE \
    --validation-id-files /path/to/accuracy/test_example_ids.json \
                          /path/to/length/test_example_ids.json \
    --cache-dir /ocean/projects/cis250162p/aparthas/.cache/huggingface/hub

The first 2500 prompt indices in each bucket are mandatory and aligned across
all models; canonical IDs remain strings and are never confused with offsets.
Validation manifests select their intersection; otherwise a deterministic 10%
prompt-hash split is used. Exact duplicate text is always kept on one side.
No generations, judges, GPU operations, or SFS predictor calls occur here.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

from sfs_core.routing.routebalance_predictor import (
    DEFAULT_MAX_COMPLETION_TOKENS,
    MiniLMCPUEmbedder,
    RouteBalancePredictor,
    encoder_metadata,
    file_sha256,
    prompt_sha256,
    save_predictor_artifact,
)


DEFAULT_MODELS = ("ministral3-3b", "ministral3-8b", "ministral3-14b")
DEFAULT_BUCKETS = ("alpaca", "govreport-summarization", "hotpot_qa", "writingprompts")


@dataclass
class CalibrationPrompt:
    bucket: str
    example_id: str
    prompt_index: int
    prompt: str
    qualities: list[float]
    output_lengths: list[int]
    completion_caps: list[int]

    def manifest(self) -> dict[str, Any]:
        return {
            "bucket": self.bucket, "example_id": self.example_id,
            "prompt_index": self.prompt_index, "prompt_sha256": prompt_sha256(self.prompt),
        }


def _finite_number(value: Any, name: str, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not lower <= result <= upper:
        raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")
    return result


def load_calibration(
    root: Path,
    *,
    model_labels: Sequence[str] = DEFAULT_MODELS,
    buckets: Sequence[str] = DEFAULT_BUCKETS,
    expected_per_bucket: int = 2500,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
) -> tuple[list[CalibrationPrompt], dict[str, Any]]:
    """Strictly audit complete canonical calibration groups before embedding."""
    if not 0 < expected_per_bucket <= 2500 or max_completion_tokens <= 0:
        raise ValueError("Calibration count must be in [1,2500] and completion cap positive")
    if not model_labels or len(set(model_labels)) != len(model_labels):
        raise ValueError("Model labels must be nonempty and unique")
    if not buckets or len(set(buckets)) != len(buckets):
        raise ValueError("Buckets must be nonempty and unique")
    groups: dict[tuple[str, str], CalibrationPrompt] = {}
    reference_keys: set[tuple[str, str]] | None = None
    sources = []
    quality_metrics: Counter[str] = Counter()
    imputation_count = 0
    for model_number, model in enumerate(model_labels):
        model_keys: set[tuple[str, str]] = set()
        for bucket in buckets:
            path = root / model / f"{bucket}_scored.jsonl"
            digest = hashlib.sha256()
            seen_indices: set[int] = set()
            count = 0
            with path.open("rb") as stream:
                for line_number, line in enumerate(stream, 1):
                    digest.update(line)
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    context = f"{path}:{line_number}"
                    if record.get("model_label") != model or record.get("bucket") != bucket:
                        raise ValueError(f"{context}: wrong model or bucket")
                    if record.get("error"):
                        raise ValueError(f"{context}: failed generation cannot train a predictor")
                    metadata = record.get("prompt_metadata") or {}
                    example_id = metadata.get("example_id")
                    if not isinstance(example_id, str) or not example_id:
                        raise ValueError(f"{context}: missing canonical prompt_metadata.example_id")
                    index = record.get("prompt_index")
                    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < expected_per_bucket:
                        raise ValueError(f"{context}: prompt_index outside calibration [0,{expected_per_bucket})")
                    # Preserve canonical string IDs. For the dataset:split:index
                    # form, also check its independent index to catch a holdout
                    # file whose local row numbering accidentally starts at 0.
                    suffix = example_id.rsplit(":", 1)[-1]
                    if ":" in example_id and suffix.isdecimal() and int(suffix) != index:
                        raise ValueError(f"{context}: canonical ID index differs from prompt_index")
                    if index in seen_indices:
                        raise ValueError(f"{context}: duplicate prompt_index")
                    seen_indices.add(index)
                    key = (bucket, example_id)
                    if key in model_keys:
                        raise ValueError(f"{context}: duplicate canonical prompt ID")
                    model_keys.add(key)
                    prompt = record.get("prompt")
                    if not isinstance(prompt, str) or not prompt.strip():
                        raise ValueError(f"{context}: missing prompt text")
                    quality = _finite_number(record.get("quality"), f"{context} quality", 0, 1)
                    cap = record.get("max_completion_tokens")
                    if isinstance(cap, bool) or not isinstance(cap, int) or not 1 <= cap <= max_completion_tokens:
                        raise ValueError(f"{context}: invalid generation completion cap")
                    length = (record.get("response") or {}).get("completion_tokens")
                    if isinstance(length, bool) or not isinstance(length, int) or not 0 <= length <= cap:
                        raise ValueError(f"{context}: invalid response completion_tokens")
                    quality_metrics[str(record.get("quality_metric", "unknown"))] += 1
                    imputation_count += int(bool(record.get("quality_imputed") or record.get("judge_imputed")))
                    if model_number == 0:
                        groups[key] = CalibrationPrompt(bucket, example_id, index, prompt, [], [], [])
                    elif key not in groups:
                        raise ValueError(f"{context}: models have misaligned canonical IDs")
                    group = groups[key]
                    if group.prompt != prompt or group.prompt_index != index:
                        raise ValueError(f"{context}: model prompt text/index mismatch")
                    group.qualities.append(quality)
                    group.output_lengths.append(length)
                    group.completion_caps.append(cap)
                    count += 1
            if count != expected_per_bucket or seen_indices != set(range(expected_per_bucket)):
                raise ValueError(f"{path}: expected every calibration index exactly once; found {count}")
            sources.append({"path": str(path.resolve()), "sha256": digest.hexdigest(), "records": count})
        if reference_keys is not None and model_keys != reference_keys:
            raise ValueError(f"{model}: incomplete or misaligned prompt groups")
        reference_keys = model_keys
    ordered = sorted(groups.values(), key=lambda row: (row.bucket, row.prompt_index, row.example_id))
    return ordered, {
        "source_files": sources,
        "calibration_boundary": {"prompt_index_min": 0, "prompt_index_max_exclusive": expected_per_bucket},
        "quality_signal": "existing_grouped_judge_scores",
        "quality_metric_counts": dict(quality_metrics),
        "explicit_imputation_flags": imputation_count,
        "calibration_serving_profile_note": "quality and length labels reused; not capacity evidence",
    }


def split_calibration(
    rows: Sequence[CalibrationPrompt],
    *,
    validation_id_files: Sequence[Path] = (),
    validation_fraction: float = 0.1,
    split_seed: int = 69,
) -> tuple[list[CalibrationPrompt], list[CalibrationPrompt], dict[str, Any]]:
    if not rows:
        raise ValueError("No calibration prompts")
    sources = []
    if validation_id_files:
        id_sets = []
        for path in validation_id_files:
            data = json.loads(path.read_text(encoding="utf-8"))
            values = data.get("example_ids") if isinstance(data, dict) else data
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                raise ValueError(f"{path}: expected a list of canonical string example_ids")
            if len(set(values)) != len(values):
                raise ValueError(f"{path}: duplicate validation IDs")
            id_sets.append(set(values))
            sources.append({"path": str(path.resolve()), "sha256": file_sha256(path), "num_ids": len(values)})
        validation_ids = set.intersection(*id_sets)
        if not validation_ids:
            raise ValueError("Supplied validation manifests have an empty intersection")
        unknown = validation_ids - {row.example_id for row in rows}
        if unknown:
            raise ValueError(f"Validation IDs are absent from calibration: {sorted(unknown)[:3]}")
        selected = [row for row in rows if row.example_id in validation_ids]
        split_method = "intersection_of_existing_validation_id_manifests"
    else:
        if not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0,1)")
        # Split by raw prompt hash so repeated prompts never straddle partitions.
        selected = [row for row in rows if int(hashlib.sha256(
            f"{split_seed}:{prompt_sha256(row.prompt)}".encode()
        ).hexdigest()[:16], 16) / 2**64 < validation_fraction]
        split_method = "seeded_prompt_hash_threshold"
    validation_hashes = {prompt_sha256(row.prompt) for row in selected}
    validation = [row for row in rows if prompt_sha256(row.prompt) in validation_hashes]
    training = [row for row in rows if prompt_sha256(row.prompt) not in validation_hashes]
    if not training or not validation:
        raise ValueError("Both training and validation must contain prompts")
    return training, validation, {
        "split_method": split_method,
        "split_seed": split_seed,
        "validation_fraction": validation_fraction,
        "validation_manifest_sources": sources,
        "validation_duplicate_text_expansion": len(validation) - len(selected),
        "validation_selection_ids": [row.example_id for row in selected],
        "split_counts_by_bucket": {
            "train": dict(Counter(row.bucket for row in training)),
            "validation": dict(Counter(row.bucket for row in validation)),
        },
        "evaluation_holdouts_included": False,
    }


def validation_metrics(rows: Sequence[CalibrationPrompt], predictions: list[dict], models: Sequence[str]) -> dict:
    """Per-bucket/model errors on excluded calibration prompts only."""
    import numpy as np

    if len(rows) != len(predictions):
        raise ValueError("Validation predictions are not aligned")
    result = {}
    for bucket in ["all", *sorted({row.bucket for row in rows})]:
        selected = [i for i, row in enumerate(rows) if bucket == "all" or row.bucket == bucket]
        result[bucket] = {}
        for model_number, model in enumerate(models):
            q_error = np.asarray([predictions[i][model]["quality"] - rows[i].qualities[model_number] for i in selected])
            l_error = np.asarray([predictions[i][model]["output_tokens"] - rows[i].output_lengths[model_number] for i in selected])
            result[bucket][model] = {
                "num_prompts": len(selected),
                "quality_mae": float(np.abs(q_error).mean()),
                "quality_rmse": float(np.sqrt((q_error ** 2).mean())),
                "length_mae_tokens": float(np.abs(l_error).mean()),
                "length_rmse_tokens": float(np.sqrt((l_error ** 2).mean())),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--completions-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--buckets", nargs="+", default=list(DEFAULT_BUCKETS))
    parser.add_argument("--expected-per-bucket", type=int, default=2500)
    parser.add_argument("--max-completion-tokens", type=int, default=8192)
    parser.add_argument("--validation-id-files", type=Path, nargs="+", default=[])
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=69)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--distance-epsilon", type=float, default=1e-6)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--allow-encoder-download", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="Audit calibration and splits without model/dependencies/output")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"Refusing to overwrite existing artifact directory: {args.output_dir}")
    start = time.perf_counter()
    rows, provenance = load_calibration(
        args.completions_root, model_labels=args.models, buckets=args.buckets,
        expected_per_bucket=args.expected_per_bucket, max_completion_tokens=args.max_completion_tokens,
    )
    training, validation, split = split_calibration(
        rows, validation_id_files=args.validation_id_files,
        validation_fraction=args.validation_fraction, split_seed=args.split_seed,
    )
    if len(training) < args.k:
        parser.error(f"Need at least k={args.k} training prompts, found {len(training)}")
    provenance.update(split)
    provenance["encoder"] = encoder_metadata()
    provenance["encoder_cache_dir"] = str(args.cache_dir.resolve()) if args.cache_dir is not None else None
    if args.validate_only:
        print(json.dumps({"status": "PASS", "num_train": len(training), "num_validation": len(validation),
                          "provenance": provenance}, indent=2, allow_nan=False))
        return
    import numpy as np

    print(f"Audited calibration: {len(training)} training / {len(validation)} validation; loading CPU MiniLM",
          file=sys.stderr, flush=True)
    embedder = MiniLMCPUEmbedder(
        provenance["encoder"], cache_dir=args.cache_dir,
        local_files_only=not args.allow_encoder_download, cpu_threads=args.cpu_threads,
    )
    provenance["encoder"] = dict(embedder.metadata)
    encode_start = time.perf_counter()
    print(f"Embedding {len(training)} training prompts with {args.cpu_threads} CPU threads",
          file=sys.stderr, flush=True)
    embeddings = embedder.encode([row.prompt for row in training], batch_size=args.embedding_batch_size)
    predictor = RouteBalancePredictor(
        embeddings=embeddings,
        qualities=np.asarray([row.qualities for row in training]),
        output_lengths=np.asarray([row.output_lengths for row in training]),
        model_labels=args.models, embedder=embedder,
        k=args.k, distance_epsilon=args.distance_epsilon,
        max_completion_tokens=args.max_completion_tokens,
        embedding_batch_size=args.embedding_batch_size, cpu_threads=args.cpu_threads,
    )
    provenance["training_embedding_and_index_seconds"] = time.perf_counter() - encode_start
    validation_start = time.perf_counter()
    print(f"Training embedding/index complete in {provenance['training_embedding_and_index_seconds']:.1f}s; "
          f"predicting {len(validation)} validation prompts", file=sys.stderr, flush=True)
    predictions = predictor.predict_batch([row.prompt for row in validation])
    provenance["validation_prediction_seconds"] = time.perf_counter() - validation_start
    provenance["validation_metrics"] = validation_metrics(validation, predictions, args.models)
    provenance["build_seconds_before_save"] = time.perf_counter() - start
    provenance["build_cpu_threads"] = args.cpu_threads
    metadata = save_predictor_artifact(
        args.output_dir, predictor, train_prompts=[row.manifest() for row in training],
        validation_prompts=[row.manifest() for row in validation], provenance=provenance,
    )
    print(json.dumps({"status": "PASS", "artifact_dir": str(args.output_dir.resolve()),
                      "num_train": metadata["num_train_prompts"],
                      "num_validation": metadata["num_validation_prompts"],
                      "validation_metrics": metadata["validation_metrics"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

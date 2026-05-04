#!/usr/bin/env python3
"""
Train and export the output-length prediction model used by wait-time simulation.

This script expects JSONL logs that include ``prompt`` text, ``prompt_tokens``,
``model_label`` (or ``model_id``), and token usage metadata for the responses.
It fits three LightGBM regressors: one optimized for the mean (L2 loss),
one optimized for the median (Huber loss), and one trained with pinball loss
for a target quantile (P90/P95). The resulting artifact contains the
PCA-reduced semantic projection, feature metadata, and the serialized boosters
consumed by :mod:`vllm.v1.engine.output_length_predictor`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA

import lightgbm as lgb

from vllm.v1.engine.output_length_predictor import (
    AdmissionFeatures,
    HashingSemanticProjector,
    PromptFeatureExtractor,
)


@dataclass
class LengthTrainingExample:
    request_id: str
    model_id: str
    prompt_text: str
    prompt_tokens: int
    output_tokens: int
    target: float
    weight: float


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iter_inputs(paths: Iterable[str]) -> Iterator[dict]:
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            for child in sorted(path.glob("*.jsonl")):
                yield from _iter_jsonl(child)
        else:
            yield from _iter_jsonl(path)


def _extract_prompt_tokens(record: dict) -> int:
    for key in ("prompt_tokens", "num_prompt_tokens", "prompt_only_tokens"):
        value = record.get(key)
        if isinstance(value, int):
            return value
    prompt = record.get("prompt")
    if isinstance(prompt, str):
        return len(prompt.split())
    return 0


def _extract_model_id(record: dict) -> str:
    return record.get("model_label") or record.get("model_id") or "unknown-model"


def _extract_output_tokens(record: dict) -> Optional[int]:
    response = record.get("response") or {}
    return int(response.get("completion_tokens", 0))

def _get_nested(record: dict, dotted: Optional[str]) -> Optional[object]:
    if not dotted:
        return None
    cur: object = record
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
        if cur is None:
            return None
    return cur


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()


def _request_identifier(record: dict, candidate_keys: Sequence[str]) -> str:
    for key in candidate_keys:
        value = _get_nested(record, key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    prompt = record.get("prompt")
    if isinstance(prompt, str) and prompt:
        meta = record.get("prompt_metadata") or {}
        dataset = meta.get("dataset_id", "unknown")
        return f"{dataset}:{_hash_prompt(prompt)}"
    fallback = record.get("request_id") or record.get("prompt_index")
    if fallback is not None:
        return str(fallback)
    return _hash_prompt(json.dumps(record, sort_keys=True))


def _load_batch_times_csv(path: Path, label: Optional[str] = None) -> dict[str, float]:
    """Derive a penalty from the async batch stats CSV output."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} is empty or missing a header row.")
        total_exec = 0.0
        count = 0
        for row in reader:
            raw_value = row.get("exec") or row.get("interval")
            if raw_value in (None, ""):
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if value < 0:
                continue
            total_exec += value
            count += 1
    if count == 0:
        raise ValueError(
            f"Could not derive a penalty from {path}; no usable exec/interval entries."
        )
    avg_exec = total_exec / count
    key = label or "__default__"
    return {key: avg_exec}


def _load_batch_times_json(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain an object mapping model ids to floats.")
    result: dict[str, float] = {}
    for key, value in data.items():
        try:
            result[str(key)] = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Batch time entry for {key!r} must be numeric.") from None
    return result


def _load_batch_times(entries: Optional[Sequence[str]]) -> dict[str, float]:
    if not entries:
        return {}
    if isinstance(entries, str):
        entries = [entries]
    batch_times: dict[str, float] = {}
    for entry in entries:
        label: Optional[str] = None
        spec = entry.strip()
        if not spec:
            continue
        if "=" in spec:
            label, path_str = spec.split("=", 1)
            label = label.strip()
            path_str = path_str.strip()
            if not label:
                raise ValueError(
                    f"Invalid --model-batch-times entry {entry!r}; missing model id."
                )
        else:
            path_str = spec
        file_path = Path(path_str).expanduser()
        if not file_path.exists():
            raise FileNotFoundError(f"Batch-times file not found: {file_path}")
        if file_path.suffix.lower() == ".csv":
            values = _load_batch_times_csv(file_path, label=label)
        else:
            values = _load_batch_times_json(file_path)
            if label:
                value = values.get(label)
                if value is None:
                    if len(values) == 1:
                        value = next(iter(values.values()))
                    else:
                        raise ValueError(
                            f"JSON file {file_path} does not contain an entry for {label!r}."
                        )
                values = {label: value}
        batch_times.update(values)
    return batch_times


def _build_model_lookup(model_ids: Iterable[str]) -> dict[str, int]:
    unique = sorted(set(model_ids))
    lookup = {model_id: idx for idx, model_id in enumerate(unique)}
    lookup["__unknown__"] = len(unique)
    return lookup


def _parse_examples(
    input_paths: Sequence[str],
    *,
    request_id_fields: str,
    delta_penalty: float,
    batch_times: dict[str, float],
) -> list[LengthTrainingExample]:
    examples: list[LengthTrainingExample] = []
    candidate_req_keys = [key.strip() for key in request_id_fields.split(",") if key.strip()]

    for record in _iter_inputs(input_paths):
        prompt_text = record.get("prompt")
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            continue
        prompt_tokens = _extract_prompt_tokens(record)
        if prompt_tokens <= 0:
            continue
        output_tokens = _extract_output_tokens(record)
        if not output_tokens:
            continue
        model_id = _extract_model_id(record)
        penalty = batch_times.get(model_id, batch_times.get("__default__", 1.0))
        request_id = _request_identifier(record, candidate_req_keys)
        examples.append(
            LengthTrainingExample(
                request_id=request_id,
                model_id=model_id,
                prompt_text=prompt_text,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                target=math.log1p(output_tokens),
                weight=float(delta_penalty) * penalty,
            )
        )
    return examples


def _split_examples_by_request(
    examples: list[LengthTrainingExample], test_fraction: float, seed: int
) -> Tuple[list[LengthTrainingExample], list[LengthTrainingExample]]:
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("--test-fraction must be in [0.0, 1.0).")
    if test_fraction == 0.0:
        return examples, []
    by_request: dict[str, list[LengthTrainingExample]] = defaultdict(list)
    for ex in examples:
        by_request[ex.request_id].append(ex)
    request_ids = list(by_request.keys())
    if len(request_ids) < 2:
        raise RuntimeError(
            "Need at least two unique request groups to create a train/test split."
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(request_ids)
    test_groups = int(round(len(request_ids) * test_fraction))
    test_groups = max(1, min(test_groups, len(request_ids) - 1))
    test_request_ids = set(request_ids[:test_groups])
    train_examples: list[LengthTrainingExample] = []
    test_examples: list[LengthTrainingExample] = []
    for req_id, req_examples in by_request.items():
        if req_id in test_request_ids:
            test_examples.extend(req_examples)
        else:
            train_examples.extend(req_examples)
    if not train_examples or not test_examples:
        raise RuntimeError(
            "Could not create a non-empty train/test split. Adjust --test-fraction."
        )
    return train_examples, test_examples


def _build_length_features(
    examples: list[LengthTrainingExample], extractor: PromptFeatureExtractor
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    admissions = [
        AdmissionFeatures(
            model_id=ex.model_id,
            prompt_text=ex.prompt_text,
            prompt_token_count=ex.prompt_tokens,
        )
        for ex in examples
    ]
    features = np.vstack(
        [extractor.build_feature_row(adm) for adm in admissions]
    ).astype(np.float32)
    log_targets = np.asarray([ex.target for ex in examples], dtype=np.float32)
    weights = np.asarray([ex.weight for ex in examples], dtype=np.float32)
    output_tokens = np.asarray([ex.output_tokens for ex in examples], dtype=np.float32)
    return features, log_targets, weights, output_tokens


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    errors = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
    }


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff)))


def _evaluate_predictions(
    y_true_log: np.ndarray,
    y_true_tokens: np.ndarray,
    y_pred_log: np.ndarray,
    *,
    alpha: Optional[float] = None,
) -> dict[str, float]:
    metrics = {
        "log": _regression_metrics(y_true_log, y_pred_log),
    }
    pred_tokens = np.clip(np.expm1(y_pred_log), 0.0, None)
    metrics["tokens"] = _regression_metrics(y_true_tokens, pred_tokens)
    if alpha is not None:
        metrics["log_pinball"] = _pinball_loss(y_true_log, y_pred_log, alpha)
        metrics["tokens_pinball"] = _pinball_loss(y_true_tokens, pred_tokens, alpha)
    return metrics


def _evaluate_token_predictions(
    y_true_log: np.ndarray,
    y_true_tokens: np.ndarray,
    y_pred_tokens: np.ndarray,
) -> dict[str, dict[str, float]]:
    pred_tokens = np.clip(y_pred_tokens, 0.0, None)
    pred_log = np.log1p(pred_tokens)
    return {
        "log": _regression_metrics(y_true_log, pred_log),
        "tokens": _regression_metrics(y_true_tokens, pred_tokens),
    }


def _write_test_example_ids(
    output_dir: Path,
    *,
    request_id_fields: str,
    test_examples: Sequence[LengthTrainingExample],
) -> str:
    test_request_ids = sorted({ex.request_id for ex in test_examples})
    payload: dict[str, object] = {
        "request_id_fields": request_id_fields,
        "num_test_request_groups": len(test_request_ids),
        "num_test_examples": len(test_examples),
        "example_ids": test_request_ids,
    }
    filename = "test_example_ids.json"
    with (output_dir / filename).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return filename


def train_model(args: argparse.Namespace) -> None:
    batch_times = _load_batch_times(args.model_batch_times)
    examples = _parse_examples(
        args.input,
        request_id_fields=args.request_id_fields,
        delta_penalty=args.delta_penalty,
        batch_times=batch_times,
    )
    if not examples:
        raise RuntimeError("No valid training records were found.")

    eval_examples: list[LengthTrainingExample] = []
    if args.eval_input:
        eval_examples = _parse_examples(
            args.eval_input,
            request_id_fields=args.request_id_fields,
            delta_penalty=args.delta_penalty,
            batch_times=batch_times,
        )
        if not eval_examples:
            raise RuntimeError("No valid evaluation records were found for --eval-input.")

    train_examples, test_examples = _split_examples_by_request(
        examples, test_fraction=args.test_fraction, seed=args.split_seed
    )
    if not train_examples:
        raise RuntimeError("No training examples remain after preprocessing.")

    projector = HashingSemanticProjector(dimension=args.hash_dim)
    hash_vectors = [projector.hash_text(ex.prompt_text) for ex in train_examples]
    model_ids = [ex.model_id for ex in train_examples]

    hash_matrix = np.vstack(hash_vectors)
    pca = PCA(n_components=args.semantic_dim, random_state=42)
    pca.fit(hash_matrix)
    projector = HashingSemanticProjector(
        dimension=args.hash_dim,
        mean=pca.mean_,
        components=pca.components_,
    )

    model_lookup = _build_model_lookup(model_ids)
    extractor = PromptFeatureExtractor(
        model_lookup=model_lookup,
        semantic_projector=projector,
    )
    feature_names = extractor.feature_names
    train_features, train_log_targets, train_weights, train_token_targets = (
        _build_length_features(train_examples, extractor)
    )
    if test_examples:
        test_features, test_log_targets, _, test_tokens = _build_length_features(
            test_examples, extractor
        )
    else:
        test_features = np.empty((0, train_features.shape[1]), dtype=np.float32)
        test_log_targets = np.empty((0,), dtype=np.float32)
        test_tokens = np.empty((0,), dtype=np.float32)

    lgb_dataset_log = lgb.Dataset(
        train_features,
        label=train_log_targets,
        weight=train_weights,
        feature_name=feature_names,
        categorical_feature=[0],
        free_raw_data=False,
    )
    lgb_dataset_tokens = lgb.Dataset(
        train_features,
        label=train_token_targets,
        weight=train_weights,
        feature_name=feature_names,
        categorical_feature=[0],
        free_raw_data=False,
    )

    median_params = {
        "objective": "huber",
        "metric": "None",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        # "max_depth": -1,
        "max_depth": 8,
        "verbose": 1,
        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        "max_bin": 63,
        # "early_stopping_rounds": 50,
    }
    mean_params = dict(median_params)
    mean_params.update({"objective": "regression"})

    tail_params = dict(median_params)
    tail_params.update(
        {
            "objective": "quantile",
            "alpha": args.tail_quantile,
        }
    )

    mean_model = lgb.train(
        params=mean_params,
        train_set=lgb_dataset_tokens,
        num_boost_round=args.num_boost_round,
    )
    median_model = lgb.train(
        params=median_params,
        train_set=lgb_dataset_log,
        num_boost_round=args.num_boost_round,
    )
    tail_model = lgb.train(
        params=tail_params,
        train_set=lgb_dataset_log,
        num_boost_round=args.num_boost_round,
    )
    test_metrics: Optional[dict[str, object]] = None
    if test_examples:
        mean_pred = np.asarray(mean_model.predict(test_features), dtype=np.float32)
        median_pred = np.asarray(median_model.predict(test_features), dtype=np.float32)
        tail_pred = np.asarray(tail_model.predict(test_features), dtype=np.float32)
        test_metrics = {
            "mean_head": _evaluate_token_predictions(
                test_log_targets, test_tokens, mean_pred
            ),
            "median_head": _evaluate_predictions(
                test_log_targets, test_tokens, median_pred, alpha=0.5
            ),
            "tail_head": _evaluate_predictions(
                test_log_targets, test_tokens, tail_pred, alpha=args.tail_quantile
            ),
        }
    eval_metrics: Optional[dict[str, object]] = None
    if eval_examples:
        eval_features, eval_log_targets, _, eval_tokens = _build_length_features(
            eval_examples, extractor
        )
        eval_mean_pred = np.asarray(mean_model.predict(eval_features), dtype=np.float32)
        eval_median_pred = np.asarray(median_model.predict(eval_features), dtype=np.float32)
        eval_tail_pred = np.asarray(tail_model.predict(eval_features), dtype=np.float32)
        eval_metrics = {
            "mean_head": _evaluate_token_predictions(
                eval_log_targets, eval_tokens, eval_mean_pred
            ),
            "median_head": _evaluate_predictions(
                eval_log_targets, eval_tokens, eval_median_pred, alpha=0.5
            ),
            "tail_head": _evaluate_predictions(
                eval_log_targets, eval_tokens, eval_tail_pred, alpha=args.tail_quantile
            ),
        }

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    mean_model.save_model(str(output_dir / "mean_model.txt"))
    median_model.save_model(str(output_dir / "median_model.txt"))
    tail_model.save_model(str(output_dir / "quantile_model.txt"))
    test_ids_filename = _write_test_example_ids(
        output_dir,
        request_id_fields=args.request_id_fields,
        test_examples=test_examples,
    )

    metadata = {
        "hash_dim": args.hash_dim,
        "pca_mean": projector.metadata()["pca_mean"],
        "pca_components": projector.metadata()["pca_components"],
        "model_id_lookup": model_lookup,
        "feature_order": feature_names,
        "tail_quantile": args.tail_quantile,
        "mean_model_file": "mean_model.txt",
        "mean_model_target": "tokens",
        "train_examples": len(train_examples),
        "test_examples": len(test_examples),
        "delta_penalty": args.delta_penalty,
        "split_seed": args.split_seed,
        "test_fraction": args.test_fraction,
        "test_example_ids_file": test_ids_filename,
        "eval_examples": len(eval_examples),
        "eval_input": list(args.eval_input) if args.eval_input else [],
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    summary = {
        "train_examples": len(train_examples),
        "test_examples": len(test_examples),
        "total_examples": len(examples),
        "model_ids": sorted(set(model_ids)),
        "tail_quantile": args.tail_quantile,
        "learning_rate": args.learning_rate,
        "num_boost_round": args.num_boost_round,
        "semantic_dim": args.semantic_dim,
        "hash_dim": args.hash_dim,
        "split_seed": args.split_seed,
        "test_fraction": args.test_fraction,
        "test_example_ids_file": test_ids_filename,
        "eval_examples": len(eval_examples),
        "eval_input": list(args.eval_input) if args.eval_input else [],
    }
    if test_metrics is not None:
        summary["test_metrics"] = test_metrics
    if eval_metrics is not None:
        summary["eval_metrics"] = eval_metrics
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Wrote output-length model to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the output-length prediction model for wait-time simulation."
    )
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="One or more JSONL files or directories containing logs.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the trained model + metadata will be written.",
    )
    parser.add_argument(
        "--model-batch-times",
        action="append",
        help=(
            "Repeatable. Either path to a JSON mapping of model_id->seconds or "
            "entries like model_id=/path/to/batch_stats.csv captured via "
            "--batch-stats-file (we derive the average exec time). Include "
            "__default__=... to override the fallback penalty."
        ),
    )
    parser.add_argument(
        "--request-id-fields",
        type=str,
        default="prompt_metadata.example_id,prompt_index,request_id",
        help="Comma-separated dotted paths used to group records into requests.",
    )
    parser.add_argument(
        "--tail-quantile",
        type=float,
        default=0.9,
        help="Quantile to learn for the tail head (e.g., 0.9 or 0.95).",
    )
    parser.add_argument(
        "--delta-penalty",
        type=float,
        default=1.0,
        help="Global scaling factor for the wait-time penalty weights.",
    )
    parser.add_argument(
        "--hash-dim",
        type=int,
        default=512,
        help="Embedding dimension for the hashed semantic feature.",
    )
    parser.add_argument(
        "--semantic-dim",
        type=int,
        default=16,
        help="Number of PCA components to retain for the semantic feature.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate for LightGBM.",
    )
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=128,
        help="Maximum number of leaves for LightGBM trees.",
    )
    parser.add_argument(
        "--min-data-in-leaf",
        type=int,
        default=50,
        help="Minimum records per leaf to reduce overfitting.",
    )
    parser.add_argument(
        "--num-boost-round",
        type=int,
        default=500,
        help="Boosting rounds for each head.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0,
        help=(
            "Fraction of request groups held out for testing/generalization checks. "
            "Set to 0 to disable holdout."
        ),
    )
    parser.add_argument(
        "--eval-input",
        nargs="+",
        default=None,
        help=(
            "Optional JSONL files or directories used only for evaluation "
            "(for example: holdout_cache-scored outputs)."
        ),
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=69,
        help="Random seed used for request-group train/test splitting.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train_model(parse_args())

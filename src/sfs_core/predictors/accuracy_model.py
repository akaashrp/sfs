#!/usr/bin/env python3
"""
Train the raw accuracy predictor used by the routing system.

The model is a LightGBM regressor that consumes prompt-level features alongside
static model descriptors (identifier index, log-parameters, max context length).
It predicts an accuracy score hat A(x, m). By default this score is trained in
raw space; optionally a logit link can be enabled so predictions are mapped
back through sigmoid and therefore remain in (0, 1). User-specified lambda and
delta are applied at inference time when computing utility:
  U(x,m) = hat A(x,m) - lambda * Cost(x,m) - delta * TTFT(x,m).

This script trains only hat A(x,m).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
from sklearn.decomposition import PCA

from vllm.v1.engine.output_length_predictor import (
    AdmissionFeatures,
    HashingSemanticProjector,
    PromptFeatureExtractor,
)
from vllm.v1.engine.accuracy_predictor import AccuracyFeatureBuilder


@dataclass
class TrainingExample:
    request_id: str
    model_id: str
    prompt_text: str
    prompt_tokens: int
    quality: float
    quality_metric: str


def _iter_jsonl(path: Path) -> Iterator[dict]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            cursor = 0
            while cursor < len(line):
                try:
                    record, end = decoder.raw_decode(line, cursor)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Failed to parse {path}:{lineno}: {exc}") from exc
                yield record
                cursor = end
                while cursor < len(line) and line[cursor].isspace():
                    cursor += 1


def _iter_inputs(paths: Iterable[str]) -> Iterator[dict]:
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            for child in sorted(path.glob("*.jsonl")):
                yield from _iter_jsonl(child)
        else:
            yield from _iter_jsonl(path)


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


def _get_nested_float(record: dict, dotted: Optional[str]) -> Optional[float]:
    value = _get_nested(record, dotted)
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def _extract_prompt_tokens(record: dict) -> int:
    for key in ("prompt_only_tokens", "prompt_tokens", "num_prompt_tokens"):
        value = record.get(key)
        if isinstance(value, int):
            return value
    prompt = record.get("prompt")
    if isinstance(prompt, str):
        return len(prompt.split())
    return 0


def _extract_model_id(record: dict) -> str:
    for key in ("model_label", "model_id", "model"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return "unknown-model"


def _extract_quality(record: dict, field: Optional[str]) -> Optional[float]:
    if field:
        return _get_nested_float(record, field)
    for key in ("quality", "quality_score", "score"):
        value = record.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    metrics = record.get("metrics")
    if isinstance(metrics, dict):
        value = metrics.get("quality")
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _extract_quality_metric(record: dict) -> str:
    metric = record.get("quality_metric")
    if isinstance(metric, str) and metric.strip():
        return metric.strip()
    metrics = record.get("metrics")
    if isinstance(metrics, dict):
        candidate = metrics.get("quality_metric") or metrics.get("metric")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return "unknown"


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
    return record.get("request_id") or record.get("prompt_index") or _hash_prompt(
        json.dumps(record, sort_keys=True)
    )


def _load_model_metadata(path: str) -> Dict[str, Dict[str, float]]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    parsed: Dict[str, Dict[str, float]] = {}
    for model_id, specs in data.items():
        num_params = float(specs.get("num_params", specs.get("parameters", 0.0)))
        if num_params <= 0.0:
            num_params = 1.0
        parsed[model_id] = {
            # Natural log is fine as long as consistent.
            "log_num_params": float(np.log(num_params)),
            "max_context_length": float(
                specs.get("max_context_length", specs.get("max_model_len", 0.0))
            ),
        }
    if "__default__" not in parsed:
        parsed["__default__"] = {"log_num_params": 0.0, "max_context_length": 0.0}
    return parsed


def _build_model_lookup(model_ids: Iterable[str]) -> Dict[str, int]:
    unique = sorted(set(model_ids))
    lookup = {model_id: idx for idx, model_id in enumerate(unique)}
    lookup["__unknown__"] = len(unique)
    return lookup


def _parse_examples(args: argparse.Namespace) -> List[TrainingExample]:
    examples: List[TrainingExample] = []
    candidate_req_keys = [key.strip() for key in args.request_id_fields.split(",") if key.strip()]

    for record in _iter_inputs(args.input):
        prompt = record.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            continue

        prompt_tokens = _extract_prompt_tokens(record)
        if prompt_tokens <= 0:
            continue

        quality = _extract_quality(record, args.quality_field)
        if quality is None:
            continue

        model_id = _extract_model_id(record)
        quality_metric = _extract_quality_metric(record)
        request_id = _request_identifier(record, candidate_req_keys)

        examples.append(
            TrainingExample(
                request_id=request_id,
                model_id=model_id,
                prompt_text=prompt,
                prompt_tokens=prompt_tokens,
                quality=float(quality),
                quality_metric=quality_metric,
            )
        )

    return examples


def _balance_quality_metrics(
    examples: List[TrainingExample], seed: int = 42
) -> List[TrainingExample]:
    """Downsample so every model has the same count per quality metric."""
    if not examples:
        return []
    by_metric: Dict[str, Dict[str, List[TrainingExample]]] = defaultdict(lambda: defaultdict(list))
    for ex in examples:
        by_metric[ex.quality_metric][ex.model_id].append(ex)
    rng = np.random.default_rng(seed)
    balanced: List[TrainingExample] = []
    for per_model in by_metric.values():
        if not per_model:
            continue
        target = min(len(items) for items in per_model.values())
        if target <= 0:
            continue
        for items in per_model.values():
            if len(items) <= target:
                balanced.extend(items)
            else:
                idx = rng.choice(len(items), size=target, replace=False)
                balanced.extend(items[i] for i in idx)
    return balanced


def _split_examples_by_request(
    examples: List[TrainingExample], test_fraction: float, seed: int
) -> Tuple[List[TrainingExample], List[TrainingExample]]:
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("--test-fraction must be in [0.0, 1.0).")
    if test_fraction == 0.0:
        return examples, []
    by_request: Dict[str, List[TrainingExample]] = defaultdict(list)
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
    train_examples: List[TrainingExample] = []
    test_examples: List[TrainingExample] = []
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


def _build_accuracy_features(
    examples: List[TrainingExample], feature_builder: AccuracyFeatureBuilder
) -> Tuple[np.ndarray, np.ndarray]:
    admissions = [
        AdmissionFeatures(
            model_id=ex.model_id,
            prompt_text=ex.prompt_text,
            prompt_token_count=ex.prompt_tokens,
        )
        for ex in examples
    ]
    features = np.vstack(
        [feature_builder.build_feature_row(adm) for adm in admissions]
    ).astype(np.float32)
    labels = np.asarray([ex.quality for ex in examples], dtype=np.float32)
    return features, labels


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    errors = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
    }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _apply_target_link(
    labels: np.ndarray, *, target_link: str, epsilon: float
) -> np.ndarray:
    if target_link == "identity":
        return labels
    if target_link == "logit":
        clipped = np.clip(labels, epsilon, 1.0 - epsilon)
        return np.log(clipped / (1.0 - clipped))
    raise ValueError(f"Unsupported target_link={target_link!r}")


def _invert_target_link(predictions: np.ndarray, *, target_link: str) -> np.ndarray:
    if target_link == "identity":
        return predictions
    if target_link == "logit":
        return _sigmoid(predictions)
    raise ValueError(f"Unsupported target_link={target_link!r}")


def _write_test_example_ids(
    output_dir: Path,
    *,
    request_id_fields: str,
    test_examples: Sequence[TrainingExample],
) -> str:
    test_request_ids = sorted({ex.request_id for ex in test_examples})
    payload: Dict[str, object] = {
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
    examples = _parse_examples(args)
    if not examples:
        raise RuntimeError("No valid training examples were extracted.")
    if args.target_link not in {"identity", "logit"}:
        raise ValueError("--target-link must be one of: identity, logit.")
    if not 0.0 < args.link_epsilon < 0.5:
        raise ValueError("--link-epsilon must be in (0.0, 0.5).")

    label_values = np.asarray([ex.quality for ex in examples], dtype=np.float32)
    if not np.all(np.isfinite(label_values)):
        raise RuntimeError("Non-finite quality labels found in training data.")
    if args.target_link == "logit":
        label_min = float(np.min(label_values))
        label_max = float(np.max(label_values))
        if label_min < 0.0 or label_max > 1.0:
            raise RuntimeError(
                "Logit link requires quality labels in [0, 1], but found "
                f"min={label_min:.6f}, max={label_max:.6f}."
            )

    train_examples, test_examples = _split_examples_by_request(
        examples, test_fraction=args.test_fraction, seed=args.split_seed
    )
    if not args.no_balance_quality_metrics:
        train_examples = _balance_quality_metrics(train_examples, seed=args.split_seed)
    if not train_examples:
        raise RuntimeError("No training examples remain after preprocessing.")

    # Hashed semantic features -> PCA projection
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

    # Feature extraction
    model_lookup = _build_model_lookup(model_ids)
    prompt_extractor = PromptFeatureExtractor(
        model_lookup=model_lookup,
        semantic_projector=projector,
    )

    model_metadata = _load_model_metadata(args.model_metadata)
    feature_builder = AccuracyFeatureBuilder(
        prompt_extractor=prompt_extractor,
        model_descriptors=model_metadata,
    )

    train_features, train_labels = _build_accuracy_features(
        train_examples, feature_builder
    )
    train_targets = _apply_target_link(
        train_labels,
        target_link=args.target_link,
        epsilon=args.link_epsilon,
    )
    test_metrics: Optional[Dict[str, float]] = None
    if test_examples:
        test_features, test_labels = _build_accuracy_features(
            test_examples, feature_builder
        )
    else:
        test_features = np.empty((0, train_features.shape[1]), dtype=np.float32)
        test_labels = np.empty((0,), dtype=np.float32)

    # Dataset (regression)
    dataset = lgb.Dataset(
        train_features,
        label=train_targets,
        feature_name=feature_builder.feature_names,
        categorical_feature=[0],  # assumes first column is model_id index
        free_raw_data=False,
    )

    params = {
        "objective": "huber",  # robust regression for noisy labels
        "metric": ["l2", "l1"],
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "max_depth": -1,
        "verbose": 1,
        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        "max_bin": 63,
    }

    booster = lgb.train(
        params=params,
        train_set=dataset,
        num_boost_round=args.num_boost_round,
    )
    if test_examples:
        test_pred_link_space = np.asarray(
            booster.predict(test_features),
            dtype=np.float32,
        )
        test_pred = _invert_target_link(
            test_pred_link_space,
            target_link=args.target_link,
        )
        test_metrics = _regression_metrics(test_labels, test_pred)

    # Write artifacts
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "accuracy_model.txt"
    booster.save_model(str(model_path))
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
        "prompt_feature_order": prompt_extractor.feature_names,
        "feature_order": feature_builder.feature_names,
        "model_descriptors": model_metadata,
        "train_examples": len(train_examples),
        "test_examples": len(test_examples),
        "semantic_dim": args.semantic_dim,
        "split_seed": args.split_seed,
        "test_fraction": args.test_fraction,
        "target_link": args.target_link,
        "link_epsilon": args.link_epsilon,
        "test_example_ids_file": test_ids_filename,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    summary = {
        "train_examples": len(train_examples),
        "test_examples": len(test_examples),
        "total_examples": len(examples),
        "models": sorted(set(model_ids)),
        "learning_rate": args.learning_rate,
        "num_boost_round": args.num_boost_round,
        "semantic_dim": args.semantic_dim,
        "hash_dim": args.hash_dim,
        "objective": params["objective"],
        "split_seed": args.split_seed,
        "test_fraction": args.test_fraction,
        "target_link": args.target_link,
        "link_epsilon": args.link_epsilon,
        "quality_balancing_enabled": not args.no_balance_quality_metrics,
        "test_example_ids_file": test_ids_filename,
    }
    if test_metrics is not None:
        summary["test_metrics"] = test_metrics
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Wrote raw-accuracy model to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the LightGBM raw-accuracy regression model."
    )
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="JSONL files or directories containing scored generations.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the trained model + metadata will be written.",
    )
    parser.add_argument(
        "--model-metadata",
        required=True,
        help="JSON mapping of model_id to descriptors (num_params, max_context_length).",
    )
    parser.add_argument(
        "--quality-field",
        help="Optional dotted path to the accuracy label field (default: 'quality').",
    )
    parser.add_argument(
        "--request-id-fields",
        type=str,
        default="prompt_metadata.example_id,prompt_index,request_id",
        help="Comma-separated dotted paths used to group records into requests.",
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
        default=256,
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
        help="Boosting rounds for the regression model.",
    )
    parser.add_argument(
        "--target-link",
        type=str,
        default="identity",
        choices=("identity", "logit"),
        help=(
            "Label-link used during training. 'identity' keeps the existing "
            "unbounded regressor; 'logit' trains in logit space and maps "
            "predictions through sigmoid to keep them in (0, 1)."
        ),
    )
    parser.add_argument(
        "--link-epsilon",
        type=float,
        default=1e-4,
        help=(
            "Numerical epsilon used when --target-link=logit to avoid logit(0) "
            "and logit(1). Effective labels are clipped to [epsilon, 1-epsilon]."
        ),
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
        "--split-seed",
        type=int,
        default=69,
        help="Random seed used for request-group train/test splitting and balancing.",
    )
    parser.add_argument(
        "--no-balance-quality-metrics",
        action="store_true",
        help="Disable quality-metric balancing (default balances to the minimum count).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train_model(parse_args())

"""
python accuracy_model.py --input /ocean/projects/cis250162p/aparthas/sfs/experiments/bucketed_prompt_outputs/qwen3-0.6b/scored /ocean/projects/cis250162p/aparthas/sfs/experiments/bucketed_prompt_outputs/qwen3-8b/scored /ocean/projects/cis250162p/aparthas/sfs/experiments/bucketed_prompt_outputs/qwen3-32b/scored --output-dir /ocean/projects/cis250162p/aparthas/sfs/src/assets/predictors/accuracy_predictor --model-metadata /ocean/projects/cis250162p/aparthas/sfs/experiments/bucketed_prompt_outputs/model_metadata.json --test-fraction 0.1 --no-balance-quality-metrics
"""

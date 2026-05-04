"""Collect per-model, per-dataset input/output lengths into one JSON file.

By default this reads:
  <sfs>/experiments/bucketed_prompt_outputs/
    qwen3-0.6b/outputs/*.jsonl
    qwen3-8b/outputs/*.jsonl
    qwen3-32b/outputs/*.jsonl

Example:
  python collect_model_dataset_lengths.py \
    --output-json /ocean/projects/cis250162p/aparthas/sfs/experiments/bucketed_prompt_outputs/model_dataset_lengths.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sfs_core.paths import BUCKETED_OUTPUTS_ROOT
from sfs_core.shared.shared_experiment_helpers import as_nonnegative_int

DEFAULT_BUCKETED_ROOT = BUCKETED_OUTPUTS_ROOT
DEFAULT_MODELS = ["qwen3-0.6b", "qwen3-8b", "qwen3-32b"]
DEFAULT_SUBDIR = "outputs"
DEFAULT_OUTPUT_JSON = (
    DEFAULT_BUCKETED_ROOT / "model_dataset_input_output_lengths.json"
)

SCORED_SUFFIX = "_scored.jsonl"

INPUT_LEN_PATHS = [
    ("prompt_only_tokens",),
    ("prompt_tokens",),
    ("num_prompt_tokens",),
    ("prompt_metadata", "prompt_tokens"),
]
OUTPUT_LEN_PATHS = [
    ("response", "completion_tokens"),
    ("completion_tokens",),
    ("usage_completion_tokens",),
    ("response", "output_tokens"),
    ("output_tokens",),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect per-model per-dataset input/output lengths from JSONL outputs."
        )
    )
    parser.add_argument(
        "--bucketed-root",
        type=Path,
        default=DEFAULT_BUCKETED_ROOT,
        help=(
            "Root containing per-model directories "
            "(default: %(default)s). Ignored when --model-dir is provided."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=(
            "Model directory names under --bucketed-root "
            "(default: %(default)s). Ignored when --model-dir is provided."
        ),
    )
    parser.add_argument(
        "--model-dir",
        action="append",
        type=Path,
        default=[],
        help=(
            "Explicit model directory (for example: .../qwen3-8b). "
            "Can be provided multiple times."
        ),
    )
    parser.add_argument(
        "--subdir",
        default=DEFAULT_SUBDIR,
        help=(
            "Subdirectory under each model dir that contains JSONL files "
            "(default: %(default)s, e.g. outputs or scored)."
        ),
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help=(
            "Glob for dataset files inside --subdir. "
            "Default: '*.jsonl' for outputs, '*_scored.jsonl' for scored."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Path to write the aggregated JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent (default: %(default)s).",
    )
    return parser.parse_args()


def resolve_model_dirs(args: argparse.Namespace) -> list[Path]:
    if args.model_dir:
        return [path.expanduser().resolve() for path in args.model_dir]

    root = args.bucketed_root.expanduser().resolve()
    return [(root / model).resolve() for model in args.models]


def infer_dataset_name(path: Path) -> str:
    name = path.name
    if name.endswith(SCORED_SUFFIX):
        return name[: -len(SCORED_SUFFIX)]
    if name.endswith(".jsonl"):
        return name[: -len(".jsonl")]
    return path.stem


def _get_nested(mapping: dict[str, Any], path: tuple[str, ...]) -> Any | None:
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def _as_nonnegative_int(value: Any) -> int | None:
    return as_nonnegative_int(value, allow_strings=False)


def _word_count(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return 0
    return len(text.split())


def _extract_length(
    row: dict[str, Any],
    token_paths: list[tuple[str, ...]],
    fallback_text_paths: list[tuple[str, ...]],
) -> int | None:
    for path in token_paths:
        token_value = _as_nonnegative_int(_get_nested(row, path))
        if token_value is not None:
            return token_value

    for path in fallback_text_paths:
        count = _word_count(_get_nested(row, path))
        if count is not None:
            return count
    return None


def _resolve_source_dir(model_dir: Path, subdir: str) -> Path:
    if model_dir.is_dir() and model_dir.name == subdir:
        return model_dir

    candidate = model_dir / subdir
    if candidate.is_dir():
        return candidate

    if model_dir.is_dir():
        return model_dir

    raise FileNotFoundError(f"Model directory does not exist: {model_dir}")


def _infer_model_name(model_dir: Path, source_dir: Path, subdir: str) -> str:
    if source_dir.name == subdir and source_dir.parent.name:
        return source_dir.parent.name
    if model_dir.name:
        return model_dir.name
    return source_dir.name


def _mean(values: list[int]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def collect_dataset_lengths(path: Path) -> dict[str, Any]:
    input_lengths: list[int] = []
    output_lengths: list[int] = []
    paired_lengths: list[dict[str, int]] = []

    total_rows = 0
    invalid_json_rows = 0
    non_object_rows = 0

    with path.open("r", encoding="utf-8", errors="replace") as src:
        for line in src:
            text = line.strip()
            if not text:
                continue
            total_rows += 1
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                invalid_json_rows += 1
                continue

            if not isinstance(row, dict):
                non_object_rows += 1
                continue

            input_len = _extract_length(
                row,
                INPUT_LEN_PATHS,
                [("prompt",)],
            )
            output_len = _extract_length(
                row,
                OUTPUT_LEN_PATHS,
                [("response", "output_text"), ("output_text",)],
            )

            if input_len is not None:
                input_lengths.append(input_len)
            if output_len is not None:
                output_lengths.append(output_len)
            if input_len is not None and output_len is not None:
                paired_lengths.append(
                    {
                        "input_length": int(input_len),
                        "output_length": int(output_len),
                    }
                )

    return {
        "file": str(path),
        "total_rows": total_rows,
        "invalid_json_rows": invalid_json_rows,
        "non_object_rows": non_object_rows,
        "input_count": len(input_lengths),
        "output_count": len(output_lengths),
        "paired_count": len(paired_lengths),
        "input_length_mean": _mean(input_lengths),
        "output_length_mean": _mean(output_lengths),
        "input_lengths": input_lengths,
        "output_lengths": output_lengths,
        "paired_lengths": paired_lengths,
    }


def main() -> None:
    args = parse_args()
    if args.indent < 0:
        raise SystemExit("--indent must be >= 0.")

    pattern = args.pattern
    if pattern is None:
        pattern = "*_scored.jsonl" if args.subdir == "scored" else "*.jsonl"

    model_dirs = resolve_model_dirs(args)
    output_path = args.output_json.expanduser().resolve()

    models_payload: dict[str, Any] = {}
    total_dataset_files = 0
    total_rows = 0

    for model_dir in model_dirs:
        resolved_model_dir = model_dir.expanduser().resolve()
        if not resolved_model_dir.exists():
            print(
                f"warning: model directory not found: {resolved_model_dir}",
                file=sys.stderr,
            )
            continue
        if not resolved_model_dir.is_dir():
            print(
                f"warning: model path is not a directory: {resolved_model_dir}",
                file=sys.stderr,
            )
            continue

        source_dir = _resolve_source_dir(resolved_model_dir, args.subdir)
        files = sorted(p for p in source_dir.glob(pattern) if p.is_file())
        if not files:
            print(
                f"warning: no files matching '{pattern}' in {source_dir}",
                file=sys.stderr,
            )
            continue

        model_name = _infer_model_name(resolved_model_dir, source_dir, args.subdir)
        datasets_payload: dict[str, Any] = {}

        model_rows = 0
        for file_path in files:
            dataset_name = infer_dataset_name(file_path)
            dataset_payload = collect_dataset_lengths(file_path)
            datasets_payload[dataset_name] = dataset_payload
            total_dataset_files += 1
            model_rows += int(dataset_payload["total_rows"])

        total_rows += model_rows
        models_payload[model_name] = {
            "model_dir": str(resolved_model_dir),
            "source_dir": str(source_dir),
            "pattern": pattern,
            "dataset_count": len(datasets_payload),
            "total_rows": model_rows,
            "datasets": datasets_payload,
        }

    if not models_payload:
        raise SystemExit("No model data was collected.")

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bucketed_root": str(args.bucketed_root.expanduser().resolve()),
        "subdir": args.subdir,
        "pattern": pattern,
        "model_count": len(models_payload),
        "dataset_file_count": total_dataset_files,
        "total_rows": total_rows,
        "length_fields": {
            "input": [list(path) for path in INPUT_LEN_PATHS],
            "output": [list(path) for path in OUTPUT_LEN_PATHS],
        },
        "models": models_payload,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as dst:
        json.dump(payload, dst, indent=args.indent, sort_keys=True)
        dst.write("\n")

    print(f"Wrote {output_path}")
    print(
        "Collected "
        f"{len(models_payload)} models, {total_dataset_files} dataset files, {total_rows} rows."
    )


if __name__ == "__main__":
    main()

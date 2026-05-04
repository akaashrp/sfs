"""Aggregate per-dataset score and length statistics from scored JSONL outputs.

By default this script reads:
  <sfs>/experiments/bucketed_prompt_outputs/
    qwen3-0.6b/scored
    qwen3-8b/scored
    qwen3-32b/scored

Each JSONL row is expected to contain a numeric score field (default: ``quality``).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sfs_core.paths import BUCKETED_OUTPUTS_ROOT
from sfs_core.shared.shared_experiment_helpers import _compute_percentile

DEFAULT_MODELS = ["qwen3-0.6b", "qwen3-8b", "qwen3-32b"]
DEFAULT_BUCKETED_ROOT = BUCKETED_OUTPUTS_ROOT
SCORED_SUFFIX = "_scored.jsonl"

PROMPT_LEN_PATHS = [
    ("prompt_only_tokens",),
    ("prompt_tokens",),
    ("prompt_metadata", "prompt_tokens"),
]
OUTPUT_LEN_PATHS = [
    ("response", "completion_tokens"),
    ("completion_tokens",),
    ("response", "output_tokens"),
    ("output_tokens",),
]


@dataclass(frozen=True)
class DatasetStats:
    model: str
    dataset: str
    average: float | None
    p50: float | None
    p99: float | None
    score_count: int
    avg_prompt_len: float | None
    prompt_len_count: int
    avg_output_len: float | None
    output_len_count: int
    missing_or_null: int
    non_numeric: int
    invalid_json: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-dataset score stats from scored JSONL files."
    )
    parser.add_argument(
        "--bucketed-root",
        type=Path,
        default=DEFAULT_BUCKETED_ROOT,
        help=(
            "Root containing <model>/scored directories "
            "(default: %(default)s). Ignored when --scored-dir is provided."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=(
            "Model directory names under --bucketed-root "
            "(default: %(default)s). Ignored when --scored-dir is provided."
        ),
    )
    parser.add_argument(
        "--scored-dir",
        action="append",
        type=Path,
        default=[],
        help=(
            "Explicit scored directory. Can be provided multiple times. "
            "If set, --bucketed-root/--models are ignored."
        ),
    )
    parser.add_argument(
        "--score-field",
        default="quality",
        help="Field name in each JSONL row that contains the numeric score.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Decimal precision for printed numeric values (default: %(default)s).",
    )
    return parser.parse_args()


def infer_model_name(scored_dir: Path) -> str:
    if scored_dir.name == "scored" and scored_dir.parent.name:
        return scored_dir.parent.name
    return scored_dir.name


def dataset_name_from_file(path: Path) -> str:
    name = path.name
    if name.endswith(SCORED_SUFFIX):
        return name[: -len(SCORED_SUFFIX)]
    return path.stem


def find_scored_dirs(args: argparse.Namespace) -> list[Path]:
    if args.scored_dir:
        return [path.expanduser().resolve() for path in args.scored_dir]

    root = args.bucketed_root.expanduser().resolve()
    return [(root / model / "scored").resolve() for model in args.models]


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _get_nested(mapping: dict[str, object], path: tuple[str, ...]) -> object | None:
    current: object = mapping
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def _extract_length(
    row: dict[str, object],
    paths: list[tuple[str, ...]],
    fallback_text_path: tuple[str, ...] | None,
) -> float | None:
    for path in paths:
        numeric_value = _as_float(_get_nested(row, path))
        if numeric_value is not None:
            return numeric_value

    if fallback_text_path is not None:
        text_value = _get_nested(row, fallback_text_path)
        if isinstance(text_value, str):
            stripped = text_value.strip()
            if stripped:
                return float(len(stripped.split()))
    return None


def compute_stats(path: Path, score_field: str) -> tuple[
    float | None,
    float | None,
    float | None,
    int,
    float | None,
    int,
    float | None,
    int,
    int,
    int,
    int,
]:
    score_values: list[float] = []
    prompt_len_total = 0.0
    prompt_len_count = 0
    output_len_total = 0.0
    output_len_count = 0

    missing_or_null = 0
    non_numeric = 0
    invalid_json = 0

    with path.open("r", encoding="utf-8", errors="replace") as src:
        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_json += 1
                continue

            if not isinstance(row, dict):
                invalid_json += 1
                continue

            score_value = row.get(score_field)
            if score_value is None:
                missing_or_null += 1
            else:
                numeric_score = _as_float(score_value)
                if numeric_score is None:
                    non_numeric += 1
                else:
                    score_values.append(numeric_score)

            prompt_len = _extract_length(row, PROMPT_LEN_PATHS, ("prompt",))
            if prompt_len is not None:
                prompt_len_total += prompt_len
                prompt_len_count += 1

            output_len = _extract_length(row, OUTPUT_LEN_PATHS, ("response", "output_text"))
            if output_len is not None:
                output_len_total += output_len
                output_len_count += 1

    score_count = len(score_values)
    average = sum(score_values) / score_count if score_count else None
    p50 = _compute_percentile(score_values, 0.50)
    p99 = _compute_percentile(score_values, 0.99)
    avg_prompt_len = (
        prompt_len_total / prompt_len_count if prompt_len_count else None
    )
    avg_output_len = (
        output_len_total / output_len_count if output_len_count else None
    )

    return (
        average,
        p50,
        p99,
        score_count,
        avg_prompt_len,
        prompt_len_count,
        avg_output_len,
        output_len_count,
        missing_or_null,
        non_numeric,
        invalid_json,
    )


def format_number(value: float | None, precision: int) -> str:
    if value is None:
        return "NA"
    return f"{value:.{precision}f}"


def print_metric_matrix(
    *,
    title: str,
    models: list[str],
    datasets: list[str],
    stats_map: dict[tuple[str, str], DatasetStats],
    value_getter: Callable[[DatasetStats], float | None],
    precision: int,
) -> None:
    print(title)
    print("\t".join(["dataset", *models]))
    for dataset in datasets:
        row = [dataset]
        for model in models:
            stats = stats_map.get((model, dataset))
            row.append(
                format_number(value_getter(stats), precision) if stats else "NA"
            )
        print("\t".join(row))
    print("")


def print_detail_table(
    rows: list[DatasetStats],
    precision: int,
    score_field: str,
) -> None:
    print(f"Detailed stats per file (score_field={score_field})")
    print(
        "\t".join(
            [
                "model",
                "dataset",
                "avg",
                "p50",
                "p99",
                "avg_prompt_len",
                "avg_output_len",
                "score_count",
                "prompt_len_count",
                "output_len_count",
                "missing_or_null",
                "non_numeric",
                "invalid_json",
            ]
        )
    )
    for row in rows:
        print(
            "\t".join(
                [
                    row.model,
                    row.dataset,
                    format_number(row.average, precision),
                    format_number(row.p50, precision),
                    format_number(row.p99, precision),
                    format_number(row.avg_prompt_len, precision),
                    format_number(row.avg_output_len, precision),
                    str(row.score_count),
                    str(row.prompt_len_count),
                    str(row.output_len_count),
                    str(row.missing_or_null),
                    str(row.non_numeric),
                    str(row.invalid_json),
                ]
            )
        )


def main() -> None:
    args = parse_args()
    if args.precision < 0:
        raise SystemExit("--precision must be >= 0.")

    scored_dirs = find_scored_dirs(args)
    rows: list[DatasetStats] = []

    for scored_dir in scored_dirs:
        if not scored_dir.exists():
            print(f"warning: scored directory not found: {scored_dir}", file=sys.stderr)
            continue
        if not scored_dir.is_dir():
            print(f"warning: not a directory: {scored_dir}", file=sys.stderr)
            continue

        model_name = infer_model_name(scored_dir)
        files = sorted(p for p in scored_dir.glob(f"*{SCORED_SUFFIX}") if p.is_file())
        if not files:
            print(
                f"warning: no files matching *{SCORED_SUFFIX} in {scored_dir}",
                file=sys.stderr,
            )
            continue

        for path in files:
            (
                average,
                p50,
                p99,
                score_count,
                avg_prompt_len,
                prompt_len_count,
                avg_output_len,
                output_len_count,
                missing_or_null,
                non_numeric,
                invalid_json,
            ) = compute_stats(path, args.score_field)

            rows.append(
                DatasetStats(
                    model=model_name,
                    dataset=dataset_name_from_file(path),
                    average=average,
                    p50=p50,
                    p99=p99,
                    score_count=score_count,
                    avg_prompt_len=avg_prompt_len,
                    prompt_len_count=prompt_len_count,
                    avg_output_len=avg_output_len,
                    output_len_count=output_len_count,
                    missing_or_null=missing_or_null,
                    non_numeric=non_numeric,
                    invalid_json=invalid_json,
                )
            )

    if not rows:
        raise SystemExit("No scored data found.")

    models = list(dict.fromkeys(row.model for row in rows))
    datasets = sorted({row.dataset for row in rows})
    stats_map = {(row.model, row.dataset): row for row in rows}

    print_metric_matrix(
        title="Per-dataset average score",
        models=models,
        datasets=datasets,
        stats_map=stats_map,
        value_getter=lambda s: s.average,
        precision=args.precision,
    )
    print_metric_matrix(
        title="Per-dataset p50 score",
        models=models,
        datasets=datasets,
        stats_map=stats_map,
        value_getter=lambda s: s.p50,
        precision=args.precision,
    )
    print_metric_matrix(
        title="Per-dataset p99 score",
        models=models,
        datasets=datasets,
        stats_map=stats_map,
        value_getter=lambda s: s.p99,
        precision=args.precision,
    )
    print_metric_matrix(
        title="Per-dataset average prompt length",
        models=models,
        datasets=datasets,
        stats_map=stats_map,
        value_getter=lambda s: s.avg_prompt_len,
        precision=args.precision,
    )
    print_metric_matrix(
        title="Per-dataset average output length",
        models=models,
        datasets=datasets,
        stats_map=stats_map,
        value_getter=lambda s: s.avg_output_len,
        precision=args.precision,
    )

    print_detail_table(
        sorted(rows, key=lambda r: (r.dataset, r.model)),
        args.precision,
        args.score_field,
    )


if __name__ == "__main__":
    main()

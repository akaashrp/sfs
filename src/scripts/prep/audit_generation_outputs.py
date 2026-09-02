#!/usr/bin/env python3
"""Strictly audit aligned multi-model generation outputs before judging."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any


DEFAULT_BUCKETS = (
    "alpaca",
    "govreport-summarization",
    "hotpot_qa",
    "writingprompts",
)


def audit_generation_outputs(
    outputs_root: Path,
    *,
    model_names: tuple[str, ...],
    bucket_names: tuple[str, ...],
    expected_per_bucket: int,
) -> dict[str, Any]:
    errors: list[str] = []
    rows: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    alignment: dict[str, dict[str, set[str]]] = defaultdict(dict)

    for model_name in model_names:
        model_dir = outputs_root / model_name
        if not model_dir.is_dir():
            errors.append(f"missing model directory: {model_dir}")
            continue
        for bucket_name in bucket_names:
            path = model_dir / f"{bucket_name}.jsonl"
            counters = {
                "records": 0,
                "request_errors": 0,
                "empty_outputs": 0,
                "invalid_completion_tokens": 0,
                "missing_finish_reason": 0,
                "missing_example_id": 0,
                "duplicate_example_id": 0,
            }
            example_ids: set[str] = set()
            if not path.is_file():
                errors.append(f"missing output file: {path}")
                rows[model_name][bucket_name] = counters
                alignment[model_name][bucket_name] = example_ids
                continue
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    counters["records"] += 1
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        errors.append(f"{path}:{line_number}: invalid JSON: {exc}")
                        continue
                    if record.get("error") is not None:
                        counters["request_errors"] += 1
                    response = record.get("response")
                    if not isinstance(response, dict):
                        counters["empty_outputs"] += 1
                        counters["invalid_completion_tokens"] += 1
                        counters["missing_finish_reason"] += 1
                    else:
                        output_text = response.get("output_text")
                        if not isinstance(output_text, str) or not output_text.strip():
                            counters["empty_outputs"] += 1
                        completion_tokens = response.get("completion_tokens")
                        if (
                            not isinstance(completion_tokens, int)
                            or completion_tokens <= 0
                        ):
                            counters["invalid_completion_tokens"] += 1
                        if not response.get("finish_reason"):
                            counters["missing_finish_reason"] += 1
                    metadata = record.get("prompt_metadata")
                    example_id = (
                        metadata.get("example_id")
                        if isinstance(metadata, dict)
                        else None
                    )
                    if not isinstance(example_id, str) or not example_id:
                        counters["missing_example_id"] += 1
                    elif example_id in example_ids:
                        counters["duplicate_example_id"] += 1
                    else:
                        example_ids.add(example_id)

            if counters["records"] != expected_per_bucket:
                errors.append(
                    f"{path}: expected {expected_per_bucket} records, "
                    f"found {counters['records']}"
                )
            for counter_name, value in counters.items():
                if counter_name != "records" and value:
                    errors.append(f"{path}: {counter_name}={value}")
            rows[model_name][bucket_name] = counters
            alignment[model_name][bucket_name] = example_ids

    for bucket_name in bucket_names:
        model_sets = [
            alignment[model_name].get(bucket_name, set())
            for model_name in model_names
        ]
        if model_sets and any(ids != model_sets[0] for ids in model_sets[1:]):
            errors.append(f"{bucket_name}: example IDs are not aligned across models")

    return {
        "status": "PASS" if not errors else "FAIL",
        "outputs_root": str(outputs_root),
        "models": list(model_names),
        "buckets": list(bucket_names),
        "expected_per_bucket": expected_per_bucket,
        "rows": rows,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--buckets", nargs="+", default=DEFAULT_BUCKETS)
    parser.add_argument("--expected-per-bucket", type=int, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_per_bucket <= 0:
        parser.error("--expected-per-bucket must be positive")
    return args


def main() -> None:
    args = parse_args()
    result = audit_generation_outputs(
        args.outputs_root.expanduser().resolve(),
        model_names=tuple(args.models),
        bucket_names=tuple(args.buckets),
        expected_per_bucket=args.expected_per_bucket,
    )
    output_path = args.output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

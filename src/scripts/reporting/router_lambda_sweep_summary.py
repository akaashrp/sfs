#!/usr/bin/env python3
"""Summarize router sweep JSONs by lambda_weight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sfs_core.shared.router_sweep_summary_helpers import (
    _collect_run_metrics_for_group,
    _nested_get,
    _ordered_utilities,
)


DEFAULT_UTILITIES = [
    "hard",
    "hard_prefill_tps",
    "shortest_queue",
    "latency_agnostic",
    "round_robin",
    "instance_affinity",
]


def _format_lambda(value: float | int | str) -> str:
    if isinstance(value, str):
        value = float(value)
    numeric = float(value)
    if numeric == 0.0:
        return "0"

    if abs(numeric) >= 1e-2:
        return f"{numeric:g}"

    mantissa_raw, exponent_raw = f"{numeric:.15e}".split("e")
    mantissa = mantissa_raw.rstrip("0").rstrip(".")
    exponent = int(exponent_raw)
    return f"{mantissa}e{exponent}"


def _iter_json_files(input_paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in input_paths:
        if path.is_file() and path.suffix.lower() == ".json":
            if not path.name.endswith("manifest.json"):
                files.append(path)
            continue
        if path.is_dir():
            for json_path in sorted(path.glob("*.json")):
                if json_path.name.endswith("manifest.json"):
                    continue
                files.append(json_path)
    return sorted(set(files))


def aggregate_jsons(
    input_paths: list[Path],
    request_rate_qps: float | None,
    qps_tolerance: float,
    requested_utilities: list[str] | None,
    duplicate_policy: str,
) -> dict[str, Any]:
    lambda_map: dict[str, dict[str, Any]] = {}
    source_map: dict[str, dict[str, str]] = {}
    discovered_utilities: set[str] = set()
    qps_seen: set[str] = set()

    for json_path in _iter_json_files(input_paths):
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        qps_raw = _nested_get(payload, "config", "request_rate_qps")
        lambda_weight = _nested_get(payload, "config", "lambda_weight")
        if not isinstance(qps_raw, (int, float)) or not isinstance(
            lambda_weight, (int, float)
        ):
            continue

        qps_value = float(qps_raw)
        qps_seen.add(f"{qps_value:g}")
        if request_rate_qps is not None and abs(qps_value - request_rate_qps) > qps_tolerance:
            continue

        lambda_key = _format_lambda(lambda_weight)
        lambda_map.setdefault(lambda_key, {})
        source_map.setdefault(lambda_key, {})

        runs = _nested_get(payload, "router", "runs")
        if not isinstance(runs, list):
            continue

        for run in runs:
            if not isinstance(run, dict):
                continue
            _collect_run_metrics_for_group(
                run=run,
                per_group_metrics=lambda_map[lambda_key],
                per_group_sources=source_map[lambda_key],
                lambda_weight=lambda_weight,
                source_path=json_path,
                requested_utilities=requested_utilities,
                discovered_utilities=discovered_utilities,
                duplicate_policy=duplicate_policy,
                duplicate_scope=f"lambda={lambda_key}",
            )

    utility_order = _ordered_utilities(
        discovered_utilities,
        DEFAULT_UTILITIES,
        requested_utilities=requested_utilities,
    )

    sorted_lambda_keys = sorted(lambda_map.keys(), key=lambda value: float(value))
    normalized_lambda_map: dict[str, dict[str, Any]] = {}
    normalized_source_map: dict[str, dict[str, Any]] = {}
    missing: list[dict[str, str]] = []
    for lambda_key in sorted_lambda_keys:
        normalized_lambda_map[lambda_key] = {}
        normalized_source_map[lambda_key] = {}
        for utility in utility_order:
            metrics = lambda_map[lambda_key].get(utility)
            normalized_lambda_map[lambda_key][utility] = metrics
            source = source_map[lambda_key].get(utility)
            normalized_source_map[lambda_key][utility] = source
            if metrics is None:
                missing.append({"lambda": lambda_key, "utility": utility})

    return {
        "utilities": utility_order,
        "request_rate_qps_filter": request_rate_qps,
        "qps_seen_in_inputs": sorted(qps_seen, key=float),
        "duplicate_policy": duplicate_policy,
        "lambda": normalized_lambda_map,
        "source_files": normalized_source_map,
        "missing": missing,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate router sweep JSONs and produce a lambda-indexed summary JSON."
    )
    parser.add_argument(
        "input_paths",
        nargs="+",
        help="Input directories and/or JSON files to scan.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where summary JSON will be written.",
    )
    parser.add_argument(
        "--summary-name",
        default="router_lambda_sweep_summary.json",
        help="Filename for consolidated JSON output.",
    )
    parser.add_argument(
        "--request-rate-qps",
        type=float,
        default=None,
        help=(
            "Optional QPS filter. If set, only runs matching this request_rate_qps "
            "within --qps-tolerance are included."
        ),
    )
    parser.add_argument(
        "--qps-tolerance",
        type=float,
        default=1e-9,
        help="Tolerance for matching --request-rate-qps.",
    )
    parser.add_argument(
        "--utilities",
        nargs="+",
        default=None,
        help=(
            "Optional utility list to include (and output order). "
            "Example: --utilities hard hard_prefill_tps latency_agnostic"
        ),
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=["error", "first", "last"],
        default="error",
        help=(
            "How to handle duplicate (lambda, utility) entries across inputs: "
            "'error' raises, 'first' keeps first, 'last' keeps last."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(path).expanduser().resolve() for path in args.input_paths]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = aggregate_jsons(
        input_paths=input_paths,
        request_rate_qps=args.request_rate_qps,
        qps_tolerance=args.qps_tolerance,
        requested_utilities=args.utilities,
        duplicate_policy=args.duplicate_policy,
    )

    summary_path = output_dir / args.summary_name
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=False)
        f.write("\n")

    print(f"Wrote summary JSON: {summary_path}")
    print(f"Lambdas: {list(summary['lambda'].keys())}")
    print(f"Utilities: {summary['utilities']}")
    print(f"Missing pairs: {len(summary['missing'])}")


if __name__ == "__main__":
    main()

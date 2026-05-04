#!/usr/bin/env python3
"""Summarize router delta sweep JSONs into a consolidated metrics JSON."""

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
    "soft",
    "soft_prefill_tps",
    "latency_agnostic",
    "round_robin",
    "shortest_queue",
    "instance_affinity",
    "hard",
    "hard_prefill_tps",
]


def _format_delta(delta: float | int | str) -> str:
    if isinstance(delta, str):
        delta = float(delta)
    numeric = float(delta)
    if numeric.is_integer():
        return str(int(numeric))

    formatted = f"{numeric:g}"
    if "e+" in formatted:
        return formatted.replace("e+", "e")
    return formatted


def aggregate_jsons(
    input_dirs: list[Path],
    request_rate_qps: float | None,
    qps_tolerance: float,
    requested_utilities: list[str] | None,
    duplicate_policy: str,
) -> dict[str, Any]:
    delta_map: dict[str, dict[str, Any]] = {}
    source_map: dict[str, dict[str, str]] = {}
    discovered_utilities: set[str] = set()

    for input_dir in input_dirs:
        for json_path in sorted(input_dir.glob("*.json")):
            if json_path.name.endswith("manifest.json"):
                continue

            with json_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)

            delta_raw = _nested_get(payload, "config", "delta_weight")
            qps_raw = _nested_get(payload, "config", "request_rate_qps")
            lambda_weight = _nested_get(payload, "config", "lambda_weight")
            if not isinstance(delta_raw, (int, float)):
                continue
            if not isinstance(qps_raw, (int, float)):
                continue

            qps_value = float(qps_raw)
            if request_rate_qps is not None and abs(qps_value - request_rate_qps) > qps_tolerance:
                continue

            delta_value = float(delta_raw)
            delta_key = _format_delta(delta_value)
            delta_map.setdefault(delta_key, {})
            source_map.setdefault(delta_key, {})

            runs = _nested_get(payload, "router", "runs")
            if not isinstance(runs, list):
                continue

            for run in runs:
                if not isinstance(run, dict):
                    continue
                _collect_run_metrics_for_group(
                    run=run,
                    per_group_metrics=delta_map[delta_key],
                    per_group_sources=source_map[delta_key],
                    lambda_weight=lambda_weight,
                    source_path=json_path,
                    requested_utilities=requested_utilities,
                    discovered_utilities=discovered_utilities,
                    duplicate_policy=duplicate_policy,
                    duplicate_scope=f"delta={delta_key}",
                    ttft_slo_key="ttft_ms_slo_attainment",
                    system_entry_e2e_ttft_slo_key="system_entry_e2e_ttft_ms_slo_attainment",
                )

    utility_order = _ordered_utilities(
        discovered_utilities,
        DEFAULT_UTILITIES,
        requested_utilities=requested_utilities,
    )

    sorted_delta_keys = sorted(delta_map.keys(), key=lambda value: float(value))
    normalized_delta_map: dict[str, dict[str, Any]] = {}
    for delta_key in sorted_delta_keys:
        per_delta = delta_map[delta_key]
        ordered_utilities: dict[str, Any] = {}
        for utility in utility_order:
            metrics = per_delta.get(utility)
            if metrics is not None:
                ordered_utilities[utility] = metrics
        for utility in sorted(per_delta.keys()):
            if utility not in ordered_utilities:
                ordered_utilities[utility] = per_delta[utility]
        normalized_delta_map[delta_key] = {"utilities": ordered_utilities}

    return {"deltas": normalized_delta_map}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate router delta sweep JSONs and produce a consolidated summary JSON."
        )
    )
    parser.add_argument(
        "input_dirs",
        nargs="+",
        help="Input experiment directories containing per-delta router JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where summary JSON will be written.",
    )
    parser.add_argument(
        "--summary-name",
        default="router_delta_sweep_metrics_summary.json",
        help="Filename for consolidated JSON output.",
    )
    parser.add_argument(
        "--request-rate-qps",
        type=float,
        default=8.0,
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
            "Example: --utilities soft latency_agnostic round_robin shortest_queue instance_affinity"
        ),
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=["error", "first", "last"],
        default="last",
        help=(
            "How to handle duplicate (delta, utility) entries across inputs: "
            "'error' raises, 'first' keeps first, 'last' keeps last."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dirs = [Path(path).expanduser().resolve() for path in args.input_dirs]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = aggregate_jsons(
        input_dirs=input_dirs,
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
    print(f"Delta keys: {list(summary['deltas'].keys())}")
    print("Excluded delta values (baked-in): none")


if __name__ == "__main__":
    main()

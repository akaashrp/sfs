#!/usr/bin/env python3
"""Plot system-entry-to-dispatch distribution from router output JSON."""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path
from typing import Any, Dict

from sfs_core.shared.router_io_plot_helpers import _plot_distribution, _quantile, _read_json


def _resolve_router_run(*, router_output_json: Path, utility: str) -> Dict[str, Any]:
    payload = _read_json(router_output_json)
    router = payload.get("router")
    if not isinstance(router, dict):
        raise ValueError("Router JSON is missing top-level 'router' section.")
    runs = router.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Router JSON is missing 'router.runs' list.")

    target_utility = str(utility).strip()
    for run in runs:
        if not isinstance(run, dict):
            continue
        label = str(run.get("label", "")).strip()
        if label == target_utility:
            return run

    available = sorted(
        {
            str(run.get("label")).strip()
            for run in runs
            if isinstance(run, dict) and str(run.get("label")).strip()
        }
    )
    raise ValueError(
        f"Utility '{target_utility}' not found in router.runs labels. "
        f"Available labels: {available}"
    )


def _collect_system_entry_to_dispatch(
    run: Dict[str, Any],
) -> tuple[list[float], list[Dict[str, Any]], Dict[str, int]]:
    counts = {
        "per_request_total": 0,
        "per_request_with_metric": 0,
    }
    values: list[float] = []
    records: list[Dict[str, Any]] = []

    per_request = run.get("per_request")
    if not isinstance(per_request, list):
        return values, records, counts

    for idx, row in enumerate(per_request):
        counts["per_request_total"] += 1
        if not isinstance(row, dict):
            continue

        raw_value = row.get("system_entry_to_dispatch_ms")
        if not isinstance(raw_value, (int, float)):
            continue
        value = float(raw_value)
        if not math.isfinite(value):
            continue

        values.append(value)
        counts["per_request_with_metric"] += 1
        records.append(
            {
                "index": idx,
                "request_id": row.get("request_id"),
                "instance_id": row.get("instance_id"),
                "system_entry_to_dispatch_ms": value,
            }
        )

    return values, records, counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect and plot system-entry-to-dispatch distribution from a "
            "router output JSON file."
        )
    )
    parser.add_argument(
        "--router-output-json",
        type=Path,
        required=True,
        help="Path to router experiment output JSON.",
    )
    parser.add_argument(
        "--utility",
        type=str,
        default="hard",
        help="Utility label in router.runs (default: %(default)s).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=80,
        help="Number of histogram bins (default: %(default)s).",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=None,
        help="Path to output distribution plot PNG.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path to write summary JSON.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom plot title.",
    )

    args = parser.parse_args()
    if args.bins < 1:
        parser.error("--bins must be >= 1.")
    return args


def main() -> None:
    args = parse_args()
    router_output_json = args.router_output_json.expanduser()
    if not router_output_json.exists():
        raise SystemExit(f"Router output JSON does not exist: {router_output_json}")

    utility = str(args.utility).strip()
    run = _resolve_router_run(router_output_json=router_output_json, utility=utility)
    values, records, counts = _collect_system_entry_to_dispatch(run)
    if not values:
        raise SystemExit(
            "No system_entry_to_dispatch_ms values found in router run per-request data."
        )

    title = args.title or (
        f"{router_output_json.stem} [{utility}]: system entry to dispatch distribution"
    )
    default_stem = f"{router_output_json.stem}_{utility}_system_entry_to_dispatch_distribution"
    output_png = (
        args.output_png.expanduser()
        if args.output_png is not None
        else router_output_json.with_name(f"{default_stem}.png")
    )
    output_json = (
        args.output_json.expanduser()
        if args.output_json is not None
        else router_output_json.with_name(f"{default_stem}.json")
    )

    _plot_distribution(
        values=values,
        output_path=output_png,
        title=title,
        bins=int(args.bins),
        x_label="System entry to dispatch (ms)",
    )

    summary = {
        "mode": "router_system_entry_to_dispatch_distribution",
        "router_output_json": str(router_output_json),
        "utility": utility,
        "route_strategy": run.get("route_strategy"),
        "wait_estimator": run.get("wait_estimator"),
        "count": len(values),
        "mean_ms": float(statistics.fmean(values)),
        "min_ms": float(min(values)),
        "p50_ms": _quantile(values, 0.50),
        "p90_ms": _quantile(values, 0.90),
        "p95_ms": _quantile(values, 0.95),
        "p99_ms": _quantile(values, 0.99),
        "max_ms": float(max(values)),
        "system_entry_to_dispatch_values_ms": values,
        "system_entry_to_dispatch_records": records,
        "collection_counts": counts,
        "artifacts": {
            "distribution_plot": str(output_png),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"router_output_json={router_output_json}")
    print(f"utility={utility}")
    print(f"count={summary['count']}")
    print(f"mean_ms={summary['mean_ms']:.6f}")
    print(f"p50_ms={summary['p50_ms']:.6f}")
    print(f"p90_ms={summary['p90_ms']:.6f}")
    print(f"p95_ms={summary['p95_ms']:.6f}")
    print(f"p99_ms={summary['p99_ms']:.6f}")
    print(f"max_ms={summary['max_ms']:.6f}")
    print(f"plot={output_png}")
    print(f"summary={output_json}")


if __name__ == "__main__":
    main()

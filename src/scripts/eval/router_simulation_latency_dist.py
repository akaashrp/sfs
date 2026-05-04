#!/usr/bin/env python3
"""Plot simulation-latency distribution from router predicted-wait logs."""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path
from typing import Any, Dict

from sfs_core.shared.router_io_plot_helpers import (
    _iter_literal_records,
    _plot_distribution,
    _quantile,
    _read_json,
    _resolve_path_from_router_json,
)


def _resolve_predicted_wait_log(
    *,
    router_json_path: Path | None,
    predicted_wait_log: Path | None,
    utility: str,
) -> tuple[Path, str | None]:
    if predicted_wait_log is not None:
        return predicted_wait_log.expanduser(), None
    if router_json_path is None:
        raise ValueError(
            "Provide --router-output-json or --predicted-wait-log."
        )

    router_json = _read_json(router_json_path)
    router = router_json.get("router")
    if not isinstance(router, dict):
        raise ValueError("Router JSON is missing top-level 'router' section.")
    request_log_paths = router.get("request_log_paths")
    if not isinstance(request_log_paths, dict):
        raise ValueError("Router JSON is missing 'router.request_log_paths'.")

    raw_path = request_log_paths.get(utility)
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"Utility '{utility}' missing request log path in router JSON.")

    configured_estimator: str | None = None
    runs = router.get("runs")
    if isinstance(runs, list):
        for run in runs:
            if not isinstance(run, dict):
                continue
            if str(run.get("label")) != utility:
                continue
            value = run.get("wait_estimator")
            if isinstance(value, str) and value:
                configured_estimator = value
                break

    return _resolve_path_from_router_json(raw_path, router_json_path=router_json_path), configured_estimator


def _collect_simulation_latencies(
    path: Path, *, wait_estimator: str | None
) -> tuple[list[float], Dict[str, int]]:
    counts = {
        "records_total": 0,
        "records_matching_estimator": 0,
        "records_with_reports": 0,
        "reports_scanned": 0,
        "reports_with_simulation_latency": 0,
    }
    values: list[float] = []

    normalized_estimator = (
        str(wait_estimator).strip().lower() if wait_estimator is not None else None
    )

    for record in _iter_literal_records(path):
        counts["records_total"] += 1
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue

        estimator_value = payload.get("_wait_estimator")
        if normalized_estimator is not None:
            if str(estimator_value).strip().lower() != normalized_estimator:
                continue

        counts["records_matching_estimator"] += 1
        reports = payload.get("reports")
        if not isinstance(reports, list):
            continue
        counts["records_with_reports"] += 1

        for report in reports:
            counts["reports_scanned"] += 1
            if not isinstance(report, dict):
                continue
            latency_ms = report.get("simulation_latency_ms")
            if not isinstance(latency_ms, (int, float)):
                continue
            latency = float(latency_ms)
            if not math.isfinite(latency):
                continue
            values.append(latency)
            counts["reports_with_simulation_latency"] += 1

    return values, counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect and plot simulation-latency distribution from a router "
            "predicted-wait log."
        )
    )
    parser.add_argument(
        "--router-output-json",
        type=Path,
        default=None,
        help=(
            "Path to router experiment JSON output. Used to resolve request log "
            "from --utility."
        ),
    )
    parser.add_argument(
        "--predicted-wait-log",
        type=Path,
        default=None,
        help="Optional direct path to predicted waits log (overrides router JSON).",
    )
    parser.add_argument(
        "--utility",
        type=str,
        default="hard",
        help="Utility label in router JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--wait-estimator",
        type=str,
        default="live",
        help=(
            "Estimator filter for payload._wait_estimator. "
            "Use empty string to disable filtering."
        ),
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
    if args.router_output_json is None and args.predicted_wait_log is None:
        parser.error("Provide --router-output-json or --predicted-wait-log.")
    return args


def main() -> None:
    args = parse_args()
    router_json_path = (
        args.router_output_json.expanduser() if args.router_output_json is not None else None
    )
    if router_json_path is not None and not router_json_path.exists():
        raise SystemExit(f"Router output JSON does not exist: {router_json_path}")

    wait_estimator_filter = str(args.wait_estimator).strip() or None
    predicted_wait_log, configured_estimator = _resolve_predicted_wait_log(
        router_json_path=router_json_path,
        predicted_wait_log=args.predicted_wait_log,
        utility=str(args.utility).strip(),
    )
    if not predicted_wait_log.exists():
        raise SystemExit(f"Predicted wait log does not exist: {predicted_wait_log}")

    values, counts = _collect_simulation_latencies(
        predicted_wait_log, wait_estimator=wait_estimator_filter
    )
    if not values:
        raise SystemExit(
            "No simulation_latency_ms values found after filtering. "
            "Check --wait-estimator and input log."
        )

    title = (
        args.title
        or (
            f"{predicted_wait_log.stem}: simulation latency distribution"
            + (f" ({wait_estimator_filter})" if wait_estimator_filter else "")
        )
    )
    default_stem = f"{predicted_wait_log.stem}_simulation_latency_distribution"
    output_png = (
        args.output_png.expanduser()
        if args.output_png is not None
        else predicted_wait_log.with_name(f"{default_stem}.png")
    )
    output_json = (
        args.output_json.expanduser()
        if args.output_json is not None
        else predicted_wait_log.with_name(f"{default_stem}.json")
    )

    _plot_distribution(
        values=values,
        output_path=output_png,
        title=title,
        bins=int(args.bins),
        x_label="Simulation latency (ms)",
    )

    summary = {
        "mode": "router_simulation_latency_distribution",
        "router_output_json": str(router_json_path) if router_json_path is not None else None,
        "utility": str(args.utility).strip(),
        "configured_wait_estimator": configured_estimator,
        "wait_estimator_filter": wait_estimator_filter,
        "predicted_wait_log": str(predicted_wait_log),
        "count": len(values),
        "mean_ms": float(statistics.fmean(values)),
        "min_ms": float(min(values)),
        "p50_ms": _quantile(values, 0.50),
        "p90_ms": _quantile(values, 0.90),
        "p95_ms": _quantile(values, 0.95),
        "p99_ms": _quantile(values, 0.99),
        "max_ms": float(max(values)),
        "simulation_latency_values_ms": values,
        "collection_counts": counts,
        "artifacts": {
            "distribution_plot": str(output_png),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"predicted_wait_log={predicted_wait_log}")
    print(f"wait_estimator_filter={wait_estimator_filter}")
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

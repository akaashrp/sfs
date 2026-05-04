#!/usr/bin/env python3
"""Summarize router arrival-sweep JSONs and plot gated utility vs QPS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from sfs_core.shared.router_sweep_summary_helpers import (
    _collect_run_metrics_for_group,
    _nested_get,
)


DEFAULT_UTILITIES = [
    "hard",
    # "hard_prefill_tps",
    "latency_agnostic",
    "shortest_queue",
    "round_robin",
]
PREFERRED_ARRIVAL_ORDER = [
    "poisson",
    "mmpp2_r3",
    "mmpp2_r6",
]


def _format_qps(qps: float | int | str) -> str:
    if isinstance(qps, str):
        return qps
    return f"{float(qps):g}"


def _format_ratio_for_key(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "unknown"
    formatted = f"{float(value):g}"
    return formatted.replace(".", "p")


def _arrival_key(config: dict[str, Any]) -> str:
    arrival_process = config.get("arrival_process")
    if not isinstance(arrival_process, str):
        return "unknown"
    arrival_process = arrival_process.strip().lower()

    if arrival_process == "poisson":
        return "poisson"
    if arrival_process == "mmpp2":
        ratio_key = _format_ratio_for_key(config.get("mmpp2_rate_ratio"))
        return f"mmpp2_r{ratio_key}"
    return arrival_process


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

    seen: set[Path] = set()
    unique: list[Path] = []
    for file_path in files:
        resolved = file_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(file_path)
    return unique


def aggregate_jsons(
    input_paths: list[Path],
    target_utilities: list[str],
) -> dict[str, Any]:
    data_map: dict[str, dict[str, dict[str, Any]]] = {}
    source_map: dict[str, dict[str, dict[str, str]]] = {}
    qps_keys_seen: set[str] = set()
    arrivals_seen: set[str] = set()

    for json_path in _iter_json_files(input_paths):
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        config = payload.get("config", {})
        if not isinstance(config, dict):
            continue
        qps_raw = config.get("request_rate_qps")
        if not isinstance(qps_raw, (int, float)):
            continue
        qps_key = _format_qps(qps_raw)
        arrival_key = _arrival_key(config)

        qps_keys_seen.add(qps_key)
        arrivals_seen.add(arrival_key)
        data_map.setdefault(arrival_key, {})
        data_map[arrival_key].setdefault(qps_key, {})
        source_map.setdefault(arrival_key, {})
        source_map[arrival_key].setdefault(qps_key, {})

        lambda_weight = config.get("lambda_weight")
        runs = _nested_get(payload, "router", "runs")
        if not isinstance(runs, list):
            continue

        for run in runs:
            if not isinstance(run, dict):
                continue
            _collect_run_metrics_for_group(
                run=run,
                per_group_metrics=data_map[arrival_key][qps_key],
                per_group_sources=source_map[arrival_key][qps_key],
                lambda_weight=lambda_weight,
                source_path=json_path,
                requested_utilities=target_utilities,
                discovered_utilities=None,
                duplicate_policy="error",
                duplicate_scope=f"arrival={arrival_key}, qps={qps_key}",
                resolve_source_path=True,
                include_existing_source_on_duplicate=False,
            )

    qps_keys = sorted(qps_keys_seen, key=float)
    arrival_order = [
        arrival for arrival in PREFERRED_ARRIVAL_ORDER if arrival in arrivals_seen
    ] + sorted(arrival for arrival in arrivals_seen if arrival not in PREFERRED_ARRIVAL_ORDER)

    normalized_data: dict[str, dict[str, dict[str, Any]]] = {}
    normalized_source_files: dict[str, dict[str, dict[str, Any]]] = {}
    missing: list[dict[str, str]] = []
    for arrival in arrival_order:
        normalized_data[arrival] = {}
        normalized_source_files[arrival] = {}
        for qps_key in qps_keys:
            normalized_data[arrival][qps_key] = {}
            normalized_source_files[arrival][qps_key] = {}
            for utility in target_utilities:
                metrics = _nested_get(data_map, arrival, qps_key, utility)
                source_file = _nested_get(source_map, arrival, qps_key, utility)
                normalized_data[arrival][qps_key][utility] = metrics
                normalized_source_files[arrival][qps_key][utility] = source_file
                if metrics is None:
                    missing.append(
                        {
                            "arrival_distribution": arrival,
                            "qps": qps_key,
                            "utility": utility,
                        }
                    )

    return {
        "utilities": target_utilities,
        "arrival_distributions": arrival_order,
        "qps_keys": qps_keys,
        "data": normalized_data,
        "source_files": normalized_source_files,
        "missing": missing,
    }


def plot_actual_slo_gated_utility_vs_qps(
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    utility_colors = {
        "hard": "#1f77b4",
        "hard_prefill_tps": "#ff7f0e",
        "latency_agnostic": "#2ca02c",
        "shortest_queue": "#d62728",
        "round_robin": "#9467bd",
    }
    arrival_styles = {
        "poisson": ("-", "o"),
        "mmpp2_r3": ("--", "s"),
        "mmpp2_r6": (":", "^"),
    }

    plt.figure(figsize=(12, 7))
    for utility in summary["utilities"]:
        for arrival in summary["arrival_distributions"]:
            x_values: list[float] = []
            y_values: list[float] = []
            for qps_key in summary["qps_keys"]:
                utility_data = _nested_get(summary, "data", arrival, qps_key, utility)
                if not isinstance(utility_data, dict):
                    continue
                value = utility_data.get("actual_slo_gated_utility_mean")
                if not isinstance(value, (int, float)):
                    continue
                x_values.append(float(qps_key))
                y_values.append(float(value))

            if not x_values:
                continue

            line_style, marker = arrival_styles.get(arrival, ("-.", "D"))
            color = utility_colors.get(utility, "#7f7f7f")
            plt.plot(
                x_values,
                y_values,
                linestyle=line_style,
                marker=marker,
                linewidth=1.9,
                markersize=7.5,
                color=color,
                label=f"{utility} | {arrival}",
            )

    plt.xlabel("QPS")
    plt.ylabel("actual_slo_gated_utility_mean")
    plt.title("Actual SLO-Gated Utility Mean vs QPS (by Utility and Arrival Distribution)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate router arrival-sweep JSONs and plot "
            "actual_slo_gated_utility_mean vs QPS."
        )
    )
    parser.add_argument(
        "input_paths",
        nargs="+",
        help="Input directories and/or JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where summary JSON and plots will be written.",
    )
    parser.add_argument(
        "--summary-name",
        default="router_arrival_sweep_summary.json",
        help="Filename for consolidated JSON output.",
    )
    parser.add_argument(
        "--plot-name",
        default="actual_slo_gated_utility_mean_vs_qps.png",
        help="Filename for plot output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(p) for p in args.input_paths]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = aggregate_jsons(input_paths, DEFAULT_UTILITIES)

    summary_path = output_dir / args.summary_name
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=False)
        f.write("\n")

    plot_path = output_dir / args.plot_name
    plot_actual_slo_gated_utility_vs_qps(summary, plot_path)

    print(f"Wrote summary JSON: {summary_path}")
    print(f"Wrote plot: {plot_path}")
    print(f"Arrival distributions: {summary['arrival_distributions']}")
    print(f"QPS keys: {summary['qps_keys']}")
    print(f"Missing combinations: {len(summary['missing'])}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize router QPS sweep JSONs and generate requested plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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


def _format_qps(qps: float | int | str) -> str:
    if isinstance(qps, str):
        return qps
    return f"{float(qps):g}"


def aggregate_jsons(input_dirs: list[Path]) -> tuple[dict[str, Any], list[str]]:
    qps_map: dict[str, dict[str, Any]] = {}
    discovered_utilities: set[str] = set()

    for input_dir in input_dirs:
        input_dir_resolved = str(input_dir.expanduser().resolve())
        for json_path in sorted(input_dir.glob("*.json")):
            if json_path.name.endswith("manifest.json"):
                continue

            with json_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)

            qps_raw = _nested_get(payload, "config", "request_rate_qps")
            lambda_weight = _nested_get(payload, "config", "lambda_weight")
            if qps_raw is None:
                continue
            qps_key = _format_qps(qps_raw)
            qps_map.setdefault(qps_key, {})

            runs = _nested_get(payload, "router", "runs")
            if not isinstance(runs, list):
                continue

            for run in runs:
                if not isinstance(run, dict):
                    continue
                _collect_run_metrics_for_group(
                    run=run,
                    per_group_metrics=qps_map[qps_key],
                    per_group_sources=None,
                    lambda_weight=lambda_weight,
                    source_path=json_path,
                    requested_utilities=None,
                    discovered_utilities=discovered_utilities,
                    duplicate_policy="error",
                    duplicate_scope=f"qps={qps_key}",
                    include_existing_source_on_duplicate=False,
                )

    utility_order = _ordered_utilities(discovered_utilities, DEFAULT_UTILITIES)

    sorted_qps_keys = sorted(qps_map.keys(), key=lambda x: float(x))
    normalized_qps_map: dict[str, dict[str, Any]] = {}
    for qps_key in sorted_qps_keys:
        normalized_qps_map[qps_key] = {
            utility: qps_map[qps_key].get(utility) for utility in utility_order
        }

    summary = {
        "utilities": utility_order,
        "qps": normalized_qps_map,
    }
    return summary, sorted_qps_keys


def _plot_metric(
    summary: dict[str, Any],
    qps_keys: list[str],
    metric_key: str,
    xlabel: str,
    title: str,
    output_path: Path,
    x_log_scale: bool = False,
) -> None:
    utilities = summary["utilities"]

    plt.figure(figsize=(10, 6))
    for utility in utilities:
        x_values = []
        y_values = []
        for qps_key in qps_keys:
            utility_data = summary["qps"][qps_key].get(utility)
            if not isinstance(utility_data, dict):
                continue
            metric_value = utility_data.get(metric_key)
            if not isinstance(metric_value, (int, float)):
                continue
            if x_log_scale and metric_value <= 0:
                continue
            x_values.append(float(metric_value))
            y_values.append(float(qps_key))
        if not x_values:
            continue
        plt.plot(x_values, y_values, marker="o", linewidth=1.8, label=utility)

    if x_log_scale:
        plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("QPS")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _qps_with_all_utilities_present(
    summary: dict[str, Any], qps_keys: list[str]
) -> list[str]:
    utilities = summary["utilities"]
    filtered: list[str] = []
    for qps_key in qps_keys:
        qps_data = summary["qps"].get(qps_key, {})
        has_all = all(isinstance(qps_data.get(utility), dict) for utility in utilities)
        if has_all:
            filtered.append(qps_key)
    return filtered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate router QPS sweep JSONs and produce requested summary JSON/plots."
        )
    )
    parser.add_argument(
        "input_dirs",
        nargs="+",
        help="Input experiment directories containing per-QPS router JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where summary JSON and plots will be written.",
    )
    parser.add_argument(
        "--summary-name",
        default="router_qps_sweep_summary.json",
        help="Filename for consolidated JSON output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dirs = [Path(p) for p in args.input_dirs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary, qps_keys = aggregate_jsons(input_dirs)

    summary_path = output_dir / args.summary_name
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=False)
        f.write("\n")

    plot_qps_keys = _qps_with_all_utilities_present(summary, qps_keys)

    _plot_metric(
        summary=summary,
        qps_keys=plot_qps_keys,
        metric_key="system_entry_e2e_ttft_ms_slo_attainment_pct",
        xlabel="SLO attainment (%)",
        title="System-Entry E2E TTFT SLO Attainment vs QPS",
        output_path=output_dir / "qps_vs_slo_attainment.png",
    )
    _plot_metric(
        summary=summary,
        qps_keys=plot_qps_keys,
        metric_key="average_system_entry_e2e_ttft_ms",
        xlabel="Average system_entry_e2e_ttft_ms",
        title="Average System-Entry E2E TTFT (ms) vs QPS",
        output_path=output_dir / "qps_vs_avg_system_entry_e2e_ttft_ms.png",
        x_log_scale=True,
    )
    _plot_metric(
        summary=summary,
        qps_keys=plot_qps_keys,
        metric_key="p90_system_entry_e2e_ttft_ms",
        xlabel="P90 system_entry_e2e_ttft_ms",
        title="P90 System-Entry E2E TTFT (ms) vs QPS",
        output_path=output_dir / "qps_vs_p90_system_entry_e2e_ttft_ms.png",
        x_log_scale=True,
    )

    print(f"Wrote summary JSON: {summary_path}")
    print(f"Plot QPS keys (all utilities present): {plot_qps_keys}")
    print(f"Wrote plots to: {output_dir}")


if __name__ == "__main__":
    main()

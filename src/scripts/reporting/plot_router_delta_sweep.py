#!/usr/bin/env python3

"""Generate utility-vs-latency/SLO plots from router delta sweep summary JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

DEFAULT_UTILITY_FIELD = (
    "actual_accuracy_minus_lambda_weight_times_actual_cost"
)
DEFAULT_SWEEP_UTILITIES = ["soft", "soft_prefill_tps"]
DEFAULT_UMAX_UTILITY = "latency_agnostic"
BASELINE_UTILITIES = [
    "latency_agnostic",
    "round_robin",
    "shortest_queue",
    "instance_affinity",
]

PLOT_SPECS = [
    {
        "field": "system_entry_e2e_ttft_ms_slo_attainment",
        "metric_label": "System-entry E2E TTFT SLO attainment (%)",
        "filename": "utility_vs_system_entry_e2e_ttft_ms_slo_attainment.png",
        "log_x": False,
    },
    {
        "field": "ttft_ms_slo_attainment",
        "metric_label": "TTFT SLO attainment (%)",
        "filename": "utility_vs_ttft_ms_slo_attainment.png",
        "log_x": False,
    },
    {
        "field": "average_system_entry_e2e_ttft_ms",
        "metric_label": "Average system-entry E2E TTFT (ms)",
        "filename": "utility_vs_average_system_entry_e2e_ttft_ms.png",
        "log_x": True,
    },
    {
        "field": "p90_system_entry_e2e_ttft_ms",
        "metric_label": "P90 system-entry E2E TTFT (ms)",
        "filename": "utility_vs_p90_system_entry_e2e_ttft_ms.png",
        "log_x": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot utility (actual_accuracy - lambda_weight * actual_cost) "
            "against SLO/latency metrics across delta values."
        )
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("experiments/router_delta_sweep_metrics_summary.json"),
        help="Path to router_delta_sweep_metrics_summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/delta_plots"),
        help="Directory to save plot images.",
    )
    parser.add_argument(
        "--exclude-delta",
        action="append",
        default=[],
        help=(
            "Delta value to exclude from sweep-point plotting. "
            "Pass multiple times to exclude multiple deltas."
        ),
    )
    parser.add_argument(
        "--sweep-utilities",
        nargs="+",
        default=DEFAULT_SWEEP_UTILITIES,
        help=(
            "Utility series to plot across delta values "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--metric-field",
        default=None,
        help=(
            "Optional metric field to plot (for a single figure). "
            "If omitted, all built-in plot specs are generated."
        ),
    )
    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="Do not draw delta=0 baseline utility points.",
    )
    parser.add_argument(
        "--utility-field",
        default=DEFAULT_UTILITY_FIELD,
        help=(
            "Utility field in summary JSON utilities entries "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--utility-label",
        default="Utility = actual_acc - lambda_weight * actual_cost",
        help="Y-axis label for utility (default: %(default)s).",
    )
    parser.add_argument(
        "--u-max-utility",
        default=DEFAULT_UMAX_UTILITY,
        help=(
            "Utility name at delta=0 used for u_max in the extra "
            "u_max - utility vs wait plot (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--filename-prefix",
        default="",
        help="Optional prefix prepended to output filenames.",
    )
    return parser.parse_args()


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float))


def is_excluded_delta(delta_value: float, excluded_deltas: list[float]) -> bool:
    for excluded in excluded_deltas:
        tolerance = max(1e-15, abs(excluded) * 1e-12)
        if math.isclose(delta_value, excluded, rel_tol=0.0, abs_tol=tolerance):
            return True
    return False


def extract_utility_points(
    deltas: dict[str, Any],
    utility_name: str,
    y_field: str,
    excluded_deltas: list[float],
    utility_field: str,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for delta_key, delta_entry in deltas.items():
        delta_value = float(delta_key)
        if is_excluded_delta(delta_value, excluded_deltas):
            continue
        utilities = delta_entry.get("utilities", {})
        utility_metrics = utilities.get(utility_name)
        if not isinstance(utility_metrics, dict):
            continue
        x_val = utility_metrics.get(utility_field)
        y_val = utility_metrics.get(y_field)
        if not (is_number(x_val) and is_number(y_val)):
            continue
        points.append(
            {
                "utility": utility_name,
                "delta": delta_value,
                "delta_label": delta_key,
                "x": float(x_val),
                "y": float(y_val),
            }
        )
    points.sort(key=lambda p: p["delta"])
    return points


def get_delta_zero_utilities(deltas: dict[str, Any]) -> dict[str, Any]:
    for delta_key, delta_entry in deltas.items():
        if math.isclose(float(delta_key), 0.0, abs_tol=1e-15):
            utilities = delta_entry.get("utilities", {})
            if isinstance(utilities, dict):
                return utilities
    return {}


def extract_baseline_points(
    deltas: dict[str, Any], y_field: str, utility_field: str
) -> dict[str, tuple[float, float]]:
    # Tuple format: (utility, metric).
    baselines: dict[str, tuple[float, float]] = {}
    zero_utils = get_delta_zero_utilities(deltas)
    for util_name in BASELINE_UTILITIES:
        metrics = zero_utils.get(util_name)
        if not isinstance(metrics, dict):
            continue
        x_val = metrics.get(utility_field)
        y_val = metrics.get(y_field)
        if not (is_number(x_val) and is_number(y_val)):
            continue
        baselines[util_name] = (float(x_val), float(y_val))
    return baselines


def get_u_max(
    deltas: dict[str, Any], u_max_utility: str, utility_field: str
) -> float:
    zero_utils = get_delta_zero_utilities(deltas)
    utility_metrics = zero_utils.get(u_max_utility)
    if not isinstance(utility_metrics, dict):
        raise SystemExit(
            "Could not compute u_max: "
            f"delta=0 missing utility '{u_max_utility}'."
        )
    value = utility_metrics.get(utility_field)
    if not is_number(value):
        raise SystemExit(
            "Could not compute u_max: "
            f"field '{utility_field}' missing for utility '{u_max_utility}' at delta=0."
        )
    return float(value)


def convert_to_u_max_gap(
    sweep_points_by_utility: dict[str, list[dict[str, Any]]],
    baseline_points: dict[str, tuple[float, float]],
    u_max: float,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, tuple[float, float]]]:
    gap_sweep: dict[str, list[dict[str, Any]]] = {}
    for utility_name, points in sweep_points_by_utility.items():
        gap_sweep[utility_name] = [
            {
                **point,
                "x": u_max - point["x"],
            }
            for point in points
        ]

    gap_baselines = {
        util_name: (u_max - utility_val, metric_val)
        for util_name, (utility_val, metric_val) in baseline_points.items()
    }
    return gap_sweep, gap_baselines


def plot_metric(
    sweep_points_by_utility: dict[str, list[dict[str, Any]]],
    baseline_points: dict[str, tuple[float, float]],
    metric_label: str,
    utility_label: str,
    log_x: bool,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))

    sweep_style = {
        "soft": {"marker": "o", "color": "#1f77b4", "prefix": "S"},
        "soft_prefill_tps": {"marker": "s", "color": "#ff7f0e", "prefix": "SP"},
    }
    label_offsets = [
        (6, 6),
        (6, -12),
        (-22, 6),
        (-22, -12),
        (14, 0),
        (-30, 0),
        (10, 12),
        (10, -18),
        (-26, 12),
        (-26, -18),
    ]
    lookup_sections: list[str] = []

    for utility_name, points in sweep_points_by_utility.items():
        if not points:
            continue
        style = sweep_style.get(
            utility_name,
            {"marker": "o", "color": "#4C72B0", "prefix": utility_name[:2].upper()},
        )
        id_width = max(2, len(str(len(points))))
        for idx, point in enumerate(points, start=1):
            point["id_label"] = f"{style['prefix']}{idx:0{id_width}d}"

        xs = [p["y"] for p in points]
        if utility_name == "soft":
            ys = [p["x"] for p in points]
        else:
            ys = [p["x"] for p in points]
            
        ax.scatter(
            xs,
            ys,
            marker=style["marker"],
            color=style["color"],
            s=74,
            edgecolors="black",
            linewidths=0.5,
            label=f"{utility_name.replace('_', ' ')} (across delta)",
            zorder=3,
        )
        ax.plot(xs, ys, color=style["color"], linewidth=1.2, alpha=0.65, zorder=2)

        for idx, point in enumerate(points):
            x_offset, y_offset = label_offsets[idx % len(label_offsets)]
            ax.annotate(
                point["id_label"],
                (point["y"], point["x"]),
                textcoords="offset points",
                xytext=(x_offset, y_offset),
                fontsize=8,
                weight="bold",
                alpha=0.95,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "fc": "white",
                    "ec": "#666666",
                    "lw": 0.4,
                    "alpha": 0.9,
                },
                arrowprops={
                    "arrowstyle": "-",
                    "lw": 0.5,
                    "color": "#666666",
                    "alpha": 0.6,
                },
            )

        lookup_lines = [f"{utility_name} delta map"] + [
            f"{p['id_label']} -> {p['delta_label']}" for p in points
        ]
        lookup_sections.append("\n".join(lookup_lines))

    if lookup_sections:
        fig.text(
            0.79,
            0.97,
            "\n\n".join(lookup_sections),
            ha="left",
            va="top",
            fontsize=8,
            family="monospace",
            bbox={
                "boxstyle": "round,pad=0.3",
                "fc": "#f7f7f7",
                "ec": "#cccccc",
                "alpha": 0.95,
            },
        )

    baseline_style = {
        "latency_agnostic": ("^", "#2ca02c"),
        "round_robin": ("D", "#9467bd"),
        "shortest_queue": ("P", "#8c564b"),
        "instance_affinity": ("X", "#7f7f7f"),
    }
    for util_name, (utility_val, metric_val) in baseline_points.items():
        marker, color = baseline_style.get(util_name, ("o", "black"))
        ax.scatter(
            [metric_val],
            [utility_val],
            marker=marker,
            color=color,
            s=88,
            edgecolors="black",
            linewidths=0.35,
            label=f"{util_name.replace('_', ' ')} (delta 0)",
            zorder=5,
        )

    ax.set_xlabel(metric_label)
    ax.set_ylabel(utility_label)
    if log_x:
        ax.set_xscale("log")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout(rect=(0, 0, 0.77, 1))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.summary_json.exists():
        raise SystemExit(f"Summary JSON not found: {args.summary_json}")

    with args.summary_json.open("r", encoding="utf-8") as src:
        payload = json.load(src)

    deltas = payload.get("deltas", {})
    if not isinstance(deltas, dict) or not deltas:
        raise SystemExit("No delta entries found in summary JSON.")

    excluded_deltas: list[float] = []
    for raw in args.exclude_delta:
        try:
            excluded_deltas.append(float(raw))
        except ValueError:
            raise SystemExit(f"Invalid --exclude-delta value: {raw}")

    selected_specs = PLOT_SPECS
    if args.metric_field is not None:
        selected_specs = [spec for spec in PLOT_SPECS if spec["field"] == args.metric_field]
        if not selected_specs:
            valid = ", ".join(spec["field"] for spec in PLOT_SPECS)
            raise SystemExit(
                f"Unknown --metric-field value: {args.metric_field}. Valid: {valid}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    u_max = get_u_max(deltas, args.u_max_utility, args.utility_field)
    for spec in selected_specs:
        sweep_points_by_utility = {
            utility_name: extract_utility_points(
                deltas,
                utility_name,
                spec["field"],
                excluded_deltas,
                args.utility_field,
            )
            for utility_name in args.sweep_utilities
        }
        baseline_points = (
            {}
            if args.no_baselines
            else extract_baseline_points(deltas, spec["field"], args.utility_field)
        )
        output_path = args.output_dir / f"{args.filename_prefix}{spec['filename']}"
        plot_metric(
            sweep_points_by_utility=sweep_points_by_utility,
            baseline_points=baseline_points,
            metric_label=spec["metric_label"],
            utility_label=args.utility_label,
            log_x=spec["log_x"],
            output_path=output_path,
        )
        print(f"Saved {output_path}")

        if spec["field"] == "average_system_entry_e2e_ttft_ms":
            gap_sweep, gap_baselines = convert_to_u_max_gap(
                sweep_points_by_utility=sweep_points_by_utility,
                baseline_points=baseline_points,
                u_max=u_max,
            )
            gap_output = (
                args.output_dir
                / f"{args.filename_prefix}u_max_minus_utility_vs_average_system_entry_e2e_ttft_ms.png"
            )
            plot_metric(
                sweep_points_by_utility=gap_sweep,
                baseline_points=gap_baselines,
                metric_label=spec["metric_label"],
                utility_label=(
                    f"u_max - utility (u_max from delta=0 {args.u_max_utility})"
                ),
                log_x=spec["log_x"],
                output_path=gap_output,
            )
            print(f"Saved {gap_output}")


if __name__ == "__main__":
    main()

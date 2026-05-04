#!/usr/bin/env python3

"""Plot utility metrics across lambda values from router lambda sweep summary."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


DEFAULT_UTILITIES = ["hard", "hard_prefill_tps"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot actual_slo_gated_utility_mean for selected utilities "
            "against lambda on a log-scaled x-axis."
        )
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("experiments/lambda_sweep/router_qps8_lambda_sweep_summary.json"),
        help="Path to lambda sweep summary JSON.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(
            "experiments/lambda_sweep/actual_slo_gated_utility_mean_vs_lambda.png"
        ),
        help="Path to output PNG.",
    )
    parser.add_argument(
        "--utilities",
        nargs="+",
        default=DEFAULT_UTILITIES,
        help="Utilities to include in the plot.",
    )
    parser.add_argument(
        "--exclude-lambda",
        action="append",
        default=[],
        help=(
            "Lambda value to exclude from plotting. "
            "Pass multiple times to exclude multiple lambdas."
        ),
    )
    return parser.parse_args()


def is_excluded_lambda(lambda_value: float, excluded_lambdas: list[float]) -> bool:
    for excluded in excluded_lambdas:
        tolerance = max(1e-15, abs(excluded) * 1e-12)
        if math.isclose(lambda_value, excluded, rel_tol=0.0, abs_tol=tolerance):
            return True
    return False


def load_points(
    payload: dict[str, Any],
    utility: str,
    y_field: str,
    excluded_lambdas: list[float],
) -> list[tuple[float, float]]:
    by_lambda = payload.get("by_lambda", {})
    if not isinstance(by_lambda, dict):
        return []

    points: list[tuple[float, float]] = []
    for lambda_key, util_map in by_lambda.items():
        if not isinstance(util_map, dict):
            continue
        metrics = util_map.get(utility)
        if not isinstance(metrics, dict):
            continue
        y_val = metrics.get(y_field)
        try:
            x_val = float(lambda_key)
        except (TypeError, ValueError):
            continue
        if x_val <= 0 or not isinstance(y_val, (int, float)):
            continue
        if is_excluded_lambda(x_val, excluded_lambdas):
            continue
        points.append((x_val, float(y_val)))

    points.sort(key=lambda item: item[0])
    return points


def choose_y_step(span: float) -> float:
    if span <= 0:
        return 0.01
    target_step = span / 10.0
    steps = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    for step in steps:
        if step >= target_step:
            return step
    return 0.2


def main() -> None:
    args = parse_args()
    if not args.summary_json.exists():
        raise SystemExit(f"Summary JSON not found: {args.summary_json}")

    with args.summary_json.open("r", encoding="utf-8") as src:
        payload = json.load(src)

    y_field = "actual_slo_gated_utility_mean"
    fig, ax = plt.subplots(figsize=(10, 6))

    marker_map = {
        "hard": "o",
        "hard_prefill_tps": "s",
        "latency_agnostic": "^",
    }

    excluded_lambdas: list[float] = []
    for raw in args.exclude_lambda:
        try:
            excluded_lambdas.append(float(raw))
        except ValueError:
            raise SystemExit(f"Invalid --exclude-lambda value: {raw}")

    all_y_values: list[float] = []
    any_points = False
    for utility in args.utilities:
        points = load_points(payload, utility, y_field, excluded_lambdas)
        if not points:
            continue
        any_points = True
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        all_y_values.extend(ys)
        ax.plot(
            xs,
            ys,
            marker=marker_map.get(utility, "o"),
            linewidth=1.5,
            markersize=5,
            label=utility.replace("_", " "),
        )

    if not any_points:
        raise SystemExit("No valid points found for requested utilities.")

    y_min = min(all_y_values)
    y_max = max(all_y_values)
    span = y_max - y_min
    pad = max(0.01, span * 0.15)
    ax.set_ylim(y_min - pad, y_max + pad)
    y_step = choose_y_step((y_max + pad) - (y_min - pad))
    ax.yaxis.set_major_locator(MultipleLocator(y_step))

    ax.set_xscale("log")
    ax.set_xlabel("Lambda")
    ax.set_ylabel("Actual SLO-gated utility mean")
    ax.set_title("Actual SLO-gated utility vs lambda (QPS 8)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc="best")
    fig.tight_layout()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, dpi=220)
    plt.close(fig)
    print(f"Saved {args.output_path}")


if __name__ == "__main__":
    main()

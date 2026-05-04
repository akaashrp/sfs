import csv
import math
from pathlib import Path
from typing import Any, Dict, Optional

from sfs_core.shared.shared_experiment_helpers import _compute_percentile
from sfs_core.regression.two_part_fit import (
    BATCH_FIT_REQUIRED_COLS as _BATCH_FIT_REQUIRED_COLS,
    BATCH_FIT_OPTIONAL_COLS as _BATCH_FIT_OPTIONAL_COLS,
    build_feature_matrix as _build_batch_fit_matrix,
)


def _pearson_correlation(xs: list[float], ys: list[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var <= 0 or y_var <= 0:
        return None
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    return float(cov / math.sqrt(x_var * y_var))


def build_tradeoff_summary(per_request: list[Dict[str, Any]]) -> Dict[str, Any]:
    pairs = [
        (float(item["latency_ms"]), float(item["predicted_accuracy"]))
        for item in per_request
        if item.get("error") is None
        and item.get("latency_ms") is not None
        and item.get("predicted_accuracy") is not None
    ]
    if not pairs:
        return {
            "num_points": 0,
            "mean_predicted_accuracy": None,
            "latency_accuracy_pearson_r": None,
            "quantile_curve": [],
        }

    latencies = [lat for lat, _ in pairs]
    accuracies = [acc for _, acc in pairs]
    quantile_curve: list[Dict[str, float | None]] = []
    for quantile in (0.5, 0.75, 0.9, 0.95):
        latency_cap = _compute_percentile(latencies, quantile)
        if latency_cap is None:
            continue
        filtered = [acc for lat, acc in pairs if lat <= latency_cap]
        quantile_curve.append(
            {
                "latency_quantile": quantile,
                "latency_cap_ms": latency_cap,
                "mean_predicted_accuracy": (
                    float(sum(filtered) / len(filtered)) if filtered else None
                ),
            }
        )

    return {
        "num_points": len(pairs),
        "mean_predicted_accuracy": float(sum(accuracies) / len(accuracies)),
        "latency_accuracy_pearson_r": _pearson_correlation(latencies, accuracies),
        "quantile_curve": quantile_curve,
    }

def _r2_score(actual: list[float], predicted: list[float]) -> Optional[float]:
    if len(actual) != len(predicted) or len(actual) < 2:
        return None
    mean_actual = sum(actual) / len(actual)
    ss_tot = sum((value - mean_actual) ** 2 for value in actual)
    if ss_tot <= 0:
        return None
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    return float(1.0 - (ss_res / ss_tot))


def _fit_error_metrics(actual: list[float], predicted: list[float]) -> Dict[str, Optional[float] | int]:
    if not actual or len(actual) != len(predicted):
        return {
            "count": 0,
            "mae": None,
            "rmse": None,
            "mean_error": None,
            "mape_pct": None,
            "r2": None,
            "pearson_r": None,
        }
    abs_errors = [abs(p - a) for a, p in zip(actual, predicted)]
    sq_errors = [(p - a) ** 2 for a, p in zip(actual, predicted)]
    signed_errors = [p - a for a, p in zip(actual, predicted)]
    nonzero_pairs = [(a, p) for a, p in zip(actual, predicted) if a != 0]
    mape = (
        100.0
        * sum(abs((p - a) / a) for a, p in nonzero_pairs)
        / len(nonzero_pairs)
        if nonzero_pairs
        else None
    )
    return {
        "count": len(actual),
        "mae": float(sum(abs_errors) / len(abs_errors)),
        "rmse": float(math.sqrt(sum(sq_errors) / len(sq_errors))),
        "mean_error": float(sum(signed_errors) / len(signed_errors)),
        "mape_pct": float(mape) if mape is not None else None,
        "r2": _r2_score(actual, predicted),
        "pearson_r": _pearson_correlation(actual, predicted),
    }


def _add_inlier_r2(
    *,
    metrics: Dict[str, Optional[float] | int],
    actual: list[float],
    predicted: list[float],
    inlier_mask: list[bool],
) -> Dict[str, Optional[float] | int]:
    enriched = dict(metrics)
    if len(actual) != len(predicted) or len(actual) != len(inlier_mask):
        enriched["inlier_count"] = 0
        enriched["inlier_fraction"] = None
        enriched["r2_inliers"] = None
        return enriched

    actual_inliers = [a for a, is_inlier in zip(actual, inlier_mask) if is_inlier]
    predicted_inliers = [p for p, is_inlier in zip(predicted, inlier_mask) if is_inlier]
    inlier_count = len(actual_inliers)
    enriched["inlier_count"] = inlier_count
    enriched["inlier_fraction"] = (
        float(inlier_count / len(inlier_mask)) if inlier_mask else None
    )
    enriched["r2_inliers"] = _r2_score(actual_inliers, predicted_inliers)
    return enriched


def _plot_batch_fit_3d(
    *,
    prefill: list[float],
    decode: list[float],
    actual_exec: list[float],
    estimated_exec: list[float],
    title: str,
    output_path: Path,
    max_points: int,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    total = len(actual_exec)
    if total > max_points:
        rng = np.random.default_rng(69)
        idx = rng.choice(total, size=max_points, replace=False)
        prefill_vals = np.asarray(prefill)[idx]
        decode_vals = np.asarray(decode)[idx]
        actual_vals = np.asarray(actual_exec)[idx]
        estimated_vals = np.asarray(estimated_exec)[idx]
    else:
        prefill_vals = prefill
        decode_vals = decode
        actual_vals = actual_exec
        estimated_vals = estimated_exec

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        prefill_vals,
        decode_vals,
        actual_vals,
        marker="o",
        alpha=0.4,
        label="Actual",
    )
    ax.scatter(
        prefill_vals,
        decode_vals,
        estimated_vals,
        marker="x",
        alpha=0.4,
        label="Estimated",
    )
    ax.set_xlabel("Prefill tokens")
    ax.set_ylabel("Decode tokens")
    ax.set_zlabel("Batch exec time")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    return str(output_path)


def _plot_batch_fit_parity(
    *,
    actual_exec: list[float],
    estimated_exec: list[float],
    title: str,
    output_path: Path,
    max_points: int,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    total = len(actual_exec)
    if total > max_points:
        rng = np.random.default_rng(69)
        idx = rng.choice(total, size=max_points, replace=False)
        actual_vals = np.asarray(actual_exec)[idx]
        estimated_vals = np.asarray(estimated_exec)[idx]
    else:
        actual_vals = actual_exec
        estimated_vals = estimated_exec

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(actual_vals, estimated_vals, s=10, alpha=0.6)
    max_val = max(max(actual_vals), max(estimated_vals))
    plt.plot([0.0, max_val], [0.0, max_val], linestyle="--", linewidth=1.0)
    plt.xlabel("Actual exec time")
    plt.ylabel("Estimated exec time")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return str(output_path)


def _read_batch_fit_df(csv_path: Path):
    import pandas as pd
    return pd.read_csv(csv_path, header=0)


def _resolve_batch_fit_paths(paths: list[Path]) -> list[Path]:
    resolved: list[Path] = []
    for path in paths:
        expanded = path.expanduser()
        if not expanded.exists():
            raise ValueError(f"Batch-fit CSV does not exist: {expanded}")
        resolved.append(expanded)
    return resolved


def _write_batch_fit_predictions(
    *,
    prefill: list[float],
    decode: list[float],
    actual_exec: list[float],
    typical_exec: list[float],
    expected_exec: list[float],
    output_path: Path,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "prefill",
                "decode",
                "actual_exec",
                "predicted_typical_exec",
                "predicted_expected_exec",
            ]
        )
        for row in zip(prefill, decode, actual_exec, typical_exec, expected_exec):
            writer.writerow(row)
    return str(output_path)


def _evaluate_batch_fit_df(
    *,
    fit_result: Any,
    df: Any,
    output_dir: Path,
    output_prefix: str,
    title_prefix: str,
    max_plot_points: int,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    feature_set = getattr(fit_result, "feature_set", None)
    x, _ = _build_batch_fit_matrix(df, feature_set=feature_set)
    actual_exec = [float(v) for v in df["exec"].to_list()]
    typical_exec = [float(v) for v in fit_result.predict_typical(x)]
    expected_exec = [float(v) for v in fit_result.predict_expected(x)]
    prefill = [float(v) for v in df["prefill"].to_list()]
    decode = [float(v) for v in df["decode"].to_list()]
    robust_exec = [float(v) for v in fit_result.robust_model.predict(x)]
    threshold = float(fit_result.residual_threshold)
    inlier_mask = [
        float(actual - robust) <= threshold
        for actual, robust in zip(actual_exec, robust_exec)
    ]

    typical_metrics = _add_inlier_r2(
        metrics=_fit_error_metrics(actual_exec, typical_exec),
        actual=actual_exec,
        predicted=typical_exec,
        inlier_mask=inlier_mask,
    )
    expected_metrics = _add_inlier_r2(
        metrics=_fit_error_metrics(actual_exec, expected_exec),
        actual=actual_exec,
        predicted=expected_exec,
        inlier_mask=inlier_mask,
    )

    typical_surface_path = _plot_batch_fit_3d(
        prefill=prefill,
        decode=decode,
        actual_exec=actual_exec,
        estimated_exec=typical_exec,
        title=f"{title_prefix}: actual vs typical estimate",
        output_path=output_dir / f"{output_prefix}_actual_vs_typical_3d.png",
        max_points=max_plot_points,
    )
    expected_surface_path = _plot_batch_fit_3d(
        prefill=prefill,
        decode=decode,
        actual_exec=actual_exec,
        estimated_exec=expected_exec,
        title=f"{title_prefix}: actual vs expected estimate",
        output_path=output_dir / f"{output_prefix}_actual_vs_expected_3d.png",
        max_points=max_plot_points,
    )
    expected_parity_path = _plot_batch_fit_parity(
        actual_exec=actual_exec,
        estimated_exec=expected_exec,
        title=f"{title_prefix}: expected estimate parity",
        output_path=output_dir / f"{output_prefix}_expected_parity.png",
        max_points=max_plot_points,
    )
    typical_parity_path = _plot_batch_fit_parity(
        actual_exec=actual_exec,
        estimated_exec=typical_exec,
        title=f"{title_prefix}: typical estimate parity",
        output_path=output_dir / f"{output_prefix}_typical_parity.png",
        max_points=max_plot_points,
    )
    predictions_path = _write_batch_fit_predictions(
        prefill=prefill,
        decode=decode,
        actual_exec=actual_exec,
        typical_exec=typical_exec,
        expected_exec=expected_exec,
        output_path=output_dir / f"{output_prefix}_predictions.csv",
    )

    return (
        {
            "goodness_of_fit": {
                "typical": typical_metrics,
                "expected": expected_metrics,
            },
            "artifacts": {
                "actual_vs_typical_3d": typical_surface_path,
                "actual_vs_expected_3d": expected_surface_path,
                "typical_parity": typical_parity_path,
                "expected_parity": expected_parity_path,
                "predictions_csv": predictions_path,
            },
        },
        {
            "prefill": prefill,
            "decode": decode,
            "actual_exec": actual_exec,
            "typical_exec": typical_exec,
            "expected_exec": expected_exec,
            "inlier_mask": inlier_mask,
        },
    )


def _plot_wait_fit(
    *,
    predicted: list[float],
    actual: list[float],
    output_path: Path,
    title: str,
) -> Optional[str]:
    if not predicted or len(predicted) != len(actual):
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.scatter(predicted, actual, s=10, alpha=0.6)
    max_val = max(max(predicted), max(actual))
    plt.plot([0.0, max_val], [0.0, max_val], linestyle="--", linewidth=1.0)
    plt.xlabel("Predicted wait (ms)")
    plt.ylabel("Actual wait (ms)")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return str(output_path)

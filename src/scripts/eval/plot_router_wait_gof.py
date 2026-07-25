#!/usr/bin/env python3
"""Evaluate routed wait-estimator GoF using router logs and per-model wait logs."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_THIS_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _THIS_DIR.parent.parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.append(str(_WORKSPACE_ROOT))

from sfs_core.eval.fit_utils import _fit_error_metrics
from sfs_core.shared.router_io_plot_helpers import (
    _iter_literal_records,
    _read_json,
    _resolve_path_from_router_json,
)

QUEUE_PATTERN = re.compile(r"queue_ms=([0-9.+-eE]+)")
PREFILL_PATTERN = re.compile(r"prefill_s=([0-9.+-eE]+)")
REQUEST_ID_PATTERN = re.compile(r"request_id=([^\s]+)")

DEFAULT_UTILITIES = ("hard", "slo_aware")
TTFT_TARGET_ESTIMATORS = {"live", "prefill_tps_ttft", "pk_mg1"}


def _extract_prefill_backlog_tokens(payload: Dict[str, Any]) -> Optional[float]:
    reports = payload.get("reports")
    if isinstance(reports, list) and reports:
        report0 = reports[0]
        if isinstance(report0, dict):
            metadata = report0.get("metadata")
            if isinstance(metadata, dict):
                value = metadata.get("prefill_backlog_total_tokens")
                if isinstance(value, (int, float)):
                    return float(value)
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get("prefill_backlog_total_tokens")
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _read_predicted_wait_records(path: Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    for record in _iter_literal_records(path):
        request_id = record.get("request_id")
        wait_time_ms = record.get("wait_time_ms")
        instance_id = record.get("instance_id")
        payload = record.get("payload")
        if not isinstance(request_id, str) or not request_id:
            continue
        if not isinstance(wait_time_ms, (int, float)):
            continue
        if not isinstance(instance_id, str) or not instance_id:
            continue
        prefill_backlog_tokens: Optional[float] = None
        wait_estimator: Optional[str] = None
        if isinstance(payload, dict):
            prefill_backlog_tokens = _extract_prefill_backlog_tokens(payload)
            estimator_value = payload.get("_wait_estimator")
            if isinstance(estimator_value, str) and estimator_value:
                wait_estimator = estimator_value
        records[request_id] = {
            "request_id": request_id,
            "instance_id": instance_id,
            "predicted_wait_ms": float(wait_time_ms),
            "prefill_backlog_total_tokens": prefill_backlog_tokens,
            "wait_estimator": wait_estimator,
        }
    return records

def _read_response_map(path: Path) -> Dict[str, Dict[str, str]]:
    mappings: Dict[str, Dict[str, str]] = {}
    for record in _iter_literal_records(path):
        request_id = record.get("request_id")
        response_id = record.get("response_id")
        instance_id = record.get("instance_id")
        if not isinstance(request_id, str) or not request_id:
            continue
        if not isinstance(response_id, str) or not response_id:
            continue
        mappings[request_id] = {
            "request_id": request_id,
            "response_id": response_id,
            "instance_id": str(instance_id) if instance_id is not None else "",
        }
    return mappings


def _read_actual_wait_logs(
    paths: list[Path],
) -> tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    queue_values: Dict[str, float] = {}
    ttft_values: Dict[str, float] = {}
    prefill_values: Dict[str, float] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as src:
            for line in src:
                queue_match = QUEUE_PATTERN.search(line)
                request_match = REQUEST_ID_PATTERN.search(line)
                if not queue_match or not request_match:
                    continue
                response_id = request_match.group(1)
                try:
                    queue_ms = float(queue_match.group(1))
                except ValueError:
                    continue
                queue_values[response_id] = queue_ms
                prefill_match = PREFILL_PATTERN.search(line)
                if prefill_match is not None:
                    try:
                        prefill_ms = float(prefill_match.group(1)) * 1000.0
                    except ValueError:
                        continue
                    prefill_values[response_id] = prefill_ms
                    ttft_values[response_id] = float(queue_ms + prefill_ms)
    return queue_values, ttft_values, prefill_values


def _resolve_actual_metric_name(wait_estimator: Optional[str]) -> str:
    normalized = str(wait_estimator or "").strip().lower()
    if normalized in TTFT_TARGET_ESTIMATORS:
        return "ttft_ms"
    return "queue_ms"


def _resolve_utility_wait_estimator(
    *,
    router_json: Dict[str, Any],
    utility: str,
) -> Optional[str]:
    router = router_json.get("router")
    if not isinstance(router, dict):
        return None
    runs = router.get("runs")
    if not isinstance(runs, list):
        return None
    for run in runs:
        if not isinstance(run, dict):
            continue
        label = run.get("label")
        if str(label) != utility:
            continue
        wait_estimator = run.get("wait_estimator")
        if isinstance(wait_estimator, str) and wait_estimator:
            return wait_estimator
    return None

def _sample_points(
    predicted: list[float],
    actual: list[float],
    *,
    max_points: int,
) -> tuple[list[float], list[float]]:
    if len(predicted) <= max_points:
        return predicted, actual
    rng = random.Random(69)
    idxs = sorted(rng.sample(range(len(predicted)), k=max_points))
    return [predicted[i] for i in idxs], [actual[i] for i in idxs]


def _plot_parity(
    *,
    predicted: list[float],
    actual: list[float],
    output_path: Path,
    title: str,
    max_points: int,
    actual_label: str,
) -> Optional[str]:
    if not predicted or len(predicted) != len(actual):
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs, ys = _sample_points(predicted, actual, max_points=max_points)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.scatter(xs, ys, s=10, alpha=0.6)
    max_val = max(max(xs), max(ys))
    plt.plot([0.0, max_val], [0.0, max_val], linestyle="--", linewidth=1.0)
    plt.xlabel("Predicted wait (ms)")
    plt.ylabel(actual_label)
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return str(output_path)


def _plot_residual_histogram(
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

    residuals = [p - a for a, p in zip(actual, predicted)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=60, alpha=0.8)
    plt.xlabel("Residual (predicted - actual) ms")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return str(output_path)


def _write_pairs_csv(
    *,
    rows: list[Dict[str, Any]],
    estimate_field: str,
    actual_field: str,
    actual_metric_name: str,
    output_path: Path,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "request_id",
                "response_id",
                "instance_id",
                "actual_metric_name",
                "actual_metric_ms",
                "actual_queue_ms",
                "actual_ttft_ms",
                "actual_prefill_ms",
                "predicted_wait_ms",
                "live_wait_estimate_ms",
                "prefill_tps_estimate_ms",
                "prefill_backlog_total_tokens",
                "prefill_tokens_for_tps",
                "prompt_tokens",
                "prefill_tps",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.get("request_id"),
                    row.get("response_id"),
                    row.get("instance_id"),
                    actual_metric_name,
                    row.get(actual_field),
                    row.get("actual_queue_ms"),
                    row.get("actual_ttft_ms"),
                    row.get("actual_prefill_ms"),
                    row.get(estimate_field),
                    row.get("live_wait_estimate_ms"),
                    row.get("prefill_tps_estimate_ms"),
                    row.get("prefill_backlog_total_tokens"),
                    row.get("prefill_tokens_for_tps"),
                    row.get("prompt_tokens"),
                    row.get("prefill_tps"),
                ]
            )
    return str(output_path)

def _evaluate_estimator(
    *,
    rows: list[Dict[str, Any]],
    estimate_field: str,
    actual_field: str,
    actual_metric_name: str,
    burst_threshold_ms: float,
    min_matched_pairs: int,
    max_plot_points: int,
    output_prefix: str,
    output_dir: Path,
) -> Dict[str, Any]:
    usable_rows = [
        row
        for row in rows
        if isinstance(row.get(estimate_field), (int, float))
        and isinstance(row.get(actual_field), (int, float))
    ]
    if len(usable_rows) < min_matched_pairs:
        raise RuntimeError(
            f"Estimator '{estimate_field}' matched_pairs={len(usable_rows)} "
            f"is below --min-matched-pairs={min_matched_pairs}."
        )

    predicted = [float(row[estimate_field]) for row in usable_rows]
    actual = [float(row[actual_field]) for row in usable_rows]
    burst_rows = [row for row in usable_rows if float(row[actual_field]) > burst_threshold_ms]
    burst_predicted = [float(row[estimate_field]) for row in burst_rows]
    burst_actual = [float(row[actual_field]) for row in burst_rows]

    per_instance: Dict[str, Dict[str, Any]] = {}
    per_instance_rows: Dict[str, list[Dict[str, Any]]] = {}
    instance_ids = sorted({str(row["instance_id"]) for row in usable_rows})
    for instance_id in instance_ids:
        local_rows = [row for row in usable_rows if row.get("instance_id") == instance_id]
        per_instance_rows[instance_id] = local_rows
        local_pred = [float(row[estimate_field]) for row in local_rows]
        local_actual = [float(row[actual_field]) for row in local_rows]
        per_instance[instance_id] = {
            "count": len(local_rows),
            "metrics_ms": _fit_error_metrics(local_actual, local_pred),
        }

    parity_paths_by_instance: Dict[str, Optional[str]] = {}
    burst_parity_paths_by_instance: Dict[str, Optional[str]] = {}
    actual_axis_label = f"Actual {actual_metric_name} (ms)"
    for instance_id, local_rows in per_instance_rows.items():
        safe_instance_id = re.sub(r"[^A-Za-z0-9._-]+", "_", instance_id)
        local_pred = [float(row[estimate_field]) for row in local_rows]
        local_actual = [float(row[actual_field]) for row in local_rows]
        parity_paths_by_instance[instance_id] = _plot_parity(
            predicted=local_pred,
            actual=local_actual,
            output_path=output_dir / f"{output_prefix}_{estimate_field}_{actual_metric_name}_{safe_instance_id}_parity.png",
            title=f"{output_prefix} {estimate_field} {instance_id}: parity",
            max_points=max_plot_points,
            actual_label=actual_axis_label,
        )
        local_burst_rows = [
            row for row in local_rows if float(row[actual_field]) > burst_threshold_ms
        ]
        local_burst_pred = [float(row[estimate_field]) for row in local_burst_rows]
        local_burst_actual = [float(row[actual_field]) for row in local_burst_rows]
        burst_parity_paths_by_instance[instance_id] = _plot_parity(
            predicted=local_burst_pred,
            actual=local_burst_actual,
            output_path=output_dir / f"{output_prefix}_{estimate_field}_{actual_metric_name}_{safe_instance_id}_parity_burst.png",
            title=f"{output_prefix} {estimate_field} {instance_id}: burst parity",
            max_points=max_plot_points,
            actual_label=actual_axis_label,
        )

    residual_hist_path = _plot_residual_histogram(
        predicted=predicted,
        actual=actual,
        output_path=output_dir / f"{output_prefix}_{estimate_field}_{actual_metric_name}_residual_hist.png",
        title=f"{output_prefix} {estimate_field}: residual histogram",
    )
    pairs_csv_path = _write_pairs_csv(
        rows=usable_rows,
        estimate_field=estimate_field,
        actual_field=actual_field,
        actual_metric_name=actual_metric_name,
        output_path=output_dir / f"{output_prefix}_{estimate_field}_{actual_metric_name}_matched_pairs.csv",
    )

    return {
        "actual_metric_name": actual_metric_name,
        "matched_pairs": len(usable_rows),
        "overall_metrics_ms": _fit_error_metrics(actual, predicted),
        "burst_threshold_ms": burst_threshold_ms,
        "burst_matched_pairs": len(burst_rows),
        "burst_metrics_ms": _fit_error_metrics(burst_actual, burst_predicted),
        "per_instance_metrics": per_instance,
        "artifacts": {
            "parity_plots_by_instance": parity_paths_by_instance,
            "burst_parity_plots_by_instance": burst_parity_paths_by_instance,
            "residual_histogram": residual_hist_path,
            "matched_pairs_csv": pairs_csv_path,
        },
    }

def _resolve_utility_paths(
    *,
    router_json: Dict[str, Any],
    router_json_path: Path,
    utility: str,
) -> tuple[Path, Path]:
    router = router_json.get("router")
    if not isinstance(router, dict):
        raise ValueError("Router JSON is missing top-level 'router' section.")
    request_log_paths = router.get("request_log_paths")
    response_map_paths = router.get("response_map_paths")
    if not isinstance(request_log_paths, dict) or not isinstance(response_map_paths, dict):
        raise ValueError(
            "Router JSON is missing 'router.request_log_paths' or 'router.response_map_paths'."
        )
    raw_request_path = request_log_paths.get(utility)
    raw_mapping_path = response_map_paths.get(utility)
    if not isinstance(raw_request_path, str) or not raw_request_path:
        raise ValueError(f"Utility '{utility}' missing request log path in router JSON.")
    if not isinstance(raw_mapping_path, str) or not raw_mapping_path:
        raise ValueError(f"Utility '{utility}' missing response map path in router JSON.")

    request_path = _resolve_path_from_router_json(
        raw_request_path,
        router_json_path=router_json_path,
    )
    mapping_path = _resolve_path_from_router_json(
        raw_mapping_path,
        router_json_path=router_json_path,
    )
    if not request_path.exists():
        raise FileNotFoundError(
            f"Predicted-wait log does not exist: {request_path}. "
            f"Also checked router-json directory fallback under {router_json_path.parent}."
        )
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Response-map log does not exist: {mapping_path}. "
            f"Also checked router-json directory fallback under {router_json_path.parent}."
        )
    return request_path, mapping_path


def _analyze_utility(
    *,
    utility: str,
    utility_wait_estimator: Optional[str],
    predicted_records: Dict[str, Dict[str, Any]],
    response_map: Dict[str, Dict[str, str]],
    actual_queue_map: Dict[str, float],
    actual_ttft_map: Dict[str, float],
    actual_prefill_map: Dict[str, float],
    burst_threshold_ms: float,
    min_matched_pairs: int,
    max_plot_points: int,
    output_dir: Path,
    output_prefix: str,
) -> Dict[str, Any]:
    inferred_estimator = utility_wait_estimator
    if not inferred_estimator:
        for pred in predicted_records.values():
            candidate = pred.get("wait_estimator")
            if isinstance(candidate, str) and candidate:
                inferred_estimator = candidate
                break
    actual_metric_name = _resolve_actual_metric_name(inferred_estimator)
    actual_field = "actual_ttft_ms" if actual_metric_name == "ttft_ms" else "actual_queue_ms"
    normalized_estimator = str(inferred_estimator or "").strip().lower()
    use_prefill_estimate_label = normalized_estimator == "prefill_tps_ttft"
    estimate_field = (
        "prefill_tps_estimate_ms" if use_prefill_estimate_label else "live_wait_estimate_ms"
    )

    rows: list[Dict[str, Any]] = []
    missing_mapping = 0
    missing_queue_ms = 0
    missing_actual_metric = 0
    missing_actual_prefill = 0
    routed_instances: set[str] = set()

    for request_id, pred in predicted_records.items():
        instance_id = str(pred.get("instance_id", ""))
        if instance_id:
            routed_instances.add(instance_id)
        predicted_wait_ms = pred.get("predicted_wait_ms")
        if not isinstance(predicted_wait_ms, (int, float)):
            continue
        mapping = response_map.get(request_id)
        if mapping is None:
            missing_mapping += 1
            continue
        response_id = mapping.get("response_id")
        if not isinstance(response_id, str) or not response_id:
            missing_mapping += 1
            continue

        actual_queue = actual_queue_map.get(response_id)
        actual_ttft = actual_ttft_map.get(response_id)
        actual_prefill = actual_prefill_map.get(response_id)
        if actual_queue is None:
            missing_queue_ms += 1
        actual_metric = actual_ttft if actual_metric_name == "ttft_ms" else actual_queue
        if actual_metric is None:
            missing_actual_metric += 1
            continue
        if actual_prefill is None:
            missing_actual_prefill += 1

        prefill_backlog_tokens = pred.get("prefill_backlog_total_tokens")
        live_wait_estimate_ms = (
            float(predicted_wait_ms) if not use_prefill_estimate_label else None
        )
        prefill_tps_estimate_ms = (
            float(predicted_wait_ms) if use_prefill_estimate_label else None
        )

        rows.append(
            {
                "request_id": request_id,
                "response_id": response_id,
                "instance_id": instance_id,
                "actual_queue_ms": float(actual_queue) if actual_queue is not None else None,
                "actual_ttft_ms": float(actual_ttft) if actual_ttft is not None else None,
                "actual_prefill_ms": (
                    float(actual_prefill) if actual_prefill is not None else None
                ),
                "predicted_wait_ms": float(predicted_wait_ms),
                "live_wait_estimate_ms": live_wait_estimate_ms,
                "prefill_tps_estimate_ms": prefill_tps_estimate_ms,
                "prefill_backlog_total_tokens": (
                    float(prefill_backlog_tokens)
                    if isinstance(prefill_backlog_tokens, (int, float))
                    else None
                ),
                "prefill_tokens_for_tps": None,
                "prompt_tokens": None,
                "prefill_tps": None,
            }
        )

    matched_instances = {str(row["instance_id"]) for row in rows if row.get("instance_id")}
    missing_instance_coverage = sorted(routed_instances - matched_instances)
    if missing_instance_coverage:
        raise RuntimeError(
            f"Utility '{utility}' has no matched {actual_metric_name} rows for routed instance(s): "
            f"{', '.join(missing_instance_coverage)}. "
            "Provide all per-model actual wait logs."
        )

    primary_eval = _evaluate_estimator(
        rows=rows,
        estimate_field=estimate_field,
        actual_field=actual_field,
        actual_metric_name=actual_metric_name,
        burst_threshold_ms=burst_threshold_ms,
        min_matched_pairs=min_matched_pairs,
        max_plot_points=max_plot_points,
        output_prefix=output_prefix,
        output_dir=output_dir,
    )

    return {
        "utility": utility,
        "wait_estimator": inferred_estimator,
        "actual_metric_name": actual_metric_name,
        "evaluated_estimate_field": estimate_field,
        "coverage": {
            "predicted_records": len(predicted_records),
            "mapping_records": len(response_map),
            "matched_pairs": len(rows),
            "missing_mapping": missing_mapping,
            "missing_queue_ms": missing_queue_ms,
            "missing_actual_metric": missing_actual_metric,
            "missing_actual_prefill": missing_actual_prefill,
            "routed_instances": sorted(routed_instances),
            "matched_instances": sorted(matched_instances),
        },
        "estimators": {
            estimate_field: primary_eval,
        },
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate wait-estimator GoF for routed router runs using predicted-wait "
            "logs, response maps, and per-model actual wait logs."
        )
    )
    parser.add_argument(
        "--router-output-json",
        type=Path,
        required=True,
        help="Path to a router experiment JSON output produced by experiments.py.",
    )
    parser.add_argument(
        "--actual-wait-log",
        type=Path,
        action="append",
        required=True,
        help=(
            "Path to a per-model actual wait log "
            "(queue_ms/ttft_s/prefill_s lines). "
            "Pass once per model log."
        ),
    )
    parser.add_argument(
        "--utilities",
        nargs="+",
        default=list(DEFAULT_UTILITIES),
        help="Utilities to evaluate (default: hard slo_aware).",
    )
    parser.add_argument(
        "--min-matched-pairs",
        type=int,
        default=1,
        help="Minimum matched pairs required per utility and estimator.",
    )
    parser.add_argument(
        "--burst-threshold-ms",
        type=float,
        default=1.0,
        help="Burst subset threshold on selected actual metric (default: 1.0).",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=20000,
        help="Maximum points to plot per chart (subsampled deterministically if exceeded).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for artifacts and summary JSON (default: router JSON directory).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional explicit summary JSON output path.",
    )

    args = parser.parse_args()
    if args.min_matched_pairs < 1:
        parser.error("--min-matched-pairs must be >= 1.")
    if args.burst_threshold_ms < 0:
        parser.error("--burst-threshold-ms must be >= 0.")
    if args.max_plot_points < 1:
        parser.error("--max-plot-points must be >= 1.")
    return args


def main() -> None:
    args = parse_args()
    router_json_path = args.router_output_json.expanduser()
    if not router_json_path.exists():
        raise FileNotFoundError(f"Router output JSON does not exist: {router_json_path}")

    router_json = _read_json(router_json_path)
    actual_wait_logs = [Path(p).expanduser() for p in args.actual_wait_log]
    for path in actual_wait_logs:
        if not path.exists():
            raise FileNotFoundError(f"Actual wait log does not exist: {path}")

    actual_queue_map, actual_ttft_map, actual_prefill_map = _read_actual_wait_logs(
        actual_wait_logs
    )
    if not actual_queue_map:
        raise RuntimeError(
            "No queue_ms entries found in provided --actual-wait-log files."
        )

    output_dir = (
        args.output_dir.expanduser()
        if args.output_dir is not None
        else router_json_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    utilities = [str(u).strip() for u in args.utilities if str(u).strip()]
    if not utilities:
        raise ValueError("No utilities selected.")

    runs: list[Dict[str, Any]] = []
    for utility in utilities:
        predicted_path, mapping_path = _resolve_utility_paths(
            router_json=router_json,
            router_json_path=router_json_path,
            utility=utility,
        )
        predicted_records = _read_predicted_wait_records(predicted_path)
        response_map = _read_response_map(mapping_path)
        if not predicted_records:
            raise RuntimeError(
                f"Utility '{utility}' has no usable predicted-wait records in {predicted_path}."
            )
        if not response_map:
            raise RuntimeError(
                f"Utility '{utility}' has no usable response mappings in {mapping_path}."
            )

        utility_prefix = f"{router_json_path.stem}_{utility}"
        utility_wait_estimator = _resolve_utility_wait_estimator(
            router_json=router_json,
            utility=utility,
        )
        run = _analyze_utility(
            utility=utility,
            utility_wait_estimator=utility_wait_estimator,
            predicted_records=predicted_records,
            response_map=response_map,
            actual_queue_map=actual_queue_map,
            actual_ttft_map=actual_ttft_map,
            actual_prefill_map=actual_prefill_map,
            burst_threshold_ms=float(args.burst_threshold_ms),
            min_matched_pairs=int(args.min_matched_pairs),
            max_plot_points=int(args.max_plot_points),
            output_dir=output_dir,
            output_prefix=utility_prefix,
        )
        run["request_log_path"] = str(predicted_path)
        run["response_map_path"] = str(mapping_path)
        runs.append(run)

    summary = {
        "mode": "router_wait_gof",
        "router_output_json": str(router_json_path),
        "actual_wait_logs": [str(p) for p in actual_wait_logs],
        "utilities": utilities,
        "burst_threshold_ms": float(args.burst_threshold_ms),
        "min_matched_pairs": int(args.min_matched_pairs),
        "runs": runs,
    }

    output_json = (
        args.output_json.expanduser()
        if args.output_json is not None
        else output_dir / f"{router_json_path.stem}_router_wait_gof.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote router wait GoF summary to {output_json}")


if __name__ == "__main__":
    main()

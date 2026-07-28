from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

from sfs_core.shared.shared_experiment_helpers import _metric_summary


SYSTEM_ENTRY_E2E_COMPLETION_METRIC_DEFINITION = {
    "unit": "ms",
    "start": "system entry (system_entry_perf)",
    "end": "terminal completion of the non-streaming request (completed_perf)",
    "per_request_formula": (
        "latency_ms + system_entry_to_dispatch_ms - arrival_to_dispatch_ms"
    ),
    "equivalent_formula": "(completed_perf - system_entry_perf) * 1000",
    "aggregation": (
        "computed per request before calculating the mean and percentiles"
    ),
    "role": (
        "supplementary full-response latency; system_entry_e2e_ttft_ms remains "
        "the primary paper latency metric"
    ),
}


def _nested_get(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _mean_numeric_values(values: Iterable[Any]) -> float | None:
    numeric = [float(v) for v in values if isinstance(v, (int, float))]
    if not numeric:
        return None
    return float(sum(numeric) / len(numeric))


def _compute_actual_slo_gated_utility_mean(
    run: dict[str, Any], lambda_weight: Any
) -> float | None:
    if not isinstance(lambda_weight, (int, float)):
        return None
    per_request = run.get("per_request")
    if not isinstance(per_request, list) or len(per_request) == 0:
        return None

    lambda_weight_f = float(lambda_weight)
    total_queries = len(per_request)
    gated_utility_sum = 0.0
    for item in per_request:
        if not isinstance(item, dict):
            continue
        if item.get("system_entry_e2e_ttft_slo_met") is not True:
            continue
        actual_accuracy = item.get("actual_accuracy")
        actual_cost = item.get("actual_cost")
        if not isinstance(actual_accuracy, (int, float)):
            continue
        if not isinstance(actual_cost, (int, float)):
            continue
        gated_utility_sum += float(actual_accuracy) - lambda_weight_f * float(actual_cost)

    return gated_utility_sum / total_queries


def _finite_numeric_value(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _system_entry_e2e_completion_metrics(run: dict[str, Any]) -> dict[str, Any]:
    per_request = run.get("per_request")
    if not isinstance(per_request, list):
        per_request = []

    values: list[float] = []
    for item in per_request:
        if not isinstance(item, dict):
            continue
        latency_ms = _finite_numeric_value(item.get("latency_ms"))
        system_entry_to_dispatch_ms = _finite_numeric_value(
            item.get("system_entry_to_dispatch_ms")
        )
        arrival_to_dispatch_ms = _finite_numeric_value(
            item.get("arrival_to_dispatch_ms")
        )
        if (
            latency_ms is None
            or system_entry_to_dispatch_ms is None
            or arrival_to_dispatch_ms is None
        ):
            continue

        value = (
            latency_ms
            + system_entry_to_dispatch_ms
            - arrival_to_dispatch_ms
        )
        if math.isfinite(value) and value >= 0.0:
            values.append(value)

    metric_summary = _metric_summary(values)
    total_records = len(per_request)
    coverage_pct = (
        100.0 * len(values) / total_records if total_records > 0 else 0.0
    )
    return {
        "average_system_entry_e2e_completion_ms": metric_summary["mean"],
        "p50_system_entry_e2e_completion_ms": metric_summary["p50"],
        "p90_system_entry_e2e_completion_ms": metric_summary["p90"],
        "system_entry_e2e_completion_ms_count": metric_summary["count"],
        "system_entry_e2e_completion_ms_coverage_pct": coverage_pct,
    }


def _extract_metrics(
    run: dict[str, Any],
    summary: dict[str, Any],
    lambda_weight: Any,
    *,
    ttft_slo_key: str = "ttft_ms_slo_attainment_pct",
    system_entry_e2e_ttft_slo_key: str = "system_entry_e2e_ttft_ms_slo_attainment_pct",
    include_system_entry_e2e_completion_metrics: bool = False,
) -> dict[str, Any]:
    predicted_accuracy_mean = _nested_get(summary, "predicted_accuracy", "mean")
    predicted_cost_mean = _nested_get(summary, "predicted_cost", "mean")
    actual_cost_mean = _nested_get(summary, "actual_cost", "mean")
    actual_accuracy_mean = _nested_get(summary, "actual_accuracy", "mean")

    if not isinstance(actual_accuracy_mean, (int, float)):
        per_request = run.get("per_request")
        if isinstance(per_request, list):
            actual_accuracy_mean = _mean_numeric_values(
                item.get("actual_accuracy") for item in per_request if isinstance(item, dict)
            )

    predicted_composite = None
    if all(
        isinstance(v, (int, float))
        for v in (predicted_accuracy_mean, predicted_cost_mean, lambda_weight)
    ):
        predicted_composite = float(predicted_accuracy_mean) - float(lambda_weight) * float(
            predicted_cost_mean
        )

    actual_composite = None
    if all(
        isinstance(v, (int, float))
        for v in (actual_accuracy_mean, actual_cost_mean, lambda_weight)
    ):
        actual_composite = float(actual_accuracy_mean) - float(lambda_weight) * float(
            actual_cost_mean
        )

    actual_slo_gated_utility_mean = _compute_actual_slo_gated_utility_mean(
        run, lambda_weight
    )

    metrics = {
        ttft_slo_key: summary.get("ttft_slo_attainment_pct"),
        system_entry_e2e_ttft_slo_key: summary.get("system_entry_e2e_ttft_slo_attainment_pct"),
        "system_entry_to_dispatch_ms_mean": _nested_get(
            summary, "system_entry_to_dispatch_ms", "mean"
        ),
        "average_system_entry_e2e_ttft_ms": _nested_get(
            summary, "system_entry_e2e_ttft_ms", "mean"
        ),
        "p50_system_entry_e2e_ttft_ms": _nested_get(
            summary, "system_entry_e2e_ttft_ms", "p50"
        ),
        "p90_system_entry_e2e_ttft_ms": _nested_get(
            summary, "system_entry_e2e_ttft_ms", "p90"
        ),
        "predicted_accuracy_mean": predicted_accuracy_mean,
        "actual_accuracy_mean": actual_accuracy_mean,
        "predicted_cost_mean": predicted_cost_mean,
        "actual_cost_mean": actual_cost_mean,
        "predicted_accuracy_minus_lambda_weight_times_predicted_cost": predicted_composite,
        "actual_accuracy_minus_lambda_weight_times_actual_cost": actual_composite,
        "actual_slo_gated_utility_mean": actual_slo_gated_utility_mean,
        "throughput_qps_all": summary.get("throughput_qps_all"),
        "throughput_rpm_all": summary.get("throughput_rpm_all"),
    }
    if include_system_entry_e2e_completion_metrics:
        metrics.update(_system_entry_e2e_completion_metrics(run))
    return metrics


def _ordered_utilities(
    discovered_utilities: set[str],
    default_utilities: list[str],
    requested_utilities: list[str] | None = None,
) -> list[str]:
    if requested_utilities is not None:
        return list(requested_utilities)
    return [
        utility for utility in default_utilities if utility in discovered_utilities
    ] + sorted(utility for utility in discovered_utilities if utility not in default_utilities)


def _collect_run_metrics_for_group(
    *,
    run: dict[str, Any],
    per_group_metrics: dict[str, Any],
    per_group_sources: dict[str, str] | None,
    lambda_weight: Any,
    source_path: Path,
    requested_utilities: list[str] | None,
    discovered_utilities: set[str] | None,
    duplicate_policy: str,
    duplicate_scope: str,
    ttft_slo_key: str = "ttft_ms_slo_attainment_pct",
    system_entry_e2e_ttft_slo_key: str = "system_entry_e2e_ttft_ms_slo_attainment_pct",
    include_system_entry_e2e_completion_metrics: bool = False,
    resolve_source_path: bool = False,
    include_existing_source_on_duplicate: bool = True,
) -> None:
    utility = run.get("utility")
    if not isinstance(utility, str):
        return
    if requested_utilities is not None and utility not in requested_utilities:
        return
    if discovered_utilities is not None:
        discovered_utilities.add(utility)

    metrics = _extract_metrics(
        run,
        run.get("summary", {}),
        lambda_weight,
        ttft_slo_key=ttft_slo_key,
        system_entry_e2e_ttft_slo_key=system_entry_e2e_ttft_slo_key,
        include_system_entry_e2e_completion_metrics=(
            include_system_entry_e2e_completion_metrics
        ),
    )
    existing = per_group_metrics.get(utility)
    if existing is not None:
        if duplicate_policy == "error":
            source_path_str = str(source_path)
            if (
                include_existing_source_on_duplicate
                and per_group_sources is not None
                and utility in per_group_sources
            ):
                raise ValueError(
                    "Duplicate data for "
                    f"{duplicate_scope}, utility={utility}: existing="
                    f"{per_group_sources[utility]}, duplicate={source_path_str}"
                )
            raise ValueError(
                f"Duplicate data for {duplicate_scope}, utility={utility}: {source_path_str}"
            )
        if duplicate_policy == "first":
            return

    per_group_metrics[utility] = metrics
    if per_group_sources is not None:
        per_group_sources[utility] = str(
            source_path.resolve() if resolve_source_path else source_path
        )

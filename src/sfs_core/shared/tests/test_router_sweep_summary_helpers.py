from __future__ import annotations

import json

import pytest

from scripts.reporting.router_qps_sweep_summary import aggregate_jsons
from sfs_core.shared.router_sweep_summary_helpers import (
    SYSTEM_ENTRY_E2E_COMPLETION_METRIC_DEFINITION,
    _extract_metrics,
)


def _summary() -> dict[str, object]:
    return {
        "ttft_slo_attainment_pct": 91.0,
        "system_entry_e2e_ttft_ms_slo_attainment_pct": 89.0,
        "system_entry_to_dispatch_ms": {"mean": 3.0},
        "system_entry_e2e_ttft_ms": {
            "mean": 40.0,
            "p50": 35.0,
            "p90": 70.0,
        },
        "throughput_qps_all": 8.75,
        "throughput_rpm_all": 525.0,
    }


def test_completion_metrics_are_derived_per_request() -> None:
    run = {
        "per_request": [
            {
                "latency_ms": 100.0,
                "system_entry_to_dispatch_ms": 30.0,
                "arrival_to_dispatch_ms": 20.0,
            },
            {
                "latency_ms": 200.0,
                "system_entry_to_dispatch_ms": 50.0,
                "arrival_to_dispatch_ms": 10.0,
            },
            {"latency_ms": 300.0},
        ],
    }

    metrics = _extract_metrics(
        run,
        _summary(),
        lambda_weight=0.0005,
        include_system_entry_e2e_completion_metrics=True,
    )

    assert metrics["average_system_entry_e2e_completion_ms"] == pytest.approx(175.0)
    assert metrics["p50_system_entry_e2e_completion_ms"] == pytest.approx(175.0)
    assert metrics["p90_system_entry_e2e_completion_ms"] == pytest.approx(227.0)
    assert metrics["system_entry_e2e_completion_ms_count"] == 2
    assert metrics["system_entry_e2e_completion_ms_coverage_pct"] == pytest.approx(
        200.0 / 3.0
    )
    assert metrics["average_system_entry_e2e_ttft_ms"] == 40.0
    assert metrics["throughput_qps_all"] == 8.75


def test_completion_metrics_reject_invalid_timing_records() -> None:
    run = {
        "per_request": [
            {
                "latency_ms": True,
                "system_entry_to_dispatch_ms": 2.0,
                "arrival_to_dispatch_ms": 1.0,
            },
            {
                "latency_ms": float("inf"),
                "system_entry_to_dispatch_ms": 2.0,
                "arrival_to_dispatch_ms": 1.0,
            },
            {
                "latency_ms": 1.0,
                "system_entry_to_dispatch_ms": 2.0,
                "arrival_to_dispatch_ms": 10.0,
            },
        ],
    }

    metrics = _extract_metrics(
        run,
        _summary(),
        lambda_weight=0.0005,
        include_system_entry_e2e_completion_metrics=True,
    )

    assert metrics["average_system_entry_e2e_completion_ms"] is None
    assert metrics["p50_system_entry_e2e_completion_ms"] is None
    assert metrics["p90_system_entry_e2e_completion_ms"] is None
    assert metrics["system_entry_e2e_completion_ms_count"] == 0
    assert metrics["system_entry_e2e_completion_ms_coverage_pct"] == 0.0


def test_other_sweep_summaries_do_not_gain_completion_fields_by_default() -> None:
    metrics = _extract_metrics({}, _summary(), lambda_weight=0.0005)

    assert "average_system_entry_e2e_completion_ms" not in metrics
    assert "system_entry_e2e_completion_ms_count" not in metrics


def test_qps_aggregation_adds_definition_and_completion_metrics(tmp_path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    payload = {
        "config": {
            "request_rate_qps": 8.75,
            "lambda_weight": 0.0005,
        },
        "router": {
            "runs": [
                {
                    "utility": "hard",
                    "summary": _summary(),
                    "per_request": [
                        {
                            "latency_ms": 100.0,
                            "system_entry_to_dispatch_ms": 30.0,
                            "arrival_to_dispatch_ms": 20.0,
                        }
                    ],
                },
                {
                    "utility": "hard_pk_mg1",
                    "summary": _summary(),
                    "per_request": [],
                }
            ]
        },
    }
    with (input_dir / "run.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file)

    summary, qps_keys = aggregate_jsons([input_dir])

    assert qps_keys == ["8.75"]
    assert summary["utilities"] == ["hard", "hard_pk_mg1"]
    assert summary["metric_definitions"]["system_entry_e2e_completion_ms"] == (
        SYSTEM_ENTRY_E2E_COMPLETION_METRIC_DEFINITION
    )
    hard = summary["qps"]["8.75"]["hard"]
    assert hard["average_system_entry_e2e_completion_ms"] == pytest.approx(110.0)
    assert hard["system_entry_e2e_completion_ms_count"] == 1
    assert hard["system_entry_e2e_completion_ms_coverage_pct"] == 100.0

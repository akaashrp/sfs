from __future__ import annotations

import json

import pytest

from scripts.runs.service_metrics_config import (
    MODEL_KEYS,
    NONNEGATIVE_COEFFICIENT_CONSTRAINT,
    SFS_COEFFICIENT_KEYS,
    build_simulation_args,
    load_and_validate,
    summarize_capacity,
)


def _row(service_rate_qps: float = 1.0):
    return {
        "num_queries": 100,
        "succeeded": 100,
        "failed": 0,
        "service_rate_qps": service_rate_qps,
        "prefill_theta": {
            "matched_pairs": 100,
        },
        "score_proxy": {
            "prefill_tps": 1000.0,
            "decode_tps": 500.0,
            "mean_decode_batch_ms": 2.0,
            "decode_batch_stats_rows_used": 100,
        },
        "sfs_simulation": {
            "feature_set": "legacy",
            "fit_rows": 100,
            "fit_inlier_rows": 99,
            **{
                key: float(index)
                for index, key in enumerate(SFS_COEFFICIENT_KEYS)
            },
        },
    }


def test_validate_requires_all_score_and_sfs_metrics(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps({model: _row() for model in MODEL_KEYS}),
        encoding="utf-8",
    )

    rows = load_and_validate(path, expected_feature_set="legacy")

    assert tuple(rows) == MODEL_KEYS


def test_validate_rejects_mismatched_feature_set(tmp_path):
    payload = {model: _row() for model in MODEL_KEYS}
    payload["qwen3-32b"]["sfs_simulation"]["feature_set"] = "cross_term"
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="expected 'legacy'"):
        load_and_validate(path, expected_feature_set="legacy")


def test_validate_rejects_incomplete_request_or_trace_coverage(tmp_path):
    payload = {model: _row() for model in MODEL_KEYS}
    payload["qwen3-8b"]["failed"] = 1
    payload["qwen3-8b"]["succeeded"] = 99
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must all succeed"):
        load_and_validate(path, expected_feature_set="legacy")

    payload["qwen3-8b"] = _row()
    payload["qwen3-8b"]["prefill_theta"]["matched_pairs"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="match every successful request"):
        load_and_validate(path, expected_feature_set="legacy")


def test_capacity_summary_is_explicit_sum_of_standalone_rates():
    rows = {
        model: _row(service_rate_qps=float(index))
        for index, model in enumerate(MODEL_KEYS, start=1)
    }

    summary = summarize_capacity(rows)

    assert summary == {
        "method": "sum_of_standalone_saturated_service_rates",
        "per_model_qps": {
            "qwen3-0.6b": 1.0,
            "qwen3-8b": 2.0,
            "qwen3-32b": 3.0,
        },
        "aggregate_capacity_qps": 6.0,
        "workload_specific": True,
        "concurrent_contention_validated": False,
    }


def test_cross_term_simulation_args_are_parser_safe_and_use_cross_term_option():
    row = _row()
    sfs = row["sfs_simulation"]
    sfs["feature_set"] = "cross_term"
    sfs["intercept"] = -0.0615
    sfs["sum_coeff"] = -9.68e-7
    sfs["sum_sq_coeff"] = 2.5e-9

    args = build_simulation_args(row)

    by_option = dict(arg.split("=", 1) for arg in args)
    assert float(by_option["--simulation-intercept"]) == -0.0615
    assert float(by_option["--simulation-sum-coeff"]) == -9.68e-7
    assert (
        float(by_option["--simulation-prefill-x-context-coeff"])
        == 2.5e-9
    )
    assert not any(arg.startswith("--simulation-sum-sq-coeff") for arg in args)
    assert all(
        "=" in arg
        for arg in args
        if arg.startswith("--simulation-")
    )


def test_validate_can_require_physical_nonnegative_sfs_fit(tmp_path):
    payload = {model: _row() for model in MODEL_KEYS}
    for row in payload.values():
        sfs = row["sfs_simulation"]
        sfs["coefficient_constraint"] = NONNEGATIVE_COEFFICIENT_CONSTRAINT
        sfs["fit_prediction_diagnostics"] = {
            "minimum_s": 0.001,
            "negative_rows": 0,
            "minimum_nonempty_s": 0.001,
            "negative_nonempty_rows": 0,
            "r2_all_rows": 0.99,
            "mae_s_all_rows": 0.001,
        }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    load_and_validate(
        path,
        expected_feature_set="legacy",
        require_nonnegative_sfs=True,
    )

    payload["qwen3-0.6b"]["sfs_simulation"]["intercept"] = -0.01
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="intercept must be nonnegative"):
        load_and_validate(
            path,
            expected_feature_set="legacy",
            require_nonnegative_sfs=True,
        )

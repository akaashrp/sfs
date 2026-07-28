from __future__ import annotations

import json

import pytest

from scripts.runs.service_metrics_config import (
    MODEL_KEYS,
    SFS_COEFFICIENT_KEYS,
    load_and_validate,
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

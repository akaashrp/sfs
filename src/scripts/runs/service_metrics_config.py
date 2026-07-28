#!/usr/bin/env python3
"""Validate and consume hardware-specific router calibration JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


MODEL_KEYS = ("qwen3-0.6b", "qwen3-8b", "qwen3-32b")
SFS_COEFFICIENT_KEYS = (
    "intercept",
    "prefill_coeff",
    "prefill_sq_coeff",
    "decode_coeff",
    "sum_coeff",
    "sum_sq_coeff",
)


def summarize_capacity(
    rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    per_model_qps = {
        model: float(rows[model]["service_rate_qps"])
        for model in MODEL_KEYS
    }
    return {
        "method": "sum_of_standalone_saturated_service_rates",
        "per_model_qps": per_model_qps,
        "aggregate_capacity_qps": sum(per_model_qps.values()),
        "workload_specific": True,
        "concurrent_contention_validated": False,
    }


def _positive(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return numeric


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    return numeric


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return value


def load_and_validate(
    path: Path,
    *,
    expected_feature_set: str,
) -> dict[str, dict[str, Any]]:
    resolved_path = path.expanduser().resolve()
    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Calibration JSON does not exist: {resolved_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid calibration JSON {resolved_path}: {exc}") from exc
    rows = payload.get("models", payload) if isinstance(payload, dict) else None
    if not isinstance(rows, dict):
        raise ValueError("Calibration JSON must be an object keyed by model")

    validated: dict[str, dict[str, Any]] = {}
    for model_key in MODEL_KEYS:
        row = rows.get(model_key)
        if not isinstance(row, dict):
            raise ValueError(f"Calibration is missing model row {model_key!r}")
        score = row.get("score_proxy")
        prefill = row.get("prefill_theta")
        sfs = row.get("sfs_simulation")
        if not isinstance(score, dict):
            raise ValueError(f"{model_key}.score_proxy must be an object")
        if not isinstance(prefill, dict):
            raise ValueError(f"{model_key}.prefill_theta must be an object")
        if not isinstance(sfs, dict):
            raise ValueError(f"{model_key}.sfs_simulation must be an object")

        num_queries = _nonnegative_int(
            row.get("num_queries"),
            label=f"{model_key}.num_queries",
        )
        succeeded = _nonnegative_int(
            row.get("succeeded"),
            label=f"{model_key}.succeeded",
        )
        failed = _nonnegative_int(
            row.get("failed"),
            label=f"{model_key}.failed",
        )
        if num_queries <= 0 or succeeded != num_queries or failed != 0:
            raise ValueError(
                f"{model_key} calibration requests must all succeed; "
                f"num_queries={num_queries}, succeeded={succeeded}, failed={failed}"
            )
        matched_pairs = _nonnegative_int(
            prefill.get("matched_pairs"),
            label=f"{model_key}.prefill_theta.matched_pairs",
        )
        if matched_pairs != succeeded:
            raise ValueError(
                f"{model_key} prefill calibration must match every successful "
                f"request; matched_pairs={matched_pairs}, succeeded={succeeded}"
            )
        decode_rows = _nonnegative_int(
            score.get("decode_batch_stats_rows_used"),
            label=f"{model_key}.score_proxy.decode_batch_stats_rows_used",
        )
        if decode_rows <= 0:
            raise ValueError(
                f"{model_key}.score_proxy.decode_batch_stats_rows_used must be positive"
            )
        for metric in ("prefill_tps", "decode_tps", "mean_decode_batch_ms"):
            _positive(score.get(metric), label=f"{model_key}.score_proxy.{metric}")
        _positive(
            row.get("service_rate_qps"),
            label=f"{model_key}.service_rate_qps",
        )
        feature_set = str(sfs.get("feature_set", "")).strip()
        if feature_set != expected_feature_set:
            raise ValueError(
                f"{model_key}.sfs_simulation.feature_set={feature_set!r}; "
                f"expected {expected_feature_set!r}"
            )
        for coefficient in SFS_COEFFICIENT_KEYS:
            _finite(
                sfs.get(coefficient),
                label=f"{model_key}.sfs_simulation.{coefficient}",
            )
        fit_rows = _nonnegative_int(
            sfs.get("fit_rows"),
            label=f"{model_key}.sfs_simulation.fit_rows",
        )
        fit_inlier_rows = _nonnegative_int(
            sfs.get("fit_inlier_rows"),
            label=f"{model_key}.sfs_simulation.fit_inlier_rows",
        )
        if fit_rows <= len(SFS_COEFFICIENT_KEYS) or not (
            len(SFS_COEFFICIENT_KEYS) < fit_inlier_rows <= fit_rows
        ):
            raise ValueError(
                f"{model_key} has insufficient SFS fit rows; "
                f"fit_rows={fit_rows}, fit_inlier_rows={fit_inlier_rows}"
            )
        validated[model_key] = row
    return validated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument(
        "--expected-feature-set",
        choices=("legacy", "cross_term"),
        default="legacy",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    subparsers.add_parser("capacity-summary")

    coefficients = subparsers.add_parser("coefficients")
    coefficients.add_argument("--model", choices=MODEL_KEYS, required=True)

    qps = subparsers.add_parser("qps-values")
    qps.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=(0.4, 0.6, 0.8, 1.0, 1.1),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_and_validate(
        args.path,
        expected_feature_set=args.expected_feature_set,
    )
    if args.command == "validate":
        print(
            json.dumps(
                {
                    "path": str(args.path.expanduser().resolve()),
                    "models": list(MODEL_KEYS),
                    "feature_set": args.expected_feature_set,
                },
                sort_keys=True,
            )
        )
        return
    if args.command == "coefficients":
        sfs = rows[args.model]["sfs_simulation"]
        for key in SFS_COEFFICIENT_KEYS:
            print(f"{float(sfs[key]):.17g}")
        return
    if args.command == "capacity-summary":
        print(json.dumps(summarize_capacity(rows), sort_keys=True))
        return
    if args.command == "qps-values":
        capacity_qps = float(summarize_capacity(rows)["aggregate_capacity_qps"])
        values: list[float] = []
        for fraction in args.fractions:
            if not math.isfinite(fraction) or fraction <= 0:
                raise ValueError("QPS fractions must be finite and positive")
            value = round(capacity_qps * float(fraction), 3)
            if value > 0 and value not in values:
                values.append(value)
        print(" ".join(f"{value:g}" for value in values))
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()

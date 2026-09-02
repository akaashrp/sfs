#!/usr/bin/env python3
"""Validate and consume hardware-specific router calibration JSON."""

from __future__ import annotations

import argparse
import copy
import hashlib
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
NONNEGATIVE_COEFFICIENT_CONSTRAINT = "nonnegative_intercept_and_slopes"
MINIMUM_SFS_FIT_R2 = 0.95


def summarize_capacity(
    rows: dict[str, dict[str, Any]],
    *,
    model_keys: tuple[str, ...] = MODEL_KEYS,
) -> dict[str, Any]:
    per_model_qps = {
        model: float(rows[model]["service_rate_qps"])
        for model in model_keys
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


def _nonnegative(value: Any, *, label: str) -> float:
    numeric = _finite(value, label=label)
    if numeric < 0:
        raise ValueError(f"{label} must be nonnegative")
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
    require_nonnegative_sfs: bool = False,
    model_keys: tuple[str, ...] = MODEL_KEYS,
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
    for model_key in model_keys:
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
            validate_coefficient = (
                _nonnegative if require_nonnegative_sfs else _finite
            )
            validate_coefficient(
                sfs.get(coefficient),
                label=f"{model_key}.sfs_simulation.{coefficient}",
            )
        if require_nonnegative_sfs:
            constraint = str(sfs.get("coefficient_constraint", "")).strip()
            if constraint != NONNEGATIVE_COEFFICIENT_CONSTRAINT:
                raise ValueError(
                    f"{model_key}.sfs_simulation.coefficient_constraint="
                    f"{constraint!r}; expected "
                    f"{NONNEGATIVE_COEFFICIENT_CONSTRAINT!r}"
                )
            diagnostics = sfs.get("fit_prediction_diagnostics")
            if not isinstance(diagnostics, dict):
                raise ValueError(
                    f"{model_key}.sfs_simulation.fit_prediction_diagnostics "
                    "must be an object"
                )
            minimum_s = _nonnegative(
                diagnostics.get("minimum_s"),
                label=(
                    f"{model_key}.sfs_simulation."
                    "fit_prediction_diagnostics.minimum_s"
                ),
            )
            negative_rows = _nonnegative_int(
                diagnostics.get("negative_rows"),
                label=(
                    f"{model_key}.sfs_simulation."
                    "fit_prediction_diagnostics.negative_rows"
                ),
            )
            if negative_rows != 0:
                raise ValueError(
                    f"{model_key} has {negative_rows} negative fitted batch "
                    "latencies"
                )
            _positive(
                diagnostics.get("minimum_nonempty_s"),
                label=(
                    f"{model_key}.sfs_simulation."
                    "fit_prediction_diagnostics.minimum_nonempty_s"
                ),
            )
            negative_nonempty_rows = _nonnegative_int(
                diagnostics.get("negative_nonempty_rows"),
                label=(
                    f"{model_key}.sfs_simulation."
                    "fit_prediction_diagnostics.negative_nonempty_rows"
                ),
            )
            if negative_nonempty_rows != 0:
                raise ValueError(
                    f"{model_key} has {negative_nonempty_rows} negative "
                    "nonempty fitted batch latencies"
                )
            r2_all_rows = _finite(
                diagnostics.get("r2_all_rows"),
                label=(
                    f"{model_key}.sfs_simulation."
                    "fit_prediction_diagnostics.r2_all_rows"
                ),
            )
            if r2_all_rows < MINIMUM_SFS_FIT_R2:
                raise ValueError(
                    f"{model_key} SFS fit R^2={r2_all_rows:.6f} is below "
                    f"{MINIMUM_SFS_FIT_R2:.2f}"
                )
            _nonnegative(
                diagnostics.get("mae_s_all_rows"),
                label=(
                    f"{model_key}.sfs_simulation."
                    "fit_prediction_diagnostics.mae_s_all_rows"
                ),
            )
            if minimum_s == 0 and all(
                float(sfs[key]) == 0 for key in SFS_COEFFICIENT_KEYS
            ):
                raise ValueError(
                    f"{model_key} has an all-zero SFS batch-latency model"
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


def build_simulation_args(row: dict[str, Any]) -> list[str]:
    """Return parser-safe vLLM simulation arguments for one model row."""
    sfs = row["sfs_simulation"]
    feature_set = str(sfs["feature_set"])
    final_option = (
        "simulation-prefill-x-context-coeff"
        if feature_set == "cross_term"
        else "simulation-sum-sq-coeff"
    )
    values = (
        ("simulation-intercept", sfs["intercept"]),
        ("simulation-prefill-coeff", sfs["prefill_coeff"]),
        ("simulation-prefill-sq-coeff", sfs["prefill_sq_coeff"]),
        ("simulation-decode-coeff", sfs["decode_coeff"]),
        ("simulation-sum-coeff", sfs["sum_coeff"]),
        (final_option, sfs["sum_sq_coeff"]),
    )
    # Keep each floating-point option and value in one argv token. vLLM's
    # FlexibleArgumentParser otherwise mistakes a standalone negative decimal
    # for a dotted JSON option.
    return [
        f"--simulation-batch-time-feature-set={feature_set}",
        *(f"--{option}={float(value):.17g}" for option, value in values),
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fit_coefficients(fit: Any) -> dict[str, float]:
    coefficient_by_name = {
        str(name): float(coefficient)
        for name, coefficient in zip(fit.feature_names, fit.base_model.coef_)
    }
    final_feature_name = "s_sq" if fit.feature_set == "legacy" else "p_x_ctx"
    return {
        "intercept": float(fit.base_model.intercept_),
        "prefill_coeff": coefficient_by_name["p"],
        "decode_coeff": coefficient_by_name["d"],
        "sum_coeff": coefficient_by_name["s"],
        "prefill_sq_coeff": coefficient_by_name["p_sq_sum"],
        "sum_sq_coeff": coefficient_by_name[final_feature_name],
    }


def _fit_prediction_diagnostics(
    batch_df: Any,
    predictions: Any,
) -> dict[str, float | int]:
    observed = batch_df["exec"].to_numpy()
    total_variation = float(((observed - observed.mean()) ** 2).sum())
    residual_variation = float(((observed - predictions) ** 2).sum())
    nonempty = (
        batch_df["prefill"].to_numpy() + batch_df["decode"].to_numpy()
    ) > 0
    if not nonempty.any():
        raise ValueError("Batch-latency fit contains no nonempty batches")
    return {
        "minimum_s": float(predictions.min()),
        "negative_rows": int((predictions < 0).sum()),
        "minimum_nonempty_s": float(predictions[nonempty].min()),
        "negative_nonempty_rows": int((predictions[nonempty] < 0).sum()),
        "r2_all_rows": float(
            1.0 - residual_variation / total_variation
            if total_variation > 0
            else 1.0
        ),
        "mae_s_all_rows": float(abs(observed - predictions).mean()),
    }


def derive_nonnegative_calibration(
    source_path: Path,
    output_path: Path,
    *,
    expected_feature_set: str,
    model_keys: tuple[str, ...] = MODEL_KEYS,
) -> Path:
    """Refit only the SFS latency coefficients from an existing calibration."""
    import pandas as pd

    from sfs_core.regression.two_part_fit import (
        build_feature_matrix,
        fit_two_part_from_df,
    )

    source_path = source_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if output_path.exists():
        raise ValueError(f"Refusing to overwrite existing output: {output_path}")
    source_rows = load_and_validate(
        source_path,
        expected_feature_set=expected_feature_set,
        model_keys=model_keys,
    )
    derived_rows = copy.deepcopy(source_rows)
    source_sha256 = _sha256(source_path)

    for model_key in model_keys:
        source_row = source_rows[model_key]
        source_sfs = source_row["sfs_simulation"]
        batch_path = (
            Path(str(source_row["batch_stats_csv_path"]))
            .expanduser()
            .resolve()
        )
        if not batch_path.is_file():
            raise ValueError(
                f"{model_key} batch-stats trace does not exist: {batch_path}"
            )
        fit_rows = int(source_sfs["fit_rows"])
        batch_df = pd.read_csv(batch_path)
        if len(batch_df) < fit_rows:
            raise ValueError(
                f"{model_key} batch-stats trace has {len(batch_df)} rows, "
                f"fewer than fit_rows={fit_rows}"
            )
        # The calibration snapshots the byte offset after warmup, then fits all
        # rows appended after it. Those are exactly the final fit_rows rows.
        fit_df = batch_df.tail(fit_rows).reset_index(drop=True)
        reproduced_fit = fit_two_part_from_df(
            fit_df,
            stall_percentile=99.9,
            feature_set=expected_feature_set,
        )
        reproduced = _fit_coefficients(reproduced_fit)
        max_abs_error = max(
            abs(float(reproduced[key]) - float(source_sfs[key]))
            for key in SFS_COEFFICIENT_KEYS
        )
        if max_abs_error > 1e-9 or int(reproduced_fit.inlier_mask.sum()) != int(
            source_sfs["fit_inlier_rows"]
        ):
            raise ValueError(
                f"{model_key} source fit could not be reproduced exactly; "
                f"max_abs_error={max_abs_error:.3g}, "
                f"inliers={int(reproduced_fit.inlier_mask.sum())}, "
                f"expected_inliers={int(source_sfs['fit_inlier_rows'])}"
            )

        constrained_fit = fit_two_part_from_df(
            fit_df,
            stall_percentile=99.9,
            feature_set=expected_feature_set,
            nonnegative_coefficients=True,
        )
        constrained = _fit_coefficients(constrained_fit)
        features, _ = build_feature_matrix(
            fit_df,
            feature_set=expected_feature_set,
        )
        predictions = constrained_fit.predict_typical(features)
        updated_sfs = derived_rows[model_key]["sfs_simulation"]
        updated_sfs.update(constrained)
        updated_sfs.update(
            {
                "coefficient_constraint": constrained_fit.coefficient_constraint,
                "fit_prediction_diagnostics": _fit_prediction_diagnostics(
                    fit_df,
                    predictions,
                ),
                "source_unconstrained_fit_reproduced": True,
                "source_unconstrained_fit_max_abs_error": float(max_abs_error),
                "source_batch_stats_sha256": _sha256(batch_path),
            }
        )

    payload = {
        "models": derived_rows,
        "derivation": {
            "method": "nnls_on_original_huber_inliers",
            "source_calibration_path": str(source_path),
            "source_calibration_sha256": source_sha256,
            "preserved_metrics": [
                "service_rate_qps",
                "prefill_theta",
                "score_proxy",
            ],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    output_path.with_name(f"{output_path.name}.sha256").write_text(
        f"{_sha256(output_path)}  {output_path.name}\n",
        encoding="utf-8",
    )
    load_and_validate(
        output_path,
        expected_feature_set=expected_feature_set,
        require_nonnegative_sfs=True,
        model_keys=model_keys,
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument(
        "--model-key",
        action="append",
        dest="model_keys",
        default=None,
        help=(
            "Expected model key in the calibration JSON; repeat once per "
            "model. Defaults to the legacy Qwen family."
        ),
    )
    parser.add_argument(
        "--expected-feature-set",
        choices=("legacy", "cross_term"),
        default="legacy",
    )
    parser.add_argument(
        "--require-nonnegative-sfs",
        action="store_true",
        help=(
            "Require a nonnegative SFS latency fit and its prediction "
            "diagnostics."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    subparsers.add_parser("capacity-summary")

    coefficients = subparsers.add_parser("coefficients")
    coefficients.add_argument("--model", required=True)

    simulation_args = subparsers.add_parser("simulation-args")
    simulation_args.add_argument("--model", required=True)

    refit = subparsers.add_parser("refit-nonnegative")
    refit.add_argument("--output-path", type=Path, required=True)

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
    model_keys = tuple(
        str(model).strip() for model in (args.model_keys or MODEL_KEYS)
    )
    if not model_keys or any(not model for model in model_keys):
        raise ValueError(
            "--model-key must contain at least one non-empty model key"
        )
    if len(set(model_keys)) != len(model_keys):
        raise ValueError("--model-key must not contain duplicate model keys")
    rows = load_and_validate(
        args.path,
        expected_feature_set=args.expected_feature_set,
        require_nonnegative_sfs=args.require_nonnegative_sfs,
        model_keys=model_keys,
    )
    if args.command == "validate":
        print(
            json.dumps(
                {
                    "path": str(args.path.expanduser().resolve()),
                    "models": list(model_keys),
                    "feature_set": args.expected_feature_set,
                },
                sort_keys=True,
            )
        )
        return
    if args.command == "coefficients":
        if args.model not in rows:
            raise ValueError(
                f"--model {args.model!r} is not one of {list(model_keys)!r}"
            )
        sfs = rows[args.model]["sfs_simulation"]
        for key in SFS_COEFFICIENT_KEYS:
            print(f"{float(sfs[key]):.17g}")
        return
    if args.command == "simulation-args":
        if args.model not in rows:
            raise ValueError(
                f"--model {args.model!r} is not one of {list(model_keys)!r}"
            )
        print("\n".join(build_simulation_args(rows[args.model])))
        return
    if args.command == "refit-nonnegative":
        output_path = derive_nonnegative_calibration(
            args.path,
            args.output_path,
            expected_feature_set=args.expected_feature_set,
            model_keys=model_keys,
        )
        print(output_path)
        return
    if args.command == "capacity-summary":
        print(
            json.dumps(
                summarize_capacity(rows, model_keys=model_keys),
                sort_keys=True,
            )
        )
        return
    if args.command == "qps-values":
        capacity_qps = float(
            summarize_capacity(rows, model_keys=model_keys)[
                "aggregate_capacity_qps"
            ]
        )
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

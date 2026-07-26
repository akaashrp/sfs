"""Fit router-estimate to EngineCore-readiness delay from wait_gof runs."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack


READY_TS_PATTERN = re.compile(
    r"\brequest_id=(?P<request_id>\S+).*?\bready_ts_s=(?P<ready_ts>[0-9.eE+-]+)"
)
MODEL_FEATURES = {
    "constant": (),
    "prompt_tokens": ("prompt_tokens",),
    "prompt_tokens_pending": (
        "prompt_tokens",
        "pending_dispatch_count",
    ),
}


@dataclass(frozen=True, slots=True)
class ReadinessSample:
    run_name: str
    request_id: str
    prompt_tokens: int
    pending_dispatch_count: int
    delay_ms: float


def _read_actual_ready_times(path: Path) -> dict[str, float]:
    ready_times: dict[str, float] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as source:
        for line in source:
            match = READY_TS_PATTERN.search(line)
            if match is None:
                continue
            ready_times[match.group("request_id")] = float(
                match.group("ready_ts")
            )
    return ready_times


def _read_router_records(path: Path) -> dict[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as source:
        for line in source:
            try:
                record = ast.literal_eval(line)
            except (SyntaxError, ValueError):
                continue
            if not isinstance(record, Mapping):
                continue
            request_id = record.get("request_id")
            if isinstance(request_id, str):
                records[request_id] = record
    return records


def load_run_samples(result_path: Path) -> list[ReadinessSample]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    run = payload["wait_gof"]["runs"][0]
    request_log_path = Path(run["request_log_path"])
    actual_log_path = result_path.parent / "actual_wait_times.log"
    router_records = _read_router_records(request_log_path)
    actual_ready_times = _read_actual_ready_times(actual_log_path)

    samples: list[ReadinessSample] = []
    for item in run["per_request"]:
        scheduler_request_id = item.get("scheduler_request_id")
        response_id = item.get("response_id")
        if not isinstance(scheduler_request_id, str) or not isinstance(
            response_id,
            str,
        ):
            continue
        record = router_records.get(scheduler_request_id)
        ready_ts = actual_ready_times.get(response_id)
        if record is None or ready_ts is None:
            continue
        record_payload = record.get("payload")
        if not isinstance(record_payload, Mapping):
            continue
        reports = record_payload.get("reports")
        inputs = record_payload.get("_readiness_predictor_inputs")
        if (
            not isinstance(reports, list)
            or not reports
            or not isinstance(reports[0], Mapping)
            or not isinstance(inputs, Mapping)
        ):
            continue
        simulation_timestamp = reports[0].get("simulation_timestamp")
        prompt_tokens = inputs.get("prompt_tokens")
        pending_count = inputs.get("pending_dispatch_count")
        if (
            not isinstance(simulation_timestamp, (int, float))
            or not isinstance(prompt_tokens, int)
            or not isinstance(pending_count, int)
        ):
            continue
        delay_ms = (ready_ts - float(simulation_timestamp)) * 1000.0
        if not math.isfinite(delay_ms) or delay_ms < 0.0:
            continue
        samples.append(
            ReadinessSample(
                run_name=result_path.parent.parent.name,
                request_id=scheduler_request_id,
                prompt_tokens=prompt_tokens,
                pending_dispatch_count=pending_count,
                delay_ms=delay_ms,
            )
        )

    expected = len(run["per_request"])
    if len(samples) < expected * 0.9:
        raise RuntimeError(
            f"Matched only {len(samples)} of {expected} readiness samples "
            f"for {result_path}"
        )
    return samples


def _design_matrix(
    samples: Sequence[ReadinessSample],
    feature_names: Sequence[str],
) -> np.ndarray:
    columns: list[np.ndarray] = [np.ones(len(samples), dtype=np.float64)]
    for feature_name in feature_names:
        columns.append(
            np.asarray(
                [float(getattr(sample, feature_name)) for sample in samples],
                dtype=np.float64,
            )
        )
    return np.column_stack(columns)


def fit_nonnegative_mae(
    samples: Sequence[ReadinessSample],
    feature_names: Sequence[str],
) -> np.ndarray:
    if not samples:
        raise ValueError("At least one readiness sample is required")
    x = _design_matrix(samples, feature_names)
    y = np.asarray([sample.delay_ms for sample in samples], dtype=np.float64)
    sample_count, coefficient_count = x.shape
    residual_identity = eye(sample_count, format="csr")
    x_sparse = csr_matrix(x)
    constraints = vstack(
        (
            hstack((x_sparse, -residual_identity)),
            hstack((-x_sparse, -residual_identity)),
        ),
        format="csr",
    )
    bounds = [(0.0, None)] * (coefficient_count + sample_count)
    objective = np.concatenate(
        (
            np.zeros(coefficient_count, dtype=np.float64),
            np.ones(sample_count, dtype=np.float64),
        )
    )
    result = linprog(
        objective,
        A_ub=constraints,
        b_ub=np.concatenate((y, -y)),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Readiness MAE fit failed: {result.message}")
    return np.asarray(result.x[:coefficient_count], dtype=np.float64)


def _predict(
    samples: Sequence[ReadinessSample],
    feature_names: Sequence[str],
    coefficients: np.ndarray,
) -> np.ndarray:
    return _design_matrix(samples, feature_names) @ coefficients


def _metrics(
    samples: Sequence[ReadinessSample],
    predictions: np.ndarray,
) -> dict[str, float | int]:
    actual = np.asarray(
        [sample.delay_ms for sample in samples],
        dtype=np.float64,
    )
    errors = predictions - actual
    return {
        "count": len(samples),
        "mae_ms": float(np.mean(np.abs(errors))),
        "rmse_ms": float(np.sqrt(np.mean(errors**2))),
        "bias_ms": float(np.mean(errors)),
        "actual_mean_ms": float(np.mean(actual)),
        "predicted_mean_ms": float(np.mean(predictions)),
    }


def evaluate_leave_one_run_out(
    samples_by_run: Mapping[str, Sequence[ReadinessSample]],
) -> dict[str, dict[str, Any]]:
    if len(samples_by_run) < 2:
        raise ValueError("Leave-one-run-out evaluation requires at least two runs")
    evaluations: dict[str, dict[str, Any]] = {}
    for model_name, feature_names in MODEL_FEATURES.items():
        folds: list[dict[str, Any]] = []
        all_errors: list[float] = []
        for heldout_name, heldout_samples in samples_by_run.items():
            training_samples = [
                sample
                for run_name, run_samples in samples_by_run.items()
                if run_name != heldout_name
                for sample in run_samples
            ]
            coefficients = fit_nonnegative_mae(
                training_samples,
                feature_names,
            )
            predictions = _predict(
                heldout_samples,
                feature_names,
                coefficients,
            )
            fold_metrics = _metrics(heldout_samples, predictions)
            folds.append(
                {
                    "heldout_run": heldout_name,
                    **fold_metrics,
                }
            )
            all_errors.extend(
                abs(
                    float(predicted) - sample.delay_ms
                )
                for sample, predicted in zip(heldout_samples, predictions)
            )
        evaluations[model_name] = {
            "features": list(feature_names),
            "macro_mae_ms": float(
                np.mean([fold["mae_ms"] for fold in folds])
            ),
            "pooled_mae_ms": float(np.mean(all_errors)),
            "folds": folds,
        }
    return evaluations


def build_fit_payload(
    samples_by_run: Mapping[str, Sequence[ReadinessSample]],
) -> dict[str, Any]:
    evaluations = evaluate_leave_one_run_out(samples_by_run)
    selected_model = min(
        MODEL_FEATURES,
        key=lambda name: (
            evaluations[name]["macro_mae_ms"],
            len(MODEL_FEATURES[name]),
        ),
    )
    selected_features = MODEL_FEATURES[selected_model]
    all_samples = [
        sample for run_samples in samples_by_run.values() for sample in run_samples
    ]
    fitted = fit_nonnegative_mae(all_samples, selected_features)
    coefficient_by_name = {
        "intercept_ms": float(fitted[0]),
        "prompt_token_ms": 0.0,
        "pending_dispatch_ms": 0.0,
    }
    for name, value in zip(selected_features, fitted[1:]):
        output_name = (
            "prompt_token_ms"
            if name == "prompt_tokens"
            else "pending_dispatch_ms"
        )
        coefficient_by_name[output_name] = float(value)

    return {
        "version": 1,
        "target": "router_estimate_to_engine_ready_ms",
        "fit_objective": "nonnegative_mean_absolute_error",
        "selected_model": selected_model,
        "features": list(selected_features),
        "coefficients": coefficient_by_name,
        "selection": {
            "criterion": "lowest leave-one-run-out macro MAE",
            "models": evaluations,
        },
        "training": {
            "sample_count": len(all_samples),
            "runs": {
                name: len(samples) for name, samples in samples_by_run.items()
            },
            "in_sample": _metrics(
                all_samples,
                _predict(all_samples, selected_features, fitted),
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        action="append",
        type=Path,
        required=True,
        help="Corrected readiness-mode wait_gof result.json; repeat per run.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples_by_run: dict[str, list[ReadinessSample]] = {}
    for result_path in args.result:
        samples = load_run_samples(result_path)
        run_name = result_path.parent.parent.name
        if run_name in samples_by_run:
            raise ValueError(f"Duplicate readiness run name: {run_name}")
        samples_by_run[run_name] = samples
    payload = build_fit_payload(samples_by_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

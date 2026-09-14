"""Fit Mooncake prefill and RouteBalance TPOT heads from declared calibration traces.

Input manifest example::

  {"data_role": "calibration", "serving_profile_verified": true,
   "serving_profile": {"max_num_batched_tokens": 32768, "...": "full configuration"},
   "models": {"ministral3-3b": {
     "service_rate_qps": 1.0, "service_rate_definition": "corrected whole-run requests/elapsed",
     "traces": ["/absolute/path/batch_stats_ministral3-3b.csv"]}}}

Supply the actual full serving profile and fresh measured speeds. This script
does not discover old traces or infer throughput from inherited defaults.
Each trace is split chronologically 80/20 before fitting; diagnostics are
computed on its held-out tail, then all eligible calibration rows are fitted.
Mixed prefill/decode rows train TPOT but never the isolated prefill model.
Prefill uses an explicit causal-quadratic work model:
``intercept + a*p + b*(p*p + 2*p*processed_context)``. The tied context
coefficient permits calibration from ordinary full-prefill length probes;
absence of partial-prefill observations remains visible in coverage reports.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from sfs_core.routing.methodology_calibration import (
    CALIBRATION_SCHEMA_VERSION, PREFILL_FEATURE_NAMES, TPOT_FEATURE_NAMES,
    file_sha256, serving_profile_sha256,
)
from sfs_core.routing.methodology_policies import _count, _positive

_REQUIRED_COLUMNS = ("prefill", "prefill_sq_sum", "decode", "num_seqs",
                     "sum_tokens", "prefill_x_processed_ctx_sum", "exec")


def load_trace_rows(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    rows, provenance = [], []
    for trace_index, path in enumerate(paths):
        valid, invalid = 0, 0
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            missing = set(_REQUIRED_COLUMNS) - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} lacks required trace features {sorted(missing)}")
            for row_index, record in enumerate(reader):
                try:
                    row = {name: float(record[name]) for name in _REQUIRED_COLUMNS}
                except (ValueError, TypeError):
                    invalid += 1
                    continue
                if any(not math.isfinite(value) or value < 0 for value in row.values()) or row["exec"] <= 0:
                    invalid += 1
                    continue
                if any(int(row[name]) != row[name] for name in
                       ("prefill", "decode", "num_seqs", "sum_tokens")):
                    invalid += 1
                    continue
                row.update(trace_index=trace_index, row_index=row_index)
                rows.append(row)
                valid += 1
        provenance.append({"path": str(path), "sha256": file_sha256(path),
                           "valid_rows": valid, "invalid_rows": invalid})
    return rows, provenance


def _chronological_split(rows: list[dict]) -> tuple[list[int], list[int]]:
    grouped = {}
    for index, row in enumerate(rows):
        grouped.setdefault(row["trace_index"], []).append(index)
    train, test = [], []
    for indices in grouped.values():
        split = max(1, int(len(indices) * .8))
        train.extend(indices[:split])
        test.extend(indices[split:])
    if not train or not test:
        raise ValueError("need sufficient independent calibration rows for a chronological holdout")
    return train, test


def _error_report(observed, predicted) -> dict:
    import numpy as np
    error = np.asarray(predicted) - np.asarray(observed)
    return {"rows": int(len(error)), "mae_ms": float(np.mean(np.abs(error))),
            "median_absolute_error_ms": float(np.median(np.abs(error))),
            "p95_absolute_error_ms": float(np.percentile(np.abs(error), 95)),
            "mean_bias_ms": float(np.mean(error)),
            "mean_observed_ms": float(np.mean(observed))}


def _ranges_report(values, observed, predicted, upper_bounds) -> list[dict]:
    import numpy as np
    report = []
    lower = 0
    for upper in upper_bounds:
        mask = (values >= lower) & (values < upper)
        if np.any(mask):
            report.append({"lower_inclusive": lower,
                           "upper_exclusive": upper if math.isfinite(upper) else None,
                           **_error_report(observed[mask], predicted[mask])})
        lower = upper
    return report


def fit_prefill(rows: list[dict], *, chunk_tokens: int, min_rows: int = 40) -> dict:
    import numpy as np
    from scipy.optimize import nnls

    # Singleton execution gives identifiable per-request timing; a batch
    # intercept from multi-request batches cannot be charged to each request.
    pure = [r for r in rows if r["decode"] == 0 and r["num_seqs"] == 1 and r["prefill"] > 0]
    if len(pure) < min_rows:
        raise ValueError(f"need >= {min_rows} pure singleton prefill rows; found {len(pure)}; add isolated probes")
    if any(r["prefill"] > chunk_tokens for r in pure):
        raise ValueError("prefill rows exceed declared max_num_batched_tokens")
    if any(not math.isclose(r["prefill_sq_sum"], r["prefill"] ** 2, rel_tol=1e-9) for r in pure):
        raise ValueError("singleton prefill squares do not match per-request work")
    x = np.array([[1, r["prefill"],
                   r["prefill_sq_sum"] + 2 * r["prefill_x_processed_ctx_sum"]]
                  for r in pure], dtype=float)
    y = np.array([r["exec"] * 1000 for r in pure])
    scale = np.array([1, 1e4, 1e8])
    train, test = _chronological_split(pure)
    if np.linalg.matrix_rank(x[train] / scale) < 3:
        raise ValueError("prefill training rows cannot identify the causal-quadratic fit; add varied-length prefill probes")
    coefficients, _ = nnls(x[train] / scale, y[train])
    predicted = (x[test] / scale) @ coefficients
    context = np.array([r["prefill_x_processed_ctx_sum"] / r["prefill"] for r in pure])
    prompt_extent = x[:, 1] + context
    report = {
        "fit_rows": len(pure), "heldout": _error_report(y[test], predicted),
        "heldout_by_prompt_extent_tokens": _ranges_report(
            prompt_extent[test], y[test], predicted, [512, 2048, 8192, 32769, 131073, float("inf")]),
        "partial_prefill_rows": int(np.sum(context > 0)),
        "partial_prefill_observed": bool(np.any(context > 0)),
        "chunk_token_range": [float(x[:, 1].min()), float(x[:, 1].max())],
        "processed_context_range": [float(context.min()), float(context.max())],
        "heldout_split": "per_trace_chronological_last_20_percent",
        "requires_runtime_residual_audit": True,
    }
    all_coefficients, _ = nnls(x / scale, y)
    intercept, linear, quadratic = (all_coefficients / scale).tolist()
    return {"feature_names": list(PREFILL_FEATURE_NAMES),
            "coefficients_ms": [intercept, linear, quadratic, 2 * quadratic],
            "chunk_tokens": chunk_tokens, "fit_variant": "nonnegative_singleton_chunk_execution",
            "parameterization": "intercept_linear_causal_quadratic",
            "context_constraint": "context_coefficient_equals_twice_quadratic",
            "coverage": report}


def fit_tpot(rows: list[dict], *, seed: int = 42, min_rows: int = 40,
             n_estimators: int = 120):
    import numpy as np
    import xgboost as xgb

    eligible = [r for r in rows if r["decode"] > 0]
    if len(eligible) < min_rows:
        raise ValueError(f"need >= {min_rows} decode-active rows; found {len(eligible)}")
    x = np.array([[r["decode"], r["prefill"], r["sum_tokens"]] for r in eligible])
    y = np.array([r["exec"] * 1000 for r in eligible])
    train, test = _chronological_split(eligible)
    kwargs = dict(n_estimators=n_estimators, max_depth=4, learning_rate=.05,
                  objective="reg:squarederror", tree_method="hist", n_jobs=1,
                  random_state=seed, subsample=1.0, colsample_bytree=1.0)
    head = xgb.XGBRegressor(**kwargs)
    head.fit(x[train], y[train])
    predicted = head.predict(x[test])
    report = {
        "fit_rows": len(eligible), "heldout": _error_report(y[test], predicted),
        "heldout_by_decode_tokens": _ranges_report(
            x[test, 0], y[test], predicted, [2, 8, 32, 128, 513, float("inf")]),
        "feature_ranges": {name: [float(x[:, index].min()), float(x[:, index].max())]
                           for index, name in enumerate(TPOT_FEATURE_NAMES)},
        "pure_decode_rows": int(np.sum(x[:, 1] == 0)),
        "mixed_prefill_decode_rows": int(np.sum(x[:, 1] > 0)),
        "heldout_split": "per_trace_chronological_last_20_percent",
        "requires_runtime_residual_audit": True,
    }
    head.fit(x, y)
    return head, {"feature_names": list(TPOT_FEATURE_NAMES),
                  "target": "decode_active_iteration_exec_ms",
                  "fit_variant": "xgboost_decode_iteration_execution",
                  "hyperparameters": kwargs, "coverage": report}


def fit_manifest(manifest_path: str | Path, output_dir: str | Path, *,
                 seed: int = 42, min_rows: int = 40, n_estimators: int = 120) -> Path:
    manifest_path = Path(manifest_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise ValueError("output directory already exists; preserve artifacts and choose a new directory")
    with manifest_path.open() as stream:
        manifest = json.load(stream)
    if manifest.get("data_role") != "calibration" or manifest.get("serving_profile_verified") is not True:
        raise ValueError("manifest must declare calibration data and verify the corrected serving profile")
    profile = manifest.get("serving_profile")
    if not isinstance(profile, dict) or not profile:
        raise ValueError("full serving_profile provenance is required")
    chunk_tokens = _count(profile.get("max_num_batched_tokens", 0), "max_num_batched_tokens")
    if not chunk_tokens:
        raise ValueError("serving profile must provide positive max_num_batched_tokens")
    if not isinstance(manifest.get("models"), dict) or not manifest["models"]:
        raise ValueError("manifest must declare model-specific rates and trace paths")
    fitted, heads = {}, {}
    for model, entry in manifest["models"].items():
        speed = _positive(entry["service_rate_qps"], "service_rate_qps")
        if not entry.get("service_rate_definition"):
            raise ValueError("each fresh service rate needs its measurement definition")
        if not entry.get("traces"):
            raise ValueError("each model requires explicit trace paths")
        paths = []
        for raw in entry["traces"]:
            path = Path(raw).expanduser()
            paths.append((path if path.is_absolute() else manifest_path.parent / path).resolve())
        rows, provenance = load_trace_rows(paths)
        prefill = fit_prefill(rows, chunk_tokens=chunk_tokens, min_rows=min_rows)
        head, tpot = fit_tpot(rows, seed=seed, min_rows=min_rows, n_estimators=n_estimators)
        fitted[model] = {"service_rate_qps": speed,
                         "service_rate_definition": entry["service_rate_definition"],
                         "prefill": prefill, "tpot": tpot, "trace_provenance": provenance}
        heads[model] = head
    # No output is created until every model has adequate calibration rows.
    output_dir.mkdir(parents=True)
    for index, (model, artifact) in enumerate(fitted.items()):
        model_file = f"tpot_{index}.json"
        heads[model].save_model(output_dir / model_file)
        artifact["tpot"].update(model_file=model_file,
                                model_sha256=file_sha256(output_dir / model_file))
    result = {
        "schema_version": CALIBRATION_SCHEMA_VERSION, "data_role": "calibration",
        "serving_profile_verified": True, "serving_profile": profile,
        "serving_profile_sha256": serving_profile_sha256(profile),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(manifest_path), "source_manifest_sha256": file_sha256(manifest_path),
        "models": fitted,
    }
    artifact_path = output_dir / "methodology_calibration.json"
    with artifact_path.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return artifact_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-rows", type=int, default=40)
    args = parser.parse_args()
    print(fit_manifest(args.manifest, args.output_dir, seed=args.seed, min_rows=args.min_rows))


if __name__ == "__main__":
    main()

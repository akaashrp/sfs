from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import HuberRegressor, LinearRegression


BATCH_FIT_REQUIRED_COLS = [
    "ts",
    "engine",
    "prefill",
    "prefill_sq_sum",
    "decode",
    "decode_sq_sum",
    "total",
    "sched",
    "exec",
    "interval",
    "num_seqs",
    "sum_tokens",
    "sum_sq_tokens",
    "avg_tokens",
    "max_tokens",
]
BATCH_FIT_OPTIONAL_COLS = [
    "prefill_x_processed_ctx_sum",
]
DEFAULT_COLS = BATCH_FIT_REQUIRED_COLS + BATCH_FIT_OPTIONAL_COLS

FEATURE_SET_LEGACY = "legacy"
FEATURE_SET_CROSS_TERM = "cross_term"
DEFAULT_FEATURE_SET = FEATURE_SET_LEGACY
FEATURE_SET_CHOICES = (FEATURE_SET_LEGACY, FEATURE_SET_CROSS_TERM)
FEATURE_NAMES_BY_SET = {
    FEATURE_SET_LEGACY: ["p", "d", "s", "p_sq_sum", "s_sq"],
    FEATURE_SET_CROSS_TERM: ["p", "d", "s", "p_sq_sum", "p_x_ctx"],
}


@dataclass
class TwoPartFitResult:
    robust_model: HuberRegressor
    base_model: LinearRegression
    feature_set: str
    feature_names: list[str]
    coefficient_constraint: str
    inlier_mask: np.ndarray
    residual_threshold: float
    stall_probability: float
    mean_stall_delay: float

    def predict_typical(self, x: np.ndarray) -> np.ndarray:
        return self.base_model.predict(x)

    def predict_expected(self, x: np.ndarray) -> np.ndarray:
        return self.predict_typical(x) + self.stall_probability * self.mean_stall_delay


def _normalize_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        candidate = Path("/") / path
        if candidate.exists():
            path = candidate
        else:
            raise ValueError(f"Expected absolute path for file, got '{path_like}'.")
    return path


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    feature_set: str = DEFAULT_FEATURE_SET,
) -> tuple[np.ndarray, list[str]]:
    required_base_cols = ("prefill", "decode", "sum_tokens", "prefill_sq_sum")
    missing_base_cols = [col for col in required_base_cols if col not in df.columns]
    if missing_base_cols:
        raise ValueError(
            "Batch-fit dataframe is missing required columns: "
            + ", ".join(missing_base_cols)
        )

    prefill_tokens = df["prefill"].to_numpy()
    decode_tokens = df["decode"].to_numpy()

    if feature_set == FEATURE_SET_LEGACY:
        x = np.column_stack(
            [
                prefill_tokens,
                decode_tokens,
                df["sum_tokens"].to_numpy(),
                df["prefill_sq_sum"].to_numpy(),
                df["sum_sq_tokens"].to_numpy(),
            ]
        )
    elif feature_set == FEATURE_SET_CROSS_TERM:
        x = np.column_stack(
            [
                prefill_tokens,
                decode_tokens,
                df["sum_tokens"].to_numpy(),
                df["prefill_sq_sum"].to_numpy(),
                df["prefill_x_processed_ctx_sum"].to_numpy(),
            ]
        )
    else:
        raise ValueError(
            f"Unsupported resolved feature_set='{feature_set}'. "
            f"Choose one of: {', '.join(FEATURE_SET_CHOICES)}."
        )

    return x, FEATURE_NAMES_BY_SET[feature_set].copy()


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0
    return 1.0 - ss_res / ss_tot


def _fit_base_model(
    x: np.ndarray,
    y: np.ndarray,
    *,
    nonnegative_coefficients: bool,
) -> LinearRegression:
    if not nonnegative_coefficients:
        model = LinearRegression()
        model.fit(x, y)
        return model

    # Batch latency and every feature are nonnegative physical quantities.
    # Constrain both the intercept and slopes so the fitted latency cannot
    # become negative or decrease as work is added.
    design = np.column_stack([np.ones(len(x), dtype=x.dtype), x])
    parameters, _ = nnls(design, y)
    model = LinearRegression()
    model.intercept_ = float(parameters[0])
    model.coef_ = np.asarray(parameters[1:], dtype=float)
    model.n_features_in_ = int(x.shape[1])
    return model


def fit_two_part_from_df(
    df: pd.DataFrame,
    *,
    stall_percentile: float = 99.9,
    huber_epsilon: float = 1.35,
    feature_set: str = DEFAULT_FEATURE_SET,
    nonnegative_coefficients: bool = False,
) -> TwoPartFitResult:
    if not 0.0 < stall_percentile < 100.0:
        raise ValueError("stall_percentile must be in (0, 100).")

    x, feature_names = build_feature_matrix(df, feature_set=feature_set)
    y = df["exec"].to_numpy()

    robust_model = HuberRegressor(epsilon=huber_epsilon, max_iter=1000)
    robust_model.fit(x, y)

    robust_residuals = y - robust_model.predict(x)
    residual_threshold = np.percentile(robust_residuals, stall_percentile)
    inlier_mask = robust_residuals <= residual_threshold

    if inlier_mask.sum() <= x.shape[1]:
        raise ValueError(
            "Too few inliers after residual filtering. "
            "Lower stall_percentile or inspect data quality."
        )

    base_model = _fit_base_model(
        x[inlier_mask],
        y[inlier_mask],
        nonnegative_coefficients=nonnegative_coefficients,
    )

    typical_pred = base_model.predict(x)
    stall_mask = robust_residuals > residual_threshold
    stall_delay = np.maximum(0.0, y - typical_pred)
    stall_probability = float(np.mean(stall_mask))
    mean_stall_delay = float(stall_delay[stall_mask].mean()) if stall_mask.any() else 0.0

    return TwoPartFitResult(
        robust_model=robust_model,
        base_model=base_model,
        feature_set=feature_set,
        feature_names=feature_names,
        coefficient_constraint=(
            "nonnegative_intercept_and_slopes"
            if nonnegative_coefficients
            else "unconstrained"
        ),
        inlier_mask=inlier_mask,
        residual_threshold=float(residual_threshold),
        stall_probability=stall_probability,
        mean_stall_delay=mean_stall_delay,
    )


def fit_two_part(
    path: str | Path,
    *,
    cols: Iterable[str] = DEFAULT_COLS,
    stall_percentile: float = 99.99,
    huber_epsilon: float = 1.35,
    feature_set: str = DEFAULT_FEATURE_SET,
    start_offset: int = 0,
    nonnegative_coefficients: bool = False,
) -> tuple[TwoPartFitResult, pd.DataFrame]:
    path = _normalize_path(path)
    if start_offset < 0:
        raise ValueError("start_offset must be nonnegative.")
    if start_offset == 0:
        df = pd.read_csv(path, header=0)
    else:
        with path.open("rb") as src:
            header = src.readline().decode("utf-8")
            src.seek(start_offset)
            rows = src.read().decode("utf-8")
        if not header.strip():
            raise ValueError(f"Batch-stats file has no header: {path}")
        if not rows.strip():
            raise ValueError(
                f"Batch-stats file has no rows after start_offset={start_offset}: "
                f"{path}"
            )
        df = pd.read_csv(io.StringIO(header + rows), header=0)
    result = fit_two_part_from_df(
        df,
        stall_percentile=stall_percentile,
        huber_epsilon=huber_epsilon,
        feature_set=feature_set,
        nonnegative_coefficients=nonnegative_coefficients,
    )
    return result, df


def summarize_two_part_fit(result: TwoPartFitResult, df: pd.DataFrame) -> str:
    x, _ = build_feature_matrix(df, feature_set=result.feature_set)
    y = df["exec"].to_numpy()

    typical_pred = result.predict_typical(x)
    expected_pred = result.predict_expected(x)
    inliers = result.inlier_mask

    lines = [
        "Two-part fit summary:",
        f"feature_set={result.feature_set}",
        f"coefficient_constraint={result.coefficient_constraint}",
        f"stall_residual_threshold={result.residual_threshold}",
        f"stall_probability={result.stall_probability}",
        f"mean_stall_delay={result.mean_stall_delay}",
        f"R^2_typical_all_rows={_r2(y, typical_pred)}",
        f"R^2_typical_inliers={_r2(y[inliers], typical_pred[inliers])}",
        f"R^2_expected_all_rows={_r2(y, expected_pred)}",
        "Base-model coefficients:",
    ]

    for name, coef in zip(result.feature_names, result.base_model.coef_):
        lines.append(f"beta_{name}={coef}")
    lines.append(f"intercept={result.base_model.intercept_}")
    return "\n".join(lines) + "\n"


def run_two_part_fit(
    output_prefix: str,
    path: str | Path,
    *,
    cols: Iterable[str] = DEFAULT_COLS,
    stall_percentile: float = 99.9,
    huber_epsilon: float = 1.35,
    feature_set: str = DEFAULT_FEATURE_SET,
    nonnegative_coefficients: bool = False,
) -> str:
    result, df = fit_two_part(
        path,
        cols=cols,
        stall_percentile=stall_percentile,
        huber_epsilon=huber_epsilon,
        feature_set=feature_set,
        nonnegative_coefficients=nonnegative_coefficients,
    )
    out = summarize_two_part_fit(result, df)
    out_path = f"{output_prefix}_two_part_coefficients.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out)
    return out


if __name__ == "__main__":
    # path = Path("experiments/metrics/batch_stats_qwen3-0_6b.csv")
    # print(run_two_part_fit("regression_qwen3_0.6b_final", path))
    # path = Path("experiments/metrics/batch_stats_qwen3-8b.csv")
    # print(run_two_part_fit("regression_qwen3_8b_final", path))
    # path = Path("experiments/metrics/batch_stats_qwen3-32b.csv")
    # print(run_two_part_fit("regression_qwen3_32b_final", path))
    
    path = Path("experiments/batch_stats_qwen3-0_6b_38075153_2026-03-20_075741.csv")
    print(run_two_part_fit("regression_qwen3_0.6b_test_router_experiment", path))

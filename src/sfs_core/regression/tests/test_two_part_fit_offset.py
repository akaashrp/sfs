from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sfs_core.regression.two_part_fit import (
    DEFAULT_COLS,
    fit_two_part,
    fit_two_part_from_df,
)


def _row(index: int, *, exec_s: float) -> dict[str, float | int]:
    prefill = index + 1
    decode = 2 * index + 1
    total = prefill + decode
    return {
        "ts": float(index),
        "engine": 0,
        "prefill": prefill,
        "prefill_sq_sum": prefill**2,
        "decode": decode,
        "decode_sq_sum": decode**2,
        "total": total,
        "sched": 0.0001,
        "exec": exec_s,
        "interval": exec_s,
        "num_seqs": 1,
        "sum_tokens": total,
        "sum_sq_tokens": total**2,
        "avg_tokens": total,
        "max_tokens": total,
        "prefill_x_processed_ctx_sum": prefill * total,
    }


def test_fit_two_part_start_offset_excludes_prior_rows(tmp_path: Path):
    path = tmp_path / "batch_stats.csv"
    prior = pd.DataFrame(
        [_row(index, exec_s=100.0 + index) for index in range(8)],
        columns=DEFAULT_COLS,
    )
    prior.to_csv(path, index=False)
    start_offset = path.stat().st_size

    current = pd.DataFrame(
        [
            _row(index, exec_s=0.01 + index * 0.001)
            for index in range(8, 28)
        ],
        columns=DEFAULT_COLS,
    )
    current.to_csv(path, mode="a", header=False, index=False)

    _, fitted_df = fit_two_part(
        path,
        stall_percentile=95.0,
        start_offset=start_offset,
    )

    assert len(fitted_df) == len(current)
    assert fitted_df["ts"].tolist() == current["ts"].tolist()
    assert float(fitted_df["exec"].max()) < 1.0


def test_nonnegative_fit_cannot_predict_negative_or_decreasing_latency():
    rng = np.random.default_rng(7)
    rows = []
    for index in range(200):
        prefill = int(rng.integers(0, 512))
        decode = int(rng.integers(1, 65))
        sum_tokens = int(rng.integers(decode, 4096))
        prefill_sq_sum = prefill**2
        cross_term = prefill * max(sum_tokens - prefill, 0)
        exec_s = max(
            0.001,
            0.002
            + 2e-6 * prefill
            + 3e-4 * decode
            + 1e-7 * sum_tokens
            + rng.normal(0, 0.001),
        )
        row = _row(index, exec_s=exec_s)
        row.update(
            {
                "prefill": prefill,
                "prefill_sq_sum": prefill_sq_sum,
                "decode": decode,
                "sum_tokens": sum_tokens,
                "sum_sq_tokens": sum_tokens**2,
                "prefill_x_processed_ctx_sum": cross_term,
            }
        )
        rows.append(row)

    fit = fit_two_part_from_df(
        pd.DataFrame(rows, columns=DEFAULT_COLS),
        feature_set="cross_term",
        nonnegative_coefficients=True,
    )

    assert fit.coefficient_constraint == "nonnegative_intercept_and_slopes"
    assert fit.base_model.intercept_ >= 0
    assert np.all(fit.base_model.coef_ >= 0)
    assert np.all(fit.predict_typical(np.zeros((1, 5))) >= 0)

from __future__ import annotations

import pytest

from sfs_core.shared.trace_theta import (
    estimate_score_proxy_metrics_from_batch_stats,
)


def test_score_proxy_calibration_uses_only_pure_decode_rows(tmp_path):
    path = tmp_path / "batch_stats.csv"
    path.write_text(
        "\n".join(
            [
                (
                    "ts,engine,prefill,prefill_sq_sum,decode,decode_sq_sum,"
                    "total,sched,exec,interval,num_seqs,sum_tokens,"
                    "sum_sq_tokens,avg_tokens,max_tokens,"
                    "prefill_x_processed_ctx_sum"
                ),
                "1,0,0,0,10,10,10,0.001,0.010,0.011,10,100,1000,10,10,0",
                "2,0,0,0,20,20,20,0.001,0.020,0.021,20,200,2000,10,10,0",
                # A mixed iteration must not contaminate decode calibration.
                "3,0,100,10000,100,10000,200,0.001,1.000,1.001,100,1,1,1,1,0",
                "malformed,row",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = estimate_score_proxy_metrics_from_batch_stats(
        batch_stats_csv_path=path
    )

    assert metrics["decode_tps"] == pytest.approx(1000.0)
    assert metrics["mean_decode_batch_ms"] == pytest.approx(15.0)
    assert metrics["decode_batch_stats_rows_used"] == 2
    assert metrics["decode_tokens_used"] == 30
    assert metrics["mixed_prefill_decode_rows_ignored"] == 1
    assert metrics["malformed_rows_ignored"] == 1


def test_score_proxy_calibration_rejects_mixed_only_trace(tmp_path):
    path = tmp_path / "batch_stats.csv"
    path.write_text(
        "ts,engine,prefill,prefill_sq_sum,decode,decode_sq_sum,total,sched,"
        "exec,interval,num_seqs,sum_tokens,sum_sq_tokens,avg_tokens,max_tokens,"
        "prefill_x_processed_ctx_sum\n"
        "1,0,100,10000,10,100,110,0.001,0.010,0.011,10,100,1000,10,10,0\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="No usable pure-decode"):
        estimate_score_proxy_metrics_from_batch_stats(
            batch_stats_csv_path=path
        )

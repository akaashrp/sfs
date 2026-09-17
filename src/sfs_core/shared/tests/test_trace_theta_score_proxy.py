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


_HEADER = (
    "ts,engine,prefill,prefill_sq_sum,decode,decode_sq_sum,total,sched,exec,"
    "interval,num_seqs,sum_tokens,sum_sq_tokens,avg_tokens,max_tokens,"
    "prefill_x_processed_ctx_sum"
)


def _decode_row(ts, num_seqs, exec_s):
    return (
        f"{ts},0,0,0,{num_seqs},{num_seqs},{num_seqs},0.001,{exec_s},"
        f"{exec_s},{num_seqs},{num_seqs * 1000},0,1000,1000,0"
    )


def _loaded_trace(path, *, runaway_rows):
    """Loaded phase: prefill burst, batched decode draining to one sequence.

    ``runaway_rows`` single-sequence iterations model one output that runs to
    the token cap after every other sequence has finished.
    """
    lines = [_HEADER, "0,0,4096,0,0,0,4096,0.001,0.200,0.2,128,4096,0,32,32,0"]
    ts = 1
    for num_seqs in (128,) * 40 + (64,) * 20 + (8,) * 10 + (2,) * 5:
        lines.append(_decode_row(ts, num_seqs, 0.025 if num_seqs > 8 else 0.010))
        ts += 1
    for _ in range(runaway_rows):
        lines.append(_decode_row(ts, 1, 0.007))
        ts += 1
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_multi_sequence_window_removes_single_sequence_runaway_tail(tmp_path):
    from sfs_core.shared.trace_theta import SCORE_PROXY_MULTI_SEQUENCE_PURE_DECODE

    healthy = _loaded_trace(tmp_path / "healthy.csv", runaway_rows=3)
    runaway = _loaded_trace(tmp_path / "runaway.csv", runaway_rows=600)

    current_healthy = estimate_score_proxy_metrics_from_batch_stats(
        batch_stats_csv_path=healthy
    )
    current_runaway = estimate_score_proxy_metrics_from_batch_stats(
        batch_stats_csv_path=runaway
    )
    # The historical summary is dominated by the lone runaway sequence.
    assert current_runaway["decode_batch_stats_rows_used"] == 675
    assert current_runaway["decode_tps"] < 0.5 * current_healthy["decode_tps"]
    assert (
        current_runaway["mean_decode_batch_ms"]
        < 0.5 * current_healthy["mean_decode_batch_ms"]
    )
    assert "decode_window_rule" not in current_runaway

    windowed = {
        name: estimate_score_proxy_metrics_from_batch_stats(
            batch_stats_csv_path=path,
            decode_window=SCORE_PROXY_MULTI_SEQUENCE_PURE_DECODE,
        )
        for name, path in (("healthy", healthy), ("runaway", runaway))
    }
    # With single-sequence iterations excluded both traces summarize to the
    # same batched decode sample: 40x128 + 20x64 + 10x8 + 5x2 tokens over
    # 60x25 ms + 15x10 ms.
    for name, excluded in (("healthy", 3), ("runaway", 600)):
        metrics = windowed[name]
        assert metrics["decode_window_rule"] == "multi_sequence_pure_decode"
        assert metrics["decode_rows_excluded_by_window"] == excluded
        assert metrics["decode_batch_stats_rows_used"] == 75
        assert metrics["decode_tokens_used"] == 6490
        assert metrics["decode_tps"] == pytest.approx(6490 / 1.65)
        assert metrics["mean_decode_batch_ms"] == pytest.approx(1650 / 75)
        assert metrics["mixed_prefill_decode_rows_ignored"] == 0
    # A healthy trace moves only by its few natural single-sequence rows.
    assert windowed["healthy"]["decode_tps"] == pytest.approx(
        current_healthy["decode_tps"], rel=0.02
    )


def test_multi_sequence_window_rejects_single_sequence_only_trace(tmp_path):
    from sfs_core.shared.trace_theta import SCORE_PROXY_MULTI_SEQUENCE_PURE_DECODE

    path = tmp_path / "single.csv"
    path.write_text(
        "\n".join([_HEADER, _decode_row(1, 1, 0.007), _decode_row(2, 1, 0.007)])
        + "\n",
        encoding="utf-8",
    )
    assert estimate_score_proxy_metrics_from_batch_stats(batch_stats_csv_path=path)[
        "decode_batch_stats_rows_used"
    ] == 2
    with pytest.raises(RuntimeError, match="multi_sequence_pure_decode"):
        estimate_score_proxy_metrics_from_batch_stats(
            batch_stats_csv_path=path,
            decode_window=SCORE_PROXY_MULTI_SEQUENCE_PURE_DECODE,
        )
    with pytest.raises(ValueError, match="Unknown SCORE-proxy decode window"):
        estimate_score_proxy_metrics_from_batch_stats(
            batch_stats_csv_path=path, decode_window="tail"
        )

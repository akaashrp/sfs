from __future__ import annotations

import pytest

from sfs_core.routing.score_proxy import (
    estimate_ttft_ms,
    hard_slo_candidate_value,
)


def test_score_proxy_ttft_terms_are_additive():
    estimate_ms, terms = estimate_ttft_ms(
        prompt_tokens=20,
        prefill_backlog_tokens=100,
        decode_backlog_tokens=40,
        prefill_tps=2000.0,
        decode_tps=1000.0,
        mean_decode_batch_ms=5.0,
    )

    assert terms == {
        "total_prefill_tokens": 120.0,
        "prefill_ms": 60.0,
        "decode_interference_ms": 40.0,
        "mean_decode_batch_ms": 5.0,
    }
    assert estimate_ms == pytest.approx(105.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prefill_backlog_tokens", -1.0),
        ("decode_backlog_tokens", -1.0),
        ("prefill_tps", 0.0),
        ("decode_tps", 0.0),
        ("mean_decode_batch_ms", 0.0),
    ],
)
def test_score_proxy_ttft_rejects_invalid_inputs(field, value):
    kwargs = {
        "prompt_tokens": 20,
        "prefill_backlog_tokens": 100.0,
        "decode_backlog_tokens": 40.0,
        "prefill_tps": 2000.0,
        "decode_tps": 1000.0,
        "mean_decode_batch_ms": 5.0,
    }
    kwargs[field] = value

    with pytest.raises(ValueError):
        estimate_ttft_ms(**kwargs)


def test_hard_slo_value_masks_infeasible_quality_upgrade():
    waits = {"small": 90.0, "large": 110.0}

    small = hard_slo_candidate_value(
        instance_id="small",
        wait_ms_by_instance=waits,
        slo_ms=100.0,
        predicted_quality=0.5,
        predicted_cost=10.0,
        lambda_weight=0.01,
    )
    large = hard_slo_candidate_value(
        instance_id="large",
        wait_ms_by_instance=waits,
        slo_ms=100.0,
        predicted_quality=0.99,
        predicted_cost=1.0,
        lambda_weight=0.01,
    )

    assert small == pytest.approx(0.4)
    assert large == float("-inf")


def test_hard_slo_value_falls_back_to_minimum_wait_when_none_feasible():
    waits = {"small": 90.0, "large": 110.0}

    assert hard_slo_candidate_value(
        instance_id="small",
        wait_ms_by_instance=waits,
        slo_ms=10.0,
        predicted_quality=0.0,
        predicted_cost=999.0,
        lambda_weight=999.0,
    ) == -90.0
    assert hard_slo_candidate_value(
        instance_id="large",
        wait_ms_by_instance=waits,
        slo_ms=10.0,
        predicted_quality=1.0,
        predicted_cost=0.0,
        lambda_weight=0.0,
    ) == -110.0

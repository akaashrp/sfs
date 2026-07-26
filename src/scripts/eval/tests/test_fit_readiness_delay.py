from __future__ import annotations

import pytest

from scripts.eval.fit_readiness_delay import (
    ReadinessSample,
    build_fit_payload,
)


def _samples(run_name: str, offset_ms: float):
    samples = []
    prior_ready = []
    rows = (
        (0.000, 100),
        (0.001, 300),
        (0.002, 100),
        (0.030, 400),
        (0.031, 200),
    )
    for index, (estimate_timestamp_s, prompt_tokens) in enumerate(rows):
        unready_count = sum(
            ready_at_s > estimate_timestamp_s
            for ready_at_s in prior_ready
        )
        delay_ms = (
            offset_ms
            + 0.01 * prompt_tokens
            + 4.0 * unready_count
        )
        actual_ready_at_s = estimate_timestamp_s + delay_ms / 1000.0
        prior_ready.append(actual_ready_at_s)
        samples.append(
            ReadinessSample(
                run_name=run_name,
                request_id=f"{run_name}-{index}",
                prompt_tokens=prompt_tokens,
                pending_dispatch_count=unready_count,
                delay_ms=delay_ms,
                estimate_timestamp_s=estimate_timestamp_s,
                actual_ready_at_s=actual_ready_at_s,
            )
        )
    return samples


def test_fit_selects_prompt_and_pending_model():
    payload = build_fit_payload(
        {
            "run-a": _samples("run-a", 2.0),
            "run-b": _samples("run-b", 2.0),
            "run-c": _samples("run-c", 2.0),
        }
    )

    assert payload["selected_model"] == "prompt_tokens_unready"
    coefficients = payload["coefficients"]
    assert coefficients["intercept_ms"] == pytest.approx(2.0)
    assert coefficients["prompt_token_ms"] == pytest.approx(0.01)
    assert coefficients["pending_dispatch_ms"] == pytest.approx(4.0)

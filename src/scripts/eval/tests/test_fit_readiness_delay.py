from __future__ import annotations

import pytest

from scripts.eval.fit_readiness_delay import (
    ReadinessSample,
    build_fit_payload,
)


def _samples(run_name: str, offset_ms: float):
    return [
        ReadinessSample(
            run_name=run_name,
            request_id=f"{run_name}-{index}",
            prompt_tokens=prompt_tokens,
            pending_dispatch_count=pending_count,
            delay_ms=(
                offset_ms
                + 0.01 * prompt_tokens
                + 4.0 * pending_count
            ),
        )
        for index, (prompt_tokens, pending_count) in enumerate(
            ((100, 0), (300, 0), (100, 2), (400, 1), (200, 3))
        )
    ]


def test_fit_selects_prompt_and_pending_model():
    payload = build_fit_payload(
        {
            "run-a": _samples("run-a", 2.0),
            "run-b": _samples("run-b", 2.0),
            "run-c": _samples("run-c", 2.0),
        }
    )

    assert payload["selected_model"] == "prompt_tokens_pending"
    coefficients = payload["coefficients"]
    assert coefficients["intercept_ms"] == pytest.approx(2.0)
    assert coefficients["prompt_token_ms"] == pytest.approx(0.01)
    assert coefficients["pending_dispatch_ms"] == pytest.approx(4.0)

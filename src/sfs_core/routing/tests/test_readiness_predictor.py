from __future__ import annotations

import pytest

from sfs_core.routing.readiness_predictor import ReadinessDelayPredictor


def test_readiness_predictor_uses_only_router_visible_inputs():
    predictor = ReadinessDelayPredictor.from_mapping(
        {
            "coefficients": {
                "intercept_ms": 2.0,
                "prompt_token_ms": 0.01,
                "pending_dispatch_ms": 3.0,
            }
        }
    )

    assert predictor.predict_ms(
        prompt_tokens=100,
        pending_dispatch_count=2,
    ) == pytest.approx(9.0)


@pytest.mark.parametrize(
    "coefficient",
    (-1.0, float("inf"), float("nan")),
)
def test_readiness_predictor_rejects_invalid_coefficients(coefficient: float):
    with pytest.raises(ValueError):
        ReadinessDelayPredictor.from_mapping(
            {
                "coefficients": {
                    "intercept_ms": coefficient,
                    "prompt_token_ms": 0.0,
                    "pending_dispatch_ms": 0.0,
                }
            }
        )

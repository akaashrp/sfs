"""Router-to-EngineCore readiness-delay prediction."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ReadinessDelayPredictor:
    """Small linear predictor evaluated once per candidate instance."""

    intercept_ms: float
    prompt_token_ms: float
    pending_dispatch_ms: float

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> "ReadinessDelayPredictor":
        coefficients = payload.get("coefficients")
        if not isinstance(coefficients, Mapping):
            raise ValueError("Readiness predictor is missing coefficients")
        predictor = cls(
            intercept_ms=float(coefficients["intercept_ms"]),
            prompt_token_ms=float(coefficients["prompt_token_ms"]),
            pending_dispatch_ms=float(coefficients["pending_dispatch_ms"]),
        )
        for name, value in (
            ("intercept_ms", predictor.intercept_ms),
            ("prompt_token_ms", predictor.prompt_token_ms),
            ("pending_dispatch_ms", predictor.pending_dispatch_ms),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"Readiness predictor coefficient {name} must be finite "
                    "and nonnegative"
                )
        return predictor

    @classmethod
    def load(cls, path: str | Path) -> "ReadinessDelayPredictor":
        resolved = Path(path)
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("Readiness predictor file must contain an object")
        return cls.from_mapping(payload)

    def predict_ms(
        self,
        *,
        prompt_tokens: int,
        pending_dispatch_count: int,
    ) -> float:
        return max(
            0.0,
            self.intercept_ms
            + self.prompt_token_ms * max(0, int(prompt_tokens))
            + self.pending_dispatch_ms
            * max(0, int(pending_dispatch_count)),
        )

"""Pure SCORE-inspired proxy math shared by routing code and unit tests."""

from __future__ import annotations

import math
from typing import Mapping, Optional


def estimate_ttft_ms(
    *,
    prompt_tokens: int,
    prefill_backlog_tokens: float,
    decode_backlog_tokens: float,
    prefill_tps: float,
    decode_tps: float,
    mean_decode_batch_ms: float,
) -> tuple[float, dict[str, float]]:
    values = {
        "prefill_backlog_tokens": float(prefill_backlog_tokens),
        "decode_backlog_tokens": float(decode_backlog_tokens),
        "prefill_tps": float(prefill_tps),
        "decode_tps": float(decode_tps),
        "mean_decode_batch_ms": float(mean_decode_batch_ms),
    }
    if int(prompt_tokens) < 0:
        raise ValueError("prompt_tokens must be nonnegative")
    for name in ("prefill_backlog_tokens", "decode_backlog_tokens"):
        if not math.isfinite(values[name]) or values[name] < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    for name in ("prefill_tps", "decode_tps", "mean_decode_batch_ms"):
        if not math.isfinite(values[name]) or values[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")

    total_prefill_tokens = values["prefill_backlog_tokens"] + int(
        prompt_tokens
    )
    prefill_ms = total_prefill_tokens / values["prefill_tps"] * 1000.0
    decode_interference_ms = (
        values["decode_backlog_tokens"] / values["decode_tps"] * 1000.0
    )
    ttft_ms = (
        prefill_ms
        + decode_interference_ms
        + values["mean_decode_batch_ms"]
    )
    return ttft_ms, {
        "total_prefill_tokens": total_prefill_tokens,
        "prefill_ms": prefill_ms,
        "decode_interference_ms": decode_interference_ms,
        "mean_decode_batch_ms": values["mean_decode_batch_ms"],
    }


def hard_slo_candidate_value(
    *,
    instance_id: str,
    wait_ms_by_instance: Mapping[str, float],
    slo_ms: Optional[float],
    predicted_quality: float,
    predicted_cost: float,
    lambda_weight: float,
) -> float:
    """Return the common hard-SLO value used by SFS and all hard baselines."""
    if instance_id not in wait_ms_by_instance:
        raise KeyError(instance_id)
    if slo_ms is None:
        feasible = True
        any_feasible = True
    else:
        any_feasible = any(
            float(wait_ms) <= float(slo_ms)
            for wait_ms in wait_ms_by_instance.values()
        )
        feasible = (
            float(wait_ms_by_instance[instance_id]) <= float(slo_ms)
        )
    if any_feasible:
        if not feasible:
            return float("-inf")
        return float(predicted_quality) - (
            float(lambda_weight) * float(predicted_cost)
        )
    return -float(wait_ms_by_instance[instance_id])

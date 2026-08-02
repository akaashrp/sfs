"""Published SCORE routing objective with explicit serving-time adaptations.

The SCORE paper defines the online model choice as::

    argmax_i Q_i - lambda * (w_C * C_i + w_L * L_i)

where ``C_i = c_i * predicted_output_tokens`` and
``L_i = W_i + s_i * predicted_output_tokens``.  The paper does not specify
how to update ``lambda`` online.  This module therefore implements the
published fixed-lambda decision rule and retains the cumulative cost-budget
residual for auditability; the residual is candidate-invariant apart from the
current request cost and does not introduce an undocumented controller.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional


def _finite_nonnegative(value: float, *, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return numeric


def _finite_positive(value: float, *, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric


def estimate_total_response_latency_ms(
    *,
    prompt_tokens: int,
    predicted_output_tokens: float,
    prefill_backlog_tokens: float,
    decode_backlog_tokens: float,
    prefill_tps: float,
    decode_tps: float,
    mean_decode_batch_ms: float,
) -> tuple[float, dict[str, float]]:
    """Estimate SCORE's ``W_i + s_i * predicted_output_tokens``.

    SCORE leaves ``W_i`` unspecified for a continuous-batching engine.  The
    serving adaptation used here maps the effective prefill and decode
    backlogs to time with the matching V100 service rates.  A pure-decode
    batch advances each active request by one token, so the calibrated mean
    decode-batch latency is the per-response-token ``s_i`` term.
    """

    if isinstance(prompt_tokens, bool) or int(prompt_tokens) < 0:
        raise ValueError("prompt_tokens must be nonnegative")
    prompt_tokens_value = int(prompt_tokens)
    output_tokens = _finite_nonnegative(
        predicted_output_tokens,
        name="predicted_output_tokens",
    )
    prefill_backlog = _finite_nonnegative(
        prefill_backlog_tokens,
        name="prefill_backlog_tokens",
    )
    decode_backlog = _finite_nonnegative(
        decode_backlog_tokens,
        name="decode_backlog_tokens",
    )
    prefill_rate = _finite_positive(prefill_tps, name="prefill_tps")
    decode_rate = _finite_positive(decode_tps, name="decode_tps")
    decode_step_ms = _finite_positive(
        mean_decode_batch_ms,
        name="mean_decode_batch_ms",
    )

    total_prefill_tokens = prefill_backlog + prompt_tokens_value
    prefill_wait_ms = total_prefill_tokens / prefill_rate * 1000.0
    decode_backlog_wait_ms = decode_backlog / decode_rate * 1000.0
    waiting_time_ms = prefill_wait_ms + decode_backlog_wait_ms
    predicted_runtime_ms = output_tokens * decode_step_ms
    total_latency_ms = waiting_time_ms + predicted_runtime_ms

    return total_latency_ms, {
        "total_prefill_tokens": total_prefill_tokens,
        "prefill_wait_ms": prefill_wait_ms,
        "decode_backlog_wait_ms": decode_backlog_wait_ms,
        "waiting_time_ms": waiting_time_ms,
        "predicted_runtime_ms": predicted_runtime_ms,
        "predicted_total_latency_ms": total_latency_ms,
    }


@dataclass(slots=True)
class ScorePolicyState:
    """Online state needed to retain SCORE's cumulative cost expression."""

    total_requests: int
    total_cost_budget: Optional[float] = None
    cumulative_predicted_cost: float = 0.0
    routed_requests: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.total_requests, bool) or int(self.total_requests) <= 0:
            raise ValueError("total_requests must be positive")
        self.total_requests = int(self.total_requests)
        if self.total_cost_budget is not None:
            self.total_cost_budget = _finite_nonnegative(
                self.total_cost_budget,
                name="total_cost_budget",
            )

    @property
    def next_request_index(self) -> int:
        return self.routed_requests + 1

    def prorated_budget_for_next_request(self) -> Optional[float]:
        if self.total_cost_budget is None:
            return None
        fraction = min(self.next_request_index, self.total_requests) / float(
            self.total_requests
        )
        return float(self.total_cost_budget * fraction)

    def record_selection(self, predicted_cost: float) -> None:
        selected_cost = _finite_nonnegative(
            predicted_cost,
            name="selected_predicted_cost",
        )
        self.cumulative_predicted_cost += selected_cost
        self.routed_requests += 1

    def as_dict(self) -> dict[str, float | int | None | str]:
        final_budget_residual = (
            self.cumulative_predicted_cost - self.total_cost_budget
            if self.total_cost_budget is not None
            else None
        )
        return {
            "total_requests": self.total_requests,
            "routed_requests": self.routed_requests,
            "total_cost_budget": self.total_cost_budget,
            "cumulative_predicted_cost": self.cumulative_predicted_cost,
            "final_cost_budget_residual": final_budget_residual,
            "budget_control": "fixed_lambda_published_argmax",
        }


def score_candidate_terms(
    *,
    predicted_quality: float,
    predicted_response_cost: float,
    predicted_total_latency_ms: float,
    latency_limit_ms: float,
    lambda_weight: float,
    cost_weight: float,
    latency_weight: float,
    state: ScorePolicyState,
) -> dict[str, float | int | None]:
    """Return the expanded SCORE Lagrangian terms for one candidate."""

    quality = float(predicted_quality)
    if not math.isfinite(quality):
        raise ValueError("predicted_quality must be finite")
    cost = _finite_nonnegative(
        predicted_response_cost,
        name="predicted_response_cost",
    )
    latency_ms = _finite_nonnegative(
        predicted_total_latency_ms,
        name="predicted_total_latency_ms",
    )
    limit_ms = _finite_nonnegative(latency_limit_ms, name="latency_limit_ms")
    multiplier = _finite_nonnegative(lambda_weight, name="lambda_weight")
    w_cost = _finite_nonnegative(cost_weight, name="cost_weight")
    w_latency = _finite_nonnegative(latency_weight, name="latency_weight")

    projected_cumulative_cost = state.cumulative_predicted_cost + cost
    prorated_budget = state.prorated_budget_for_next_request()
    cost_constraint_residual = (
        projected_cumulative_cost - prorated_budget
        if prorated_budget is not None
        else cost
    )
    predicted_total_latency_s = latency_ms / 1000.0
    latency_limit_s = limit_ms / 1000.0
    latency_constraint_residual_s = predicted_total_latency_s - latency_limit_s
    weighted_constraint_residual = (
        w_cost * cost_constraint_residual + w_latency * latency_constraint_residual_s
    )
    candidate_value = quality - (multiplier * weighted_constraint_residual)

    return {
        "request_index": state.next_request_index,
        "cumulative_predicted_cost_before": state.cumulative_predicted_cost,
        "projected_cumulative_cost": projected_cumulative_cost,
        "prorated_cost_budget": prorated_budget,
        "cost_constraint_residual": cost_constraint_residual,
        "predicted_total_latency_ms": latency_ms,
        "predicted_total_latency_s": predicted_total_latency_s,
        "latency_limit_ms": limit_ms,
        "latency_limit_s": latency_limit_s,
        "latency_constraint_residual_s": latency_constraint_residual_s,
        "weighted_constraint_residual": weighted_constraint_residual,
        "score_candidate_value": candidate_value,
    }

from __future__ import annotations

import pytest

from sfs_core.routing.score_policy import (
    ScorePolicyState,
    estimate_total_response_latency_ms,
    score_candidate_terms,
)


def test_total_latency_is_wait_plus_predicted_response_runtime():
    total_ms, terms = estimate_total_response_latency_ms(
        prompt_tokens=20,
        predicted_output_tokens=10,
        prefill_backlog_tokens=100,
        decode_backlog_tokens=40,
        prefill_tps=2000.0,
        decode_tps=1000.0,
        mean_decode_batch_ms=5.0,
    )

    assert terms["prefill_wait_ms"] == pytest.approx(60.0)
    assert terms["decode_backlog_wait_ms"] == pytest.approx(40.0)
    assert terms["waiting_time_ms"] == pytest.approx(100.0)
    assert terms["predicted_runtime_ms"] == pytest.approx(50.0)
    assert total_ms == pytest.approx(150.0)


def test_published_score_matches_expanded_lagrangian():
    state = ScorePolicyState(
        total_requests=10,
        total_cost_budget=1000.0,
        cumulative_predicted_cost=80.0,
        routed_requests=1,
    )

    terms = score_candidate_terms(
        predicted_quality=0.8,
        predicted_response_cost=30.0,
        predicted_total_latency_ms=2500.0,
        latency_limit_ms=2000.0,
        lambda_weight=0.01,
        cost_weight=1.0,
        latency_weight=2.0,
        state=state,
    )

    # At t=2, the prorated budget is 200. Projected cost is 110, so the
    # cost residual is -90. Latency residual is 0.5 seconds.
    assert terms["prorated_cost_budget"] == pytest.approx(200.0)
    assert terms["cost_constraint_residual"] == pytest.approx(-90.0)
    assert terms["latency_constraint_residual_s"] == pytest.approx(0.5)
    assert terms["weighted_constraint_residual"] == pytest.approx(-89.0)
    assert terms["score_candidate_value"] == pytest.approx(1.69)


def test_cumulative_budget_offsets_do_not_change_fixed_lambda_ranking():
    def values(state: ScorePolicyState) -> tuple[float, float]:
        cheap = score_candidate_terms(
            predicted_quality=0.6,
            predicted_response_cost=10.0,
            predicted_total_latency_ms=1000.0,
            latency_limit_ms=500.0,
            lambda_weight=0.01,
            cost_weight=1.0,
            latency_weight=1.0,
            state=state,
        )
        expensive = score_candidate_terms(
            predicted_quality=0.9,
            predicted_response_cost=50.0,
            predicted_total_latency_ms=2000.0,
            latency_limit_ms=500.0,
            lambda_weight=0.01,
            cost_weight=1.0,
            latency_weight=1.0,
            state=state,
        )
        return (
            float(cheap["score_candidate_value"]),
            float(expensive["score_candidate_value"]),
        )

    under_budget = values(
        ScorePolicyState(
            total_requests=10,
            total_cost_budget=1000.0,
            cumulative_predicted_cost=0.0,
            routed_requests=0,
        )
    )
    over_budget = values(
        ScorePolicyState(
            total_requests=10,
            total_cost_budget=1.0,
            cumulative_predicted_cost=1000.0,
            routed_requests=5,
        )
    )

    assert (under_budget[0] > under_budget[1]) == (over_budget[0] > over_budget[1])
    assert (under_budget[0] - under_budget[1]) == pytest.approx(
        over_budget[0] - over_budget[1]
    )


def test_score_state_records_only_selected_predicted_cost():
    state = ScorePolicyState(total_requests=2, total_cost_budget=50.0)
    state.record_selection(12.5)

    assert state.routed_requests == 1
    assert state.cumulative_predicted_cost == pytest.approx(12.5)
    assert state.prorated_budget_for_next_request() == pytest.approx(50.0)
    assert state.as_dict()["final_cost_budget_residual"] == pytest.approx(-37.5)

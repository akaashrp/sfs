from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.runs import experiments
from sfs_core.routing.score_policy import ScorePolicyState
from sfs_core.routing.wait_time_scheduler import WaitTimeResult


def _snapshot_wait(
    instance_id: str,
    *,
    prefill_backlog_tokens: int,
    decode_backlog_tokens: int,
) -> WaitTimeResult:
    return WaitTimeResult(
        instance_id=instance_id,
        wait_ms=0.0,
        fetched_at_s=1.0,
        raw_payload={
            "reports": [
                {
                    "metadata": {
                        "simulation_mode": "snapshot_only",
                        "prefill_backlog_total_tokens": prefill_backlog_tokens,
                        "decode_backlog_total_tokens": decode_backlog_tokens,
                        "pending_overlay_count": 2,
                    }
                }
            ]
        },
    )


def _estimated_wait(
    instance_id: str,
    *,
    predicted_output_tokens: float,
) -> WaitTimeResult:
    source = _snapshot_wait(
        instance_id,
        prefill_backlog_tokens=100,
        decode_backlog_tokens=40,
    )
    diagnostics: dict[str, dict] = {}
    estimate_ms = experiments._wait_estimator_score_total_latency(
        instance_id,
        SimpleNamespace(last_wait=None),
        {instance_id: source},
        {
            "prompt_tokens": 20,
            "output_lengths": {instance_id: predicted_output_tokens},
            # Deliberately contradictory realized/oracle-looking data. The
            # estimator must use only output_lengths above.
            "realized_output_lengths": {instance_id: 999999.0},
            "prefill_tps_by_instance": {instance_id: 2000.0},
            "decode_tps_by_instance": {instance_id: 1000.0},
            "mean_decode_batch_ms_by_instance": {instance_id: 5.0},
            "wait_estimator_diagnostics_by_instance": diagnostics,
        },
    )
    payload = dict(source.raw_payload)
    payload["_wait_estimator_diagnostics"] = diagnostics[instance_id]
    return WaitTimeResult(
        instance_id=instance_id,
        wait_ms=float(estimate_ms),
        fetched_at_s=source.fetched_at_s,
        raw_payload=payload,
    )


def test_score_total_latency_uses_predicted_length_and_calibrated_step_time():
    result = _estimated_wait("vllm-8b", predicted_output_tokens=10.0)

    assert result.wait_ms == pytest.approx(150.0)
    diagnostics = result.raw_payload["_wait_estimator_diagnostics"]
    assert diagnostics["method"] == experiments.SCORE_TOTAL_LATENCY_ESTIMATOR_NAME
    assert diagnostics["published_score_formula"] is True
    assert diagnostics["estimated_waiting_time_ms"] == pytest.approx(100.0)
    assert diagnostics["estimated_runtime_ms"] == pytest.approx(50.0)


def test_score_uses_snapshot_summary_path_not_sfs_critical_path():
    assert experiments.CollectingWaitTimeScheduler._wait_fetch_args_for_estimator(
        experiments.SCORE_TOTAL_LATENCY_ESTIMATOR_NAME,
        prompt_tokens=4096,
    ) == (None, None)
    assert not experiments._should_include_unconditional_live_fetch(
        experiments.SCORE_TOTAL_LATENCY_ESTIMATOR_NAME
    )


def test_score_utility_uses_published_quality_cost_latency_argmax_without_gate():
    state = experiments.UtilityState(
        score_latency_limit_ms=100.0,
        score_policy_state=ScorePolicyState(
            total_requests=1,
            total_cost_budget=100.0,
        ),
    )
    utility = experiments.build_utility_fn(
        "score",
        lambda_weight=0.01,
        delta_weight=999.0,
        instance_costs={
            "fast": {"prompt": 1000.0, "output": 1.0},
            "slow": {"prompt": 1000.0, "output": 1.0},
        },
        utility_state=state,
        score_cost_weight=1.0,
        score_latency_weight=1.0,
    )
    waits = {
        "fast": WaitTimeResult("fast", 1000.0, 1.0, {}),
        "slow": WaitTimeResult("slow", 2000.0, 1.0, {}),
    }
    outputs = {"fast": 10.0, "slow": 10.0}
    quality = {"fast": 0.5, "slow": 0.9}

    fast_value = utility("fast", waits, quality, outputs, 100)
    slow_value = utility("slow", waits, quality, outputs, 100)

    # Prompt cost is intentionally ignored, matching C_i=c_i*S_hat_i.
    assert fast_value == pytest.approx(1.391)
    assert slow_value == pytest.approx(1.781)
    # Both violate the 100 ms limit; SCORE penalizes rather than hard-gates.
    assert slow_value > fast_value


def test_score_compact_terms_recompute_the_exact_routing_values():
    waits = {
        "small": _estimated_wait("small", predicted_output_tokens=10.0),
        "large": _estimated_wait("large", predicted_output_tokens=20.0),
    }
    state = ScorePolicyState(total_requests=2, total_cost_budget=100.0)
    terms = experiments._build_compact_published_score_candidate_terms(
        wait_results=waits,
        selected_instance_id="large",
        latency_limit_ms=100.0,
        accuracy_scores={"small": 0.5, "large": 0.9},
        output_lengths={"small": 10.0, "large": 20.0},
        instance_costs={
            "small": {"prompt": 999.0, "output": 1.0},
            "large": {"prompt": 999.0, "output": 1.0},
        },
        lambda_weight=0.01,
        cost_weight=1.0,
        latency_weight=1.0,
        state=state,
    )

    assert terms["selected_instance_id"] == "large"
    assert terms["published_fixed_lambda_argmax"] is True
    assert terms["candidates"]["small"]["predicted_response_cost"] == 10.0
    assert terms["candidates"]["large"]["predicted_runtime_ms"] == 100.0


def test_new_score_policy_does_not_replace_the_existing_proxy_mapping():
    proxy = experiments._baseline_runtime_params(
        "hard_score_proxy",
        lambda_weight=0.1,
        delta_weight=0.2,
    )
    score = experiments._baseline_runtime_params(
        "score",
        lambda_weight=0.1,
        delta_weight=0.2,
    )

    assert proxy[4] == experiments.SCORE_PROXY_TTFT_ESTIMATOR_NAME
    assert proxy[3] is experiments._wait_estimator_score_proxy_ttft
    assert score[4] == experiments.SCORE_TOTAL_LATENCY_ESTIMATOR_NAME
    assert score[3] is experiments._wait_estimator_score_total_latency

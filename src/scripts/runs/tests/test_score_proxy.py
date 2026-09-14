from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts.runs import experiments
from sfs_core.routing.wait_time_scheduler import WaitTimeResult


def test_calibrated_metrics_do_not_alias_ministral_8b_to_qwen(tmp_path):
    path = tmp_path/"metrics.json"
    path.write_text(json.dumps({
        "mistralai/Ministral-3-8B-Instruct-2512-BF16": {"prefill_tps": 1234},
        "qwen3-8b": {"prefill_tps": 5678},
    }))
    rows = experiments._load_calibrated_service_metrics(path)
    assert rows["ministral3-8b"]["prefill_tps"] == 1234
    assert rows["qwen3-8b"]["prefill_tps"] == 5678
    assert experiments._normalize_prefill_tps_key("vllm-ministral3-8b") == "ministral3-8b"


def _wait_result(
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


def test_score_proxy_computes_each_ttft_term_and_ignores_output_labels():
    instance_id = "vllm-8b"
    wait_result = _wait_result(
        instance_id,
        prefill_backlog_tokens=100,
        decode_backlog_tokens=40,
    )
    diagnostics: dict[str, dict] = {}
    context = {
        "prompt_tokens": 20,
        "prefill_tps_by_instance": {instance_id: 2000.0},
        "decode_tps_by_instance": {instance_id: 1000.0},
        "mean_decode_batch_ms_by_instance": {instance_id: 5.0},
        # The estimator must not use realized or oracle output lengths.
        "output_lengths": {instance_id: 999999.0},
        "wait_estimator_diagnostics_by_instance": diagnostics,
    }

    estimate_ms = experiments._wait_estimator_score_proxy_ttft(
        instance_id,
        SimpleNamespace(last_wait=None),
        {instance_id: wait_result},
        context,
    )

    assert estimate_ms == pytest.approx(105.0)
    details = diagnostics[instance_id]
    assert details["estimated_prefill_ms"] == pytest.approx(60.0)
    assert details["estimated_decode_interference_ms"] == pytest.approx(40.0)
    assert details["estimated_first_decode_batch_ms"] == pytest.approx(5.0)
    assert details["estimated_ttft_ms"] == pytest.approx(105.0)
    assert details["score_exact_reproduction"] is False
    assert details["details"]["pending_overlay_count"] == 2


def test_score_proxy_fails_closed_when_decode_backlog_is_missing():
    instance_id = "vllm-8b"
    wait_result = WaitTimeResult(
        instance_id=instance_id,
        wait_ms=0.0,
        fetched_at_s=1.0,
        raw_payload={
            "reports": [
                {"metadata": {"prefill_backlog_total_tokens": 100}}
            ]
        },
    )
    context = {
        "prompt_tokens": 20,
        "prefill_tps_by_instance": {instance_id: 2000.0},
        "decode_tps_by_instance": {instance_id: 1000.0},
        "mean_decode_batch_ms_by_instance": {instance_id: 5.0},
        "wait_estimator_diagnostics_by_instance": {},
    }

    with pytest.raises(RuntimeError, match="effective prefill and decode backlog"):
        experiments._wait_estimator_score_proxy_ttft(
            instance_id,
            SimpleNamespace(last_wait=None),
            {instance_id: wait_result},
            context,
        )


def _score_wait_result(
    instance_id: str,
    *,
    prompt_tokens: int,
    prefill_backlog_tokens: int,
    decode_backlog_tokens: int,
    prefill_tps: float,
    decode_tps: float,
    mean_decode_batch_ms: float,
) -> WaitTimeResult:
    source = _wait_result(
        instance_id,
        prefill_backlog_tokens=prefill_backlog_tokens,
        decode_backlog_tokens=decode_backlog_tokens,
    )
    diagnostics: dict[str, dict] = {}
    estimate_ms = experiments._wait_estimator_score_proxy_ttft(
        instance_id,
        SimpleNamespace(last_wait=None),
        {instance_id: source},
        {
            "prompt_tokens": prompt_tokens,
            "prefill_tps_by_instance": {instance_id: prefill_tps},
            "decode_tps_by_instance": {instance_id: decode_tps},
            "mean_decode_batch_ms_by_instance": {
                instance_id: mean_decode_batch_ms
            },
            "wait_estimator_diagnostics_by_instance": diagnostics,
        },
    )
    raw_payload = dict(source.raw_payload)
    raw_payload["_wait_estimator_diagnostics"] = diagnostics[instance_id]
    return WaitTimeResult(
        instance_id=instance_id,
        wait_ms=float(estimate_ms),
        fetched_at_s=source.fetched_at_s,
        raw_payload=raw_payload,
    )


def test_compact_score_candidate_logging_records_exact_terms_and_hard_gate():
    waits = {
        "small": _score_wait_result(
            "small",
            prompt_tokens=20,
            prefill_backlog_tokens=100,
            decode_backlog_tokens=40,
            prefill_tps=2000.0,
            decode_tps=1000.0,
            mean_decode_batch_ms=5.0,
        ),
        "large": _score_wait_result(
            "large",
            prompt_tokens=20,
            prefill_backlog_tokens=200,
            decode_backlog_tokens=80,
            prefill_tps=1000.0,
            decode_tps=1000.0,
            mean_decode_batch_ms=5.0,
        ),
    }
    logged = experiments._build_compact_score_candidate_terms(
        wait_results=waits,
        selected_instance_id="small",
        slo_ms=110.0,
        prompt_tokens=20,
        accuracy_scores={"small": 0.5, "large": 0.99},
        output_lengths={"small": 10.0, "large": 20.0},
        instance_costs={
            "small": {"prompt": 1.0, "output": 2.0},
            "large": {"prompt": 3.0, "output": 4.0},
        },
        lambda_weight=0.01,
    )

    assert logged["selected_instance_id"] == "small"
    assert logged["slo_ms"] == pytest.approx(110.0)
    assert logged["any_slo_feasible"] is True
    assert set(logged["candidates"]) == {"small", "large"}

    small = logged["candidates"]["small"]
    assert small == {
        "prefill_backlog_tokens": 100.0,
        "decode_backlog_tokens": 40.0,
        "pending_overlay_count": 2,
        "prefill_ms": 60.0,
        "decode_interference_ms": 40.0,
        "first_decode_batch_ms": 5.0,
        "predicted_ttft_ms": 105.0,
        "slo_feasible": True,
        "predicted_quality": 0.5,
        "predicted_output_tokens": 10.0,
        "predicted_cost": 40.0,
        "hard_candidate_value": pytest.approx(0.1),
    }
    large = logged["candidates"]["large"]
    assert large["predicted_ttft_ms"] == pytest.approx(305.0)
    assert large["slo_feasible"] is False
    assert large["hard_candidate_value"] is None


def test_compact_score_candidate_logging_rejects_ttft_drift():
    wait = _score_wait_result(
        "small",
        prompt_tokens=20,
        prefill_backlog_tokens=100,
        decode_backlog_tokens=40,
        prefill_tps=2000.0,
        decode_tps=1000.0,
        mean_decode_batch_ms=5.0,
    )
    wait.wait_ms += 1.0

    with pytest.raises(ValueError, match="differs from the value used for routing"):
        experiments._build_compact_score_candidate_terms(
            wait_results={"small": wait},
            selected_instance_id="small",
            slo_ms=110.0,
            prompt_tokens=20,
            accuracy_scores={"small": 0.5},
            output_lengths={"small": 10.0},
            instance_costs={"small": {"prompt": 1.0, "output": 2.0}},
            lambda_weight=0.01,
        )


def test_score_proxy_uses_snapshot_summary_path_not_sfs_critical_path():
    assert experiments.CollectingWaitTimeScheduler._wait_fetch_args_for_estimator(
        experiments.SCORE_PROXY_TTFT_ESTIMATOR_NAME,
        prompt_tokens=4096,
    ) == (None, None)
    assert not experiments._should_include_unconditional_live_fetch(
        experiments.SCORE_PROXY_TTFT_ESTIMATOR_NAME
    )


def test_hard_score_proxy_has_identical_quality_cost_and_slo_logic_to_hard():
    state = experiments.UtilityState(current_slo_ms=100.0)
    costs = {
        "small": {"prompt": 1.0, "output": 2.0},
        "large": {"prompt": 3.0, "output": 4.0},
    }
    hard = experiments.build_utility_fn(
        "hard",
        lambda_weight=0.01,
        delta_weight=0.0,
        instance_costs=costs,
        utility_state=state,
    )
    score = experiments.build_utility_fn(
        "hard_score_proxy",
        lambda_weight=0.01,
        delta_weight=999.0,
        instance_costs=costs,
        utility_state=state,
    )
    waits = {
        "small": WaitTimeResult("small", 90.0, 1.0, {}),
        "large": WaitTimeResult("large", 110.0, 1.0, {}),
    }
    accuracy = {"small": 0.5, "large": 0.99}
    outputs = {"small": 10.0, "large": 20.0}

    hard_scores = {
        key: hard(key, waits, accuracy, outputs, 10) for key in waits
    }
    score_scores = {
        key: score(key, waits, accuracy, outputs, 10) for key in waits
    }
    assert score_scores == hard_scores
    assert max(score_scores, key=score_scores.get) == "small"

    state.current_slo_ms = 10.0
    assert score("small", waits, accuracy, outputs, 10) == -90.0
    assert score("large", waits, accuracy, outputs, 10) == -110.0


def test_service_metrics_json_maps_aliases_and_requires_decode_terms(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "qwen3-8b": {
                    "prefill_theta": {
                        "theta_p_tps_from_wait_logs": 2000.0,
                        "theta_p_tps_from_batch_stats": 6000.0,
                    },
                    "score_proxy": {
                        "prefill_tps": 1000.0,
                        "decode_tps": 1000.0,
                        "mean_decode_batch_ms": 5.0,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    calibrated = experiments._load_calibrated_service_metrics(metrics_path)
    instances = {
        "vllm-8b": SimpleNamespace(
            model_id="qwen3-8b",
            default_model="/models/Qwen3-8B",
        )
    }

    prefill, _ = experiments._resolve_router_prefill_tps_by_instance(
        instances=instances,
        calibrated_metrics=calibrated,
    )
    decode, decode_batch_ms, diagnostics = (
        experiments._resolve_score_proxy_metrics_by_instance(
            instances=instances,
            calibrated_metrics=calibrated,
        )
    )

    assert prefill == {"vllm-8b": 6000.0}
    assert decode == {"vllm-8b": 1000.0}
    assert decode_batch_ms == {"vllm-8b": 5.0}
    assert diagnostics["vllm-8b"]["matched_key"] == "qwen3-8b"

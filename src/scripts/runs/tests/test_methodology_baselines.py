"""CPU harness/reporting integration for the three methodology policies.

The real dispatch coordinator, calibration JSON loader, request normalization,
sidecar writers, CLI parser, and report collation run here. Network clients,
tokenization, and learned model inference are explicit fakes; these tests are
not measured serving-capacity or prediction-quality evidence.
"""

from __future__ import annotations

import asyncio
import ast
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.runs import experiments, experiments_sweep
from scripts.reporting.router_qps_sweep_summary import aggregate_jsons
from sfs_core.routing import wait_time_scheduler
from sfs_core.routing.methodology_calibration import (
    MethodologyCalibration, PREFILL_FEATURE_NAMES, TPOT_FEATURE_NAMES,
    file_sha256, serving_profile_sha256,
)
from sfs_core.routing.methodology_snapshot import parse_baseline_snapshot
from sfs_core.routing.routebalance_predictor import RouteBalancePredictor


POLICIES = ("lmdeploy_proxy", "mooncake_prefill", "routebalance")
MODELS = ("model-small", "model-large")


def _requests():
    return [experiments.ExperimentRequest(
        request_id=f"q{index}", prompt=f"Calibration fixture prompt {index}",
        prompt_tokens=8, bucket="alpaca", latency_slo_ms=10000,
        queue_slo_ms=1000, ttft_slo_ms=1000,
    ) for index in range(3)]


def _snapshot(active, version):
    now = time.monotonic()
    return parse_baseline_snapshot({
        "version": version, "created_at": now,
        "num_running": 0, "num_waiting": len(active),
        "running_request_ids": [], "waiting_request_ids": list(active),
        "requests": {request_id: {
            "request_id": request_id, "status": "WAITING",
            "num_prompt_tokens": 8, "num_computed_tokens": 0,
            "num_output_processed_tokens": 0, "kv_block_counts": [0],
        } for request_id in active},
        "inflight_batch": None,
        "config": {"max_num_batched_tokens": 128, "max_num_seqs": 512,
                   "max_model_len": 4096, "chunked_prefill_enabled": True},
        "kv_cache_config": {"block_size": 16, "kv_cache_free_blocks": 200,
                            "kv_cache_total_blocks": 256, "kv_cache_groups": [{}]},
        "parallel_config": {"decode_context_parallel_size": 1},
    }, observed_at=now)


class FakeClient:
    def __init__(self, index, *, fail=False):
        self.instance_id = f"instance-{index}"
        self.model_id = MODELS[index]
        self.default_model = self.model_id
        self.calls = []
        self.active = set()
        self.seen_engine_ids = set()
        self.snapshot_calls = 0
        self.closed = False
        self.fail = fail

    async def refresh_baseline_state(self):
        self.snapshot_calls += 1
        self.seen_engine_ids.update(self.active)
        return _snapshot(self.active, self.snapshot_calls)

    async def submit_request(self, **payload):
        self.calls.append(payload)
        # Match the real OpenAI endpoint's ID namespace. A router-ID-only
        # ledger cannot reconcile these observed requests correctly.
        engine_id = "chatcmpl-" + payload["extra_body"]["request_id"]
        self.active.add(engine_id)
        try:
            await asyncio.sleep(0.02)
            if self.fail:
                raise RuntimeError("fixture network failure")
            return SimpleNamespace(
                id=engine_id, model=self.model_id,
                usage=SimpleNamespace(prompt_tokens=8, completion_tokens=2, total_tokens=10),
            )
        finally:
            self.active.remove(engine_id)

    def prime_wait_source(self):
        raise AssertionError("Methodology baseline started SFS native watcher")

    def close(self):
        self.closed = True


class FakePredictor:
    model_labels = MODELS
    metadata = {"fixture": "fake-inference", "architecture": "MiniLM/KNN"}

    def predict_batch(self, prompts, caps):
        time.sleep(0.002)
        return [{model: {"quality": 0.75 + 0.1 * index,
                         "output_tokens": float(min(cap, 4 + index))}
                 for index, model in enumerate(MODELS)}
                for prompt, cap in zip(prompts, caps)]


@pytest.fixture
def calibration_path(tmp_path):
    # Real artifact parsing/checksums, fake learned-head inference below.
    head = tmp_path / "fixture-head.json"
    head.write_text("{}", encoding="utf-8")
    profile = {"fixture": True, "dtype": "bfloat16", "max_model_len": 4096}
    payload = {
        "schema_version": 1, "data_role": "calibration",
        "serving_profile": profile, "serving_profile_verified": True,
        "serving_profile_sha256": serving_profile_sha256(profile),
        "models": {model: {
            "service_rate_qps": 4.0 / (index + 1),
            "service_rate_definition": "fixture arrivals-window throughput",
            "prefill": {"feature_names": list(PREFILL_FEATURE_NAMES),
                        "coefficients_ms": [0.1, float(index + 1), 0, 0],
                        "parameterization": "intercept_linear_causal_quadratic",
                        "chunk_tokens": 128},
            "tpot": {"feature_names": list(TPOT_FEATURE_NAMES),
                     "model_file": head.name, "model_sha256": file_sha256(head)},
        } for index, model in enumerate(MODELS)},
    }
    path = tmp_path / "methodology-calibration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def harness(monkeypatch, calibration_path):
    warmups = []

    async def warmup(clients):
        warmups.append([client.instance_id for client in clients])

    monkeypatch.setattr(experiments, "warm_up_instances", warmup)
    monkeypatch.setattr(wait_time_scheduler, "load_tokenizer", lambda *a, **k: object())
    monkeypatch.setattr(MethodologyCalibration, "tpot_ms", lambda self, model, features: 2.0)
    monkeypatch.setattr(RouteBalancePredictor, "load", classmethod(lambda cls, path: FakePredictor()))

    def forbidden_predictor(*args, **kwargs):
        raise AssertionError("SFS predictor loaded for a methodology policy")

    monkeypatch.setattr(wait_time_scheduler, "AccuracyPredictor", forbidden_predictor)
    monkeypatch.setattr(wait_time_scheduler, "OutputLengthPredictor", forbidden_predictor)
    return warmups


def _run_kwargs(policy, clients, calibration_path, tmp_path):
    runtime = experiments._baseline_runtime_params(policy, lambda_weight=0.3, delta_weight=0.5)
    return dict(
        utility_name=policy, requests=_requests(), instances=clients,
        instance_costs={key: {"prompt": 0.1, "output": 0.2} for key in clients},
        accuracy_model_path="must-not-load-sfs-accuracy", output_length_model_path="must-not-load-sfs-length",
        lambda_weight=runtime[0], delta_weight=runtime[1],
        worker_count=4, max_queue_size=2, request_rate_qps=0,
        arrival_process="deterministic", arrival_seed=7,
        max_completion_tokens=32, temperature=0, top_p=1,
        tokenizer_id="fixture-tokenizer", system_prompt="You are a helpful assistant.",
        response_map_path=str(tmp_path / f"{policy}-responses.log"),
        request_log_path=str(tmp_path / f"{policy}-decisions.log"),
        route_strategy=runtime[2], wait_estimator=runtime[3], wait_estimator_name=runtime[4],
        methodology_calibration_path=str(calibration_path),
        routebalance_predictor_path=str(tmp_path / "fixture-predictor"),
        routebalance_weights=(0.5, 0.25, 0.25), routebalance_batch_wait_ms=2,
        route_random_seed=7, close_instances_on_stop=False,
    )


def test_scout_backlog_stop_drains_only_actual_arrivals(harness, calibration_path, tmp_path):
    from scripts.runs.capacity_scout import TrialMonitor

    async def exercise():
        clients = {f"instance-{i}": FakeClient(i) for i in range(2)}
        monitor = TrialMonitor(tmp_path/"events.jsonl", max_outstanding=1)
        run = await asyncio.wait_for(experiments.run_policy(
            **_run_kwargs("lmdeploy_proxy", clients, calibration_path, tmp_path),
            trial_monitor=monitor), timeout=10)
        assert run["summary"]["total_requests"] == 1
        assert run["summary"]["succeeded_requests"] == 1
        assert monitor.arrivals == monitor.completed == 1
        assert next(e for e in monitor.events if e["event"] == "arrivals_end")["reason"] == "backlog_limit"
        assert not any(run["methodology_config"]["unfinished_after_drain"].values())
    asyncio.run(exercise())


def test_dispatch_failure_resolves_harness_completion_without_hanging(
    harness, calibration_path, tmp_path, monkeypatch,
):
    async def fail_dispatch(self, queued):
        raise RuntimeError("snapshot unavailable")

    monkeypatch.setattr(experiments.CollectingWaitTimeScheduler, "_dispatch", fail_dispatch)

    async def exercise():
        clients = {f"instance-{i}": FakeClient(i) for i in range(2)}
        for client in clients.values():
            client.prime_wait_source = lambda: None
        kwargs = _run_kwargs("round_robin", clients, calibration_path, tmp_path)
        kwargs.update(accuracy_model_path=None, output_length_model_path=None,
                      enable_wait_time_polling=False)
        run = await asyncio.wait_for(experiments.run_policy(**kwargs), timeout=10)
        assert run["summary"]["failed_requests"] == 3
        assert all("snapshot unavailable" in row["error"] for row in run["per_request"])
    asyncio.run(exercise())


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("decouple_arrivals", [False, True])
def test_run_policy_preserves_decisions_costs_timing_and_engine_id_reconciliation(
    policy, decouple_arrivals, harness, calibration_path, tmp_path, monkeypatch,
):
    clients = {client.instance_id: client for client in (FakeClient(0), FakeClient(1))}
    kwargs = _run_kwargs(policy, clients, calibration_path, tmp_path)
    kwargs["decouple_arrivals"] = decouple_arrivals
    kwargs["per_request_wait_logs"] = [tmp_path / "engine-waits.log"]
    monkeypatch.setattr(experiments, "_read_latency_components_from_logs", lambda *a, **k: {
        f"chatcmpl-{policy}-q{index}": experiments.RequestLatencyComponents(
            queue_ms=2, prefill_ms=3, frontend_ttft_ms=7,
        ) for index in range(3)
    })
    result = asyncio.run(experiments.run_policy(**kwargs))
    assert result["summary"]["succeeded_requests"] == 3, result["per_request"]
    assert result["summary"]["failed_requests"] == 0
    assert harness == [list(clients)]
    assert result["utility"] == policy
    assert result["route_strategy"] == policy
    assert result["methodology_config"]["policy"] == policy
    assert result["methodology_config"]["sfs_predictor_substitution"] is False
    assert set(result["methodology_config"]["unfinished_after_drain"].values()) == {0}
    assert all(not client.closed and not client.active for client in clients.values())
    assert result["summary"]["actual_cost"]["mean"] == pytest.approx(1.2)
    for item in result["per_request"]:
        assert item["methodology_terms"]["policy"] == policy
        assert set(item["methodology_terms"]["candidates"]) == set(clients)
        assert item["methodology_terms"]["random_seed"] == 7
        assert item["route_strategy"] == policy
        assert item["cost_source"] == "actual_usage"
        assert item["system_entry_to_dispatch_ms"] >= item["arrival_to_dispatch_ms"]
        assert item["system_entry_e2e_ttft_ms"] == pytest.approx(item["system_entry_to_dispatch_ms"] + 5)
        assert item["ttft_ms"] == 5
        assert item["frontend_ttft_ms"] == 7
        assert item["response_id"] == "chatcmpl-" + item["scheduler_request_id"]
        assert item["methodology_terms"]["engine_request_id"] == item["response_id"]
        if policy != "routebalance":
            assert item["predicted_accuracy"] is None
            assert item["predicted_output_tokens"] is None
        else:
            assert item["predicted_accuracy"] in (0.75, 0.85)
            assert item["methodology_terms"]["prediction_batch_ms"] > 0
            assert item["system_entry_to_dispatch_ms"] >= item["methodology_terms"]["prediction_batch_ms"]
    if policy == "lmdeploy_proxy":
        assert sum(client.snapshot_calls for client in clients.values()) == 0
        assert result["summary"]["predicted_accuracy"]["count"] == 0
    else:
        assert any(client.seen_engine_ids for client in clients.values())
    # These shared legacy sidecars deliberately use Python literal records.
    decisions = [ast.literal_eval(line) for line in Path(kwargs["request_log_path"]).read_text().splitlines()]
    mappings = [ast.literal_eval(line) for line in Path(kwargs["response_map_path"]).read_text().splitlines()]
    assert len(decisions) == len(mappings) == 3
    assert all(row["payload"]["methodology_terms"]["policy"] == policy for row in decisions)
    assert {row["response_id"] for row in mappings} == {row["response_id"] for row in result["per_request"]}


def test_run_policy_failed_response_preserves_policy_diagnostics_and_releases_counter(
    harness, calibration_path, tmp_path,
):
    clients = {client.instance_id: client for client in (FakeClient(0, fail=True), FakeClient(1, fail=True))}
    result = asyncio.run(experiments.run_policy(**_run_kwargs("lmdeploy_proxy", clients, calibration_path, tmp_path)))
    assert result["summary"]["failed_requests"] == 3
    assert result["summary"]["succeeded_requests"] == 0
    assert set(result["methodology_config"]["unfinished_after_drain"].values()) == {0}
    assert all("fixture network failure" in row["error"] for row in result["per_request"])
    assert all(row["methodology_terms"]["policy"] == "lmdeploy_proxy" for row in result["per_request"])


def test_run_policy_rejects_mismatched_serving_profile_before_warmup(
    harness, calibration_path, tmp_path,
):
    clients = {client.instance_id: client for client in (FakeClient(0), FakeClient(1))}
    kwargs = _run_kwargs("lmdeploy_proxy", clients, calibration_path, tmp_path)
    kwargs["methodology_serving_profile"] = {"dtype": "different-profile"}
    with pytest.raises(ValueError, match="profile"):
        asyncio.run(experiments.run_policy(**kwargs))
    assert harness == []
    assert not any(client.calls for client in clients.values())


@pytest.mark.parametrize("policy", POLICIES)
def test_runtime_selection_bypasses_sfs_wait_estimators(policy):
    assert experiments._baseline_runtime_params(policy, lambda_weight=0.3, delta_weight=9) == (
        0.3, 0.0, policy, None, policy, True,
    )


def test_cli_accepts_baseline_artifact_and_batch_options(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "experiments.py", "--utilities", *POLICIES,
        "--methodology-calibration-json", "calibration.json",
        "--routebalance-predictor-path", "knn-artifact",
        "--routebalance-weights", "0.5", "0.3", "0.2",
        "--routebalance-batch-max-size", "8", "--routebalance-batch-wait-ms", "12",
        "--methodology-snapshot-max-age-ms", "750",
    ])
    args = experiments.parse_args()
    assert args.utilities == list(POLICIES)
    assert args.methodology_calibration_json == "calibration.json"
    assert args.routebalance_predictor_path == "knn-artifact"
    assert tuple(args.routebalance_weights) == (0.5, 0.3, 0.2)
    assert args.routebalance_batch_max_size == 8
    assert args.routebalance_batch_wait_ms == 12
    assert args.methodology_snapshot_max_age_ms == 750


@pytest.mark.parametrize("flags", [
    ["--utilities", "lmdeploy_proxy"],
    ["--utilities", "routebalance", "--methodology-calibration-json", "x.json"],
    ["--routebalance-weights", "1", "1", "1"],
    ["--routebalance-batch-max-size", "0"],
    ["--routebalance-batch-wait-ms", "nan"],
    ["--methodology-snapshot-max-age-ms", "-1"],
])
def test_cli_rejects_incomplete_or_invalid_baseline_configuration(monkeypatch, flags):
    monkeypatch.setattr(sys, "argv", ["experiments.py", *flags])
    with pytest.raises(SystemExit) as exc:
        experiments.parse_args()
    assert exc.value.code == 2


def test_qps_wrapper_forwards_new_policy_artifacts_at_every_shared_point(tmp_path):
    wrapper, forwarded = experiments_sweep._parse_wrapper_args([
        "--sweep", "qps", "--qps-values", "2", "3", "--qps-utilities", *POLICIES,
        "--output-dir", str(tmp_path), "--utilities", *POLICIES,
        "--methodology-calibration-json", "calibration.json",
        "--routebalance-predictor-path", "predictor", "--routebalance-weights", "0.5", "0.3", "0.2",
        "--routebalance-batch-wait-ms", "12",
    ])
    base = experiments_sweep._parse_experiment_args(forwarded)
    points = experiments_sweep._build_run_points(wrapper, base)
    assert len(points) == 2
    for point in points:
        args = experiments_sweep._as_run_args(
            base_args=base, utilities=point["utilities"], lambda_weight=point["lambda_weight"],
            delta_weight=point["delta_weight"], request_rate_qps=point["request_rate_qps"],
        )
        assert args.utilities == list(POLICIES)
        assert args.methodology_calibration_json == "calibration.json"
        assert args.routebalance_predictor_path == "predictor"
        assert tuple(args.routebalance_weights) == (0.5, 0.3, 0.2)
        assert args.routebalance_batch_wait_ms == 12
    assert [point["request_rate_qps"] for point in points] == [2, 3]


def test_default_delta_sweep_does_not_silently_require_new_baseline_artifacts(tmp_path):
    wrapper, forwarded = experiments_sweep._parse_wrapper_args([
        "--sweep", "delta", "--delta-values", "0", "--output-dir", str(tmp_path),
    ])
    base = experiments_sweep._parse_experiment_args(forwarded)
    points = experiments_sweep._build_run_points(wrapper, base)
    assert set(points[0]["utilities"]) == set(experiments.BUILTIN_UTILITIES) - set(POLICIES) - {"vllm_sr_latency"}


def test_sweep_execution_serializes_baseline_configuration_for_every_point(
    tmp_path, calibration_path, monkeypatch,
):
    requests = _requests()
    clients = {client.instance_id: client for client in (FakeClient(0), FakeClient(1))}
    monkeypatch.setattr(experiments, "_build_request_set", lambda args: (
        requests, [{"request_id": req.request_id} for req in requests], {"fixture": True},
    ))
    monkeypatch.setattr(experiments, "load_instances", lambda path: (clients, {}, {}))
    observed = []

    async def run_router(**kwargs):
        args = kwargs["args"]
        observed.append(args)
        return {"runs": [{"utility": policy, "methodology_config": {"policy": policy}}
                         for policy in args.utilities]}

    monkeypatch.setattr(experiments, "run_router_experiment", run_router)
    output_dir = tmp_path / "sweep-outputs"
    asyncio.run(experiments_sweep._async_main([
        "--sweep", "qps", "--qps-values", "2", "3", "--qps-utilities", *POLICIES,
        "--output-dir", str(output_dir), "--utilities", *POLICIES,
        "--num-requests", "3", "--methodology-calibration-json", str(calibration_path),
        "--routebalance-predictor-path", "fixture-predictor",
        "--routebalance-weights", "0.5", "0.3", "0.2",
        "--routebalance-batch-max-size", "8", "--routebalance-batch-wait-ms", "12",
        "--methodology-snapshot-max-age-ms", "750",
    ]))
    assert [args.request_rate_qps for args in observed] == [2, 3]
    assert all(args.utilities == list(POLICIES) for args in observed)
    assert all(client.closed for client in clients.values())
    point_files = [path for path in output_dir.glob("*.json") if not path.name.endswith("manifest.json")]
    assert len(point_files) == 2
    for path in point_files:
        payload = json.loads(path.read_text())
        config = payload["config"]
        assert config["utilities"] == list(POLICIES)
        assert config["methodology_calibration_json"] == str(calibration_path)
        assert config["routebalance_predictor_path"] == "fixture-predictor"
        assert config["routebalance_weights"] == [0.5, 0.3, 0.2]
        assert config["routebalance_batch_max_size"] == 8
        assert config["routebalance_batch_wait_ms"] == 12
        assert config["methodology_snapshot_max_age_ms"] == 750
        assert [run["utility"] for run in payload["router"]["runs"]] == list(POLICIES)


def test_report_collates_all_policies_using_actual_quality_and_arrival_slo_gate(tmp_path):
    runs = []
    for policy in POLICIES:
        runs.append({
            "utility": policy,
            "methodology_config": {"policy": policy},
            "summary": {"predicted_accuracy": {"mean": None},
                        "actual_cost": {"mean": 2.0}, "throughput_qps_all": 2.0},
            "per_request": [
                {"actual_accuracy": 0.8, "actual_cost": 2.0,
                 "system_entry_e2e_ttft_slo_met": True,
                 "predicted_accuracy": None, "methodology_terms": {"policy": policy}},
                {"actual_accuracy": 0.4, "actual_cost": 2.0,
                 "ttft_slo_met": True, "system_entry_e2e_ttft_slo_met": False,
                 "predicted_accuracy": None, "methodology_terms": {"policy": policy}},
            ],
        })
    path = tmp_path / "scored-derived.json"
    path.write_text(json.dumps({"config": {"request_rate_qps": 2.0, "lambda_weight": 0.1},
                                "router": {"runs": runs}}), encoding="utf-8")
    original = path.read_bytes()
    summary, qps = aggregate_jsons([tmp_path])
    assert qps == ["2"]
    assert set(summary["utilities"]) == set(POLICIES)
    for policy in POLICIES:
        metrics = summary["qps"]["2"][policy]
        assert metrics["predicted_accuracy_mean"] is None
        assert metrics["actual_accuracy_mean"] == pytest.approx(0.6)
        assert metrics["actual_accuracy_minus_lambda_weight_times_actual_cost"] == pytest.approx(0.4)
        assert metrics["actual_slo_gated_utility_mean"] == pytest.approx(0.3)
    assert path.read_bytes() == original

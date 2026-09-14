"""Exercise the complete stage controller with explicit fake serving/fit APIs."""
import asyncio
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.runs import experiments as exp
from scripts.runs import ministral3_methodology_stage as stage
from scripts.prep import fit_methodology_calibration
from scripts.prep.prepare_methodology_service import MODELS, PROFILE, sha256
from sfs_core.routing.methodology_calibration import MethodologyCalibration
from sfs_core.shared import shared_experiment_helpers


@pytest.mark.parametrize("mode", ["complete", "budget_exhausted", "resume"])
def test_complete_stage_preserves_probe_traces_and_requires_measured_bracket(tmp_path, monkeypatch, mode):
    prepared, root = tmp_path/"prepared", tmp_path/"pool"
    prepared.mkdir()
    root.mkdir()
    metrics = tmp_path/"model_metrics.json"
    metrics.write_text(json.dumps({m: {"prefill_tps": 1234, "decode_tps": 56,
        "mean_decode_batch_ms": 7} for m in MODELS}))
    manifest = {"coverage_audit": {m: {"requires_prefill_probes": True} for m in MODELS},
                "models": {m: {"traces": [str(metrics)]} for m in MODELS},
                "service_metrics_sha256": sha256(metrics)}
    manifest_path = tmp_path/"manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    source_bytes = manifest_path.read_bytes()
    rows = [exp.ExperimentRequest(str(i), "calibration prompt", i+1, stage.BUCKETS[i%4],
                                  10000, 1000, 1000) for i in range(10000)]
    with (prepared/"requests.jsonl").open('w') as stream:
        stream.writelines(json.dumps(asdict(row))+'\n' for row in rows)
    (prepared/"metadata.json").write_text(json.dumps({"status": "READY", "data_role": "calibration",
        "num_requests": 10000, "holdout_start_index": 0, "prompts_per_bucket": 2500,
        "requests_sha256": sha256(prepared/"requests.jsonl"), "forwarded_argv": [],
        "service_manifest_sha256": sha256(manifest_path)}))
    for m in MODELS:
        (root/f"batch_stats_{m}.csv").write_text("probe-only-trace\n")
    probes, calls = [], []

    class Client:
        def __init__(self, model_id, instance_id, **_):
            self.model_id, self.instance_id = model_id, instance_id
            self.closed = False
            clients[instance_id] = self

        async def submit_request(self, **payload):
            probes.append((self.model_id, payload))
            assert payload["max_completion_tokens"] == 1
            return SimpleNamespace(id=payload["extra_body"]["request_id"],
                                   usage=SimpleNamespace(model_dump=lambda: {"completion_tokens": 1}))

        def close(self):
            self.closed = True

    clients = {}
    # Keep the real loader and its metadata propagation in the controller path.
    # Only the network client is replaced at the external-serving boundary.
    monkeypatch.setattr(exp, "InstanceClient", Client)
    (tmp_path/"instances.json").write_text(json.dumps({"serving_profile": PROFILE,
        "instances": [{"instance_id": m, "model_id": m, "default_model": m,
                       "address": "http://127.0.0.1:1"} for m in MODELS]}))

    async def noop(*_, **__):
        pass
    monkeypatch.setattr(stage, "wait_drained", noop)
    monkeypatch.setattr(shared_experiment_helpers, "warm_up_instances", noop)

    def fit(path, destination):
        loaded = json.loads(Path(path).read_text())
        assert all(len(loaded["models"][m]["traces"]) == 2 for m in MODELS)
        destination.mkdir()
        output = destination/"methodology_calibration.json"
        output.write_text('{}')
        return output
    monkeypatch.setattr(fit_methodology_calibration, "fit_manifest", fit)
    monkeypatch.setattr(MethodologyCalibration, "load", classmethod(lambda *a:
        SimpleNamespace(validate_runtime_profile=lambda profile: None)))

    async def route(**kw):
        assert set(kw["args"].calibrated_service_metrics) == set(MODELS)
        assert all(row["prefill_tps"] == 1234 for row in
                   kw["args"].calibrated_service_metrics.values())
        policy = kw["args"].utilities[0]
        rate = kw["args"].request_rate_qps
        calls.append((policy, rate))
        monitor = kw["trial_monitor"]
        if str(kw["response_map_base_path"].name).startswith("smoke"):
            count = 192
        else:
            count = int(rate*360)
            throughput = rate if rate < 6.7 else 6.2
            monitor.events = ([{"event": "arrival", "t": i/rate} for i in range(count)] +
                [{"event": "completion", "t": 10+i/throughput, "error": None}
                 for i in range(int(throughput*350))] +
                [{"event": "sample", "t": 240}, {"event": "arrivals_end", "t": 360,
                                                 "reason": "duration_limit"}])
        for m in MODELS:
            with (root/f"batch_stats_{m}.csv").open('a') as stream:
                stream.write("later-trial-data\n")
        result = {"summary": {"total_requests": count}, "per_request": [
            {"request_id": str(i), "response_id": str(i), "queue_delay_ms": 1, "ttft_ms": 2,
             "system_entry_to_dispatch_ms": 1, "actual_cost": .1, "usage_completion_tokens": 1,
             "methodology_terms": {"policy": policy}} for i in range(count)]}
        return {"runs": [result]}
    monkeypatch.setattr(exp, "run_router_experiment", route)
    options = SimpleNamespace(prepared_dir=prepared, output_root=root,
        service_manifest=manifest_path, service_metrics_json=metrics,
        instances_config=tmp_path/"instances.json", routebalance_predictor=tmp_path/"predictor",
        wait_log=[], duration_s=360, max_trials=3 if mode == "budget_exhausted" else 10)
    if mode == "resume":
        source = tmp_path/"previous_stage"
        (source/"timing_models").mkdir(parents=True)
        (source/"timing_models/methodology_calibration.json").write_text('{}')
        (source/"smoke_audit.json").write_text('{"status":"PASS"}')
        for policy in stage.POLICIES:
            (source/f"smoke_{policy}.json").write_text('{"reused":true}')
        options.resume_from = source
        reused = {"source": str(source), "file_sha256": {}, "trials": [
            {"classification": "stable", "requested_qps": 6.5},
            {"classification": "unstable", "requested_qps": 7},
            {"classification": "inconclusive", "requested_qps": 6.75,
             "reason": "borderline_or_nonstationary_windows", "stop_reason": "duration_limit"}]}
        monkeypatch.setattr(stage, "validate_resume_stage", lambda options: reused)
    asyncio.run(stage.run_stage(options))
    if mode == "resume":
        assert not probes
        assert calls[0] == ("shortest_queue", 6.75)
        assert (root/"smoke_hard.json").read_bytes() == (source/"smoke_hard.json").read_bytes()
        assert json.loads((root/"reuse_manifest.json").read_text())["source"] == str(source)
    else:
        assert Counter(model for model, _ in probes) == Counter({m: 64 for m in MODELS})
        assert [policy for policy, _ in calls[:8]] == list(stage.POLICIES)
        for m in MODELS:
            assert (root/f"prefill_probe_trace_{m}.csv").read_text() == "probe-only-trace\n"
    assert manifest_path.read_bytes() == source_bytes
    assert all(client.closed for client in clients.values())
    if mode == "budget_exhausted":
        assert not (root/"capacity_scout.json").exists()
        checkpoint = json.loads((root/"capacity_scout_progress.json").read_text())
        assert checkpoint["status"] == "NEEDS_MORE_TRIALS"
        assert checkpoint["next_trial"]["qps"] == 6
        assert len(checkpoint["trials"]) == 3
        return
    result = json.loads((root/"capacity_scout.json").read_text())
    assert result["status"] == "PASS"
    assert result["stable_qps"] == 6.5 and result["unstable_qps"] == 6.75
    assert result["final_matrix_cells"] == 32
    assert result["final_requests_per_cell"] == 16000
    assert not result["evaluation_started"]
    assert [row["classification"] for row in result["endpoint_confirmations"]] == ["stable", "unstable"]
    assert all(client.closed for client in clients.values())

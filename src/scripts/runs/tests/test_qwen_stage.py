"""Rehearse Qwen calibration orchestration; synthetic traces are not GPU evidence."""
import asyncio
from collections import Counter
import csv
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.runs import experiments as exp
from scripts.runs import qwen_baselines as qwen
from scripts.prep.tests.test_fit_methodology_calibration import synthetic_rows
from scripts.prep.paper_ablation_data import write_json
from sfs_core.routing.methodology_calibration import MethodologyCalibration


@pytest.mark.parametrize("missing_usage", [False, True])
def test_qwen_controller_validates_calibration_smoke_and_drain(tmp_path, monkeypatch, missing_usage):
    output = tmp_path/"stage"
    output.mkdir()
    requests = [exp.ExperimentRequest(str(i), f"calibration prompt {i}", i+1,
                qwen.BUCKETS[i%4], 10000, 1000, 1000) for i in range(10000)]
    request_path = tmp_path/"calibration_requests.jsonl"
    request_path.write_text("".join(json.dumps(asdict(r))+"\n" for r in requests))
    manifest = {"manifest_path":str(tmp_path/"manifest.json"), "calibration_requests":str(request_path),
                "experiment_argv":qwen.base_argv(tmp_path,tmp_path/"unused-cache")}
    write_json(manifest["manifest_path"],manifest)
    predictor = tmp_path/"predictor"
    write_json(predictor/"metadata.json",{"synthetic_test":True})
    instance_path = tmp_path/"instances.json"
    write_json(instance_path,qwen.pool_config(manifest))
    columns = ["ts","engine","prefill","prefill_sq_sum","decode","num_seqs",
               "sum_tokens","sum_tokens_sq","exec","prefill_x_processed_ctx_sum"]
    frozen_expected = {}
    for model in qwen.MODELS:
        path = output/f"batch_stats_{model}.csv"
        with path.open("w",newline="") as stream:
            writer = csv.DictWriter(stream,fieldnames=columns,extrasaction="ignore")
            writer.writeheader()
            for row in synthetic_rows():
                writer.writerow({**row,"ts":1,"engine":0,"sum_tokens_sq":row["sum_tokens"]**2})
        frozen_expected[model] = path.read_bytes()
    clients, submissions, policies = {}, [], []

    class Client:
        def __init__(self,instance_id,model_id,**_):
            self.instance_id, self.model_id, self.closed = instance_id, model_id, False
            self.published = False
            clients[instance_id] = self

        async def submit_request(self,**payload):
            self.published = True
            request_id = payload.get("extra_body",{}).get("request_id","warmup")
            if request_id != "warmup":
                assert payload["extra_body"]["chat_template_kwargs"] == {"enable_thinking":False}
                assert payload["messages"][0]["content"].startswith("You are Qwen")
                assert payload["max_completion_tokens"] == (1 if request_id.startswith("probe-") else 8192)
                submissions.append((self.model_id,request_id.split("-",1)[0]))
            return SimpleNamespace(id=request_id,usage=None if missing_usage else
                SimpleNamespace(model_dump=lambda:{"prompt_tokens":32,"completion_tokens":1}))

        async def refresh_baseline_state(self):
            if not self.published:
                raise RuntimeError("Baseline scheduler snapshot is not published yet")
            return SimpleNamespace(requests=(),inflight_total_tokens=0)

        def close(self):
            self.closed = True

    monkeypatch.setattr(exp,"InstanceClient",Client)  # Preserve production loader and metadata path.

    async def route(**kwargs):
        args = kwargs["args"]
        assert kwargs["instance_metadata"]["serving_profile"] == qwen.PROFILE
        assert set(args.calibrated_service_metrics) == set(qwen.MODELS)
        assert all(r["prefill_tps"] > 0 and r["decode_tps"] > 0
                   for r in args.calibrated_service_metrics.values())
        assert args.num_requests == 192 and args.request_rate_qps == 2.0
        policy = args.utilities[0]
        policies.append(policy)
        for model in qwen.MODELS:
            with (output/f"batch_stats_{model}.csv").open("a") as stream:
                stream.write("later-smoke-data\n")
        extra = {}
        rowextras = {}
        if policy == "vllm_sr_latency":
            from sfs_core.routing.latency_history import LatencyHistory
            history = LatencyHistory(list(qwen.MODELS))
            history.reset()
            for i in range(32):
                for model in qwen.MODELS:
                    for metric in ("ttft","tpot"):
                        history.update(model,metric,.1,request_id=f"warmup:{model}:{i}",generation=1)
            for request in kwargs["requests"]:
                decision=history.select(); model=decision['selected_model']
                for metric in ("ttft","tpot"):
                    history.update(model,metric,.1,request_id=request.request_id,generation=1)
                rowextras[request.request_id] = {"methodology_terms":decision,"instance_id":model,
                    "response_model":model,"scheduler_request_id":request.request_id}
            extra["methodology_config"] = {**history.metadata(),"warmup_completions":96,"instance_models":{m:m for m in qwen.MODELS}}
        return {"runs":[{"utility":policy, **extra, "per_request":[{
            "request_id":r.request_id,"response_id":f"response-{r.request_id}",
            "queue_delay_ms":1,"ttft_ms":2,"system_entry_to_dispatch_ms":1,
            "actual_cost":.1,"usage_completion_tokens":1, **rowextras.get(r.request_id,{})} for r in kwargs["requests"]]}]}

    monkeypatch.setattr(exp,"run_router_experiment",route)
    if missing_usage:
        with pytest.raises(ValueError,match="Missing measured probe"):
            asyncio.run(qwen.calibrate(manifest,output,instance_path,predictor))
        assert not (output/"smoke_audit.json").exists()
        assert not policies
    else:
        asyncio.run(qwen.calibrate(manifest,output,instance_path,predictor))
        assert Counter(submissions) == Counter({(m,kind):n for m in qwen.MODELS
                                               for kind,n in (("probe",64),("rate",512))})
        assert policies == list(qwen.POLICIES)
        for model in qwen.MODELS:
            assert (output/f"calibration_trace_{model}.csv").read_bytes() == frozen_expected[model]
        timing = MethodologyCalibration.load(output/"timing_models/methodology_calibration.json")
        timing.preload()
        timing.validate_runtime_profile(qwen.PROFILE)
        assert set(timing.speeds) == set(qwen.MODELS)
        qwen.validate_smoke(output,manifest,predictor)
        manifest["sfs_root"] = str(Path(__file__).resolve().parents[4])
        refreshed = tmp_path/"refreshed"
        refreshed.mkdir()
        policies.clear()
        asyncio.run(qwen.smoke_existing(manifest, output, refreshed, instance_path, predictor, qwen.POLICIES))
        assert policies == list(qwen.POLICIES)
        assert (refreshed/"calibration_responses.json").read_bytes() == (output/"calibration_responses.json").read_bytes()
        assert json.loads((refreshed/"smoke_audit.json").read_text())["evaluation_started"] is False
        qwen.validate_smoke(refreshed, manifest, predictor)
        (output/"model_metrics.json").write_text("{}")
        with pytest.raises(ValueError,match="smoke gate"):
            qwen.validate_smoke(output,manifest,predictor)
    assert all(client.closed for client in clients.values())

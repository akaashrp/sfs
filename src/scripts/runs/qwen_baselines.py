"""Prepare canonical Qwen baseline jobs; reuse measured reference cells.

Preparation/validation is CPU-only. Calibration, smoke and evaluation are
separate modes so no evaluation can start without audited baseline inputs.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time

from scripts.prep.paper_ablation_data import BUCKETS, MODELS, rows, sha256, validate as validate_data, write_json
from scripts.eval.estimator_ablation import PRICES
from scripts.runs.ministral3_methodology_stage import POLICIES as CORE_POLICIES, audit_run, length_stratified_requests, smoke_requests, wait_drained
from scripts.runs.service_metrics_config import build_simulation_args

POLICIES = (*CORE_POLICIES, "vllm_sr_latency")
NEW_POLICIES = ("mooncake_prefill", "lmdeploy_proxy", "routebalance", "score", "vllm_sr_latency")
QPS = (7.0, 8.3, 8.6, 8.9)
PAPER_QPS = (3, 4, 5, 6, 7, 8, 8.3, 8.6, 8.75, 9, 9.2, 9.3, 9.5)
PINS = ("c1899de289a04d12100db370d81485cdf75e47ca", "b968826d9c46dd6066d109eabc6255188de91218", "9216db5781bf21249d130ec9da846c4624c16137")
HF_NAMES = ("Qwen3-0.6B", "Qwen3-8B", "Qwen3-32B")
PROFILE = {"dtype": "auto (BF16 checkpoints)", "max_model_len": 131072,
    "context_length": 65536, "prompt_token_limit": 32768, "max_completion_tokens": 8192,
    "chunked_prefill": True, "max_num_batched_tokens": 32768, "max_num_seqs": 512,
    "prefix_caching": False, "gpu_memory_utilization": .9,
    "tensor_parallel_sizes": [1, 1, 2], "gpu": "H100-80GB",
    "rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}}
COEFFICIENT_NAMES = ("intercept", "prefill_coeff", "prefill_sq_coeff", "decode_coeff", "sum_coeff", "sum_sq_coeff")
# These are legacy-feature-set fits: the sixth coefficient multiplies the sum of squared context lengths.
FEATURE_SET = "legacy"
COEFFICIENTS = (
    (.002582076153059547, 2.0335711505545034e-6, 1.8925400079684374e-10, 1.6889500804006565e-5, 3.5429418765562916e-8, 1.5754738957656986e-13),
    (.007295929680901453, 2.0789009922784725e-5, 5.119559033569263e-10, 2.5632010897991973e-5, 4.6490885513702363e-8, 2.6851287984524727e-13),
    (.014347877296187458, 5.294185118159464e-5, 8.774059443201359e-10, 6.304876078328351e-5, 3.6863941393981496e-8, 4.519593317761812e-13))


def base_argv(root, cache, count=16000, per_bucket=4000):
    return ["--experiment", "router", "--num-requests", str(count), "--seed", "69",
        "--bucket-dir", str(cache), "--holdout-cache-dir", str(cache),
        "--holdout-bucket-dir", str(root.parent/"vllm_utils/bucketed_prompts_0.6B"),
        "--holdout-prompts-per-bucket", str(per_bucket), "--holdout-start-index", "2500",
        "--holdout-context-length", "65536", "--tokenizer-id", "Qwen/Qwen3-8B",
        "--system-prompt", "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        "--chat-template-kwargs-json", '{"enable_thinking":false}', "--max-completion-tokens", "8192",
        "--accuracy-model-path", str(root/"src/assets/predictors/accuracy_predictor"),
        "--output-length-model-path", str(root/"src/assets/predictors/output_length_predictor"),
        "--worker-count", "1", "--max-queue-size", "1", "--temperature", "0", "--top-p", "1",
        "--lambda-weight", "5e-4", "--delta-weight", "0", "--feasible-slo-mode", "ttft",
        "--queue-slo-min-ms", "1", "--queue-slo-max-ms", "5", "--ttft-slo-min-ms", "150",
        "--ttft-slo-max-ms", "1120", "--ttft-slo-base-ms", "155", "--ttft-slo-per-prompt-token-ms", ".035",
        "--ttft-slo-jitter-min", ".98", "--ttft-slo-jitter-max", "1.02", "--arrival-process", "poisson",
        "--decouple-arrivals", "--require-existing-holdout-cache", "--frozen-legacy-holdout-cache"]


def pool_config(manifest, ports=(9200, 9201, 9202), tag="cpu"):
    instances = []
    for index, model in enumerate(MODELS):
        instances.append({"instance_id": "vllm-"+model.removeprefix("qwen3-"),
            "address": f"http://127.0.0.1:{ports[index]}", "default_model": model,
            "model_id": model, "snapshot_shm_name": f"sfs_{tag}_{index}",
            "snapshot_shm_size_bytes": 8*1024*1024, "max_num_batched_tokens": 32768,
            "max_num_seqs": 512, "chunked_prefill_enabled": True, "long_prefill_token_threshold": 0,
            "ttft_batch_model": dict(zip(COEFFICIENT_NAMES, COEFFICIENTS[index])),
            "batch_time_feature_set": FEATURE_SET})
    return {"instances": instances, "serving_profile": PROFILE,
        "instance_costs": {r["instance_id"]: dict(zip(("prompt", "output"), PRICES[r["model_id"]])) for r in instances},
        "cost_units": "USD per million tokens", "canonical_manifest": manifest.get("manifest_path")}


def prepare(root, prepared, output):
    from scripts.runs.experiments_sweep import _parse_experiment_args
    from scripts.runs import experiments as exp
    root, prepared, output = Path(root).resolve(), Path(prepared).resolve(), Path(output).resolve()
    validate_data(prepared)
    if output.exists():
        raise ValueError("Preserve existing Qwen job preparation")
    source = root.parent/"vllm_utils"
    cache = prepared.parent/"qwen_canonical_holdout_4000"
    argv = base_argv(root, cache)
    requests, _, _ = exp._build_request_set(_parse_experiment_args(argv))
    import csv
    reqmap = source/"bucketed_prompt_outputs/req_map_qps_seed69_holdout4000_n16000.csv"
    with reqmap.open() as stream:
        mapped = {r["req_id"]: r for r in csv.DictReader(stream)}
    cached = {(b, r["prompt_metadata"]["example_id"]): r for b in BUCKETS for r in rows(cache/f"{b}.jsonl")}
    if len(requests) != 16000 or len(mapped) != 16000:
        raise ValueError("Canonical Qwen must retain 16000 requests")
    for req in requests:
        row = mapped[req.request_id]
        if req.bucket != row["bucket"] or req.prompt != cached[(row["bucket"], row["example_id"])]["prompt"] or req.prompt_tokens != int(row["prompt_tokens"]):
            raise ValueError("Actual ingestion differs from the canonical request map")
    # Explicit original Poisson sweeps: do not substitute later PK-M/G/1,
    # V100, or changed-engine-regime runs merely because their dates are newer.
    original = source/"experiments"
    reference_dirs = ("router_qps_sweep_38498357_2026-04-06_070030",
        "router_qps_sweep_38498366_2026-04-06_222644", "router_qps_sweep_38725846_2026-04-10_034010",
        "router_qps_sweep_38498376_2026-04-07_001106", "router_qps_sweep_no_snapshot_38531619_2026-04-07_080337",
        "router_qps_sweep_no_snapshot_38531626_2026-04-07_081928", "router_qps_sweep_no_snapshot_38635156_2026-04-08_153554",
        "router_qps_sweep_38498349_2026-04-06_053230", "router_qps_sweep_38596188_2026-04-07_182115",
        "router_qps_sweep_38698462_2026-04-09_212250", "router_qps_sweep_38703672_2026-04-10_005848",
        "router_qps_sweep_no_snapshot_38635146_2026-04-08_144928", "router_qps_sweep_no_snapshot_38698417_2026-04-09_205330",
        "router_qps_sweep_no_snapshot_38698456_2026-04-09_212250")
    references, full_references = {}, {}
    for folder in reference_dirs:
        for path in sorted((original/folder).glob("*_point*.json")):
            if not re.search(r"_point\d+\.json$", path.name): continue
            payload = json.loads(path.read_text())
            config = payload["config"]
            qps = config["request_rate_qps"]
            if qps not in set(QPS)|set(PAPER_QPS):
                continue
            if any(config.get(k) != v for k, v in {"num_requests": 16000, "seed": 69,
                "arrival_process": "poisson", "lambda_weight": .0005, "delta_weight": 0,
                "max_completion_tokens": 8192, "temperature": 0, "top_p": 1}.items()):
                raise ValueError(f"Canonical reference configuration differs: {path}")
            for run in payload["router"]["runs"]:
                if run["utility"] not in ("hard", "shortest_queue", "latency_agnostic", "round_robin"):
                    continue
                if len(run["per_request"]) != 16000 or any(r.get("error") for r in run["per_request"]):
                    raise ValueError(f"Incomplete reference policy: {path}")
                if any(req.bucket != r["bucket"] or req.prompt_tokens != r["prompt_tokens"] or
                       abs(req.ttft_slo_ms-r["ttft_slo_ms"]) > 1e-8
                       for req, r in zip(requests, sorted(run["per_request"], key=lambda r:int(r["request_id"].split('-')[-1])))):
                    raise ValueError(f"Canonical reference workload/SLO differs: {path}")
                key = f"{qps:g}:{run['utility']}"
                full_references[key] = str(path)
                if qps in QPS: references[key] = str(path)
    if not full_references:
        raise ValueError("No verified canonical historical reference cells")
    model_paths = [root.parent/".cache/huggingface/hub"/f"models--Qwen--{name}"/"snapshots"/pin for name,pin in zip(HF_NAMES,PINS)]
    files = [reqmap, cache/"manifest.json", *(cache/f"{b}.jsonl" for b in BUCKETS),
             root/"src/assets/templates/chat_template_qwen3.jinja"]
    files += [Path(p) for p in set(full_references.values())]
    files += [p for name in ("accuracy_predictor", "output_length_predictor") for p in (root/"src/assets/predictors"/name).glob("*") if p.is_file()]
    for path in model_paths:
        index = path/"model.safetensors.index.json"
        if index.exists():
            shards = set(json.loads(index.read_text())["weight_map"].values())
        else:
            shards = {"model.safetensors"}
        if any(not (path/s).is_file() or (path/s).stat().st_size == 0 for s in shards):
            raise ValueError(f"Missing pinned model weights: {path}")
        files += [path/"config.json", path/"tokenizer_config.json"]
    output.mkdir(parents=True)
    cal_cache = output/"calibration_cache"
    cal_cache.mkdir()
    for bucket in BUCKETS:
        shutil.copy2(prepared/"calibration"/MODELS[0]/f"{bucket}_scored.jsonl", cal_cache/f"{bucket}.jsonl")
    args = _parse_experiment_args(argv)
    calibration = exp.build_experiment_requests(cal_cache, num_requests=10000, seed=69,
        **{k: getattr(args,k) for k in ("slo_min_ms", "slo_max_ms", "queue_slo_min_ms", "queue_slo_max_ms",
           "ttft_slo_min_ms", "ttft_slo_max_ms", "ttft_slo_base_ms", "ttft_slo_per_prompt_token_ms",
           "ttft_slo_jitter_min", "ttft_slo_jitter_max")}, prompt_sampling_mode="mixed_then_shuffle", holdout_prompts_per_bucket=2500)
    request_path = output/"calibration_requests.jsonl"
    with request_path.open("x") as stream:
        for req in calibration:
            stream.write(json.dumps(asdict(req))+"\n")
    warmup_path = output/"latency_warmup.json"
    write_json(warmup_path, {"data_role":"calibration", "requests":[asdict(r) for r in smoke_requests(calibration, per_bucket=8)]})
    from sfs_core.routing.latency_warmup import load_warmup
    load_warmup(warmup_path)
    argv += ["--latency-warmup-requests", str(warmup_path)]
    files += [request_path, warmup_path, prepared/"data_audit.json"]
    manifest = {"schema_version": 2, "status": "PREPARED", "sfs_root": str(root), "prepared_data": str(prepared),
        "manifest_path": str(output/"qwen_manifest.json"), "requests_per_cell": 16000,
        "policies": list(POLICIES), "new_policies": list(NEW_POLICIES), "qps_values": QPS,
        "profile": PROFILE, "experiment_argv": argv, "request_map": str(reqmap),
        "canonical_reference_cells": references,
        "missing_reference_cells": [f"{q:g}:{p}" for q in QPS for p in ("hard","shortest_queue","latency_agnostic","round_robin") if f"{q:g}:{p}" not in references],
        "latency_warmup": str(warmup_path), "model_paths": list(map(str, model_paths)),
        "canonical_full_curve_cells": full_references,
        "calibration_requests": str(request_path), "calibration_groups": len(calibration),
        "file_sha256": {str(p): sha256(p) for p in files},
        "runtime_gates": ["fresh Qwen timing inputs", "native Qwen RouteBalance predictor", "all-nine-policy GPU smoke"],
        "gpu_submitted": False}
    write_json(output/"qwen_manifest.json", manifest)
    write_json(output/"instances_cpu.json", pool_config(manifest))
    return manifest


def validate_manifest(path):
    from scripts.runs import experiments as exp
    manifest = json.loads(Path(path).read_text())
    if (manifest.get("schema_version") != 2 or manifest.get("requests_per_cell") != 16000 or manifest.get("policies") != list(POLICIES)
            or manifest.get("qps_values") != list(QPS) or manifest.get("profile") != PROFILE):
        raise ValueError("Incorrect canonical Qwen experiment contract")
    for name, expected in manifest["file_sha256"].items():
        if sha256(name) != expected:
            raise ValueError(f"Canonical input changed: {name}")
    root = Path(manifest["sfs_root"])
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin or Path(spec.origin).resolve() != root/"vllm/vllm/__init__.py":
        raise ValueError("Active editable vLLM import differs from the selected worktree")
    instances, costs, meta = exp.load_instances(Path(path).parent/"instances_cpu.json")
    try:
        if set(c.model_id for c in instances.values()) != set(MODELS) or set(costs) != set(instances) or meta["serving_profile"] != PROFILE:
            raise ValueError("Invalid Qwen instances loader contract")
    finally:
        for client in instances.values(): client.close()
    return manifest


def server_argv(root, model_path, row, index, output):
    argv = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(model_path),
        "--served-model-name", row["model_id"], "--disable-uvicorn-access-log", "--dtype", "auto",
        "--max-model-len", "131072", "--rope-scaling", json.dumps(PROFILE["rope_scaling"]),
        "--chat-template", str(root/"src/assets/templates/chat_template_qwen3.jinja"),
        "--enable-chunked-prefill", "--no-enable-prefix-caching", "--max-num-batched-tokens", "32768",
        "--max-num-seqs", "512", "--gpu-memory-utilization", "0.9", "--tensor-parallel-size", str((1,1,2)[index]),
        "--batch-stats-file", str(output/f"batch_stats_{row['model_id']}.csv"),
        "--port", row["address"].rsplit(":",1)[-1], "--no-enable-wait-time-simulation",
        "--enable-snapshot-shm-publishing", "--snapshot-shm-name", row["snapshot_shm_name"],
        "--snapshot-shm-size-bytes", str(row["snapshot_shm_size_bytes"]), "--snapshot-shm-publish-interval-ms", "0",
        "--output-length-model-path", str(root/"src/assets/predictors/output_length_predictor")]
    # The feature set decides whether vLLM reads the sixth coefficient as the sum of squared context
    # lengths or as the prefill x processed-context cross term, so it travels with the coefficients.
    argv += build_simulation_args({"sfs_simulation": {**row["ttft_batch_model"],
                                                     "feature_set": row.get("batch_time_feature_set", FEATURE_SET)}})
    return argv


async def calibrate(manifest, output, instances_path, predictor):
    from scripts.runs import experiments as exp
    from scripts.runs.experiments_sweep import _parse_experiment_args
    from scripts.prep.fit_methodology_calibration import fit_manifest, load_trace_rows
    from sfs_core.shared.shared_experiment_helpers import build_messages, warm_up_instances
    from sfs_core.shared.trace_theta import estimate_score_proxy_metrics_from_batch_stats
    requests = [exp.ExperimentRequest(**r) for r in rows(manifest["calibration_requests"])]
    instances, costs, metadata = exp.load_instances(instances_path)
    service, recorded = {}, []
    try:
        await warm_up_instances(list(instances.values()))
        await wait_drained(instances)
        probes = length_stratified_requests(requests)
        sample = smoke_requests(requests, per_bucket=128)
        async def measure(client):
            for index, req in enumerate(probes):
                reply = await client.submit_request(messages=build_messages(req.prompt,
                    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."),
                    temperature=0, top_p=1, max_completion_tokens=1,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}, "request_id": f"probe-{client.model_id}-{index}"})
                if reply.usage is None or not reply.id:
                    raise ValueError("Missing measured probe response/usage")
                recorded.append({"model": client.model_id, "probe_id": req.request_id,
                                 "response_id": reply.id, "usage": reply.usage.model_dump()})
                await wait_drained({client.instance_id: client})
            semaphore = asyncio.Semaphore(128)
            async def submit(index, req):
                async with semaphore:
                    reply = await client.submit_request(messages=build_messages(req.prompt,
                        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."),
                        temperature=0, top_p=1, max_completion_tokens=8192,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}, "request_id": f"rate-{client.model_id}-{index}"})
                    if reply.usage is None or not reply.id:
                        raise ValueError("Missing measured service response/usage")
                    recorded.append({"model": client.model_id, "request_id": req.request_id,
                                     "response_id": reply.id, "usage": reply.usage.model_dump()})
            started = time.monotonic()
            await asyncio.gather(*(submit(i,r) for i,r in enumerate(sample)))
            elapsed = time.monotonic()-started
            service[client.model_id] = {"service_rate_qps": len(sample)/elapsed, "num_queries": len(sample),
                "succeeded": len(sample), "failed": 0, "elapsed_s": elapsed,
                "service_rate_definition": "512 calibration responses / bounded-concurrency whole-run elapsed including drain; not router capacity"}
        await asyncio.gather(*(measure(client) for client in instances.values()))
        await wait_drained(instances)
        await asyncio.sleep(2)
        write_json(output/"calibration_responses.json", {"data_role": "calibration", "responses": recorded})
        fit_input = {"data_role": "calibration", "serving_profile_verified": True, "serving_profile": PROFILE, "models": {}}
        for model in MODELS:
            trace = output/f"batch_stats_{model}.csv"
            frozen = output/f"calibration_trace_{model}.csv"
            shutil.copy2(trace, frozen)
            values, _ = load_trace_rows([frozen])
            positive = [r for r in values if r["prefill"] > 0]
            if not positive:
                raise ValueError("Missing prefill trace observations")
            proxy = estimate_score_proxy_metrics_from_batch_stats(batch_stats_csv_path=frozen, batch_stats_offset=0)
            proxy["prefill_tps"] = sum(r["prefill"] for r in positive)/sum(r["exec"] for r in positive)
            service[model]["score_proxy"] = proxy
            fit_input["models"][model] = {**service[model], "traces": [str(frozen)]}
        write_json(output/"model_metrics.json", service)
        write_json(output/"service_manifest.json", fit_input)
        timing = await asyncio.to_thread(fit_manifest, output/"service_manifest.json", output/"timing_models")
        args = _parse_experiment_args([*manifest["experiment_argv"], "--service-metrics-json", str(output/"model_metrics.json"),
            "--methodology-calibration-json", str(timing), "--routebalance-predictor-path", str(predictor)])
        args.per_request_wait_log = [str(output/f"wait_{model}.log") for model in MODELS]
        warmup_path = output/"latency_warmup.json"
        write_json(warmup_path, {"data_role":"calibration", "requests":[asdict(r) for r in smoke_requests(requests, per_bucket=8)]})
        args.latency_warmup_requests = str(warmup_path)
        smoke = smoke_requests(requests)
        for policy in POLICIES:
            args.utilities, args.num_requests, args.request_rate_qps = [policy], len(smoke), 2.0
            await wait_drained(instances)
            result = await exp.run_router_experiment(args=args, requests=smoke, instances=instances,
                instance_costs=costs, instance_metadata=metadata,
                response_map_base_path=output/f"smoke_{policy}_responses.log", request_log_base_path=output/f"smoke_{policy}_waits.log")
            write_json(output/f"smoke_{policy}.json", result)
            audit_run(result["runs"][0], len(smoke))
        await wait_drained(instances)
        from scripts.runs.latency_validation import source_hashes
        write_json(output/"latency_smoke_gate.json", {"status":"PASS_GPU_SMOKE", "requests":len(smoke),
            "source_sha256":source_hashes(Path(manifest.get("sfs_root", Path(__file__).resolve().parents[3]))),
            "result_sha256":sha256(output/"smoke_vllm_sr_latency.json")})
        write_json(output/"smoke_audit.json", {"status": "PASS", "policies": list(POLICIES),
            "profile": PROFILE, "manifest_sha256": sha256(manifest["manifest_path"]),
            "timing_sha256": sha256(timing), "metrics_sha256": sha256(output/"model_metrics.json"),
            "predictor_metadata_sha256": sha256(Path(predictor)/"metadata.json"), "evaluation_started": False})
    finally:
        for client in instances.values(): client.close()


def validate_reusable_calibration(stage, predictor):
    stage = Path(stage)
    audit = json.loads((stage/"smoke_audit.json").read_text())
    if (audit.get("status") != "PASS" or audit.get("profile") != PROFILE
            or audit.get("timing_sha256") != sha256(stage/"timing_models/methodology_calibration.json")
            or audit.get("metrics_sha256") != sha256(stage/"model_metrics.json")
            or audit.get("predictor_metadata_sha256") != sha256(Path(predictor)/"metadata.json")):
        raise ValueError("Reusable measured calibration artifact gate failed")
    from sfs_core.routing.methodology_calibration import MethodologyCalibration
    calibration = MethodologyCalibration.load(stage/"timing_models/methodology_calibration.json")
    calibration.validate_runtime_profile(PROFILE)
    calibration.preload()


async def smoke_existing(manifest, stage, output, instances_path, predictor, policies):
    """Reuse measured calibration; this mode never runs an evaluation cell."""
    from scripts.runs import experiments as exp
    from scripts.runs.experiments_sweep import _parse_experiment_args
    from scripts.runs.latency_validation import source_hashes
    from sfs_core.shared.shared_experiment_helpers import warm_up_instances
    validate_reusable_calibration(stage,predictor)
    sources = source_hashes(manifest["sfs_root"])
    shutil.copy2(stage/"model_metrics.json",output/"model_metrics.json")
    # The prefill bootstrap rechecks these original measured response IDs.
    shutil.copy2(stage/"calibration_responses.json",output/"calibration_responses.json")
    (output/"timing_models").symlink_to((stage/"timing_models").resolve(),target_is_directory=True)
    args = _parse_experiment_args([*manifest["experiment_argv"], "--service-metrics-json", str(output/"model_metrics.json"),
        "--methodology-calibration-json", str(output/"timing_models/methodology_calibration.json"),
        "--routebalance-predictor-path",str(predictor)])
    args.per_request_wait_log = [str(output/f"wait_{m}.log") for m in MODELS]
    requests = [exp.ExperimentRequest(**r) for r in rows(manifest["calibration_requests"])]
    smoke = smoke_requests(requests)
    instances,costs,metadata=exp.load_instances(instances_path)
    try:
        # HTTP readiness precedes the first scheduler snapshot on a cold server.
        await warm_up_instances(list(instances.values()))
        for policy in policies:
            await wait_drained(instances)
            args.utilities,args.num_requests,args.request_rate_qps=[policy],len(smoke),2.
            result=await asyncio.wait_for(exp.run_router_experiment(args=args,requests=smoke,instances=instances,instance_costs=costs,
                instance_metadata=metadata,response_map_base_path=output/f"smoke_{policy}_responses.log",
                request_log_base_path=output/f"smoke_{policy}_waits.log"), timeout=1200)
            write_json(output/f"smoke_{policy}.json",result)
            audit_run(result["runs"][0],len(smoke))
        await wait_drained(instances)
        if source_hashes(manifest["sfs_root"]) != sources:
            raise ValueError("Sources changed during Qwen smoke; preserve results for review")
        if "vllm_sr_latency" in policies:
            write_json(output/"latency_smoke_gate.json",{"status":"PASS_GPU_SMOKE","requests":len(smoke),
                "source_sha256":source_hashes(manifest["sfs_root"]),"result_sha256":sha256(output/"smoke_vllm_sr_latency.json")})
        write_json(output/"smoke_audit.json",{"status":"PASS","policies":list(policies),"profile":PROFILE,
            "manifest_sha256":sha256(manifest["manifest_path"]),"timing_sha256":sha256(output/"timing_models/methodology_calibration.json"),
            "metrics_sha256":sha256(output/"model_metrics.json"),"predictor_metadata_sha256":sha256(Path(predictor)/"metadata.json"),
            "evaluation_started":False,"reused_calibration":str(stage)})
    finally:
        for client in instances.values():client.close()


def run_sweep(manifest, stage, output, instances_path, predictor, timing_gate=None, policies=None, qps_values=None):
    policies = tuple(policies or NEW_POLICIES)
    rates = tuple(qps_values or QPS)
    if len(set(rates)) != len(rates) or not set(rates).issubset(QPS):
        raise ValueError("Invalid canonical Qwen QPS subset")
    if len(set(policies)) != len(policies) or not set(policies).issubset(NEW_POLICIES):
        raise ValueError("Invalid Qwen policy subset")
    if "mooncake_prefill" in policies and timing_gate is None:
        raise ValueError("Mooncake evaluation requires the corrected-prefill bootstrap timing gate")
    timing = stage/"timing_models/methodology_calibration.json"
    if timing_gate is not None:
        timing = Path(timing_gate(instances_path, output))
    argv = [sys.executable, "-m", "scripts.runs.experiments_sweep", "--sweep", "qps",
        "--qps-values", *map(str,rates), "--qps-utilities", *policies, "--output-dir", str(output/"outputs"),
        "--output-prefix", "qwen_baselines", *manifest["experiment_argv"], "--instances-config", str(instances_path),
        "--service-metrics-json", str(stage/"model_metrics.json"), "--methodology-calibration-json",
        str(timing), "--routebalance-predictor-path", str(predictor)]
    for model in MODELS: argv += ["--per-request-wait-log", str(output/f"wait_{model}.log")]
    from scripts.runs.latency_validation import source_hashes
    sources = source_hashes(manifest["sfs_root"])
    write_json(output/"sweep_started.json", {"source_sha256":sources,"policies":policies,
        "qps_values":rates,"requests_per_cell":16000,"manifest_sha256":sha256(manifest["manifest_path"])})
    with (output/"driver.log").open("x") as log:
        subprocess.run(argv, cwd=Path(manifest["sfs_root"])/"src", stdout=log, stderr=subprocess.STDOUT, check=True)
    if source_hashes(manifest["sfs_root"]) != sources:
        raise ValueError("Sources changed during Qwen evaluation; preserve outputs for review")
    write_json(output/"sweep_completed.json", {"status": "COMPLETE_UNCOLLATED",
        "source_sha256":sources, "manifest_sha256": sha256(manifest["manifest_path"]), "timing_sha256": sha256(timing)})


def start_and_run(options, manifest, *, timing_gate=None):
    from scripts.runs.latency_validation import validate_cpu
    validate_cpu(getattr(options,"cpu_validation",None),manifest["sfs_root"])
    from sfs_core.routing.routebalance_predictor import RouteBalancePredictor
    predictor = Path(options.predictor).resolve()
    RouteBalancePredictor.load(predictor)  # Fail before loading any GPU model.
    root, output = Path(manifest["sfs_root"]), Path(options.output_root).resolve()
    if output.exists(): raise ValueError("Choose a new GPU output root")
    job = os.environ.get("SLURM_JOB_ID", "")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if not job.isdecimal() or len(visible) != 4:
        raise ValueError("Requires a four-GPU Slurm allocation")
    if options.mode == "sweep":
        validate_smoke(options.stage_dir, manifest, predictor, getattr(options,"policies",None) or NEW_POLICIES)
    elif options.mode == "smoke":
        validate_reusable_calibration(options.stage_dir,predictor)
    output.mkdir(parents=True)
    local = Path("/local")/os.environ["USER"]/f"sfs-qwen-{job}"
    local.mkdir(parents=True, exist_ok=False)
    from scripts.runs.serving_ipc import qwen_ipc_environment
    server_env = qwen_ipc_environment(local, os.environ)
    ports = [20000+(int(job)%10000)*3+i for i in range(3)]
    config = pool_config(manifest, ports, job)
    instances_path = output/"instances.json"
    write_json(instances_path, config)
    processes, logs = [], []
    try:
        for i, row in enumerate(config["instances"]):
            dest = local/HF_NAMES[i]
            dest.mkdir()
            subprocess.run(["rsync", "-aL", str(manifest["model_paths"][i])+"/", str(dest)+"/"], check=True)
            argv = server_argv(root, dest, row, i, output)
            env = dict(server_env, CUDA_VISIBLE_DEVICES=visible[i] if i<2 else ",".join(visible[2:]),
                VLLM_ATTENTION_BACKEND="FLASH_ATTN", VLLM_USE_FLASHINFER_SAMPLER="0",
                VLLM_PER_REQUEST_WAIT_LOG_PATH=str(output/f"wait_{row['model_id']}.log"),
                XDG_CACHE_HOME=str(local/".cache"), TRITON_CACHE_DIR=str(local/".triton"), CUDA_CACHE_PATH=str(local/".nv"))
            log = (output/f"server_{row['model_id']}.log").open("x"); logs.append(log)
            write_json(output/f"server_argv_{row['model_id']}.json", argv)
            processes.append(subprocess.Popen(argv, cwd=root/"src", env=env, stdout=log, stderr=subprocess.STDOUT))
        import urllib.request
        deadline = time.monotonic()+1200
        for row in config["instances"]:
            while True:
                if any(p.poll() is not None for p in processes):
                    raise RuntimeError("Qwen server exited before readiness; see preserved server logs")
                try:
                    with urllib.request.urlopen(row["address"]+"/v1/models", timeout=2) as response:
                        models = json.load(response)
                    if row["model_id"] in {r["id"] for r in models["data"]}: break
                except (OSError, ValueError): pass
                if time.monotonic() > deadline: raise TimeoutError("Qwen pool readiness deadline")
                time.sleep(2)
        if options.mode == "calibrate":
            asyncio.run(calibrate(manifest, output, instances_path, predictor))
        elif options.mode == "smoke":
            asyncio.run(smoke_existing(manifest,Path(options.stage_dir),output,instances_path,predictor,
                getattr(options,"policies",None) or POLICIES))
        else:
            stage = Path(options.stage_dir).resolve()
            run_sweep(manifest, stage, output, instances_path, predictor, timing_gate, getattr(options,"policies",None), getattr(options,"qps_values",None))
    finally:
        for process in processes:
            if process.poll() is None: process.terminate()
        for process in processes:
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
        for log in logs: log.close()


def validate_smoke(stage_dir, manifest, predictor, policies=None):
    policies = tuple(policies or POLICIES)
    if stage_dir is None: raise ValueError("Sweep requires --stage-dir with successful GPU smoke")
    stage = Path(stage_dir)
    from scripts.runs.latency_validation import validate_selector_smoke
    audit = json.loads((stage/"smoke_audit.json").read_text())
    if (audit.get("status") != "PASS" or not set(policies).issubset(audit.get("policies",[])) or audit.get("profile") != PROFILE
            or audit.get("manifest_sha256") != sha256(manifest["manifest_path"])
            or audit.get("timing_sha256") != sha256(stage/"timing_models/methodology_calibration.json")
            or audit.get("metrics_sha256") != sha256(stage/"model_metrics.json")
            or audit.get("predictor_metadata_sha256") != sha256(Path(predictor)/"metadata.json")):
        raise ValueError("Qwen sweep calibration/smoke gate did not pass")
    if "vllm_sr_latency" in policies:
        validate_selector_smoke(stage, manifest.get("sfs_root",Path(__file__).resolve().parents[3]))



def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "validate", "calibrate", "smoke", "sweep"))
    p.add_argument("--sfs-root", type=Path, default=Path(os.environ.get("SFS_ROOT", ".")))
    p.add_argument("--prepared-dir", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--output-root", type=Path)
    p.add_argument("--reuse-calibration-only", action="store_true")
    p.add_argument("--cpu-validation", type=Path)
    p.add_argument("--qps-values", nargs="+", type=float, choices=QPS)
    p.add_argument("--policies", nargs="+", choices=NEW_POLICIES)
    p.add_argument("--predictor", type=Path)
    p.add_argument("--stage-dir", type=Path)
    a = p.parse_args()
    if a.mode == "prepare":
        if a.prepared_dir is None or a.output_root is None: p.error("prepare requires input/output paths")
        result = prepare(a.sfs_root, a.prepared_dir, a.output_root)
        print(json.dumps({"status": result["status"], "manifest": result["manifest_path"]}))
    else:
        if a.manifest is None: p.error("--manifest is required")
        manifest = validate_manifest(a.manifest)
        if a.mode == "validate":
            if a.cpu_validation is not None:
                from scripts.runs.latency_validation import validate_cpu
                validate_cpu(a.cpu_validation,manifest["sfs_root"])
            from scripts.runs.serving_ipc import validate_ipc_paths
            local = Path("/local")/os.environ["USER"]/"sfs-qwen-99999999"
            ipc_contract = validate_ipc_paths(local/"ipc", local/"tmp")
            scratch = Path(manifest["sfs_root"])/"experiments/ipc"
            socket_bind = validate_ipc_paths(scratch, scratch, bind=True)
            if a.predictor is not None:
                from sfs_core.routing.routebalance_predictor import RouteBalancePredictor
                predictor = RouteBalancePredictor.load(a.predictor)
                if set(predictor.model_labels) != set(MODELS):
                    raise ValueError("RouteBalance predictor family differs from Qwen")
                predictor.predict_batch(["Summarize why the sky appears blue."])
            if a.stage_dir is not None:
                if a.reuse_calibration_only:
                    validate_reusable_calibration(a.stage_dir,a.predictor)
                else:
                    validate_smoke(a.stage_dir, manifest, a.predictor, a.policies or NEW_POLICIES)
            from scripts.runs.serving_contract_preflight import validate as validate_serving
            serving_contract = validate_serving(Path(manifest["sfs_root"]), "qwen",
                Path(manifest["sfs_root"])/"src/assets/predictors/output_length_predictor")
            print(json.dumps({"status": "PASS", "requests_per_cell": 16000, "new_policies": NEW_POLICIES,
                              "qps": QPS, "gpu_executed": False, "serving_contract": serving_contract,
                              "ipc_contract": ipc_contract, "cpu_socket_bind": socket_bind}))
        else:
            if a.predictor is None or a.output_root is None: p.error("GPU modes require predictor/output paths")
            start_and_run(a, manifest)


if __name__ == "__main__": main()

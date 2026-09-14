"""Qualify a prefill-only correction on the loaded Qwen evaluation pool."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from scripts.runs import qwen_baselines as qwen
from scripts.runs.ministral3_prefill_bootstrap import (
    read_json, validate_prefill_revision, validate_smoke,
)
from scripts.runs.capacity_scout import TrialMonitor


def validate_inputs(options, manifest, *, ingest=False):
    stage = Path(options.stage_dir).resolve()
    qwen.validate_smoke(stage, manifest, options.predictor)
    revision = validate_prefill_revision(stage/"timing_models/methodology_calibration.json",
        options.candidate, options.candidate_audit, options.timing_review,
        models=qwen.MODELS, profile=qwen.PROFILE)
    from scripts.runs import experiments as exp
    from scripts.runs.experiments_sweep import _parse_experiment_args
    requests = [exp.ExperimentRequest(**r) for r in qwen.rows(manifest["calibration_requests"])]
    if (len(requests) != 10000 or
            Counter(r.bucket for r in requests) != Counter({b: 2500 for b in qwen.BUCKETS})):
        raise ValueError("Qwen smoke requires the balanced calibration pool")
    expected = {(r.request_id, r.bucket, r.prompt_tokens) for r in qwen.smoke_requests(requests)}
    files = {**manifest["file_sha256"], **revision["file_sha256"]}
    for policy in qwen.POLICIES:
        path = stage/f"smoke_{policy}.json"
        payload = read_json(path)
        if len(payload.get("runs", [])) != 1 or payload["runs"][0].get("utility") != policy:
            raise ValueError("Unexpected original Qwen policy smoke")
        run = payload["runs"][0]
        qwen.audit_run(run, 192)
        if {(r["request_id"], r["bucket"], r["prompt_tokens"]) for r in run["per_request"]} != expected:
            raise ValueError("Original Qwen smoke identity differs")
        files[str(path)] = qwen.sha256(path)
    original = read_json(stage/"timing_models/methodology_calibration.json")
    candidate = read_json(options.candidate)
    for model in qwen.MODELS:
        traces = (original["models"][model]["trace_provenance"] +
                  candidate["models"][model]["prefill"]["calibration_row_selection"]["trace_provenance"])
        for trace in traces:
            if qwen.sha256(trace["path"]) != trace["sha256"]:
                raise ValueError("Measured Qwen fit input changed")
            files[trace["path"]] = trace["sha256"]
    for path in (stage/"smoke_audit.json", stage/"model_metrics.json", stage/"calibration_responses.json",
                 Path(manifest["manifest_path"]), Path(options.predictor)/"metadata.json", Path(__file__)):
        files[str(path.resolve())] = qwen.sha256(path)
    ingestion = None
    if ingest:
        args = _parse_experiment_args([*manifest["experiment_argv"],
            "--service-metrics-json", str(stage/"model_metrics.json"),
            "--methodology-calibration-json", str(options.candidate)])
        evaluation, _, _ = exp._build_request_set(args)
        with Path(manifest["request_map"]).open() as stream:
            mapped = {r["req_id"]: r for r in csv.DictReader(stream)}
        cache = Path(args.holdout_cache_dir)
        cached = {(b, r["prompt_metadata"]["example_id"]): r["prompt"]
                  for b in qwen.BUCKETS for r in qwen.rows(cache/f"{b}.jsonl")}
        if len(evaluation) != 16000 or len(mapped) != 16000:
            raise ValueError("Canonical Qwen evaluation must retain 16000 requests")
        if Counter(r.bucket for r in evaluation) != Counter({b: 4000 for b in qwen.BUCKETS}):
            raise ValueError("Canonical Qwen bucket balance differs")
        for request in evaluation:
            row = mapped[request.request_id]
            if (request.bucket != row["bucket"] or request.prompt_tokens != int(row["prompt_tokens"]) or
                    request.prompt != cached[(row["bucket"], row["example_id"])]):
                raise ValueError("Actual Qwen ingestion differs from the canonical request map")
        ingestion = {"status": "PASS_CPU", "requests": 16000, "per_bucket": 4000}
    return {"status": "PASS_CPU", "gpu_executed": False, "revision": revision,
            "file_sha256": files, "evaluation_ingestion": ingestion}


async def qualify(options, manifest, inputs, instances_path, output):
    from scripts.runs import experiments as exp
    from scripts.runs.experiments_sweep import _parse_experiment_args
    from sfs_core.shared.shared_experiment_helpers import warm_up_instances
    output, stage = Path(output), Path(options.stage_dir)
    folder = output/"prefill_smoke"
    folder.mkdir(exist_ok=False)
    args = _parse_experiment_args([*manifest["experiment_argv"],
        "--service-metrics-json", str(stage/"model_metrics.json"),
        "--methodology-calibration-json", str(options.candidate),
        "--routebalance-predictor-path", str(options.predictor)])
    args.instances_config = Path(instances_path)
    args.utilities, args.num_requests, args.request_rate_qps = ["mooncake_prefill"], 192, 2.0
    args.per_request_wait_log = [str(output/f"wait_{m}.log") for m in qwen.MODELS]
    requests = qwen.smoke_requests([exp.ExperimentRequest(**r)
                                   for r in qwen.rows(manifest["calibration_requests"])])
    instances, costs, metadata = exp.load_instances(instances_path)
    try:
        if metadata.get("serving_profile") != qwen.PROFILE or {c.model_id for c in instances.values()} != set(qwen.MODELS):
            raise ValueError("Loaded Qwen serving configuration differs")
        await warm_up_instances(list(instances.values()))
        await qwen.wait_drained(instances)
        monitor = TrialMonitor(folder/"events.jsonl", duration_s=1200, max_outstanding=512)
        payload = await asyncio.wait_for(exp.run_router_experiment(args=args, requests=requests,
            instances=instances, instance_costs=costs, instance_metadata=metadata,
            response_map_base_path=folder/"responses.log", request_log_base_path=folder/"waits.log",
            trial_monitor=monitor), timeout=1200)
        raw = folder/"smoke_mooncake_prefill.json"
        qwen.write_json(raw, payload)
        await qwen.wait_drained(instances)
        audit = validate_smoke(payload, options.candidate, requests, instance_ids=set(instances))
        for name, expected in inputs["file_sha256"].items():
            if qwen.sha256(name) != expected:
                raise ValueError(f"Validated Qwen input changed: {name}")
        audit.update(data_role="calibration", raw_smoke_sha256=qwen.sha256(raw),
                     events_sha256=qwen.sha256(folder/"events.jsonl"),
                     original_smoke_audit_sha256=qwen.sha256(stage/"smoke_audit.json"),
                     prefill_revision=inputs["revision"], file_sha256=inputs["file_sha256"],
                     evaluation_requests_per_cell=16000)
        qwen.write_json(folder/"audit.json", audit)
        derived = output/"qualified_timing"
        shutil.copytree(Path(options.candidate).resolve().parent, derived)
        timing = derived/"methodology_calibration.json"
        if qwen.sha256(timing) != audit["timing_calibration_sha256"]:
            raise ValueError("Qualified Qwen timing copy changed")
        return timing
    finally:
        for client in instances.values():
            client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate", "run"))
    for name in ("manifest", "stage-dir", "predictor", "candidate", "candidate-audit", "timing-review"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cpu-validation", type=Path)
    parser.add_argument("--qps-values", nargs="+", type=float, choices=qwen.QPS)
    parser.add_argument("--policies", nargs="+", choices=qwen.NEW_POLICIES)
    options = parser.parse_args()
    manifest = qwen.validate_manifest(options.manifest)
    inputs = validate_inputs(options, manifest, ingest=options.mode == "validate")
    if options.mode == "validate":
        print(json.dumps(inputs))
        return
    if options.output_root is None:
        parser.error("Run requires --output-root")
    def timing_gate(instances, output):
        return asyncio.run(qualify(options, manifest, inputs, instances, output))
    run_options = SimpleNamespace(**{**vars(options), "mode": "sweep"})
    qwen.start_and_run(run_options, manifest, timing_gate=timing_gate)


if __name__ == "__main__":
    main()

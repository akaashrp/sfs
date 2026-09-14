"""Validate revised prefill timing, then reuse the pool for Figure 5 baselines.

The original shortest-queue scout and seven unaffected policy smokes stay raw.
A derived stage records the changed timing artifact and a fresh Mooncake smoke.
No evaluation request is sent until composition and the original freeze gates pass.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from copy import deepcopy
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

from scripts.prep.prepare_methodology_service import MODELS, PROFILE, sha256
from scripts.runs.capacity_scout import TrialMonitor, classify_trial, freeze_loads
from scripts.runs.ministral3_methodology_stage import (
    BUCKETS, POLICIES, audit_run, smoke_requests, validate_pool,
    validate_resume_stage, wait_drained, write_json,
)
from scripts.runs import ministral3_figure5 as figure5


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_prefill_revision(original_path, candidate_path, audit_path, review_path,
                             *, models=MODELS, profile=PROFILE):
    """Only the approved prefill fit may change; all other policy inputs match."""
    from sfs_core.routing.methodology_calibration import MethodologyCalibration
    original, candidate = map(MethodologyCalibration.load, (original_path, candidate_path))
    candidate.validate_runtime_profile(profile)
    original.validate_runtime_profile(profile)
    audit, review = read_json(audit_path), read_json(review_path)
    original_hash, candidate_hash = sha256(original_path), sha256(candidate_path)
    if (candidate.payload.get("parent_calibration_sha256") != original_hash or
            audit.get("status") != "CPU_VALIDATED_GPU_SMOKE_REQUIRED" or
            audit.get("candidate_sha256") != candidate_hash or
            audit.get("parent_calibration_sha256") != original_hash or
            review.get("candidate_timing_calibration_sha256") != candidate_hash or
            review.get("original_timing_calibration_sha256") != original_hash or
            review.get("status") != "PREFILL_CORRECTION_REQUIRES_MATCHED_GPU_SMOKE"):
        raise ValueError("Revised timing is not the CPU-validated and reviewed prefill artifact")
    for key in ("prefill_fit_review", "tpot_fit_review", "arrival_review"):
        if not isinstance(review.get(key), str) or not review[key].strip():
            raise ValueError("Substantive timing and arrival review is required")
    allowed = {"models", "created_at", "parent_calibration_sha256", "prefill_correction"}
    if ({k: v for k, v in original.payload.items() if k not in allowed} !=
            {k: v for k, v in candidate.payload.items() if k not in allowed} or
            set(original.models) != set(candidate.models) or set(candidate.models) != set(models)):
        raise ValueError("The prefill revision changed unrelated calibration metadata")
    files = {str(Path(p).resolve()): sha256(p)
             for p in (original_path, candidate_path, audit_path, review_path)}
    for model in models:
        old, new = original.models[model], candidate.models[model]
        if ({k: v for k, v in old.items() if k != "prefill"} !=
                {k: v for k, v in new.items() if k != "prefill"}):
            raise ValueError(f"TPOT, service rates or provenance changed for {model}")
        for calibration in (original, candidate):
            head = calibration.path.parent / calibration.models[model]["tpot"]["model_file"]
            files[str(head)] = sha256(head)
        if new["prefill"]["coverage"]["fit_rows"] < 40:
            raise ValueError("Insufficient retained prefill calibration")
    candidate.preload()
    return {"status": "PASS_CPU", "original_sha256": original_hash,
            "candidate_sha256": candidate_hash, "file_sha256": files,
            "changed_policy": "mooncake_prefill", "unaffected_policies": [
                policy for policy in POLICIES if policy != "mooncake_prefill"]}


def validate_completed_scout(stage, reuse):
    """Reproduce the measured bracket and both endpoint decisions from events."""
    stage = Path(stage).resolve()
    path = stage / "capacity_scout.json"
    scout = read_json(path)
    loads = freeze_loads(reuse["trials"])
    if (scout.get("status") != "PASS" or scout.get("data_role") != "calibration" or
            scout.get("reference_policy") != "shortest_queue" or
            scout.get("serving_profile") != PROFILE or
            scout.get("timing_calibration_sha256") != sha256(stage/"timing_models/methodology_calibration.json") or
            scout.get("final_policy_ids") != list(POLICIES) or
            scout.get("final_requests_per_cell") != 16000 or scout.get("final_matrix_cells") != 32 or
            any(scout.get(k) != v for k, v in loads.items()) or
            freeze_loads(scout["trials"]) != loads):
        raise ValueError("Completed measured scout does not satisfy the original Figure 5 gates")
    normalized = lambda rows: [{k: v for k, v in row.items() if k not in ("run_file", "events_file")}
                               for row in rows]
    if normalized(scout["trials"]) != normalized(reuse["trials"]):
        raise ValueError("Saved scout trials differ from the rederived measurements")
    endpoints = scout.get("endpoint_confirmations", [])
    if len(endpoints) != 2:
        raise ValueError("Both endpoint confirmations are required")
    files = {str(path): sha256(path)}
    for saved, rate, expected in zip(endpoints,
                                    (loads["qps_values"][0], loads["qps_values"][-1]),
                                    ("stable", "unstable"), strict=True):
        run_path = (stage / saved["run_file"]).resolve()
        if not run_path.is_relative_to(stage) or not run_path.stem.startswith("confirm_"):
            raise ValueError("Endpoint must refer to this measured stage")
        events_path = run_path.with_name(run_path.stem+"_events.jsonl")
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        actual = classify_trial(events, requested_qps=rate)
        if (saved.get("requested_qps") != rate or actual["classification"] != expected or
                any(saved.get(k) != v for k, v in actual.items())):
            raise ValueError("Endpoint classification differs from its raw measured events")
        audit_run(read_json(run_path)["runs"][0], expected=actual["num_arrivals"])
        files.update({str(p): sha256(p) for p in (events_path, run_path)})
    return scout, files


def validate_inputs(options, *, require_capacity=True):
    stage = Path(options.stage_dir).resolve()
    prepared = Path(read_json(stage/"stage_started.json")["prepared_dir"])
    meta = read_json(prepared/"metadata.json")
    spec = importlib.util.find_spec("vllm")
    if (spec is None or spec.origin is None or Path(spec.origin).resolve().parent.parent !=
            Path(os.environ["SFS_ROOT"]).resolve()/"vllm"):
        raise ValueError("Editable vLLM import differs from the validated serving worktree")
    reuse = validate_resume_stage(SimpleNamespace(
        resume_from=stage, prepared_dir=prepared, service_manifest=meta["service_manifest"],
        source_review=getattr(options, "source_review", None)))
    revision = validate_prefill_revision(stage/"timing_models/methodology_calibration.json",
        options.candidate, options.candidate_audit, options.timing_review)
    files = {**reuse["file_sha256"], **reuse["serving_source_sha256"], **revision["file_sha256"]}
    inherited_paths = [stage/"reuse_manifest.json"]
    seen = set()
    while inherited_paths:
        inherited_path = inherited_paths.pop()
        if inherited_path in seen or not inherited_path.is_file():
            continue
        seen.add(inherited_path)
        for name, expected in read_json(inherited_path)["file_sha256"].items():
            if sha256(name) != expected:
                raise ValueError(f"Inherited measured artifact changed: {name}")
            files[name] = expected
            if Path(name).name == "reuse_manifest.json":
                inherited_paths.append(Path(name))
    scout = None
    if require_capacity:
        scout, scout_files = validate_completed_scout(stage, reuse)
        if scout.get("request_set_sha256") != meta["requests_sha256"]:
            raise ValueError("Scout request identity differs from the prepared calibration")
        files.update(scout_files)
    if getattr(options, "source_review", None):
        from sfs_core.routing.routebalance_predictor import RouteBalancePredictor
        path = getattr(options, "routebalance_predictor", None)
        if not path:
            raise ValueError("Full policy refresh requires the native RouteBalance predictor")
        predictor = RouteBalancePredictor.load(path)
        if set(predictor.model_labels) != set(MODELS):
            raise ValueError("Refreshed RouteBalance predictor has the wrong model family")
        predictor.predict_batch(["Summarize why the sky appears blue."])
        metadata_path = Path(path)/"metadata.json"
        files[str(metadata_path.resolve())] = sha256(metadata_path)
        for name, digest in read_json(metadata_path)["file_sha256"].items():
            files[str((Path(path)/name).resolve())] = digest
    return {"status": "PASS_CPU" if require_capacity else "PASS_CPU_PENDING_CAPACITY",
            "prepared_dir": str(prepared), "service_manifest": meta["service_manifest"],
            "scout": scout, "revision": revision, "file_sha256": files}


def configure_refreshed_smoke(args, options, inputs):
    if getattr(options, "source_review", None):
        args.routebalance_predictor_path = str(Path(options.routebalance_predictor).resolve())
        # Keep the measured calibration's routing settings, including batching.
        for name, value in inputs["scout"]["configuration"].items():
            if not hasattr(args, name):
                raise ValueError(f"Unknown measured routing setting: {name}")
            setattr(args, name, value)
        return list(POLICIES)
    return ["mooncake_prefill"]


def validate_evaluation_inputs(options, inputs):
    """Exercise real 8000-request ingestion and identity joins before allocation."""
    from scripts.runs import experiments as exp
    from scripts.runs.experiments_sweep import _parse_experiment_args
    root = Path(os.environ["SFS_ROOT"]).resolve()
    cache = root/"experiments/data/prompts/ministral3/holdout_cache_2000"
    mapping = root/"experiments/ministral3_paper/request_maps/req_map_delta_seed69_holdout2000_n8000.csv"
    meta = read_json(Path(inputs["prepared_dir"])/"metadata.json")
    service = read_json(inputs["service_manifest"])
    metrics = Path(service["models"][MODELS[0]]["traces"][0]).parent/"model_metrics.json"
    args = _parse_experiment_args([*meta["forwarded_argv"], "--num-requests", "8000",
        "--seed", "69", "--holdout-start-index", "2500", "--holdout-prompts-per-bucket", "2000",
        "--bucket-dir", str(cache), "--holdout-cache-dir", str(cache),
        "--require-existing-holdout-cache", "--service-metrics-json", str(metrics),
        "--decouple-arrivals", "--methodology-calibration-json", str(options.candidate)])
    configure_refreshed_smoke(args, options, inputs)
    requests, _, _ = exp._build_request_set(args)
    with mapping.open() as stream:
        mapped = list(csv.DictReader(stream))
    cached = {}
    for bucket in BUCKETS:
        for line in (cache/f"{bucket}.jsonl").read_text().splitlines():
            row = json.loads(line)
            cached[(bucket, row["prompt_metadata"]["example_id"])] = row["prompt"]
    by_id = {row["req_id"]: row for row in mapped}
    if (len(requests) != 8000 or len(mapped) != 8000 or len(by_id) != 8000 or
            Counter(r.bucket for r in requests) != Counter({b: 2000 for b in BUCKETS})):
        raise ValueError("Evaluation request budget or mapping differs")
    for request in requests:
        row = by_id.get(request.request_id)
        if (row is None or row["bucket"] != request.bucket or
                not 2500 <= int(row["holdout_prompt_index"]) < 4500 or
                int(row["prompt_tokens"]) != request.prompt_tokens or
                cached.get((row["bucket"], row["example_id"])) != request.prompt):
            raise ValueError("Evaluation prompt identity differs from the canonical request map")
    inputs["file_sha256"].update({str(p): sha256(p) for p in
        (mapping, metrics, cache/"manifest.json", *[cache/f"{b}.jsonl" for b in BUCKETS])})
    return {"status": "PASS_CPU", "requests": 8000, "per_bucket": 2000,
            "request_map_sha256": sha256(mapping), "gpu_executed": False}


def validate_smoke(payload, candidate_path, expected_requests, *, instance_ids=None):
    runs = payload.get("runs", [])
    if len(runs) != 1 or runs[0].get("utility") != "mooncake_prefill":
        raise ValueError("A fresh Mooncake smoke is required")
    run = runs[0]
    audit_run(run, expected=192)
    expected = {(r.request_id, r.bucket, r.prompt_tokens) for r in expected_requests}
    actual = {(r["request_id"], r["bucket"], r["prompt_tokens"]) for r in run["per_request"]}
    config = run.get("methodology_config", {})
    if (len(expected) != 192 or actual != expected or
            config.get("calibration", {}).get("artifact_sha256") != sha256(candidate_path) or
            run["summary"].get("succeeded_requests") != 192 or
            run["summary"].get("failed_requests") != 0 or
            any(not row.get("methodology_terms") or
                row["methodology_terms"].get("policy") != "mooncake_prefill"
                for row in run["per_request"])):
        raise ValueError("Mooncake smoke has mismatched inputs, timing, outcomes or decision logs")
    instance_ids = set(instance_ids) if instance_ids is not None else {f"vllm-{model}" for model in MODELS}
    for row in run["per_request"]:
        terms = row["methodology_terms"].get("candidates", {})
        if set(terms) != instance_ids or row.get("instance_id") not in terms:
            raise ValueError("Mooncake candidate coverage is incomplete")
        for term in terms.values():
            values = [term.get(key) for key in ("queued_prefill_ms", "incoming_prefill_ms", "total_prefill_ms")]
            if (any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values) or
                    not math.isclose(values[0]+values[1], values[2], rel_tol=1e-9, abs_tol=1e-9)):
                raise ValueError("Mooncake prefill accounting is invalid")
        if terms[row["instance_id"]]["total_prefill_ms"] != min(t["total_prefill_ms"] for t in terms.values()):
            raise ValueError("Mooncake selected a non-minimal candidate")
    arrivals = [row["system_entry_offset_s"] for row in run["per_request"]]
    duration = max(arrivals)-min(arrivals)
    realized = 191/duration if duration > 0 else float("inf")
    if not 1.8 <= realized <= 2.2:
        raise ValueError("Mooncake smoke did not attain the declared two-QPS arrivals")
    return {"status": "PASS", "requests": 192, "realized_arrival_qps": realized,
            "timing_calibration_sha256": sha256(candidate_path)}


async def run_smoke(options, inputs):
    from scripts.runs import experiments as exp
    from scripts.runs.experiments_sweep import _parse_experiment_args
    from sfs_core.shared.shared_experiment_helpers import warm_up_instances
    prepared = Path(inputs["prepared_dir"])
    meta = read_json(prepared/"metadata.json")
    requests = [exp.ExperimentRequest(**json.loads(line))
                for line in (prepared/"requests.jsonl").read_text().splitlines()]
    if len(requests) != 10000 or Counter(r.bucket for r in requests) != Counter({b: 2500 for b in BUCKETS}):
        raise ValueError("Smoke must use the balanced calibration request pool")
    rows = smoke_requests(requests)
    args = _parse_experiment_args([*meta["forwarded_argv"],
                                  "--service-metrics-json", options.service_metrics_json])
    args.instances_config = Path(options.instances_config)
    args.service_metrics_json = Path(options.service_metrics_json)
    args.methodology_calibration_json = str(Path(options.candidate).resolve())
    args.per_request_wait_log = options.wait_log
    args.decouple_arrivals = True
    args.utilities, args.request_rate_qps, args.num_requests = ["mooncake_prefill"], 2.0, 192
    instances, costs, metadata = exp.load_instances(args.instances_config)
    folder = Path(options.output_root)/"prefill_smoke"
    folder.mkdir(exist_ok=False)
    try:
        validate_pool(instances, metadata, options.service_metrics_json, inputs["service_manifest"])
        await warm_up_instances(list(instances.values()))
        await wait_drained(instances)
        monitor = TrialMonitor(folder/"events.jsonl", duration_s=1200, max_outstanding=512)
        policies = configure_refreshed_smoke(args, options, inputs)
        for policy in policies:
            args.utilities = [policy]
            await wait_drained(instances)
            selected_monitor = monitor if policy == "mooncake_prefill" else None
            payload = await asyncio.wait_for(exp.run_router_experiment(
                args=args, requests=rows, instances=instances, instance_costs=costs,
                instance_metadata=metadata, response_map_base_path=folder/f"{policy}_responses.log",
                request_log_base_path=folder/f"{policy}_waits.log", trial_monitor=selected_monitor), timeout=1200)
            write_json(folder/f"smoke_{policy}.json", payload)
            audit_run(payload["runs"][0], expected=192)
            await wait_drained(instances)
        payload = read_json(folder/"smoke_mooncake_prefill.json")
        audit = validate_smoke(payload, options.candidate, rows)
        audit.update(raw_smoke_sha256=sha256(folder/"smoke_mooncake_prefill.json"),
                     events_sha256=sha256(folder/"events.jsonl"), data_role="calibration",
                     refreshed_policies=policies,
                     policy_sha256={p: sha256(folder/f"smoke_{p}.json") for p in policies})
        write_json(folder/"audit.json", audit)
    finally:
        for client in instances.values():
            client.close()
    return folder


def compose_and_freeze(options, inputs, smoke_folder):
    """Produce explicit derived provenance; never edit a measured source record."""
    stage, root = Path(options.stage_dir).resolve(), Path(options.output_root).resolve()
    for name, expected in inputs["file_sha256"].items():
        if sha256(name) != expected:
            raise ValueError(f"Validated input changed before composition: {name}")
    smoke_folder = Path(smoke_folder)
    fresh_path, audit_path = smoke_folder/"smoke_mooncake_prefill.json", smoke_folder/"audit.json"
    fresh, fresh_audit = read_json(fresh_path), read_json(audit_path)
    if (fresh_audit.get("status") != "PASS" or fresh_audit.get("data_role") != "calibration" or
            fresh_audit.get("timing_calibration_sha256") != sha256(options.candidate) or
            fresh_audit.get("raw_smoke_sha256") != sha256(fresh_path) or
            fresh_audit.get("events_sha256") != sha256(smoke_folder/"events.jsonl")):
        raise ValueError("Fresh GPU smoke is absent or changed")
    derived = root/"derived_stage"
    refreshed = list(POLICIES) if getattr(options, "source_review", None) else ["mooncake_prefill"]
    if getattr(options, "source_review", None):
        if fresh_audit.get("refreshed_policies") != list(POLICIES):
            raise ValueError("Reviewed source changes require fresh all-policy GPU smoke")
        for policy in POLICIES:
            path = smoke_folder/f"smoke_{policy}.json"
            if fresh_audit.get("policy_sha256", {}).get(policy) != sha256(path):
                raise ValueError("Refreshed policy smoke changed")
            run = read_json(path)["runs"][0]
            audit_run(run, expected=192)
            if run.get("utility") != policy:
                raise ValueError("Refreshed smoke policy mismatch")
    derived.mkdir(exist_ok=False)
    shutil.copytree(Path(options.candidate).resolve().parent, derived/"timing_models")
    shutil.copy2(stage/"stage_started.json", derived/"stage_started.json")
    combined = deepcopy(read_json(stage/"smoke_audit.json"))
    evidence = {}
    for policy in POLICIES:
        source = smoke_folder/f"smoke_{policy}.json" if policy in refreshed else stage/f"smoke_{policy}.json"
        shutil.copy2(source, derived/f"smoke_{policy}.json")
        evidence[policy] = {"source": str(source), "sha256": sha256(source),
                            "reused": policy not in refreshed}
        combined["summaries"][policy] = read_json(source)["runs"][0]["summary"] if policy in refreshed else combined["summaries"][policy]
    combined["summaries"]["mooncake_prefill"] = fresh["runs"][0]["summary"]
    combined["policy_evidence"] = evidence
    combined["prefill_revision"] = inputs["revision"]
    write_json(derived/"smoke_audit.json", combined)
    scout = deepcopy(inputs["scout"])
    scout["timing_calibration_sha256"] = sha256(options.candidate)
    scout["derived_after_prefill_revision"] = {
        "source": str(stage/"capacity_scout.json"), "sha256": sha256(stage/"capacity_scout.json"),
        "original_timing_sha256": inputs["revision"]["original_sha256"],
        "reason": "Only Mooncake prefill timing changed; shortest-queue routing does not consume this fit. All measured loads, trials, endpoints and serving configuration are unchanged.",
    }
    write_json(derived/"capacity_scout.json", scout)
    reviewed = read_json(options.timing_review)
    review = {"status": "PASS", "timing_calibration_sha256": sha256(options.candidate),
        "smoke_audit_sha256": sha256(derived/"smoke_audit.json"),
        "prefill_fit_review": "CPU review before GPU qualification: "+reviewed["prefill_fit_review"]+" Current qualification: the warmed-prefill correction now has a matched, successful 192-request GPU smoke with measured arrivals and full decision logs. Original timing and smoke remain preserved; the corrected artifact is adopted for evaluation.",
        "tpot_fit_review": reviewed["tpot_fit_review"]+" Final TPOT heads and service rates are byte-identical across this correction.",
        "arrival_review": reviewed["arrival_review"]+" The completed resumed scout and both endpoint classifications were rederived from raw events without changing thresholds; fresh Mooncake smoke attained two-QPS arrivals. All batch-collection and cold-start delay remains in measured latency.",
        "source_review": str(Path(options.timing_review).resolve()), "policy_evidence": evidence}
    write_json(derived/"timing_review.json", review)
    base_path = derived/"figure5_manifest.base.json"
    manifest = figure5.freeze(derived, derived/"timing_review.json", base_path, os.environ["SFS_ROOT"])
    manifest["prefill_revision"] = scout["derived_after_prefill_revision"]
    manifest["file_sha256"].update(inputs["file_sha256"])
    for policy in refreshed:
        path = smoke_folder/f"smoke_{policy}.json"
        manifest["file_sha256"][str(path)] = sha256(path)
    for path in (fresh_path, audit_path, smoke_folder/"events.jsonl", Path(__file__).resolve()):
        manifest["file_sha256"][str(path)] = sha256(path)
    write_json(options.manifest, manifest)
    return figure5.load_manifest(options.manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare-check", "validate", "run"))
    for name in ("stage-dir", "candidate", "candidate-audit", "timing-review"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--source-review", type=Path)
    parser.add_argument("--routebalance-predictor", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--output-root")
    parser.add_argument("--instances-config")
    parser.add_argument("--service-metrics-json")
    parser.add_argument("--wait-log", action="append", default=[])
    options = parser.parse_args()
    if options.mode == "run" and (not all((options.manifest, options.output_root,
            options.instances_config, options.service_metrics_json)) or len(options.wait_log) != 3):
        parser.error("Run requires manifest, output root, instances, service metrics and three wait logs")
    if options.manifest and Path(options.manifest).exists():
        raise ValueError("Refusing to overwrite an existing Figure 5 manifest")
    inputs = validate_inputs(options, require_capacity=options.mode != "prepare-check")
    if options.mode != "run":
        ingestion = validate_evaluation_inputs(options, inputs)
        print(json.dumps({"status": inputs["status"], "gpu_executed": False,
                          "revision": inputs["revision"], "evaluation_ingestion": ingestion,
                          "file_sha256": inputs["file_sha256"]}))
        return
    folder = asyncio.run(run_smoke(options, inputs))
    manifest = compose_and_freeze(options, inputs, folder)
    if not options.smoke_only:
        figure5.run_sweep(SimpleNamespace(output_root=options.output_root, manifest=options.manifest,
            group="baseline", instances_config=options.instances_config, wait_log=options.wait_log), manifest)


if __name__ == "__main__":
    main()

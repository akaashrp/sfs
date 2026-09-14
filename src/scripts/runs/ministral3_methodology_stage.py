"""Reuse one Ministral pool for missing prefill probes, smoke, and capacity scout.

Prepare the calibration request set on CPU first. GPU mode never selects final
loads from standalone service-rate sums, and never runs evaluation holdouts.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import shutil
import sys
import tarfile
import time

from scripts.prep.prepare_methodology_service import MODELS, PROFILE, sha256
from scripts.prep.paper_ablation_data import EVALUATION_REQUESTS
from scripts.runs.capacity_scout import TrialMonitor, classify_trial, next_bracket_rate, freeze_loads, plan_scout_trial

POLICIES = ("hard", "score", "lmdeploy_proxy", "mooncake_prefill", "routebalance",
            "shortest_queue", "latency_agnostic", "round_robin")
BUCKETS = ("alpaca", "govreport-summarization", "hotpot_qa", "writingprompts")
FINAL_REQUESTS_PER_CELL = EVALUATION_REQUESTS


def validate_pool(instances, metadata, service_metrics, manifest_path):
    """The identical pool contract is checked on CPU and before stage traffic."""
    if metadata.get("serving_profile") != PROFILE:
        raise ValueError("Pool serving profile differs from the audited calibration profile")
    manifest = json.loads(Path(manifest_path).read_text())
    if sha256(service_metrics) != manifest["service_metrics_sha256"]:
        raise ValueError("Pool service metrics differ from the audited fit inputs")
    if len(instances) != 3 or {client.model_id for client in instances.values()} != set(MODELS):
        raise ValueError("Pool must cover all three Ministral candidates")
    return manifest


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def length_stratified_requests(requests, count=64, seed=69):
    if len(requests) < count or count < 3:
        raise ValueError("Insufficient calibration prompts for isolated probes")
    ordered = sorted(requests, key=lambda row: row.prompt_tokens)
    chosen = [ordered[round(i*(len(ordered)-1)/(count-1))] for i in range(count)]
    if len({row.prompt_tokens for row in chosen}) < 3:
        raise ValueError("Probe pool lacks prompt-length variation")
    random.Random(seed).shuffle(chosen)
    return chosen


def smoke_requests(requests, per_bucket=48):
    chosen = []
    for bucket in BUCKETS:
        rows = [row for row in requests if row.bucket == bucket][:per_bucket]
        if len(rows) != per_bucket:
            raise ValueError(f"Insufficient smoke prompts for {bucket}")
        chosen.extend(rows)
    random.Random(69).shuffle(chosen)
    return chosen


async def wait_drained(instances, timeout_s=30):
    deadline = time.monotonic()+timeout_s
    while True:
        states = await asyncio.gather(*(client.refresh_baseline_state() for client in instances.values()))
        if all(not row.requests and not row.inflight_total_tokens for row in states):
            return
        if time.monotonic() > deadline:
            raise RuntimeError("Engine state did not drain between trials")
        await asyncio.sleep(.05)


def audit_run(run, expected=None):
    if run.get("utility") == "vllm_sr_latency":
        from sfs_core.routing.latency_history import audit_history
        metadata = run.get("methodology_config", {})
        replay = audit_history(metadata)
        if metadata.get("warmup_completions") != 96 or replay["selections"] != len(run["per_request"]):
            raise ValueError("Incomplete latency warm-up/selection trace")
        selected_sequences = set()
        observations = {}
        for event in metadata["events"]:
            if event["type"] == "observation" and event["accepted"]:
                observations.setdefault(event["request_id"], []).append(event)
        for row in run["per_request"]:
            terms = row.get("methodology_terms", {})
            sequence = terms.get("sequence")
            if type(sequence) is not int or sequence in selected_sequences or not 0 <= sequence < len(metadata["events"]):
                raise ValueError("Invalid per-request latency selection sequence")
            selected_sequences.add(sequence)
            event = metadata["events"][sequence]
            model = metadata.get("instance_models", {}).get(row.get("instance_id"))
            if (event["type"] != "selection" or event["selected_model"] != model or
                    terms.get("selected_model") != model or row.get("response_model") != model or
                    terms.get("scores") != event["scores"]):
                raise ValueError("Latency selection/response attribution mismatch")
            feedback = observations.get(row.get("scheduler_request_id"), [])
            expected_metrics = {"ttft", "tpot"} if row.get("usage_completion_tokens",0)>0 else {"ttft"}
            if {e["metric"] for e in feedback} != expected_metrics or len(feedback)!=len(expected_metrics) or any(e["sequence"]<=sequence or e["model"]!=model for e in feedback):
                raise ValueError("Missing or noncausal per-request latency feedback")
    rows = run["per_request"]
    if expected is not None and len(rows) != expected:
        raise ValueError("Incomplete smoke request coverage")
    if not rows or any(row.get("error") or not row.get("response_id") for row in rows):
        raise ValueError("Trial has request errors or missing responses")
    if len({row["response_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate responses in trial")
    for row in rows:
        for field in ("queue_delay_ms", "ttft_ms", "system_entry_to_dispatch_ms", "actual_cost"):
            value = row.get(field)
            if value is None or not math.isfinite(value) or value < 0:
                raise ValueError(f"Missing/invalid measured {field}")
        if row.get("usage_completion_tokens") is None:
            raise ValueError("Trial is missing measured token usage")
    if any(run.get("methodology_config", {}).get("unfinished_after_drain", {}).values()):
        raise ValueError("Methodology trial leaked reservations")


def prepare(options, forwarded):
    from scripts.runs.experiments_sweep import _parse_experiment_args
    from scripts.runs import experiments as exp
    args = _parse_experiment_args(forwarded)
    if args.holdout_start_index != 0 or args.holdout_prompts_per_bucket != 2500 or args.num_requests != 10000:
        raise ValueError("Preparation requires exactly the first 2500 prompts per bucket")
    root = Path(options.prepared_dir).resolve()
    if root.exists():
        raise ValueError("Prepared output already exists; choose a new directory")
    requests, request_manifest, source = exp._build_request_set(args)
    if Counter(row.bucket for row in requests) != Counter({bucket: 2500 for bucket in BUCKETS}):
        raise ValueError("Calibration request pool is not balanced")
    root.mkdir(parents=True)
    with (root / "requests.jsonl").open("x") as stream:
        for row in requests:
            stream.write(json.dumps(asdict(row), allow_nan=False) + "\n")
    write_json(root / "metadata.json", {"status": "READY", "data_role": "calibration",
        "num_requests": len(requests), "holdout_start_index": 0,
        "prompts_per_bucket": 2500, "forwarded_argv": forwarded,
        "request_manifest": request_manifest, "prompt_source": source,
        "requests_sha256": sha256(root / "requests.jsonl"),
        "service_manifest": str(Path(options.service_manifest).resolve()),
        "service_manifest_sha256": sha256(options.service_manifest)})
    print(json.dumps({"status": "READY", "prepared_dir": str(root), "num_requests": len(requests)}))


def validate_prepared(options):
    import importlib.util
    import os
    from sfs_core.routing.routebalance_predictor import validate_encoder_metadata
    from scripts.runs.experiments_sweep import _parse_experiment_args
    from scripts.runs import experiments as exp
    root = Path(options.prepared_dir).resolve()
    meta = json.loads((root/"metadata.json").read_text())
    if (meta.get("status"), meta.get("data_role"), meta.get("num_requests"),
            meta.get("holdout_start_index")) != ("READY", "calibration", 10000, 0):
        raise ValueError("Prepared inputs must be the complete calibration pool")
    if sha256(root/"requests.jsonl") != meta["requests_sha256"]:
        raise ValueError("Prepared request checksum mismatch")
    if sha256(options.service_manifest) != meta["service_manifest_sha256"]:
        raise ValueError("Service manifest checksum mismatch")
    requests = [exp.ExperimentRequest(**json.loads(line)) for line in
                (root/"requests.jsonl").read_text().splitlines()]
    if (len(requests) != 10000 or len({r.request_id for r in requests}) != 10000 or
            Counter(r.bucket for r in requests) != Counter({b: 2500 for b in BUCKETS})):
        raise ValueError("Prepared calibration payload has incomplete or duplicate requests")
    pool_check = validate_rendered_pool(Path(os.environ["SFS_ROOT"]), options.service_manifest)
    args = _parse_experiment_args([*meta["forwarded_argv"], "--service-metrics-json",
                                  pool_check["service_metrics_json"]])
    if set(args.calibrated_service_metrics) != set(MODELS):
        raise ValueError("Parsed SCORE service metrics do not match the Ministral candidates")
    artifact = Path(options.routebalance_predictor).resolve()
    predictor = json.loads((artifact/"metadata.json").read_text())
    validate_encoder_metadata(predictor["encoder"])
    for name, expected in predictor["file_sha256"].items():
        if sha256(artifact/name) != expected:
            raise ValueError(f"Native predictor checksum mismatch: {name}")
    spec = importlib.util.find_spec("vllm")
    if (spec is None or spec.origin is None or
            Path(spec.origin).resolve().parent.parent != Path(os.environ["SFS_ROOT"]).resolve()/"vllm"):
        raise ValueError("Active editable vLLM import must point to this worktree")
    from scripts.runs.serving_contract_preflight import validate as validate_serving
    serving_contract = validate_serving(Path(os.environ["SFS_ROOT"]).resolve(), "ministral3",
        Path(os.environ["PREDICTOR_RUN_DIR"])/"output_length_predictor")
    reuse = validate_resume_stage(options) if getattr(options, "resume_from", None) else None
    return {"status": "PASS", "prepared_dir": str(root), "native_predictor": str(artifact),
            "vllm_import": spec.origin, "requests": 10000, "gpu_executed": False,
            "pool_contract": pool_check, "serving_contract": serving_contract,
            "reuse": reuse}


def validate_measured_sources(source, root, review_path=None):
    # Controllers may change to resume work; serving, routing, and predictor
    # implementations must still match the code archived by the measured job.
    critical = list((root/"src/sfs_core/routing").glob("*.py"))
    critical += [root/p for p in (
        "src/scripts/runs/experiments.py", "src/scripts/runs/experiments_sweep.py",
        "src/sfs_core/shared/shared_experiment_helpers.py",
        "src/slurm/runs/ministral3_router_common.sh",
        "vllm/vllm/v1/engine/async_llm.py",
        "vllm/vllm/v1/core/sched/scheduler.py", "vllm/vllm/v1/core/sched/state_snapshot.py")]
    # A reviewed additive controller change may reuse capacity measurements,
    # but never its old policy smoke: the bootstrap must rerun all policies.
    review = json.loads(Path(review_path).read_text()) if review_path else None
    if review and (review.get("status") != "PASS_CPU_REQUIRES_ALL_POLICY_GPU_SMOKE" or
                   review.get("archive_sha256") != sha256(source/"source_snapshot.tar.gz") or
                   not review.get("reason")):
        raise ValueError("Invalid measured-source compatibility review")
    reviewed_changes = {}
    with tarfile.open(source/"source_snapshot.tar.gz") as archive:
        for path in critical:
            name = str(path.relative_to(root))
            try:
                member = archive.extractfile(name)
            except KeyError:
                member = None
            original = member.read() if member else None
            if original != path.read_bytes():
                import hashlib
                change = {"original_sha256": hashlib.sha256(original).hexdigest() if original is not None else None,
                          "current_sha256": sha256(path)}
                if not review or review.get("changes", {}).get(name) != change:
                    raise ValueError(f"Measured serving source differs: {path}")
                reviewed_changes[name] = change
    if review and reviewed_changes != review.get("changes"):
        raise ValueError("Source review does not exactly match the measured/current differences")
    return critical


def validate_resume_stage(options):
    """Re-audit preserved measured traffic and serving code before reuse."""
    import os
    from sfs_core.routing.methodology_calibration import MethodologyCalibration
    source = Path(options.resume_from).resolve()
    root = Path(os.environ["SFS_ROOT"]).resolve()
    started = json.loads((source/"stage_started.json").read_text())
    prepared = Path(options.prepared_dir).resolve()
    if (Path(started["prepared_dir"]).resolve() != prepared or
            started["service_manifest_sha256"] != sha256(options.service_manifest) or
            started.get("prepared_metadata_sha256") != sha256(prepared/"metadata.json") or
            started.get("data_role") != "calibration" or started.get("policies") != list(POLICIES)):
        raise ValueError("Resume inputs differ from the measured stage")
    if json.loads((source/"instances.json").read_text()).get("serving_profile") != PROFILE:
        raise ValueError("Measured serving profile differs")
    smoke = json.loads((source/"smoke_audit.json").read_text())
    if smoke.get("status") != "PASS" or smoke.get("policies") != list(POLICIES):
        raise ValueError("Resume requires successful eight-policy GPU smoke")
    review_path = getattr(options, "source_review", None)
    critical = validate_measured_sources(source, root, review_path)
    timing = source/"timing_models/methodology_calibration.json"
    calibration = MethodologyCalibration.load(timing)
    calibration.validate_runtime_profile(PROFILE)
    calibration.preload()
    files = [source/"stage_started.json", source/"smoke_audit.json", source/"instances.json", timing,
             prepared/"metadata.json", prepared/"requests.jsonl"]
    if review_path:
        files.append(Path(review_path).resolve())
    files += list((source/"timing_models").glob("*.json"))
    for policy in POLICIES:
        path = source/f"smoke_{policy}.json"
        payload = json.loads(path.read_text())
        run = payload["runs"][0]
        audit_run(run, expected=192)
        if run["utility"] != policy:
            raise ValueError("Measured smoke policy differs")
        files.append(path)
    # Current classification must reproduce the original result from raw events.
    # Inconclusive points remain inconclusive; no threshold is relaxed on reuse.
    trials = []
    for path in sorted(source.glob("scout_*_classification.json")):
        saved = json.loads(path.read_text())
        stem = path.name.removesuffix("_classification.json")
        events_path, run_path = source/f"{stem}_events.jsonl", source/f"{stem}.json"
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        actual = classify_trial(events, requested_qps=saved["requested_qps"])
        if any(saved.get(k) != v for k, v in actual.items()):
            raise ValueError(f"Scout classification differs from raw events: {path}")
        audit_run(json.loads(run_path.read_text())["runs"][0])
        trials.append({**actual, "run_file": str(run_path), "events_file": str(events_path)})
        files.extend((path, events_path, run_path))
    prior = source/"reuse_manifest.json"
    if prior.exists():
        inherited = json.loads(prior.read_text())
        for name, expected in inherited["file_sha256"].items():
            if sha256(name) != expected:
                raise ValueError(f"Previously reused measured input changed: {name}")
        trials = inherited["trials"] + trials
        files.append(prior)
    return {"status": "PASS", "source": str(source), "trials": trials,
            "next_trial": plan_scout_trial(trials),
            "file_sha256": {str(p): sha256(p) for p in files},
            "serving_source_sha256": {str(p): sha256(p) for p in critical}}


def checkpoint_scout(root, trials, status, **extra):
    value = {"status": status, "trials": trials,
             "next_trial": plan_scout_trial(trials), **extra}
    path = Path(root)/"capacity_scout_progress.json"
    pending = path.with_suffix(".json.new")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    pending.replace(path)
    return value


def validate_rendered_pool(sfs_root, manifest_path):
    """Render with the production shell helper, then use the real client loader.

    Client construction is network-free; no server, GPU, or shared-memory segment
    is opened. Temporary configuration files stay inside the workspace.
    """
    import os
    import subprocess
    import tempfile
    from scripts.runs import experiments as exp

    root = Path(sfs_root).resolve()
    manifest = json.loads(Path(manifest_path).read_text())
    metrics = Path(manifest["models"][MODELS[0]]["traces"][0]).parent/"model_metrics.json"
    scratch = root/"experiments/methodology_baselines/scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pool-preflight-", dir=scratch) as temporary:
        config = Path(temporary)/"instances.json"
        env = dict(os.environ, SFS_ROOT=str(root), MINISTRAL_INSTANCES_CONFIG=str(config),
                   MINISTRAL_SERVICE_METRICS=str(metrics), MINISTRAL_PORT_3B="9100",
                   MINISTRAL_PORT_8B="9101", MINISTRAL_PORT_14B="9102",
                   MINISTRAL_SHM_3B="preflight-3b", MINISTRAL_SHM_8B="preflight-8b",
                   MINISTRAL_SHM_14B="preflight-14b", MINISTRAL_SNAPSHOT_SHM_SIZE_BYTES="8388608")
        subprocess.run(["bash", "-c", 'set -euo pipefail; source "$SFS_ROOT/src/slurm/runs/ministral3_router_common.sh"; ministral3_write_instances_config'],
                       env=env, check=True, capture_output=True, text=True)
        instances, costs, metadata = exp.load_instances(config)
        try:
            validate_pool(instances, metadata, metrics, manifest_path)
            if set(costs) != set(instances) or any(
                    not math.isfinite(v) or v < 0 for c in costs.values() for v in c.values()):
                raise ValueError("Invalid candidate costs")
            return {"status": "PASS", "models": sorted(c.model_id for c in instances.values()),
                    "serving_profile": metadata["serving_profile"],
                    "service_metrics_json": str(metrics), "renderer_and_loader_exercised": True}
        finally:
            for client in instances.values():
                client.close()


async def run_stage(options):
    from scripts.runs import experiments as exp
    from scripts.runs.experiments_sweep import _parse_experiment_args
    from scripts.prep.fit_methodology_calibration import fit_manifest
    from sfs_core.routing.methodology_calibration import MethodologyCalibration
    from sfs_core.shared.shared_experiment_helpers import build_messages, warm_up_instances

    prepared = Path(options.prepared_dir).resolve()
    meta = json.loads((prepared / "metadata.json").read_text())
    if (meta.get("status"), meta.get("data_role"), meta.get("holdout_start_index"),
            meta.get("prompts_per_bucket")) != ("READY", "calibration", 0, 2500):
        raise ValueError("Invalid prepared calibration artifact")
    if sha256(prepared / "requests.jsonl") != meta["requests_sha256"]:
        raise ValueError("Prepared request checksum mismatch")
    if sha256(options.service_manifest) != meta["service_manifest_sha256"]:
        raise ValueError("Service manifest differs from the prepared inputs")
    requests = [exp.ExperimentRequest(**json.loads(line)) for line in
                (prepared / "requests.jsonl").read_text().splitlines()]
    if len(requests) != 10000 or Counter(row.bucket for row in requests) != Counter(
            {bucket: 2500 for bucket in BUCKETS}):
        raise ValueError("Prepared calibration payload has incomplete bucket coverage")
    root = Path(options.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)  # Pool logs already live here.
    if (root / "stage_started.json").exists():
        raise ValueError("This stage output already exists; preserve the prior run")
    # Parsing derives SCORE's calibrated metrics, so supply the runtime file
    # before parsing rather than only replacing its path on the namespace.
    args = _parse_experiment_args([*meta["forwarded_argv"],
                                  "--service-metrics-json", str(options.service_metrics_json)])
    args.instances_config = Path(options.instances_config)
    args.service_metrics_json = Path(options.service_metrics_json)
    args.routebalance_predictor_path = str(Path(options.routebalance_predictor).resolve())
    args.per_request_wait_log = options.wait_log
    args.decouple_arrivals = True
    args.routebalance_weights = (1/3, 1/3, 1/3)
    args.routebalance_batch_max_size = 16
    args.routebalance_batch_wait_ms = 25.0
    instances, costs, instance_meta = exp.load_instances(args.instances_config)
    try:
        manifest = validate_pool(instances, instance_meta, options.service_metrics_json,
                                 options.service_manifest)
    except BaseException:
        for client in instances.values():
            client.close()
        raise
    write_json(root / "stage_started.json", {"data_role": "calibration", "policies": POLICIES,
        "prepared_dir": str(prepared), "prepared_metadata_sha256": sha256(prepared/"metadata.json"),
        "probe_count_per_model": 64, "duration_s": options.duration_s,
        "max_trials": options.max_trials, "sensitivity_check": False,
        "service_manifest_sha256": sha256(options.service_manifest)})

    async def trial(label, policy, rows, rate, monitor=None):
        await wait_drained(instances)
        args.utilities = [policy]
        args.request_rate_qps = rate
        args.num_requests = len(rows)
        result = await asyncio.wait_for(exp.run_router_experiment(
            args=args, requests=rows, instances=instances, instance_costs=costs,
            instance_metadata=instance_meta,
            response_map_base_path=root/f"{label}_responses.log",
            request_log_base_path=root/f"{label}_waits.log", trial_monitor=monitor),
            timeout=options.duration_s+1200)
        write_json(root / f"{label}.json", result)
        await wait_drained(instances)
        return result["runs"][0]

    try:
        await warm_up_instances(list(instances.values()))
        await wait_drained(instances)
        reuse = validate_resume_stage(options) if getattr(options, "resume_from", None) else None
        if reuse:
            source = Path(reuse["source"])
            shutil.copytree(source/"timing_models", root/"timing_models")
            for name in ("smoke_audit.json", *[f"smoke_{p}.json" for p in POLICIES]):
                shutil.copy2(source/name, root/name)
            write_json(root/"reuse_manifest.json", reuse)
            calibration_path = root/"timing_models/methodology_calibration.json"
            calibration = MethodologyCalibration.load(calibration_path)
            calibration.validate_runtime_profile(instance_meta["serving_profile"])
            args.methodology_calibration_json = str(calibration_path)
        else:
            probes = length_stratified_requests(requests)
            probe_records = []

            async def probe_model(key, client):
                if not manifest["coverage_audit"][client.model_id]["requires_prefill_probes"]:
                    return
                for index, req in enumerate(probes):
                    started = time.perf_counter()
                    response = await client.submit_request(
                        messages=build_messages(req.prompt, args.system_prompt),
                        temperature=0, top_p=1, max_completion_tokens=1,
                        extra_body={"chat_template_kwargs": {},
                                    "request_id": f"isolated-{client.model_id}-{index}"})
                    probe_records.append({"model": client.model_id, "calibration_request_id": req.request_id,
                        "bucket": req.bucket, "prompt_tokens": req.prompt_tokens,
                        "response_id": response.id, "elapsed_s": time.perf_counter()-started,
                        "usage": response.usage.model_dump()})
                    await wait_drained({key: client})

            await asyncio.gather(*(probe_model(key, client) for key, client in instances.items()))
            write_json(root / "isolated_prefill_probes.json", {"data_role": "calibration",
                       "max_completion_tokens": 1, "requests": probe_records})
            # The pinned vLLM CSV logger flushes asynchronously every one second.
            # Let its tail reach disk while the drained pool has no new traffic.
            await asyncio.sleep(2.0)
            fit_input = deepcopy(manifest)
            # The pool traces contain only warmup/probes at this point. Freeze
            # derived copies now, before smoke/scout appends unrelated traffic.
            for model in MODELS:
                source = root / f"batch_stats_{model}.csv"
                target = root / f"prefill_probe_trace_{model}.csv"
                target.write_bytes(source.read_bytes())
                fit_input["models"][model]["traces"].append(str(target))
            fit_manifest_path = root / "fit_input_manifest.json"
            write_json(fit_manifest_path, fit_input)
            calibration_path = await asyncio.to_thread(fit_manifest, fit_manifest_path, root/"timing_models")
            calibration = MethodologyCalibration.load(calibration_path)
            calibration.validate_runtime_profile(instance_meta["serving_profile"])
            args.methodology_calibration_json = str(calibration_path)

            smoke_rows = smoke_requests(requests)
            summaries = {}
            for policy in POLICIES:
                monitor = TrialMonitor(root/f"smoke_{policy}_events.jsonl", duration_s=1200,
                                       max_outstanding=512)
                run = await trial(f"smoke_{policy}", policy, smoke_rows, 2.0, monitor)
                audit_run(run, expected=len(smoke_rows))
                if policy in {"lmdeploy_proxy", "mooncake_prefill", "routebalance"} and any(
                        not row.get("methodology_terms") for row in run["per_request"]):
                    raise ValueError("Methodology decision logs are incomplete")
                summaries[policy] = run["summary"]
            write_json(root / "smoke_audit.json", {"status": "PASS", "requests_per_policy": 192,
                       "policies": list(POLICIES), "qps": 2.0, "summaries": summaries})

        trials = list(reuse["trials"]) if reuse else []
        checkpoint_scout(root, trials, "COLLECTING")
        for index in range(options.max_trials):
            plan = plan_scout_trial(trials, duration_s=options.duration_s)
            if plan["status"] == "BRACKET_READY":
                break
            if plan["status"] == "NEEDS_REVIEW":
                print(json.dumps(checkpoint_scout(root, trials, "NEEDS_REVIEW")))
                return
            rate = plan["qps"]
            monitor = TrialMonitor(root/f"scout_{index:02d}_events.jsonl",
                                   duration_s=plan["duration_s"], max_outstanding=1024)
            run = await trial(f"scout_{index:02d}", "shortest_queue", requests, rate, monitor)
            audit_run(run)
            result = classify_trial(monitor.events, requested_qps=rate)
            result["run_file"] = f"scout_{index:02d}.json"
            trials.append(result)
            write_json(root/f"scout_{index:02d}_classification.json", result)
            checkpoint_scout(root, trials, "COLLECTING")
        plan = plan_scout_trial(trials, duration_s=options.duration_s)
        if plan["status"] != "BRACKET_READY":
            status = "NEEDS_REVIEW" if plan["status"] == "NEEDS_REVIEW" else "NEEDS_MORE_TRIALS"
            print(json.dumps(checkpoint_scout(root, trials, status)))
            return
        loads = freeze_loads(trials)
        # Verify the proposed light/overload endpoints on fresh drained trials.
        confirmations = []
        for index, (rate, expected) in enumerate(((loads["qps_values"][0], "stable"),
                                                (loads["qps_values"][-1], "unstable"))):
            for attempt in range(2):
                label = f"confirm_{index}" + (f"_retry{attempt}" if attempt else "")
                monitor = TrialMonitor(root/f"{label}_events.jsonl",
                    duration_s=options.duration_s if not attempt else max(options.duration_s, 600),
                    max_outstanding=1024)
                run = await trial(label, "shortest_queue", requests, rate, monitor)
                audit_run(run)
                result = classify_trial(monitor.events, requested_qps=rate)
                result["run_file"] = f"{label}.json"
                write_json(root/f"{label}_classification.json", result)
                if result["classification"] == expected:
                    break
                if result["classification"] != "inconclusive":
                    break
            confirmations.append(result)
            if result["classification"] != expected:
                print(json.dumps(checkpoint_scout(root, trials, "NEEDS_REVIEW",
                    reason="endpoint_confirmation_incomplete", endpoint_confirmations=confirmations)))
                return
            checkpoint_scout(root, trials, "CONFIRMING_ENDPOINTS", endpoint_confirmations=confirmations)
        write_json(root / "capacity_scout.json", {"status": "PASS", "data_role": "calibration",
            "reference_policy": "shortest_queue", **loads, "trials": trials,
            "endpoint_confirmations": confirmations,
            "serving_profile": instance_meta["serving_profile"],
            "timing_calibration_sha256": sha256(calibration_path),
            "request_set_sha256": meta["requests_sha256"],
            "final_policy_ids": list(POLICIES), "final_requests_per_cell": FINAL_REQUESTS_PER_CELL,
            "final_matrix_cells": 32, "evaluation_started": False,
            "configuration": {"lambda_weight": args.lambda_weight, "delta_weight": args.delta_weight,
                "score_cost_weight": args.score_cost_weight, "score_latency_weight": args.score_latency_weight,
                "score_total_cost_budget": args.score_total_cost_budget,
                "routebalance_weights": list(args.routebalance_weights),
                "routebalance_batch_max_size": 16, "routebalance_batch_wait_ms": 25.0}})
        checkpoint_scout(root, trials, "PASS", endpoint_confirmations=confirmations)
        print(json.dumps({"status": "PASS", "capacity_scout": str(root/"capacity_scout.json"), **loads}))
    finally:
        for client in instances.values():
            client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--service-manifest", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--instances-config")
    parser.add_argument("--service-metrics-json")
    parser.add_argument("--routebalance-predictor")
    parser.add_argument("--wait-log", action="append", default=[])
    parser.add_argument("--duration-s", type=int, default=360)
    parser.add_argument("--max-trials", type=int, default=16)
    parser.add_argument("--resume-from", type=Path)
    options, forwarded = parser.parse_known_args()
    if options.prepare_only:
        prepare(options, forwarded)
    elif options.validate_only:
        if options.routebalance_predictor is None:
            parser.error("Validation requires the native predictor artifact")
        print(json.dumps(validate_prepared(options)))
    else:
        if forwarded or any(getattr(options, name) is None for name in
                           ("output_root", "instances_config", "service_metrics_json", "routebalance_predictor")):
            parser.error("GPU stage requires output, instances, service metrics, predictor, and no extra arguments")
        if options.duration_s < 360 or options.max_trials < 3:
            parser.error("Scout requires >=360 second trials and >=3 bracket trials")
        asyncio.run(run_stage(options))


if __name__ == "__main__":
    main()

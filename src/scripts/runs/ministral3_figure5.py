"""Freeze and consume the measured eight-policy/four-load Figure 5 contract.

Freezing requires completed scout/smoke evidence and a recorded timing review.
It does not submit jobs. Sweep and collation use the same checksummed inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.prep.prepare_methodology_service import MODELS, PROFILE, sha256
from scripts.runs.capacity_scout import freeze_loads
from scripts.runs.ministral3_methodology_stage import BUCKETS, POLICIES, write_json
from scripts.runs.measured_audit import require_complete_ttft

EVALUATION_REQUESTS_PER_CELL = 8000
HOLDOUT_PROMPTS_PER_BUCKET = 2000

GROUPS = {"snapshot": ["hard", "score", "routebalance", "shortest_queue"],
          "baseline": ["lmdeploy_proxy", "mooncake_prefill", "latency_agnostic", "round_robin"],
          "all": list(POLICIES)}


def read_json(path):
    return json.loads(Path(path).read_text())


def freeze(stage_dir, review_path, output_path, sfs_root):
    stage, root = Path(stage_dir).resolve(), Path(sfs_root).resolve()
    scout_path, smoke_path = stage/"capacity_scout.json", stage/"smoke_audit.json"
    scout, smoke, review = read_json(scout_path), read_json(smoke_path), read_json(review_path)
    timing = stage/"timing_models/methodology_calibration.json"
    loads = freeze_loads(scout["trials"])
    if (scout.get("status") != "PASS" or scout.get("data_role") != "calibration" or
            scout.get("reference_policy") != "shortest_queue" or
            scout.get("serving_profile") != PROFILE or
            any(scout.get(key) != value for key, value in loads.items()) or
            scout.get("final_policy_ids") != list(POLICIES) or
            scout.get("final_requests_per_cell") != 16000 or
            scout.get("timing_calibration_sha256") != sha256(timing)):
        raise ValueError("Scout does not establish the declared evaluation contract")
    for row, rate, expected in zip(scout.get("endpoint_confirmations", []),
                                  (loads["qps_values"][0], loads["qps_values"][-1]),
                                  ("stable", "unstable"), strict=True):
        if row["classification"] != expected or row["requested_qps"] != rate:
            raise ValueError("Missing light/overload endpoint confirmation")
    if smoke.get("status") != "PASS" or smoke.get("policies") != list(POLICIES):
        raise ValueError("All eight policies must pass GPU smoke before freezing")
    if (review.get("status") != "PASS" or review.get("timing_calibration_sha256") != sha256(timing)
            or review.get("smoke_audit_sha256") != sha256(smoke_path) or
            any(not isinstance(review.get(key), str) or not review[key].strip()
                for key in ("prefill_fit_review", "tpot_fit_review", "arrival_review"))):
        raise ValueError("Record timing residual and warmed-arrival reviews tied to these artifacts")

    prepared = Path(read_json(stage/"stage_started.json")["prepared_dir"])
    meta = read_json(prepared/"metadata.json")
    service_manifest = Path(meta["service_manifest"])
    service = read_json(service_manifest)
    service_metrics = Path(service["models"][MODELS[0]]["traces"][0]).parent/"model_metrics.json"
    cache = root/"experiments/data/prompts/ministral3/holdout_cache_2000"
    cache_meta = read_json(cache/"manifest.json")
    req_map = root/"experiments/ministral3_paper/request_maps/req_map_delta_seed69_holdout2000_n8000.csv"
    judge_audit = root/"experiments/methodology_baselines/holdout_judge_audit_20260906.json"
    judge = read_json(judge_audit)
    if (judge.get("status"), judge.get("prompt_groups"), judge.get("model_scores")) != ("PASS", 16000, 48000):
        raise ValueError("Complete judged holdout audit required")
    if (cache_meta.get("holdout_start_index"), cache_meta.get("holdout_prompts_per_bucket"),
            cache_meta.get("bucket_counts")) != (2500, HOLDOUT_PROMPTS_PER_BUCKET,
                                                {f"{b}.jsonl": HOLDOUT_PROMPTS_PER_BUCKET for b in BUCKETS}):
        raise ValueError("Holdout cache must contain indices 2500..4499 in each bucket")
    with req_map.open() as stream:
        mappings = list(csv.DictReader(stream))
    if (len(mappings) != EVALUATION_REQUESTS_PER_CELL or len({r["req_id"] for r in mappings}) != EVALUATION_REQUESTS_PER_CELL or
            len({(r["bucket"], r["example_id"]) for r in mappings}) != EVALUATION_REQUESTS_PER_CELL or
            Counter(r["bucket"] for r in mappings) != Counter({b: HOLDOUT_PROMPTS_PER_BUCKET for b in BUCKETS}) or
            any(not 2500 <= int(r["holdout_prompt_index"]) < 4500 for r in mappings)):
        raise ValueError("Incomplete or overlapping evaluation request map")

    predictor = root/"experiments/ministral3_paper/routebalance_predictor/run_20260905_native"
    backend = root/"experiments/ministral3_paper/predictors/run_45089699"
    config = scout["configuration"]
    argv = [*meta["forwarded_argv"], "--num-requests", str(EVALUATION_REQUESTS_PER_CELL), "--seed", "69",
        "--holdout-start-index", "2500", "--holdout-prompts-per-bucket", str(HOLDOUT_PROMPTS_PER_BUCKET),
        "--bucket-dir", str(cache), "--holdout-cache-dir", str(cache),
        "--service-metrics-json", str(service_metrics), "--decouple-arrivals", "--require-existing-holdout-cache",
        "--methodology-calibration-json", str(timing),
        "--routebalance-predictor-path", str(predictor)]
    for key in ("lambda_weight", "delta_weight", "score_cost_weight", "score_latency_weight",
                "routebalance_batch_max_size", "routebalance_batch_wait_ms"):
        argv.extend(["--"+key.replace("_", "-"), str(config[key])])
    argv.extend(["--routebalance-weights", *map(str, config["routebalance_weights"])])
    if config.get("score_total_cost_budget") is not None:
        argv.extend(["--score-total-cost-budget", str(config["score_total_cost_budget"])])

    # Check actual ingestion order against the pre-existing judged-response map
    # on CPU, before any final serving allocation.
    from scripts.runs.experiments_sweep import _parse_experiment_args
    from scripts.runs import experiments as exp
    args = _parse_experiment_args(argv)
    requests, _, _ = exp._build_request_set(args)
    if len(requests) != EVALUATION_REQUESTS_PER_CELL:
        raise ValueError("Evaluation construction did not produce 8000 requests")
    mapped = {r["req_id"]: r for r in mappings}
    cached = {}
    for bucket in BUCKETS:
        with (cache/f"{bucket}.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                cached[(bucket, row["prompt_metadata"]["example_id"])] = row
    for req in requests:
        row = mapped.get(req.request_id)
        source = cached.get((row["bucket"], row["example_id"])) if row else None
        if (row is None or row["bucket"] != req.bucket or int(row["prompt_tokens"]) != req.prompt_tokens
                or source is None or source["prompt"] != req.prompt):
            raise ValueError("Evaluation ingestion differs from the saved request map")
    # The cache and map hashes preserve the full prompt identities, including
    # when distinct requests happen to have the same length.
    scored = Path(judge["scored_root"])
    files = [scout_path, smoke_path, Path(review_path).resolve(), timing, prepared/"metadata.json",
             service_manifest, service_metrics, cache/"manifest.json", req_map, judge_audit,
             Path(judge["judge_summary"]), backend/"SHA256SUMS", backend/"manual_audit.json",
             predictor/"metadata.json"]
    files += [cache/f"{bucket}.jsonl" for bucket in BUCKETS]
    files += [scored/model/f"{bucket}_scored.jsonl" for model in MODELS for bucket in BUCKETS]
    files += [predictor/name for name in read_json(predictor/"metadata.json")["file_sha256"]]
    files += [timing.parent/row["tpot"]["model_file"] for row in read_json(timing)["models"].values()]
    files += list((root/"src/sfs_core/routing").glob("*.py"))
    files += [root/"src/scripts/runs"/name for name in ("experiments.py", "experiments_sweep.py")]
    files += [root/"vllm/vllm/v1/core/sched"/name for name in ("scheduler.py", "state_snapshot.py")]
    manifest = {"schema_version": 2, "data_role": "evaluation", "policies": list(POLICIES),
        "groups": GROUPS, "requests_per_cell": EVALUATION_REQUESTS_PER_CELL, "matrix_cells": 32, "seed": 69,
        "holdout_start_index": 2500, "holdout_prompts_per_bucket": HOLDOUT_PROMPTS_PER_BUCKET,
        "serving_profile": PROFILE, "loads": loads, "configuration": config,
        "experiment_argv": argv, "request_map": str(req_map), "scored_root": str(scored),
        "service_run_dir": str(service_metrics.parent), "predictor_run_dir": str(backend),
        "judge_imputed_model_scores": judge["imputed_model_scores"],
        "capacity_evidence_requests_per_cell": 16000,
        "evaluation_budget_note": "Balanced 2000-per-bucket subset; original capacity and Figure 2 data preserved",
        "stage_dir": str(stage), "review_path": str(Path(review_path).resolve()),
        "file_sha256": {str(p.resolve()): sha256(p) for p in files}}
    write_json(output_path, manifest)
    return manifest


def load_manifest(path, *, verify_files=True):
    manifest = read_json(path)
    qps = manifest.get("loads", {}).get("qps_values", [])
    if (manifest.get("schema_version") != 2 or manifest.get("data_role") != "evaluation" or
            manifest.get("policies") != list(POLICIES) or manifest.get("groups") != GROUPS or
            manifest.get("requests_per_cell") != EVALUATION_REQUESTS_PER_CELL or manifest.get("matrix_cells") != 32 or
            manifest.get("holdout_start_index") != 2500 or
            manifest.get("holdout_prompts_per_bucket") != HOLDOUT_PROMPTS_PER_BUCKET or
            manifest.get("serving_profile") != PROFILE or
            len(qps) != 4 or any(not math.isfinite(q) or q <= 0 for q in qps) or
            sorted(set(qps)) != qps or not manifest.get("file_sha256")):
        raise ValueError("Invalid eight-policy/four-load Figure 5 manifest")
    if verify_files:
        for name, expected in manifest["file_sha256"].items():
            if sha256(name) != expected:
                raise ValueError(f"Frozen Figure 5 input changed: {name}")
    return manifest


def audit_points(paths, manifest, policies, *, augmented=False):
    expected = {(rate, policy) for rate in manifest["loads"]["qps_values"] for policy in policies}
    seen, records, realized_rates = set(), 0, []
    wanted_ids = {f"req-{i}" for i in range(manifest["requests_per_cell"])}
    for path in paths:
        payload = read_json(path)
        config = payload.get("config", {})
        rate = config.get("request_rate_qps")
        for key in ("lambda_weight", "delta_weight", "routebalance_weights",
                    "routebalance_batch_max_size", "routebalance_batch_wait_ms"):
            if config.get(key) != manifest["configuration"][key]:
                raise ValueError(f"Frozen configuration differs for {key}: {path}")
        if config.get("seed") != manifest["seed"] or config.get("arrival_process") != "poisson":
            raise ValueError(f"Incorrect seed/arrival process: {path}")
        if config.get("instance_metadata", {}).get("serving_profile") != manifest["serving_profile"]:
            raise ValueError(f"Incorrect serving profile: {path}")
        if payload.get("request_set", {}).get("num_requests") != len(wanted_ids):
            raise ValueError(f"Incomplete request set: {path}")
        for run in payload.get("router", {}).get("runs", []):
            cell = (rate, run["utility"])
            if cell not in expected or cell in seen:
                raise ValueError(f"Duplicate/unexpected Figure 5 cell: {cell}")
            seen.add(cell)
            rows, summary = run["per_request"], run["summary"]
            if (len(rows) != len(wanted_ids) or {r["request_id"] for r in rows} != wanted_ids or
                    summary.get("succeeded_requests") != len(rows) or summary.get("failed_requests") != 0 or
                    summary.get("system_entry_e2e_ttft_slo_missing_count") != 0 or
                    len({r.get("response_id") for r in rows}) != len(rows) or
                    any(r.get("error") or not r.get("response_id") for r in rows)):
                raise ValueError(f"Incomplete/failed measured cell: {cell}")
            require_complete_ttft(run)
            arrivals = [r.get("system_entry_offset_s") for r in rows]
            if any(not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0 for t in arrivals):
                raise ValueError(f"Missing measured arrival timestamps: {cell}")
            duration = max(arrivals)-min(arrivals)
            realized = (len(rows)-1)/duration if duration > 0 else float("inf")
            if abs(realized/rate-1) > .10:
                raise ValueError(f"Client did not attain the declared arrival rate: {cell}")
            realized_rates.append({"qps": rate, "policy": run["utility"], "realized_arrival_qps": realized})
            if augmented and any(not isinstance(r.get("actual_accuracy"), (int, float)) or
                                 not math.isfinite(r["actual_accuracy"]) or
                                 not 0 <= r["actual_accuracy"] <= 1 for r in rows):
                raise ValueError(f"Missing judged quality: {cell}")
            records += len(rows)
    if seen != expected:
        raise ValueError(f"Missing Figure 5 cells: {sorted(expected-seen)}")
    return {"status": "PASS", "matrix_cells": len(seen), "records": records,
            "policies": list(policies), "qps_values": manifest["loads"]["qps_values"],
            "arrival_attainment": realized_rates}


def point_paths(folder):
    # The real router also emits per-policy simulation-distribution JSONs in
    # this folder. They are diagnostics, not additional experiment points.
    return sorted(p for p in Path(folder).glob("*.json")
                  if re.search(r"(?:_point\d+|^point_\d+)\.json$", p.name))


def run_sweep(options, manifest):
    root = Path(options.output_root).resolve()
    outputs = root/"outputs"
    outputs.mkdir(exist_ok=False)
    shutil.copy2(options.manifest, root/"figure5_manifest.json")
    argv = [sys.executable, "-m", "scripts.runs.ministral3_reliable", "--sweep", "qps",
        "--qps-values", *map(str, manifest["loads"]["qps_values"]),
        "--qps-utilities", *GROUPS[options.group], "--output-dir", str(outputs),
        "--output-prefix", "ministral3_measured_"+options.group,
        *manifest["experiment_argv"], "--instances-config", options.instances_config]
    for path in options.wait_log:
        argv.extend(["--per-request-wait-log", path])
    write_json(root/"invocation.json", {"argv": argv, "manifest_sha256": sha256(options.manifest)})
    with (root/"driver.log").open("x") as stream:
        subprocess.run(argv, check=True, stdout=stream, stderr=subprocess.STDOUT)
    paths = point_paths(outputs)
    audit = audit_points(paths, manifest, GROUPS[options.group])
    audit.update(group=options.group, manifest_sha256=sha256(options.manifest),
                 point_sha256={str(p): sha256(p) for p in paths})
    write_json(root/"audit.json", audit)


def collate(options, manifest):
    from scripts.eval.augment_router_actual_accuracy import load_req_map, load_quality_index, augment_file
    root = Path(options.output_root).resolve()
    if root.exists():
        raise ValueError("Preserve prior collation outputs; choose a new directory")
    raw_paths = []
    digest = sha256(options.manifest)
    for raw in map(Path, options.raw_runs):
        audit = read_json(raw/"audit.json")
        if audit.get("status") != "PASS" or audit.get("manifest_sha256") != digest:
            raise ValueError("Raw sweep is not audited against this manifest")
        paths = point_paths(raw/"outputs")
        if {str(p): sha256(p) for p in paths} != audit["point_sha256"]:
            raise ValueError("Raw sweep points changed after their audit")
        raw_paths.extend(paths)
    audit_points(raw_paths, manifest, manifest["policies"])
    before = {str(p): sha256(p) for p in raw_paths}
    root.mkdir(parents=True)
    shutil.copy2(options.manifest, root/"figure5_manifest.json")
    req_map = load_req_map(Path(manifest["request_map"]))
    quality = load_quality_index(Path(manifest["scored_root"]))
    derived_dirs, derived_paths = [], []
    for index, raw in enumerate(map(Path, options.raw_runs)):
        dest = root/"derived_sweeps"/f"part_{index}"
        dest.mkdir(parents=True)
        derived_dirs.append(dest)
        for source in point_paths(raw/"outputs"):
            path = dest/source.name
            shutil.copy2(source, path)
            stats = augment_file(json_path=path, req_maps_by_holdout={manifest["holdout_prompts_per_bucket"]: req_map},
                                 quality_index=quality, dry_run=False)
            if (stats.skipped_reason or stats.missing_req_map or stats.missing_example_id or
                    stats.unresolved_model or stats.missing_quality):
                raise ValueError(f"Incomplete actual-quality join: {path}")
            derived_paths.append(path)
    audit = audit_points(derived_paths, manifest, manifest["policies"], augmented=True)
    figure = root/"figure5_offered_load"
    subprocess.run([sys.executable, "-m", "scripts.reporting.router_qps_sweep_summary",
                    *map(str, derived_dirs), "--output-dir", str(figure)], check=True)
    summary = read_json(figure/"router_qps_sweep_summary.json")
    if (set(summary["utilities"]) != set(manifest["policies"]) or
            {float(q) for q in summary["qps"]} != set(manifest["loads"]["qps_values"]) or
            any(not isinstance(row.get("actual_slo_gated_utility_mean"), (int, float)) or
                not math.isfinite(row["actual_slo_gated_utility_mean"])
                for entries in summary["qps"].values() for row in entries.values())):
        raise ValueError("Figure 5 summary is missing measured OnTimeUtility cells")
    if any(sha256(p) != expected for p, expected in before.items()):
        raise ValueError("Raw sweep changed during collation")
    plot = figure/"actual_slo_gated_utility_mean_vs_qps.png"
    if not plot.is_file() or not plot.stat().st_size:
        raise ValueError("Missing Figure 5 plot")
    audit.update(manifest_sha256=digest, raw_sweeps_modified=False,
        raw_point_sha256=before, judge_imputed_model_scores=manifest["judge_imputed_model_scores"],
        plot=str(plot))
    write_json(root/"audit.json", audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "validate", "sweep", "collate"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stage-dir")
    parser.add_argument("--review-json")
    parser.add_argument("--output-root")
    parser.add_argument("--group", choices=GROUPS, default="all")
    parser.add_argument("--instances-config")
    parser.add_argument("--wait-log", action="append", default=[])
    parser.add_argument("--raw-runs", nargs="+")
    parser.add_argument("--service-run-dir")
    parser.add_argument("--predictor-run-dir")
    options = parser.parse_args()
    if options.mode == "freeze":
        if not options.stage_dir or not options.review_json:
            parser.error("freeze requires --stage-dir and --review-json")
        freeze(options.stage_dir, options.review_json, options.manifest, os.environ["SFS_ROOT"])
    else:
        manifest = load_manifest(options.manifest)
        for name in ("service_run_dir", "predictor_run_dir"):
            supplied = getattr(options, name)
            if supplied is not None and Path(supplied).resolve() != Path(manifest[name]).resolve():
                parser.error(f"{name} differs from the frozen serving inputs")
        if options.mode == "sweep":
            if not options.output_root or not options.instances_config or len(options.wait_log) != 3:
                parser.error("sweep requires output root, instances, and three wait logs")
            run_sweep(options, manifest)
        elif options.mode == "collate":
            if not options.output_root or not options.raw_runs:
                parser.error("collate requires output root and raw runs")
            collate(options, manifest)
    print(json.dumps({"status": "PASS", "mode": options.mode, "manifest": options.manifest}))


if __name__ == "__main__":
    main()

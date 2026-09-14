from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from scripts.runs import ministral3_figure5 as fig
from scripts.runs.ministral3_methodology_stage import POLICIES
from scripts.prep.prepare_methodology_service import PROFILE, sha256


def contract(tmp_path, count=3):
    frozen = tmp_path/"source.txt"
    frozen.write_text("frozen input")
    return {"schema_version": 2, "data_role": "evaluation", "policies": list(POLICIES),
        "holdout_start_index": 2500, "holdout_prompts_per_bucket": 2000,
        "groups": fig.GROUPS, "requests_per_cell": count, "matrix_cells": 32, "seed": 69,
        "serving_profile": PROFILE, "loads": {"qps_values": [2.000001, 3, 4, 5]},
        "configuration": {"lambda_weight": .0005, "delta_weight": 0,
            "routebalance_weights": [1/3]*3, "routebalance_batch_max_size": 16,
            "routebalance_batch_wait_ms": 25}, "file_sha256": {str(frozen): sha256(frozen)}}


def points(tmp_path, manifest):
    paths = []
    for index, rate in enumerate(manifest["loads"]["qps_values"]):
        rows = [{"request_id": f"req-{i}", "response_id": f"response-{i}",
                 "system_entry_offset_s": i/rate, "actual_accuracy": .8}
                for i in range(manifest["requests_per_cell"]) ]
        payload = {"config": {**manifest["configuration"], "request_rate_qps": rate,
            "seed": 69, "arrival_process": "poisson", "instance_metadata": {"serving_profile": PROFILE}},
            "request_set": {"num_requests": len(rows)}, "router": {"runs": [
                {"utility": p, "summary": {"succeeded_requests": len(rows), "failed_requests": 0,
                    "system_entry_e2e_ttft_missing_count": 0}, "per_request": deepcopy(rows)} for p in POLICIES]}}
        path = tmp_path/f"point_{index}.json"
        path.write_text(json.dumps(payload))
        paths.append(path)
    return paths


def test_frozen_manifest_rejects_changed_input_and_legacy_matrix(tmp_path):
    manifest = contract(tmp_path, 8000)
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps(manifest))
    assert fig.load_manifest(path)["matrix_cells"] == 32
    for reduced in ({"requests_per_cell": 16000}, {"holdout_prompts_per_bucket": 4000}):
        path.write_text(json.dumps({**manifest, **reduced}))
        with pytest.raises(ValueError, match="Invalid eight-policy"):
            fig.load_manifest(path)
    path.write_text(json.dumps(manifest))
    (tmp_path/"source.txt").write_text("changed")
    with pytest.raises(ValueError, match="input changed"):
        fig.load_manifest(path)
    manifest["policies"] = ["hard", "shortest_queue", "latency_agnostic", "round_robin"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Invalid eight-policy"):
        fig.load_manifest(path, verify_files=False)


def test_matrix_audit_requires_every_cell_and_actual_arrivals(tmp_path):
    manifest = contract(tmp_path)
    paths = points(tmp_path, manifest)
    audit = fig.audit_points(paths, manifest, POLICIES, augmented=True)
    assert audit["matrix_cells"] == 32 and audit["records"] == 96
    assert len(audit["arrival_attainment"]) == 32
    with pytest.raises(ValueError, match="Missing Figure 5 cells"):
        fig.audit_points(paths[:-1], manifest, POLICIES)
    with pytest.raises(ValueError, match="Duplicate/unexpected"):
        fig.audit_points(paths+[paths[0]], manifest, POLICIES)
    payload = fig.read_json(paths[0])
    payload["router"]["runs"][0]["per_request"][-1]["system_entry_offset_s"] *= 2
    paths[0].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="arrival rate"):
        fig.audit_points(paths, manifest, POLICIES)


@pytest.mark.parametrize("mutation,match", [
    (lambda p: p["config"].update(lambda_weight=1), "Frozen configuration"),
    (lambda p: p["router"]["runs"][0]["per_request"][0].update(error="failed"), "failed measured"),
    (lambda p: p["router"]["runs"][0]["per_request"][0].update(actual_accuracy=None), "judged quality"),
])
def test_bad_cells_cannot_pass_collation(tmp_path, mutation, match):
    manifest = contract(tmp_path)
    paths = points(tmp_path, manifest)
    payload = fig.read_json(paths[0])
    mutation(payload)
    paths[0].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=match):
        fig.audit_points(paths, manifest, POLICIES, augmented=True)


@pytest.mark.parametrize("group", ["all", "snapshot", "baseline"])
def test_sweep_invocation_uses_only_frozen_rates_and_group(tmp_path, monkeypatch, group):
    manifest = contract(tmp_path)
    manifest["experiment_argv"] = ["--num-requests", "8000"]
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps(manifest))
    root = tmp_path/"run"
    root.mkdir()
    commands = []
    monkeypatch.setattr(fig.subprocess, "run", lambda argv, **kw: commands.append(argv))
    monkeypatch.setattr(fig, "audit_points", lambda *a, **kw: {"status": "PASS"})
    options = SimpleNamespace(manifest=path, output_root=root, group=group,
        instances_config="instances.json", wait_log=["a", "b", "c"])
    fig.run_sweep(options, manifest)
    argv = commands[0]
    assert argv[argv.index("--qps-values")+1:argv.index("--qps-utilities")] == ["2.000001", "3", "4", "5"]
    assert argv[argv.index("--qps-utilities")+1:argv.index("--output-dir")] == fig.GROUPS[group]
    assert argv[argv.index("--num-requests")+1] == "8000"
    assert fig.read_json(root/"audit.json")["manifest_sha256"] == sha256(path)


def test_qps_summary_preserves_precision_and_adaptation_labels():
    from scripts.reporting.router_qps_sweep_summary import _format_qps, DISPLAY_LABELS
    assert float(_format_qps(2.000001)) == 2.000001
    assert "adaptation" in DISPLAY_LABELS["mooncake_prefill"]


def test_point_discovery_excludes_real_router_diagnostic_json(tmp_path):
    point = tmp_path/"router_qps6_point01.json"
    point.write_text('{}')
    (tmp_path/"router_qps6_point01_predicted_waits_router_hard_simulation_latency_distribution.json").write_text('{}')
    (tmp_path/"router_manifest.json").write_text('{}')
    assert fig.point_paths(tmp_path) == [point]


@pytest.mark.parametrize("grouped", [False, True])
def test_collation_joins_only_derived_copies_and_checks_complete_matrix(tmp_path, monkeypatch, grouped):
    from scripts.reporting.router_qps_sweep_summary import aggregate_jsons
    manifest = contract(tmp_path)
    req_map = tmp_path/"requests.csv"
    req_map.write_text("req_id,bucket,example_id\n" +
                      "".join(f"req-{i},alpaca,example-{i}\n" for i in range(3)))
    scored = tmp_path/"scores/ministral3-3b"
    scored.mkdir(parents=True)
    (scored/"alpaca_scored.jsonl").write_text("".join(json.dumps({
        "quality": .8, "prompt_metadata": {"example_id": f"example-{i}"}})+"\n" for i in range(3)))
    manifest.update(request_map=str(req_map), scored_root=str(scored.parent), judge_imputed_model_scores=0)
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps(manifest))
    raw_runs, hashes = [], {}
    for group in (["snapshot", "baseline"] if grouped else ["all"]):
        raw = tmp_path/group
        raw_runs.append(str(raw))
        (raw/"outputs").mkdir(parents=True)
        paths = points(raw/"outputs", manifest)
        for point in paths:
            payload = fig.read_json(point)
            payload["config"]["prompt_source"] = {"holdout_prompts_per_bucket": 2000}
            payload["router"]["runs"] = [run for run in payload["router"]["runs"] if run["utility"] in fig.GROUPS[group]]
            for run in payload["router"]["runs"]:
                for row in run["per_request"]:
                    row.pop("actual_accuracy")
                    row.update(bucket="alpaca", instance_id="vllm-ministral3-3b", actual_cost=1.0,
                        actual_cost_source="usage", system_entry_e2e_ttft_slo_met=True)
            point.write_text(json.dumps(payload))
        group_hashes = {str(p): sha256(p) for p in paths}
        hashes.update(group_hashes)
        (raw/"audit.json").write_text(json.dumps({"status": "PASS", "manifest_sha256": sha256(path),
                                                 "point_sha256": group_hashes}))

    def report(argv, **kw):
        end = argv.index("--output-dir")
        summary, _ = aggregate_jsons([fig.Path(p) for p in argv[3:end]])
        dest = fig.Path(argv[end+1])
        dest.mkdir()
        (dest/"router_qps_sweep_summary.json").write_text(json.dumps(summary))
        (dest/"actual_slo_gated_utility_mean_vs_qps.png").write_bytes(b"plot fixture")
    monkeypatch.setattr(fig.subprocess, "run", report)
    options = SimpleNamespace(manifest=path, raw_runs=raw_runs, output_root=tmp_path/"derived")
    fig.collate(options, manifest)
    assert all(sha256(p) == digest for p, digest in hashes.items())
    audit = fig.read_json(options.output_root/"audit.json")
    assert audit["records"] == 96 and audit["raw_sweeps_modified"] is False
    summary = fig.read_json(options.output_root/"figure5_offered_load/router_qps_sweep_summary.json")
    assert summary["qps"]["2.000001"]["mooncake_prefill"]["actual_slo_gated_utility_mean"] == pytest.approx(.7995)

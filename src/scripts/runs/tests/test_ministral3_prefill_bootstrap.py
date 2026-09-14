from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.runs import ministral3_prefill_bootstrap as boot


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def revision(tmp_path, monkeypatch):
    old_path, new_path = tmp_path/"old/timing.json", tmp_path/"new/timing.json"
    models = {}
    for index, model in enumerate(boot.MODELS):
        name = f"head_{index}.json"
        for folder in (old_path.parent, new_path.parent):
            dump(folder/name, {"test_only": True})
        models[model] = {"service_rate_qps": index+1,
                         "tpot": {"model_file": name, "feature_names": ["decode_tokens"]},
                         "prefill": {"coverage": {"fit_rows": 64}, "coefficients_ms": [10]}}
    old = {"schema_version": 1, "data_role": "calibration", "serving_profile": boot.PROFILE,
           "models": models, "source_manifest": "original_calibration"}
    dump(old_path, old)
    new = deepcopy(old)
    new["parent_calibration_sha256"] = boot.sha256(old_path)
    for model in new["models"]:
        new["models"][model]["prefill"]["coefficients_ms"] = [5]
    dump(new_path, new)
    audit = dump(tmp_path/"audit.json", {
        "status": "CPU_VALIDATED_GPU_SMOKE_REQUIRED", "candidate_sha256": boot.sha256(new_path),
        "parent_calibration_sha256": boot.sha256(old_path)})
    review = dump(tmp_path/"review.json", {
        "status": "PREFILL_CORRECTION_REQUIRES_MATCHED_GPU_SMOKE",
        "candidate_timing_calibration_sha256": boot.sha256(new_path),
        "original_timing_calibration_sha256": boot.sha256(old_path),
        "prefill_fit_review": "test-only prefill review", "tpot_fit_review": "unchanged heads",
        "arrival_review": "actual arrivals must be checked"})

    class Calibration:
        def __init__(self, path):
            self.path = Path(path)
            self.payload = boot.read_json(path)
            self.models = self.payload["models"]

        def validate_runtime_profile(self, profile):
            if self.payload["serving_profile"] != profile:
                raise ValueError("profile mismatch")

        def preload(self):
            pass

    from sfs_core.routing.methodology_calibration import MethodologyCalibration
    monkeypatch.setattr(MethodologyCalibration, "load", Calibration)
    return old_path, new_path, audit, review


def test_only_prefill_revision_can_reuse_other_policy_evidence(revision):
    result = boot.validate_prefill_revision(*revision)
    assert result["status"] == "PASS_CPU"
    assert result["changed_policy"] == "mooncake_prefill"
    assert len(result["unaffected_policies"]) == 7


@pytest.mark.parametrize("mutation", ["speed", "tpot", "profile", "source", "hash", "unreviewed"])
def test_revision_rejects_unrelated_changes_even_with_updated_candidate_hash(revision, mutation):
    old, new, audit, review = revision
    payload = boot.read_json(new)
    if mutation == "speed":
        payload["models"][boot.MODELS[0]]["service_rate_qps"] += 1
    elif mutation == "tpot":
        payload["models"][boot.MODELS[0]]["tpot"]["feature_names"] = ["different"]
    elif mutation == "profile":
        payload["serving_profile"]["max_num_seqs"] = 1
    elif mutation == "source":
        payload["source_manifest"] = "unrelated"
    elif mutation == "hash":
        payload["parent_calibration_sha256"] = "incorrect"
    dump(new, payload)
    a, r = boot.read_json(audit), boot.read_json(review)
    a["candidate_sha256"] = r["candidate_timing_calibration_sha256"] = boot.sha256(new)
    if mutation == "unreviewed":
        r["status"] = "PENDING"
    dump(audit, a)
    dump(review, r)
    with pytest.raises(ValueError):
        boot.validate_prefill_revision(*revision)


def smoke(candidate):
    expected, rows = [], []
    for i in range(192):
        req = SimpleNamespace(request_id=f"r-{i}", bucket=boot.BUCKETS[i//48], prompt_tokens=i+1)
        expected.append(req)
        rows.append({**vars(req), "response_id": f"response-{i}", "error": None,
                     "queue_delay_ms": 1, "ttft_ms": 2, "system_entry_to_dispatch_ms": 3,
                     "system_entry_offset_s": i/2, "actual_cost": .1,
                     "usage_completion_tokens": 4,
                     "instance_id": f"vllm-{boot.MODELS[0]}",
                     "methodology_terms": {"policy": "mooncake_prefill", "candidates": {
                         f"vllm-{model}": {"queued_prefill_ms": 0, "incoming_prefill_ms": j+1,
                                            "total_prefill_ms": j+1}
                         for j, model in enumerate(boot.MODELS)}}})
    payload = {"test_only": True, "runs": [{"utility": "mooncake_prefill", "per_request": rows,
        "summary": {"succeeded_requests": 192, "failed_requests": 0},
        "methodology_config": {"calibration": {"artifact_sha256": boot.sha256(candidate)},
                               "unfinished_after_drain": {model: 0 for model in boot.MODELS}}}]}
    return payload, expected


@pytest.mark.parametrize("mutation", [None, "identity", "timing", "missing_decision", "error", "leak", "arrival", "candidate", "accounting", "choice"])
def test_fresh_smoke_checks_actual_outcomes_and_arrivals(tmp_path, mutation):
    candidate = dump(tmp_path/"candidate.json", {"test_only": True})
    payload, requests = smoke(candidate)
    run = payload["runs"][0]
    if mutation == "identity":
        run["per_request"][0]["request_id"] = "wrong"
    elif mutation == "timing":
        run["methodology_config"]["calibration"]["artifact_sha256"] = "wrong"
    elif mutation == "missing_decision":
        run["per_request"][0]["methodology_terms"] = None
    elif mutation == "error":
        run["per_request"][0]["error"] = "failed"
    elif mutation == "leak":
        run["methodology_config"]["unfinished_after_drain"][boot.MODELS[0]] = 1
    elif mutation == "arrival":
        for row in run["per_request"]:
            row["system_entry_offset_s"] *= 2
    elif mutation == "candidate":
        run["per_request"][0]["methodology_terms"]["candidates"].pop(f"vllm-{boot.MODELS[-1]}")
    elif mutation == "accounting":
        run["per_request"][0]["methodology_terms"]["candidates"][f"vllm-{boot.MODELS[0]}"]["total_prefill_ms"] = 999
    elif mutation == "choice":
        run["per_request"][0]["instance_id"] = f"vllm-{boot.MODELS[-1]}"
    if mutation:
        with pytest.raises(ValueError):
            boot.validate_smoke(payload, candidate, requests)
    else:
        assert boot.validate_smoke(payload, candidate, requests)["realized_arrival_qps"] == 2


def endpoint_events(rate, completion_rate):
    arrivals = [{"event": "arrival", "t": i/rate} for i in range(int(rate*360))]
    completed = [{"event": "completion", "t": 10+i/completion_rate, "error": None}
                 for i in range(int(completion_rate*350))]
    completed += [{"event": "completion", "t": 400+i/10, "error": None}
                  for i in range(len(arrivals)-len(completed))]
    return sorted(arrivals+completed+[
        {"event": "arrivals_end", "t": 360, "reason": "duration_limit"},
        {"event": "sample", "t": 240, "engines": {"a": {}, "b": {}, "c": {}}}], key=lambda r: r["t"])


@pytest.fixture
def completed_scout(tmp_path):
    stage = tmp_path/"stage"
    timing = dump(stage/"timing_models/methodology_calibration.json", {"test_only": True})
    trials = [{"classification": "stable", "requested_qps": 7.75},
              {"classification": "unstable", "requested_qps": 8}]
    loads = boot.freeze_loads(trials)
    endpoints = []
    for index, rate in enumerate((loads["qps_values"][0], loads["qps_values"][-1])):
        events = endpoint_events(rate, rate if index == 0 else rate/2)
        label = f"confirm_{index}"
        (stage/f"{label}_events.jsonl").write_text("\n".join(map(json.dumps, events))+"\n")
        payload, _ = smoke(timing)
        prototype = payload["runs"][0]["per_request"][0]
        payload["runs"][0]["per_request"] = [
            {**deepcopy(prototype), "request_id": str(i), "response_id": f"response-{i}"}
            for i in range(int(rate*360))]
        dump(stage/f"{label}.json", payload)
        endpoints.append({**boot.classify_trial(events, requested_qps=rate), "run_file": f"{label}.json"})
    scout = {"status": "PASS", "data_role": "calibration", "reference_policy": "shortest_queue",
             "serving_profile": boot.PROFILE, "timing_calibration_sha256": boot.sha256(timing),
             "final_policy_ids": list(boot.POLICIES), "final_requests_per_cell": 16000,
             "final_matrix_cells": 32, "trials": trials, "endpoint_confirmations": endpoints, **loads}
    dump(stage/"capacity_scout.json", scout)
    return stage, {"trials": trials}


@pytest.mark.parametrize("mutation", [None, "unfinished", "wide", "endpoint_events", "trial_record", "budget"])
def test_completed_scout_preserves_measurement_gates(completed_scout, mutation):
    stage, reuse = completed_scout
    scout = boot.read_json(stage/"capacity_scout.json")
    if mutation == "unfinished":
        scout["status"] = "COLLECTING"
    elif mutation == "wide":
        reuse["trials"][0]["requested_qps"] = 6
    elif mutation == "endpoint_events":
        with (stage/"confirm_1_events.jsonl").open("a") as stream:
            stream.write(json.dumps({"event": "sample_error", "t": 250, "error": "lost telemetry"})+"\n")
    elif mutation == "trial_record":
        scout["trials"][0]["reason"] = "fabricated"
    elif mutation == "budget":
        scout["final_requests_per_cell"] = 8000
    dump(stage/"capacity_scout.json", scout)
    if mutation:
        with pytest.raises(ValueError):
            boot.validate_completed_scout(stage, reuse)
    else:
        actual, files = boot.validate_completed_scout(stage, reuse)
        assert actual["qps_values"] == [5.2, 6.8, 7.6, 8.4]
        assert len(files) == 5


@pytest.mark.parametrize("failure", [None, "smoke", "freeze"])
@pytest.mark.parametrize("smoke_only", [False, True])
def test_evaluation_is_unreachable_until_smoke_and_freeze_pass(tmp_path, monkeypatch, failure, smoke_only):
    import sys
    calls = []
    monkeypatch.setattr(sys, "argv", ["bootstrap", "run", "--stage-dir", "stage",
        "--candidate", "candidate", "--candidate-audit", "audit", "--timing-review", "review",
        "--manifest", str(tmp_path/"final.json"), "--output-root", str(tmp_path),
        "--instances-config", "instances", "--service-metrics-json", "metrics",
        "--wait-log", "a", "--wait-log", "b", "--wait-log", "c", *(["--smoke-only"] if smoke_only else [])])

    def validated(options, *, require_capacity):
        assert require_capacity is True
        calls.append("validate")
        return {}

    async def run_smoke(options, inputs):
        calls.append("smoke")
        if failure == "smoke":
            raise ValueError("smoke rejected")
        return tmp_path/"smoke"

    def freeze(options, inputs, folder):
        calls.append("freeze")
        if failure == "freeze":
            raise ValueError("freeze rejected")
        return {"requests_per_cell": 8000}

    def sweep(options, manifest):
        assert options.group == "baseline" and manifest["requests_per_cell"] == 8000
        calls.append("sweep")

    monkeypatch.setattr(boot, "validate_inputs", validated)
    monkeypatch.setattr(boot, "run_smoke", run_smoke)
    monkeypatch.setattr(boot, "compose_and_freeze", freeze)
    monkeypatch.setattr(boot.figure5, "run_sweep", sweep)
    if failure:
        with pytest.raises(ValueError, match="rejected"):
            boot.main()
        assert "sweep" not in calls
    else:
        boot.main()
        assert calls == ["validate", "smoke", "freeze"] + ([] if smoke_only else ["sweep"])


@pytest.mark.parametrize("corrupt_fresh", [False, True])
@pytest.mark.parametrize("refresh_all", [False, True])
def test_composition_preserves_raw_measurements_and_records_each_policy(tmp_path, monkeypatch, corrupt_fresh, refresh_all):
    monkeypatch.setenv("SFS_ROOT", str(tmp_path))
    stage, output, fresh = tmp_path/"source_stage", tmp_path/"output", tmp_path/"fresh_smoke"
    output.mkdir()
    candidate = dump(tmp_path/"candidate/methodology_calibration.json", {"test_only": True})
    dump(stage/"stage_started.json", {"prepared_dir": "calibration"})
    scout = {"test_only": True, "status": "PASS", "trials": [{"requested_qps": 9}],
             "timing_calibration_sha256": "original", "endpoint_confirmations": []}
    dump(stage/"capacity_scout.json", scout)
    for policy in boot.POLICIES:
        dump(stage/f"smoke_{policy}.json", {"test_only": True, "policy": policy})
    dump(stage/"smoke_audit.json", {"status": "PASS", "policies": list(boot.POLICIES),
                                  "summaries": {p: {"old": True} for p in boot.POLICIES}})
    payload, _ = smoke(candidate)
    raw = dump(fresh/"smoke_mooncake_prefill.json", payload)
    event_path = fresh/"events.jsonl"
    event_path.write_text('{"test_only":true}\n')
    dump(fresh/"audit.json", {"status": "PASS", "data_role": "calibration",
        "timing_calibration_sha256": boot.sha256(candidate), "raw_smoke_sha256": boot.sha256(raw),
        "events_sha256": boot.sha256(event_path)})
    review = dump(tmp_path/"review.json", {"prefill_fit_review": "reviewed CPU residuals",
        "tpot_fit_review": "unchanged", "arrival_review": "latency includes collection delay"})
    if refresh_all:
        for policy in boot.POLICIES:
            content = deepcopy(payload)
            content["runs"][0]["utility"] = policy
            dump(fresh/f"smoke_{policy}.json", content)
        audit = boot.read_json(fresh/"audit.json")
        audit.update(refreshed_policies=list(boot.POLICIES),
                     policy_sha256={p: boot.sha256(fresh/f"smoke_{p}.json") for p in boot.POLICIES})
        dump(fresh/"audit.json", audit)
    before = {str(p): boot.sha256(p) for p in stage.glob('*.json')}
    inputs = {"file_sha256": before, "scout": scout, "revision": {"original_sha256": "original"}}
    options = SimpleNamespace(stage_dir=stage, output_root=output, candidate=candidate,
                              timing_review=review, manifest=tmp_path/"published.json",
                              source_review=review if refresh_all else None)
    calls = []

    def freeze(derived, review, base_path, root):
        calls.append("freeze")
        combined = boot.read_json(derived/"smoke_audit.json")
        assert set(combined["policy_evidence"]) == set(boot.POLICIES)
        assert sum(row["reused"] for row in combined["policy_evidence"].values()) == (0 if refresh_all else 7)
        assert boot.read_json(derived/"capacity_scout.json")["trials"] == scout["trials"]
        return {"test_only": True, "file_sha256": {}}

    monkeypatch.setattr(boot.figure5, "freeze", freeze)
    monkeypatch.setattr(boot.figure5, "load_manifest", boot.read_json)
    if corrupt_fresh:
        raw.write_text('{}')
        with pytest.raises(ValueError, match="Fresh GPU smoke"):
            boot.compose_and_freeze(options, inputs, fresh)
        assert not options.manifest.exists() and calls == []
    else:
        result = boot.compose_and_freeze(options, inputs, fresh)
        assert result["prefill_revision"]["sha256"] == before[str(stage/"capacity_scout.json")]
        assert result["file_sha256"][str(raw)] == boot.sha256(raw)
        assert calls == ["freeze"]
    assert {p: boot.sha256(p) for p in before} == before

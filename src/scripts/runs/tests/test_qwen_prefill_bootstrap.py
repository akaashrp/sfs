import asyncio
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.runs import qwen_baselines as qwen
from scripts.runs import qwen_prefill_bootstrap as boot
from scripts.runs.tests.test_ministral3_prefill_bootstrap import dump, revision, smoke


def test_qwen_revision_checks_its_own_profile_and_candidate_pool(revision):
    old, new, audit, review = revision
    from scripts.runs.ministral3_prefill_bootstrap import MODELS
    for path in (old, new):
        value = boot.read_json(path)
        value["models"] = {q: value["models"][m] for q, m in zip(qwen.MODELS, MODELS)}
        value["serving_profile"] = deepcopy(qwen.PROFILE)
        if path == new:
            value["parent_calibration_sha256"] = qwen.sha256(old)
        dump(path, value)
    a, r = boot.read_json(audit), boot.read_json(review)
    a["candidate_sha256"] = r["candidate_timing_calibration_sha256"] = qwen.sha256(new)
    a["parent_calibration_sha256"] = r["original_timing_calibration_sha256"] = qwen.sha256(old)
    dump(audit, a)
    dump(review, r)
    assert boot.validate_prefill_revision(*revision, models=qwen.MODELS, profile=qwen.PROFILE)["status"] == "PASS_CPU"
    with pytest.raises(ValueError):
        boot.validate_prefill_revision(*revision)


@pytest.mark.parametrize("failure", [False, True])
def test_evaluation_is_unreachable_until_timing_gate_returns(tmp_path, monkeypatch, failure):
    output = tmp_path/"output"
    output.mkdir()
    candidate = dump(tmp_path/"new/timing.json", {"test_only": True})
    manifest = {"sfs_root": str(tmp_path), "experiment_argv": ["--num-requests", "16000"],
                "manifest_path": str(dump(tmp_path/"manifest.json", {"test_only": True}))}
    calls = []
    def gate(instances, folder):
        calls.append("gate")
        assert instances == tmp_path/"instances.json" and folder == output
        if failure:
            raise ValueError("test smoke rejected")
        return candidate
    def execute(argv, **kwargs):
        calls.append("evaluate")
        assert calls == ["gate", "evaluate"]
        assert argv[argv.index("--methodology-calibration-json")+1] == str(candidate)
        assert argv[argv.index("--service-metrics-json")+1] == str(tmp_path/"old/model_metrics.json")
        assert argv[argv.index("--num-requests")+1] == "16000"
        assert argv[argv.index("--qps-values")+1:argv.index("--qps-utilities")] == list(map(str, qwen.QPS))
    monkeypatch.setattr(qwen.subprocess, "run", execute)
    if failure:
        with pytest.raises(ValueError, match="smoke rejected"):
            qwen.run_sweep(manifest, tmp_path/"old", output, tmp_path/"instances.json", tmp_path/"predictor", gate)
        assert calls == ["gate"]
        assert not list(output.iterdir())
    else:
        qwen.run_sweep(manifest, tmp_path/"old", output, tmp_path/"instances.json", tmp_path/"predictor", gate)
        assert boot.read_json(output/"sweep_completed.json")["timing_sha256"] == qwen.sha256(candidate)


@pytest.mark.parametrize("corrupt", [False, True])
def test_qualified_qwen_timing_requires_measured_smoke_and_preserves_raw(tmp_path, monkeypatch, corrupt):
    from scripts.runs import experiments as exp
    from scripts.runs import experiments_sweep
    from sfs_core.shared import shared_experiment_helpers
    candidate = dump(tmp_path/"candidate/timing.json", {"test_only": True})
    payload, expected = smoke(candidate)
    for row in payload["runs"][0]["per_request"]:
        row["instance_id"] = "vllm-0.6b"
        values = list(row["methodology_terms"]["candidates"].values())
        row["methodology_terms"]["candidates"] = dict(zip(("vllm-0.6b", "vllm-8b", "vllm-32b"), values))
    requests = [exp.ExperimentRequest(r.request_id, "test-only prompt", r.prompt_tokens, r.bucket, 1000, 100, 100)
                for r in expected]
    monkeypatch.setattr(qwen, "rows", lambda _: [asdict(r) for r in requests])
    monkeypatch.setattr(experiments_sweep, "_parse_experiment_args", lambda _: SimpleNamespace())
    clients = {f"vllm-{m.removeprefix('qwen3-')}": SimpleNamespace(model_id=m) for m in qwen.MODELS}
    closed = []
    for name, client in clients.items():
        client.close = lambda name=name: closed.append(name)
    monkeypatch.setattr(exp, "load_instances", lambda _: (clients, {}, {"serving_profile": qwen.PROFILE}))
    async def noop(*_):
        pass
    monkeypatch.setattr(qwen, "wait_drained", noop)
    monkeypatch.setattr(shared_experiment_helpers, "warm_up_instances", noop)
    output = tmp_path/"output"
    output.mkdir()
    async def route(**kwargs):
        assert kwargs["args"].num_requests == 192 and kwargs["args"].request_rate_qps == 2.0
        (output/"prefill_smoke/events.jsonl").write_text('{"test_only":true}\n')
        if corrupt:
            payload["runs"][0]["methodology_config"]["calibration"]["artifact_sha256"] = "wrong"
        return payload
    monkeypatch.setattr(exp, "run_router_experiment", route)
    dump(tmp_path/"original/smoke_audit.json", {"test_only": True})
    # The production artifact name is fixed by composition.
    candidate.rename(candidate.with_name("methodology_calibration.json"))
    candidate = candidate.with_name("methodology_calibration.json")
    options = SimpleNamespace(stage_dir=tmp_path/"original", candidate=candidate, predictor=tmp_path/"predictor")
    manifest = {"experiment_argv": [], "calibration_requests": "test-only"}
    inputs = {"file_sha256": {}, "revision": {"test_only": True}}
    task = boot.qualify(options, manifest, inputs, tmp_path/"instances.json", output)
    if corrupt:
        with pytest.raises(ValueError, match="mismatched"):
            asyncio.run(task)
        assert not (output/"qualified_timing").exists()
        assert not (output/"prefill_smoke/audit.json").exists()
    else:
        result = asyncio.run(task)
        assert qwen.sha256(result) == qwen.sha256(candidate)
        assert boot.read_json(output/"prefill_smoke/audit.json")["status"] == "PASS"
    assert len(closed) == 3
    assert (output/"prefill_smoke/smoke_mooncake_prefill.json").exists()
    assert (tmp_path/"original/smoke_audit.json").exists()

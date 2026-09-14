"""Regress the post-startup failure through real config rendering and loading."""
import json
from pathlib import Path

import pytest

from scripts.runs import experiments as exp
from scripts.runs import ministral3_methodology_stage as stage
from scripts.prep.prepare_methodology_service import MODELS, PROFILE, sha256


def test_production_renderer_loader_and_stage_contract(tmp_path):
    metrics = tmp_path/"model_metrics.json"
    metrics.write_text(json.dumps({model: {"sfs_simulation": dict.fromkeys(
        ("intercept", "prefill_coeff", "prefill_sq_coeff", "decode_coeff", "sum_coeff", "sum_sq_coeff"), .01)}
        for model in MODELS}))
    manifest = tmp_path/"service_manifest.json"
    manifest.write_text(json.dumps({"models": {m: {"traces": [str(tmp_path/"trace.csv")]}
                                              for m in MODELS},
                                    "service_metrics_sha256": sha256(metrics)}))
    root = Path(__file__).resolve().parents[4]
    result = stage.validate_rendered_pool(root, manifest)
    assert result["status"] == "PASS"
    assert result["serving_profile"] == PROFILE
    assert result["renderer_and_loader_exercised"]


@pytest.mark.parametrize("profile", [PROFILE, None, {**PROFILE, "max_num_seqs": 128}])
def test_real_loader_preserves_provenance_and_rejects_wrong_stage_profile(tmp_path, profile):
    path, metrics, manifest = (tmp_path/name for name in ("instances.json", "metrics.json", "manifest.json"))
    metrics.write_text('{}')
    manifest.write_text(json.dumps({"service_metrics_sha256": sha256(metrics)}))
    payload = {"instances": [{"instance_id": m, "model_id": m, "default_model": m,
                               "address": "http://127.0.0.1:1"} for m in MODELS],
               "cost_units": "USD per million tokens", "service_metrics_json": str(metrics),
               "provenance": {"revision": "test"}}
    if profile is not None:
        payload["serving_profile"] = profile
    path.write_text(json.dumps(payload))
    instances, _, metadata = exp.load_instances(path)
    try:
        assert metadata["cost_units"] == payload["cost_units"]
        assert metadata["service_metrics_json"] == str(metrics)
        assert metadata["provenance"] == payload["provenance"]
        if profile == PROFILE:
            stage.validate_pool(instances, metadata, metrics, manifest)
        else:
            with pytest.raises(ValueError, match="serving profile"):
                stage.validate_pool(instances, metadata, metrics, manifest)
    finally:
        for client in instances.values():
            client.close()


def test_malformed_profile_fails_before_client_construction(tmp_path, monkeypatch):
    path = tmp_path/"instances.json"
    path.write_text(json.dumps({"instances": [{}], "serving_profile": "invalid"}))
    def unexpected(**_):
        pytest.fail("Malformed metadata must fail before client construction")
    monkeypatch.setattr(exp, "InstanceClient", unexpected)
    with pytest.raises(ValueError, match="must be an object"):
        exp.load_instances(path)

import hashlib
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.runs import ministral3_methodology_stage as stage


@pytest.mark.parametrize("fault", [None, "missing", "stale", "extra", "archive", "status"])
def test_source_review_is_exact_and_requires_new_gpu_smoke(tmp_path, fault):
    root, measured = tmp_path/"root", tmp_path/"measured"
    measured.mkdir()
    names = ["src/sfs_core/routing/existing.py", "src/scripts/runs/experiments.py",
             "src/scripts/runs/experiments_sweep.py", "src/sfs_core/shared/shared_experiment_helpers.py",
             "src/slurm/runs/ministral3_router_common.sh", "vllm/vllm/v1/engine/async_llm.py",
             "vllm/vllm/v1/core/sched/scheduler.py", "vllm/vllm/v1/core/sched/state_snapshot.py"]
    archive = measured/"source_snapshot.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for name in names:
            path = root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"original")
            info = tarfile.TarInfo(name)
            info.size = 8
            stream.addfile(info, io.BytesIO(b"original"))
    stage.validate_measured_sources(measured, root)
    changed = root/names[1]
    changed.write_bytes(b"reviewed addition")
    review = {"status": "PASS_CPU_REQUIRES_ALL_POLICY_GPU_SMOKE", "reason": "fixture review",
              "archive_sha256": stage.sha256(archive), "changes": {names[1]: {
                  "original_sha256": hashlib.sha256(b"original").hexdigest(),
                  "current_sha256": stage.sha256(changed)}}}
    if fault == "stale":
        changed.write_bytes(b"unreviewed change")
    elif fault == "extra":
        review["changes"]["extra"] = {}
    elif fault == "archive":
        review["archive_sha256"] = "wrong"
    elif fault == "status":
        review["status"] = "PASS"
    path = tmp_path/"review.json"
    path.write_text(json.dumps(review))
    if fault:
        with pytest.raises(ValueError):
            stage.validate_measured_sources(measured, root, None if fault == "missing" else path)
    else:
        assert len(stage.validate_measured_sources(measured, root, path)) == len(names)


def test_full_smoke_uses_native_predictor_and_frozen_settings(tmp_path):
    from scripts.runs.ministral3_prefill_bootstrap import configure_refreshed_smoke
    args = SimpleNamespace(routebalance_predictor_path=None, routebalance_batch_wait_ms=99,
                           routebalance_weights=[1, 0, 0], lambda_weight=1)
    options = SimpleNamespace(source_review=tmp_path/"review", routebalance_predictor=tmp_path/"native")
    config = {"routebalance_batch_wait_ms": 25, "routebalance_weights": [1/3]*3, "lambda_weight": .0005}
    assert configure_refreshed_smoke(args, options, {"scout": {"configuration": config}}) == list(stage.POLICIES)
    assert args.routebalance_predictor_path == str((tmp_path/"native").resolve())
    assert all(getattr(args, k) == v for k, v in config.items())
    with pytest.raises(ValueError, match="Unknown measured routing setting"):
        configure_refreshed_smoke(args, options, {"scout": {"configuration": {"invented": 1}}})

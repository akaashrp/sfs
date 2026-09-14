import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.runs import qwen_baselines as qwen
from scripts.runs import experiments as exp


def test_qwen_renderer_uses_real_loader_and_canonical_prices(tmp_path):
    config = qwen.pool_config({"manifest_path": "manifest.json"})
    path = tmp_path/"instances.json"
    path.write_text(json.dumps(config))
    instances, costs, metadata = exp.load_instances(path)
    try:
        assert metadata["serving_profile"] == qwen.PROFILE
        assert costs["vllm-32b"] == {"prompt": .287, "output": .64}
        assert {c.model_id for c in instances.values()} == set(qwen.MODELS)
        assert metadata["ttft_batch_params_by_instance"]["vllm-32b"]["max_num_batched_tokens"] == 32768
    finally:
        for client in instances.values(): client.close()


def test_production_qwen_server_argv_parses_with_pinned_vllm(tmp_path, monkeypatch):
    from vllm.platforms import current_platform
    # Supply the known target device type at the hardware-discovery boundary.
    # The real parser still checks every production CLI option and value.
    monkeypatch.setattr(current_platform, "device_type", "cuda")
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.utils import FlexibleArgumentParser
    root = Path(__file__).resolve().parents[4]
    parser = make_arg_parser(FlexibleArgumentParser())
    for i, row in enumerate(qwen.pool_config({})["instances"]):
        argv = qwen.server_argv(root, tmp_path/"model", row, i, tmp_path)
        args = parser.parse_args(argv[3:])
        assert args.tensor_parallel_size == (1,1,2)[i]
        assert args.max_num_batched_tokens == 32768
        assert args.max_num_seqs == 512
        assert args.enable_chunked_prefill
        assert not args.enable_prefix_caching
        assert args.snapshot_shm_name == row["snapshot_shm_name"]


def test_qwen_sweep_rejects_incomplete_smoke_before_gpu(tmp_path):
    (tmp_path/"smoke_audit.json").write_text(json.dumps({"status": "FAILED"}))
    with pytest.raises(ValueError, match="smoke gate"):
        qwen.validate_smoke(tmp_path, {"manifest_path": "does-not-exist"}, tmp_path)


def test_current_family_budgets_preserve_original_capacity_evidence():
    from scripts.runs.ministral3_methodology_stage import FINAL_REQUESTS_PER_CELL
    from scripts.runs.ministral3_figure5 import EVALUATION_REQUESTS_PER_CELL
    assert EVALUATION_REQUESTS_PER_CELL == 8000
    assert FINAL_REQUESTS_PER_CELL == 16000
    argv = qwen.base_argv(Path("/sfs"), Path("/cache"))
    assert argv[argv.index("--num-requests")+1] == "16000"
    assert set(qwen.NEW_POLICIES) == {"mooncake_prefill", "lmdeploy_proxy", "routebalance", "score", "vllm_sr_latency"}


def test_qwen_grid_and_policies_do_not_expand_ministral():
    assert qwen.QPS == (7.,8.3,8.6,8.9)
    assert len(qwen.POLICIES)==9 and "vllm_sr_latency" not in qwen.CORE_POLICIES

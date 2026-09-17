"""Remaining-length tables: build excludes evaluation ids and is deterministic; per-model rules; pool plumbing off by default."""
import csv
import hashlib
import json

import pytest

from scripts.cloud.pool import parse_remaining_length_rules, remaining_length_provenance, server_argv
from scripts.prep.remaining_length_tables import build

MODELS = ("qwen3-0.6b", "qwen3-8b")


def _calibration(root, heldout_ids=()):
    summary = {"models": {}}
    for model in MODELS:
        summary["models"][model] = {"datasets": {}}
        for bucket, prompt in (("alpaca", 40), ("govreport-summarization", 9000)):
            lengths = [(i * 7 + len(model)) % 300 + 1 for i in range(2500)]
            path = root / model / "outputs" / f"{bucket}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as f:
                for i, n in enumerate(lengths):
                    f.write(json.dumps({"model_label": model, "prompt_index": i, "error": None,
                                        "prompt_metadata": {"example_id": f"{bucket}:train:{i}"},
                                        "prompt_tokens": prompt + i % 3, "max_completion_tokens": 8192,
                                        "response": {"completion_tokens": n}}) + "\n")
            summary["models"][model]["datasets"][bucket] = {"output_lengths": lengths}
    (root / "model_dataset_input_output_lengths.json").write_text(json.dumps(summary))
    request_map = root / "request_map.csv"
    with request_map.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["req_id", "bucket", "example_id"]); writer.writeheader()
        for i, (bucket, example) in enumerate(heldout_ids or (("alpaca", "alpaca:train:99999"),)):
            writer.writerow({"req_id": f"req-{i}", "bucket": bucket, "example_id": example})
    return request_map


def _tables(tmp_path):
    build(tmp_path / "cal", _calibration(tmp_path / "cal"), tmp_path / "tables", MODELS)
    return tmp_path / "tables"


def test_build_is_deterministic_and_binned(tmp_path):
    request_map = _calibration(tmp_path / "cal")
    first = build(tmp_path / "cal", request_map, tmp_path / "out1", MODELS)
    second = build(tmp_path / "cal", request_map, tmp_path / "out2", MODELS)
    for model in MODELS:
        assert first["tables"][model]["sha256"] == second["tables"][model]["sha256"]
        assert first["tables"][model]["sha256"] == hashlib.sha256((tmp_path / "out1" / f"{model}.json").read_bytes()).hexdigest()
        table = json.loads((tmp_path / "out1" / f"{model}.json").read_text())
        assert table["model"] == model and table["heldout_overlap"] == 0
        assert len(table["bins"]["0"]) == 2500 and len(table["bins"]["4"]) == 2500 and len(table["bins"]["all"]) == 5000
        assert table["bins"]["all"] == sorted(table["bins"]["all"])
        assert table["request_map_sha256"] == hashlib.sha256(request_map.read_bytes()).hexdigest()


def test_build_rejects_evaluation_overlap(tmp_path):
    request_map = _calibration(tmp_path / "cal", heldout_ids=(("alpaca", "alpaca:train:7"),))
    with pytest.raises(ValueError, match="overlaps the evaluation set"):
        build(tmp_path / "cal", request_map, tmp_path / "out", MODELS)


def test_rule_specs():
    assert parse_remaining_length_rules(["qwen3-0.6b=running_all:0.5:prompt_bin", "qwen3-8b=exhausted_only:0.65:model",
                                         "qwen3-32b=off"]) == {
        "qwen3-0.6b": {"mode": "running_all", "quantile": 0.5, "conditioning": "prompt_bin"},
        "qwen3-8b": {"mode": "exhausted_only", "quantile": 0.65, "conditioning": "model"},
        "qwen3-32b": {"mode": "off", "quantile": 0.5, "conditioning": "prompt_bin"}}
    assert parse_remaining_length_rules(["qwen3-0.6b=running_all"])["qwen3-0.6b"]["quantile"] == 0.5
    for bad in ("qwen3-0.6b=always", "qwen3-0.6b=running_all:2", "qwen3-0.6b=running_all:0.5:bucket", "=off"):
        with pytest.raises(ValueError):
            parse_remaining_length_rules([bad])


def test_pool_flag_off_is_byte_identical_and_per_model_rules_add_server_args(tmp_path):
    from scripts.runs.qwen_baselines import pool_config
    tables = _tables(tmp_path)
    definition = {"models": list(MODELS)}
    rows = pool_config({}, (1, 2, 3), "t")["instances"][:2]
    off = [server_argv("qwen", "/m", row, i, tmp_path, "/len") for i, row in enumerate(rows)]
    assert off == [server_argv("qwen", "/m", row, i, tmp_path, "/len", {"rule": "current", "models": {}}) for i, row in enumerate(rows)]
    assert not any("--remaining-length-mode" in argv for argv in off)
    assert remaining_length_provenance(definition, None)["rule"] == "current"
    assert all(rule["mode"] == "off" and rule["rule"] == "current" and "table" not in rule
               for rule in remaining_length_provenance(definition, None)["models"].values())
    block = remaining_length_provenance(definition, {"tables": tables, "rules": parse_remaining_length_rules(
        ["qwen3-0.6b=running_all:0.5:prompt_bin"])})
    assert block["rule"] == "qwen3-0.6b=running_all_prompt_bin_q50"
    assert block["models"]["qwen3-8b"]["mode"] == "off" and "table" not in block["models"]["qwen3-8b"]
    small = block["models"]["qwen3-0.6b"]
    assert small["rule"] == "running_all_prompt_bin_q50" and len(small["table"]["sha256"]) == 64
    assert small["table"]["support"]["all"] == 5000
    on = [server_argv("qwen", "/m", row, i, tmp_path, "/len", block) for i, row in enumerate(rows)]
    assert on[1] == off[1]                                            # 8B stays byte-identical
    assert off[0][-2:] == on[0][-2:] == ["--host", "127.0.0.1"] and on[0][:len(off[0]) - 2] == off[0][:-2]
    added = on[0][len(off[0]) - 2:-2]
    assert added == ["--remaining-length-mode", "running_all", "--remaining-length-table", small["table"]["path"],
                     "--remaining-length-quantile", "0.5", "--remaining-length-conditioning", "prompt_bin"]
    with pytest.raises(ValueError, match="unknown models"):
        remaining_length_provenance(definition, {"tables": tables, "rules": {"qwen3-32b": {"mode": "off"}}})
    with pytest.raises(ValueError):
        server_argv("ministral", "/m", {"default_model": "x", "model_id": "qwen3-0.6b", "address": "http://h:1",
                    "snapshot_shm_name": "s", "snapshot_shm_size_bytes": 1, "ttft_batch_model": {}}, 0, tmp_path, "/len", block)


def test_router_attaches_tables_only_for_engines_with_a_rule(tmp_path):
    from scripts.runs import experiments as exp
    from types import SimpleNamespace
    tables = _tables(tmp_path)
    block = remaining_length_provenance({"models": list(MODELS)}, {"tables": tables, "rules": parse_remaining_length_rules(
        ["qwen3-0.6b=running_all:0.5:prompt_bin"])})
    instances = {"vllm-0.6b": SimpleNamespace(model_id="qwen3-0.6b"), "vllm-8b": SimpleNamespace(model_id="qwen3-8b")}
    assert exp._load_remaining_length_tables(None, instances) is None
    assert exp._load_remaining_length_tables({"rule": "current"}, instances) is None
    attached = exp._load_remaining_length_tables(block, instances)
    assert set(attached) == {"vllm-0.6b"} and attached["vllm-0.6b"].model == "qwen3-0.6b"
    block["models"]["qwen3-0.6b"]["table"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="changed"):
        exp._load_remaining_length_tables(block, instances)


def test_canonical_control_default_rules():
    from scripts.cloud.canonical_control import PAIRED_RUN_RULES
    assert parse_remaining_length_rules(PAIRED_RUN_RULES) == {
        "qwen3-0.6b": {"mode": "running_all", "quantile": 0.5, "conditioning": "prompt_bin"}}

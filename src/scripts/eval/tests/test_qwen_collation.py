from copy import deepcopy
import json
from pathlib import Path
import sys

from scripts.reporting import collate_qwen_baselines as collator
from scripts.reporting import router_qps_sweep_summary as reporting
from scripts.runs import qwen_baselines as qwen
from scripts.prep.paper_ablation_data import sha256, write_json


def test_qwen_collation_retains_original_curve_and_adds_all_baselines(tmp_path, monkeypatch):
    project = tmp_path/"project"
    sfs = project/"sfs"
    raw, old, output = tmp_path/"raw", tmp_path/"old", tmp_path/"derived"
    (raw/"outputs").mkdir(parents=True); old.mkdir()
    request_map = tmp_path/"request_map.csv"
    request_map.write_text("req_id,bucket,example_id\n"+"".join(f"req-{i},alpaca,example-{i}\n" for i in range(3)))
    scored = project/"vllm_utils/bucketed_prompt_outputs/holdout_4000_scored/qwen3-0.6b/scored"
    scored.mkdir(parents=True)
    (scored/"alpaca_scored.jsonl").write_text("".join(json.dumps({"bucket":"alpaca", "model_label":"qwen3-0.6b",
        "prompt_metadata":{"example_id":f"example-{i}"}, "quality":.8})+"\n" for i in range(3)))
    manifest = {"sfs_root":str(sfs), "qps_values":[6,8.3], "request_map":str(request_map),
                "canonical_reference_cells":{}, "canonical_full_curve_cells":{}}
    original_policies = ("hard","shortest_queue","latency_agnostic","round_robin")
    def point(rate, policies):
        records = [{"request_id":f"req-{i}", "response_id":f"resp-{i}", "bucket":"alpaca",
                    "instance_id":"vllm-0.6b", "system_entry_offset_s":i/rate, "actual_cost":1.,
                    "actual_cost_source":"usage", "system_entry_e2e_ttft_slo_met":True} for i in range(3)]
        return {"config":{"request_rate_qps":rate,"seed":69,"arrival_process":"poisson",
            "lambda_weight":.0005,"delta_weight":0,"routebalance_weights":[1/3]*3,
            "routebalance_batch_max_size":16,"routebalance_batch_wait_ms":25,
            "instance_metadata":{"serving_profile":qwen.PROFILE}, "prompt_source":{"holdout_prompts_per_bucket":4000}},
            "request_set":{"num_requests":3}, "router":{"runs":[{"utility":p,
                "summary":{"succeeded_requests":3,"failed_requests":0,"system_entry_e2e_ttft_missing_count":0,
                    "system_entry_e2e_ttft_slo_attainment_pct":100,
                    "system_entry_e2e_ttft_ms":{"mean":2,"p50":2,"p90":2}},
                "per_request":deepcopy(records)} for p in policies]}}
    for i, rate in enumerate((6,8.3,9.5)):
        path = old/f"old_qps{rate}_point{i:02d}.json"
        write_json(path,point(rate,original_policies))
        for policy in original_policies:
            manifest["canonical_full_curve_cells"][f"{rate:g}:{policy}"] = str(path)
            if rate in manifest["qps_values"]: manifest["canonical_reference_cells"][f"{rate:g}:{policy}"] = str(path)
    for i,rate in enumerate(manifest["qps_values"]):
        write_json(raw/"outputs"/f"new_qps{rate}_point{i:02d}.json",point(rate,qwen.NEW_POLICIES))
    write_json(raw/"outputs/new_qps6_point00_simulation_latency_distribution.json",{"diagnostic":True})
    paths = list(old.glob('*.json'))+list((raw/"outputs").glob('*.json'))
    before = {p:sha256(p) for p in paths}
    manifest_path = tmp_path/"manifest.json"
    write_json(manifest_path,manifest)
    monkeypatch.setattr(qwen,"validate_manifest",lambda _:manifest)
    real_contract = collator.contract
    monkeypatch.setattr(collator,"contract",lambda m:{**real_contract(m),"requests_per_cell":3})
    plotted_qps = []
    real_plot = reporting._plot_utility_vs_qps
    def record_plot(**kwargs):
        plotted_qps.extend(kwargs["qps_keys"])
        return real_plot(**kwargs)
    monkeypatch.setattr(reporting, "_plot_utility_vs_qps", record_plot)
    def report(argv, **_):
        with monkeypatch.context() as context:
            context.setattr(sys, "argv", [argv[2], *argv[3:]])
            reporting.main()  # Exercise real aggregation, CLI selection and all four plots.
    monkeypatch.setattr(collator.subprocess,"run",report)
    result=collator.collate(manifest_path,raw,output)
    assert result["matrix_cells"] == 20  # 12 original cells + 8 added baseline cells.
    summary=json.loads((output/"figures/router_qps_sweep_summary.json").read_text())
    assert set(summary["utilities"]) == set(qwen.POLICIES)
    assert '9.5' in summary["qps"]
    assert '9.5' in plotted_qps
    assert summary["qps"]["9.5"]["score"] is None  # No synthesized baseline point.
    assert {p:sha256(p) for p in paths} == before

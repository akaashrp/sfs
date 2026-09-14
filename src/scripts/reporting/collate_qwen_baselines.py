"""Merge audited 16k baseline cells into derived canonical Qwen policy figures."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys

from scripts.prep.paper_ablation_data import sha256, write_json
from scripts.runs import qwen_baselines as qwen
from scripts.runs.ministral3_figure5 import audit_points, point_paths


def contract(manifest):
    return {"requests_per_cell": 16000, "loads": {"qps_values": manifest["qps_values"]},
        "seed": 69, "serving_profile": qwen.PROFILE, "configuration": {
            "lambda_weight": .0005, "delta_weight": 0, "routebalance_weights": [1/3]*3,
            "routebalance_batch_max_size": 16, "routebalance_batch_wait_ms": 25}}


def collate(manifest_path, raw_root, output_root):
    from scripts.eval.augment_router_actual_accuracy import load_req_map, load_quality_index, augment_file
    manifest = qwen.validate_manifest(manifest_path)
    raw_roots = [Path(r).resolve() for r in (raw_root if isinstance(raw_root,(list,tuple)) else [raw_root])]
    output = Path(output_root).resolve()
    if output.exists(): raise ValueError("Preserve existing derived figures")
    new_paths = [p for raw in raw_roots for p in point_paths(raw/"outputs")]
    if len(set(new_paths)) != len(new_paths): raise ValueError("Duplicate raw roots")
    audit = audit_points(new_paths, contract(manifest), qwen.NEW_POLICIES)
    for source in new_paths:
        for run in json.loads(source.read_text())["router"]["runs"]:
            if run["utility"] == "vllm_sr_latency":
                qwen.audit_run(run, 16000)
    references = manifest.get("canonical_full_curve_cells", manifest["canonical_reference_cells"])
    raw_paths = {*new_paths, *map(Path, references.values())}
    before = {str(p): sha256(p) for p in raw_paths}
    derived = output/"derived"
    derived.mkdir(parents=True)
    mapping = load_req_map(Path(manifest["request_map"]))
    quality = load_quality_index(Path(manifest["sfs_root"]).parent/"vllm_utils/bucketed_prompt_outputs/holdout_4000_scored")
    cells = Counter()
    new_derived = []
    for source in sorted(raw_paths):
        payload = json.loads(source.read_text())
        rate = payload["config"]["request_rate_qps"]
        retained = []
        for run in payload["router"]["runs"]:
            policy = run["utility"]
            if source not in new_paths and references.get(f"{rate:g}:{policy}") != str(source):
                continue
            retained.append(deepcopy(run)); cells[(rate,policy)] += 1
        if not retained: continue
        payload["router"]["runs"] = retained
        path = derived/f"{len(list(derived.glob('*.json'))):02d}_{source.name}"
        write_json(path, payload)
        stats = augment_file(json_path=path, req_maps_by_holdout={4000:mapping}, quality_index=quality, dry_run=False)
        if stats.skipped_reason or stats.missing_req_map or stats.missing_quality or stats.unresolved_model or stats.missing_example_id:
            raise ValueError(f"Incomplete canonical quality join: {path}")
        if source in new_paths: new_derived.append(path)
    expected = {(q,p) for q in manifest["qps_values"] for p in qwen.NEW_POLICIES}
    expected |= {(float(key.split(":",1)[0]), key.split(":",1)[1]) for key in references}
    if set(cells) != expected or set(cells.values()) != {1}:
        raise ValueError("Integrated figure is missing/duplicating new or original policy/load cells")
    audit_points(new_derived, contract(manifest), qwen.NEW_POLICIES, augmented=True)
    subprocess.run([sys.executable, "-m", "scripts.reporting.router_qps_sweep_summary",
        str(derived), "--output-dir", str(output/"figures"), "--retain-all-qps"], check=True)
    summary = json.loads((output/"figures/router_qps_sweep_summary.json").read_text())
    if set(summary["utilities"]) != set(qwen.POLICIES):
        raise ValueError("A required baseline is absent from the integrated figures")
    summary_cells = {(float(rate), policy) for rate, entries in summary["qps"].items()
                     for policy, row in entries.items() if isinstance(row, dict)}
    if summary_cells != expected or any(
        not isinstance(row.get("actual_slo_gated_utility_mean"), (int, float)) or
        not math.isfinite(row["actual_slo_gated_utility_mean"])
        for entries in summary["qps"].values() for row in entries.values() if isinstance(row, dict)
    ):
        raise ValueError("Integrated summary is missing measured OnTimeUtility cells")
    for rate in manifest["qps_values"]:
        for policy in qwen.NEW_POLICIES:
            row = summary["qps"][f"{rate:g}"][policy]
            for metric in ("system_entry_e2e_ttft_ms_slo_attainment_pct", "average_system_entry_e2e_ttft_ms", "p90_system_entry_e2e_ttft_ms"):
                value = row.get(metric)
                if not isinstance(value,(int,float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"Missing measured figure metric: {rate}:{policy}:{metric}")
    plots = ("actual_slo_gated_utility_mean_vs_qps.png", "qps_vs_slo_attainment.png",
             "qps_vs_avg_system_entry_e2e_ttft_ms.png", "qps_vs_p90_system_entry_e2e_ttft_ms.png")
    if any(not (output/"figures"/name).is_file() or not (output/"figures"/name).stat().st_size for name in plots):
        raise ValueError("Missing integrated policy figure")
    if any(sha256(p) != digest for p,digest in before.items()):
        raise ValueError("Canonical/raw results changed during derived collation")
    audit.update(matrix_cells=len(cells), shared_comparison_cells=len(manifest["qps_values"])*len(qwen.POLICIES),
        requests_per_cell=16000, all_policies=list(qwen.POLICIES),
        source_sha256=before, raw_modified=False, manifest_sha256=sha256(manifest_path))
    write_json(output/"audit.json", audit)
    return audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--raw-root", required=True, type=Path, action="append")
    p.add_argument("--output-root", required=True, type=Path)
    a=p.parse_args()
    result=collate(a.manifest,a.raw_root,a.output_root)
    print(json.dumps({k:result[k] for k in ("status","matrix_cells","requests_per_cell")}))


if __name__ == "__main__": main()

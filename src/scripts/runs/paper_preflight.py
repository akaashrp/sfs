"""Rehearse real holdout construction, score joins, and predictor inference on CPU."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

from scripts.prep.paper_ablation_data import BUCKETS, rows, sha256, write_json


def validate_holdout(argv, request_map, scored_root, models, count, per_bucket):
    from scripts.runs.experiments_sweep import _parse_experiment_args
    from scripts.runs import experiments as exp
    from scripts.eval.augment_router_actual_accuracy import load_quality_index, normalize_model_label
    args = _parse_experiment_args(argv)
    requests, _, source = exp._build_request_set(args)
    with Path(request_map).open() as stream:
        mappings = list(csv.DictReader(stream))
    index = {row["req_id"]:row for row in mappings}
    if (len(requests) != count or len(index) != count or len(mappings) != count or
            Counter(row.bucket for row in requests) != Counter({b:per_bucket for b in BUCKETS})):
        raise ValueError("Holdout coverage or request identity mismatch")
    cache = Path(source["effective_bucket_dir"])
    cached = {(b,r["prompt_metadata"]["example_id"]):r for b in BUCKETS for r in rows(cache/f"{b}.jsonl")}
    quality = load_quality_index(Path(scored_root))
    for req in requests:
        mapped = index[req.request_id]
        row = cached.get((mapped["bucket"],mapped["example_id"]))
        if row is None or req.bucket != mapped["bucket"] or req.prompt != row["prompt"] or req.prompt_tokens != int(mapped["prompt_tokens"]):
            raise ValueError("Actual ingestion differs from saved canonical map")
        for model in models:
            value = quality.get((normalize_model_label(model), req.bucket, mapped["example_id"]))
            if not isinstance(value, (int,float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Missing/invalid held-out quality for a candidate")
    from vllm.v1.engine.accuracy_predictor import AccuracyPredictor
    from vllm.v1.engine.output_length_predictor import AdmissionFeatures, OutputLengthPredictor
    sample = [next(r for r in requests if r.bucket == b) for b in BUCKETS]
    admissions = [AdmissionFeatures(m,r.prompt,r.prompt_tokens) for r in sample for m in models]
    accuracies = AccuracyPredictor(str(args.accuracy_model_path)).predict_batch(admissions)
    lengths = OutputLengthPredictor(str(args.output_length_model_path)).predict_batch(admissions)
    if any(not math.isfinite(v) for v in accuracies) or any(v is None or not math.isfinite(v.mean_tokens) or v.mean_tokens <= 0 for v in lengths):
        raise ValueError("Saved SFS predictor inference failed")
    return {"status":"PASS", "requests":count, "candidate_score_joins":count*len(models),
        "actual_request_builder":True, "real_predictor_inferences":len(admissions)*2,
        "request_map_sha256":sha256(request_map), "cache_manifest_sha256":sha256(cache/"manifest.json"),
        "gpu_executed":False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sfs-root",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args(); root=a.sfs_root.resolve()
    from scripts.runs.qwen_baselines import base_argv, MODELS
    source=root.parent/"vllm_utils/bucketed_prompt_outputs"
    qwen=validate_holdout(base_argv(root,root/"experiments/paper_ablation_20260907/qwen_canonical_holdout_4000"),
        source/"req_map_qps_seed69_holdout4000_n16000.csv",source/"holdout_4000_scored",MODELS,16000,4000)
    meta=json.loads((root/"experiments/methodology_baselines/prepared_20260906/metadata.json").read_text())
    cache=root/"experiments/data/prompts/ministral3/holdout_cache_4000"
    argv=[*meta["forwarded_argv"],"--num-requests","16000","--holdout-start-index","2500",
        "--holdout-prompts-per-bucket","4000","--bucket-dir",str(cache),"--holdout-cache-dir",str(cache),
        "--require-existing-holdout-cache"]
    from scripts.prep.prepare_methodology_service import MODELS as MINISTRAL
    ministral=validate_holdout(argv,root/"experiments/ministral3_paper/request_maps/req_map_qps_seed69_holdout4000_n16000.csv",
        root/"experiments/ministral3_paper/holdout_4000_generation/completions",MINISTRAL,16000,4000)
    write_json(a.output,{"status":"PASS","qwen":qwen,"ministral":ministral})
    print(json.dumps({"status":"PASS","output":str(a.output)}))


if __name__ == "__main__":main()

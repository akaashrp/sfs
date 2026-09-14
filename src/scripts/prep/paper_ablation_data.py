"""Strict, network-free preparation of canonical Qwen response ablations.

Canonical identity comes from example_id, never the generation-local row index.
Raw responses/scores are read only; all selected copies are checksummed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

MODELS = ("qwen3-0.6b", "qwen3-8b", "qwen3-32b")
BUCKETS = ("alpaca", "govreport-summarization", "hotpot_qa", "writingprompts")
EVALUATION_REQUESTS = 16000
HOLDOUT_PROMPTS_PER_BUCKET = EVALUATION_REQUESTS // len(BUCKETS)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def identity(row):
    key = row.get("prompt_metadata", {}).get("example_id")
    bucket = row.get("bucket")
    if not isinstance(key, str) or not key or bucket not in BUCKETS:
        raise ValueError("Missing canonical bucket/example_id")
    return bucket, key


def indexed(path, *, model=None):
    result = {}
    for row in rows(path):
        key = identity(row)
        if key in result:
            raise ValueError(f"Duplicate canonical key in {path}: {key}")
        if row.get("error") or (model and row.get("model_label") != model):
            raise ValueError(f"Failed generation or incorrect model in {path}")
        if model:
            q = row.get("quality")
            length = row.get("response", {}).get("completion_tokens")
            if (isinstance(q, bool) or not isinstance(q, (int, float)) or
                    not math.isfinite(q) or not 0 <= q <= 1 or
                    not isinstance(length, int) or not 0 <= length <= 8192):
                raise ValueError(f"Invalid quality or output length in {path}: {key}")
        result[key] = row
    return result


def prepare(source_root, output_root, per_bucket=HOLDOUT_PROMPTS_PER_BUCKET):
    source, output = Path(source_root).resolve(), Path(output_root).resolve()
    if output.exists():
        raise ValueError("Preserve existing preparation; select a new output directory")
    if per_bucket != HOLDOUT_PROMPTS_PER_BUCKET:
        raise ValueError("All paper evaluation workloads require 16000 prompts, 4000 per bucket")
    cache = source/f"holdout_cache_{per_bucket}"
    manifest = json.loads((cache/"manifest.json").read_text())
    if (manifest.get("holdout_start_index"), manifest.get("holdout_prompts_per_bucket"),
            manifest.get("system_prompt"), manifest.get("max_completion_tokens")) != (
            2500, per_bucket, "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.", 8192):
        raise ValueError("Incorrect canonical Qwen holdout profile")
    files, prepared_files, counts = {}, {}, Counter()
    output.mkdir(parents=True)
    for bucket in BUCKETS:
        cache_path = cache/f"{bucket}.jsonl"
        selected = indexed(cache_path)
        if (len(selected) != per_bucket or {r["prompt_index"] for r in selected.values()} !=
                set(range(2500, 2500+per_bucket))):
            raise ValueError(f"Incomplete holdout cache: {bucket}")
        files[str(cache_path)] = sha256(cache_path)
        calibration_reference = None
        for model in MODELS:
            for role, path in (
                ("calibration", source/model/"scored"/f"{bucket}_scored.jsonl"),
                ("holdout", source/"holdout_4000_scored"/model/"scored"/f"{bucket}_scored.jsonl"),
            ):
                records = indexed(path, model=model)
                files[str(path)] = sha256(path)
                if role == "calibration":
                    if len(records) != 2500 or {r["prompt_index"] for r in records.values()} != set(range(2500)):
                        raise ValueError(f"Incomplete calibration: {path}")
                    prompts = {key: row["prompt"] for key, row in records.items()}
                    if calibration_reference is None:
                        calibration_reference = prompts
                    if prompts != calibration_reference or set(records) & set(selected):
                        raise ValueError("Calibration model alignment or holdout separation failed")
                    ordered = sorted(records.values(), key=lambda r: r["prompt_index"])
                else:
                    if not set(selected) <= set(records):
                        raise ValueError(f"Missing scored holdout keys: {path}")
                    ordered = []
                    for key, cached in sorted(selected.items(), key=lambda pair: pair[1]["prompt_index"]):
                        row = records[key]
                        if row["prompt"] != cached["prompt"]:
                            raise ValueError(f"Holdout prompt differs from generation: {key}")
                        ordered.append(row)
                dest = output/role/model/f"{bucket}_scored.jsonl"
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("x") as stream:
                    for row in ordered:
                        stream.write(json.dumps(row, allow_nan=False)+"\n")
                        counts[f"{role}_model_scores"] += 1
                        if row.get("quality_imputed") or row.get("quality_metric") == "judge_default_bucket_mean":
                            counts[f"{role}_imputed"] += 1
                prepared_files[str(dest)] = sha256(dest)
                if role == "holdout":
                    dest = output/"judge_inputs"/model/f"{bucket}.jsonl"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with dest.open("x") as stream:
                        for row in ordered:
                            fresh = {k: v for k, v in row.items() if
                                     not k.startswith("quality") and not k.startswith("judge_")}
                            stream.write(json.dumps(fresh, allow_nan=False)+"\n")
                    prepared_files[str(dest)] = sha256(dest)
    result = {"status": "PASS", "model_family": "qwen3", "models": MODELS,
              "holdout_prompt_groups": EVALUATION_REQUESTS, "calibration_prompt_groups": 10000,
              "counts": dict(counts), "canonical_cache_profile": manifest,
              "source_sha256": files, "prepared_sha256": prepared_files,
              "source_root": str(source), "output_root": str(output), "gpu_executed": False}
    write_json(output/"data_audit.json", result)
    return result


def validate(root):
    audit = json.loads((Path(root)/"data_audit.json").read_text())
    if (audit.get("status") != "PASS" or audit.get("holdout_prompt_groups") != EVALUATION_REQUESTS
            or audit.get("calibration_prompt_groups") != 10000
            or audit.get("models") != list(MODELS)
            or audit.get("counts", {}).get("holdout_model_scores") != EVALUATION_REQUESTS*len(MODELS)
            or audit.get("counts", {}).get("calibration_model_scores") != 10000*len(MODELS)
            or audit.get("canonical_cache_profile", {}).get("holdout_prompts_per_bucket") != HOLDOUT_PROMPTS_PER_BUCKET
            or audit.get("canonical_cache_profile", {}).get("holdout_start_index") != 2500):
        raise ValueError("Invalid prepared data audit")
    for key in ("source_sha256", "prepared_sha256"):
        for path, expected in audit[key].items():
            if sha256(path) != expected:
                raise ValueError(f"Prepared or source data changed: {path}")
    return audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--validate-only", action="store_true")
    a = p.parse_args()
    if not a.validate_only and a.source_root is None:
        p.error("Preparation requires --source-root")
    result = validate(a.output_root) if a.validate_only else prepare(a.source_root, a.output_root)
    print(json.dumps({k: result[k] for k in ("status", "counts", "holdout_prompt_groups")}))


if __name__ == "__main__":
    main()

"""Audit a completed corrected Ministral service run and declare its fit inputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

MODELS = ("ministral3-3b", "ministral3-8b", "ministral3-14b")
PROFILE = {"dtype": "auto (BF16 checkpoints)", "max_model_len": 131072,
           "context_length": 65536, "prompt_token_limit": 32768,
           "max_completion_tokens": 8192, "chunked_prefill": True,
           "max_num_batched_tokens": 32768, "max_num_seqs": 512,
           "prefix_caching": False}
PINS = {"3b": "b6d637bef2393152b3da2b2fde72eecdee30557e",
        "8b": "f6fae9795746f63c9be8344932f01275f3c63734",
        "14b": "3cea74c1ebaf5ce5f5a2553de470e2ceab825142"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_manifest(service_dir):
    root = Path(service_dir).expanduser().resolve()
    provenance = dict(line.split("=", 1) for line in
                      (root / "provenance.txt").read_text().splitlines() if "=" in line)
    required = {"workload": "service_metrics", "gpu_profile": "h100-80",
        "dtype": "auto", "attention_backend": "FLASH_ATTN", "tokenizer_mode": "mistral",
        "config_format": "mistral", "load_format": "mistral", "max_model_len": "131072",
        "context_length": "65536", "chunked_prefill": "1", "prefix_caching": "false",
        "max_num_batched_tokens": "32768", "max_num_seqs": "512",
        "gpu_memory_utilization": "0.90", "max_completion_tokens": "8192",
        "prompt_token_limit": "32768", "temperature": "0.0", "top_p": "1.0",
        "system_prompt": "You are a helpful assistant.", "chat_template_kwargs_json": "{}",
        "calibration_num_requests": "10000", "record_completion_caps_ignored": "true"}
    required.update({f"tensor_parallel_{tier}": "1" for tier in PINS})
    for key, value in required.items():
        if provenance.get(key) != value:
            raise ValueError(f"Service launch provenance mismatch for {key}: {provenance.get(key)!r}")
    for tier, pin in PINS.items():
        expected = f"mistralai/Ministral-3-{tier.upper()}-Instruct-2512-BF16@{pin}"
        if provenance.get(f"model_{tier}") != expected:
            raise ValueError(f"Incorrect model snapshot for {tier}")
    metrics_path = root / "model_metrics.json"
    expected_checksum = Path(str(metrics_path)+".sha256").read_text().split()[0]
    if sha256(metrics_path) != expected_checksum:
        raise ValueError("Service metrics checksum mismatch")
    metrics = json.loads(metrics_path.read_text())
    if set(metrics) != set(MODELS):
        raise ValueError("Service metrics must cover exactly the three candidates")
    models, coverage = {}, {}
    for model in MODELS:
        entry = metrics[model]
        if (entry.get("num_queries"), entry.get("succeeded"), entry.get("failed")) != (10000, 10000, 0):
            raise ValueError(f"Incomplete or failed service run for {model}")
        rate = float(entry["service_rate_qps"])
        elapsed = float(entry["elapsed_s"])
        if not math.isfinite(rate) or rate <= 0 or not math.isclose(rate, 10000/elapsed):
            raise ValueError(f"Invalid measured request throughput for {model}")
        path = root / f"batch_stats_{model}.csv"
        pure, decode, total = 0, 0, 0
        lengths = set()
        with path.open() as stream:
            for row in csv.DictReader(stream):
                total += 1
                d, p, n = float(row["decode"]), float(row["prefill"]), float(row["num_seqs"])
                if d > 0:
                    decode += 1
                if d == 0 and p > 0 and n == 1:
                    pure += 1
                    lengths.add(p)
        coverage[model] = {"trace_rows": total, "pure_singleton_prefill_rows": pure,
            "distinct_pure_prefill_lengths": len(lengths), "decode_active_rows": decode,
            "requires_prefill_probes": pure < 40 or len(lengths) < 3}
        models[model] = {"service_rate_qps": rate,
            "service_rate_definition": "10000 successful calibration requests / whole-run elapsed_s including drain",
            "traces": [str(path)]}
    return {"data_role": "calibration", "serving_profile_verified": True,
        "serving_profile": PROFILE.copy(), "models": models,
        "service_run_provenance": provenance,
        "service_metrics_sha256": expected_checksum,
        "coverage_audit": coverage,
        "source_sha256": {p.name: sha256(p) for p in
            [root/"provenance.txt", root/"gpus.csv", *[root/f"batch_stats_{m}.csv" for m in MODELS]]},
        "router_capacity_measured": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-run-dir", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    manifest = prepare_manifest(args.service_run_dir)
    path = Path(args.output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "PASS", "manifest": str(path), "coverage": manifest["coverage_audit"]}))


if __name__ == "__main__":
    main()

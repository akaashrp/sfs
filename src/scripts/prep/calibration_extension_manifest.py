#!/usr/bin/env python3
"""Verify a calibration-extension family run and write its manifest.json.

The manifest records the source checkout (commit, uncommitted patch), the
pinned vLLM source, the frozen campaign bundle, the pinned model revisions,
the raw bucket index ranges, the generation settings, a verification of
every completion file against the raw bucket rows, and a SHA-256 for every
file under the family output root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BUCKETS = ("alpaca", "govreport-summarization", "hotpot_qa", "writingprompts")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(root: Path, *argv: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *argv], text=True).strip()


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def verify_model(
    completions: Path,
    model: str,
    bucket_rows: dict[str, list[dict[str, Any]]],
    start: int,
    count: int,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    per_bucket: dict[str, Any] = {}
    expected_indices = set(range(start, start + count))
    for bucket in BUCKETS:
        path = completions / model / f"{bucket}.jsonl"
        if not path.is_file():
            errors.append(f"missing {path}")
            continue
        seen: dict[int, str] = {}
        finish = Counter()
        prompt_tokens: list[int] = []
        completion_tokens: list[int] = []
        request_errors = 0
        for record in iter_jsonl(path):
            idx = record.get("prompt_index")
            if record.get("model_label") != model:
                errors.append(f"{path}: model_label {record.get('model_label')!r} != {model!r}")
            if record.get("bucket") != bucket:
                errors.append(f"{path}: bucket {record.get('bucket')!r} != {bucket!r}")
            if not isinstance(idx, int) or idx not in expected_indices:
                errors.append(f"{path}: prompt_index {idx!r} outside [{start}, {start + count})")
                continue
            if idx in seen:
                errors.append(f"{path}: duplicate prompt_index {idx}")
                continue
            example_id = (record.get("prompt_metadata") or {}).get("example_id")
            raw_id = bucket_rows[bucket][idx]["example_id"]
            if example_id != raw_id:
                errors.append(f"{path}: prompt_index {idx} example_id {example_id!r} != raw {raw_id!r}")
            seen[idx] = str(example_id)
            if record.get("error") is not None:
                request_errors += 1
                continue
            response = record.get("response") or {}
            reason = response.get("finish_reason")
            finish[str(reason)] += 1
            if reason not in ("stop", "length"):
                errors.append(f"{path}: prompt_index {idx} finish_reason {reason!r}")
            ct = response.get("completion_tokens")
            if not isinstance(ct, int) or ct < 0 or ct > record.get("max_completion_tokens", 1 << 30):
                errors.append(f"{path}: prompt_index {idx} completion_tokens {ct!r}")
            else:
                completion_tokens.append(ct)
            pt = record.get("prompt_tokens")
            if isinstance(pt, int):
                prompt_tokens.append(pt)
        missing = sorted(expected_indices - set(seen))
        if missing:
            errors.append(f"{path}: {len(missing)} missing indices (first {missing[:5]})")
        if request_errors:
            errors.append(f"{path}: {request_errors} request errors")
        per_bucket[bucket] = {
            "records": len(seen),
            "request_errors": request_errors,
            "index_min": min(seen) if seen else None,
            "index_max": max(seen) if seen else None,
            "finish_reasons": dict(finish),
            "prompt_tokens_mean": round(sum(prompt_tokens) / len(prompt_tokens), 2) if prompt_tokens else None,
            "completion_tokens_mean": round(sum(completion_tokens) / len(completion_tokens), 2)
            if completion_tokens else None,
            "completion_tokens_max": max(completion_tokens) if completion_tokens else None,
        }
    return per_bucket, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=("qwen", "ministral"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--sfs-root", type=Path, required=True)
    parser.add_argument("--bucket-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--served-names", nargs="+", default=None)
    parser.add_argument("--tensor-parallel", nargs="+", type=int, default=None)
    parser.add_argument("--settings-json", type=str, default="{}")
    parser.add_argument("--code-files", nargs="*", type=Path, default=())
    parser.add_argument("--slurm-job-id", type=str, default=os.environ.get("SLURM_JOB_ID"))
    parser.add_argument("--driver-exit-status", type=int, default=None)
    parser.add_argument("--purpose", type=str, default="")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    run_dir = args.run_dir.resolve()
    completions = run_dir / "completions"
    bundle = json.loads(args.bundle.read_text())
    family = bundle["families"][args.family]
    tps = args.tensor_parallel or family["profile"].get("tensor_parallel_sizes") or [1] * len(args.models)
    served = args.served_names or args.models
    if not (len(tps) == len(served) == len(args.models)):
        raise SystemExit("models, served names and tensor-parallel sizes must align")

    bucket_rows = {b: list(iter_jsonl(args.bucket_dir / f"{b}.jsonl")) for b in BUCKETS}
    verification: dict[str, Any] = {}
    errors: list[str] = []
    for model in args.models:
        per_bucket, model_errors = verify_model(completions, model, bucket_rows, args.start_index, args.count)
        verification[model] = per_bucket
        errors.extend(model_errors)
    index_ranges = {
        b: {
            "raw_bucket_file": str(args.bucket_dir / f"{b}.jsonl"),
            "raw_bucket_rows": len(bucket_rows[b]),
            "start_index": args.start_index,
            "end_index_exclusive": args.start_index + args.count,
            "count": args.count,
            "first_example_id": bucket_rows[b][args.start_index]["example_id"],
            "last_example_id": bucket_rows[b][args.start_index + args.count - 1]["example_id"],
            "raw_bucket_sha256": sha256_file(args.bucket_dir / f"{b}.jsonl"),
        }
        for b in BUCKETS
    }

    # Preserve the exact code used (patched pipeline files) beside the outputs.
    code_dir = output_root / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    for src in args.code_files:
        shutil.copy2(src, code_dir / src.name)
    diff_text = git(args.sfs_root, "diff", "HEAD")
    (code_dir / "source_patch.diff").write_text(diff_text + ("\n" if diff_text and not diff_text.endswith("\n") else ""))
    source = {
        "sfs_root": str(args.sfs_root.resolve()),
        "sfs_commit": git(args.sfs_root, "rev-parse", "HEAD"),
        "sfs_status_porcelain": git(args.sfs_root, "status", "--porcelain"),
        "sfs_uncommitted_patch": "code/source_patch.diff",
        "vllm_root": str((args.sfs_root / "vllm").resolve()),
        "vllm_commit": git(args.sfs_root / "vllm", "rev-parse", "HEAD"),
        "vllm_status_porcelain": git(args.sfs_root / "vllm", "status", "--porcelain"),
    }
    try:
        import vllm  # noqa: F401
        source["vllm_python_resolved"] = str(Path(vllm.__file__).resolve())
    except Exception as exc:  # pragma: no cover - informational only
        source["vllm_python_resolved"] = f"unavailable: {exc}"
    try:
        import torch
        source["torch_version"] = torch.__version__
    except Exception as exc:  # pragma: no cover
        source["torch_version"] = f"unavailable: {exc}"

    run_summaries = {}
    for model in args.models:
        path = completions / f"run_summary_{model}.json"
        if path.is_file():
            run_summaries[model] = json.loads(path.read_text())
    provenance_txt = run_dir / "provenance.txt"
    gpus_csv = run_dir / "gpus.csv"
    audit_path = run_dir / "generation_audit.json"

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": args.purpose,
        "family": args.family,
        "slurm_job_id": args.slurm_job_id,
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "driver_exit_status": args.driver_exit_status,
        "source": source,
        "bundle": {
            "path": str(args.bundle.resolve()),
            "sha256": sha256_file(args.bundle),
            "family_profile": family["profile"],
            "family_experiment_argv": family["experiment_argv"],
        },
        "models": {
            model: {
                "repo": bundle["models"][model]["repo"],
                "revision": bundle["models"][model]["revision"],
                "served_model_name": served_name,
                "tensor_parallel_size": tp,
            }
            for model, served_name, tp in zip(args.models, served, tps)
        },
        "index_ranges": index_ranges,
        "settings": json.loads(args.settings_json),
        "client_run_summaries": run_summaries,
        "provenance_txt": provenance_txt.read_text() if provenance_txt.is_file() else None,
        "gpus_csv": gpus_csv.read_text() if gpus_csv.is_file() else None,
        "generation_audit": json.loads(audit_path.read_text()) if audit_path.is_file() else None,
        "verification": {"status": "PASS" if not errors else "FAIL", "errors": errors, "per_model": verification},
        "files": {},
    }
    manifest_path = output_root / "manifest.json"
    files = {}
    for path in sorted(p for p in output_root.rglob("*") if p.is_file() and p != manifest_path):
        files[str(path.relative_to(output_root))] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    manifest["files"] = files
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "verification": manifest["verification"]["status"],
                      "errors": errors[:20], "files": len(files)}, indent=2))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()

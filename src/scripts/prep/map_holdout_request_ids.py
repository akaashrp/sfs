#!/usr/bin/env python3
"""
Build deterministic req-i -> holdout request mapping tables used by router sweeps.

The router sweep path samples holdout prompts with:
1) round-robin mixing across bucket files up to per-bucket limit, then
2) deterministic random shuffle using the run seed.

This script reproduces that exact ordering and outputs a table for:
- qps/qps_no_snapshot presets (holdout_cache_4000, 16000 requests), and
- delta presets (holdout_cache_2000, 8000 requests).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterator, List

from sfs_core.paths import EXPERIMENTS_ROOT, PROMPTS_DATA_ROOT

DEFAULT_OUTPUT_DIR = EXPERIMENTS_ROOT

QPS_CACHE_DIR = PROMPTS_DATA_ROOT / "holdout_cache_4000"
DELTA_CACHE_DIR = PROMPTS_DATA_ROOT / "holdout_cache_2000"


def get_prompt_bucket_files(bucket_dir: Path) -> List[str]:
    files: List[str] = []
    for bucket_file in bucket_dir.iterdir():
        if not bucket_file.is_file():
            continue
        if bucket_file.suffix != ".jsonl":
            continue
        if bucket_file.name == "summary.json":
            continue
        files.append(bucket_file.name)
    return sorted(files)


def iter_bucketed_prompts(path: Path, limit: int) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        for idx, line in enumerate(fp):
            if idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompt = record.get("prompt")
            if isinstance(prompt, str) and prompt.strip():
                yield record


def iter_mixed_bucketed_prompts(
    bucket_dir: Path,
    limit: int,
    *,
    per_bucket_limit: int,
) -> Iterator[Dict[str, Any]]:
    if limit <= 0:
        return

    prompt_bucket_files = get_prompt_bucket_files(bucket_dir)
    if not prompt_bucket_files:
        return

    if per_bucket_limit <= 0:
        raise ValueError("per_bucket_limit must be > 0.")

    iterators = [
        iter_bucketed_prompts(bucket_dir / bucket_file, per_bucket_limit)
        for bucket_file in prompt_bucket_files
    ]

    yielded = 0
    while yielded < limit:
        progressed = False
        for bucket_iter in iterators:
            if yielded >= limit:
                break
            prompt = next(bucket_iter, None)
            if prompt is None:
                continue
            progressed = True
            yielded += 1
            yield prompt
        if not progressed:
            break


def iter_mixed_then_random_bucketed_prompts(
    bucket_dir: Path,
    *,
    per_bucket_limit: int,
    limit: int,
    seed: int,
) -> Iterator[Dict[str, Any]]:
    if per_bucket_limit <= 0:
        raise ValueError("per_bucket_limit must be > 0.")
    if limit <= 0:
        return

    prompt_bucket_files = get_prompt_bucket_files(bucket_dir)
    if not prompt_bucket_files:
        return

    total_pool = per_bucket_limit * len(prompt_bucket_files)
    mixed_records = list(
        iter_mixed_bucketed_prompts(
            bucket_dir,
            total_pool,
            per_bucket_limit=per_bucket_limit,
        )
    )
    rng = random.Random(seed)
    rng.shuffle(mixed_records)
    yield from islice(mixed_records, limit)


def build_mapping_rows(
    *,
    cache_dir: Path,
    per_bucket_limit: int,
    num_requests: int,
    seed: int,
) -> List[Dict[str, Any]]:
    records = list(
        iter_mixed_then_random_bucketed_prompts(
            cache_dir,
            per_bucket_limit=per_bucket_limit,
            limit=num_requests,
            seed=seed,
        )
    )

    rows: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        metadata = record.get("prompt_metadata") if isinstance(record, dict) else None
        if not isinstance(metadata, dict):
            metadata = {}
        rows.append(
            {
                "req_id": f"req-{idx}",
                "req_index": idx,
                "bucket": record.get("bucket"),
                "holdout_request_id": record.get("request_id"),
                "holdout_prompt_index": record.get("prompt_index"),
                "prompt_tokens": record.get("prompt_tokens"),
                "prompt_only_tokens": record.get("prompt_only_tokens"),
                "dataset_id": metadata.get("dataset_id"),
                "example_id": metadata.get("example_id"),
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "req_id",
        "req_index",
        "bucket",
        "holdout_request_id",
        "holdout_prompt_index",
        "prompt_tokens",
        "prompt_only_tokens",
        "dataset_id",
        "example_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_jobs(preset: str) -> List[Dict[str, Any]]:
    if preset == "qps":
        return [
            {
                "name": "qps",
                "cache_dir": QPS_CACHE_DIR,
                "per_bucket_limit": 4000,
                "num_requests": 16000,
            }
        ]
    if preset == "delta":
        return [
            {
                "name": "delta",
                "cache_dir": DELTA_CACHE_DIR,
                "per_bucket_limit": 2000,
                "num_requests": 8000,
            }
        ]
    if preset == "all":
        return [
            {
                "name": "qps",
                "cache_dir": QPS_CACHE_DIR,
                "per_bucket_limit": 4000,
                "num_requests": 16000,
            },
            {
                "name": "delta",
                "cache_dir": DELTA_CACHE_DIR,
                "per_bucket_limit": 2000,
                "num_requests": 8000,
            },
        ]
    raise ValueError(f"Unsupported preset: {preset}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic req-i -> holdout request-id mapping tables "
            "for router sweep runs."
        )
    )
    parser.add_argument(
        "--preset",
        choices=("qps", "delta", "all"),
        default="all",
        help=(
            "qps: holdout_cache_4000 mapping (QPS and QPS no snapshot runs), "
            "delta: holdout_cache_2000 mapping (delta runs), "
            "all: both."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=69,
        help="Deterministic shuffle seed (must match sweep run seed).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated CSV mapping tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args.preset)
    manifest: Dict[str, Any] = {"seed": args.seed, "jobs": []}

    for job in jobs:
        cache_dir = Path(job["cache_dir"]).expanduser().resolve()
        if not cache_dir.exists():
            raise FileNotFoundError(f"Holdout cache directory does not exist: {cache_dir}")

        rows = build_mapping_rows(
            cache_dir=cache_dir,
            per_bucket_limit=int(job["per_bucket_limit"]),
            num_requests=int(job["num_requests"]),
            seed=int(args.seed),
        )
        output_path = (
            output_dir
            / (
                f"req_map_{job['name']}_"
                f"seed{int(args.seed)}_"
                f"holdout{int(job['per_bucket_limit'])}_"
                f"n{int(job['num_requests'])}.csv"
            )
        )
        write_csv(output_path, rows)
        manifest["jobs"].append(
            {
                "name": job["name"],
                "cache_dir": str(cache_dir),
                "per_bucket_limit": int(job["per_bucket_limit"]),
                "num_requests": int(job["num_requests"]),
                "rows_written": len(rows),
                "output_csv": str(output_path),
            }
        )
        print(f"[DONE] {job['name']}: wrote {len(rows)} rows -> {output_path}")

    manifest_path = output_dir / f"req_map_manifest_seed{int(args.seed)}_{args.preset}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()

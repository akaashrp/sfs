#!/usr/bin/env python3
"""Augment router sweep per-request records with judged accuracy scores.

This script joins three data sources:
1) Router-side prompt-ingestion maps (req_id -> holdout prompt/example),
2) Router run JSON files (contains per_request routed model), and
3) Scored holdout outputs (contains judge quality per model/prompt).

It writes `actual_accuracy` into each `router.runs[*].per_request[*]` record.

Safety behavior:
- Before writing a JSON, create a temporary backup copy next to the file.
- Write updated content to a temporary write file.
- Atomically replace the original.
- Delete temporary backup only after successful persistence.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sfs_core.paths import BUCKETED_OUTPUTS_ROOT
from sfs_core.shared.model_label_helpers import normalize_model_label, resolve_model_label


DEFAULT_BUCKETED_PROMPT_OUTPUTS = BUCKETED_OUTPUTS_ROOT
DEFAULT_DELTA_REQ_MAP = (
    DEFAULT_BUCKETED_PROMPT_OUTPUTS / "req_map_delta_seed69_holdout2000_n8000.csv"
)
DEFAULT_QPS_REQ_MAP = (
    DEFAULT_BUCKETED_PROMPT_OUTPUTS / "req_map_qps_seed69_holdout4000_n16000.csv"
)
DEFAULT_SCORED_ROOT = DEFAULT_BUCKETED_PROMPT_OUTPUTS / "holdout_4000_scored"

HOLDOUT_CACHE_RE = re.compile(r"holdout_cache_(\d+)")


@dataclass(frozen=True)
class ReqMapEntry:
    req_id: str
    bucket: str
    example_id: str


@dataclass
class FileStats:
    path: Path
    holdout_prompts_per_bucket: Optional[int] = None
    per_request_total: int = 0
    updated: int = 0
    unchanged: int = 0
    missing_req_map: int = 0
    missing_example_id: int = 0
    unresolved_model: int = 0
    missing_quality: int = 0
    skipped_reason: Optional[str] = None


def load_req_map(path: Path) -> Dict[str, ReqMapEntry]:
    if not path.exists():
        raise FileNotFoundError(f"Request mapping CSV not found: {path}")

    mapping: Dict[str, ReqMapEntry] = {}
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            req_id = (row.get("req_id") or "").strip()
            bucket = (row.get("bucket") or "").strip()
            example_id = (row.get("example_id") or "").strip()
            if not req_id:
                continue
            mapping[req_id] = ReqMapEntry(
                req_id=req_id,
                bucket=bucket,
                example_id=example_id,
            )
    return mapping


def load_quality_index(
    scored_root: Path,
) -> Dict[Tuple[str, str, str], float]:
    if not scored_root.exists():
        raise FileNotFoundError(f"Scored root directory not found: {scored_root}")

    index: Dict[Tuple[str, str, str], float] = {}

    model_dirs = [p for p in sorted(scored_root.iterdir()) if p.is_dir()]
    for model_dir in model_dirs:
        default_model_label = normalize_model_label(model_dir.name)
        scored_dir = model_dir / "scored"
        if not scored_dir.is_dir():
            continue

        scored_files = sorted(scored_dir.glob("*_scored.jsonl"))
        for scored_file in scored_files:
            default_bucket = scored_file.stem.replace("_scored", "")
            with scored_file.open("r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)

                    model_label = normalize_model_label(
                        str(record.get("model_label") or default_model_label)
                    )
                    bucket = str(record.get("bucket") or default_bucket)
                    prompt_metadata = record.get("prompt_metadata")
                    if not isinstance(prompt_metadata, dict):
                        continue
                    example_id = prompt_metadata.get("example_id")
                    if not isinstance(example_id, str) or not example_id:
                        continue

                    quality = record.get("quality")
                    if not isinstance(quality, (int, float)):
                        continue

                    index[(model_label, bucket, example_id)] = float(quality)

    if not index:
        raise RuntimeError(
            "Quality index is empty. Check --scored-root and scored JSONL contents."
        )

    return index


def _extract_holdout_prompts_per_bucket(payload: Dict[str, Any]) -> Optional[int]:
    config = payload.get("config")
    if not isinstance(config, dict):
        return None

    prompt_source = config.get("prompt_source")
    if isinstance(prompt_source, dict):
        holdout_value = prompt_source.get("holdout_prompts_per_bucket")
        if isinstance(holdout_value, int):
            return holdout_value
        if isinstance(holdout_value, str) and holdout_value.isdigit():
            return int(holdout_value)

        holdout_cache_dir = prompt_source.get("holdout_cache_dir")
        if isinstance(holdout_cache_dir, str):
            match = HOLDOUT_CACHE_RE.search(holdout_cache_dir)
            if match:
                return int(match.group(1))

    return None


def _safe_write_json(path: Path, payload: Dict[str, Any]) -> None:
    temp_backup = path.with_name(f"{path.name}.tmp_backup")

    fd, temp_write_str = tempfile.mkstemp(
        prefix=f"{path.name}.",
        suffix=".tmp_write",
        dir=str(path.parent),
    )
    os.close(fd)
    temp_write = Path(temp_write_str)

    shutil.copy2(path, temp_backup)

    try:
        with temp_write.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)
            fp.write("\n")

        os.replace(temp_write, path)
        temp_backup.unlink(missing_ok=True)
    except Exception:
        temp_write.unlink(missing_ok=True)
        # Keep backup on failure for manual recovery.
        raise


def _iter_run_jsons(run_dir: Path, include_manifests: bool) -> Iterable[Path]:
    if not run_dir.is_dir():
        return

    for json_path in sorted(run_dir.glob("*.json")):
        if not include_manifests and json_path.name.endswith("_manifest.json"):
            continue
        yield json_path


def augment_file(
    *,
    json_path: Path,
    req_maps_by_holdout: Dict[int, Dict[str, ReqMapEntry]],
    quality_index: Dict[Tuple[str, str, str], float],
    dry_run: bool,
) -> FileStats:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    stats = FileStats(path=json_path)

    holdout_value = _extract_holdout_prompts_per_bucket(payload)
    stats.holdout_prompts_per_bucket = holdout_value
    if holdout_value is None:
        stats.skipped_reason = "could_not_resolve_holdout_prompts_per_bucket"
        return stats

    req_map = req_maps_by_holdout.get(holdout_value)
    if req_map is None:
        stats.skipped_reason = f"no_req_map_for_holdout_{holdout_value}"
        return stats

    router = payload.get("router")
    if not isinstance(router, dict):
        stats.skipped_reason = "missing_router_section"
        return stats

    runs = router.get("runs")
    if not isinstance(runs, list):
        stats.skipped_reason = "missing_router_runs"
        return stats

    any_changes = False

    for run in runs:
        if not isinstance(run, dict):
            continue

        per_request = run.get("per_request")
        if not isinstance(per_request, list):
            continue

        for record in per_request:
            if not isinstance(record, dict):
                continue

            stats.per_request_total += 1
            request_id = record.get("request_id")
            if not isinstance(request_id, str):
                stats.missing_req_map += 1
                continue

            req_entry = req_map.get(request_id)
            if req_entry is None:
                stats.missing_req_map += 1
                continue

            if not req_entry.example_id:
                stats.missing_example_id += 1
                continue

            model_label = resolve_model_label(
                response_model=record.get("response_model"),
                instance_id=record.get("instance_id"),
            )
            if model_label is None:
                stats.unresolved_model += 1
                continue

            request_bucket = record.get("bucket")
            if isinstance(request_bucket, str) and request_bucket:
                bucket_for_lookup = request_bucket
            else:
                bucket_for_lookup = req_entry.bucket

            quality = quality_index.get(
                (model_label, bucket_for_lookup, req_entry.example_id)
            )
            if quality is None:
                # Fall back to mapping bucket if record bucket was inconsistent.
                quality = quality_index.get(
                    (model_label, req_entry.bucket, req_entry.example_id)
                )

            if quality is None:
                stats.missing_quality += 1
                continue

            previous = record.get("actual_accuracy")
            if previous == quality:
                stats.unchanged += 1
                continue

            record["actual_accuracy"] = quality
            stats.updated += 1
            any_changes = True

    if any_changes and not dry_run:
        _safe_write_json(json_path, payload)

    if stats.per_request_total == 0:
        stats.skipped_reason = "no_per_request_records"

    return stats


def _read_run_dirs(run_dirs: List[Path], run_dir_files: List[Path]) -> List[Path]:
    resolved = [p.expanduser().resolve() for p in run_dirs]

    for list_path in run_dir_files:
        with list_path.expanduser().resolve().open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                resolved.append(Path(line).expanduser().resolve())

    # Preserve order while de-duplicating.
    seen: set[Path] = set()
    unique: List[Path] = []
    for path in resolved:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)

    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Augment router per-request records with actual_accuracy from scored holdout outputs."
        )
    )
    parser.add_argument(
        "run_dirs",
        nargs="*",
        type=Path,
        help="Router sweep run directories containing router *.json files.",
    )
    parser.add_argument(
        "--run-dir-file",
        action="append",
        default=[],
        type=Path,
        help="Optional text file with one run-directory path per line (supports comments with #).",
    )
    parser.add_argument(
        "--delta-req-map",
        type=Path,
        default=DEFAULT_DELTA_REQ_MAP,
        help=(
            "CSV req-map for holdout 2000 runs (delta sweeps). "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--qps-req-map",
        type=Path,
        default=DEFAULT_QPS_REQ_MAP,
        help=(
            "CSV req-map for holdout 4000 runs (qps sweeps). "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--scored-root",
        type=Path,
        default=DEFAULT_SCORED_ROOT,
        help=(
            "Root directory with per-model scored holdout files. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--include-manifests",
        action="store_true",
        help="Also inspect *_manifest.json files (normally skipped).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and report updates without writing JSON files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dirs = _read_run_dirs(args.run_dirs, args.run_dir_file)
    if not run_dirs:
        raise SystemExit("No run directories were provided.")

    delta_req_map = load_req_map(args.delta_req_map.expanduser().resolve())
    qps_req_map = load_req_map(args.qps_req_map.expanduser().resolve())
    quality_index = load_quality_index(args.scored_root.expanduser().resolve())

    req_maps_by_holdout = {
        2000: delta_req_map,
        4000: qps_req_map,
    }

    print(
        "[LOAD] req-maps: "
        f"holdout2000={len(delta_req_map)} rows, "
        f"holdout4000={len(qps_req_map)} rows"
    )
    print(f"[LOAD] quality-index: {len(quality_index)} (model, bucket, example_id) entries")

    file_stats: List[FileStats] = []
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            print(f"[WARN] run dir not found, skipping: {run_dir}")
            continue

        json_files = list(_iter_run_jsons(run_dir, include_manifests=args.include_manifests))
        if not json_files:
            print(f"[WARN] no JSON files found in: {run_dir}")
            continue

        print(f"[SCAN] {run_dir} ({len(json_files)} json files)")

        for json_path in json_files:
            stats = augment_file(
                json_path=json_path,
                req_maps_by_holdout=req_maps_by_holdout,
                quality_index=quality_index,
                dry_run=args.dry_run,
            )
            file_stats.append(stats)

            if stats.skipped_reason is not None:
                print(f"  [SKIP] {json_path.name}: {stats.skipped_reason}")
                continue

            action = "DRYRUN" if args.dry_run else "WRITE"
            print(
                f"  [{action}] {json_path.name}: "
                f"total={stats.per_request_total}, "
                f"updated={stats.updated}, "
                f"unchanged={stats.unchanged}, "
                f"missing_map={stats.missing_req_map}, "
                f"missing_example={stats.missing_example_id}, "
                f"unresolved_model={stats.unresolved_model}, "
                f"missing_quality={stats.missing_quality}"
            )

    seen = len(file_stats)
    skipped = sum(1 for s in file_stats if s.skipped_reason is not None)
    processed = seen - skipped
    updated_records = sum(s.updated for s in file_stats)
    total_records = sum(s.per_request_total for s in file_stats)
    missing_map = sum(s.missing_req_map for s in file_stats)
    missing_example = sum(s.missing_example_id for s in file_stats)
    unresolved_model = sum(s.unresolved_model for s in file_stats)
    missing_quality = sum(s.missing_quality for s in file_stats)

    print("[DONE]")
    print(f"  files_seen={seen}")
    print(f"  files_processed={processed}")
    print(f"  files_skipped={skipped}")
    print(f"  per_request_total={total_records}")
    print(f"  records_updated={updated_records}")
    print(f"  missing_req_map={missing_map}")
    print(f"  missing_example_id={missing_example}")
    print(f"  unresolved_model={unresolved_model}")
    print(f"  missing_quality={missing_quality}")
    print(f"  dry_run={bool(args.dry_run)}")


if __name__ == "__main__":
    main()

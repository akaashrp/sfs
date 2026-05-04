#!/usr/bin/env python3
"""Deduplicate holdout scored JSONL files by keeping last occurrence per key.

Key definition and traversal/normalization semantics intentionally mirror
`augment_router_actual_accuracy.py` so surviving qualities match the values used
when augmenting router experiment JSONs.

Workflow:
1) Pass 1: scan in augment order and record last occurrence for each key.
2) Pass 2: rewrite scored files, preserving only winning keyed lines.
   - Non-keyed lines are preserved verbatim.
   - Keyed winner lines are preserved verbatim.
   - Keyed non-winner lines are removed.
3) Validation (optional, enabled by default): compare scored qualities against
   `actual_accuracy` in a router run JSON without re-running augmentation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sfs_core.paths import BUCKETED_OUTPUTS_ROOT, EXPERIMENTS_ROOT
from sfs_core.shared.model_label_helpers import normalize_model_label, resolve_model_label

DEFAULT_BUCKETED_PROMPT_OUTPUTS = BUCKETED_OUTPUTS_ROOT
DEFAULT_SCORED_ROOT = DEFAULT_BUCKETED_PROMPT_OUTPUTS / "holdout_4000_scored"
DEFAULT_DELTA_REQ_MAP = (
    DEFAULT_BUCKETED_PROMPT_OUTPUTS / "req_map_delta_seed69_holdout2000_n8000.csv"
)
DEFAULT_VALIDATE_ROUTER_JSON = (
    EXPERIMENTS_ROOT
    / "router_delta_sweep_38645567_2026-04-08_213911"
    / "router_delta_qwen3_lambda5em4_20260408_214117_delta1em07_point01.json"
)


@dataclass(frozen=True)
class KeyOccurrence:
    file_path: Path
    line_number: int
    quality: float


@dataclass(frozen=True)
class ReqMapEntry:
    req_id: str
    bucket: str
    example_id: str


@dataclass
class DedupStats:
    files_scanned: int = 0
    files_changed: int = 0
    files_unchanged: int = 0
    keyed_lines_seen: int = 0
    lines_removed: int = 0
    unique_keys_retained: int = 0
    duplicate_keys: int = 0
    conflicting_duplicate_keys: int = 0


@dataclass
class ValidationStats:
    router_json: Path
    per_request_total: int = 0
    per_request_with_actual_accuracy: int = 0
    missing_req_map: int = 0
    unresolved_model: int = 0
    missing_quality_lookup: int = 0
    mismatches: int = 0
    mismatch_examples: List[str] = None

    def __post_init__(self) -> None:
        if self.mismatch_examples is None:
            self.mismatch_examples = []


def iter_scored_files(scored_root: Path) -> Iterable[Tuple[Path, str, str]]:
    model_dirs = [p for p in sorted(scored_root.iterdir()) if p.is_dir()]
    for model_dir in model_dirs:
        default_model_label = normalize_model_label(model_dir.name)
        scored_dir = model_dir / "scored"
        if not scored_dir.is_dir():
            continue
        for scored_file in sorted(scored_dir.glob("*_scored.jsonl")):
            default_bucket = scored_file.stem.replace("_scored", "")
            yield scored_file, default_model_label, default_bucket


def key_from_line(
    line: str,
    *,
    default_model_label: str,
    default_bucket: str,
    file_path: Path,
    line_number: int,
) -> Tuple[Optional[Tuple[str, str, str]], Optional[float]]:
    stripped = line.strip()
    if not stripped:
        return None, None

    try:
        record = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {file_path}:{line_number}: {exc.msg}"
        ) from exc

    prompt_metadata = record.get("prompt_metadata")
    if not isinstance(prompt_metadata, dict):
        return None, None

    example_id = prompt_metadata.get("example_id")
    if not isinstance(example_id, str) or not example_id:
        return None, None

    quality = record.get("quality")
    if not isinstance(quality, (int, float)):
        return None, None

    model_label = normalize_model_label(str(record.get("model_label") or default_model_label))
    bucket = str(record.get("bucket") or default_bucket)

    return (model_label, bucket, example_id), float(quality)


def safe_write_lines(path: Path, lines: List[str]) -> None:
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
            fp.writelines(lines)
        os.replace(temp_write, path)
        temp_backup.unlink(missing_ok=True)
    except Exception:
        temp_write.unlink(missing_ok=True)
        raise


def dedup_scored_files(scored_root: Path, *, dry_run: bool) -> Tuple[DedupStats, Dict[Tuple[str, str, str], KeyOccurrence]]:
    stats = DedupStats()

    key_counts: Dict[Tuple[str, str, str], int] = {}
    key_quality_values: Dict[Tuple[str, str, str], set[float]] = {}
    winners: Dict[Tuple[str, str, str], KeyOccurrence] = {}

    scored_file_specs = list(iter_scored_files(scored_root))
    stats.files_scanned = len(scored_file_specs)

    # Pass 1: determine last occurrence winner for each key.
    for scored_file, default_model_label, default_bucket in scored_file_specs:
        with scored_file.open("r", encoding="utf-8") as fp:
            for line_number, line in enumerate(fp, start=1):
                key, quality = key_from_line(
                    line,
                    default_model_label=default_model_label,
                    default_bucket=default_bucket,
                    file_path=scored_file,
                    line_number=line_number,
                )
                if key is None or quality is None:
                    continue

                stats.keyed_lines_seen += 1
                key_counts[key] = key_counts.get(key, 0) + 1
                key_quality_values.setdefault(key, set()).add(quality)
                winners[key] = KeyOccurrence(
                    file_path=scored_file,
                    line_number=line_number,
                    quality=quality,
                )

    stats.unique_keys_retained = len(winners)
    stats.duplicate_keys = sum(1 for c in key_counts.values() if c > 1)
    stats.conflicting_duplicate_keys = sum(
        1
        for key, count in key_counts.items()
        if count > 1 and len(key_quality_values.get(key, set())) > 1
    )

    # Pass 2: rewrite each file keeping only winner keyed lines.
    for scored_file, default_model_label, default_bucket in scored_file_specs:
        with scored_file.open("r", encoding="utf-8") as fp:
            original_lines = fp.readlines()

        kept_lines: List[str] = []
        removed_in_file = 0

        for line_number, line in enumerate(original_lines, start=1):
            key, _quality = key_from_line(
                line,
                default_model_label=default_model_label,
                default_bucket=default_bucket,
                file_path=scored_file,
                line_number=line_number,
            )

            if key is None:
                kept_lines.append(line)
                continue

            winner = winners.get(key)
            if winner is not None and winner.file_path == scored_file and winner.line_number == line_number:
                kept_lines.append(line)
            else:
                removed_in_file += 1

        if removed_in_file == 0:
            stats.files_unchanged += 1
            continue

        stats.files_changed += 1
        stats.lines_removed += removed_in_file

        if not dry_run:
            safe_write_lines(scored_file, kept_lines)

    return stats, winners


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
            mapping[req_id] = ReqMapEntry(req_id=req_id, bucket=bucket, example_id=example_id)
    return mapping


def load_quality_index(scored_root: Path) -> Dict[Tuple[str, str, str], float]:
    index: Dict[Tuple[str, str, str], float] = {}
    for scored_file, default_model_label, default_bucket in iter_scored_files(scored_root):
        with scored_file.open("r", encoding="utf-8") as fp:
            for line_number, line in enumerate(fp, start=1):
                key, quality = key_from_line(
                    line,
                    default_model_label=default_model_label,
                    default_bucket=default_bucket,
                    file_path=scored_file,
                    line_number=line_number,
                )
                if key is None or quality is None:
                    continue
                index[key] = quality

    if not index:
        raise RuntimeError("Quality index is empty after dedup/load.")

    return index


def validate_against_router_json(
    *,
    router_json_path: Path,
    quality_index: Dict[Tuple[str, str, str], float],
    req_map: Dict[str, ReqMapEntry],
    epsilon: float,
    max_examples: int,
) -> ValidationStats:
    payload = json.loads(router_json_path.read_text(encoding="utf-8"))
    stats = ValidationStats(router_json=router_json_path)

    router = payload.get("router")
    if not isinstance(router, dict):
        raise ValueError(f"Router JSON missing 'router' section: {router_json_path}")

    runs = router.get("runs")
    if not isinstance(runs, list):
        raise ValueError(f"Router JSON missing 'router.runs': {router_json_path}")

    for run in runs:
        if not isinstance(run, dict):
            continue
        per_request = run.get("per_request")
        if not isinstance(per_request, list):
            continue

        for rec in per_request:
            if not isinstance(rec, dict):
                continue
            stats.per_request_total += 1

            actual_accuracy = rec.get("actual_accuracy")
            if not isinstance(actual_accuracy, (int, float)):
                continue
            stats.per_request_with_actual_accuracy += 1

            request_id = rec.get("request_id")
            if not isinstance(request_id, str):
                stats.missing_req_map += 1
                continue

            map_entry = req_map.get(request_id)
            if map_entry is None:
                stats.missing_req_map += 1
                continue

            model_label = resolve_model_label(
                response_model=rec.get("response_model"),
                instance_id=rec.get("instance_id"),
            )
            if model_label is None:
                stats.unresolved_model += 1
                continue

            request_bucket = rec.get("bucket")
            if isinstance(request_bucket, str) and request_bucket:
                bucket_for_lookup = request_bucket
            else:
                bucket_for_lookup = map_entry.bucket

            quality = quality_index.get((model_label, bucket_for_lookup, map_entry.example_id))
            if quality is None:
                quality = quality_index.get((model_label, map_entry.bucket, map_entry.example_id))

            if quality is None:
                stats.missing_quality_lookup += 1
                if len(stats.mismatch_examples) < max_examples:
                    stats.mismatch_examples.append(
                        f"missing quality: req={request_id} model={model_label} "
                        f"bucket={bucket_for_lookup} example_id={map_entry.example_id}"
                    )
                continue

            if abs(float(quality) - float(actual_accuracy)) > epsilon:
                stats.mismatches += 1
                if len(stats.mismatch_examples) < max_examples:
                    stats.mismatch_examples.append(
                        f"mismatch: req={request_id} model={model_label} bucket={bucket_for_lookup} "
                        f"example_id={map_entry.example_id} actual_accuracy={actual_accuracy} expected={quality}"
                    )

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deduplicate holdout scored JSONLs by keeping last occurrence per augment-order key, "
            "then validate against a router JSON's actual_accuracy."
        )
    )
    parser.add_argument(
        "--scored-root",
        type=Path,
        default=DEFAULT_SCORED_ROOT,
        help="Root directory containing holdout_4000_scored model folders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report dedup changes without rewriting files.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip router actual_accuracy consistency validation step.",
    )
    parser.add_argument(
        "--validate-router-json",
        type=Path,
        default=DEFAULT_VALIDATE_ROUTER_JSON,
        help="Router experiment JSON path used for post-dedup validation.",
    )
    parser.add_argument(
        "--delta-req-map",
        type=Path,
        default=DEFAULT_DELTA_REQ_MAP,
        help="Delta req-map CSV used for request_id -> example_id mapping.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-12,
        help="Absolute tolerance for quality vs actual_accuracy comparison.",
    )
    parser.add_argument(
        "--max-mismatch-examples",
        type=int,
        default=20,
        help="Max mismatch/missing examples to print.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    scored_root = args.scored_root.expanduser().resolve()
    if not scored_root.exists():
        raise FileNotFoundError(f"Scored root does not exist: {scored_root}")

    print(f"[START] scored_root={scored_root}")
    dedup_stats, _winners = dedup_scored_files(scored_root, dry_run=bool(args.dry_run))

    print("[DEDUP]")
    print(f"  files_scanned={dedup_stats.files_scanned}")
    print(f"  files_changed={dedup_stats.files_changed}")
    print(f"  files_unchanged={dedup_stats.files_unchanged}")
    print(f"  keyed_lines_seen={dedup_stats.keyed_lines_seen}")
    print(f"  lines_removed={dedup_stats.lines_removed}")
    print(f"  unique_keys_retained={dedup_stats.unique_keys_retained}")
    print(f"  duplicate_keys={dedup_stats.duplicate_keys}")
    print(f"  conflicting_duplicate_keys={dedup_stats.conflicting_duplicate_keys}")
    print(f"  dry_run={bool(args.dry_run)}")

    if args.skip_validation:
        print("[VALIDATION] skipped")
        return 0

    router_json_path = args.validate_router_json.expanduser().resolve()
    req_map_path = args.delta_req_map.expanduser().resolve()

    quality_index = load_quality_index(scored_root)
    req_map = load_req_map(req_map_path)

    validation = validate_against_router_json(
        router_json_path=router_json_path,
        quality_index=quality_index,
        req_map=req_map,
        epsilon=float(args.epsilon),
        max_examples=int(args.max_mismatch_examples),
    )

    print("[VALIDATION]")
    print(f"  router_json={validation.router_json}")
    print(f"  per_request_total={validation.per_request_total}")
    print(f"  per_request_with_actual_accuracy={validation.per_request_with_actual_accuracy}")
    print(f"  missing_req_map={validation.missing_req_map}")
    print(f"  unresolved_model={validation.unresolved_model}")
    print(f"  missing_quality_lookup={validation.missing_quality_lookup}")
    print(f"  mismatches={validation.mismatches}")
    print(f"  epsilon={args.epsilon}")

    if validation.mismatch_examples:
        print("  examples:")
        for example in validation.mismatch_examples:
            print(f"    - {example}")

    ok = (
        validation.missing_req_map == 0
        and validation.unresolved_model == 0
        and validation.missing_quality_lookup == 0
        and validation.mismatches == 0
    )

    if not ok:
        print("[FAIL] validation criteria not met")
        return 2

    print("[PASS] validation criteria met")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

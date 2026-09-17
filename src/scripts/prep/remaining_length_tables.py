"""Build prompt-length-conditioned survival tables from the calibration outputs only.

One JSON table per model (sorted total output lengths per prompt-length bin plus a
model-level back-off bin) for the flag-gated exhausted-only remaining-decode target
(`--remaining-length-table` on the vLLM server; `scripts.cloud.pool` plumbs it). Bins,
support minimum and quantile semantics are those of
scripts/cloud/reports/reserve-tail-20260916/evaluate_tail.py.

Inputs are verified exactly as reserve-alternatives-20260916/build_calibration.py does:
no errored record, matching model label, prompt indices 0-2499 per bucket, lengths equal
to the summary arrays, and zero (bucket, example_id) overlap with the evaluation request map.

Example:
  python -m scripts.prep.remaining_length_tables \
    --calibration /path/to/bucketed_prompt_outputs \
    --request-map /path/to/bundle/qwen/request_map.csv \
    --output /path/to/tables
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from vllm.v1.core.sched.remaining_length import (
    DEFAULT_MIN_SUPPORT,
    MODEL_LEVEL_KEY,
    PROMPT_BIN_EDGES,
    RemainingLengthTable,
    prompt_bin,
)

DEFAULT_MODELS = ("qwen3-0.6b", "qwen3-8b", "qwen3-32b")
DEFAULT_CAP = 8192


def load_heldout(request_map: Path) -> set[tuple[str, str]]:
    with request_map.open() as stream:
        return {(row["bucket"], row["example_id"]) for row in csv.DictReader(stream)}


def load_model_outputs(calibration: Path, model: str, heldout: set[tuple[str, str]],
                       prompt_field: str = "prompt_tokens") -> tuple[list[dict], list[dict]]:
    """Verified (prompt_tokens, completion_tokens) rows and per-source provenance."""
    summary = json.loads((calibration / "model_dataset_input_output_lengths.json").read_text())
    rows, sources = [], []
    for bucket, expected in summary["models"][model]["datasets"].items():
        path = calibration / model / "outputs" / f"{bucket}.jsonl"
        lengths, indices = [], set()
        with path.open() as stream:
            for line in stream:
                record = json.loads(line)
                if record.get("error"):
                    raise ValueError(f"errored calibration record in {path}")
                if record["model_label"] != model:
                    raise ValueError(f"model label mismatch in {path}")
                key = (bucket, str(record["prompt_metadata"]["example_id"]))
                if key in heldout:
                    raise ValueError(f"calibration record overlaps the evaluation set: {key}")
                indices.add(record["prompt_index"])
                completion = int(record["response"]["completion_tokens"])
                if not 0 <= completion <= int(record["max_completion_tokens"]):
                    raise ValueError(f"completion length outside the cap in {path}")
                lengths.append(completion)
                rows.append({"bucket": bucket, "prompt_tokens": int(record[prompt_field]),
                             "completion_tokens": completion})
        if indices != set(range(2500)) or lengths != expected["output_lengths"]:
            raise ValueError(f"calibration indices or lengths differ from the summary for {path}")
        sources.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "records": len(lengths)})
    return rows, sources


def build_table(model: str, rows: list[dict], *, min_support: int = DEFAULT_MIN_SUPPORT,
                cap: int = DEFAULT_CAP, sources: list[dict] | None = None,
                request_map_sha256: str | None = None) -> RemainingLengthTable:
    bins: dict[str, list[int]] = {str(b): [] for b in range(len(PROMPT_BIN_EDGES) - 1)}
    bins[MODEL_LEVEL_KEY] = []
    for row in rows:
        length = min(int(row["completion_tokens"]), int(cap))
        bins[str(prompt_bin(row["prompt_tokens"]))].append(length)
        bins[MODEL_LEVEL_KEY].append(length)
    bins = {key: sorted(values) for key, values in bins.items()}
    metadata = {
        "rule": "prompt_survival_quantile",
        "fitted_from": "calibration outputs only (predictor training prompts); disjoint from evaluation ids",
        "records": len(rows),
        "records_per_bucket": {b: sum(r["bucket"] == b for r in rows) for b in sorted({r["bucket"] for r in rows})},
        "prompt_field": "prompt_tokens",
        "heldout_overlap": 0,
        "request_map_sha256": request_map_sha256,
        "sources": sources or [],
    }
    return RemainingLengthTable(model=model, bins=bins, prompt_bin_edges=PROMPT_BIN_EDGES,
                                min_support=min_support, cap=cap, metadata=metadata)


def write_table(table: RemainingLengthTable, path: Path) -> str:
    payload = json.dumps(table.to_payload(), sort_keys=True, separators=(",", ":")) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)
    return hashlib.sha256(payload.encode()).hexdigest()


def build(calibration: Path, request_map: Path, output: Path, models=DEFAULT_MODELS,
          *, min_support: int = DEFAULT_MIN_SUPPORT, cap: int = DEFAULT_CAP) -> dict:
    heldout = load_heldout(request_map)
    request_map_sha256 = hashlib.sha256(request_map.read_bytes()).hexdigest()
    manifest = {"schema_version": 1, "request_map": str(request_map), "request_map_sha256": request_map_sha256,
                "evaluation_ids": len(heldout), "calibration": str(calibration), "min_support": min_support,
                "cap": cap, "prompt_bin_edges": list(PROMPT_BIN_EDGES), "tables": {}}
    for model in models:
        rows, sources = load_model_outputs(calibration, model, heldout)
        table = build_table(model, rows, min_support=min_support, cap=cap, sources=sources,
                            request_map_sha256=request_map_sha256)
        path = output / f"{model}.json"
        sha256 = write_table(table, path)
        manifest["tables"][model] = {"path": str(path), "sha256": sha256, "records": len(rows),
                                     "support_per_bin": {k: table.support(k) for k in table.to_payload()["bins"]}}
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=1) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration", type=Path, required=True, help="bucketed_prompt_outputs root")
    parser.add_argument("--request-map", type=Path, required=True, help="evaluation request map CSV (bucket, example_id)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--min-support", type=int, default=DEFAULT_MIN_SUPPORT)
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP)
    args = parser.parse_args()
    manifest = build(args.calibration.resolve(), args.request_map.resolve(), args.output.resolve(),
                     [m for m in args.models.split(",") if m], min_support=args.min_support, cap=args.cap)
    for model, entry in manifest["tables"].items():
        print(model, entry["records"], entry["sha256"], entry["support_per_bin"])


if __name__ == "__main__":
    main()

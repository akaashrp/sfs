import ast
import csv
import re
from pathlib import Path
from typing import Any


REQUEST_ID_PATTERN = re.compile(r"request_id=([^\s]+)")
PREFILL_S_PATTERN = re.compile(r"prefill_s=([0-9.+-eE]+)")


def snapshot_trace_file_offsets(
    *,
    response_map_path: Path,
    actual_wait_log_path: Path,
    batch_stats_csv_path: Path | None = None,
) -> dict[str, int]:
    offsets = {
        "response_map_offset": response_map_path.expanduser().stat().st_size
        if response_map_path.expanduser().exists()
        else 0,
        "actual_wait_log_offset": actual_wait_log_path.expanduser().stat().st_size
        if actual_wait_log_path.expanduser().exists()
        else 0,
    }
    if batch_stats_csv_path is not None:
        batch_path = batch_stats_csv_path.expanduser()
        offsets["batch_stats_offset"] = batch_path.stat().st_size if batch_path.exists() else 0
    return offsets


def _read_new_lines(path: Path, start_offset: int) -> list[str]:
    with path.expanduser().open("r", encoding="utf-8", errors="ignore") as src:
        if start_offset > 0:
            src.seek(start_offset)
        return [line.rstrip("\n") for line in src if line.strip()]


def estimate_prefill_theta_from_trace(
    *,
    request_prompt_tokens: dict[str, int],
    response_map_path: Path,
    actual_wait_log_path: Path,
    response_map_offset: int = 0,
    actual_wait_log_offset: int = 0,
    batch_stats_csv_path: Path | None = None,
    batch_stats_offset: int = 0,
) -> dict[str, Any]:
    response_map_lines = _read_new_lines(response_map_path, response_map_offset)
    request_to_response: dict[str, str] = {}
    parsed_response_map_rows = 0
    for line in response_map_lines:
        try:
            record = ast.literal_eval(line)
        except Exception:
            continue
        if not isinstance(record, dict):
            continue
        request_id = record["request_id"]
        response_id = record["response_id"]
        request_to_response[str(request_id)] = str(response_id)
        parsed_response_map_rows += 1

    wait_log_lines = _read_new_lines(actual_wait_log_path, actual_wait_log_offset)
    response_prefill_s: dict[str, float] = {}
    for line in wait_log_lines:
        request_match = REQUEST_ID_PATTERN.search(line)
        prefill_match = PREFILL_S_PATTERN.search(line)
        if request_match is None or prefill_match is None:
            continue
        response_id = request_match.group(1)
        prefill_s = float(prefill_match.group(1))
        response_prefill_s[response_id] = prefill_s

    matched: list[dict[str, Any]] = []
    total_tokens = 0.0
    total_prefill_s = 0.0
    seen_response_ids: set[str] = set()

    for request_id, prompt_tokens in request_prompt_tokens.items():
        # request_prompt_tokens may be keyed by scheduler request-id, user request-id,
        # or directly by response-id. Try response-map lookup first, then direct id.
        response_id = request_to_response.get(request_id, request_id)
        if not isinstance(response_id, str) or not response_id:
            continue
        prefill_s = response_prefill_s.get(response_id)
        if prefill_s is None:
            continue
        if response_id in seen_response_ids:
            continue
        seen_response_ids.add(response_id)
        matched.append(
            {
                "request_id": request_id,
                "response_id": response_id,
                "prompt_tokens": int(prompt_tokens),
                "prefill_s": float(prefill_s),
                "prefill_tps": float(prompt_tokens) / float(prefill_s),
            }
        )
        total_tokens += float(prompt_tokens)
        total_prefill_s += float(prefill_s)

    if not matched:
        raise RuntimeError(
            "No matched request-response-prefill samples found. "
            f"request_prompt_tokens={len(request_prompt_tokens)} "
            f"response_map_rows={parsed_response_map_rows}/{len(response_map_lines)} "
            f"wait_log_rows={len(wait_log_lines)} "
            f"wait_prefill_samples={len(response_prefill_s)}"
        )

    result: dict[str, Any] = {
        "theta_p_tps_from_wait_logs": total_tokens / total_prefill_s,
        "matched_pairs": len(matched),
        "samples": matched,
    }

    if batch_stats_csv_path is not None:
        batch_lines = _read_new_lines(batch_stats_csv_path, batch_stats_offset)
        batch_prefill = 0.0
        batch_exec = 0.0
        parsed_rows = 0
        for line in batch_lines:
            if line.startswith("ts,engine,prefill"):
                continue
            row = next(csv.reader([line]))
            prefill_tokens = float(row[2])
            exec_s = float(row[8])
            if prefill_tokens > 0 and exec_s > 0:
                batch_prefill += prefill_tokens
                batch_exec += exec_s
                parsed_rows += 1
        if parsed_rows == 0:
            raise RuntimeError("No usable batch-stats rows found after batch_stats_offset.")
        result["theta_p_tps_from_batch_stats"] = batch_prefill / batch_exec
        result["batch_stats_rows_used"] = parsed_rows

    return result

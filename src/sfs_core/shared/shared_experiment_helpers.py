from __future__ import annotations

import asyncio
from itertools import islice
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Iterator

DEFAULT_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
DEFAULT_RANDOM_SEED = 69


def as_nonnegative_int(
    value: Any,
    *,
    allow_strings: bool = True,
    require_integral: bool = False,
) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        if not math.isfinite(value) or value < 0:
            return None
        if require_integral and not value.is_integer():
            return None
        return int(value)
    if allow_strings and isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if require_integral:
            try:
                parsed = int(stripped)
            except ValueError:
                return None
            return parsed if parsed >= 0 else None
        try:
            numeric = float(stripped)
        except ValueError:
            return None
        if not math.isfinite(numeric) or numeric < 0:
            return None
        return int(numeric)
    return None


def _compute_percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    if q <= 0:
        return float(min(values))
    if q >= 1:
        return float(max(values))

    sorted_values = sorted(values)
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])

    weight = pos - lo
    return float(sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight)


def build_messages(prompt: str, system_prompt: str | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def iter_bucketed_prompts(
    path: str | Path,
    limit: int,
    include_complete_record: bool = False,
) -> Iterator[str | dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fp:
        for idx, line in enumerate(fp):
            if idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompt = record.get("prompt")
            if isinstance(prompt, str) and prompt.strip():
                if include_complete_record:
                    yield record
                else:
                    yield prompt


def get_prompt_bucket_files(prompt_bucket_dir: str | os.PathLike[str] | Path) -> list[str]:
    prompt_bucket_files: list[str] = []
    for bucket_file in os.listdir(prompt_bucket_dir):
        if not bucket_file.endswith(".jsonl"):
            continue
        if bucket_file == "summary.json":
            continue
        prompt_bucket_files.append(bucket_file)
    return sorted(prompt_bucket_files)


def iter_mixed_bucketed_prompts(
    bucket_dir: Path,
    limit: int,
    include_complete_record: bool = False,
    *,
    per_bucket_limit: int | None = None,
) -> Iterator[str | dict[str, Any]]:
    """Yield prompts round-robin across every bucket file."""
    if limit <= 0:
        return
    prompt_bucket_files = get_prompt_bucket_files(bucket_dir)
    if not prompt_bucket_files:
        return

    per_file_limit = per_bucket_limit if per_bucket_limit is not None else limit
    if per_file_limit <= 0:
        raise ValueError("per_bucket_limit must be > 0 when provided.")

    iterators = [
        iter_bucketed_prompts(
            str(bucket_dir / bucket_file),
            per_file_limit,
            include_complete_record,
        )
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


def iter_random_bucketed_prompts(
    bucket_dir: Path,
    limit: int,
    *,
    seed: int = DEFAULT_RANDOM_SEED,
    include_complete_record: bool = False,
) -> Iterator[str | dict[str, Any]]:
    """Yield prompts in a deterministic random order across all bucket files."""
    prompts: list[str | dict[str, Any]] = []
    prompt_bucket_files = get_prompt_bucket_files(bucket_dir)
    for bucket_file in prompt_bucket_files:
        bucket_path = bucket_dir / bucket_file
        if not bucket_path.exists():
            continue
        prompts.extend(iter_bucketed_prompts(str(bucket_path), limit, include_complete_record))

    rng = random.Random(seed)
    rng.shuffle(prompts)
    yield from islice(prompts, limit)


def iter_mixed_then_random_bucketed_prompts(
    bucket_dir: Path,
    *,
    per_bucket_limit: int,
    limit: int,
    seed: int = DEFAULT_RANDOM_SEED,
    include_complete_record: bool = False,
) -> Iterator[str | dict[str, Any]]:
    """Yield records by round-robin bucket mixing followed by deterministic shuffle."""
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
            include_complete_record=include_complete_record,
            per_bucket_limit=per_bucket_limit,
        )
    )
    rng = random.Random(seed)
    rng.shuffle(mixed_records)
    yield from islice(mixed_records, limit)


async def warm_up_instances(clients: list[Any]) -> None:
    """Send a warm-up request to each instance to seed wait-time snapshots."""
    messages = [{"role": "user", "content": "warm-up", "temperature": 0}]
    tasks = []
    for client in clients:
        tasks.append(
            asyncio.create_task(
                client.submit_request(messages=messages, max_completion_tokens=1)
            )
        )
    await asyncio.gather(*tasks, return_exceptions=True)


def select_prompt_subset(
    prompts: Iterator[str | dict[str, Any]],
    num_queries: int,
) -> list[str | dict[str, Any]]:
    """Materialize only the first ``num_queries`` prompts from an iterator."""
    if num_queries <= 0:
        return []
    return list(islice(prompts, num_queries))


def _metric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": float(sum(values) / len(values)),
        "p10": _compute_percentile(values, 0.10),
        "p50": _compute_percentile(values, 0.50),
        "p90": _compute_percentile(values, 0.90),
        "min": float(min(values)),
        "max": float(max(values)),
    }

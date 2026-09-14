from __future__ import annotations

import asyncio
from collections.abc import Mapping
from itertools import islice
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Iterator

DEFAULT_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
DEFAULT_CHAT_TEMPLATE_KWARGS: dict[str, Any] = {"enable_thinking": False}
DEFAULT_RANDOM_SEED = 69

_RESERVED_CHAT_TEMPLATE_KWARGS = frozenset(
    {
        "add_generation_prompt",
        "tokenize",
    }
)


def resolve_chat_template_kwargs(
    chat_template_kwargs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return validated template kwargs, preserving Qwen's default mode."""
    if chat_template_kwargs is None:
        return dict(DEFAULT_CHAT_TEMPLATE_KWARGS)
    if not isinstance(chat_template_kwargs, Mapping):
        raise ValueError("chat template kwargs must be a JSON object")
    if not all(isinstance(key, str) for key in chat_template_kwargs):
        raise ValueError("chat template kwarg names must be strings")

    reserved = sorted(_RESERVED_CHAT_TEMPLATE_KWARGS.intersection(chat_template_kwargs))
    if reserved:
        raise ValueError(
            "chat template kwargs cannot override runner-controlled argument(s): "
            + ", ".join(reserved)
        )
    return dict(chat_template_kwargs)


def parse_chat_template_kwargs_json(value: str) -> dict[str, Any]:
    """Parse the CLI JSON representation of tokenizer chat-template kwargs."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid chat template kwargs JSON: {exc.msg}") from exc
    return resolve_chat_template_kwargs(parsed)


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


def resolve_prompt_bucket_dir(
    path: str | os.PathLike[str] | Path,
    *,
    label: str = "Prompt-bucket",
) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {resolved}")
    if not get_prompt_bucket_files(resolved):
        raise ValueError(
            f"No JSONL files found in {label.lower()} directory: {resolved}"
        )
    return resolved


def resolve_max_completion_tokens(
    requested_max_completion_tokens: int,
    prompt_tokens: int,
    *,
    context_length: int | None = None,
    record_max_completion_tokens: Any = None,
) -> int:
    if requested_max_completion_tokens <= 0:
        raise ValueError("requested_max_completion_tokens must be > 0")
    if prompt_tokens < 0:
        raise ValueError("prompt_tokens must be >= 0")

    resolved = int(requested_max_completion_tokens)
    record_limit = as_nonnegative_int(
        record_max_completion_tokens,
        require_integral=True,
    )
    if record_limit is not None and record_limit > 0:
        resolved = min(resolved, record_limit)

    if context_length is not None:
        if context_length <= 0:
            raise ValueError("context_length must be > 0 when provided")
        remaining_context = int(context_length) - int(prompt_tokens)
        if remaining_context <= 0:
            raise ValueError(
                f"Prompt has {prompt_tokens} tokens but context length is "
                f"{context_length}"
            )
        resolved = min(resolved, remaining_context)

    return max(1, resolved)


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
    messages = [{"role": "user", "content": "warm-up"}]
    tasks = []
    for client in clients:
        tasks.append(
            asyncio.create_task(
                client.submit_request(messages=messages, temperature=0, top_p=1,
                                      max_completion_tokens=1)
            )
        )
    results = await asyncio.gather(*tasks, return_exceptions=True)
    failures = [(getattr(client, "instance_id", str(index)), str(result))
                for index, (client, result) in enumerate(zip(clients, results))
                if isinstance(result, BaseException)]
    if failures:
        raise RuntimeError(f"Instance warm-up failed; snapshots are not ready: {failures}")


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

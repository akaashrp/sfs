from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sfs_core.shared.shared_experiment_helpers import (
    DEFAULT_CHAT_TEMPLATE_KWARGS,
    DEFAULT_SYSTEM_PROMPT,
    as_nonnegative_int,
    build_messages,
    resolve_chat_template_kwargs,
)
from sfs_core.shared.tokenizer_helpers import (
    load_tokenizer as load_configured_tokenizer,
    normalize_tokenizer_mode,
)

MANIFEST_FILENAME = "manifest.json"
PROMPT_TOKENIZATION_VERSION = 2


def _as_nonnegative_int(value: Any) -> int | None:
    return as_nonnegative_int(value, require_integral=True)


def _list_bucket_files(source_bucket_dir: Path) -> list[str]:
    if not source_bucket_dir.exists():
        raise FileNotFoundError(
            f"Holdout source directory does not exist: {source_bucket_dir}"
        )
    if not source_bucket_dir.is_dir():
        raise ValueError(f"Holdout source path is not a directory: {source_bucket_dir}")

    bucket_files = sorted(
        p.name
        for p in source_bucket_dir.iterdir()
        if p.is_file() and p.suffix == ".jsonl" and p.name != "summary.json"
    )
    if not bucket_files:
        raise ValueError(
            f"No bucket JSONL files found in holdout source: {source_bucket_dir}"
        )
    return bucket_files


def _count_jsonl_rows(path: Path) -> int:
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows += 1
    return rows


def _load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _build_expected_manifest_config(
    *,
    source_bucket_dir: Path,
    tokenizer_id: str,
    tokenizer_mode: str,
    holdout_start_index: int,
    holdout_prompts_per_bucket: int,
    holdout_context_length: int,
    max_completion_tokens: int,
    prompt_token_limit: int,
    system_prompt: str,
    chat_template_kwargs: dict[str, Any],
    bucket_files: list[str],
) -> dict[str, Any]:
    return {
        "prompt_tokenization_version": PROMPT_TOKENIZATION_VERSION,
        "source_bucket_dir": str(source_bucket_dir.resolve()),
        "tokenizer_id": str(tokenizer_id),
        "tokenizer_mode": str(tokenizer_mode),
        "holdout_start_index": int(holdout_start_index),
        "holdout_prompts_per_bucket": int(holdout_prompts_per_bucket),
        "holdout_context_length": int(holdout_context_length),
        "max_completion_tokens": int(max_completion_tokens),
        "prompt_token_limit": int(prompt_token_limit),
        "system_prompt": str(system_prompt),
        "chat_template_kwargs": dict(chat_template_kwargs),
        "bucket_files": list(bucket_files),
    }


def _is_cache_reusable(
    *,
    manifest: dict[str, Any] | None,
    expected_manifest_config: dict[str, Any],
    cache_dir: Path,
    holdout_prompts_per_bucket: int,
    bucket_files: list[str],
) -> bool:
    if manifest is None:
        return False

    for key, expected_value in expected_manifest_config.items():
        actual_value = manifest.get(key)
        if key == "chat_template_kwargs" and key not in manifest:
            # Version-2 caches predate the explicit field but were always
            # tokenized with Qwen's non-thinking template policy.
            actual_value = DEFAULT_CHAT_TEMPLATE_KWARGS
        elif key == "tokenizer_mode" and key not in manifest:
            # Version-2 caches used Hugging Face AutoTokenizer exclusively.
            actual_value = "auto"
        if actual_value != expected_value:
            return False

    bucket_counts = manifest.get("bucket_counts")
    if not isinstance(bucket_counts, dict):
        return False

    cache_jsonl_files = sorted(
        p.name for p in cache_dir.iterdir() if p.is_file() and p.suffix == ".jsonl"
    )
    if cache_jsonl_files != sorted(bucket_files):
        return False

    for bucket_file in bucket_files:
        if (
            _as_nonnegative_int(bucket_counts.get(bucket_file))
            != holdout_prompts_per_bucket
        ):
            return False
        output_path = cache_dir / bucket_file
        if not output_path.exists():
            return False
        if _count_jsonl_rows(output_path) != holdout_prompts_per_bucket:
            return False

    return True


def _load_tokenizer(tokenizer_id: str, *, tokenizer_mode: str = "auto"):
    try:
        return load_configured_tokenizer(
            tokenizer_id,
            tokenizer_mode=tokenizer_mode,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load tokenizer '{tokenizer_id}' for holdout cache conversion."
        ) from exc


def _compute_prompt_tokens(
    prompt: str,
    tokenizer: Any,
    system_prompt: str,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> int:
    messages = build_messages(prompt=prompt, system_prompt=system_prompt)
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        **resolve_chat_template_kwargs(chat_template_kwargs),
    )
    return int(len(token_ids))


def _truncate_prompt_tokens(
    prompt: str,
    tokenizer: Any,
    max_tokens: int | None,
) -> tuple[str, int]:
    if not prompt:
        return "", 0
    if max_tokens is None or max_tokens <= 0:
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        return prompt, int(len(token_ids))

    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return prompt, int(len(token_ids))

    truncated_ids = token_ids[:max_tokens]
    truncated_text = tokenizer.decode(truncated_ids)
    return truncated_text, int(len(truncated_ids))


def _slice_and_convert_bucket(
    *,
    source_path: Path,
    holdout_start_index: int,
    holdout_prompts_per_bucket: int,
    holdout_context_length: int,
    max_completion_tokens: int,
    prompt_token_limit: int,
    tokenizer: Any,
    system_prompt: str,
    chat_template_kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    slice_end = holdout_start_index + holdout_prompts_per_bucket
    converted_records: list[dict[str, Any]] = []

    with source_path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if line_index < holdout_start_index:
                continue
            if line_index >= slice_end:
                break

            line = line.strip()
            if not line:
                continue

            try:
                raw_record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON in holdout source {source_path} line "
                    f"{line_index + 1}: {exc}"
                ) from exc

            raw_prompt = raw_record.get("prompt")
            if not isinstance(raw_prompt, str) or not raw_prompt.strip():
                raise ValueError(
                    f"Missing/empty prompt in holdout source {source_path} line "
                    f"{line_index + 1}."
                )
            prompt, prompt_only_tokens = _truncate_prompt_tokens(
                raw_prompt,
                tokenizer,
                prompt_token_limit,
            )

            prompt_index = line_index
            bucket_name = raw_record.get("bucket")
            if not isinstance(bucket_name, str) or not bucket_name:
                bucket_name = source_path.stem

            prompt_metadata = {
                k: v
                for k, v in raw_record.items()
                if k
                not in {
                    "prompt",
                    "response",
                    "quality",
                    "quality_metric",
                    "timings",
                    "model_label",
                    "model_id",
                    "error",
                }
            }

            prompt_tokens = _compute_prompt_tokens(
                prompt,
                tokenizer,
                system_prompt,
                chat_template_kwargs,
            )

            remaining_context = int(holdout_context_length) - int(prompt_tokens)
            record_max_completion_tokens = max(
                1,
                min(int(max_completion_tokens), remaining_context),
            )

            converted_records.append(
                {
                    "bucket": bucket_name,
                    "prompt_index": int(prompt_index),
                    "request_id": f"holdout-{source_path.stem}-{prompt_index:05d}",
                    "prompt": prompt,
                    "prompt_metadata": prompt_metadata,
                    "prompt_tokens": int(prompt_tokens),
                    "prompt_only_tokens": int(prompt_only_tokens),
                    "max_completion_tokens": int(record_max_completion_tokens),
                }
            )

    if len(converted_records) != holdout_prompts_per_bucket:
        raise ValueError(
            "Insufficient records for holdout slice in "
            f"{source_path}: requested {holdout_prompts_per_bucket} records from "
            f"start index {holdout_start_index}, found {len(converted_records)}."
        )

    converted_records.sort(key=lambda record: str(record["request_id"]))
    return converted_records


def prepare_holdout_prompt_cache(
    *,
    source_bucket_dir: Path,
    cache_dir: Path,
    tokenizer_id: str,
    tokenizer_mode: str = "auto",
    holdout_start_index: int,
    holdout_prompts_per_bucket: int,
    holdout_context_length: int,
    max_completion_tokens: int,
    prompt_token_limit: int = 32768,
    rebuild: bool = False,
    require_existing: bool = False,
    frozen_legacy: bool = False,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    chat_template_kwargs: dict[str, Any] | None = None,
    tokenizer: Any | None = None,
) -> tuple[Path, dict[str, Any], bool]:
    if holdout_prompts_per_bucket <= 0:
        raise ValueError("holdout_prompts_per_bucket must be > 0.")
    if holdout_start_index < 0:
        raise ValueError("holdout_start_index must be >= 0.")
    if holdout_context_length <= 0:
        raise ValueError("holdout_context_length must be > 0.")
    if max_completion_tokens <= 0:
        raise ValueError("max_completion_tokens must be > 0.")
    if prompt_token_limit <= 0:
        raise ValueError("prompt_token_limit must be > 0.")

    source_bucket_dir = source_bucket_dir.expanduser().resolve()
    cache_dir = cache_dir.expanduser().resolve()
    if require_existing and (rebuild or not cache_dir.is_dir()):
        raise ValueError("Existing-only holdout cache mode forbids creation or rebuilding")
    cache_dir.mkdir(parents=True, exist_ok=True)
    resolved_chat_template_kwargs = resolve_chat_template_kwargs(
        chat_template_kwargs
    )
    resolved_tokenizer_mode = normalize_tokenizer_mode(tokenizer_mode)

    bucket_files = _list_bucket_files(source_bucket_dir)
    expected_manifest_config = _build_expected_manifest_config(
        source_bucket_dir=source_bucket_dir,
        tokenizer_id=tokenizer_id,
        tokenizer_mode=resolved_tokenizer_mode,
        holdout_start_index=holdout_start_index,
        holdout_prompts_per_bucket=holdout_prompts_per_bucket,
        holdout_context_length=holdout_context_length,
        max_completion_tokens=max_completion_tokens,
        prompt_token_limit=prompt_token_limit,
        system_prompt=system_prompt,
        chat_template_kwargs=resolved_chat_template_kwargs,
        bucket_files=bucket_files,
    )

    manifest_path = cache_dir / MANIFEST_FILENAME
    existing_manifest = _load_manifest(manifest_path)
    if frozen_legacy:
        if not require_existing or rebuild:
            raise ValueError("Frozen legacy caches require existing-only mode and forbid rebuilding")
        # A derived paper cache can intentionally retain the original routing
        # token counts. Validate its separate contract rather than relabeling
        # those counts as the current tokenizer version.
        legacy = dict(existing_manifest or {})
        if legacy.get("data_role") != "frozen_canonical_paper_holdout" or not legacy.get("file_sha256"):
            raise ValueError("Missing frozen canonical paper cache provenance")
        from hashlib import sha256
        for name, expected in legacy["file_sha256"].items():
            if Path(name).name != name or sha256((cache_dir/name).read_bytes()).hexdigest() != expected:
                raise ValueError("Frozen canonical cache contents changed")
        comparable = {k:v for k,v in expected_manifest_config.items() if k != "prompt_tokenization_version"}
        if not _is_cache_reusable(manifest=legacy, expected_manifest_config=comparable, cache_dir=cache_dir,
                holdout_prompts_per_bucket=holdout_prompts_per_bucket, bucket_files=bucket_files):
            raise ValueError("Frozen canonical cache does not match the requested experiment")
        return cache_dir, existing_manifest, False
    if not rebuild and _is_cache_reusable(
        manifest=existing_manifest,
        expected_manifest_config=expected_manifest_config,
        cache_dir=cache_dir,
        holdout_prompts_per_bucket=holdout_prompts_per_bucket,
        bucket_files=bucket_files,
    ):
        return cache_dir, existing_manifest, False

    if require_existing:
        raise ValueError("Existing holdout cache is incompatible; refusing to rebuild or modify it")

    if tokenizer is None:
        tokenizer = _load_tokenizer(
            tokenizer_id,
            tokenizer_mode=resolved_tokenizer_mode,
        )

    for existing_file in cache_dir.iterdir():
        if (
            existing_file.is_file()
            and existing_file.suffix == ".jsonl"
            and existing_file.name not in bucket_files
        ):
            existing_file.unlink()

    bucket_counts: dict[str, int] = {}
    for bucket_file in bucket_files:
        source_path = source_bucket_dir / bucket_file
        converted_records = _slice_and_convert_bucket(
            source_path=source_path,
            holdout_start_index=holdout_start_index,
            holdout_prompts_per_bucket=holdout_prompts_per_bucket,
            holdout_context_length=holdout_context_length,
            max_completion_tokens=max_completion_tokens,
            prompt_token_limit=prompt_token_limit,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            chat_template_kwargs=resolved_chat_template_kwargs,
        )

        output_path = cache_dir / bucket_file
        with output_path.open("w", encoding="utf-8") as handle:
            for record in converted_records:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")

        bucket_counts[bucket_file] = len(converted_records)

    manifest = dict(expected_manifest_config)
    manifest["bucket_counts"] = bucket_counts
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return cache_dir, manifest, True


__all__ = ["MANIFEST_FILENAME", "prepare_holdout_prompt_cache"]

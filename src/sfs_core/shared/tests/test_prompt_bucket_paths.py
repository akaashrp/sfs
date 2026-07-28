from __future__ import annotations

from pathlib import Path

import pytest

from sfs_core.shared.shared_experiment_helpers import (
    resolve_max_completion_tokens,
    resolve_prompt_bucket_dir,
)


def test_resolve_prompt_bucket_dir_requires_existing_directory(tmp_path: Path):
    missing = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_prompt_bucket_dir(missing, label="Calibration prompt-bucket")


def test_resolve_prompt_bucket_dir_requires_jsonl_file(tmp_path: Path):
    with pytest.raises(ValueError, match="No JSONL files"):
        resolve_prompt_bucket_dir(tmp_path, label="Calibration prompt-bucket")


def test_resolve_prompt_bucket_dir_accepts_jsonl_directory(tmp_path: Path):
    (tmp_path / "alpaca_scored.jsonl").write_text("{}\n", encoding="utf-8")

    assert (
        resolve_prompt_bucket_dir(tmp_path, label="Calibration prompt-bucket")
        == tmp_path.resolve()
    )


def test_resolve_max_completion_tokens_applies_context_limit():
    assert (
        resolve_max_completion_tokens(
            8192,
            32801,
            context_length=40960,
            record_max_completion_tokens=8192,
        )
        == 8159
    )


def test_resolve_max_completion_tokens_preserves_smaller_record_limit():
    assert (
        resolve_max_completion_tokens(
            8192,
            100,
            context_length=40960,
            record_max_completion_tokens=512,
        )
        == 512
    )

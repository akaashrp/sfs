from pathlib import Path

from sfs_core.prep.holdout_cache import (
    PROMPT_TOKENIZATION_VERSION,
    _build_expected_manifest_config,
    _compute_prompt_tokens,
    _is_cache_reusable,
)


class _RecordingTokenizer:
    def __init__(self) -> None:
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return [1, 2, 3]


def test_prompt_tokens_disable_thinking_with_direct_template_kwarg():
    tokenizer = _RecordingTokenizer()

    assert _compute_prompt_tokens("hello", tokenizer, "system") == 3
    assert tokenizer.kwargs is not None
    assert tokenizer.kwargs["enable_thinking"] is False
    assert "chat_template_kwargs" not in tokenizer.kwargs


def test_prompt_tokens_allow_family_with_no_template_kwargs():
    tokenizer = _RecordingTokenizer()

    assert _compute_prompt_tokens("hello", tokenizer, "system", {}) == 3
    assert tokenizer.kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
    }


def test_manifest_versions_prompt_tokenization_semantics():
    config = _build_expected_manifest_config(
        source_bucket_dir=Path("/source"),
        tokenizer_id="tokenizer",
        tokenizer_mode="auto",
        holdout_start_index=1,
        holdout_prompts_per_bucket=2,
        holdout_context_length=3,
        max_completion_tokens=4,
        prompt_token_limit=5,
        system_prompt="system",
        chat_template_kwargs={"enable_thinking": False},
        bucket_files=["bucket.jsonl"],
    )

    assert config["prompt_tokenization_version"] == PROMPT_TOKENIZATION_VERSION
    assert config["tokenizer_mode"] == "auto"
    assert config["chat_template_kwargs"] == {"enable_thinking": False}


def test_existing_qwen_v2_cache_remains_reusable(tmp_path):
    bucket_file = "bucket.jsonl"
    (tmp_path / bucket_file).write_text("{}\n{}\n", encoding="utf-8")
    expected = _build_expected_manifest_config(
        source_bucket_dir=Path("/source"),
        tokenizer_id="tokenizer",
        tokenizer_mode="auto",
        holdout_start_index=1,
        holdout_prompts_per_bucket=2,
        holdout_context_length=3,
        max_completion_tokens=4,
        prompt_token_limit=5,
        system_prompt="system",
        chat_template_kwargs={"enable_thinking": False},
        bucket_files=[bucket_file],
    )
    legacy_manifest = dict(expected)
    legacy_manifest.pop("chat_template_kwargs")
    legacy_manifest.pop("tokenizer_mode")
    legacy_manifest["bucket_counts"] = {bucket_file: 2}

    assert _is_cache_reusable(
        manifest=legacy_manifest,
        expected_manifest_config=expected,
        cache_dir=tmp_path,
        holdout_prompts_per_bucket=2,
        bucket_files=[bucket_file],
    )

    ministral_expected = dict(expected)
    ministral_expected["tokenizer_mode"] = "mistral"
    ministral_expected["chat_template_kwargs"] = {}
    assert not _is_cache_reusable(
        manifest=legacy_manifest,
        expected_manifest_config=ministral_expected,
        cache_dir=tmp_path,
        holdout_prompts_per_bucket=2,
        bucket_files=[bucket_file],
    )

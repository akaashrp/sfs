from pathlib import Path

from sfs_core.prep.holdout_cache import (
    PROMPT_TOKENIZATION_VERSION,
    _build_expected_manifest_config,
    _compute_prompt_tokens,
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


def test_manifest_versions_prompt_tokenization_semantics():
    config = _build_expected_manifest_config(
        source_bucket_dir=Path("/source"),
        tokenizer_id="tokenizer",
        holdout_start_index=1,
        holdout_prompts_per_bucket=2,
        holdout_context_length=3,
        max_completion_tokens=4,
        prompt_token_limit=5,
        system_prompt="system",
        bucket_files=["bucket.jsonl"],
    )

    assert config["prompt_tokenization_version"] == PROMPT_TOKENIZATION_VERSION

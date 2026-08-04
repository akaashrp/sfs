from __future__ import annotations

from typing import Any


TOKENIZER_MODES = ("auto", "mistral")


def normalize_tokenizer_mode(tokenizer_mode: str) -> str:
    """Validate the tokenizer backend used for prompt accounting."""
    normalized = str(tokenizer_mode).strip().lower()
    if normalized not in TOKENIZER_MODES:
        raise ValueError(
            f"Unsupported tokenizer mode {tokenizer_mode!r}; "
            f"choose from {', '.join(TOKENIZER_MODES)}."
        )
    return normalized


def load_tokenizer(model_or_path: str, *, tokenizer_mode: str = "auto") -> Any:
    """Load the same tokenizer backend selected for the vLLM server."""
    normalized = normalize_tokenizer_mode(tokenizer_mode)
    if normalized == "mistral":
        # Keep this import lazy so Qwen and other Hugging Face tokenizer users
        # do not pay vLLM's import cost or require mistral_common at startup.
        from vllm.transformers_utils.tokenizers import MistralTokenizer

        return MistralTokenizer.from_pretrained(model_or_path)

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_or_path, use_fast=True)

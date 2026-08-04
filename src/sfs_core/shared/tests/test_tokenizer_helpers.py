import sys
import types

import pytest

from sfs_core.shared.tokenizer_helpers import (
    load_tokenizer,
    normalize_tokenizer_mode,
)


def test_normalize_tokenizer_mode_is_case_insensitive():
    assert normalize_tokenizer_mode(" AUTO ") == "auto"
    assert normalize_tokenizer_mode("Mistral") == "mistral"


def test_normalize_tokenizer_mode_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported tokenizer mode"):
        normalize_tokenizer_mode("sentencepiece")


def test_load_tokenizer_uses_mistral_backend(monkeypatch):
    sentinel = object()

    class _MistralTokenizer:
        @classmethod
        def from_pretrained(cls, model_or_path):
            assert model_or_path == "/models/ministral"
            return sentinel

    tokenizer_module = types.ModuleType("vllm.transformers_utils.tokenizers")
    tokenizer_module.MistralTokenizer = _MistralTokenizer
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(
        sys.modules,
        "vllm.transformers_utils",
        types.ModuleType("vllm.transformers_utils"),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.transformers_utils.tokenizers",
        tokenizer_module,
    )

    assert load_tokenizer("/models/ministral", tokenizer_mode="mistral") is sentinel


def test_load_tokenizer_uses_hugging_face_backend(monkeypatch):
    sentinel = object()

    class _AutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_or_path, *, use_fast):
            assert model_or_path == "Qwen/Qwen3-8B"
            assert use_fast is True
            return sentinel

    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoTokenizer = _AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    assert load_tokenizer("Qwen/Qwen3-8B", tokenizer_mode="auto") is sentinel

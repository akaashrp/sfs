from __future__ import annotations

import re
from typing import Any, Optional


MODEL_PATTERNS = (
    (
        "qwen3",
        re.compile(r"qwen3[-_]?([0-9]+(?:\.[0-9]+)?)b", re.IGNORECASE),
    ),
    (
        "ministral3",
        re.compile(r"ministral[-_]?3[-_]?([0-9]+(?:\.[0-9]+)?)b", re.IGNORECASE),
    ),
)

SNAPSHOT_TO_MODEL_LABEL = {
    "c1899de289a04d12100db370d81485cdf75e47ca": "qwen3-0.6b",
    "b968826d9c46dd6066d109eabc6255188de91218": "qwen3-8b",
    "9216db5781bf21249d130ec9da846c4624c16137": "qwen3-32b",
    "b6d637bef2393152b3da2b2fde72eecdee30557e": "ministral3-3b",
    "f6fae9795746f63c9be8344932f01275f3c63734": "ministral3-8b",
    "3cea74c1ebaf5ce5f5a2553de470e2ceab825142": "ministral3-14b",
}

INSTANCE_TO_MODEL_LABEL = {
    "vllm-0.6b": "qwen3-0.6b",
    "vllm-8b": "qwen3-8b",
    "vllm-32b": "qwen3-32b",
    "vllm-ministral3-3b": "ministral3-3b",
    "vllm-ministral3-8b": "ministral3-8b",
    "vllm-ministral3-14b": "ministral3-14b",
}


def _normalize_model_number(number: str) -> str:
    if "." in number:
        number = number.rstrip("0").rstrip(".")
    return number


def normalize_model_label(label: str) -> str:
    raw = label.strip().lower()
    for family, pattern in MODEL_PATTERNS:
        match = pattern.search(raw)
        if match:
            return f"{family}-{_normalize_model_number(match.group(1))}b"
    return raw


def resolve_model_label(response_model: Any, instance_id: Any) -> Optional[str]:
    if isinstance(response_model, str) and response_model:
        for snapshot_hash, label in SNAPSHOT_TO_MODEL_LABEL.items():
            if snapshot_hash in response_model:
                return label

        for family, pattern in MODEL_PATTERNS:
            match = pattern.search(response_model)
            if match:
                return f"{family}-{_normalize_model_number(match.group(1))}b"

    if isinstance(instance_id, str):
        normalized = instance_id.strip().lower()
        mapped = INSTANCE_TO_MODEL_LABEL.get(normalized)
        if mapped is not None:
            return mapped

    return None

from sfs_core.shared.model_label_helpers import (
    normalize_model_label,
    resolve_model_label,
)


def test_normalize_model_label_supports_qwen_and_ministral():
    assert normalize_model_label("Qwen3-0.6B") == "qwen3-0.6b"
    assert (
        normalize_model_label("ministral3-14b-instruct")
        == "ministral3-14b"
    )


def test_resolve_ministral_model_from_name_snapshot_or_instance():
    assert (
        resolve_model_label("ministral3-8b-instruct", None)
        == "ministral3-8b"
    )
    assert (
        resolve_model_label(
            "/models/snapshots/3cea74c1ebaf5ce5f5a2553de470e2ceab825142",
            None,
        )
        == "ministral3-14b"
    )
    assert (
        resolve_model_label(None, "vllm-ministral3-3b")
        == "ministral3-3b"
    )

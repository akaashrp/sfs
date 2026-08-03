import pytest

from sfs_core.shared.shared_experiment_helpers import (
    parse_chat_template_kwargs_json,
    resolve_chat_template_kwargs,
)


def test_omitted_policy_preserves_qwen_non_thinking_default():
    assert resolve_chat_template_kwargs(None) == {"enable_thinking": False}


def test_empty_policy_supports_model_families_without_qwen_kwarg():
    assert parse_chat_template_kwargs_json("{}") == {}


@pytest.mark.parametrize(
    "value",
    (
        "[]",
        '{"tokenize": false}',
        '{"add_generation_prompt": false}',
    ),
)
def test_invalid_or_runner_controlled_policy_is_rejected(value):
    with pytest.raises(ValueError):
        parse_chat_template_kwargs_json(value)

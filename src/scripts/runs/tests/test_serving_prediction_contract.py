"""API admission must supply valid output work before the engine receives it."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def engine():
    from vllm.platforms.cpu import CpuPlatform
    with patch("vllm.platforms._current_platform", CpuPlatform()):
        from vllm.v1.engine.async_llm import AsyncLLM
        value = AsyncLLM.__new__(AsyncLLM)
        value.processor = SimpleNamespace(tokenizer=Mock())
        value.processor.tokenizer.decode.return_value = "decoded chat prompt"
        value.model_config = SimpleNamespace(model="mistralai/Ministral-3-3B-Instruct-2512-BF16")
        value.vllm_config = SimpleNamespace(scheduler_config=SimpleNamespace(
            enable_snapshot_shm_publishing=True, enable_wait_time_simulation=False))
        value.output_length_predictor = Mock()
        value.output_length_predictor.predict.return_value = SimpleNamespace(mean_tokens=12.5)
        yield value


def request(tokens=None):
    return SimpleNamespace(prompt_token_ids=[1, 2, 3] if tokens is None else tokens,
                           predicted_output_tokens_mean=None)


def test_token_only_chat_is_predicted_without_retokenizing(engine):
    value = request()
    original = value.prompt_token_ids.copy()
    engine._attach_output_length_prediction(value, None)
    assert value.predicted_output_tokens_mean == 12.5
    assert value.prompt_token_ids == original
    engine.tokenizer.decode.assert_called_once_with(original)
    admission = engine.output_length_predictor.predict.call_args.args[0]
    assert admission.prompt_text == "decoded chat prompt"
    assert admission.prompt_token_count == 3


def test_text_chat_preserves_existing_predictor_input(engine):
    value = request()
    engine._attach_output_length_prediction(value, "canonical rendered Qwen chat")
    assert value.predicted_output_tokens_mean == 12.5
    engine.tokenizer.decode.assert_not_called()
    assert engine.output_length_predictor.predict.call_args.args[0].prompt_text == "canonical rendered Qwen chat"


@pytest.mark.parametrize("mean", [None, float("nan"), float("inf"), -1.0, 0.0])
def test_invalid_prediction_is_rejected_before_engine_admission(engine, mean):
    engine.output_length_predictor.predict.return_value = None if mean is None else SimpleNamespace(mean_tokens=mean)
    value = request()
    with pytest.raises(ValueError, match="output-length prediction"):
        engine._attach_output_length_prediction(value, "prompt")
    assert value.predicted_output_tokens_mean is None


def test_snapshot_mode_rejects_missing_predictor(engine):
    engine.output_length_predictor = None
    with pytest.raises(ValueError, match="predictor"):
        engine._attach_output_length_prediction(request(), "prompt")


def test_token_only_prompt_without_tokenizer_is_rejected(engine):
    engine.processor.tokenizer = None
    with pytest.raises(ValueError, match="tokenizer"):
        engine._attach_output_length_prediction(request(), None)


def test_plain_serving_without_prediction_remains_supported(engine):
    engine.vllm_config.scheduler_config.enable_snapshot_shm_publishing = False
    engine.output_length_predictor = None
    value = request()
    engine._attach_output_length_prediction(value, "prompt")
    assert value.predicted_output_tokens_mean is None


def test_broken_configured_predictor_fails_before_starting_engine(engine, monkeypatch):
    from vllm.v1.engine import async_llm
    config = engine.vllm_config
    config.model_config = engine.model_config
    config.observability_config = SimpleNamespace(otlp_traces_endpoint=None)
    config.scheduler_config.output_length_model_path = "unusable-predictor"
    config.scheduler_config.output_length_tail_quantile = 0.9
    monkeypatch.setattr(async_llm, "Processor", Mock(return_value=engine.processor))
    monkeypatch.setattr(async_llm, "OutputLengthPredictor", Mock(side_effect=ValueError("bad model file")))
    start = Mock()
    monkeypatch.setattr(async_llm.EngineCoreClient, "make_async_mp_client", start)
    with pytest.raises(ValueError, match="usable output-length predictor"):
        async_llm.AsyncLLM(config, executor_class=object, log_stats=False)
    start.assert_not_called()

from __future__ import annotations

from scripts.runs.compute_model_service_metrics import _build_calibration_payload


def test_calibration_payload_disables_qwen_thinking():
    payload = _build_calibration_payload(
        prompt="Summarize this report.",
        max_completion_tokens=512,
        model_id="qwen-test",
    )

    assert payload["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert payload["max_completion_tokens"] == 512
    assert payload["model"] == "qwen-test"

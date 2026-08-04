from __future__ import annotations

import asyncio
import json

from scripts.runs import compute_model_service_metrics
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


def test_calibration_payload_supports_neutral_non_qwen_policy():
    payload = _build_calibration_payload(
        prompt="Summarize this report.",
        max_completion_tokens=512,
        model_id="ministral-test",
        system_prompt="You are a helpful assistant.",
        chat_template_kwargs={},
    )

    assert payload["messages"] == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Summarize this report."},
    ]
    assert payload["extra_body"] == {"chat_template_kwargs": {}}


def test_model_calibration_streams_run_concurrently(monkeypatch, tmp_path):
    model_names = ("qwen3-0.6b", "qwen3-8b", "qwen3-32b")
    clients = {
        model_name: type(
            "Client",
            (),
            {"default_model": model_name, "instance_id": model_name},
        )()
        for model_name in model_names
    }
    schedulers = {}
    scheduler_kwargs = {}

    class Scheduler:
        def __init__(self, model_name):
            self.model_name = model_name
            self.stopped = False

        async def stop(self):
            self.stopped = True

    def build_scheduler(model_name, _model_clients, **kwargs):
        scheduler = Scheduler(model_name)
        schedulers[model_name] = scheduler
        scheduler_kwargs[model_name] = kwargs
        return scheduler

    warmed_clients = []

    async def warm_up(all_clients):
        warmed_clients.extend(all_clients)

    entered = set()
    all_entered = asyncio.Event()

    async def compute_one(client, _prompt_records, **_kwargs):
        entered.add(client.instance_id)
        if len(entered) == len(model_names):
            all_entered.set()
        await asyncio.wait_for(all_entered.wait(), timeout=0.5)
        return {"model_id": client.instance_id}

    monkeypatch.setattr(
        compute_model_service_metrics,
        "build_single_model_wait_time_scheduler",
        build_scheduler,
    )
    monkeypatch.setattr(
        compute_model_service_metrics,
        "warm_up_instances",
        warm_up,
    )
    monkeypatch.setattr(
        compute_model_service_metrics,
        "_compute_single_model_metrics",
        compute_one,
    )
    monkeypatch.setattr(
        compute_model_service_metrics,
        "BATCH_STATS_FLUSH_GRACE_S",
        0,
    )

    output_path = tmp_path / "metrics.json"
    results = asyncio.run(
        compute_model_service_metrics.compute_metrics_for_models(
            clients,
            [{"prompt": "test", "prompt_tokens": 1}],
            tokenizer_id="/models/ministral-tokenizer",
            tokenizer_mode="mistral",
            output_path=output_path,
        )
    )

    assert entered == set(model_names)
    assert warmed_clients == list(clients.values())
    assert all(scheduler.stopped for scheduler in schedulers.values())
    assert all(
        kwargs == {
            "enable_wait_time_polling": False,
            "tokenizer_id": "/models/ministral-tokenizer",
            "tokenizer_mode": "mistral",
        }
        for kwargs in scheduler_kwargs.values()
    )
    assert list(results) == list(model_names)
    assert json.loads(output_path.read_text(encoding="utf-8")) == results

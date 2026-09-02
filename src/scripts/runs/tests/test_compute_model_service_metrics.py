from __future__ import annotations

import asyncio
import json

from scripts.runs import compute_model_service_metrics
from scripts.runs.compute_model_service_metrics import _build_calibration_payload
from scripts.runs.compute_model_service_metrics import get_instance_clients_from_config


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


def test_family_neutral_instances_config(monkeypatch, tmp_path):
    created = []

    class Client:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            created.append(kwargs)

    monkeypatch.setattr(compute_model_service_metrics, "InstanceClient", Client)
    path = tmp_path / "instances.json"
    path.write_text(
        json.dumps(
            {
                "instances": [
                    {
                        "model_key": "ministral3-3b",
                        "instance_id": "vllm-ministral3-3b",
                        "address": "http://localhost:8100",
                        "default_model": "ministral3-3b-instruct",
                        "model_id": "ministral3-3b-instruct",
                    },
                    {
                        "model_key": "ministral3-8b",
                        "instance_id": "vllm-ministral3-8b",
                        "address": "http://localhost:8101",
                        "default_model": "ministral3-8b-instruct",
                        "model_id": "ministral3-8b-instruct",
                    },
                ],
                "instance_costs": {
                    "vllm-ministral3-3b": {"prompt": 0.1, "output": 0.1},
                    "vllm-ministral3-8b": {"prompt": 0.15, "output": 0.15},
                },
            }
        ),
        encoding="utf-8",
    )

    clients, costs = get_instance_clients_from_config(path)

    assert list(clients) == ["ministral3-3b", "ministral3-8b"]
    assert clients["ministral3-3b"].instance_id == "vllm-ministral3-3b"
    assert clients["ministral3-8b"].default_model == "ministral3-8b-instruct"
    assert costs == {
        "vllm-ministral3-3b": {"prompt": 0.1, "output": 0.1},
        "vllm-ministral3-8b": {"prompt": 0.15, "output": 0.15},
    }
    assert len(created) == 2

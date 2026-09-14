from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from scripts.runs import experiments
from sfs_core.shared.shared_experiment_helpers import warm_up_instances


def test_warmup_uses_valid_mistral_messages_and_request_sampling():
    from mistral_common.protocol.instruct.request import ChatCompletionRequest

    calls = []

    async def submit(**payload):
        ChatCompletionRequest(messages=payload["messages"])
        calls.append(payload)

    asyncio.run(warm_up_instances([SimpleNamespace(submit_request=submit)]))
    assert len(calls) == 1
    assert calls[0]["temperature"] == 0
    assert calls[0]["top_p"] == 1
    assert calls[0]["max_completion_tokens"] == 1


def test_warmup_reports_failure_after_all_instances_finish():
    completed = []

    async def fail(**_):
        raise ValueError("rejected warm-up")

    async def succeed(**_):
        await asyncio.sleep(0)
        completed.append(True)

    clients = [SimpleNamespace(instance_id="bad", submit_request=fail),
               SimpleNamespace(instance_id="good", submit_request=succeed)]
    with pytest.raises(RuntimeError, match="bad.*rejected warm-up"):
        asyncio.run(warm_up_instances(clients))
    assert completed == [True]


def test_batch_fit_has_no_prompt_directory_dependency(tmp_path, monkeypatch):
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["experiments", "--experiment", "batch_fit",
        "--bucket-dir", str(tmp_path / "missing-qwen-directory"),
        "--output-path", str(output)])
    args = experiments.parse_args()
    monkeypatch.setattr(experiments, "run_batch_fit_experiment", lambda **_: {"cpu_fit": True})
    asyncio.run(experiments.async_main(args))
    result = json.loads(output.read_text())
    assert result["batch_fit"] == {"cpu_fit": True}
    assert result["config"]["bucket_files"] == []

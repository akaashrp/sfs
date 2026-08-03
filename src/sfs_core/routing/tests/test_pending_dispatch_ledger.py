from __future__ import annotations

import asyncio
import time
from types import MethodType, SimpleNamespace

import vllm.v1.engine

try:
    from vllm.v1.engine import _scheduler_sim  # noqa: F401
except ImportError:
    vllm.v1.engine._scheduler_sim = SimpleNamespace()

from sfs_core.routing.pending_dispatch_ledger import (
    PendingDispatch,
    PendingDispatchLedger,
)
from sfs_core.routing.wait_time_scheduler import (
    WaitTimeResult,
    WaitTimeScheduler,
    _QueuedRequest,
)


def test_ledger_preserves_per_instance_order_and_reconciles_observed_ids():
    ledger = PendingDispatchLedger(("a", "b"))
    first = PendingDispatch("chatcmpl-1", 8, 4.0, 16, 123.5)
    second = PendingDispatch("chatcmpl-2", 16, 8.0, 32)
    ledger.reserve("a", first)
    ledger.reserve("a", second)

    assert ledger.unobserved_for_instance("a") == (first, second)
    assert ledger.unobserved_for_instance("b") == ()
    assert first.as_native_tuple() == (
        "chatcmpl-1",
        8,
        4.0,
        16,
        123.5,
    )

    ledger.mark_observed("a", ("chatcmpl-1",))
    assert ledger.unobserved_for_instance("a") == (second,)
    ledger.release("a", "chatcmpl-2")
    assert ledger.unobserved_for_instance("a") == ()


class _FakeInstance:
    def __init__(self, instance_id: str, gate: asyncio.Event) -> None:
        self.instance_id = instance_id
        self._gate = gate

    async def submit_request(self, **_payload):
        await self._gate.wait()
        return SimpleNamespace(id=f"response-{self.instance_id}", model="test")

    def last_wait_for_mode(self, **_kwargs):
        return None


def _build_scheduler(gate: asyncio.Event) -> WaitTimeScheduler:
    scheduler = WaitTimeScheduler.__new__(WaitTimeScheduler)
    scheduler._instances = {
        "a": _FakeInstance("a", gate),
        "b": _FakeInstance("b", gate),
    }
    scheduler._routing_state_lock = asyncio.Lock()
    scheduler._pending_dispatch_ledger = PendingDispatchLedger(("a", "b"))
    scheduler._request_log = {}
    scheduler._log_path = None
    scheduler._submit_tasks = set()
    scheduler._response_map_path = None

    async def _predict_model_scores(
        _self,
        *,
        prompt_text,
        prompt_tokens,
        completion_cap,
    ):
        return {}, {"a": 4.0, "b": 4.0}

    active_estimates = 0

    async def _collect_wait_times(
        _self,
        prompt_tokens=None,
        stop_mode=None,
        pending_dispatches_by_instance=None,
        probe_ready_delay_ms_by_instance=None,
    ):
        nonlocal active_estimates
        active_estimates += 1
        assert active_estimates == 1
        await asyncio.sleep(0)
        pending = pending_dispatches_by_instance or {}
        now = time.time()
        results = {
            instance_id: WaitTimeResult(
                instance_id=instance_id,
                wait_ms=float(len(pending.get(instance_id, ()))),
                fetched_at_s=now,
                raw_payload={
                    "reports": [{"num_requests": len(pending.get(instance_id, ()))}]
                },
            )
            for instance_id in _self._instances
        }
        active_estimates -= 1
        return results

    def _select_instance(
        _self,
        wait_results,
        accuracy_scores=None,
        output_lengths=None,
        prompt_tokens=None,
    ):
        return min(
            wait_results,
            key=lambda instance_id: wait_results[instance_id].raw_payload["reports"][0][
                "num_requests"
            ],
        )

    scheduler._predict_model_scores = MethodType(
        _predict_model_scores,
        scheduler,
    )
    scheduler._collect_wait_times = MethodType(
        _collect_wait_times,
        scheduler,
    )
    scheduler._select_instance = MethodType(_select_instance, scheduler)
    return scheduler


def test_readiness_delay_uses_candidate_specific_pending_count():
    scheduler = WaitTimeScheduler.__new__(WaitTimeScheduler)
    scheduler._instances = {"a": object(), "b": object()}
    scheduler._readiness_predictor = SimpleNamespace(
        predict_ms=lambda *, prompt_tokens, pending_dispatch_count: (
            1.0 + 0.01 * prompt_tokens + 4.0 * pending_dispatch_count
        )
    )
    delays = scheduler._probe_ready_delays_by_instance(
        prompt_tokens=100,
        pending_dispatches_by_instance={
            "a": (),
            "b": (
                PendingDispatch(
                    "pending",
                    8,
                    4.0,
                    16,
                    predicted_ready_at_s=101.0,
                ),
                PendingDispatch(
                    "already-ready",
                    8,
                    4.0,
                    16,
                    predicted_ready_at_s=99.0,
                ),
            ),
        },
        reference_time_s=100.0,
    )

    assert delays == {"a": 2.0, "b": 6.0}


def test_predicted_ready_time_uses_selected_simulation_timestamp():
    wait_record = WaitTimeResult(
        instance_id="a",
        wait_ms=1.0,
        fetched_at_s=time.time(),
        raw_payload={
            "reports": [{"simulation_timestamp": 100.0}],
        },
    )

    assert WaitTimeScheduler._predicted_ready_at_s(
        wait_record=wait_record,
        delay_ms=7.5,
    ) == 100.0075


def _queued_request(request_id: str) -> _QueuedRequest:
    loop = asyncio.get_running_loop()
    return _QueuedRequest(
        request_id=request_id,
        payload={
            "messages": [{"role": "user", "content": request_id}],
            "max_completion_tokens": 16,
            "_prompt_tokens": 8,
        },
        result_future=loop.create_future(),
        enqueued_at_s=time.time(),
    )


def test_prompt_token_fallback_passes_direct_thinking_kwarg():
    class _Tokenizer:
        def __init__(self) -> None:
            self.kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            self.kwargs = kwargs
            return [1, 2, 3]

    scheduler = WaitTimeScheduler.__new__(WaitTimeScheduler)
    scheduler._tokenizer = _Tokenizer()

    assert (
        scheduler._get_prompt_tokens(
            {"messages": [{"role": "user", "content": "hello"}]},
            "hello",
        )
        == 3
    )
    assert scheduler._tokenizer.kwargs["enable_thinking"] is False
    assert "chat_template_kwargs" not in scheduler._tokenizer.kwargs


def test_prompt_token_fallback_uses_request_template_kwargs():
    class _Tokenizer:
        def __init__(self) -> None:
            self.kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            self.kwargs = kwargs
            return [1, 2, 3]

    scheduler = WaitTimeScheduler.__new__(WaitTimeScheduler)
    scheduler._tokenizer = _Tokenizer()
    payload = {
        "messages": [{"role": "user", "content": "hello"}],
        "extra_body": {"chat_template_kwargs": {}},
    }

    assert scheduler._get_prompt_tokens(payload, "hello") == 3
    assert scheduler._tokenizer.kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
    }


def test_two_dispatch_workers_route_against_atomic_pending_state():
    asyncio.run(_exercise_two_dispatch_workers())


async def _exercise_two_dispatch_workers() -> None:
    gate = asyncio.Event()
    scheduler = _build_scheduler(gate)
    first = _queued_request("request-1")
    second = _queued_request("request-2")

    await asyncio.gather(
        scheduler._dispatch(first),
        scheduler._dispatch(second),
    )

    assert (await first.result_future).instance_id == "a"
    assert (await second.result_future).instance_id == "b"
    assert first.payload["extra_body"]["request_id"] == "request-1"
    assert second.payload["extra_body"]["request_id"] == "request-2"
    assert len(scheduler._pending_dispatch_ledger.unobserved_for_instance("a")) == 1
    assert len(scheduler._pending_dispatch_ledger.unobserved_for_instance("b")) == 1

    gate.set()
    await asyncio.gather(*scheduler._submit_tasks)
    await asyncio.sleep(0)
    assert scheduler._pending_dispatch_ledger.unobserved_for_instance("a") == ()
    assert scheduler._pending_dispatch_ledger.unobserved_for_instance("b") == ()


def test_failed_submission_releases_reservation():
    asyncio.run(_exercise_failed_submission())


async def _exercise_failed_submission() -> None:
    gate = asyncio.Event()
    scheduler = _build_scheduler(gate)

    async def _fail(**_payload):
        raise RuntimeError("submission failed")

    scheduler._instances["a"].submit_request = _fail
    queued = _queued_request("failed")
    await scheduler._dispatch(queued)
    await asyncio.gather(*scheduler._submit_tasks, return_exceptions=True)
    await asyncio.sleep(0)

    assert scheduler._pending_dispatch_ledger.unobserved_for_instance("a") == ()

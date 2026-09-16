from __future__ import annotations

import asyncio
import time

import pytest

from sfs_core.routing import wait_time_scheduler
from sfs_core.routing.methodology_policies import RouteBalanceWeights
from sfs_core.routing.methodology_scheduler import MethodologyScheduler
from sfs_core.routing.methodology_snapshot import parse_baseline_snapshot
from sfs_core.routing.wait_time_scheduler import _QueuedRequest


def snapshot_payload(*, requests=None, version=1, age=0):
    requests = requests or {}
    running = [rid for rid, request in requests.items() if request.get("status") != "WAITING"]
    waiting = [rid for rid in requests if rid not in running]
    return {
        "version": version, "created_at": time.time() - age,
        "num_running": len(running), "num_waiting": len(waiting),
        "running_request_ids": running, "waiting_request_ids": waiting,
        "requests": requests, "inflight_batch": None,
        "config": {"max_num_batched_tokens": 128, "max_num_seqs": 512,
                   "max_model_len": 4096, "chunked_prefill_enabled": True},
        "kv_cache_config": {"block_size": 16,
                            "kv_cache_groups": [{"kv_cache_spec": {"block_size": 16}}],
                            "kv_cache_free_blocks": 1000, "kv_cache_total_blocks": 2000},
        "parallel_config": {"decode_context_parallel_size": 1},
    }


def request_state(rid, *, prompt=8, computed=8, generated=2):
    return {"request_id": rid, "status": "RUNNING", "num_prompt_tokens": prompt,
            "num_computed_tokens": computed, "num_output_processed_tokens": generated,
            "num_output_target_tokens": 999999, "kv_block_counts": [2]}


class FakeClient:
    def __init__(self, instance_id):
        self.instance_id = self.model_id = self.default_model = instance_id
        self.address = "fake://" + instance_id
        self.raw = snapshot_payload()
        self.snapshot_calls = 0
        self.submissions = []
        self.gate = None
        self.fail = False

    async def refresh_baseline_state(self):
        self.snapshot_calls += 1
        return parse_baseline_snapshot(self.raw, observed_at=time.time())

    async def submit_request(self, **payload):
        self.submissions.append(payload)
        if self.gate is not None:
            await self.gate.wait()
        if self.fail:
            raise RuntimeError("synthetic submit failure")
        return {"id": f"response-{len(self.submissions)}-{self.instance_id}",
                "model": self.model_id,
                "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11}}

    def close(self):
        pass


class FakeCalibration:
    speeds = {"a": 1.0, "b": 10.0}
    metadata = {"source": "synthetic CPU test"}

    def __init__(self):
        self.tpot_calls = []

    def prefill_ms(self, model, prompt, computed=0):
        return (prompt * prompt - computed * computed) / 100

    def tpot_ms(self, model, features):
        self.tpot_calls.append((model, features))
        return 1.0


class FakePredictor:
    model_labels = ("a", "b")
    metadata = {"source": "synthetic native predictor interface"}

    def __init__(self):
        self.calls = []

    def predict_batch(self, prompts, caps):
        self.calls.append((prompts, caps))
        return [{model: {"quality": .5, "output_tokens": min(cap, 100 if prompt == "long" else 10)}
                 for model in self.model_labels} for prompt, cap in zip(prompts, caps)]


def queued(rid, prompt="short", *, elapsed=.02):
    loop = asyncio.get_running_loop()
    entered = time.perf_counter() - elapsed
    return _QueuedRequest(rid, {"prompt": prompt, "max_tokens": 128, "_prompt_tokens": 8,
                               "_system_entry_perf": entered, "_started_perf": entered,
                               "_completion_future": loop.create_future()},
                          loop.create_future(), time.time())


def make_scheduler(monkeypatch, tmp_path, policy, **kwargs):
    monkeypatch.setattr(wait_time_scheduler, "load_tokenizer", lambda *args, **kwargs: None)
    clients = {name: FakeClient(name) for name in ("a", "b")}
    calibration = FakeCalibration()
    predictor = FakePredictor() if policy == "routebalance" else None
    scheduler = MethodologyScheduler(
        clients, policy=policy, calibration=calibration, predictor=predictor,
        request_log_path=str(tmp_path / "requests.jsonl"),
        response_map_path=str(tmp_path / "response_map.jsonl"),
        instance_costs={"a": {"prompt": 1, "output": 2},
                        "b": {"prompt": 2, "output": 4}},
        route_random_seed=7, **kwargs)
    return scheduler, clients, calibration, predictor


def test_lmdeploy_lifetime_includes_response_wait_and_never_invokes_snapshots(monkeypatch, tmp_path):
    async def run():
        scheduler, clients, _, _ = make_scheduler(monkeypatch, tmp_path, "lmdeploy_proxy")
        gate = asyncio.Event()
        for client in clients.values():
            client.gate = gate
        first, second = queued("first"), queued("second")
        await scheduler._dispatch_batch([first])
        await scheduler._dispatch_batch([second])
        await asyncio.sleep(0)
        assert first.result_future.result().instance_id != second.result_future.result().instance_id
        assert scheduler.lifetime.unfinished_counts() == {"a": 1, "b": 1}
        assert all(client.snapshot_calls == 0 for client in clients.values())
        assert scheduler._accuracy_predictor is None and scheduler._output_length_predictor is None
        gate.set()
        await scheduler.drain()
        assert len(scheduler.lifetime) == 0
        record = await first.payload["_completion_future"]
        assert record["latency_ms"] >= 20
        assert record["system_entry_perf"] == first.payload["_system_entry_perf"]
        assert record["usage_completion_tokens"] == 3
        assert (tmp_path / "response_map.jsonl").exists()
    asyncio.run(run())


def test_actual_routebalance_batch_lpt_updates_local_work_and_times_predictions(monkeypatch, tmp_path):
    async def run():
        scheduler, _, calibration, predictor = make_scheduler(
            monkeypatch, tmp_path, "routebalance", weights=RouteBalanceWeights(0, 0, 1))
        short, long = queued("short"), queued("long", "long")
        await scheduler._dispatch_batch([short, long])
        await scheduler.drain()
        short_record = await short.payload["_completion_future"]
        long_record = await long.payload["_completion_future"]
        assert "error" not in short_record and "error" not in long_record
        assert long_record["methodology_terms"]["batch_position"] == 0
        assert short_record["methodology_terms"]["batch_position"] == 1
        assert long_record["instance_id"] != short_record["instance_id"]
        chosen = long_record["instance_id"]
        later_terms = short_record["methodology_terms"]["candidates"][chosen]
        assert later_terms["pending_decode_tokens"] == 100
        assert later_terms["unobserved_reservations"] == 1
        assert len(predictor.calls) == 1
        assert len(calibration.tpot_calls) == 2  # one head call per tier per batch
        assert long_record["methodology_terms"]["collection_wait_ms"] >= 20
        assert short_record["methodology_terms"]["prediction_batch_ms"] >= 0
        assert long_record["methodology_terms"]["candidates"]["a"]["predicted_cost"] == 208
        assert len(scheduler.lifetime) == 0
    asyncio.run(run())


def test_native_pending_lengths_ignore_sfs_targets_and_count_observed_once(monkeypatch, tmp_path):
    async def run():
        scheduler, clients, _, _ = make_scheduler(monkeypatch, tmp_path, "routebalance")
        scheduler.lifetime.reserve("existing", "a", prompt_tokens=8, predicted_output_tokens=10)
        clients["a"].raw = snapshot_payload(requests={"existing": request_state("existing", generated=3)})
        current = queued("current")
        await scheduler._dispatch_batch([current])
        await scheduler.drain()
        record = await current.payload["_completion_future"]
        assert "error" not in record
        terms = record["methodology_terms"]["candidates"]["a"]
        assert terms["pending_decode_tokens"] == 7
        assert terms["unobserved_reservations"] == 0
        scheduler.lifetime.release("existing")
    asyncio.run(run())


def test_submission_failure_and_cancel_release_exactly_once(monkeypatch, tmp_path):
    async def run():
        scheduler, clients, _, _ = make_scheduler(monkeypatch, tmp_path, "lmdeploy_proxy")
        for client in clients.values():
            client.fail = True
        failed = queued("failed")
        await scheduler._dispatch_batch([failed])
        await scheduler.drain()
        record = await failed.payload["_completion_future"]
        assert "synthetic submit failure" in record["error"]
        assert len(scheduler.lifetime) == 0
        for client in clients.values():
            client.fail = False
            client.gate = asyncio.Event()
        cancelled = queued("cancelled")
        await scheduler._dispatch_batch([cancelled])
        await asyncio.sleep(0)
        assert len(scheduler.lifetime) == 1
        for task in tuple(scheduler._submit_tasks):
            task.cancel()
        await scheduler.drain()
        record = await cancelled.payload["_completion_future"]
        assert "CancelledError" in record["error"]
        assert len(scheduler.lifetime) == 0
    asyncio.run(run())


@pytest.mark.parametrize("cancel_completion", [False, True])
def test_submission_cancelled_before_coroutine_start_releases_reservation(monkeypatch, tmp_path, cancel_completion):
    async def run():
        scheduler, _, _, _ = make_scheduler(monkeypatch, tmp_path, "lmdeploy_proxy")
        cancelled = queued("cancelled-before-start")
        await scheduler._dispatch_batch([cancelled])
        if cancel_completion:
            cancelled.payload["_completion_future"].cancel()
        for task in tuple(scheduler._submit_tasks):
            task.cancel()
        await scheduler.drain()
        assert len(scheduler.lifetime) == 0
        assert cancelled.payload["_completion_future"].done()
        if not cancel_completion:
            assert "CancelledError" in cancelled.payload["_completion_future"].result()["error"]
    asyncio.run(run())


def test_routebalance_unknown_and_regressed_snapshots_fail_but_old_tombstones_are_ignored(monkeypatch, tmp_path):
    async def run():
        scheduler, clients, _, _ = make_scheduler(monkeypatch, tmp_path, "routebalance")
        clients["a"].raw = snapshot_payload(requests={"old": request_state("old")})
        bad = queued("bad")
        await scheduler._dispatch_batch([bad])
        assert "Untracked" in (await bad.payload["_completion_future"])["error"]
        assert len(scheduler.lifetime) == 0
        scheduler._completed_ids.add("old")
        good = queued("good")
        await scheduler._dispatch_batch([good])
        await scheduler.drain()
        assert "error" not in await good.payload["_completion_future"]
        clients["a"].raw = snapshot_payload(requests={"old": request_state("old")}, age=10)
        snapshots = await scheduler._read_snapshots()
        assert snapshots["a"].age_ms >= 10000
        assert not scheduler._capacity_current["a"]
        clients["a"].raw = snapshot_payload(version=0)
        with pytest.raises(RuntimeError, match="regressed"):
            await scheduler._read_snapshots()
    asyncio.run(run())


def test_old_empty_snapshot_retains_unobserved_requests(monkeypatch, tmp_path):
    async def run():
        scheduler, clients, _, _ = make_scheduler(monkeypatch, tmp_path, "routebalance")
        clients["a"].raw = snapshot_payload(age=10)
        snapshots = await scheduler._read_snapshots()
        assert snapshots["a"].age_ms >= 10000
        scheduler.lifetime.reserve("unobserved", "a", predicted_output_tokens=10)
        snapshots = await scheduler._read_snapshots()
        assert snapshots["a"].age_ms >= 10000
    asyncio.run(run())


def test_old_empty_snapshot_handoff_retains_work_without_age_deadline(monkeypatch, tmp_path):
    async def run():
        scheduler, clients, _, _ = make_scheduler(
            monkeypatch, tmp_path, "routebalance", weights=RouteBalanceWeights(0, 1, 0))
        gate = asyncio.Event()
        for client in clients.values():
            client.gate = gate
        clients["a"].raw = snapshot_payload(age=10)
        incoming = queued("handoff")
        await scheduler._dispatch_batch([incoming])
        assert incoming.result_future.result().instance_id == "a"
        # The engine has not published the just-forwarded request yet.
        snapshots = await scheduler._read_snapshots()
        assert snapshots["a"].age_ms >= 10000
        assert scheduler.lifetime.unfinished_counts()["a"] == 1
        scheduler._dispatch_times["handoff"] = time.perf_counter() - 2
        snapshots = await scheduler._read_snapshots()
        assert snapshots["a"].age_ms >= 10000
        gate.set()
        await scheduler.drain()
    asyncio.run(run())


def test_chat_engine_request_id_reconciles_with_router_lifetime_identity(monkeypatch, tmp_path):
    async def run():
        scheduler, clients, _, _ = make_scheduler(monkeypatch, tmp_path, "routebalance")
        gate = asyncio.Event()
        for client in clients.values():
            client.gate = gate
        first, second = queued("chat-first"), queued("chat-second")
        for request in (first, second):
            request.payload.pop("prompt")
            request.payload["messages"] = [{"role": "user", "content": "short"}]
        await scheduler._dispatch_batch([first])
        first_instance = first.result_future.result().instance_id
        engine_id = "chatcmpl-chat-first"
        clients[first_instance].raw = snapshot_payload(
            requests={engine_id: request_state(engine_id, computed=11, generated=3)})
        await scheduler._dispatch_batch([second])
        gate.set()
        await scheduler.drain()
        first_record = await first.payload["_completion_future"]
        second_record = await second.payload["_completion_future"]
        assert "error" not in second_record
        assert first_record["methodology_terms"]["engine_request_id"] == engine_id
        terms = second_record["methodology_terms"]["candidates"][first_instance]
        assert terms["pending_decode_tokens"] == 7
        assert terms["unobserved_reservations"] == 0
        assert engine_id in scheduler._completed_ids
        assert len(scheduler.lifetime) == 0
    asyncio.run(run())


def test_collector_idle_flush_busy_timeout_size_and_end_of_arrivals(monkeypatch, tmp_path):
    async def run():
        scheduler, _, _, _ = make_scheduler(monkeypatch, tmp_path, "routebalance",
                                            batch_wait_ms=30, batch_max_size=3)
        first = queued("idle", elapsed=0)
        assert await scheduler._collect_batch(first) == [first]
        scheduler.lifetime.reserve("busy-a", "a")
        scheduler.lifetime.reserve("busy-b", "b")
        before = time.perf_counter()
        expired = queued("timeout", elapsed=0)
        assert await scheduler._collect_batch(expired) == [expired]
        assert time.perf_counter() - before >= .015
        batch = [queued(str(i), elapsed=0) for i in range(3)]
        for request in batch[1:]:
            scheduler._queue.put_nowait(request)
        assert await scheduler._collect_batch(batch[0]) == batch
        for _ in batch[1:]:
            scheduler._queue.task_done()
        last = queued("last", elapsed=0)
        collect = asyncio.create_task(scheduler._collect_batch(last))
        await asyncio.sleep(.002)
        scheduler.finish_arrivals()
        assert await collect == [last]
        scheduler.lifetime.release("busy-a")
        scheduler.lifetime.release("busy-b")
    asyncio.run(run())


def test_worker_cancellation_after_partial_collection_resolves_every_dequeued_request(monkeypatch, tmp_path):
    async def run():
        scheduler, _, _, _ = make_scheduler(monkeypatch, tmp_path, "routebalance",
                                            batch_wait_ms=1000, batch_max_size=8)
        scheduler.lifetime.reserve("busy-a", "a")
        scheduler.lifetime.reserve("busy-b", "b")
        first, second = queued("first", elapsed=0), queued("second", elapsed=0)
        scheduler._queue.put_nowait(first)
        scheduler._queue.put_nowait(second)
        worker = asyncio.create_task(scheduler._worker_loop())
        await asyncio.sleep(.01)
        assert scheduler._queue.empty()
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        assert first.payload["_completion_future"].done()
        assert second.payload["_completion_future"].done()
        await asyncio.wait_for(scheduler._queue.join(), timeout=.1)
        assert "CancelledError" in first.payload["_completion_future"].result()["error"]
        assert "CancelledError" in second.payload["_completion_future"].result()["error"]
    asyncio.run(run())


def test_concurrent_start_keeps_single_batch_coordinator(monkeypatch, tmp_path):
    async def run():
        scheduler, _, _, _ = make_scheduler(monkeypatch, tmp_path, "routebalance")
        await asyncio.gather(scheduler.start(), scheduler.start())
        count = len(scheduler._workers)
        await scheduler.stop(close_instances=False)
        assert count == 1
    asyncio.run(run())

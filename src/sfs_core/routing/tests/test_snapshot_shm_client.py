from __future__ import annotations

import os
import time

import pytest

from sfs_core.routing.snapshot_shm_client import SnapshotShmClient
from vllm.v1.core.sched.snapshot_serialization import (
    encode_scheduler_state_snapshot,
)
from vllm.v1.core.sched.state_snapshot import (
    RequestStateSnapshot,
    SchedulerConfigSnapshot,
    SchedulerKVCacheSnapshot,
    SchedulerParallelSnapshot,
    SchedulerStateSnapshot,
)
from vllm.v1.engine.snapshot_shm import SnapshotShmPublisher
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheSpec


pytest.importorskip(
    "vllm.v1.engine._scheduler_sim",
    reason="scheduler simulator native extension is not built",
)


def _publisher_name() -> str:
    return f"router_snapshot_client_test_{os.getpid()}_{time.time_ns()}"


def _build_snapshot(version: int, created_at: float) -> SchedulerStateSnapshot:
    waiting_request = RequestStateSnapshot(
        request_id=f"wait-{version}",
        status="WAITING",
        priority=0,
        arrival_time=created_at,
        num_prompt_tokens=32,
        num_computed_tokens=0,
        num_output_target_tokens=64,
        num_prompt_processed_tokens=0,
        num_output_processed_tokens=0,
        max_tokens=64,
        num_preemptions=0,
        num_cached_tokens=0,
        is_long_prompt=False,
        kv_block_counts=(0,),
    )
    return SchedulerStateSnapshot(
        version=version,
        created_at=created_at,
        num_running=0,
        num_waiting=1,
        running_request_ids=[],
        waiting_request_ids=[waiting_request.request_id],
        requests={waiting_request.request_id: waiting_request},
        config=SchedulerConfigSnapshot(
            max_num_batched_tokens=128,
            max_num_seqs=4,
            max_model_len=2048,
            long_prefill_token_threshold=256,
            chunked_prefill_enabled=True,
            policy="fcfs",
        ),
        kv_cache_config=SchedulerKVCacheSnapshot(
            num_gpu_blocks=1024,
            block_size=16,
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=["layer0"],
                    kv_cache_spec=KVCacheSpec(block_size=16),
                )
            ],
            kv_cache_usage=0.1,
            kv_cache_total_blocks=2048,
            kv_cache_free_blocks=1800,
        ),
        parallel_config=SchedulerParallelSnapshot(
            decode_context_parallel_size=1,
        ),
        waiting_set_size=1,
        prefill_backlog_waiting_tokens=32,
        prefill_backlog_total_tokens=32,
        build_latency_ms=1.5,
    )


def _eventually_estimate(
    client: SnapshotShmClient,
    *,
    prompt_tokens: int | None,
    timeout_s: float = 5.0,
):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            return client.estimate(prompt_tokens=prompt_tokens)
        except RuntimeError:
            time.sleep(0.01)
    raise AssertionError("SnapshotShmClient did not produce an estimate in time")


def test_snapshot_shm_client_prompt_aware_estimate_uses_native_watcher():
    snapshot = _build_snapshot(version=51, created_at=51.0)
    publisher = SnapshotShmPublisher(
        name=_publisher_name(),
        size_bytes=1024 * 1024,
        simulation_intercept=1.0,
        simulation_prefill_coeff=0.1,
        simulation_prefill_sq_coeff=0.0,
        simulation_decode_coeff=0.2,
        simulation_sum_coeff=0.0,
        simulation_sum_sq_coeff=0.0,
    )
    client = SnapshotShmClient(
        shm_name=publisher.name,
        shm_size_bytes=publisher.size_bytes,
    )
    try:
        assert publisher.publish(snapshot, encode_scheduler_state_snapshot(snapshot))
        estimate = _eventually_estimate(client, prompt_tokens=32)
        report = estimate.payload["reports"][0]
        assert report["snapshot_version"] == 51
        assert report["snapshot_timestamp"] == 51.0
        assert report["metadata"]["simulation_mode"] == "critical_path_prefill_done"
        assert report["metadata"]["prefill_backlog_total_tokens"] == 32.0
        assert report["snapshot_build_latency_ms"] == 1.5
    finally:
        client.close()
        publisher.close()


def test_snapshot_shm_client_snapshot_only_uses_parsed_summary():
    snapshot = _build_snapshot(version=52, created_at=52.0)
    publisher = SnapshotShmPublisher(
        name=_publisher_name(),
        size_bytes=1024 * 1024,
        simulation_intercept=1.0,
        simulation_prefill_coeff=0.1,
        simulation_prefill_sq_coeff=0.0,
        simulation_decode_coeff=0.2,
        simulation_sum_coeff=0.0,
        simulation_sum_sq_coeff=0.0,
    )
    client = SnapshotShmClient(
        shm_name=publisher.name,
        shm_size_bytes=publisher.size_bytes,
    )
    try:
        assert publisher.publish(snapshot, encode_scheduler_state_snapshot(snapshot))
        _eventually_estimate(client, prompt_tokens=32)
        estimate = _eventually_estimate(client, prompt_tokens=None)
        report = estimate.payload["reports"][0]
        assert report["snapshot_version"] == 52
        assert report["snapshot_timestamp"] == 52.0
        assert report["num_requests"] == 1
        assert report["metadata"]["simulation_mode"] == "snapshot_only"
        assert report["metadata"]["prefill_backlog_total_tokens"] == 32.0
    finally:
        client.close()
        publisher.close()

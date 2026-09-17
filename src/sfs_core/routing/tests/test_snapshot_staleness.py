"""Snapshot-staleness injection: delay-buffer semantics, D = 0 unchanged, delayed feed of the native simulator."""
from __future__ import annotations

import time
import uuid

import pytest

from sfs_core.routing import snapshot_shm_client as module
from sfs_core.routing.snapshot_shm_client import SnapshotDelayBuffer, SnapshotShmClient
from vllm.v1.engine.snapshot_shm import SnapshotShmHeader, SnapshotShmPublisher

pytest.importorskip("vllm.v1.engine._scheduler_sim", reason="scheduler simulator native extension is not built")


def _header(version, created_at):
    return SnapshotShmHeader(sequence=2 * version, snapshot_version=version, created_at=created_at, payload_size=1,
                             prefill_backlog_total_tokens=0., build_latency_ms=0., simulation_intercept=0.,
                             simulation_prefill_coeff=0., simulation_prefill_sq_coeff=0., simulation_decode_coeff=0.,
                             simulation_sum_coeff=0., simulation_sum_sq_coeff=0.)


def test_buffer_serves_the_newest_publication_older_than_the_delay():
    buffer = SnapshotDelayBuffer(25.0)
    for version, created in enumerate([0.0, 0.010, 0.020, 0.030, 0.040], start=1):
        assert buffer.observe(_header(version, created), bytes([version]))
    assert not buffer.observe(_header(5, 0.041), b'dup')            # versions are monotone; repeats are ignored
    got = buffer.deliver(0.046)                                      # cutoff 0.021 -> version 3 (published 0.020)
    assert (got.header.snapshot_version, got.payload, got.fallback) == (3, b'\x03', False)
    assert got.age_ms == pytest.approx(26.0)
    assert len(buffer) == 3 and buffer.fallback_deliveries == 0      # v1 and v2 can never be served again
    assert buffer.deliver(0.054).header.snapshot_version == 3        # v4 (0.030) is not 25 ms old yet
    assert buffer.deliver(0.056).header.snapshot_version == 4
    assert buffer.deliver(1.0).header.snapshot_version == 5 and len(buffer) == 1


def test_buffer_startup_fallback_serves_the_oldest_and_records_it():
    buffer = SnapshotDelayBuffer(1000.0)
    assert buffer.deliver(5.0) is None
    buffer.observe(_header(1, 4.9), b'a'); buffer.observe(_header(2, 4.95), b'b')
    got = buffer.deliver(5.0)
    assert got.header.snapshot_version == 1 and got.fallback and buffer.fallback_deliveries == 1
    still = buffer.deliver(5.5)
    assert still.header.snapshot_version == 1 and still.fallback and buffer.fallback_deliveries == 2
    later = buffer.deliver(5.96)                                     # v2 is now 1010 ms old
    assert later.header.snapshot_version == 2 and not later.fallback and later.age_ms == pytest.approx(1010.0)


def test_buffer_rejects_bad_delays_and_bounds_its_ring():
    for bad in (0, -1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            SnapshotDelayBuffer(bad)
    with pytest.raises(ValueError):
        SnapshotDelayBuffer(1, capacity=0)
    small = SnapshotDelayBuffer(1, capacity=2)
    for version in (1, 2, 3):
        small.observe(_header(version, version / 1000), b'x')
    assert len(small) == 2 and small.deliver(1.0).header.snapshot_version == 3


def _snapshot(version, created_at):
    from vllm.v1.core.sched.state_snapshot import (SchedulerConfigSnapshot, SchedulerKVCacheSnapshot,
                                                    SchedulerParallelSnapshot, SchedulerStateSnapshot)
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheSpec
    return SchedulerStateSnapshot(
        version=version, created_at=created_at, num_running=0, num_waiting=0, running_request_ids=[],
        waiting_request_ids=[], requests={},
        config=SchedulerConfigSnapshot(max_num_batched_tokens=128, max_num_seqs=4, max_model_len=2048,
                                       long_prefill_token_threshold=256, chunked_prefill_enabled=True, policy="fcfs",
                                       batch_time_feature_set="legacy"),
        kv_cache_config=SchedulerKVCacheSnapshot(
            num_gpu_blocks=1024, block_size=16,
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["layer0"], kv_cache_spec=KVCacheSpec(block_size=16))],
            kv_cache_usage=0.0, kv_cache_total_blocks=2048, kv_cache_free_blocks=2048),
        parallel_config=SchedulerParallelSnapshot(decode_context_parallel_size=1),
        resident_set_size=0, waiting_set_size=0, prefill_backlog_running_tokens=0, prefill_backlog_waiting_tokens=0,
        prefill_backlog_total_tokens=0, running_context_length_sum_snapshot=0, decode_reserve_tokens=0,
        inflight_batch=None, build_latency_ms=0.5)


@pytest.fixture
def publisher():
    publisher = SnapshotShmPublisher(name=f"sfs_stale_{uuid.uuid4().hex[:10]}", size_bytes=1 << 20,
                                     simulation_intercept=1.0, simulation_prefill_coeff=0.01, simulation_prefill_sq_coeff=0.0,
                                     simulation_decode_coeff=0.01, simulation_sum_coeff=0.0, simulation_sum_sq_coeff=0.0)
    try:
        yield publisher
    finally:
        publisher.close()


def _publish(publisher, version, created_at):
    from vllm.v1.core.sched.snapshot_serialization import encode_scheduler_state_snapshot
    snapshot = _snapshot(version, created_at)
    assert publisher.publish(snapshot, encode_scheduler_state_snapshot(snapshot))


def test_zero_staleness_keeps_the_direct_watcher_path(publisher):
    client = SnapshotShmClient(shm_name=publisher.name, shm_size_bytes=publisher.size_bytes)
    try:
        assert client.staleness_ms == 0.0 and client._delay is None
        _publish(publisher, 1, time.monotonic())
        assert client.baseline_state().version == 1
        client.start()
        client.set_staleness_ms(0)                                   # unchanged value: nothing is torn down
        assert client._delay is None and client._shm is not None and client._feeder is None
        report = client.estimate(prompt_tokens=None).payload['reports'][0]
        assert report['snapshot_version'] == 1
        assert not any(key.startswith('snapshot_staleness') or key.startswith('snapshot_deliver') for key in report['metadata'])
    finally:
        client.close()
    with pytest.raises(ValueError):
        SnapshotShmClient(shm_name=publisher.name, staleness_ms=-1)


def test_delayed_client_feeds_the_simulator_what_a_delayed_network_delivered(publisher, monkeypatch):
    clock = {'now': 100.0}
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock['now'])
    client = SnapshotShmClient(shm_name=publisher.name, shm_size_bytes=publisher.size_bytes, staleness_ms=100.0)
    monkeypatch.setattr(client, '_start_feeder', lambda: None)      # drive the feed by hand under the fake clock
    try:
        _publish(publisher, 1, 100.0)
        client.start()                                               # observes v1; nothing is 100 ms old yet
        report = client.estimate(prompt_tokens=None).payload['reports'][0]
        metadata = report['metadata']
        assert report['snapshot_version'] == 1 and metadata['snapshot_delivery_fallback'] is True
        assert metadata['snapshot_staleness_ms'] == 100.0 and metadata['snapshot_delivered_version'] == 1
        assert client._delay.fallback_deliveries >= 1
        clock['now'] = 100.05; _publish(publisher, 2, 100.05); client._feed_once()
        clock['now'] = 100.12; _publish(publisher, 3, 100.12); client._feed_once()
        report = client.estimate(prompt_tokens=None).payload['reports'][0]     # cutoff 100.02: still v1, not the live v3
        assert report['snapshot_version'] == 1 and report['metadata']['snapshot_delivery_fallback'] is False
        assert report['metadata']['snapshot_delivered_age_ms'] == pytest.approx(120.0)
        clock['now'] = 100.16; client._feed_once()                   # cutoff 100.06 -> v2 (110 ms old)
        report = client.estimate(prompt_tokens=None).payload['reports'][0]
        assert report['snapshot_version'] == 2 and report['metadata']['snapshot_delivered_version'] == 2
        assert report['metadata']['snapshot_delivered_age_ms'] == pytest.approx(110.0)
        state = client.baseline_state()                              # the Mooncake/RouteBalance read sees the same delivery
        assert state.version == 2 and state.age_ms == pytest.approx(110.0)
        clock['now'] = 100.3; client._feed_once()
        assert client.baseline_state().version == 3 and client.estimate(prompt_tokens=None).payload['reports'][0]['snapshot_version'] == 3
    finally:
        client.close()


def test_reconfiguring_the_delay_restarts_the_transport(publisher):
    client = SnapshotShmClient(shm_name=publisher.name, shm_size_bytes=publisher.size_bytes)
    _publish(publisher, 1, time.monotonic())
    client.start()
    assert client._feeder is None and client._worker is not None
    client.set_staleness_ms(50.0)
    assert client.staleness_ms == 50.0 and client._delay is not None and client._worker is None and client._shm is None
    client.start()
    try:
        assert client._feeder is not None and client._feeder.is_alive()
        client.set_staleness_ms(50.0)                                # same value: keeps running
        assert client._feeder is not None
        client.set_staleness_ms(0.0)
        assert client._delay is None and client._feeder is None
    finally:
        client.close()


def test_instance_client_gates_staleness_on_local_telemetry(publisher):
    from sfs_core.routing.wait_time_scheduler import InstanceClient
    with pytest.raises(ValueError, match='requires local SHM telemetry'):
        InstanceClient(instance_id='a', address='http://h:1', default_model='m', model_id='m', snapshot_staleness_ms=25)
    remote = InstanceClient(instance_id='a', address='http://h:1', default_model='m', model_id='m')
    remote.set_snapshot_staleness_ms(0)
    with pytest.raises(ValueError):
        remote.set_snapshot_staleness_ms(25)
    local = InstanceClient(instance_id='b', address='http://h:2', default_model='m', model_id='m',
                           snapshot_shm_name=publisher.name, snapshot_shm_size_bytes=publisher.size_bytes)
    try:
        assert local.snapshot_staleness_ms == 0.0
        local.set_snapshot_staleness_ms(400)
        assert local.snapshot_staleness_ms == 400.0
    finally:
        local.close()

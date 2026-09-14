"""CPU integration of the actual seqlock transport and baseline parser."""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

import msgspec
import pytest

from sfs_core.routing.snapshot_shm_client import SnapshotShmClient
from vllm.v1.engine.snapshot_shm import SnapshotShmPublisher

from .test_methodology_snapshot import _payload


@pytest.fixture
def transport():
    publisher = SnapshotShmPublisher(
        name=f"baseline_snapshot_test_{os.getpid()}_{time.time_ns()}",
        size_bytes=1024 * 1024,
        simulation_intercept=1.0,
        simulation_prefill_coeff=0.1,
        simulation_prefill_sq_coeff=0.0,
        simulation_decode_coeff=0.2,
        simulation_sum_coeff=0.0,
        simulation_sum_sq_coeff=0.0,
    )
    client = SnapshotShmClient(shm_name=publisher.name, shm_size_bytes=publisher.size_bytes)
    try:
        yield publisher, client
    finally:
        client.close()
        publisher.close()


def _publish(publisher, raw):
    header = SimpleNamespace(
        version=raw["version"],
        created_at=raw["created_at"],
        prefill_backlog_total_tokens=0,
        build_latency_ms=0,
    )
    assert publisher.publish(header, msgspec.msgpack.encode(raw))


def test_baseline_transport_never_starts_or_calls_native_simulator(transport, monkeypatch):
    publisher, client = transport
    monkeypatch.setattr(client, "_ensure_worker", lambda *_: pytest.fail("native worker called"))
    _publish(publisher, _payload())
    state = client.baseline_state(now=10.1)
    assert state.version == 8
    assert state.requests["finishing-prefill"].remaining_prompt_tokens == 4
    assert state.requests["decode"].generated_tokens == 3
    assert state.decode_batch_size == 1
    assert client._worker is None


def test_baseline_transport_reuses_version_and_recomputes_age(transport):
    publisher, client = transport
    raw = _payload()
    _publish(publisher, raw)
    first = client.baseline_state(now=10)
    second = client.baseline_state(now=12)
    assert first.requests == second.requests
    assert first.age_ms == 0
    assert second.age_ms == 2000
    raw["version"] += 1
    raw["created_at"] += 1
    raw["inflight_batch"] = None
    _publish(publisher, raw)
    third = client.baseline_state(now=12)
    assert third.version == 9
    assert third.requests["finishing-prefill"].remaining_prompt_tokens == 0


def test_baseline_transport_rejects_version_regression(transport):
    publisher, client = transport
    raw = _payload()
    _publish(publisher, raw)
    client.baseline_state(now=10)
    raw["version"] -= 1
    _publish(publisher, raw)
    with pytest.raises(RuntimeError, match="regressed"):
        client.baseline_state(now=10)


def test_baseline_transport_checks_unpublished_state(transport):
    _, client = transport
    with pytest.raises(RuntimeError, match="not published"):
        client.baseline_state(now=10)


def test_baseline_transport_rejects_timestamp_change_at_same_version(transport):
    publisher, client = transport
    raw = _payload()
    _publish(publisher, raw)
    client.baseline_state(now=10)
    raw["created_at"] += 1
    _publish(publisher, raw)
    with pytest.raises(RuntimeError, match="timestamp changed"):
        client.baseline_state(now=12)

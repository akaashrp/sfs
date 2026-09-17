"""Transient snapshot-read faults degrade one candidate instead of voiding the request.

The ladder under test: one immediate retry -> this instance's last successful
estimate -> HTTP /wait_time when enabled -> exclude the candidate; only a
decision that loses every candidate still fails.
"""
from __future__ import annotations

import asyncio

import pytest

from sfs_core.routing.snapshot_shm_client import (
    SNAPSHOT_FAULT_INCONSISTENT_HEADER,
    SNAPSHOT_FAULT_KINDS,
    SNAPSHOT_FAULT_NATIVE_TIMEOUT,
    SnapshotEstimate,
    classify_snapshot_read_fault,
)
from sfs_core.routing.wait_time_scheduler import (
    InstanceClient,
    SnapshotFaultCounters,
    SnapshotReadFaultError,
    WaitTimeNotReadyError,
    WaitTimeResult,
    WaitTimeScheduler,
)

NATIVE_TIMEOUT = RuntimeError(
    "Scheduler simulation failed: Timed out waiting for the parsed scheduler "
    "snapshot to catch up"
)
BAD_HEADER = RuntimeError(
    "Failed to read a consistent scheduler snapshot header from SHM sfs_f289bdc0c8_1"
)


def _estimate(wait_ms, version=7):
    return SnapshotEstimate(
        wait_ms=float(wait_ms),
        payload={
            "reports": [
                {
                    "enabled": True,
                    "ready": True,
                    "snapshot_version": version,
                    "metadata": {
                        "estimated_wait_ms": float(wait_ms),
                        "simulation_mode": "critical_path_prefill_done",
                    },
                }
            ]
        },
    )


class FakeSnapshotClient:
    """Stub for the native simulator/SHM reader: one queued outcome per estimate()."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def estimate(self, **kwargs):
        self.calls.append(kwargs)
        if not self.outcomes:
            raise AssertionError("estimate() called more often than the test allows")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _client(instance_id="a", outcomes=(), *, http_fallback=False):
    client = InstanceClient.__new__(InstanceClient)
    client.instance_id = instance_id
    client.address = "http://engine"
    client.default_model = "m"
    client.model_id = "m"
    client._wait_time_url = "http://engine/wait_time"
    client._wait_timeout_s = 2.0
    client._client = None
    client._last_wait = None
    client._last_wait_by_mode = {}
    client._snapshot_client = FakeSnapshotClient(outcomes)
    client._wait_time_http_fallback_enabled = bool(http_fallback)
    client.snapshot_fault_counters = SnapshotFaultCounters()
    return client


def _refresh(client, prompt_tokens=128):
    return asyncio.run(
        client.refresh_wait_time(
            prompt_tokens=prompt_tokens,
            critical_wait_time_timeout_s=0.05,
        )
    )


def _fault_of(result):
    return result.raw_payload["snapshot_read_fault"]


def _scheduler(instances):
    scheduler = WaitTimeScheduler.__new__(WaitTimeScheduler)
    scheduler._instances = instances
    scheduler._enable_wait_time_polling = True
    scheduler._critical_wait_time_timeout_s = 0.05
    scheduler._snapshot_requests_without_estimate = 0
    return scheduler


def _collect(scheduler, records=None, prompt_tokens=128):
    return asyncio.run(
        scheduler._collect_wait_times(
            prompt_tokens=prompt_tokens,
            snapshot_fault_records=records,
        )
    )


def test_classifier_separates_transient_faults_from_corruption():
    assert classify_snapshot_read_fault(NATIVE_TIMEOUT) == SNAPSHOT_FAULT_NATIVE_TIMEOUT
    assert classify_snapshot_read_fault(BAD_HEADER) == SNAPSHOT_FAULT_INCONSISTENT_HEADER
    # Corruption and setup faults keep failing hard.
    for hard in (
        RuntimeError("Baseline scheduler snapshot version regressed; restart the client"),
        RuntimeError("Baseline scheduler snapshot timestamp changed without version increment"),
        ValueError("Shared memory sfs_x has size 1, expected at least 2"),
        FileNotFoundError("No such file or directory: '/dev/shm/sfs_x'"),
    ):
        assert classify_snapshot_read_fault(hard) is None
    # Any other native error degrades this candidate, inside the closed taxonomy.
    other = classify_snapshot_read_fault(RuntimeError("Scheduler simulation failed: boom"))
    assert other in SNAPSHOT_FAULT_KINDS


def test_a_not_ready_instance_is_not_treated_as_a_read_fault():
    client = _client(outcomes=[WaitTimeNotReadyError("no metadata yet")])
    with pytest.raises(WaitTimeNotReadyError):
        _refresh(client)
    assert len(client._snapshot_client.calls) == 1
    assert client.snapshot_fault_counters.transient_faults == 0


def test_native_timeout_then_successful_retry():
    client = _client(outcomes=[NATIVE_TIMEOUT, _estimate(12.0)])
    result = _refresh(client)
    assert result.wait_ms == 12.0
    assert len(client._snapshot_client.calls) == 2
    fault = _fault_of(result)
    assert fault["fault_kind"] == SNAPSHOT_FAULT_NATIVE_TIMEOUT
    assert fault["retried"] and fault["retry_succeeded"] and fault["fallback"] == "retry"
    # The same record reaches per-request wait_time_metadata.
    metadata = result.raw_payload["reports"][0]["metadata"]
    assert metadata["snapshot_read_fault"]["fallback"] == "retry"
    assert metadata["estimated_wait_ms"] == 12.0
    counters = client.snapshot_fault_counters.as_dict()
    assert counters["transient_faults"] == 1 and counters["retries_succeeded"] == 1
    assert counters["cached_fallbacks"] == 0 and counters["candidates_excluded"] == 0
    assert counters["faults_by_kind"] == {SNAPSHOT_FAULT_NATIVE_TIMEOUT: 1}
    # A retry is a real reading, so it refreshes the cache.
    assert client.last_wait.wait_ms == 12.0


def test_timeout_without_retry_success_falls_back_to_the_cached_estimate():
    client = _client(outcomes=[_estimate(5.0), NATIVE_TIMEOUT, NATIVE_TIMEOUT])
    primed = _refresh(client)
    assert primed.wait_ms == 5.0 and "snapshot_read_fault" not in primed.raw_payload

    result = _refresh(client)
    assert result.wait_ms == 5.0
    fault = _fault_of(result)
    assert fault["fault_kind"] == SNAPSHOT_FAULT_NATIVE_TIMEOUT
    assert fault["retry_succeeded"] is False and fault["fallback"] == "cached"
    assert fault["cached_estimate_age_s"] >= 0.0
    counters = client.snapshot_fault_counters.as_dict()
    assert counters["transient_faults"] == 1 and counters["cached_fallbacks"] == 1
    assert counters["retries_succeeded"] == 0
    # The cached reading is reused, not restamped, and is not itself annotated.
    assert client.last_wait is primed
    assert client.last_wait.fetched_at_s == primed.fetched_at_s
    assert "snapshot_read_fault" not in primed.raw_payload


def test_inconsistent_header_falls_back_to_the_cached_estimate():
    client = _client(outcomes=[_estimate(9.0), BAD_HEADER, BAD_HEADER])
    _refresh(client)
    result = _refresh(client)
    assert result.wait_ms == 9.0
    fault = _fault_of(result)
    assert fault["fault_kind"] == SNAPSHOT_FAULT_INCONSISTENT_HEADER
    assert fault["fallback"] == "cached"
    assert "Failed to read a consistent scheduler snapshot header" in fault["error"]
    assert client.snapshot_fault_counters.faults_by_kind == {
        SNAPSHOT_FAULT_INCONSISTENT_HEADER: 1
    }


def test_http_fallback_is_used_when_enabled_and_no_cache_exists(monkeypatch):
    client = _client(outcomes=[BAD_HEADER, BAD_HEADER], http_fallback=True)
    served = WaitTimeResult(
        instance_id="a", wait_ms=31.0, fetched_at_s=1.0,
        raw_payload={"reports": [{"enabled": True, "ready": True,
                                  "metadata": {"estimated_wait_ms": 31.0}}]},
    )
    monkeypatch.setattr(client, "_fetch_http_wait", lambda **kwargs: served)
    result = _refresh(client)
    assert result.wait_ms == 31.0 and _fault_of(result)["fallback"] == "http"
    counters = client.snapshot_fault_counters.as_dict()
    assert counters["http_fallbacks"] == 1 and counters["candidates_excluded"] == 0


def test_exhausted_ladder_excludes_the_candidate_without_failing_the_request():
    faulted = _client("a", [NATIVE_TIMEOUT, NATIVE_TIMEOUT])
    with pytest.raises(SnapshotReadFaultError) as excinfo:
        _refresh(faulted)
    assert excinfo.value.instance_id == "a"
    assert excinfo.value.fault["fallback"] == "excluded"
    assert faulted.snapshot_fault_counters.candidates_excluded == 1

    scheduler = _scheduler({
        "a": _client("a", [BAD_HEADER, BAD_HEADER]),
        "b": _client("b", [_estimate(4.0)]),
        "c": _client("c", [_estimate(6.0)]),
    })
    records: list = []
    results = _collect(scheduler, records)
    assert sorted(results) == ["b", "c"]
    assert results["b"].wait_ms == 4.0 and results["c"].wait_ms == 6.0
    assert [(r["instance_id"], r["fallback"]) for r in records] == [("a", "excluded")]
    summary = scheduler.snapshot_fault_summary()
    assert summary["totals"]["candidates_excluded"] == 1
    assert summary["totals"]["requests_without_estimate"] == 0
    assert summary["totals"]["faults_by_kind"] == {SNAPSHOT_FAULT_INCONSISTENT_HEADER: 1}
    assert summary["per_instance"]["a"]["transient_faults"] == 1
    assert summary["per_instance"]["b"]["transient_faults"] == 0


def test_request_still_fails_when_every_candidate_faults():
    scheduler = _scheduler({
        name: _client(name, [NATIVE_TIMEOUT, NATIVE_TIMEOUT])
        for name in ("a", "b", "c")
    })
    records: list = []
    with pytest.raises(RuntimeError, match="Every candidate lost its wait estimate"):
        _collect(scheduler, records)
    assert sorted(r["instance_id"] for r in records) == ["a", "b", "c"]
    assert all(r["fallback"] == "excluded" for r in records)
    totals = scheduler.snapshot_fault_summary()["totals"]
    assert totals["requests_without_estimate"] == 1
    assert totals["transient_faults"] == 3 and totals["candidates_excluded"] == 3


def test_non_transient_errors_still_fail_the_request():
    scheduler = _scheduler({
        "a": _client("a", [ValueError("Shared memory sfs_x has size 1")]),
        "b": _client("b", [_estimate(4.0)]),
    })
    with pytest.raises(ValueError, match="Shared memory"):
        _collect(scheduler)
    assert scheduler.snapshot_fault_summary()["totals"]["transient_faults"] == 0


def test_normal_path_is_unchanged():
    instances = {
        "a": _client("a", [_estimate(4.0)]),
        "b": _client("b", [_estimate(6.0)]),
    }
    scheduler = _scheduler(instances)
    records: list = []
    results = _collect(scheduler, records)
    assert {name: row.wait_ms for name, row in results.items()} == {"a": 4.0, "b": 6.0}
    # Exactly one snapshot read per instance, no annotation, no counters.
    assert all(len(c._snapshot_client.calls) == 1 for c in instances.values())
    assert records == []
    assert all(
        "snapshot_read_fault" not in row.raw_payload for row in results.values()
    )
    assert all(
        "snapshot_read_fault" not in row.raw_payload["reports"][0]["metadata"]
        for row in results.values()
    )
    assert results["a"] is instances["a"].last_wait
    summary = scheduler.snapshot_fault_summary()
    assert summary["totals"] == {
        "transient_faults": 0, "retries_succeeded": 0, "cached_fallbacks": 0,
        "http_fallbacks": 0, "candidates_excluded": 0, "faults_by_kind": {},
        "requests_without_estimate": 0,
    }

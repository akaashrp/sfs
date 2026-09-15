"""Ministral-only freshness backpressure for event-driven engine snapshots.

Keep the existing age limit and work estimator. A long GPU iteration may make
an otherwise healthy engine's last publication too old for a new decision.
Wait for admissible telemetry before dispatch, with a bounded stall deadline.
The original shared scheduler and Qwen entry points remain unchanged.
"""
import asyncio
import math
import time

from .methodology_scheduler import MethodologyScheduler


class SnapshotNotFresh(RuntimeError):
    def __init__(self, stale):
        self.stale = stale
        super().__init__(f"Awaiting fresh baseline snapshots: {stale}")


class MinistralMethodologyScheduler(MethodologyScheduler):
    def __init__(self, instances, *, snapshot_wait_timeout_s=10.0,
                 snapshot_retry_interval_s=0.025, **kwargs):
        if not instances or any(not c.model_id.startswith('ministral3-') for c in instances.values()):
            raise ValueError('Ministral freshness adapter requires only Ministral instances')
        for value in (snapshot_wait_timeout_s, snapshot_retry_interval_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Snapshot wait and polling intervals must be positive and finite')
        super().__init__(instances, **kwargs)
        self.snapshot_wait_timeout_s = snapshot_wait_timeout_s
        self.snapshot_retry_interval_s = snapshot_retry_interval_s
        self._freshness_deadline = None
        self._freshness = {}
        self._freshness_waits = []

    async def _read_snapshots(self):
        # Bound a hung SHM read as well as an unchanged publication. Cancelling
        # to_thread cannot kill its thread, so a read timeout is terminal.
        timeout = self.snapshot_wait_timeout_s
        if self._freshness_deadline is not None:
            timeout = max(0.001, self._freshness_deadline-time.monotonic())
        values = await asyncio.wait_for(asyncio.gather(*(
            c.refresh_baseline_state() for c in self._instances.values())), timeout)
        snapshots = dict(zip(self._instances, values))
        now = time.monotonic()
        stale = {}
        for key, snapshot in snapshots.items():
            snapshot = snapshots[key] = snapshot.observed_again(now)
            if snapshot.version < self._versions.get(key, -1):
                raise RuntimeError(f'Baseline snapshot version regressed for {key}')
            self._versions[key] = snapshot.version
            reservations = self.lifetime.reservations(key)
            empty = not snapshot.requests and not snapshot.inflight_total_tokens
            idle_unchanged = empty and not reservations
            idle_handoff = empty and reservations and all(
                (time.perf_counter()-self._dispatch_times.get(r.request_id, -math.inf))*1000
                <= self.snapshot_max_age_ms for r in reservations)
            if snapshot.age_ms > self.snapshot_max_age_ms and not (idle_unchanged or idle_handoff):
                stale[key] = snapshot.metadata()
        if stale:
            raise SnapshotNotFresh(stale)
        return snapshots

    async def _dispatch_batch(self, batch):
        started = time.monotonic()
        self._freshness_deadline = started+self.snapshot_wait_timeout_s
        attempts = 0
        initial_stale = {}
        old_batch_id = self._batch_id
        old_sizes = dict(self._batch_sizes)
        try:
            while True:
                self._freshness = {'wait_ms': (time.monotonic()-started)*1000 if attempts else 0.0,
                                   'retries': attempts, 'initial_stale': initial_stale}
                try:
                    # The parent releases its routing lock on SnapshotNotFresh.
                    # In-flight completions can therefore release reservations
                    # while we wait. No request has been reserved/dispatched at
                    # this failure point; retry cannot duplicate a submission.
                    await super()._dispatch_batch(batch)
                    if attempts:
                        self._freshness_waits.append(dict(self._freshness))
                    return
                except SnapshotNotFresh as exc:
                    self._batch_id = old_batch_id
                    self._batch_sizes = dict(old_sizes)
                    if not attempts:
                        initial_stale = exc.stale
                    attempts += 1
                    remaining = self._freshness_deadline-time.monotonic()
                    if remaining <= 0:
                        raise RuntimeError(
                            f'Baseline snapshot freshness stalled for {self.snapshot_wait_timeout_s:g}s; '
                            f'age limit remains {self.snapshot_max_age_ms:g}ms; last state: {exc.stale}'
                        ) from exc
                    await asyncio.sleep(min(self.snapshot_retry_interval_s, remaining))
        finally:
            self._freshness_deadline = None

    def _submit_one(self, queued, record):
        record['methodology_terms']['snapshot_freshness_wait'] = dict(self._freshness)
        super()._submit_one(queued, record)

    def run_metadata(self):
        return {**super().run_metadata(), 'ministral_freshness_adapter_version': 1,
                'snapshot_max_age_ms': self.snapshot_max_age_ms,
                'snapshot_wait_timeout_s': self.snapshot_wait_timeout_s,
                'snapshot_retry_interval_s': self.snapshot_retry_interval_s,
                'snapshot_freshness_waits': self._freshness_waits,
                'snapshot_freshness_semantics': 'wait_before_dispatch; unchanged age gate; delay included in end-to-end TTFT'}

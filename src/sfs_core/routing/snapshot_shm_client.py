"""Local SHM snapshot header reader and native critical-path simulator."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Any, Dict, Optional

from .pending_dispatch_ledger import PendingDispatch
from .methodology_snapshot import BaselineSnapshot, parse_baseline_snapshot
from vllm.v1.engine.snapshot_shm import (
    SnapshotShmHeader,
    read_snapshot_shm_header_once,
    read_snapshot_shm_once,
)

if TYPE_CHECKING:
    from vllm.v1.engine.scheduler_simulator import SimulationStopMode


@dataclass(slots=True)
class SnapshotEstimate:
    wait_ms: float
    payload: Dict[str, Any]
    observed_pending_request_ids: tuple[str, ...] = ()


@dataclass(slots=True)
class DeliveredSnapshot:
    """A publication the router acts on, as a delayed network would have delivered it."""

    header: SnapshotShmHeader
    payload: bytes
    delivered_at: float
    fallback: bool

    @property
    def age_ms(self) -> float:
        return max(0.0, (self.delivered_at - self.header.created_at) * 1000.0)


class SnapshotDelayBuffer:
    """Router-side ring of recent publications; serves the one a fixed one-way delay would deliver.

    ``observe`` records each new publication (monotone versions). ``deliver(now)`` returns the
    newest entry whose publication timestamp is at least ``delay_ms`` older than ``now``; until
    one exists (startup) it serves the oldest available and flags the fallback. Entries older than
    the delivered one are dropped, so later deliveries never move backwards.
    """

    def __init__(self, delay_ms: float, *, capacity: int = 4096) -> None:
        delay = float(delay_ms)
        if not math.isfinite(delay) or delay <= 0.0:
            raise ValueError("snapshot staleness delay must be a positive finite number of ms")
        if int(capacity) < 1:
            raise ValueError("delay buffer capacity must be positive")
        self.delay_ms = delay
        self._capacity = int(capacity)
        self._entries: deque[tuple[SnapshotShmHeader, bytes]] = deque()
        self._lock = threading.Lock()
        self.fallback_deliveries = 0

    def __len__(self) -> int:
        return len(self._entries)

    def observe(self, header: SnapshotShmHeader, payload: bytes) -> bool:
        with self._lock:
            if self._entries and int(header.snapshot_version) <= int(self._entries[-1][0].snapshot_version):
                return False
            self._entries.append((header, bytes(payload)))
            while len(self._entries) > self._capacity:
                self._entries.popleft()
            return True

    def deliver(self, now: float) -> Optional[DeliveredSnapshot]:
        with self._lock:
            if not self._entries:
                return None
            cutoff = float(now) - self.delay_ms / 1000.0
            chosen = None
            for entry in self._entries:
                if float(entry[0].created_at) <= cutoff:
                    chosen = entry
                else:
                    break
            fallback = chosen is None
            if fallback:
                chosen = self._entries[0]
                self.fallback_deliveries += 1
            while self._entries[0] is not chosen:
                self._entries.popleft()
            return DeliveredSnapshot(chosen[0], chosen[1], float(now), fallback)


class SnapshotShmClient:
    """Reads lightweight SHM metadata in Python and parses snapshots natively.

    With ``staleness_ms`` > 0 the native SHM watcher is not started; a router-side feeder polls
    the SHM into a :class:`SnapshotDelayBuffer` and hands the native simulator the publication a
    network with that one-way delay would have delivered. Engine, simulator and policy code are
    unchanged; ``staleness_ms`` = 0 keeps the direct watcher path exactly as before.
    """

    def __init__(
        self,
        *,
        shm_name: str,
        shm_size_bytes: Optional[int] = None,
        staleness_ms: float = 0.0,
    ) -> None:
        self._shm_name = str(shm_name)
        self._shm_size_bytes = (
            int(shm_size_bytes) if shm_size_bytes is not None else None
        )
        self._lock = threading.Lock()
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._worker = None
        self._worker_coefficients: Optional[tuple[float, ...]] = None
        self._baseline_cache: Optional[BaselineSnapshot] = None
        self._staleness_ms = 0.0
        self._delay: Optional[SnapshotDelayBuffer] = None
        self._delivered: Optional[DeliveredSnapshot] = None
        self._fed_version = -1
        self._feed_lock = threading.Lock()
        self._feeder: Optional[threading.Thread] = None
        self._feeder_stop = threading.Event()
        self._configure_staleness(staleness_ms)

    @property
    def staleness_ms(self) -> float:
        return self._staleness_ms

    def set_staleness_ms(self, staleness_ms: float) -> None:
        """Reconfigure the injected delay; an unchanged value is a no-op, otherwise restart on ``start``."""
        if float(staleness_ms) == self._staleness_ms:
            return
        self.close()
        self._configure_staleness(staleness_ms)

    def _configure_staleness(self, staleness_ms: float) -> None:
        delay = float(staleness_ms)
        if not math.isfinite(delay) or delay < 0.0:
            raise ValueError("snapshot staleness must be a finite, nonnegative number of ms")
        self._staleness_ms = delay
        self._delay = SnapshotDelayBuffer(delay) if delay > 0.0 else None
        self._delivered = None
        self._fed_version = -1

    def close(self) -> None:
        self._baseline_cache = None
        self._stop_feeder()
        if self._worker is not None:
            self._worker.stop()
            self._worker = None
            self._worker_coefficients = None
        if self._shm is not None:
            self._shm.close()
            self._shm = None

    def start(self) -> None:
        with self._lock:
            header = self._read_latest_header()
            self._ensure_worker(header)
            if self._delay is not None:
                self._feed_once()
                self._start_feeder()

    def baseline_state(self, *, now: Optional[float] = None) -> BaselineSnapshot:
        """Read observed baseline telemetry without native simulation/overlays.

        The transport's seqlock protects the complete header and payload pair.
        Decode msgpack directly: native snapshot summaries contain SFS output
        reserve heuristics and intentionally do not supply baseline work.
        """
        import msgspec

        with self._lock:
            if self._delay is not None:
                delivered = self._delivered or self._feed_once()
                if delivered is None:
                    raise RuntimeError("Baseline scheduler snapshot is not published yet")
                header, payload = delivered.header, delivered.payload
            else:
                result = read_snapshot_shm_once(self._ensure_shm())
                if result is None:
                    raise RuntimeError("Failed to read consistent baseline scheduler snapshot")
                header, payload = result
            if not payload:
                raise RuntimeError("Baseline scheduler snapshot is not published yet")
            observed_at = time.monotonic() if now is None else float(now)
            cached = self._baseline_cache
            if cached is not None and header.snapshot_version < cached.version:
                raise RuntimeError("Baseline scheduler snapshot version regressed; restart the client")
            if cached is not None and header.snapshot_version == cached.version:
                if header.created_at != cached.created_at:
                    raise RuntimeError("Baseline scheduler snapshot timestamp changed without version increment")
                return cached.observed_again(observed_at)
            state = parse_baseline_snapshot(
                msgspec.msgpack.decode(payload),
                observed_at=observed_at,
                expected_version=int(header.snapshot_version),
                expected_created_at=float(header.created_at),
            )
            self._baseline_cache = state
            return state

    def estimate(
        self,
        *,
        prompt_tokens: Optional[int],
        stop_mode: Optional[SimulationStopMode | str] = None,
        pending_dispatches: tuple[PendingDispatch, ...] = (),
        probe_ready_delay_ms: float = 0.0,
        catchup_timeout_s: float = 2.0,
    ) -> SnapshotEstimate:
        with self._lock:
            delivered: Optional[DeliveredSnapshot] = None
            if self._delay is not None:
                delivered = self._delivered or self._feed_once()
                if delivered is None:
                    raise RuntimeError("Baseline scheduler snapshot is not published yet")
                header = delivered.header
            else:
                header = self._read_latest_header()
            self._ensure_worker(header)
            minimum_snapshot_version = int(header.snapshot_version)
            timeout_ms = max(1, int(float(catchup_timeout_s) * 1000.0))
            native_pending = tuple(
                reservation.as_native_tuple() for reservation in pending_dispatches
            )

            num_requests = 0
            build_latency_ms = float(header.build_latency_ms)
            snapshot_version = int(header.snapshot_version)
            snapshot_timestamp = float(header.created_at)

            metadata: Dict[str, Any] = {
                "prefill_backlog_total_tokens": float(
                    header.prefill_backlog_total_tokens
                ),
            }
            simulation_timestamp = time.monotonic()
            simulation_latency_ms = 0.0
            simulation_mode = "snapshot_only"
            wait_ms = 0.0
            observed_pending_request_ids: tuple[str, ...] = ()

            if prompt_tokens is not None:
                from vllm.v1.engine.scheduler_simulator import SimulationStopMode

                resolved_stop_mode = SimulationStopMode.from_value(
                    stop_mode,
                    default=SimulationStopMode.PREFILL_DONE,
                )
                summary = self._worker.run_simulation_on_latest_snapshot(
                    int(prompt_tokens),
                    resolved_stop_mode.value,
                    native_pending,
                    minimum_snapshot_version,
                    timeout_ms,
                    float(probe_ready_delay_ms),
                )
                if summary is None:
                    raise RuntimeError(
                        "Local scheduler simulation has no parsed snapshot yet."
                    )
                (
                    snapshot_version,
                    snapshot_timestamp,
                    simulation_timestamp,
                    num_requests,
                    _parsed_build_latency_ms,
                    simulation_latency_ms,
                    native_metadata,
                    observed_ids,
                ) = summary
                metadata.update(dict(native_metadata))
                observed_pending_request_ids = tuple(observed_ids)
                simulation_mode = f"critical_path_{resolved_stop_mode.value}"
                wait_ms = float(metadata.get("estimated_wait_ms", 0.0))
            else:
                summary = self._worker.snapshot_summary_on_latest_snapshot(
                    native_pending,
                    minimum_snapshot_version,
                    timeout_ms,
                )
                if summary is None:
                    raise RuntimeError(
                        "Local scheduler simulation has no parsed snapshot yet."
                    )
                (
                    snapshot_version,
                    snapshot_timestamp,
                    num_requests,
                    _parsed_build_latency_ms,
                    native_metadata,
                    observed_ids,
                ) = summary
                metadata.update(dict(native_metadata))
                observed_pending_request_ids = tuple(observed_ids)

            build_latency_ms = float(_parsed_build_latency_ms)

            metadata["simulation_mode"] = simulation_mode
            if delivered is not None:
                # Staleness injection diagnostics: what the simulator acted on.
                metadata["snapshot_staleness_ms"] = float(self._staleness_ms)
                metadata["snapshot_delivered_version"] = int(snapshot_version)
                metadata["snapshot_delivered_age_ms"] = max(
                    0.0, (time.monotonic() - float(snapshot_timestamp)) * 1000.0
                )
                metadata["snapshot_delivery_fallback"] = bool(delivered.fallback)
            report = {
                "enabled": True,
                "ready": True,
                "snapshot_version": int(snapshot_version),
                "snapshot_timestamp": float(snapshot_timestamp),
                "simulation_timestamp": float(simulation_timestamp),
                "num_requests": int(num_requests),
                "metadata": metadata,
                "snapshot_build_latency_ms": float(build_latency_ms),
                "simulation_latency_ms": float(simulation_latency_ms),
            }
            return SnapshotEstimate(
                wait_ms=float(wait_ms),
                payload={"reports": [report]},
                observed_pending_request_ids=observed_pending_request_ids,
            )

    def _ensure_shm(self) -> shared_memory.SharedMemory:
        if self._shm is None:
            self._shm = shared_memory.SharedMemory(name=self._shm_name)
            if (
                self._shm_size_bytes is not None
                and self._shm.size < self._shm_size_bytes
            ):
                raise ValueError(
                    f"Shared memory {self._shm_name} has size {self._shm.size}, "
                    f"expected at least {self._shm_size_bytes}"
                )
        return self._shm

    def _read_latest_header(self) -> SnapshotShmHeader:
        shm = self._ensure_shm()
        header = read_snapshot_shm_header_once(shm)
        if header is None:
            raise RuntimeError(
                f"Failed to read a consistent scheduler snapshot header from SHM "
                f"{self._shm_name}"
            )
        return header

    def _ensure_worker(self, header: SnapshotShmHeader) -> None:
        from vllm.v1.engine import _scheduler_sim as _scheduler_sim_native

        coefficients = (
            float(header.simulation_intercept),
            float(header.simulation_prefill_coeff),
            float(header.simulation_prefill_sq_coeff),
            float(header.simulation_decode_coeff),
            float(header.simulation_sum_coeff),
            float(header.simulation_sum_sq_coeff),
        )
        if self._worker is not None and coefficients == self._worker_coefficients:
            return
        if self._worker is not None:
            self._worker.stop()
        self._worker = _scheduler_sim_native.SchedulerSimulationWorker(
            interval_s=0.01,
            intercept=coefficients[0],
            prefill_coeff=coefficients[1],
            decode_coeff=coefficients[3],
            sum_coeff=coefficients[4],
            prefill_sq_coeff=coefficients[2],
            sum_sq_coeff=coefficients[5],
        )
        if self._delay is None:
            self._worker.start_snapshot_shm_watcher(
                str(self._shm_name),
                int(self._shm_size_bytes or 0),
                1,
            )
        else:
            # The delay feeder replaces the watcher: it hands the worker the
            # delayed publication itself, so a replaced worker is refilled.
            self._fed_version = -1
        self._worker_coefficients = coefficients

    def _feed_once(self) -> Optional[DeliveredSnapshot]:
        """Observe the newest publication and hand the delayed one to the native worker."""
        assert self._delay is not None
        with self._feed_lock:
            result = read_snapshot_shm_once(self._ensure_shm())
            if result is not None and result[1]:
                self._delay.observe(result[0], result[1])
            delivered = self._delay.deliver(time.monotonic())
            if delivered is None:
                return None
            worker = self._worker
            version = int(delivered.header.snapshot_version)
            if worker is not None and version != self._fed_version:
                worker.update_snapshot(delivered.payload)
                self._fed_version = version
            self._delivered = delivered
            return delivered

    def _start_feeder(self, poll_interval_s: float = 0.001) -> None:
        if self._feeder is not None:
            return
        self._feeder_stop.clear()

        def run() -> None:
            while not self._feeder_stop.wait(poll_interval_s):
                try:
                    self._feed_once()
                except Exception:  # keep feeding; the reader reports errors
                    continue

        self._feeder = threading.Thread(
            target=run, name=f"snapshot-delay-feeder-{self._shm_name}", daemon=True
        )
        self._feeder.start()

    def _stop_feeder(self) -> None:
        feeder = self._feeder
        if feeder is not None:
            self._feeder_stop.set()
            feeder.join(timeout=5.0)
            self._feeder = None
        self._delivered = None
        self._fed_version = -1
        if self._delay is not None:
            self._delay = SnapshotDelayBuffer(self._staleness_ms)

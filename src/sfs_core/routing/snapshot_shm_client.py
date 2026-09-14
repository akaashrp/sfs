"""Local SHM snapshot header reader and native critical-path simulator."""

from __future__ import annotations

import threading
import time
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


class SnapshotShmClient:
    """Reads lightweight SHM metadata in Python and parses snapshots natively."""

    def __init__(
        self,
        *,
        shm_name: str,
        shm_size_bytes: Optional[int] = None,
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

    def close(self) -> None:
        self._baseline_cache = None
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

    def baseline_state(self, *, now: Optional[float] = None) -> BaselineSnapshot:
        """Read observed baseline telemetry without native simulation/overlays.

        The transport's seqlock protects the complete header and payload pair.
        Decode msgpack directly: native snapshot summaries contain SFS output
        reserve heuristics and intentionally do not supply baseline work.
        """
        import msgspec

        with self._lock:
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
        self._worker.start_snapshot_shm_watcher(
            str(self._shm_name),
            int(self._shm_size_bytes or 0),
            1,
        )
        self._worker_coefficients = coefficients

"""Scheduler that routes requests based on live /wait_time outputs."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import random
import time
from functools import partial
from dataclasses import dataclass, field
from urllib.parse import urlencode
from typing import Any, Callable, Dict, Optional

from matplotlib import text
import requests
from openai import AsyncOpenAI
from vllm.v1.engine.accuracy_predictor import AccuracyPredictor, AdmissionFeatures
from vllm.v1.engine.output_length_predictor import (
    OutputLengthPredictor,
    PromptFeatureContext,
)
from vllm.v1.engine.scheduler_simulator import SimulationStopMode

from transformers import AutoTokenizer
import sys

from .snapshot_shm_client import SnapshotShmClient
from .pending_dispatch_ledger import PendingDispatch, PendingDispatchLedger
from .readiness_predictor import ReadinessDelayPredictor

LOGGER = logging.getLogger(__name__)

def load_tokenizer(model_or_path: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_or_path, use_fast=True)
        return tokenizer
    except Exception as exc:
        print(
            f"[WARN] Failed to load tokenizer '{model_or_path}': {exc}. "
            "Falling back to whitespace token counting.",
            file=sys.stderr,
        )
        return None

class WaitTimeNotReadyError(RuntimeError):
    """Raised when an instance has not produced wait-time metadata yet."""


@dataclass(slots=True)
class WaitTimeResult:
    """Last known wait time for an instance."""

    instance_id: str
    wait_ms: float
    fetched_at_s: float
    raw_payload: Dict[str, Any] = field(default_factory=dict)
    observed_pending_request_ids: tuple[str, ...] = field(
        default=(),
        repr=False,
    )


UtilityCallable = Callable[
    [str, Dict[str, WaitTimeResult], Dict[str, float], Dict[str, float], int],
    float,
]


@dataclass(slots=True)
class RoutedRequest:
    """Metadata returned after a request is routed."""

    request_id: str
    instance_id: str
    wait_time_ms: Optional[float]
    wait_time_details: Optional[Dict[str, Any]]


@dataclass(slots=True)
class _QueuedRequest:
    request_id: str
    payload: Dict[str, Any]
    result_future: asyncio.Future
    enqueued_at_s: float


class InstanceClient:
    """Tracks per-instance state and provides async submission helpers."""

    def __init__(
        self,
        *,
        instance_id: str,
        address: str,
        default_model: str,
        model_id: str,
        wait_time_timeout_s: float = 2.0,
        snapshot_shm_name: Optional[str] = None,
        snapshot_shm_size_bytes: Optional[int] = None,
        wait_time_http_fallback_enabled: bool = False,
    ) -> None:
        self.instance_id = instance_id
        self.address = address.rstrip("/")
        self.default_model = default_model
        self.model_id = model_id
        self._wait_time_url = f"{self.address}/wait_time"
        self._wait_timeout_s = wait_time_timeout_s
        self._client = AsyncOpenAI(
            api_key="EMPTY",
            base_url=f"{self.address}/v1",
            timeout=10000,
        )
        self._last_wait: Optional[WaitTimeResult] = None
        self._last_wait_by_mode: Dict[str, WaitTimeResult] = {}
        self._snapshot_client = (
            SnapshotShmClient(
                shm_name=str(snapshot_shm_name),
                shm_size_bytes=snapshot_shm_size_bytes,
            )
            if snapshot_shm_name
            else None
        )
        self._wait_time_http_fallback_enabled = bool(wait_time_http_fallback_enabled)

    async def refresh_wait_time(
        self,
        *,
        prompt_tokens: Optional[int] = None,
        critical_wait_time_timeout_s: Optional[float] = None,
        stop_mode: Optional[SimulationStopMode | str] = None,
        pending_dispatches: tuple[PendingDispatch, ...] = (),
        probe_ready_delay_ms: float = 0.0,
    ) -> Optional[WaitTimeResult]:
        """Fetch latest wait time from the instance and cache it."""
        cache_key = self._wait_cache_key(
            prompt_tokens=prompt_tokens,
            stop_mode=stop_mode,
        )
        prompt_timeout = (
            self._wait_timeout_s
            if critical_wait_time_timeout_s is None
            else float(critical_wait_time_timeout_s)
        )

        result: Optional[WaitTimeResult] = None
        if self._snapshot_client is not None:
            result = await asyncio.to_thread(
                self._fetch_local_wait,
                prompt_tokens=prompt_tokens,
                stop_mode=stop_mode,
                pending_dispatches=pending_dispatches,
                probe_ready_delay_ms=probe_ready_delay_ms,
                catchup_timeout_s=prompt_timeout,
            )

        if result is None and (
            self._snapshot_client is None or self._wait_time_http_fallback_enabled
        ):
            result = await asyncio.to_thread(
                self._fetch_http_wait,
                prompt_tokens=prompt_tokens,
                timeout_s=prompt_timeout,
                stop_mode=stop_mode,
            )

        if result:
            self._last_wait = result
            self._last_wait_by_mode[cache_key] = result
        if self._snapshot_client is not None:
            if result is None:
                raise WaitTimeNotReadyError(
                    f"Local SHM wait estimate is unavailable for "
                    f"{self.instance_id}"
                )
            return result
        return self._last_wait_by_mode.get(cache_key) or self._last_wait

    @property
    def last_wait(self) -> Optional[WaitTimeResult]:
        return self._last_wait

    def last_wait_for_mode(
        self,
        *,
        prompt_tokens: Optional[int] = None,
        stop_mode: Optional[SimulationStopMode | str] = None,
    ) -> Optional[WaitTimeResult]:
        cache_key = self._wait_cache_key(
            prompt_tokens=prompt_tokens,
            stop_mode=stop_mode,
        )
        return self._last_wait_by_mode.get(cache_key) or self._last_wait

    def close(self) -> None:
        if self._snapshot_client is not None:
            self._snapshot_client.close()
            self._snapshot_client = None

    def prime_wait_source(self) -> None:
        if self._snapshot_client is None:
            return
        self._snapshot_client.start()

    def _wait_cache_key(
        self,
        *,
        prompt_tokens: Optional[int],
        stop_mode: Optional[SimulationStopMode | str],
    ) -> str:
        if prompt_tokens is None:
            return "snapshot_only"
        resolved = SimulationStopMode.from_value(
            stop_mode,
            default=SimulationStopMode.PREFILL_DONE,
        )
        return resolved.value

    def _build_wait_result(
        self,
        payload: Dict[str, Any],
        *,
        observed_pending_request_ids: tuple[str, ...] = (),
    ) -> Optional[WaitTimeResult]:
        try:
            wait_ms = self._parse_wait_ms(payload)
        except WaitTimeNotReadyError:
            LOGGER.debug(
                "Wait time metadata not ready for %s",
                self.instance_id,
            )
            return None
        except Exception as exc:
            reports = payload.get("reports")
            enabled = True
            if isinstance(reports, list) and reports and isinstance(reports[0], dict):
                enabled = bool(reports[0].get("enabled", True))
            if enabled:
                LOGGER.warning(
                    "Failed to parse wait time for %s: %s. Payload: %s.",
                    self.instance_id,
                    exc,
                    payload,
                )
            return None
        return WaitTimeResult(
            instance_id=self.instance_id,
            wait_ms=wait_ms,
            fetched_at_s=time.time(),
            raw_payload=payload,
            observed_pending_request_ids=observed_pending_request_ids,
        )

    def _fetch_local_wait(
        self,
        *,
        prompt_tokens: Optional[int],
        stop_mode: Optional[SimulationStopMode | str],
        pending_dispatches: tuple[PendingDispatch, ...],
        probe_ready_delay_ms: float,
        catchup_timeout_s: float,
    ) -> Optional[WaitTimeResult]:
        if self._snapshot_client is None:
            return None
        estimate = self._snapshot_client.estimate(
            prompt_tokens=prompt_tokens,
            stop_mode=stop_mode,
            pending_dispatches=pending_dispatches,
            probe_ready_delay_ms=probe_ready_delay_ms,
            catchup_timeout_s=catchup_timeout_s,
        )
        return self._build_wait_result(
            estimate.payload,
            observed_pending_request_ids=(estimate.observed_pending_request_ids),
        )

    def _fetch_http_wait(
        self,
        *,
        prompt_tokens: Optional[int],
        timeout_s: float,
        stop_mode: Optional[SimulationStopMode | str],
    ) -> Optional[WaitTimeResult]:
        def _fetch_once(
            *,
            prompt_tokens_override: Optional[int],
            request_timeout_s: float,
            request_stop_mode: Optional[SimulationStopMode | str],
        ) -> Optional[WaitTimeResult]:
            params: Dict[str, str] = {"include_timings": "true"}
            if prompt_tokens_override is not None:
                params["prompt_tokens"] = str(int(prompt_tokens_override))
                if request_stop_mode is not None:
                    params["stop_mode"] = SimulationStopMode.from_value(
                        request_stop_mode,
                        default=SimulationStopMode.PREFILL_DONE,
                    ).value
            query = urlencode(params)
            url = f"{self._wait_time_url}?{query}"
            resp = requests.get(url, timeout=request_timeout_s)
            resp.raise_for_status()
            return self._build_wait_result(resp.json())

        try:
            if prompt_tokens is not None:
                try:
                    prompt_result = _fetch_once(
                        prompt_tokens_override=prompt_tokens,
                        request_timeout_s=timeout_s,
                        request_stop_mode=stop_mode,
                    )
                except requests.Timeout:
                    LOGGER.debug(
                        "Prompt-aware /wait_time timed out for %s; "
                        "falling back to cached estimate",
                        self.instance_id,
                    )
                else:
                    if prompt_result is not None:
                        return prompt_result

            return _fetch_once(
                prompt_tokens_override=None,
                request_timeout_s=self._wait_timeout_s,
                request_stop_mode=None,
            )
        except requests.Timeout:
            LOGGER.warning(
                "Timed out fetching wait time for %s",
                self.instance_id,
            )
            return None
        except Exception as exc:
            LOGGER.warning(
                "Failed to fetch wait time for %s: %s",
                self.instance_id,
                exc,
            )
            return None

    async def submit_request(self, **payload: Any) -> Any:
        """Submit a request asynchronously."""
        args = dict(payload)
        args.setdefault("model", self.default_model)
        if "messages" in args:
            return await self._client.chat.completions.create(**args)
        return await self._client.completions.create(**args)

    def _parse_wait_ms(self, payload: Dict[str, Any]) -> float:
        """Extract wait time from the wait_time endpoint payload."""
        reports = payload.get("reports")
        if not isinstance(reports, list) or not reports:
            raise ValueError("wait_time response missing reports")

        not_ready = False
        for report in reports:
            if not isinstance(report, dict):
                continue
            if not report.get("enabled", True):
                raise RuntimeError(
                    f"Wait time simulation disabled on instance {self.instance_id}"
                )

            metadata = report.get("metadata")
            if isinstance(metadata, dict) and "estimated_wait_ms" in metadata:
                return float(metadata["estimated_wait_ms"])

            if report.get("ready", False):
                return 0.0

            not_ready = True

        if not_ready:
            raise WaitTimeNotReadyError(
                f"Instance {self.instance_id} has not produced wait-time metadata"
            )

        raise ValueError("No wait time metadata available in response")


class WaitTimeScheduler:
    """Routes requests to the instance with the highest utility score."""

    def __init__(
        self,
        instances: Dict[str, InstanceClient],
        *,
        request_log_path: str | None = None,
        response_map_path: str | None = None,
        defer_sidecar_writes: bool = False,
        worker_count: int = 4,
        max_queue_size: int = 0,
        accuracy_model_path: Optional[str] = None,
        output_length_model_path: Optional[str] = None,
        lambda_weight: float = 0.0,
        delta_weight: float = 1.0,
        instance_costs: Optional[Dict[str, Dict[str, float]]] = None,
        utility_fn: Optional[UtilityCallable] = None,
        tokenizer_id: str = "Qwen/Qwen3-0.6B",
        enable_wait_time_polling: bool = True,
        critical_wait_time_timeout_s: float = 0.05,
        route_random_seed: Optional[int] = None,
        readiness_predictor_path: Optional[str] = None,
    ) -> None:
        if not instances:
            raise ValueError("At least one instance must be provided")
        self._instances = instances
        self._request_log: Dict[str, WaitTimeResult] = {}
        self._log_path = request_log_path
        self._response_map_path = response_map_path
        self._defer_sidecar_writes = bool(defer_sidecar_writes)
        self._deferred_wait_log_lines: list[str] = []
        self._deferred_response_map_lines: list[str] = []
        self._response_map_tasks: set[asyncio.Task] = set()
        self._worker_count = worker_count
        self._queue: asyncio.Queue[_QueuedRequest] = asyncio.Queue(maxsize=max_queue_size)
        self._workers: list[asyncio.Task] = []
        self._stop_event = asyncio.Event()
        self._submit_tasks: set[asyncio.Task] = set()
        self._accuracy_predictor = (
            AccuracyPredictor(accuracy_model_path) if accuracy_model_path else None
        )
        self._output_length_predictor = (
            OutputLengthPredictor(output_length_model_path)
            if output_length_model_path
            else None
        )
        self._lambda = float(lambda_weight)
        self._delta = float(delta_weight)
        self._instance_costs = instance_costs or {}
        self._utility_fn = utility_fn
        self._tokenizer = load_tokenizer(tokenizer_id)
        self._enable_wait_time_polling = bool(enable_wait_time_polling)
        self._critical_wait_time_timeout_s = float(critical_wait_time_timeout_s)
        self._route_rng = random.Random(route_random_seed)
        self._readiness_predictor = (
            ReadinessDelayPredictor.load(readiness_predictor_path)
            if readiness_predictor_path
            else None
        )
        self._routing_state_lock = asyncio.Lock()
        self._pending_dispatch_ledger = PendingDispatchLedger(tuple(instances.keys()))

    async def start(self) -> None:
        """Start background workers if they are not already running."""
        if self._workers:
            return
        for idx in range(self._worker_count):
            self._workers.append(
                asyncio.create_task(self._worker_loop(), name=f"wait-time-worker-{idx}")
            )
        prime_tasks = [
            asyncio.to_thread(instance.prime_wait_source)
            for instance in self._instances.values()
        ]
        if prime_tasks:
            results = await asyncio.gather(*prime_tasks, return_exceptions=True)
            for instance_id, result in zip(self._instances.keys(), results):
                if isinstance(result, Exception):
                    LOGGER.debug(
                        "Failed to prime local wait source for %s: %s",
                        instance_id,
                        result,
                    )

    async def stop(self, *, close_instances: bool = True) -> None:
        """Stop workers and wait for queued dispatches to finish."""
        self._stop_event.set()
        await self._queue.join()
        for worker in self._workers:
            worker.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        if self._submit_tasks:
            await asyncio.gather(*self._submit_tasks, return_exceptions=True)
        await self._wait_for_response_map_tasks()
        self._workers.clear()
        self._submit_tasks.clear()
        self._response_map_tasks.clear()
        if close_instances:
            close_tasks = [
                asyncio.to_thread(instance.close)
                for instance in self._instances.values()
            ]
            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)

    async def route_and_submit(
        self, request_id: str, *, await_dispatch: bool = False, **payload: Any
    ) -> RoutedRequest | str:
        """
        Enqueue a request for routing. Returns immediately after queueing unless
        await_dispatch=True, in which case it waits until the request has been
        dispatched (but not completed) and returns routing metadata.
        """
        if not self._workers:
            await self.start()

        loop = asyncio.get_running_loop()
        result_future: asyncio.Future = loop.create_future()
        queued = _QueuedRequest(
            request_id=request_id,
            payload=dict(payload),
            result_future=result_future,
            enqueued_at_s=time.time(),
        )
        try:
            self._queue.put_nowait(queued)
        except asyncio.QueueFull:
            await self._queue.put(queued)

        if await_dispatch:
            return await result_future
        return request_id

    async def drain(self) -> None:
        """Wait for all queued requests to be dispatched (not completed)."""
        await self._queue.join()
        if self._submit_tasks:
            await asyncio.gather(*self._submit_tasks, return_exceptions=True)
        await self._wait_for_response_map_tasks()

    async def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                queued = await asyncio.wait_for(self._queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            try:
                await self._dispatch(queued)
            except Exception as exc:
                LOGGER.warning(
                    "Failed to dispatch request %s: %s", queued.request_id, exc
                )
                if not queued.result_future.done():
                    queued.result_future.set_exception(exc)
            finally:
                self._queue.task_done()

    async def _dispatch(self, queued: _QueuedRequest) -> None:
        prompt_text = self._extract_prompt_text(queued.payload)
        prompt_tokens = self._extract_precomputed_prompt_tokens(queued.payload)
        if prompt_tokens is None:
            LOGGER.warning(
                "Falling back to tokenizer-based prompt tokenization for request_id=%s.",
                queued.request_id,
            )
            prompt_tokens = self._get_prompt_tokens(queued.payload, prompt_text)
        completion_cap = self._extract_completion_cap(queued.payload)
        accuracy_scores, output_lengths = await self._predict_model_scores(
            prompt_text=prompt_text,
            prompt_tokens=prompt_tokens,
            completion_cap=completion_cap,
        )
        engine_request_id = self._attach_engine_request_id(
            queued.payload,
            queued.request_id,
        )
        async with self._routing_state_lock:
            pending_dispatches_by_instance = self._pending_dispatches_by_instance()
            probe_ready_delay_ms_by_instance = (
                self._probe_ready_delays_by_instance(
                    prompt_tokens=prompt_tokens,
                    pending_dispatches_by_instance=(
                        pending_dispatches_by_instance
                    ),
                )
            )
            wait_results = await self._collect_wait_times(
                prompt_tokens=prompt_tokens if prompt_tokens > 0 else None,
                pending_dispatches_by_instance=pending_dispatches_by_instance,
                probe_ready_delay_ms_by_instance=probe_ready_delay_ms_by_instance,
            )
            self._reconcile_observed_dispatches(wait_results)
            target_id = self._select_instance(
                wait_results, accuracy_scores, output_lengths, prompt_tokens
            )
            self._reserve_pending_dispatch(
                instance_id=target_id,
                engine_request_id=engine_request_id,
                prompt_tokens=prompt_tokens,
                predicted_output_tokens=output_lengths.get(target_id, 1.0),
                completion_cap=completion_cap,
                predicted_ready_at_s=self._predicted_ready_at_s(
                    wait_record=wait_results.get(target_id),
                    delay_ms=probe_ready_delay_ms_by_instance.get(
                        target_id,
                        0.0,
                    ),
                ),
            )
        target = self._instances[target_id]
        wait_record = wait_results.get(target_id) or target.last_wait_for_mode(
            prompt_tokens=prompt_tokens if prompt_tokens > 0 else None,
        )
        submit_task = asyncio.create_task(target.submit_request(**queued.payload))
        submit_task.add_done_callback(
            partial(
                self._release_pending_dispatch_on_done,
                instance_id=target_id,
                engine_request_id=engine_request_id,
            )
        )
        self._submit_tasks.add(submit_task)
        submit_task.add_done_callback(self._submit_tasks.discard)
        if self._response_map_path:
            submit_task.add_done_callback(
                partial(
                    self._handle_response_mapping,
                    request_id=queued.request_id,
                    instance_id=target_id,
                )
            )
        if wait_record:
            self._request_log[queued.request_id] = wait_record
            await self._log_to_file(queued.request_id, wait_record)

        result = RoutedRequest(
            request_id=queued.request_id,
            instance_id=target_id,
            wait_time_ms=wait_record.wait_ms if wait_record else None,
            wait_time_details=wait_record.raw_payload if wait_record else None,
        )
        if not queued.result_future.done():
            queued.result_future.set_result(result)

    async def _collect_wait_times(
        self,
        prompt_tokens: Optional[int] = None,
        stop_mode: Optional[SimulationStopMode | str] = None,
        pending_dispatches_by_instance: Optional[
            Dict[str, tuple[PendingDispatch, ...]]
        ] = None,
        probe_ready_delay_ms_by_instance: Optional[Dict[str, float]] = None,
    ) -> Dict[str, WaitTimeResult]:
        if not self._enable_wait_time_polling:
            now_s = time.time()
            return {
                instance_id: WaitTimeResult(
                    instance_id=instance_id,
                    wait_ms=0.0,
                    fetched_at_s=now_s,
                    raw_payload={
                        "wait_time_polling_disabled": True,
                    },
                )
                for instance_id, instance in self._instances.items()
            }

        tasks = {
            instance_id: asyncio.create_task(
                instance.refresh_wait_time(
                    prompt_tokens=prompt_tokens,
                    critical_wait_time_timeout_s=self._critical_wait_time_timeout_s,
                    stop_mode=stop_mode,
                    pending_dispatches=(pending_dispatches_by_instance or {}).get(
                        instance_id, ()
                    ),
                    probe_ready_delay_ms=(
                        probe_ready_delay_ms_by_instance or {}
                    ).get(instance_id, 0.0),
                )
            )
            for instance_id, instance in self._instances.items()
        }
        results: Dict[str, WaitTimeResult] = {}
        for instance_id, task in tasks.items():
            result = await task
            if result:
                results[instance_id] = result
        return results

    def _pending_dispatches_by_instance(
        self,
    ) -> Dict[str, tuple[PendingDispatch, ...]]:
        return {
            instance_id: self._pending_dispatch_ledger.unobserved_for_instance(
                instance_id
            )
            for instance_id in self._instances
        }

    def _probe_ready_delays_by_instance(
        self,
        *,
        prompt_tokens: int,
        pending_dispatches_by_instance: Dict[
            str,
            tuple[PendingDispatch, ...],
        ],
    ) -> Dict[str, float]:
        predictor = getattr(self, "_readiness_predictor", None)
        if predictor is None:
            return {}
        return {
            instance_id: predictor.predict_ms(
                prompt_tokens=prompt_tokens,
                pending_dispatch_count=len(
                    pending_dispatches_by_instance.get(instance_id, ())
                ),
            )
            for instance_id in self._instances
        }

    @staticmethod
    def _predicted_ready_at_s(
        *,
        wait_record: Optional[WaitTimeResult],
        delay_ms: float,
    ) -> float:
        if delay_ms <= 0.0:
            return 0.0
        simulation_timestamp: Optional[float] = None
        if wait_record is not None:
            reports = wait_record.raw_payload.get("reports")
            if isinstance(reports, list) and reports:
                report = reports[0]
                if isinstance(report, dict):
                    value = report.get("simulation_timestamp")
                    if isinstance(value, (int, float)):
                        simulation_timestamp = float(value)
        if simulation_timestamp is None:
            simulation_timestamp = time.monotonic()
        return simulation_timestamp + float(delay_ms) / 1000.0

    def _reconcile_observed_dispatches(
        self,
        wait_results: Dict[str, WaitTimeResult],
    ) -> None:
        for instance_id, wait_result in wait_results.items():
            self._pending_dispatch_ledger.mark_observed(
                instance_id,
                wait_result.observed_pending_request_ids,
            )

    def _reserve_pending_dispatch(
        self,
        *,
        instance_id: str,
        engine_request_id: str,
        prompt_tokens: int,
        predicted_output_tokens: float,
        completion_cap: Optional[float],
        predicted_ready_at_s: float = 0.0,
    ) -> None:
        resolved_cap = (
            max(1, int(math.ceil(completion_cap)))
            if completion_cap is not None
            else 2**31 - 1
        )
        self._pending_dispatch_ledger.reserve(
            instance_id,
            PendingDispatch(
                engine_request_id=engine_request_id,
                prompt_tokens=max(0, int(prompt_tokens)),
                predicted_output_tokens=max(
                    1.0,
                    float(predicted_output_tokens),
                ),
                completion_cap=resolved_cap,
                predicted_ready_at_s=max(
                    0.0,
                    float(predicted_ready_at_s),
                ),
            ),
        )

    def _release_pending_dispatch_on_done(
        self,
        _task: asyncio.Task,
        *,
        instance_id: str,
        engine_request_id: str,
    ) -> None:
        asyncio.create_task(
            self._release_pending_dispatch(
                instance_id,
                engine_request_id,
            )
        )

    async def _release_pending_dispatch(
        self,
        instance_id: str,
        engine_request_id: str,
    ) -> None:
        async with self._routing_state_lock:
            self._pending_dispatch_ledger.release(
                instance_id,
                engine_request_id,
            )

    @staticmethod
    def _attach_engine_request_id(
        payload: Dict[str, Any],
        router_request_id: str,
    ) -> str:
        raw_extra_body = payload.get("extra_body")
        if raw_extra_body is None:
            extra_body: Dict[str, Any] = {}
        elif isinstance(raw_extra_body, dict):
            extra_body = dict(raw_extra_body)
        else:
            raise TypeError("extra_body must be a mapping")
        existing_request_id = extra_body.get("request_id")
        if (
            existing_request_id is not None
            and str(existing_request_id) != router_request_id
        ):
            raise ValueError("extra_body.request_id conflicts with router request ID")
        extra_body["request_id"] = router_request_id
        payload["extra_body"] = extra_body
        if "messages" in payload:
            return f"chatcmpl-{router_request_id}"
        return f"cmpl-{router_request_id}-0"

    def _select_instance(
        self,
        wait_results: Dict[str, WaitTimeResult],
        accuracy_scores: Optional[Dict[str, float]] = None,
        output_lengths: Optional[Dict[str, float]] = None,
        prompt_tokens: Optional[int] = None,
    ) -> str:
        # Fall back to the first instance if we have no wait-time data.
        if not wait_results:
            return next(iter(self._instances.keys()))

        accuracy_scores = accuracy_scores or {}
        output_lengths = output_lengths or {}
        prompt_tokens = prompt_tokens or 0

        if self._utility_fn is None and (
            not accuracy_scores or not output_lengths or prompt_tokens <= 0
        ):
            return min(wait_results, key=lambda k: wait_results[k].wait_ms)

        utility_fn = self._utility_fn or self._utility_soft
        try:
            utilities: Dict[str, float] = {
                instance_id: float(
                    utility_fn(
                        instance_id,
                        wait_results,
                        accuracy_scores,
                        output_lengths,
                        prompt_tokens,
                    )
                )
                for instance_id in wait_results
            }
            max_utility = max(utilities.values())
            tied_best = [
                instance_id
                for instance_id, utility_value in utilities.items()
                if utility_value == max_utility
            ]
            if len(tied_best) == 1:
                return tied_best[0]
            return str(self._route_rng.choice(tied_best))
        except Exception as exc:
            LOGGER.warning(
                "Utility function failed while selecting instance: %s. "
                "Falling back to minimum wait time.",
                exc,
            )
            return min(wait_results, key=lambda k: wait_results[k].wait_ms)

    def _utility_soft(
        self,
        instance_id: str,
        wait_results: Dict[str, WaitTimeResult],
        accuracy_scores: Dict[str, float],
        output_lengths: Dict[str, float],
        prompt_tokens: int,
    ) -> float:
        wait = wait_results[instance_id].wait_ms
        accuracy = accuracy_scores.get(instance_id, 0.0)
        cost_info = self._instance_costs.get(instance_id) or {}
        prompt_rate = float(cost_info.get("prompt", 0.0))
        output_rate = float(cost_info.get("output", 0.0))
        predicted_output = output_lengths.get(instance_id, 0.0)
        cost = prompt_rate * prompt_tokens + output_rate * predicted_output
        LOGGER.warning(
            "Policy: soft; Instance %s: wait=%sms, accuracy=%s, cost=%s, "
            "predicted_output=%s tokens",
            instance_id,
            wait,
            accuracy,
            cost,
            predicted_output,
        )
        return accuracy - self._lambda * cost - self._delta * wait

    async def get_request_wait_time(self, request_id: str) -> Optional[WaitTimeResult]:
        """Retrieve the wait time that was recorded for a routed request."""
        return self._request_log.get(request_id)

    async def _log_to_file(self, request_id: str, wait: WaitTimeResult) -> None:
        """Append a record to disk if logging is enabled."""
        if not self._log_path:
            return

        record = {
            "request_id": request_id,
            "instance_id": wait.instance_id,
            "wait_time_ms": wait.wait_ms,
            "fetched_at_s": wait.fetched_at_s,
            "payload": wait.raw_payload,
        }
        line = self._record_to_line(record)

        if self._defer_sidecar_writes:
            self._deferred_wait_log_lines.append(line)
            return

        try:
            await asyncio.to_thread(self._append_lines, self._log_path, [line])
        except Exception as exc:
            LOGGER.warning("Failed to write wait time log: %s", exc)

    def _handle_response_mapping(
        self,
        task: asyncio.Task,
        *,
        request_id: str,
        instance_id: str,
    ) -> None:
        if not self._response_map_path:
            return
        try:
            response = task.result()
        except Exception as exc:
            LOGGER.debug(
                "Request %s on %s failed before mapping log: %s",
                request_id,
                instance_id,
                exc,
            )
            return

        response_id = getattr(response, "id", None)
        model_name = getattr(response, "model", None)
        if not response_id:
            return
        log_task = asyncio.create_task(
            self._log_response_mapping(request_id, instance_id, response_id, model_name)
        )
        self._response_map_tasks.add(log_task)
        log_task.add_done_callback(self._response_map_tasks.discard)
        log_task.add_done_callback(self._on_response_map_task_done)

    async def _log_response_mapping(
        self,
        request_id: str,
        instance_id: str,
        response_id: str,
        model_name: Optional[str],
    ) -> None:
        if not self._response_map_path:
            return
        record = {
            "request_id": request_id,
            "instance_id": instance_id,
            "response_id": response_id,
            "model": model_name,
        }
        line = self._record_to_line(record)
        if self._defer_sidecar_writes:
            self._deferred_response_map_lines.append(line)
            return
        await asyncio.to_thread(self._append_lines, self._response_map_path, [line])

    async def flush_sidecar_logs(self) -> None:
        """Flush deferred sidecar logs to disk."""
        if not self._defer_sidecar_writes:
            return

        wait_lines: list[str] = []
        response_lines: list[str] = []
        if self._deferred_wait_log_lines:
            wait_lines = self._deferred_wait_log_lines
            self._deferred_wait_log_lines = []
        if self._deferred_response_map_lines:
            response_lines = self._deferred_response_map_lines
            self._deferred_response_map_lines = []

        try:
            if wait_lines and self._log_path:
                await asyncio.to_thread(self._append_lines, self._log_path, wait_lines)
            if response_lines and self._response_map_path:
                await asyncio.to_thread(
                    self._append_lines,
                    self._response_map_path,
                    response_lines,
                )
        except Exception as exc:
            if wait_lines:
                self._deferred_wait_log_lines = (
                    wait_lines + self._deferred_wait_log_lines
                )
            if response_lines:
                self._deferred_response_map_lines = (
                    response_lines + self._deferred_response_map_lines
                )
            raise RuntimeError(f"Failed to flush deferred sidecar logs: {exc}") from exc

    async def _wait_for_response_map_tasks(self) -> None:
        while self._response_map_tasks:
            pending = list(self._response_map_tasks)
            await asyncio.gather(*pending, return_exceptions=True)

    @staticmethod
    def _on_response_map_task_done(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            LOGGER.warning("Failed to write response mapping log: %s", exc)

    @staticmethod
    def _record_to_line(record: Dict[str, Any]) -> str:
        return f"{record}\n"

    @staticmethod
    def _append_lines(path: str, lines: list[str]) -> None:
        if not lines:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as fp:
            fp.writelines(lines)

    @staticmethod
    def _extract_prompt_text(payload: Dict[str, Any]) -> Optional[str]:
        if "prompt" in payload and isinstance(payload["prompt"], str):
            return payload["prompt"]
        messages = payload.get("messages")
        if isinstance(messages, list):
            parts = []
            for msg in messages:
                content = msg.get("content")
                if isinstance(content, str):
                    parts.append(content)
            if parts:
                return "\n".join(parts)
        return None

    @staticmethod
    def _extract_completion_cap(payload: Dict[str, Any]) -> Optional[float]:
        raw_cap: Any = None
        for key in ("max_completion_tokens", "max_tokens"):
            if key in payload:
                raw_cap = payload.get(key)
                break
        if raw_cap is None:
            return None
        if isinstance(raw_cap, bool):
            return None
        if isinstance(raw_cap, (int, float)):
            cap = float(raw_cap)
        elif isinstance(raw_cap, str):
            text = raw_cap.strip()
            if not text:
                return None
            try:
                cap = float(text)
            except ValueError:
                return None
        else:
            return None
        if not math.isfinite(cap) or cap <= 0.0:
            return None
        return max(1.0, cap)

    @staticmethod
    def _extract_precomputed_prompt_tokens(payload: Dict[str, Any]) -> Optional[int]:
        raw_tokens = payload.pop("_prompt_tokens", None)
        if raw_tokens is None or isinstance(raw_tokens, bool):
            return None
        if isinstance(raw_tokens, (int, float)):
            numeric = float(raw_tokens)
        elif isinstance(raw_tokens, str):
            stripped = raw_tokens.strip()
            if not stripped:
                return None
            try:
                numeric = float(stripped)
            except ValueError:
                return None
        else:
            return None
        if not math.isfinite(numeric) or numeric < 0.0:
            return None
        return int(numeric)

    def _get_prompt_tokens(self, payload: Dict[str, any], prompt_text: str) -> int:
        if self._tokenizer:
            prompt_token_ids = self._tokenizer.apply_chat_template(
                payload.get("messages"),
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            return len(prompt_token_ids)
        return len(prompt_text.split())

    def _build_batch_admissions(
        self,
        prompt_text: Optional[str],
        prompt_tokens: int,
    ) -> tuple[list[str], list[AdmissionFeatures]]:
        if not prompt_text or prompt_tokens <= 0:
            return [], []
        instance_ids: list[str] = []
        admissions: list[AdmissionFeatures] = []
        for instance_id, instance in self._instances.items():
            instance_ids.append(instance_id)
            admissions.append(
                AdmissionFeatures(
                    model_id=instance.model_id,
                    prompt_text=prompt_text,
                    prompt_token_count=prompt_tokens,
                )
            )
        return instance_ids, admissions

    def _build_prompt_context(
        self,
        prompt_text: Optional[str],
        prompt_tokens: int,
    ) -> Optional[PromptFeatureContext]:
        if not prompt_text or prompt_tokens <= 0:
            return None
        extractor = None
        if self._output_length_predictor is not None:
            extractor = getattr(self._output_length_predictor, "_feature_extractor", None)
        elif self._accuracy_predictor is not None:
            feature_builder = getattr(self._accuracy_predictor, "_feature_builder", None)
            extractor = getattr(feature_builder, "_prompt_extractor", None)
        if extractor is None or not hasattr(extractor, "build_prompt_context"):
            return None
        try:
            return extractor.build_prompt_context(
                prompt_text=prompt_text,
                prompt_token_count=prompt_tokens,
            )
        except Exception:
            return None

    def _predict_accuracy_for_admissions(
        self,
        *,
        instance_ids: list[str],
        admissions: list[AdmissionFeatures],
        prompt_context: Optional[PromptFeatureContext],
    ) -> Dict[str, float]:
        if not self._accuracy_predictor or not admissions:
            return {}
        predictions = self._accuracy_predictor.predict_batch(
            admissions,
            prompt_context=prompt_context,
        )
        return {
            instance_id: float(score)
            for instance_id, score in zip(instance_ids, predictions)
        }

    def _predict_output_lengths_for_admissions(
        self,
        *,
        instance_ids: list[str],
        admissions: list[AdmissionFeatures],
        completion_cap: Optional[float],
        prompt_context: Optional[PromptFeatureContext],
    ) -> Dict[str, float]:
        if not self._output_length_predictor or not admissions:
            return {}
        predictions = self._output_length_predictor.predict_batch(
            admissions,
            prompt_context=prompt_context,
        )
        output_lengths: Dict[str, float] = {}
        for instance_id, result in zip(instance_ids, predictions):
            if result is None:
                continue
            predicted_tokens = float(result.mean_tokens)
            if completion_cap is not None:
                predicted_tokens = min(predicted_tokens, completion_cap)
            output_lengths[instance_id] = max(1.0, predicted_tokens)
        return output_lengths

    async def _predict_model_scores(
        self,
        *,
        prompt_text: Optional[str],
        prompt_tokens: int,
        completion_cap: Optional[float] = None,
    ) -> tuple[Dict[str, float], Dict[str, float]]:
        instance_ids, admissions = self._build_batch_admissions(prompt_text, prompt_tokens)
        if not admissions:
            return {}, {}
        prompt_context = self._build_prompt_context(prompt_text, prompt_tokens)

        tasks: list[tuple[str, Any]] = []
        if self._accuracy_predictor is not None:
            tasks.append(
                (
                    "accuracy",
                    asyncio.to_thread(
                        self._predict_accuracy_for_admissions,
                        instance_ids=instance_ids,
                        admissions=admissions,
                        prompt_context=prompt_context,
                    ),
                )
            )
        if self._output_length_predictor is not None:
            tasks.append(
                (
                    "output_length",
                    asyncio.to_thread(
                        self._predict_output_lengths_for_admissions,
                        instance_ids=instance_ids,
                        admissions=admissions,
                        completion_cap=completion_cap,
                        prompt_context=prompt_context,
                    ),
                )
            )
        if not tasks:
            return {}, {}

        resolved = await asyncio.gather(*(task for _, task in tasks))
        accuracy_scores: Dict[str, float] = {}
        output_lengths: Dict[str, float] = {}
        for (task_name, _), task_result in zip(tasks, resolved):
            if task_name == "accuracy":
                accuracy_scores = task_result
            elif task_name == "output_length":
                output_lengths = task_result
        return accuracy_scores, output_lengths

    def _predict_accuracy(
        self, prompt_text: Optional[str], prompt_tokens: int
    ) -> Dict[str, float]:
        if not self._accuracy_predictor or not prompt_text or prompt_tokens <= 0:
            return {}
        instance_ids, admissions = self._build_batch_admissions(prompt_text, prompt_tokens)
        prompt_context = self._build_prompt_context(prompt_text, prompt_tokens)
        return self._predict_accuracy_for_admissions(
            instance_ids=instance_ids,
            admissions=admissions,
            prompt_context=prompt_context,
        )

    def _predict_output_length(
        self,
        prompt_text: Optional[str],
        prompt_tokens: int,
        *,
        completion_cap: Optional[float] = None,
    ) -> Dict[str, float]:
        if not self._output_length_predictor or not prompt_text or prompt_tokens <= 0:
            return {}
        instance_ids, admissions = self._build_batch_admissions(prompt_text, prompt_tokens)
        prompt_context = self._build_prompt_context(prompt_text, prompt_tokens)
        return self._predict_output_lengths_for_admissions(
            instance_ids=instance_ids,
            admissions=admissions,
            completion_cap=completion_cap,
            prompt_context=prompt_context,
        )

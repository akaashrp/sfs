"""Timed dispatch adapters for the three methodology routing baselines.

This scheduler shares the harness's client/logging lifecycle, but does not run
the SFS quality gate, predictors, snapshot simulator, or output reserve rule.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
import math
import time
from typing import Any

from .methodology_policies import (
    PrefillWork, RequestLifetimeLedger, RouteBalanceCandidate,
    RouteBalanceWeights, estimate_routebalance_latency_ms, merge_prefill_work,
    routebalance_cost, routebalance_lpt_order, select_lmdeploy,
    select_mooncake, select_routebalance,
)
from .wait_time_scheduler import RoutedRequest, WaitTimeResult, WaitTimeScheduler


METHODOLOGY_POLICIES = ("lmdeploy_proxy", "mooncake_prefill", "routebalance")


class MethodologyScheduler(WaitTimeScheduler):
    def __init__(self, instances, *, policy: str, calibration, predictor=None,
                 weights: RouteBalanceWeights | None = None, batch_max_size=16,
                 batch_wait_ms=25.0, snapshot_max_age_ms=1000.0,
                 route_random_seed=69, **kwargs):
        if policy not in METHODOLOGY_POLICIES:
            raise ValueError(f"Unknown methodology policy {policy!r}")
        if isinstance(batch_max_size, bool) or batch_max_size < 1 or int(batch_max_size) != batch_max_size:
            raise ValueError("batch_max_size must be a positive integer")
        for name, value in (("batch_wait_ms", batch_wait_ms),
                            ("snapshot_max_age_ms", snapshot_max_age_ms)):
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if policy == "routebalance" and predictor is None:
            raise ValueError("RouteBalance requires its MiniLM/KNN predictor artifact")
        kwargs.update(accuracy_model_path=None, output_length_model_path=None,
                      enable_wait_time_polling=False, readiness_predictor_path=None)
        # One coordinator owns a whole RouteBalance batch. Other baseline
        # decisions are also serialized, matching the upstream proxy lifecycle.
        kwargs["worker_count"] = 1
        super().__init__(instances, route_random_seed=route_random_seed, **kwargs)
        self.policy = policy
        self.calibration = calibration
        self.predictor = predictor
        self.weights = weights or RouteBalanceWeights()
        self.batch_max_size = int(batch_max_size)
        self.batch_wait_ms = float(batch_wait_ms)
        self.snapshot_max_age_ms = float(snapshot_max_age_ms)
        self.seed = int(route_random_seed)
        self.lifetime = RequestLifetimeLedger(instances)
        self._completed_ids: set[str] = set()
        self._engine_ids: dict[str, str] = {}
        self._dispatch_times: dict[str, float] = {}
        self._versions: dict[str, int] = {}
        self._input_finished = asyncio.Event()
        self._batch_id = 0
        self._batch_sizes: dict[int, int] = {}
        self._model_labels = {key: client.model_id for key, client in instances.items()}
        self._speeds = {key: calibration.speeds[label]
                        for key, label in self._model_labels.items()}
        # Validate all rates before accepting any request, including all-idle ties.
        if any(not math.isfinite(float(s)) or s <= 0 for s in self._speeds.values()):
            raise ValueError("Every candidate requires a positive calibrated request speed")
        if predictor is not None and not set(self._model_labels.values()).issubset(predictor.model_labels):
            raise ValueError("RouteBalance predictor does not cover the candidate models")

    async def start(self):
        if self._workers:
            return
        if self.policy == "routebalance" and hasattr(self.calibration, "preload"):
            await asyncio.to_thread(self.calibration.preload)
        if self.policy != "lmdeploy_proxy":
            # Earlier policy/warm-up work must finish before the new ledger starts.
            deadline = time.monotonic() + 10.0
            while True:
                snapshots = await self._read_snapshots()
                if all(not s.requests and not s.inflight_total_tokens for s in snapshots.values()):
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("Baseline trial requires drained engines before start")
                await asyncio.sleep(0.05)
        if not self._workers:
            self._workers.append(asyncio.create_task(self._worker_loop(), name=f"{self.policy}-coordinator"))

    def finish_arrivals(self):
        """Wake a partly filled collection window after the last enqueue."""
        self._input_finished.set()

    async def _read_snapshots(self):
        values = await asyncio.gather(*(
            client.refresh_baseline_state() for client in self._instances.values()
        ))
        snapshots = dict(zip(self._instances, values))
        for key, snapshot in snapshots.items():
            # Publication is event-driven: a drained exclusive server can keep
            # an old empty snapshot indefinitely. This exception never proves a
            # free decode slot, and ends as soon as we dispatch local work.
            idle_unchanged = (not snapshot.requests and not snapshot.inflight_total_tokens
                              and not self.lifetime.reservations(key))
            # A newly submitted request may not yet have replaced that old
            # empty publication. Bound this handoff grace by dispatch age,
            # without altering the snapshot age used for admission evidence.
            reservations = self.lifetime.reservations(key)
            idle_handoff = (not snapshot.requests and not snapshot.inflight_total_tokens
                            and reservations and all(
                                (time.perf_counter() - self._dispatch_times.get(r.request_id, -math.inf)) * 1000
                                <= self.snapshot_max_age_ms for r in reservations))
            if snapshot.age_ms > self.snapshot_max_age_ms and not (idle_unchanged or idle_handoff):
                raise RuntimeError(f"Stale baseline snapshot for {key}: {snapshot.age_ms:.1f} ms")
            if snapshot.version < self._versions.get(key, -1):
                raise RuntimeError(f"Baseline snapshot version regressed for {key}")
            self._versions[key] = snapshot.version
        return snapshots

    async def _collect_batch(self, first, batch=None):
        if batch is None:
            batch = [first]
        if self.policy != "routebalance" or any(n == 0 for n in self.lifetime.unfinished_counts().values()):
            return batch
        oldest = float(first.payload.get("_system_entry_perf", time.perf_counter()))
        deadline = oldest + self.batch_wait_ms / 1000.0
        while len(batch) < self.batch_max_size:
            try:
                batch.append(self._queue.get_nowait())
                continue
            except asyncio.QueueEmpty:
                pass
            remaining = deadline - time.perf_counter()
            if remaining <= 0 or self._input_finished.is_set():
                break
            get_task = asyncio.create_task(self._queue.get())
            end_task = asyncio.create_task(self._input_finished.wait())
            try:
                await asyncio.wait(
                    (get_task, end_task), timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                for task in (get_task, end_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(get_task, end_task, return_exceptions=True)
                # queue.get may complete while timeout/end cleanup runs. Account
                # for every consumed item, including cancellation of the worker.
                if not get_task.cancelled() and get_task.exception() is None:
                    batch.append(get_task.result())
            if get_task.cancelled():
                break
        return batch

    async def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                first = await asyncio.wait_for(self._queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            batch = [first]
            try:
                await self._collect_batch(first, batch)
                await self._dispatch_batch(batch)
            except BaseException as exc:
                for queued in batch:
                    self._fail_request(queued, exc)
                if isinstance(exc, asyncio.CancelledError):
                    raise
            finally:
                for _ in batch:
                    self._queue.task_done()

    @staticmethod
    def _prompt(queued):
        # The calibration encoder consumes the user prompt, not the chat wrapper.
        if isinstance(queued.payload.get("prompt"), str):
            return queued.payload["prompt"]
        return "\n".join(str(m["content"]) for m in queued.payload.get("messages", [])
                         if m.get("role") == "user" and isinstance(m.get("content"), str))

    @staticmethod
    def _base_record(queued):
        now = time.perf_counter()
        return {
            "request_id": queued.request_id,
            "system_entry_perf": float(queued.payload.get("_system_entry_perf", now)),
            "started_perf": float(queued.payload.get("_started_perf", now)),
            "request_bucket": queued.payload.get("_request_bucket", "unknown"),
            "prompt_tokens": int(queued.payload.get("_prompt_tokens", 0)),
        }

    def _fail_request(self, queued, exc, record=None):
        self.lifetime.release(queued.request_id)
        self._completed_ids.add(self._engine_ids.get(queued.request_id, queued.request_id))
        result = dict(record or self._base_record(queued))
        result.update(completed_perf=time.perf_counter(), error=f"{type(exc).__name__}: {exc}",
                      route_strategy=self.policy, wait_estimator=self.policy)
        result["latency_ms"] = (result["completed_perf"] - result["started_perf"]) * 1000.0
        completion = queued.payload.get("_completion_future")
        if completion is not None and not completion.done():
            completion.set_result(result)
        if not queued.result_future.done():
            queued.result_future.set_result(RoutedRequest(queued.request_id, "", None, None))

    async def _dispatch(self, queued):
        await self._dispatch_batch([queued])

    async def _dispatch_batch(self, batch):
        collection_done = time.perf_counter()
        self._batch_id += 1
        batch_id = self._batch_id
        self._batch_sizes[len(batch)] = self._batch_sizes.get(len(batch), 0) + 1
        caps = [int(self._extract_completion_cap(q.payload) or 8192) for q in batch]
        predictions = {}
        predictor_start = time.perf_counter()
        if self.policy == "routebalance":
            rows = await asyncio.to_thread(self.predictor.predict_batch,
                                          [self._prompt(q) for q in batch], caps)
            if len(rows) != len(batch):
                raise ValueError("RouteBalance predictor returned incomplete request coverage")
            predictions = dict(zip((q.request_id for q in batch), rows))
            order = routebalance_lpt_order({rid: {m: x["output_tokens"] for m, x in row.items()
                                                 if m in self._model_labels.values()}
                                           for rid, row in predictions.items()})
            by_id = {q.request_id: q for q in batch}
            batch = [by_id[rid] for rid in order]
        predictor_done = time.perf_counter()
        async with self._routing_state_lock:
            snapshots = await self._read_snapshots() if self.policy != "lmdeploy_proxy" else {}
            self._batch_tpot = {}
            if self.policy == "routebalance":
                # One head query per model/tier per scheduling batch. Every
                # assignment still recomputes latency from updated local work.
                by_model = {}
                for key, snapshot in snapshots.items():
                    model = self._model_labels[key]
                    if model not in by_model:
                        by_model[model] = self.calibration.tpot_ms(model, snapshot.tpot_features)
                    self._batch_tpot[key] = by_model[model]
            telemetry_done = time.perf_counter()
            for position, queued in enumerate(batch):
                if queued.payload.get("_completion_future") is not None and queued.payload["_completion_future"].cancelled():
                    self._fail_request(queued, asyncio.CancelledError())
                    continue
                record = self._base_record(queued)
                record.update(route_strategy=self.policy, wait_estimator=self.policy)
                try:
                    started = time.perf_counter()
                    decision = self._choose(queued, snapshots, predictions.get(queued.request_id))
                    selected = decision.selected_instance_id
                    candidate = decision.candidates[selected]
                    chosen_prediction = predictions.get(queued.request_id, {}).get(self._model_labels[selected], {})
                    output_tokens = float(chosen_prediction.get("output_tokens", 0.0))
                    before = self.lifetime.unfinished_counts()
                    self.lifetime.reserve(queued.request_id, selected,
                                          prompt_tokens=record["prompt_tokens"],
                                          predicted_output_tokens=output_tokens)
                    record.update(instance_id=selected,
                                  predicted_accuracy=chosen_prediction.get("quality"),
                                  predicted_output_tokens=chosen_prediction.get("output_tokens"),
                                  wait_time_ms=candidate.get("total_prefill_ms"),
                                  methodology_terms={
                                      "policy": self.policy, "batch_id": batch_id,
                                      "batch_size": len(batch), "batch_position": position,
                                      "collection_wait_ms": max(0.0, (collection_done-record["system_entry_perf"])*1000),
                                      "prediction_batch_ms": (predictor_done-predictor_start)*1000,
                                      "telemetry_batch_ms": (telemetry_done-predictor_done)*1000,
                                      "selection_ms": (time.perf_counter()-started)*1000,
                                      "candidates": decision.candidates,
                                      "tie_candidates": list(decision.tie_candidates),
                                      "random_seed": self.seed,
                                      "unfinished_before": before,
                                      "unfinished_after": self.lifetime.unfinished_counts(),
                                  })
                    # create_task schedules network work after this atomic batch
                    # pass; every later assignment already sees the reservation.
                    self._submit_one(queued, record)
                except Exception as exc:
                    self._fail_request(queued, exc, record)

    def _choose(self, queued, snapshots, prediction):
        if self.policy == "lmdeploy_proxy":
            return select_lmdeploy(self.lifetime.unfinished_counts(), self._speeds, self._route_rng)
        prompt_tokens = int(queued.payload["_prompt_tokens"])
        if self.policy == "mooncake_prefill":
            work, estimators = {}, {}
            for key, snapshot in snapshots.items():
                observed = [PrefillWork(r.request_id, r.prompt_tokens, r.computed_prompt_tokens)
                            for r in snapshot.requests.values() if r.request_id not in self._completed_ids]
                local = [PrefillWork(self._engine_ids.get(r.request_id, r.request_id), r.prompt_tokens)
                         for r in self.lifetime.reservations(key)]
                work[key] = merge_prefill_work(observed, local)
                model = self._model_labels[key]
                estimators[key] = lambda p, c, model=model: self.calibration.prefill_ms(model, p, c)
            result = select_mooncake(work, prompt_tokens, estimators, self._route_rng)
        else:
            candidates = {}
            for key, snapshot in snapshots.items():
                reservations = {self._engine_ids.get(r.request_id, r.request_id): r
                                for r in self.lifetime.reservations(key)}
                unknown = set(snapshot.requests) - set(reservations) - self._completed_ids
                if unknown:
                    raise RuntimeError(f"Untracked requests in RouteBalance engine {key}: {sorted(unknown)[:3]}")
                pending = sum(max(0.0, r.predicted_output_tokens - (
                    snapshot.requests[rid].generated_tokens if rid in snapshot.requests else 0
                )) for rid, r in reservations.items())
                unobserved = len(set(reservations) - set(snapshot.requests))
                model = self._model_labels[key]
                length = float(prediction[model]["output_tokens"])
                features = snapshot.tpot_features
                tpot = self._batch_tpot[key]
                admission = snapshot.admission_evidence(
                    prompt_tokens=prompt_tokens, predicted_output_tokens=length,
                    max_age_ms=self.snapshot_max_age_ms,
                    local_outstanding_requests=unobserved,
                )
                latency, terms = estimate_routebalance_latency_ms(
                    tpot_ms=tpot, pending_decode_tokens=pending,
                    decode_batch_size=snapshot.decode_batch_size,
                    predicted_output_tokens=length,
                    free_decode_slot=admission.free_decode_slot,
                    free_slot_reason=admission.reason,
                )
                price = self._instance_costs[key]
                cost = routebalance_cost(prompt_tokens=prompt_tokens, predicted_output_tokens=length,
                                         input_token_price=price["prompt"], output_token_price=price["output"])
                candidates[key] = RouteBalanceCandidate(prediction[model]["quality"], cost, latency,
                    {**terms, "tpot_features": features, "admission": asdict(admission),
                     "unobserved_reservations": unobserved,
                     "cost_units": "USD_times_1e6", "weights": asdict(self.weights)})
            result = select_routebalance(candidates, self.weights, self._route_rng)
        for key, terms in result.candidates.items():
            terms["snapshot"] = snapshots[key].metadata()
        return result

    def _submit_one(self, queued, record):
        payload = {k: v for k, v in queued.payload.items() if not k.startswith("_")}
        engine_id = self._attach_engine_request_id(payload, queued.request_id)
        self._engine_ids[queued.request_id] = engine_id
        record["methodology_terms"]["engine_request_id"] = engine_id
        target = self._instances[record["instance_id"]]
        record["dispatch_perf"] = time.perf_counter()
        self._dispatch_times[queued.request_id] = record["dispatch_perf"]
        task = asyncio.create_task(self._complete(queued, record, target, payload))
        self._submit_tasks.add(task)
        def finished(done):
            # A task cancelled before its coroutine starts never enters that
            # coroutine's try/finally. Synchronous event-loop cleanup also owns
            # this case, and release is idempotent with normal completion.
            if done.cancelled():
                self._fail_request(queued, asyncio.CancelledError(), record)
            self._submit_tasks.discard(done)
        task.add_done_callback(finished)
        if not queued.result_future.done():
            queued.result_future.set_result(RoutedRequest(queued.request_id, target.instance_id,
                                                         record.get("wait_time_ms"), record["methodology_terms"]))

    async def _complete(self, queued, record, target, payload):
        try:
            response = await target.submit_request(**payload)
            record["completed_perf"] = time.perf_counter()
            get = (lambda name: response.get(name)) if isinstance(response, dict) else (lambda name: getattr(response, name, None))
            record.update(response_id=get("id"), response_model=get("model"))
            usage = get("usage")
            for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
                record[f"usage_{name}"] = value
            record["latency_ms"] = (record["completed_perf"]-record["started_perf"])*1000
            async with self._routing_state_lock:
                self.lifetime.release(queued.request_id)
                self._completed_ids.add(self._engine_ids.get(queued.request_id, queued.request_id))
            if record["response_id"]:
                await self._log_response_mapping(queued.request_id, target.instance_id,
                                                 record["response_id"], record["response_model"])
            wait = WaitTimeResult(target.instance_id, float(record.get("wait_time_ms") or 0),
                                  time.time(), {"methodology_terms": record["methodology_terms"]})
            self._request_log[queued.request_id] = wait
            await self._log_to_file(queued.request_id, wait)
            completion = queued.payload.get("_completion_future")
            if completion is not None and not completion.done():
                completion.set_result(record)
        except BaseException as exc:
            async with self._routing_state_lock:
                self._fail_request(queued, exc, record)
            if isinstance(exc, asyncio.CancelledError):
                raise

    def run_metadata(self):
        return {
            "policy": self.policy, "implementation_version": 1,
            "random_seed": self.seed, "batch_max_size": self.batch_max_size,
            "batch_wait_ms": self.batch_wait_ms, "weights": asdict(self.weights),
            "batch_size_histogram": dict(self._batch_sizes),
            "unfinished_after_drain": self.lifetime.unfinished_counts(),
            "calibration": getattr(self.calibration, "metadata", {}),
            "predictor": getattr(self.predictor, "metadata", None),
            "sfs_predictor_substitution": False,
            "admission_rejection_enabled": False,
            "snapshot_idle_age_exception": "exclusive_drained_server_without_local_reservations",
            "snapshot_idle_handoff_grace_ms": self.snapshot_max_age_ms,
            "shared_engine_output_predictor": "enabled_for_common_publisher; ignored_by_baseline_decisions",
        }

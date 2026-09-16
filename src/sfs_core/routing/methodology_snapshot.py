"""Observed scheduler state for routing baselines, without SFS simulation.

The scheduler publishes *planned* computed-token counts before executing a
batch. We subtract its per-request scheduled increments and retain the entire
inflight batch until a newer snapshot confirms progress. We never extrapolate
GPU progress from elapsed wall time or read SFS output targets/reserves here.
Router reservations belong to the caller and must be reconciled by request ID.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class BaselineRequestState:
    request_id: str
    queue_phase: str
    status: str
    prompt_tokens: int
    committed_computed_tokens: int
    computed_prompt_tokens: int
    generated_tokens: int
    scheduled_tokens: int
    scheduled_prefill_tokens: int
    scheduled_decode_tokens: int
    kv_block_counts: tuple[int, ...]

    @property
    def remaining_prompt_tokens(self) -> int:
        return self.prompt_tokens - self.computed_prompt_tokens


@dataclass(frozen=True, slots=True)
class AdmissionEvidence:
    """A conservative proxy, not a promise from vLLM's KV allocator."""

    free_decode_slot: bool
    reason: str
    proxy: str = "conservative_capacity_proxy"
    required_kv_blocks: int = 0
    available_sequence_slots: int = 0
    available_iteration_tokens: int = 0


@dataclass(frozen=True, slots=True)
class BaselineSnapshot:
    version: int
    created_at: float
    observed_at: float
    requests: dict[str, BaselineRequestState]
    running_request_ids: tuple[str, ...]
    waiting_request_ids: tuple[str, ...]
    decode_batch_size: int
    decode_running_count: int
    inflight_prefill_tokens: int
    inflight_decode_tokens: int
    inflight_total_tokens: int
    inflight_total_context_len: int
    max_num_batched_tokens: int
    max_num_seqs: int
    max_model_len: int
    chunked_prefill_enabled: bool
    kv_cache_free_blocks: int
    kv_cache_total_blocks: int
    kv_cache_block_size: int
    kv_cache_num_groups: int
    decode_context_parallel_size: int
    build_latency_ms: float

    @property
    def age_ms(self) -> float:
        return max(0.0, (self.observed_at - self.created_at) * 1000.0)

    @property
    def num_running(self) -> int:
        return len(self.running_request_ids)

    @property
    def num_waiting(self) -> int:
        return len(self.waiting_request_ids)

    @property
    def tpot_features(self) -> dict[str, float]:
        """Actual inflight features matching trace ``decode/prefill/sum_tokens``.

        When no batch is executing these are zero, not a prediction of the
        next batch. Estimator callers must name any prospective feature proxy.
        """
        return {
            "decode_tokens": float(self.inflight_decode_tokens),
            "prefill_tokens": float(self.inflight_prefill_tokens),
            "context_tokens": float(self.inflight_total_context_len),
        }

    def observed_again(self, now: float) -> BaselineSnapshot:
        return replace(self, observed_at=_finite(now, "observed_at"))

    def admission_evidence(
        self,
        *,
        prompt_tokens: int,
        predicted_output_tokens: float,
        max_age_ms: float | None = None,
        local_outstanding_requests: int = 0,
        capacity_current: bool = True,
    ) -> AdmissionEvidence:
        """Use the free-slot branch only with explicit capacity evidence.

        An unused sequence slot on a busy coupled server does not prove prompt
        admission. This conservative proxy requires observed capacity, no queued
        or unfinished prefills or caller-side unobserved dispatches, enough
        sequence/token capacity for a first chunk, and KV capacity for the
        predicted context plus one growth block per resident decode sequence.
        The caller must pass reservations and invalidate this evidence after
        each assignment. Missing/complex memory topology never implies room.
        """
        prompt = _positive(prompt_tokens, "prompt_tokens")
        length = _finite(predicted_output_tokens, "predicted_output_tokens")
        if length < 0:
            raise ValueError("predicted_output_tokens must be nonnegative")
        # Kept as an API compatibility argument; timestamp age is diagnostic.
        # The caller supplies publication/progress evidence separately.
        if max_age_ms is not None and _finite(max_age_ms, "max_age_ms") < 0:
            raise ValueError("max_age_ms must be nonnegative")
        outstanding = _nonnegative(local_outstanding_requests, "local_outstanding_requests")
        if not capacity_current:
            return AdmissionEvidence(False, "unchanged_busy_publication_capacity_uncertain")
        if outstanding:
            return AdmissionEvidence(False, "unobserved_local_dispatches")
        if self.waiting_request_ids:
            return AdmissionEvidence(False, "waiting_requests_have_admission_priority")
        if self.inflight_prefill_tokens or any(request.remaining_prompt_tokens for request in self.requests.values()):
            return AdmissionEvidence(False, "unfinished_prefills_no_admission_proof")
        sequence_slots = self.max_num_seqs - self.num_running
        iteration_tokens = self.max_num_batched_tokens - max(self.num_running, self.inflight_total_tokens)
        if sequence_slots < 1:
            return AdmissionEvidence(False, "no_sequence_capacity")
        context = prompt + math.ceil(length)
        if context > self.max_model_len:
            return AdmissionEvidence(False, "predicted_context_exceeds_model_limit")
        if iteration_tokens < 1 or (
            not self.chunked_prefill_enabled and prompt > iteration_tokens
        ):
            return AdmissionEvidence(False, "insufficient_prefill_token_budget")
        if self.decode_context_parallel_size != 1 or self.kv_cache_num_groups != 1:
            return AdmissionEvidence(False, "unsupported_kv_topology")
        required = math.ceil(max(1, context) / self.kv_cache_block_size) + self.num_running
        if self.kv_cache_free_blocks < required:
            return AdmissionEvidence(False, "insufficient_kv_blocks", required_kv_blocks=required,
                                     available_sequence_slots=sequence_slots, available_iteration_tokens=iteration_tokens)
        reason = ("decode_only_sequence_token_and_kv_capacity" if self.num_running
                  else "idle_sequence_token_and_kv_capacity")
        return AdmissionEvidence(True, reason, required_kv_blocks=required,
                                 available_sequence_slots=sequence_slots, available_iteration_tokens=iteration_tokens)

    def metadata(self) -> dict[str, Any]:
        """Small decision-log summary; request details remain in ``requests``."""
        return {
            "snapshot_version": self.version,
            "snapshot_timestamp": self.created_at,
            "snapshot_age_ms": self.age_ms,
            "snapshot_build_latency_ms": self.build_latency_ms,
            "snapshot_mode": "baseline_observed_no_simulation",
            "progress_semantics": "committed_before_inflight_batch",
            "num_running": self.num_running,
            "num_waiting": self.num_waiting,
            "decode_batch_size": self.decode_batch_size,
            "decode_running_count": self.decode_running_count,
            "inflight_prefill_tokens": self.inflight_prefill_tokens,
            "inflight_decode_tokens": self.inflight_decode_tokens,
            "inflight_total_tokens": self.inflight_total_tokens,
            "inflight_total_context_len": self.inflight_total_context_len,
            "kv_cache_free_blocks": self.kv_cache_free_blocks,
            "kv_cache_total_blocks": self.kv_cache_total_blocks,
        }


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _nonnegative(value: Any, name: str) -> int:
    number = int(value)
    if number < 0 or number != value:
        raise ValueError(f"{name} must be a nonnegative integer")
    return number


def _positive(value: Any, name: str) -> int:
    number = _nonnegative(value, name)
    if not number:
        raise ValueError(f"{name} must be positive")
    return number


def parse_baseline_snapshot(
    payload: Mapping[str, Any],
    *,
    observed_at: float,
    expected_version: int | None = None,
    expected_created_at: float | None = None,
) -> BaselineSnapshot:
    """Parse existing msgpack primitives; reject ambiguous inflight progress.

    No pending-dispatch argument is accepted intentionally. A visible engine
    request and a router reservation must never both become observed work.
    """
    version = _nonnegative(payload["version"], "version")
    created_at = _finite(payload["created_at"], "created_at")
    if expected_version is not None and version != expected_version:
        raise ValueError("Scheduler snapshot payload/header version mismatch")
    if expected_created_at is not None and created_at != expected_created_at:
        raise ValueError("Scheduler snapshot payload/header timestamp mismatch")
    running = tuple(str(key) for key in payload["running_request_ids"])
    waiting = tuple(str(key) for key in payload["waiting_request_ids"])
    if len(set(running + waiting)) != len(running) + len(waiting):
        raise ValueError("Scheduler snapshot has duplicate/overlapping queue IDs")
    if len(running) != int(payload["num_running"]) or len(waiting) != int(payload["num_waiting"]):
        raise ValueError("Scheduler snapshot queue counts disagree")
    inflight = payload.get("inflight_batch") or {}
    inflight_prefill = _nonnegative(inflight.get("batch_prefill_tokens", 0), "batch_prefill_tokens")
    inflight_decode = _nonnegative(inflight.get("batch_decode_tokens", 0), "batch_decode_tokens")
    total = inflight_prefill + inflight_decode
    scheduled_raw = inflight.get("scheduled_tokens_by_request")
    if total and not isinstance(scheduled_raw, dict):
        raise ValueError(
            "Baseline telemetry requires inflight scheduled_tokens_by_request; "
            "restart servers using this worktree's Python snapshot publisher"
        )
    scheduled = {
        str(key): _positive(value, f"scheduled_tokens[{key}]")
        for key, value in (scheduled_raw or {}).items()
    }
    if sum(scheduled.values()) != total:
        raise ValueError("Inflight per-request token counts disagree with batch totals")
    if not set(scheduled).issubset(running):
        raise ValueError("Inflight batch contains a non-running request")
    raw_requests = payload["requests"]
    if set(raw_requests) != set(running + waiting):
        raise ValueError("Scheduler snapshot requests disagree with active queues")
    requests: dict[str, BaselineRequestState] = {}
    running_ids = set(running)
    for key, raw in raw_requests.items():
        request_id = str(key)
        if request_id != str(raw["request_id"]):
            raise ValueError("Scheduler snapshot request ID/key mismatch")
        prompt = _nonnegative(raw["num_prompt_tokens"], "num_prompt_tokens")
        planned = _nonnegative(raw["num_computed_tokens"], "num_computed_tokens")
        executing = scheduled.get(request_id, 0)
        if executing > planned:
            raise ValueError("Inflight scheduled work exceeds planned computed tokens")
        committed = planned - executing
        computed_prompt = min(prompt, committed)
        scheduled_prefill = min(executing, prompt - computed_prompt)
        requests[request_id] = BaselineRequestState(
            request_id=request_id,
            queue_phase="running" if request_id in running_ids else "waiting",
            status=str(raw["status"]),
            prompt_tokens=prompt,
            committed_computed_tokens=committed,
            computed_prompt_tokens=computed_prompt,
            generated_tokens=_nonnegative(raw["num_output_processed_tokens"], "num_output_processed_tokens"),
            scheduled_tokens=executing,
            scheduled_prefill_tokens=scheduled_prefill,
            scheduled_decode_tokens=executing - scheduled_prefill,
            kv_block_counts=tuple(_nonnegative(n, "kv_block_count") for n in raw.get("kv_block_counts", ())),
        )
    if sum(request.scheduled_prefill_tokens for request in requests.values()) != inflight_prefill:
        raise ValueError("Inflight prefill totals disagree with per-request committed progress")
    config = payload["config"]
    kv = payload["kv_cache_config"]
    return BaselineSnapshot(
        version=version,
        created_at=created_at,
        observed_at=_finite(observed_at, "observed_at"),
        requests=requests,
        running_request_ids=running,
        waiting_request_ids=waiting,
        decode_batch_size=sum(request.scheduled_decode_tokens > 0 for request in requests.values()),
        decode_running_count=sum(request.queue_phase == "running" and request.remaining_prompt_tokens == 0 for request in requests.values()),
        inflight_prefill_tokens=inflight_prefill,
        inflight_decode_tokens=inflight_decode,
        inflight_total_tokens=total,
        inflight_total_context_len=_nonnegative(inflight.get("batch_total_context_len", 0), "batch_total_context_len"),
        max_num_batched_tokens=_nonnegative(config["max_num_batched_tokens"], "max_num_batched_tokens"),
        max_num_seqs=_nonnegative(config["max_num_seqs"], "max_num_seqs"),
        max_model_len=_positive(config["max_model_len"], "max_model_len"),
        chunked_prefill_enabled=bool(config["chunked_prefill_enabled"]),
        kv_cache_free_blocks=_nonnegative(kv["kv_cache_free_blocks"], "kv_cache_free_blocks"),
        kv_cache_total_blocks=_nonnegative(kv["kv_cache_total_blocks"], "kv_cache_total_blocks"),
        kv_cache_block_size=_positive(kv["block_size"], "block_size"),
        kv_cache_num_groups=len(kv["kv_cache_groups"]),
        decode_context_parallel_size=_positive(payload["parallel_config"]["decode_context_parallel_size"], "decode_context_parallel_size"),
        build_latency_ms=_finite(payload.get("build_latency_ms", 0), "build_latency_ms"),
    )

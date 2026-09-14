"""CPU decision rules for explicitly scoped methodology routing baselines.

LMDeploy follows pinned upstream a76d91d's unfinished/speed selection. Mooncake
retains only the paper's prefill placement objective. RouteBalance follows
arXiv:2606.17949v1 equations 1 and 2 and the latency model in section 4.2.
Telemetry, learned predictors, batching and network submission are adapters;
none of these rules invoke SFS's simulator or quality/latency gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Callable, Iterable, Mapping


def _nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _count(value: int, name: str) -> int:
    numeric = _nonnegative(value, name)
    if int(numeric) != numeric:
        raise ValueError(f"{name} must be an integer")
    return int(numeric)


def _positive(value: float, name: str) -> float:
    result = _nonnegative(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class PolicyDecision:
    selected_instance_id: str
    candidates: dict[str, dict]
    tie_candidates: tuple[str, ...]


def _select(terms: dict[str, dict], key: str, rng: random.Random,
            *, maximize: bool = False) -> PolicyDecision:
    if not terms:
        raise ValueError("at least one candidate is required")
    candidates = list(terms)
    # Upstream shuffles before strict comparison. This also avoids imposing an
    # arbitrary model preference when every LMDeploy counter is zero.
    rng.shuffle(candidates)
    selected = candidates[0]
    for candidate in candidates[1:]:
        better = (terms[candidate][key] > terms[selected][key] if maximize
                  else terms[candidate][key] < terms[selected][key])
        if better:
            selected = candidate
    best = terms[selected][key]
    ties = tuple(candidate for candidate in terms if terms[candidate][key] == best)
    return PolicyDecision(selected, terms, ties)


@dataclass(frozen=True)
class RequestReservation:
    request_id: str
    instance_id: str
    prompt_tokens: int = 0
    predicted_output_tokens: float = 0.0


class RequestLifetimeLedger:
    """Router-owned outstanding requests, released only at response completion.

    Call reserve/select/release under the router's existing lock. Observation
    by the engine deliberately does not alter this ledger. Duplicate reserve
    fails rather than silently leaking or moving an in-flight request; release
    is idempotent to tolerate cancellation/failure cleanup paths.
    """

    def __init__(self, instance_ids: Iterable[str] = ()) -> None:
        self._instance_ids = tuple(instance_ids)
        if len(set(self._instance_ids)) != len(self._instance_ids):
            raise ValueError("duplicate instance IDs")
        self._reservations: dict[str, RequestReservation] = {}

    def reserve(self, request_id: str, instance_id: str, *, prompt_tokens: int = 0,
                predicted_output_tokens: float = 0.0) -> RequestReservation:
        if not request_id or not instance_id:
            raise ValueError("request_id and instance_id must be nonempty")
        if self._instance_ids and instance_id not in self._instance_ids:
            raise ValueError(f"unknown instance {instance_id!r}")
        if request_id in self._reservations:
            raise ValueError(f"request {request_id!r} is already reserved")
        reservation = RequestReservation(
            request_id, instance_id, _count(prompt_tokens, "prompt_tokens"),
            _nonnegative(predicted_output_tokens, "predicted_output_tokens"),
        )
        self._reservations[request_id] = reservation
        return reservation

    def release(self, request_id: str) -> bool:
        return self._reservations.pop(request_id, None) is not None

    def unfinished_counts(self) -> dict[str, int]:
        counts = dict.fromkeys(self._instance_ids, 0)
        for reservation in self._reservations.values():
            counts[reservation.instance_id] = counts.get(reservation.instance_id, 0) + 1
        return counts

    def reservations(self, instance_id: str | None = None) -> tuple[RequestReservation, ...]:
        return tuple(r for r in self._reservations.values()
                     if instance_id is None or r.instance_id == instance_id)

    def __len__(self) -> int:
        return len(self._reservations)


def select_lmdeploy(unfinished_counts: Mapping[str, int],
                    instance_speeds: Mapping[str, float],
                    rng: random.Random) -> PolicyDecision:
    if set(unfinished_counts) != set(instance_speeds):
        raise ValueError("unfinished counts and speeds must cover exactly the same pool")
    terms = {}
    for instance_id, count in unfinished_counts.items():
        unfinished = _count(count, f"{instance_id}.unfinished_requests")
        speed = _positive(instance_speeds[instance_id], f"{instance_id}.instance_speed_qps")
        terms[instance_id] = {
            "unfinished_requests": unfinished, "instance_speed_qps": speed,
            "unfinished_over_speed_s": unfinished / speed,
        }
    return _select(terms, "unfinished_over_speed_s", rng)


@dataclass(frozen=True)
class PrefillWork:
    request_id: str
    prompt_tokens: int
    computed_prompt_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be nonempty")
        object.__setattr__(self, "prompt_tokens", _count(self.prompt_tokens, "prompt_tokens"))
        object.__setattr__(self, "computed_prompt_tokens",
                           _count(self.computed_prompt_tokens, "computed_prompt_tokens"))
        if self.computed_prompt_tokens > self.prompt_tokens:
            raise ValueError("computed_prompt_tokens cannot exceed prompt_tokens")

    @property
    def remaining_prompt_tokens(self) -> int:
        return self.prompt_tokens - self.computed_prompt_tokens


def merge_prefill_work(observed: Iterable[PrefillWork],
                       reservations: Iterable[PrefillWork]) -> tuple[PrefillWork, ...]:
    """Deduplicate by identity before dropping decode-only observations.

    ``observed`` must contain all snapshot requests, including completed
    prefills, and must use committed rather than planned in-flight progress.
    Freshness checks belong to the snapshot adapter.
    """
    merged = {}
    for reservation in reservations:
        if reservation.request_id in merged:
            raise ValueError("duplicate request in reservations")
        merged[reservation.request_id] = reservation
    observed_ids = set()
    for work in observed:
        if work.request_id in observed_ids:
            raise ValueError("duplicate request in snapshot")
        observed_ids.add(work.request_id)
        merged[work.request_id] = work
    return tuple(work for work in merged.values() if work.remaining_prompt_tokens > 0)


def select_mooncake(prefill_work: Mapping[str, Iterable[PrefillWork]],
                    incoming_prompt_tokens: int,
                    prefill_estimators: Mapping[str, Callable[[int, int], float]],
                    rng: random.Random) -> PolicyDecision:
    if set(prefill_work) != set(prefill_estimators):
        raise ValueError("prefill work and estimators must cover exactly the same pool")
    incoming_tokens = _count(incoming_prompt_tokens, "incoming_prompt_tokens")
    terms = {}
    for instance_id, work_items in prefill_work.items():
        estimate = prefill_estimators[instance_id]
        queued_ms = 0.0
        queued_tokens = 0
        ids = set()
        for work in work_items:
            if work.request_id in ids:
                raise ValueError("duplicate prefill request; reconcile snapshots and reservations first")
            ids.add(work.request_id)
            if work.remaining_prompt_tokens:
                queued_ms += _nonnegative(estimate(work.prompt_tokens, work.computed_prompt_tokens),
                                          "queued_prefill_ms")
                queued_tokens += work.remaining_prompt_tokens
        incoming_ms = (0.0 if incoming_tokens == 0 else
                       _nonnegative(estimate(incoming_tokens, 0), "incoming_prefill_ms"))
        total_ms = _nonnegative(queued_ms + incoming_ms, "total_prefill_ms")
        terms[instance_id] = {
            "queued_prefill_ms": queued_ms, "incoming_prefill_ms": incoming_ms,
            "total_prefill_ms": total_ms, "queued_prefill_tokens": queued_tokens,
            "cache_transfer_ms": 0.0,
        }
    return _select(terms, "total_prefill_ms", rng)


@dataclass(frozen=True)
class RouteBalanceWeights:
    quality: float = 1.0 / 3.0
    cost: float = 1.0 / 3.0
    latency: float = 1.0 / 3.0

    def __post_init__(self) -> None:
        for name in ("quality", "cost", "latency"):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        if not math.isclose(self.quality + self.cost + self.latency, 1.0,
                            rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("RouteBalance weights must sum to one")


@dataclass(frozen=True)
class RouteBalanceCandidate:
    predicted_quality: float
    predicted_cost: float
    predicted_latency_ms: float
    diagnostics: Mapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("predicted_quality", "predicted_cost", "predicted_latency_ms"):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        if self.predicted_quality > 1.0:
            raise ValueError("RouteBalance quality must be within [0, 1]")


def routebalance_cost(*, prompt_tokens: int, predicted_output_tokens: float,
                       input_token_price: float, output_token_price: float) -> float:
    return _nonnegative(
        _count(prompt_tokens, "prompt_tokens") * _nonnegative(input_token_price, "input_token_price")
        + _nonnegative(predicted_output_tokens, "predicted_output_tokens")
        * _nonnegative(output_token_price, "output_token_price"), "predicted_cost")


def estimate_routebalance_latency_ms(*, tpot_ms: float, pending_decode_tokens: float,
                                     decode_batch_size: int, predicted_output_tokens: float,
                                     free_decode_slot: bool = False,
                                     free_slot_reason: str = "conservative_no_evidence"
                                     ) -> tuple[float, dict]:
    tpot = _positive(tpot_ms, "tpot_ms")
    pending = _nonnegative(pending_decode_tokens, "pending_decode_tokens")
    batch = _count(decode_batch_size, "decode_batch_size")
    length = _nonnegative(predicted_output_tokens, "predicted_output_tokens")
    if not isinstance(free_decode_slot, bool):
        raise ValueError("free_decode_slot must be a boolean")
    if free_decode_slot and not free_slot_reason:
        raise ValueError("free-slot evidence/proxy must be named")
    # A prefill-only engine can have outstanding predicted decode work but no
    # active decode sequence. Use one as an explicitly logged conservative
    # divisor; never divide by zero or infer admission from max_num_seqs.
    effective_batch = max(batch, 1)
    waiting_steps = 0.0 if free_decode_slot else pending / effective_batch
    latency_ms = _nonnegative(tpot * (waiting_steps + length), "predicted_latency_ms")
    return latency_ms, {
        "tpot_ms": tpot, "pending_decode_tokens": pending,
        "decode_batch_size": batch, "effective_decode_batch_size": effective_batch,
        "empty_decode_batch_proxy": batch == 0 and pending > 0 and not free_decode_slot,
        "free_decode_slot": free_decode_slot, "free_slot_reason": free_slot_reason,
        "waiting_decode_steps": waiting_steps, "predicted_output_tokens": length,
        "predicted_latency_ms": latency_ms,
    }


def select_routebalance(candidates: Mapping[str, RouteBalanceCandidate],
                        weights: RouteBalanceWeights,
                        rng: random.Random) -> PolicyDecision:
    if not candidates:
        raise ValueError("at least one candidate is required")
    max_cost = max(candidate.predicted_cost for candidate in candidates.values())
    max_latency = max(candidate.predicted_latency_ms for candidate in candidates.values())
    terms = {}
    for instance_id, candidate in candidates.items():
        cost_benefit = 1.0 - candidate.predicted_cost / max_cost if max_cost else 1.0
        latency_benefit = 1.0 - candidate.predicted_latency_ms / max_latency if max_latency else 1.0
        terms[instance_id] = {
            **candidate.diagnostics,
            "predicted_quality": candidate.predicted_quality,
            "predicted_cost": candidate.predicted_cost,
            "predicted_latency_ms": candidate.predicted_latency_ms,
            "max_candidate_cost": max_cost, "max_candidate_latency_ms": max_latency,
            "normalized_cost_benefit": cost_benefit,
            "normalized_latency_benefit": latency_benefit,
            "routebalance_score": (weights.quality * candidate.predicted_quality
                                   + weights.cost * cost_benefit
                                   + weights.latency * latency_benefit),
        }
    return _select(terms, "routebalance_score", rng, maximize=True)


def routebalance_lpt_order(predicted_lengths: Mapping[str, Mapping[str, float]]) -> list[str]:
    keys = {}
    for request_id, predictions in predicted_lengths.items():
        if not predictions:
            raise ValueError("each request needs at least one predicted output length")
        keys[request_id] = max(_nonnegative(length, "predicted_output_tokens")
                               for length in predictions.values())
    # Python's stable sort preserves collection order for equal LPT keys.
    return sorted(keys, key=keys.__getitem__, reverse=True)

#!/usr/bin/env python3
"""Experiment runner for router baselines, wait-estimator GoF, and batch-fit GoF."""

from __future__ import annotations

import argparse
import asyncio
import csv
import importlib
import importlib.util
import json
import logging
import math
import os
import random
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional

_THIS_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _THIS_DIR.parent.parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.append(str(_WORKSPACE_ROOT))
    
DEFAULT_TOKENIZER_ID = "Qwen/Qwen3-8B"

from sfs_core.routing.wait_time_scheduler import (
    InstanceClient,
    RoutedRequest,
    WaitTimeResult,
    WaitTimeScheduler,
)
from sfs_core.routing.pending_dispatch_ledger import PendingDispatch
from sfs_core.routing.score_proxy import (
    estimate_ttft_ms as estimate_score_proxy_ttft_ms,
    hard_slo_candidate_value,
)
from sfs_core.routing.score_policy import (
    ScorePolicyState,
    estimate_total_response_latency_ms as estimate_score_total_latency_ms,
    score_candidate_terms as published_score_candidate_terms,
)
from vllm.v1.engine.scheduler_simulator import SimulationStopMode

from sfs_core.shared.shared_experiment_helpers import (
    as_nonnegative_int as _as_nonnegative_int,
    build_messages,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_RANDOM_SEED,
    get_prompt_bucket_files,
    _metric_summary,
    iter_bucketed_prompts,
    iter_mixed_then_random_bucketed_prompts,
    iter_random_bucketed_prompts,
    parse_chat_template_kwargs_json,
    resolve_chat_template_kwargs,
    select_prompt_subset,
    warm_up_instances,
)
from sfs_core.shared.tokenizer_helpers import TOKENIZER_MODES
from sfs_core.paths import (
    BUCKETED_OUTPUTS_ROOT,
    DEFAULT_BUCKET_POOL_QWEN3_0_6B,
    EXPERIMENTS_ROOT,
    PROMPTS_DATA_ROOT,
)
from sfs_core.prep.holdout_cache import prepare_holdout_prompt_cache
from sfs_core.regression.two_part_fit import (
    fit_two_part as _fit_two_part,
    fit_two_part_from_df as _fit_two_part_from_df,
    summarize_two_part_fit as _summarize_two_part_fit,
    DEFAULT_FEATURE_SET as _BATCH_FIT_DEFAULT_FEATURE_SET,
)

from sfs_core.eval.fit_utils import (
    build_tradeoff_summary,
    _fit_error_metrics,
    _add_inlier_r2,
    _plot_batch_fit_3d,
    _plot_batch_fit_parity,
    _read_batch_fit_df,
    _resolve_batch_fit_paths,
    _write_batch_fit_predictions,
    _evaluate_batch_fit_df,
    _plot_wait_fit,
)

SCRIPT_DIR = _THIS_DIR
DEFAULT_OUTPUT_DIR = EXPERIMENTS_ROOT
DEFAULT_PROMPT_BUCKET_DIR = BUCKETED_OUTPUTS_ROOT / "qwen3-0.6b" / "outputs"
DEFAULT_AFFINITY_SCORED_ROOT = BUCKETED_OUTPUTS_ROOT
DEFAULT_HOLDOUT_BUCKET_DIR = DEFAULT_BUCKET_POOL_QWEN3_0_6B
DEFAULT_HOLDOUT_CACHE_DIR = PROMPTS_DATA_ROOT / "holdout_cache"
DEFAULT_BATCH_FIT_GLOB = "batches_qwen3_*_4_datasets.csv"
DEFAULT_BATCH_FIT_SEARCH_DIR = EXPERIMENTS_ROOT
DEFAULT_INSTANCE_COSTS: Dict[str, Dict[str, float]] = {
    "vllm-0.6b": {"prompt": 0.044, "output": 0.173},
    "vllm-8b": {"prompt": 0.072, "output": 0.287},
    "vllm-32b": {"prompt": 0.287, "output": 0.64},
}
BUILTIN_UTILITIES = (
    "soft",
    "soft_prefill_tps",
    "soft_pk_mg1",
    "min_wait",
    "hard",
    "hard_prefill_tps",
    "hard_score_proxy",
    "score",
    "vllm_sr_latency",
    "lmdeploy_proxy",
    "mooncake_prefill",
    "routebalance",
    "hard_pk_mg1",
    "slo_aware",
    "latency_agnostic",
    "latency_and_cost_agnostic",
    "round_robin",
    "shortest_queue",
    "instance_affinity",
)
DEFAULT_ROUTER_POLICIES = (
    "soft",
    "latency_agnostic",
    "latency_and_cost_agnostic",
    "round_robin",
)
BUILTIN_WAIT_ESTIMATORS = (
    "live",
    "pk_mg1",
    "prefill_tps_ttft",
    "score_proxy_ttft",
)
EXPERIMENT_MODES = ("router", "wait_gof", "batch_fit", "all")
ARRIVAL_PROCESSES = ("poisson", "deterministic", "mmpp2")
# Arrival generator provenance recorded per run: absolute schedule (sleep until
# origin + cumulative sampled offset) rather than per-gap relative sleeps.
ARRIVAL_TIMING = "absolute_schedule"
# Decoupled arrivals are generated on a dedicated thread so that stream
# parsing or routing work on the event loop cannot starve the arrival timer.
ARRIVAL_TIMING_THREAD = "absolute_schedule_thread"
FEASIBLE_SLO_MODES = ("queue", "ttft", "e2e")
SHORTEST_QUEUE_SENTINEL = 10**12
DEFAULT_AFFINITY_QUALITY_EPSILON = 0.40
DEFAULT_AFFINITY_UPGRADE_MARGIN = 0.40
DEFAULT_MMPP2_RATE_RATIO = 4.0
DEFAULT_MMPP2_HIGH_FRACTION = 0.20
DEFAULT_MMPP2_CORRELATION_TIME_S = 2.0
QUEUE_PATTERN = re.compile(r"queue_ms=([0-9.+-eE]+)")
REQUEST_ID_PATTERN = re.compile(r"request_id=([^\s]+)")
TTFT_PATTERN = re.compile(r"ttft_s=([0-9.+-eE]+)")
PREFILL_PATTERN = re.compile(r"prefill_s=([0-9.+-eE]+)")
INFERENCE_PATTERN = re.compile(r"inference_s=([0-9.+-eE]+)")
QUEUED_TS_PATTERN = re.compile(r"queued_ts_s=([0-9.+-eE]+)")
FIRST_TOKEN_TS_PATTERN = re.compile(r"first_token_ts_s=([0-9.+-eE]+)")
LOGGER = logging.getLogger(__name__)

UtilityCallable = Callable[
    [str, Dict[str, WaitTimeResult], Dict[str, float], Dict[str, float], int],
    float,
]
WaitEstimatorCallable = Callable[
    [str, InstanceClient, Dict[str, WaitTimeResult], Dict[str, Any]],
    Optional[float],
]

TTFT_BATCH_PARAM_KEYS = (
    "max_num_batched_tokens",
    "max_num_seqs",
    "chunked_prefill_enabled",
    "long_prefill_token_threshold",
    "intercept",
    "prefill_coeff",
    "prefill_sq_coeff",
    "decode_coeff",
    "sum_coeff",
    "sum_sq_coeff",
)
TTFT_ESTIMATORS = {"live"}
TTFT_TARGET_ESTIMATORS = {
    "live",
    "prefill_tps_ttft",
    "score_proxy_ttft",
    "pk_mg1",
}
PREFILL_TPS_TTFT_ESTIMATOR_NAME = "prefill_tps_ttft"
SCORE_PROXY_TTFT_ESTIMATOR_NAME = "score_proxy_ttft"
SCORE_TOTAL_LATENCY_ESTIMATOR_NAME = "score_total_latency"
SNAPSHOT_ONLY_WAIT_ESTIMATORS = {
    PREFILL_TPS_TTFT_ESTIMATOR_NAME,
    SCORE_PROXY_TTFT_ESTIMATOR_NAME,
    SCORE_TOTAL_LATENCY_ESTIMATOR_NAME,
    "pk_mg1",
}
DEFAULT_PREFILL_TPS: Dict[str, float] = {
    "qwen3-0.6b": 130313.18274213851,
    "qwen3-8b": 32954.46194008248,
    "qwen3-32b": 14842.397169498861,
}
DEFAULT_SERVICE_RATES_RPS: Dict[str, float] = {
    "qwen3-0.6b": 4.280629753725604,
    "qwen3-8b": 1.8026517307491159,
    "qwen3-32b": 1.2172821447358584,
}
PREFILL_TPS_ALIASES: Dict[str, str] = {
    "vllm-0.6b": "qwen3-0.6b",
    "vllm-8b": "qwen3-8b",
    "vllm-32b": "qwen3-32b",
    "qwen3-0.6b": "qwen3-0.6b",
    "qwen3-8b": "qwen3-8b",
    "qwen3-32b": "qwen3-32b",
}

def _should_include_unconditional_live_fetch(wait_estimator_name: str) -> bool:
    return str(wait_estimator_name).strip().lower() not in SNAPSHOT_ONLY_WAIT_ESTIMATORS


def _extract_usage_tokens(response: Any) -> Dict[str, Optional[int]]:
    if isinstance(response, dict):
        usage = response.get("usage")
    else:
        usage = getattr(response, "usage", None)

    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
    elif usage is not None:
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
    else:
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None

    return {
        "usage_prompt_tokens": _as_nonnegative_int(prompt_tokens),
        "usage_completion_tokens": _as_nonnegative_int(completion_tokens),
        "usage_total_tokens": _as_nonnegative_int(total_tokens),
    }


@dataclass(slots=True)
class UtilityState:
    current_slo_ms: Optional[float] = None
    round_robin_cursor: int = 0
    score_latency_limit_ms: Optional[float] = None
    score_policy_state: Optional[ScorePolicyState] = None


@dataclass(slots=True)
class ExperimentRequest:
    request_id: str
    prompt: str
    prompt_tokens: int
    bucket: str
    latency_slo_ms: float
    queue_slo_ms: float
    ttft_slo_ms: float


@dataclass(slots=True)
class RequestLatencyComponents:
    queue_ms: Optional[float] = None
    frontend_ttft_ms: Optional[float] = None
    prefill_ms: Optional[float] = None
    queued_ts_s: Optional[float] = None
    first_token_ts_s: Optional[float] = None


@dataclass(slots=True)
class PKInstanceStats:
    arrival_rate_rps: float = 0.0
    last_dispatch_ts: Optional[float] = None
    arrival_samples: int = 0
    arrival_window_dispatch_ts: deque[float] = field(default_factory=deque)
    service_rate_rps: float = 0.0
    service_mean_s: float = 0.0
    service_mean_sq_s2: float = 0.0
    service_samples: int = 0
    last_service_ts: Optional[float] = None
    service_window_completion_ts: deque[float] = field(default_factory=deque)
    estimate_calls: int = 0
    fallback_counts: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MMPP2DerivedParams:
    request_rate_qps: float
    rate_ratio: float
    high_fraction: float
    correlation_time_s: float
    lambda_low_rps: float
    lambda_high_rps: float
    q_low_to_high_rps: float
    q_high_to_low_rps: float


class PKOnlineStatsCollector:
    def __init__(
        self,
        *,
        instance_ids: list[str],
        per_request_wait_logs: list[Path],
        arrival_ewma_alpha: float = 0.2,
        service_ewma_alpha: Optional[float] = None,
        arrival_rate_window_s: float = 1.0,
        service_rate_window_s: Optional[float] = None,
        service_mu_prior_rps_by_instance: Optional[Dict[str, float]] = None,
        min_service_samples: int = 16,
        service_staleness_s: float = 120.0,
        rho_cap: float = 0.98,
        max_wait_ms: float = 60000.0,
        default_service_s: float = 0.05,
    ) -> None:
        self._states = {instance_id: PKInstanceStats() for instance_id in instance_ids}
        self._log_paths = list(per_request_wait_logs)
        self._offsets = _snapshot_log_offsets(self._log_paths)
        self._response_to_instance: dict[str, str] = {}
        self._arrival_alpha = float(max(min(arrival_ewma_alpha, 1.0), 0.01))
        resolved_service_alpha = (
            arrival_ewma_alpha if service_ewma_alpha is None else service_ewma_alpha
        )
        self._service_alpha = float(max(min(resolved_service_alpha, 1.0), 0.01))
        self._arrival_rate_window_s = max(float(arrival_rate_window_s), 1e-3)
        resolved_service_window_s = (
            arrival_rate_window_s
            if service_rate_window_s is None
            else service_rate_window_s
        )
        self._service_rate_window_s = max(float(resolved_service_window_s), 1e-3)
        self._min_service_samples = max(int(min_service_samples), 1)
        self._service_staleness_s = max(float(service_staleness_s), 1.0)
        self._rho_cap = float(max(min(rho_cap, 0.999), 0.5))
        self._max_wait_ms = max(float(max_wait_ms), 1.0)
        self._default_service_s = max(float(default_service_s), 1e-4)
        self._service_mu_prior_rps_by_instance: Dict[str, float] = {}
        if isinstance(service_mu_prior_rps_by_instance, dict):
            for instance_id in instance_ids:
                prior_value = service_mu_prior_rps_by_instance.get(instance_id)
                if isinstance(prior_value, bool) or not isinstance(
                    prior_value, (int, float)
                ):
                    continue
                numeric_prior = float(prior_value)
                if not math.isfinite(numeric_prior) or numeric_prior <= 0:
                    continue
                self._service_mu_prior_rps_by_instance[instance_id] = numeric_prior
                state = self._states.get(instance_id)
                if state is not None:
                    state.service_rate_rps = numeric_prior
        self._num_ingested_service_lines = 0
        self._num_unmatched_service_lines = 0

    def note_dispatch(self, instance_id: str, now_s: Optional[float] = None) -> None:
        state = self._states.get(instance_id)
        if state is None:
            return
        now = float(now_s if now_s is not None else time.time())
        window = state.arrival_window_dispatch_ts
        window.append(now)
        cutoff_ts = now - self._arrival_rate_window_s
        while window and window[0] < cutoff_ts:
            window.popleft()

        window_rate_rps = float(len(window)) / self._arrival_rate_window_s
        if math.isfinite(window_rate_rps) and window_rate_rps > 0:
            if (
                state.arrival_samples <= 0
                or state.arrival_rate_rps <= 0
                or not math.isfinite(state.arrival_rate_rps)
            ):
                state.arrival_rate_rps = window_rate_rps
            else:
                alpha = self._arrival_alpha
                state.arrival_rate_rps = (alpha * window_rate_rps) + (
                    (1.0 - alpha) * state.arrival_rate_rps
                )
            state.arrival_samples += 1
        state.last_dispatch_ts = now

    def note_response(self, response_id: Optional[str], instance_id: str) -> None:
        if response_id and instance_id in self._states:
            self._response_to_instance[str(response_id)] = instance_id

    def _record_service_sample(self, instance_id: str, inference_s: float) -> None:
        if inference_s <= 0:
            return
        state = self._states.get(instance_id)
        if state is None:
            return
        n = state.service_samples + 1
        state.service_mean_s += (inference_s - state.service_mean_s) / n
        square = inference_s * inference_s
        state.service_mean_sq_s2 += (square - state.service_mean_sq_s2) / n
        state.service_samples = n
        now = time.time()
        state.last_service_ts = now

        service_window = state.service_window_completion_ts
        service_window.append(now)
        cutoff_ts = now - self._service_rate_window_s
        while service_window and service_window[0] < cutoff_ts:
            service_window.popleft()

        window_mu_rps = float(len(service_window)) / self._service_rate_window_s
        if math.isfinite(window_mu_rps) and window_mu_rps > 0:
            if state.service_rate_rps <= 0 or not math.isfinite(state.service_rate_rps):
                state.service_rate_rps = window_mu_rps
            else:
                alpha = self._service_alpha
                state.service_rate_rps = (alpha * window_mu_rps) + (
                    (1.0 - alpha) * state.service_rate_rps
                )

    def ingest_service_logs(self) -> None:
        if not self._log_paths:
            return
        for path in self._log_paths:
            if not path.exists():
                continue
            start_offset = self._offsets.get(path, 0)
            with path.open("r", encoding="utf-8", errors="ignore") as src:
                if start_offset > 0:
                    src.seek(start_offset)
                for line in src:
                    req_match = REQUEST_ID_PATTERN.search(line)
                    inf_match = INFERENCE_PATTERN.search(line)
                    if not req_match or not inf_match:
                        continue
                    request_id = req_match.group(1)
                    try:
                        inference_s = float(inf_match.group(1))
                    except ValueError:
                        LOGGER.warning(
                            "Skipping service-log line with non-numeric inference_s; "
                            "path=%s request_id=%s raw_inference_s=%r",
                            path,
                            request_id,
                            inf_match.group(1),
                        )
                        continue
                    instance_id = self._response_to_instance.pop(request_id, None)
                    if instance_id is None:
                        self._num_unmatched_service_lines += 1
                        continue
                    self._record_service_sample(instance_id, inference_s)
                    self._num_ingested_service_lines += 1
                self._offsets[path] = src.tell()

    def estimate_wait_ms(
        self,
        instance_id: str,
        *,
        prompt_tokens: int,
        prefill_tps: float,
        running_at_snapshot: Optional[int],
    ) -> tuple[float, dict[str, Any]]:
        self.ingest_service_logs()
        prompt_tokens_value = max(int(prompt_tokens), 0)
        prefill_tps_value = max(float(prefill_tps), 1e-9)
        prefill_term_ms = float(prompt_tokens_value / prefill_tps_value * 1000.0)
        if not math.isfinite(prefill_term_ms) or prefill_term_ms < 0.0:
            prefill_term_ms = 0.0
        state = self._states.get(instance_id)
        if state is None:
            adjustments = ["unknown_instance"]
            if running_at_snapshot is None:
                adjustments.append("k_missing_default_zero")
            return prefill_term_ms, {
                "method": "pk_mg1",
                "instance_id": instance_id,
                "fallback_reason": None,
                "lambda_rps": 0.0,
                "alpha_rps": 0.0,
                "mu_rps": 1.0 / self._default_service_s,
                "mu_prior_rps": None,
                "mu_ewma_rps": None,
                "mu_online_sample_rps": None,
                "mu_minus_alpha_rps": 1.0 / self._default_service_s,
                "e_s": self._default_service_s,
                "e_s2": self._default_service_s * self._default_service_s,
                "rho": 0.0,
                "k_running_at_snapshot": int(running_at_snapshot or 0),
                "running_at_snapshot_input": running_at_snapshot,
                "prompt_tokens": int(prompt_tokens_value),
                "prefill_tps": float(prefill_tps_value),
                "pk_wait_ms": 0.0,
                "pk_queue_term_ms": 0.0,
                "pk_prefill_term_ms": float(prefill_term_ms),
                "estimated_ttft_ms": float(prefill_term_ms),
                "service_samples": 0,
                "arrival_samples": 0,
                "estimate_calls": 0,
                "adjustments": adjustments,
            }

        state.estimate_calls += 1
        now = time.time()
        adjustments: list[str] = []
        lambda_rps = state.arrival_rate_rps if state.arrival_rate_rps > 0 else 0.0
        if state.arrival_samples <= 0:
            adjustments.append("lambda_uninitialized_zero")
        e_s_sample = state.service_mean_s
        e_s2 = state.service_mean_sq_s2
        if state.service_samples < self._min_service_samples:
            adjustments.append("service_samples_below_threshold")
        if state.last_service_ts is None:
            adjustments.append("service_uninitialized_reused_prior")
        elif (now - state.last_service_ts) > self._service_staleness_s:
            adjustments.append("service_stale_reused")

        mu_prior_rps = self._service_mu_prior_rps_by_instance.get(instance_id)
        mu_online_sample_rps: Optional[float] = None
        if e_s_sample > 0 and math.isfinite(e_s_sample):
            mu_online_sample_rps = 1.0 / e_s_sample
            if not math.isfinite(mu_online_sample_rps) or mu_online_sample_rps <= 0:
                mu_online_sample_rps = None
        mu_ewma_rps = (
            float(state.service_rate_rps)
            if state.service_rate_rps > 0 and math.isfinite(state.service_rate_rps)
            else None
        )

        if mu_ewma_rps is not None:
            mu_rps = float(mu_ewma_rps)
        elif mu_online_sample_rps is not None:
            adjustments.append("mu_ewma_missing_using_sample_mean")
            mu_rps = float(mu_online_sample_rps)
        elif mu_prior_rps is not None and mu_prior_rps > 0:
            adjustments.append("mu_ewma_missing_using_prior")
            mu_rps = float(mu_prior_rps)
        else:
            adjustments.append("mu_non_finite_clamped")
            mu_rps = 1.0 / self._default_service_s

        e_s = 1.0 / mu_rps
        if e_s2 <= 0 or not math.isfinite(e_s2):
            adjustments.append("e_s2_clamped_square")
            e_s2 = e_s * e_s
        if e_s2 < (e_s * e_s):
            adjustments.append("e_s2_raised_to_square")
            e_s2 = e_s * e_s

        alpha_rps = lambda_rps
        rho = alpha_rps * e_s
        if not math.isfinite(rho):
            adjustments.append("rho_non_finite_zero")
            rho = 0.0

        k = running_at_snapshot
        if k is None or int(k) < 0:
            adjustments.append("k_missing_default_zero")
            k = 0
        k_value = int(k)

        utilization_ratio = alpha_rps / mu_rps
        if not math.isfinite(utilization_ratio):
            adjustments.append("alpha_over_mu_non_finite_zero")
            utilization_ratio = 0.0
        elif utilization_ratio < 0.0:
            adjustments.append("alpha_over_mu_negative_zero")
            utilization_ratio = 0.0

        mu_minus_alpha = mu_rps - alpha_rps
        if mu_minus_alpha <= 0.0:
            adjustments.append("mu_minus_alpha_clamped")
            state.fallback_counts["mu_minus_alpha_clamped"] = state.fallback_counts.get(
                "mu_minus_alpha_clamped", 0
            ) + 1
            wait_q_ms = float("inf")
        else:
            denominator_rps = max(mu_minus_alpha, 1e-6)
            wait_q_s = math.pow(utilization_ratio, k_value) / denominator_rps
            wait_q_ms = wait_q_s * 1000.0
        if math.isnan(wait_q_ms) or wait_q_ms < 0.0:
            adjustments.append("queue_term_non_finite_zero")
            state.fallback_counts["queue_term_non_finite_zero"] = state.fallback_counts.get(
                "queue_term_non_finite_zero", 0
            ) + 1
            wait_q_ms = 0.0
        if math.isfinite(wait_q_ms) and wait_q_ms > self._max_wait_ms:
            adjustments.append("queue_term_capped_max_wait")
            wait_q_ms = self._max_wait_ms
        elif math.isinf(wait_q_ms) and wait_q_ms > 0.0:
            adjustments.append("queue_term_infinite_overload")
        wait_q_ms = max(wait_q_ms, 0.0) if math.isfinite(wait_q_ms) else wait_q_ms

        if not math.isfinite(prefill_term_ms):
            adjustments.append("prefill_term_non_finite_zero")
            prefill_term_ms = 0.0
        prefill_term_ms = max(prefill_term_ms, 0.0)

        wait_ttft_ms = wait_q_ms + prefill_term_ms
        if math.isnan(wait_ttft_ms) or wait_ttft_ms < 0.0:
            adjustments.append("ttft_non_finite_zero")
            state.fallback_counts["ttft_non_finite_zero"] = state.fallback_counts.get(
                "ttft_non_finite_zero", 0
            ) + 1
            wait_ttft_ms = 0.0

        return wait_ttft_ms, {
            "method": "pk_mg1",
            "instance_id": instance_id,
            "fallback_reason": None,
            "lambda_rps": lambda_rps,
            "alpha_rps": alpha_rps,
            "mu_rps": mu_rps,
            "mu_prior_rps": mu_prior_rps,
            "mu_ewma_rps": mu_ewma_rps,
            "mu_online_sample_rps": mu_online_sample_rps,
            "mu_minus_alpha_rps": mu_minus_alpha,
            "e_s": e_s,
            "e_s2": e_s2,
            "rho": rho,
            "rho_raw": rho,
            "k_running_at_snapshot": k_value,
            "running_at_snapshot_input": running_at_snapshot,
            "prompt_tokens": int(prompt_tokens_value),
            "prefill_tps": float(prefill_tps_value),
            "service_samples": state.service_samples,
            "arrival_samples": state.arrival_samples,
            "estimate_calls": state.estimate_calls,
            "pk_wait_ms": wait_q_ms,
            "pk_queue_term_ms": wait_q_ms,
            "pk_prefill_term_ms": prefill_term_ms,
            "estimated_ttft_ms": wait_ttft_ms,
            "adjustments": adjustments,
        }

    def summary(self) -> dict[str, Any]:
        instances: dict[str, Any] = {}
        for instance_id, state in self._states.items():
            rho = (
                state.arrival_rate_rps * state.service_mean_s
                if state.arrival_rate_rps > 0 and state.service_mean_s > 0
                else None
            )
            ewma_rho = (
                state.arrival_rate_rps / state.service_rate_rps
                if state.arrival_rate_rps > 0 and state.service_rate_rps > 0
                else None
            )
            instances[instance_id] = {
                "lambda_rps": state.arrival_rate_rps if state.arrival_rate_rps > 0 else None,
                "mu_prior_rps": self._service_mu_prior_rps_by_instance.get(instance_id),
                "mu_ewma_rps": state.service_rate_rps if state.service_rate_rps > 0 else None,
                "e_s": state.service_mean_s if state.service_samples > 0 else None,
                "e_s2": state.service_mean_sq_s2 if state.service_samples > 0 else None,
                "rho": rho,
                "rho_ewma_mu": ewma_rho,
                "arrival_samples": state.arrival_samples,
                "service_samples": state.service_samples,
                "estimate_calls": state.estimate_calls,
                "last_service_ts": state.last_service_ts,
                "fallback_counts": dict(state.fallback_counts),
            }
        return {
            "arrival_ewma_alpha": self._arrival_alpha,
            "service_ewma_alpha": self._service_alpha,
            "arrival_rate_window_s": self._arrival_rate_window_s,
            "service_rate_window_s": self._service_rate_window_s,
            "min_service_samples": self._min_service_samples,
            "service_staleness_s": self._service_staleness_s,
            "rho_cap": self._rho_cap,
            "max_wait_ms": self._max_wait_ms,
            "default_service_s": self._default_service_s,
            "num_ingested_service_lines": self._num_ingested_service_lines,
            "num_unmatched_service_lines": self._num_unmatched_service_lines,
            "instances": instances,
        }


class CollectingWaitTimeScheduler(WaitTimeScheduler):
    """WaitTimeScheduler wrapper that exposes per-request completion metadata."""

    def __init__(
        self,
        *args: Any,
        utility_state: UtilityState,
        wait_estimator: Optional[WaitEstimatorCallable] = None,
        wait_estimator_name: str = "live",
        wait_estimator_context: Optional[Dict[str, Any]] = None,
        wait_estimators: Optional[list[tuple[str, WaitEstimatorCallable]]] = None,
        wait_estimator_contexts: Optional[Dict[str, Dict[str, Any]]] = None,
        route_strategy: str = "utility",
        route_random_seed: Optional[int] = None,
        affinity_bucket_to_instances: Optional[Dict[str, list[str]]] = None,
        affinity_global_fallback_instances: Optional[list[str]] = None,
        affinity_upgrade_margin: float = DEFAULT_AFFINITY_UPGRADE_MARGIN,
        score_cost_weight: float = 1.0,
        score_latency_weight: float = 1.0,
        skip_wait_result_build: bool = False,
        include_unconditional_live_fetch: bool = True,
        readiness_diagnostics: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._utility_state = utility_state
        self._wait_estimator = wait_estimator
        self._wait_estimator_name = wait_estimator_name
        self._wait_estimator_context = dict(wait_estimator_context or {})
        self._wait_estimators: list[tuple[str, WaitEstimatorCallable]] = []
        self._wait_estimator_contexts: Dict[str, Dict[str, Any]] = {}
        self._skip_wait_result_build = bool(skip_wait_result_build)
        self._include_unconditional_live_fetch = bool(include_unconditional_live_fetch)
        self._readiness_diagnostics = bool(readiness_diagnostics)
        self._route_rng = random.Random(route_random_seed)
        self._shortest_queue_sentinel = int(SHORTEST_QUEUE_SENTINEL)
        self._last_seen_num_requests_by_instance: Dict[str, int] = {}
        self._affinity_bucket_to_instances: Dict[str, list[str]] = {}
        for bucket, instance_ids in (affinity_bucket_to_instances or {}).items():
            normalized_bucket = _normalize_bucket_name(bucket)
            filtered_ids = [
                str(instance_id)
                for instance_id in instance_ids
                if str(instance_id) in self._instances
            ]
            if filtered_ids:
                self._affinity_bucket_to_instances[normalized_bucket] = list(
                    dict.fromkeys(filtered_ids)
                )
        self._affinity_global_fallback_instances = [
            str(instance_id)
            for instance_id in (affinity_global_fallback_instances or [])
            if str(instance_id) in self._instances
        ]
        self._affinity_global_fallback_instances = list(
            dict.fromkeys(self._affinity_global_fallback_instances)
        )
        self._affinity_upgrade_margin = max(float(affinity_upgrade_margin), 0.0)
        self._score_cost_weight = float(score_cost_weight)
        self._score_latency_weight = float(score_latency_weight)

        if wait_estimators:
            for estimator_name, estimator_fn in wait_estimators:
                if not callable(estimator_fn):
                    raise TypeError(
                        f"Wait estimator '{estimator_name}' is not callable."
                    )
                self._wait_estimators.append((str(estimator_name), estimator_fn))

            for estimator_name, _ in self._wait_estimators:
                context_payload: Dict[str, Any] = {}
                if wait_estimator_contexts:
                    source_context = wait_estimator_contexts.get(estimator_name)
                    if isinstance(source_context, dict):
                        context_payload.update(source_context)
                if estimator_name == wait_estimator_name:
                    context_payload.update(self._wait_estimator_context)
                self._wait_estimator_contexts[estimator_name] = context_payload

            if self._wait_estimator is None and self._wait_estimators:
                self._wait_estimator_name, self._wait_estimator = self._wait_estimators[0]

            if (
                self._wait_estimator is not None
                and self._wait_estimator_name not in self._wait_estimator_contexts
            ):
                self._wait_estimators.insert(
                    0,
                    (self._wait_estimator_name, self._wait_estimator),
                )
                self._wait_estimator_contexts[self._wait_estimator_name] = dict(
                    self._wait_estimator_context
                )
        elif self._wait_estimator is not None:
            self._wait_estimators = [(self._wait_estimator_name, self._wait_estimator)]
            self._wait_estimator_contexts[self._wait_estimator_name] = dict(
                self._wait_estimator_context
            )

        for context_payload in self._wait_estimator_contexts.values():
            diagnostics_map = context_payload.get("wait_estimator_diagnostics_by_instance")
            if not isinstance(diagnostics_map, dict):
                context_payload["wait_estimator_diagnostics_by_instance"] = {}

        self._route_strategy = route_strategy.strip().lower()
        if self._route_strategy not in {
            "utility",
            "round_robin",
            "shortest_queue",
            "instance_affinity",
        }:
            raise ValueError(
                f"Unsupported route_strategy={route_strategy!r}. "
                "Use 'utility', 'round_robin', 'shortest_queue', or "
                "'instance_affinity'."
            )
        if self._route_strategy == "instance_affinity":
            if (
                not self._affinity_bucket_to_instances
                and not self._affinity_global_fallback_instances
            ):
                raise ValueError(
                    "route_strategy='instance_affinity' requires affinity mapping "
                    "data (bucket preferences or global fallback instances)."
                )

    def _iter_pk_trackers(self):
        seen_ids: set[int] = set()
        for context in self._wait_estimator_contexts.values():
            tracker = context.get("pk_tracker")
            if not isinstance(tracker, PKOnlineStatsCollector):
                continue
            tracker_id = id(tracker)
            if tracker_id in seen_ids:
                continue
            seen_ids.add(tracker_id)
            yield tracker

    def _attach_readiness_predictor_inputs(
        self,
        *,
        wait_record: WaitTimeResult,
        prompt_tokens: int,
        pending_dispatches: tuple[PendingDispatch, ...],
    ) -> None:
        if not self._readiness_diagnostics:
            return
        wait_record.raw_payload["_readiness_predictor_inputs"] = {
            "prompt_tokens": int(prompt_tokens),
            "pending_dispatch_count": len(pending_dispatches),
        }

    @staticmethod
    def _wait_fetch_args_for_estimator(
        estimator_name: str,
        *,
        prompt_tokens: int,
    ) -> tuple[Optional[int], Optional[SimulationStopMode]]:
        prompt_tokens_override = prompt_tokens if prompt_tokens > 0 else None
        estimator_key = str(estimator_name).strip().lower()
        if estimator_key in SNAPSHOT_ONLY_WAIT_ESTIMATORS:
            return None, None
        if prompt_tokens_override is None:
            return None, None
        return prompt_tokens_override, SimulationStopMode.PREFILL_DONE

    @staticmethod
    def _wait_fetch_cache_key(
        *,
        prompt_tokens: Optional[int],
        stop_mode: Optional[SimulationStopMode],
    ) -> str:
        if prompt_tokens is None:
            return "snapshot_only"
        resolved_stop_mode = SimulationStopMode.from_value(
            stop_mode,
            default=SimulationStopMode.PREFILL_DONE,
        )
        return f"{int(prompt_tokens)}:{resolved_stop_mode.value}"

    def _build_zero_wait_results(
        self,
        *,
        reason: str,
    ) -> Dict[str, WaitTimeResult]:
        now_s = time.time()
        return {
            instance_id: WaitTimeResult(
                instance_id=instance_id,
                wait_ms=0.0,
                fetched_at_s=now_s,
                raw_payload={
                    "_wait_estimator": self._wait_estimator_name,
                    "_wait_estimator_skipped": True,
                    "_wait_estimator_skip_reason": reason,
                },
            )
            for instance_id in self._instances
        }

    def _estimate_wait_results(
        self,
        *,
        estimator_name: str,
        estimator_fn: WaitEstimatorCallable,
        base_wait_results: Dict[str, WaitTimeResult],
        live_wait_results: Dict[str, WaitTimeResult],
        context_base: Dict[str, Any],
    ) -> Dict[str, WaitTimeResult]:
        shared_context = self._wait_estimator_contexts.get(estimator_name, {})
        diagnostics_map = shared_context.get("wait_estimator_diagnostics_by_instance")
        if isinstance(diagnostics_map, dict):
            diagnostics_map.clear()

        context: Dict[str, Any] = dict(context_base)
        context.update(shared_context)
        now_s = time.time()
        estimated_wait_results: Dict[str, WaitTimeResult] = {}
        for instance_id, instance in self._instances.items():
            estimate_ms = estimator_fn(instance_id, instance, base_wait_results, context)
            base_wait = base_wait_results.get(instance_id)
            live_wait = live_wait_results.get(instance_id)
            source_wait = base_wait or live_wait
            if estimate_ms is None and source_wait is not None:
                estimate_ms = source_wait.wait_ms
            if estimate_ms is None:
                continue

            raw_payload = dict(source_wait.raw_payload) if source_wait else {}
            raw_payload["_wait_estimator"] = estimator_name
            if live_wait is not None:
                raw_payload["_live_wait_ms"] = float(live_wait.wait_ms)
            if base_wait is not None:
                raw_payload["_base_wait_ms"] = float(base_wait.wait_ms)
            if isinstance(diagnostics_map, dict):
                estimator_diag = diagnostics_map.get(instance_id)
                if isinstance(estimator_diag, dict):
                    raw_payload["_wait_estimator_diagnostics"] = dict(estimator_diag)

            estimated_wait_results[instance_id] = WaitTimeResult(
                instance_id=instance_id,
                wait_ms=float(estimate_ms),
                fetched_at_s=now_s,
                raw_payload=raw_payload,
            )

        if estimated_wait_results:
            return estimated_wait_results

        if base_wait_results:
            base_cloned: Dict[str, WaitTimeResult] = {}
            for instance_id, base_wait in base_wait_results.items():
                raw_payload = dict(base_wait.raw_payload) if base_wait else {}
                raw_payload["_wait_estimator"] = estimator_name
                raw_payload["_base_wait_ms"] = float(base_wait.wait_ms)
                live_wait = live_wait_results.get(instance_id)
                if live_wait is not None:
                    raw_payload["_live_wait_ms"] = float(live_wait.wait_ms)
                base_cloned[instance_id] = WaitTimeResult(
                    instance_id=instance_id,
                    wait_ms=float(base_wait.wait_ms),
                    fetched_at_s=now_s,
                    raw_payload=raw_payload,
                )
            return base_cloned

        if live_wait_results:
            live_cloned: Dict[str, WaitTimeResult] = {}
            for instance_id, live_wait in live_wait_results.items():
                raw_payload = dict(live_wait.raw_payload) if live_wait else {}
                raw_payload["_wait_estimator"] = estimator_name
                raw_payload["_live_wait_ms"] = float(live_wait.wait_ms)
                live_cloned[instance_id] = WaitTimeResult(
                    instance_id=instance_id,
                    wait_ms=float(live_wait.wait_ms),
                    fetched_at_s=now_s,
                    raw_payload=raw_payload,
                )
            return live_cloned

        return {
            instance_id: WaitTimeResult(
                instance_id=instance_id,
                wait_ms=0.0,
                fetched_at_s=now_s,
                raw_payload={
                    "_wait_estimator": estimator_name,
                    "_fallback": "zero",
                },
            )
            for instance_id in self._instances
        }

    async def _build_wait_results_for_request(
        self,
        *,
        queued_payload: Dict[str, Any],
        request_id: str,
        prompt_tokens: int,
        accuracy_scores: Dict[str, float],
        output_lengths: Dict[str, float],
        pending_dispatches_by_instance: Optional[
            Dict[str, tuple[PendingDispatch, ...]]
        ] = None,
        probe_ready_delay_ms_by_instance: Optional[Dict[str, float]] = None,
    ) -> tuple[
        Dict[str, WaitTimeResult],
        Dict[str, WaitTimeResult],
        Dict[str, Dict[str, WaitTimeResult]],
    ]:
        fetch_specs: Dict[str, tuple[Optional[int], Optional[SimulationStopMode]]] = {}
        live_fetch_args = self._wait_fetch_args_for_estimator(
            "live",
            prompt_tokens=prompt_tokens,
        )
        live_fetch_key = self._wait_fetch_cache_key(
            prompt_tokens=live_fetch_args[0],
            stop_mode=live_fetch_args[1],
        )
        if self._include_unconditional_live_fetch:
            fetch_specs[live_fetch_key] = live_fetch_args

        estimator_fetch_keys: Dict[str, str] = {}
        for estimator_name, _ in self._wait_estimators:
            fetch_args = self._wait_fetch_args_for_estimator(
                estimator_name,
                prompt_tokens=prompt_tokens,
            )
            fetch_key = self._wait_fetch_cache_key(
                prompt_tokens=fetch_args[0],
                stop_mode=fetch_args[1],
            )
            estimator_fetch_keys[estimator_name] = fetch_key
            fetch_specs[fetch_key] = fetch_args

        parent_collect_wait_times = super()._collect_wait_times
        fetch_tasks = {
            fetch_key: asyncio.create_task(
                parent_collect_wait_times(
                    prompt_tokens=fetch_args[0],
                    stop_mode=fetch_args[1],
                    pending_dispatches_by_instance=(pending_dispatches_by_instance),
                    probe_ready_delay_ms_by_instance=(
                        probe_ready_delay_ms_by_instance
                    ),
                )
            )
            for fetch_key, fetch_args in fetch_specs.items()
        }
        fetched_wait_results: Dict[str, Dict[str, WaitTimeResult]] = {}
        for fetch_key, task in fetch_tasks.items():
            fetched_wait_results[fetch_key] = await task
            self._reconcile_observed_dispatches(fetched_wait_results[fetch_key])

        if self._include_unconditional_live_fetch:
            live_wait_results = fetched_wait_results.get(live_fetch_key, {})
        else:
            primary_fetch_key = estimator_fetch_keys.get(self._wait_estimator_name)
            live_wait_results = (
                fetched_wait_results.get(primary_fetch_key, {})
                if primary_fetch_key is not None
                else {}
            )
        if not self._wait_estimators:
            return live_wait_results, live_wait_results, {}

        context_base: Dict[str, Any] = {
            "request_id": request_id,
            "payload": queued_payload,
            "prompt_tokens": prompt_tokens,
            "accuracy_scores": accuracy_scores,
            "output_lengths": output_lengths,
            "instances": self._instances,
            "live_wait_results": live_wait_results,
        }

        estimator_wait_results: Dict[str, Dict[str, WaitTimeResult]] = {}
        for estimator_name, estimator_fn in self._wait_estimators:
            base_wait_results = fetched_wait_results.get(
                estimator_fetch_keys.get(estimator_name, live_fetch_key),
                live_wait_results,
            )
            estimator_wait_results[estimator_name] = self._estimate_wait_results(
                estimator_name=estimator_name,
                estimator_fn=estimator_fn,
                base_wait_results=base_wait_results,
                live_wait_results=live_wait_results,
                context_base=context_base,
            )

        primary_wait_results = estimator_wait_results.get(self._wait_estimator_name)
        if primary_wait_results is None and estimator_wait_results:
            primary_wait_results = next(iter(estimator_wait_results.values()))

        if primary_wait_results is None:
            now_s = time.time()
            primary_wait_results = {
                instance_id: WaitTimeResult(
                    instance_id=instance_id,
                    wait_ms=0.0,
                    fetched_at_s=now_s,
                    raw_payload={
                        "_wait_estimator": self._wait_estimator_name,
                        "_fallback": "zero",
                    },
                )
                for instance_id in self._instances
            }

        return primary_wait_results, live_wait_results, estimator_wait_results

    def _select_round_robin_instance(self) -> str:
        instance_ids = list(self._instances.keys())
        if not instance_ids:
            raise RuntimeError("No instances configured for round-robin selection.")
        cursor = self._utility_state.round_robin_cursor % len(instance_ids)
        self._utility_state.round_robin_cursor += 1
        return instance_ids[cursor]

    def _select_random_instance(self, candidates: list[str]) -> str:
        if not candidates:
            raise RuntimeError("No instances configured for random selection.")
        return str(self._route_rng.choice(candidates))

    @staticmethod
    def _extract_snapshot_num_requests(wait_record: Optional[WaitTimeResult]) -> Optional[int]:
        if wait_record is None or not isinstance(wait_record.raw_payload, dict):
            return None
        return _extract_wait_num_requests(wait_record.raw_payload)

    def _resolve_effective_num_requests(
        self,
        *,
        instance_ids: list[str],
        wait_results: Dict[str, WaitTimeResult],
    ) -> tuple[Dict[str, int], Dict[str, str], Dict[str, Optional[int]]]:
        effective_num_requests: Dict[str, int] = {}
        num_requests_source: Dict[str, str] = {}
        snapshot_num_requests: Dict[str, Optional[int]] = {}

        for instance_id in instance_ids:
            wait_record = wait_results.get(instance_id)
            current_num_requests = self._extract_snapshot_num_requests(wait_record)
            snapshot_num_requests[instance_id] = (
                int(current_num_requests) if current_num_requests is not None else None
            )

            if current_num_requests is not None:
                resolved_num_requests = int(current_num_requests)
                self._last_seen_num_requests_by_instance[instance_id] = resolved_num_requests
                effective_num_requests[instance_id] = resolved_num_requests
                num_requests_source[instance_id] = "snapshot"
                continue

            previous_num_requests = self._last_seen_num_requests_by_instance.get(instance_id)
            if previous_num_requests is not None:
                effective_num_requests[instance_id] = int(previous_num_requests)
                num_requests_source[instance_id] = "last_seen"
                continue

            effective_num_requests[instance_id] = int(self._shortest_queue_sentinel)
            num_requests_source[instance_id] = "sentinel"

        return effective_num_requests, num_requests_source, snapshot_num_requests

    def _select_min_effective_queue_instance(
        self,
        *,
        candidate_instance_ids: list[str],
        effective_num_requests: Dict[str, int],
    ) -> tuple[str, list[str], bool, str]:
        if not candidate_instance_ids:
            raise RuntimeError("No candidate instances available for queue selection.")

        all_sentinel = all(
            effective_num_requests.get(instance_id, self._shortest_queue_sentinel)
            == self._shortest_queue_sentinel
            for instance_id in candidate_instance_ids
        )
        if all_sentinel:
            selected_instance_id = self._select_random_instance(candidate_instance_ids)
            tie_candidates = list(candidate_instance_ids)
            return selected_instance_id, tie_candidates, True, "all_sentinel_random"

        min_num_requests = min(
            effective_num_requests.get(instance_id, self._shortest_queue_sentinel)
            for instance_id in candidate_instance_ids
        )
        tie_candidates = [
            instance_id
            for instance_id in candidate_instance_ids
            if effective_num_requests.get(instance_id, self._shortest_queue_sentinel)
            == min_num_requests
        ]
        if len(tie_candidates) == 1:
            return tie_candidates[0], tie_candidates, False, "unique_min"
        return (
            self._select_random_instance(tie_candidates),
            tie_candidates,
            False,
            "tie_random",
        )

    def _select_shortest_queue_instance(
        self,
        *,
        wait_results: Dict[str, WaitTimeResult],
    ) -> tuple[str, Dict[str, Any]]:
        instance_ids = list(self._instances.keys())
        if not instance_ids:
            raise RuntimeError("No instances configured for shortest-queue selection.")

        (
            effective_num_requests,
            num_requests_source,
            snapshot_num_requests,
        ) = self._resolve_effective_num_requests(
            instance_ids=instance_ids,
            wait_results=wait_results,
        )
        selected_instance_id, tie_candidates, all_sentinel, selection_reason = (
            self._select_min_effective_queue_instance(
                candidate_instance_ids=instance_ids,
                effective_num_requests=effective_num_requests,
            )
        )

        routing_details: Dict[str, Any] = {
            "selected_effective_num_requests": int(
                effective_num_requests[selected_instance_id]
            ),
            "selected_num_requests_source": num_requests_source[selected_instance_id],
            "effective_num_requests_by_instance": {
                instance_id: int(count)
                for instance_id, count in effective_num_requests.items()
            },
            "num_requests_source_by_instance": dict(num_requests_source),
            "snapshot_num_requests_by_instance": dict(snapshot_num_requests),
            "tie_candidates": list(tie_candidates),
            "all_sentinel": bool(all_sentinel),
            "selection_reason": selection_reason,
            "sentinel_num_requests": int(self._shortest_queue_sentinel),
        }
        return selected_instance_id, routing_details

    def _select_instance_affinity(
        self,
        *,
        request_bucket: str,
        accuracy_scores: Dict[str, float],
    ) -> tuple[str, Dict[str, Any]]:
        instance_ids = list(self._instances.keys())
        if not instance_ids:
            raise RuntimeError("No instances configured for instance-affinity selection.")

        normalized_bucket = _normalize_bucket_name(request_bucket)
        preferred_pool = self._affinity_bucket_to_instances.get(normalized_bucket, [])
        preferred_pool_source = "bucket_preference"
        bucket_known = bool(preferred_pool)
        if not preferred_pool:
            preferred_pool = list(self._affinity_global_fallback_instances)
            preferred_pool_source = "global_fallback"
        if not preferred_pool:
            preferred_pool = list(instance_ids)
            preferred_pool_source = "all_instances_fallback"
        preferred_pool = list(dict.fromkeys(preferred_pool))

        if len(preferred_pool) == 1:
            preferred_selected_id = preferred_pool[0]
            preferred_selection_reason = "single_preferred"
        else:
            preferred_selected_id = self._select_random_instance(preferred_pool)
            preferred_selection_reason = "preferred_pool_random"

        selected_instance_id = preferred_selected_id
        selection_reason = preferred_selection_reason
        upgrade_triggered = False

        global_best_accuracy: Optional[float] = None
        preferred_best_accuracy: Optional[float] = None
        predicted_accuracy_gap: Optional[float] = None
        global_best_candidates: list[str] = []

        if accuracy_scores:
            global_best_accuracy = max(
                float(accuracy_scores.get(instance_id, 0.0))
                for instance_id in instance_ids
            )
            preferred_best_accuracy = max(
                float(accuracy_scores.get(instance_id, 0.0))
                for instance_id in preferred_pool
            )
            predicted_accuracy_gap = float(global_best_accuracy - preferred_best_accuracy)
            if predicted_accuracy_gap >= self._affinity_upgrade_margin:
                global_best_candidates = [
                    instance_id
                    for instance_id in instance_ids
                    if math.isclose(
                        float(accuracy_scores.get(instance_id, 0.0)),
                        float(global_best_accuracy),
                        rel_tol=1e-9,
                        abs_tol=1e-12,
                    )
                ]
                if not global_best_candidates:
                    global_best_candidates = list(instance_ids)
                if len(global_best_candidates) == 1:
                    selected_instance_id = global_best_candidates[0]
                    selection_reason = "upgrade_global_best_unique"
                else:
                    selected_instance_id = self._select_random_instance(
                        global_best_candidates
                    )
                    selection_reason = "upgrade_global_best_tie_random"
                upgrade_triggered = True

        routing_details: Dict[str, Any] = {
            "bucket": normalized_bucket,
            "bucket_known": bool(bucket_known),
            "preferred_pool_source": preferred_pool_source,
            "preferred_instance_pool": list(preferred_pool),
            "preferred_selection_reason": preferred_selection_reason,
            "preferred_selected_instance": preferred_selected_id,
            "global_best_predicted_accuracy": (
                float(global_best_accuracy)
                if global_best_accuracy is not None
                else None
            ),
            "preferred_best_predicted_accuracy": (
                float(preferred_best_accuracy)
                if preferred_best_accuracy is not None
                else None
            ),
            "predicted_accuracy_gap": (
                float(predicted_accuracy_gap)
                if predicted_accuracy_gap is not None
                else None
            ),
            "upgrade_margin_threshold": float(self._affinity_upgrade_margin),
            "upgrade_triggered": bool(upgrade_triggered),
            "global_best_accuracy_candidates": list(global_best_candidates),
            "selected_instance_id": selected_instance_id,
            "selection_reason": selection_reason,
        }
        return selected_instance_id, routing_details

    async def _dispatch(self, queued) -> None:
        completion_future = queued.payload.pop("_completion_future", None)
        system_entry_perf = float(
            queued.payload.pop("_system_entry_perf", time.perf_counter())
        )
        started_perf = float(queued.payload.pop("_started_perf", time.perf_counter()))
        request_slo_ms = float(queued.payload.pop("_request_slo_ms", 0.0))
        score_latency_limit_ms_raw = queued.payload.pop(
            "_score_latency_limit_ms",
            None,
        )
        score_latency_limit_ms = (
            float(score_latency_limit_ms_raw)
            if isinstance(score_latency_limit_ms_raw, (int, float))
            and not isinstance(score_latency_limit_ms_raw, bool)
            else None
        )
        request_bucket = _normalize_bucket_name(
            queued.payload.pop("_request_bucket", "unknown")
        )
        target_id: str | None = None
        dispatch_perf: Optional[float] = None
        pending_dispatches_by_instance: Dict[
            str, tuple[PendingDispatch, ...]
        ] = {}
        probe_ready_delay_ms_by_instance: Dict[str, float] = {}
        shortest_queue_routing: Optional[Dict[str, Any]] = None
        affinity_routing: Optional[Dict[str, Any]] = None
        score_candidate_terms: Optional[Dict[str, Any]] = None
        score_policy_terms: Optional[Dict[str, Any]] = None

        try:
            prompt_text = self._extract_prompt_text(queued.payload)
            prompt_tokens = self._extract_precomputed_prompt_tokens(queued.payload)
            if prompt_tokens is None:
                LOGGER.warning(
                    "Falling back to tokenizer-based prompt tokenization for request_id=%s.",
                    queued.request_id,
                )
                prompt_tokens = (
                    self._get_prompt_tokens(queued.payload, prompt_text)
                    if prompt_text
                    else 0
                )
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
                pending_dispatches_by_instance = (
                    self._pending_dispatches_by_instance()
                )
                probe_ready_delay_ms_by_instance = (
                    self._probe_ready_delays_by_instance(
                        prompt_tokens=prompt_tokens,
                        pending_dispatches_by_instance=(
                            pending_dispatches_by_instance
                        ),
                    )
                )
                if self._skip_wait_result_build:
                    wait_results = self._build_zero_wait_results(
                        reason="latency_agnostic_fast_path",
                    )
                    live_wait_results = wait_results
                    estimator_wait_results = (
                        {self._wait_estimator_name: wait_results}
                        if self._wait_estimators
                        else {}
                    )
                else:
                    (
                        wait_results,
                        live_wait_results,
                        estimator_wait_results,
                    ) = await self._build_wait_results_for_request(
                        queued_payload=queued.payload,
                        request_id=queued.request_id,
                        prompt_tokens=prompt_tokens,
                        accuracy_scores=accuracy_scores,
                        output_lengths=output_lengths,
                        pending_dispatches_by_instance=(
                            pending_dispatches_by_instance
                        ),
                        probe_ready_delay_ms_by_instance=(
                            probe_ready_delay_ms_by_instance
                        ),
                    )

                self._utility_state.current_slo_ms = request_slo_ms
                self._utility_state.score_latency_limit_ms = (
                    score_latency_limit_ms
                )
                try:
                    if self._route_strategy == "round_robin":
                        target_id = self._select_round_robin_instance()
                    elif self._route_strategy == "shortest_queue":
                        (
                            target_id,
                            shortest_queue_routing,
                        ) = self._select_shortest_queue_instance(
                            wait_results=live_wait_results or wait_results,
                        )
                    elif self._route_strategy == "instance_affinity":
                        (
                            target_id,
                            affinity_routing,
                        ) = self._select_instance_affinity(
                            request_bucket=request_bucket,
                            accuracy_scores=accuracy_scores,
                        )
                    else:
                        target_id = self._select_instance(
                            wait_results,
                            accuracy_scores,
                            output_lengths,
                            prompt_tokens,
                        )
                    if (
                        self._wait_estimator_name
                        == SCORE_PROXY_TTFT_ESTIMATOR_NAME
                    ):
                        score_candidate_terms = (
                            _build_compact_score_candidate_terms(
                                wait_results=wait_results,
                                selected_instance_id=target_id,
                                slo_ms=request_slo_ms,
                                prompt_tokens=prompt_tokens,
                                accuracy_scores=accuracy_scores,
                                output_lengths=output_lengths,
                                instance_costs=self._instance_costs,
                                lambda_weight=self._lambda,
                            )
                        )
                    score_state = self._utility_state.score_policy_state
                    if score_state is not None:
                        if score_latency_limit_ms is None:
                            raise ValueError(
                                "Published SCORE requires a total-latency limit"
                            )
                        score_policy_terms = (
                            _build_compact_published_score_candidate_terms(
                                wait_results=wait_results,
                                selected_instance_id=target_id,
                                latency_limit_ms=score_latency_limit_ms,
                                accuracy_scores=accuracy_scores,
                                output_lengths=output_lengths,
                                instance_costs=self._instance_costs,
                                lambda_weight=self._lambda,
                                cost_weight=self._score_cost_weight,
                                latency_weight=self._score_latency_weight,
                                state=score_state,
                            )
                        )
                        selected_predicted_cost = (
                            _score_predicted_response_cost(
                                target_id,
                                output_lengths=output_lengths,
                                instance_costs=self._instance_costs,
                            )
                        )
                        score_state.record_selection(selected_predicted_cost)
                        score_policy_terms["cumulative_predicted_cost_after"] = (
                            score_state.cumulative_predicted_cost
                        )
                finally:
                    self._utility_state.current_slo_ms = None
                    self._utility_state.score_latency_limit_ms = None
                self._reserve_pending_dispatch(
                    instance_id=target_id,
                    engine_request_id=engine_request_id,
                    prompt_tokens=prompt_tokens,
                    predicted_output_tokens=output_lengths.get(
                        target_id,
                        1.0,
                    ),
                    completion_cap=completion_cap,
                    predicted_ready_at_s=self._predicted_ready_at_s(
                        wait_record=(
                            live_wait_results.get(target_id)
                            or wait_results.get(target_id)
                        ),
                        delay_ms=probe_ready_delay_ms_by_instance.get(
                            target_id,
                            0.0,
                        ),
                    ),
                )

            target = self._instances[target_id]
            dispatch_perf = time.perf_counter()
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

            wait_record = (
                wait_results.get(target_id)
                or live_wait_results.get(target_id)
                or target.last_wait
            )
            if wait_record is not None:
                self._attach_readiness_predictor_inputs(
                    wait_record=wait_record,
                    prompt_tokens=prompt_tokens,
                    pending_dispatches=pending_dispatches_by_instance.get(
                        target_id,
                        (),
                    ),
                )
            live_wait_record = live_wait_results.get(
                target_id
            ) or target.last_wait_for_mode(
                prompt_tokens=prompt_tokens if prompt_tokens > 0 else None,
                stop_mode=(
                    SimulationStopMode.PREFILL_DONE if prompt_tokens > 0 else None
                ),
            )

            estimator_waits_ms: Dict[str, float] = {}
            for estimator_name, estimator_results in estimator_wait_results.items():
                estimator_record = estimator_results.get(target_id)
                if estimator_record is None:
                    estimator_record = (
                        live_wait_results.get(target_id) or target.last_wait
                    )
                if estimator_record is not None:
                    estimator_waits_ms[estimator_name] = float(estimator_record.wait_ms)

            selected_accuracy = accuracy_scores.get(target_id)
            selected_output = output_lengths.get(target_id)

            for pk_tracker in self._iter_pk_trackers():
                pk_tracker.note_dispatch(target_id, time.time())

            if wait_record:
                self._request_log[queued.request_id] = wait_record
                await self._log_to_file(queued.request_id, wait_record)

            if completion_future is not None:

                def _handle_done(task: asyncio.Task) -> None:
                    completed_perf = time.perf_counter()
                    payload: Dict[str, Any] = {
                        "request_id": queued.request_id,
                        "instance_id": target_id,
                        "wait_time_ms": wait_record.wait_ms if wait_record else None,
                        "live_wait_time_ms": (
                            live_wait_record.wait_ms if live_wait_record else None
                        ),
                        "wait_estimator": self._wait_estimator_name,
                        "wait_estimates_ms": estimator_waits_ms,
                        "score_candidate_terms": score_candidate_terms,
                        "score_policy_terms": score_policy_terms,
                        "route_strategy": self._route_strategy,
                        "predicted_accuracy": (
                            float(selected_accuracy)
                            if selected_accuracy is not None
                            else None
                        ),
                        "predicted_output_tokens": (
                            float(selected_output)
                            if selected_output is not None
                            else None
                        ),
                        "request_bucket": request_bucket,
                        "prompt_tokens": int(prompt_tokens),
                        "system_entry_perf": system_entry_perf,
                        "started_perf": started_perf,
                        "dispatch_perf": dispatch_perf,
                        "completed_perf": completed_perf,
                        "latency_ms": (completed_perf - started_perf) * 1000.0,
                    }
                    if isinstance(shortest_queue_routing, dict):
                        payload["shortest_queue_routing"] = dict(shortest_queue_routing)
                        payload["selected_effective_num_requests"] = (
                            shortest_queue_routing.get(
                                "selected_effective_num_requests"
                            )
                        )
                        payload["selected_num_requests_source"] = (
                            shortest_queue_routing.get("selected_num_requests_source")
                        )
                    if isinstance(affinity_routing, dict):
                        payload["affinity_routing"] = dict(affinity_routing)
                    wait_payload = wait_record.raw_payload if wait_record else None
                    if isinstance(wait_payload, dict):
                        metadata = None
                        reports = wait_payload.get("reports")
                        if isinstance(reports, list) and reports:
                            report0 = reports[0]
                            if isinstance(report0, dict):
                                metadata = report0.get("metadata")
                        payload["wait_time_metadata"] = metadata

                    if task.cancelled():
                        payload["error"] = "CancelledError"
                    else:
                        exc = task.exception()
                        if exc:
                            payload["error"] = f"{exc.__class__.__name__}: {exc}"
                        else:
                            response = task.result()
                            payload.update(_extract_usage_tokens(response))
                            if isinstance(response, dict):
                                payload["response_id"] = response.get("id")
                                payload["response_model"] = response.get("model")
                            else:
                                payload["response_id"] = getattr(response, "id", None)
                                payload["response_model"] = getattr(
                                    response, "model", None
                                )

                    payload_response_id = payload.get("response_id")
                    for pk_tracker_cb in self._iter_pk_trackers():
                        pk_tracker_cb.note_response(
                            (
                                str(payload_response_id)
                                if isinstance(payload_response_id, str)
                                else None
                            ),
                            str(target_id) if target_id is not None else "",
                        )
                    if not completion_future.done():
                        completion_future.set_result(payload)

                submit_task.add_done_callback(_handle_done)

            routed = RoutedRequest(
                request_id=queued.request_id,
                instance_id=target_id,
                wait_time_ms=wait_record.wait_ms if wait_record else None,
                wait_time_details=wait_record.raw_payload if wait_record else None,
            )
            if not queued.result_future.done():
                queued.result_future.set_result(routed)
        except Exception as exc:
            LOGGER.exception(
                "Dispatch failed; request_id=%s target_id=%s route_strategy=%s "
                "wait_estimator=%s error=%s",
                queued.request_id,
                target_id,
                self._route_strategy,
                self._wait_estimator_name,
                exc,
            )
            self._utility_state.current_slo_ms = None
            self._utility_state.score_latency_limit_ms = None
            if completion_future is not None and not completion_future.done():
                completed_perf = time.perf_counter()
                completion_future.set_result(
                    {
                        "request_id": queued.request_id,
                        "instance_id": target_id,
                        "system_entry_perf": system_entry_perf,
                        "started_perf": started_perf,
                        "dispatch_perf": dispatch_perf,
                        "completed_perf": completed_perf,
                        "latency_ms": (completed_perf - started_perf) * 1000.0,
                        "wait_estimator": self._wait_estimator_name,
                        "route_strategy": self._route_strategy,
                        "request_bucket": request_bucket,
                        "error": f"{exc.__class__.__name__}: {exc}",
                    }
                )
            raise


def _sample_latency_slo_ms(
    prompt_tokens: int,
    *,
    rng: random.Random,
    min_ms: float,
    max_ms: float,
) -> float:
    base_ms = 350.0 + (min(prompt_tokens, 8192) * 1.35)
    jitter = rng.uniform(0.75, 1.5)
    return float(max(min(base_ms * jitter, max_ms), min_ms))


def _sample_queue_slo_ms(
    *,
    rng: random.Random,
    min_ms: float,
    max_ms: float,
) -> float:
    if min_ms == max_ms:
        return float(min_ms)
    return float(rng.uniform(min_ms, max_ms))


def _sample_ttft_slo_ms(
    prompt_tokens: int,
    queue_slo_ms: float,
    *,
    rng: random.Random,
    min_ms: float,
    max_ms: float,
    base_ms: float,
    per_prompt_token_ms: float,
    jitter_min: float,
    jitter_max: float,
) -> float:
    prefill_budget_ms = float(base_ms) + (
        float(per_prompt_token_ms) * float(prompt_tokens)
    )
    prefill_budget_ms *= float(rng.uniform(jitter_min, jitter_max))
    prefill_budget_ms = max(prefill_budget_ms, 0.0)

    # Enforce TTFT SLO >= queue SLO + prefill budget while honoring max bound.
    max_prefill_budget_ms = max(float(max_ms) - float(queue_slo_ms), 0.0)
    prefill_budget_ms = min(prefill_budget_ms, max_prefill_budget_ms)

    ttft_floor_ms = float(queue_slo_ms) + prefill_budget_ms
    ttft_slo_ms = max(ttft_floor_ms, float(min_ms))
    return float(min(ttft_slo_ms, float(max_ms)))


def _derive_mmpp2_params(
    *,
    request_rate_qps: float,
    rate_ratio: float,
    high_fraction: float,
    correlation_time_s: float,
) -> MMPP2DerivedParams:
    if rate_ratio < 1.0:
        raise ValueError("--mmpp2-rate-ratio must be >= 1.0.")
    if not (0.0 < high_fraction < 1.0):
        raise ValueError("--mmpp2-high-fraction must be strictly between 0 and 1.")
    if correlation_time_s <= 0.0:
        raise ValueError("--mmpp2-correlation-time-s must be > 0.")

    base_rate_qps = max(float(request_rate_qps), 0.0)
    denominator = (1.0 - float(high_fraction)) + (
        float(high_fraction) * float(rate_ratio)
    )
    if denominator <= 0.0:
        raise ValueError("Invalid MMPP-2 derived parameters: denominator must be > 0.")

    lambda_low_rps = base_rate_qps / denominator
    lambda_high_rps = float(rate_ratio) * lambda_low_rps
    transition_sum = 1.0 / float(correlation_time_s)
    q_low_to_high_rps = float(high_fraction) * transition_sum
    q_high_to_low_rps = (1.0 - float(high_fraction)) * transition_sum

    return MMPP2DerivedParams(
        request_rate_qps=base_rate_qps,
        rate_ratio=float(rate_ratio),
        high_fraction=float(high_fraction),
        correlation_time_s=float(correlation_time_s),
        lambda_low_rps=float(lambda_low_rps),
        lambda_high_rps=float(lambda_high_rps),
        q_low_to_high_rps=float(q_low_to_high_rps),
        q_high_to_low_rps=float(q_high_to_low_rps),
    )


def _sample_mmpp2_interarrival_s(
    *,
    params: MMPP2DerivedParams,
    state: Dict[str, Any],
    rng: random.Random,
) -> float:
    if params.request_rate_qps <= 0.0:
        return 0.0

    current_state = state.get("current_state")
    if current_state not in {"low", "high"}:
        current_state = "high" if rng.random() < params.high_fraction else "low"

    elapsed_s = 0.0
    while True:
        if current_state == "high":
            arrival_rate = params.lambda_high_rps
            switch_rate = params.q_high_to_low_rps
            switched_state = "low"
        else:
            arrival_rate = params.lambda_low_rps
            switch_rate = params.q_low_to_high_rps
            switched_state = "high"

        total_rate = arrival_rate + switch_rate
        if total_rate <= 0.0:
            state["current_state"] = current_state
            return 0.0

        elapsed_s += rng.expovariate(total_rate)
        if switch_rate <= 0.0:
            state["current_state"] = current_state
            return float(elapsed_s)
        if arrival_rate <= 0.0:
            current_state = switched_state
            continue

        if rng.random() < (arrival_rate / total_rate):
            state["current_state"] = current_state
            return float(elapsed_s)
        current_state = switched_state


class _ArrivalSchedule:
    """Absolute arrival timeline: sleep until origin + cumulative sampled offset.

    A late event-loop wakeup (e.g. under heavy streaming load) delays only the
    current arrival instead of shifting every later one, so lag never
    accumulates into the realized rate. The sampled gap sequence is unchanged.
    """

    def __init__(self, *, clock=time.perf_counter, sleep=asyncio.sleep,
                 blocking_sleep=time.sleep) -> None:
        self._clock = clock
        self._sleep = sleep
        self._blocking_sleep = blocking_sleep
        self.origin_perf: Optional[float] = None
        self.offset_s = 0.0

    def start(self) -> None:
        self.origin_perf = self._clock()

    def deadline(self, interarrival_s: float) -> Optional[float]:
        """Advance the schedule; None means no sleep (non-positive gap)."""
        if self.origin_perf is None:
            self.start()
        if interarrival_s <= 0:
            return None
        self.offset_s += float(interarrival_s)
        return self.origin_perf + self.offset_s

    async def wait(self, interarrival_s: float) -> None:
        deadline = self.deadline(interarrival_s)
        if deadline is not None:
            await self._sleep(max(deadline - self._clock(), 0.0))

    def wait_blocking(self, interarrival_s: float) -> None:
        """Thread variant: block until the deadline without touching the loop."""
        deadline = self.deadline(interarrival_s)
        if deadline is not None:
            self._blocking_sleep(max(deadline - self._clock(), 0.0))


def _sample_interarrival_s(
    *,
    request_rate_qps: float,
    arrival_process: str,
    rng: random.Random,
    mmpp2_params: Optional[MMPP2DerivedParams] = None,
    mmpp2_state: Optional[Dict[str, Any]] = None,
) -> float:
    if request_rate_qps <= 0:
        return 0.0
    if arrival_process == "deterministic":
        return 1.0 / request_rate_qps
    if arrival_process == "poisson":
        return rng.expovariate(request_rate_qps)
    if arrival_process == "mmpp2":
        if mmpp2_params is None or mmpp2_state is None:
            raise ValueError("MMPP-2 arrival process requires mmpp2_params and mmpp2_state.")
        return _sample_mmpp2_interarrival_s(
            params=mmpp2_params,
            state=mmpp2_state,
            rng=rng,
        )
    raise ValueError(
        f"Unsupported arrival_process '{arrival_process}'. "
        f"Supported: {', '.join(ARRIVAL_PROCESSES)}."
    )


def _normalize_bucket_name(raw_bucket: Any) -> str:
    if isinstance(raw_bucket, str):
        bucket = raw_bucket.strip().lower()
        if bucket:
            return bucket
    return "unknown"


def _bucket_from_prompt_record(record: Dict[str, Any]) -> str:
    bucket = record.get("bucket")
    if isinstance(bucket, str) and bucket.strip():
        return _normalize_bucket_name(bucket)

    metadata = record.get("prompt_metadata")
    if isinstance(metadata, dict):
        metadata_bucket = metadata.get("bucket")
        if isinstance(metadata_bucket, str) and metadata_bucket.strip():
            return _normalize_bucket_name(metadata_bucket)
    return "unknown"


def build_experiment_requests(
    bucket_dir: Path,
    *,
    num_requests: int,
    seed: int,
    slo_min_ms: float,
    slo_max_ms: float,
    queue_slo_min_ms: float,
    queue_slo_max_ms: float,
    ttft_slo_min_ms: float,
    ttft_slo_max_ms: float,
    ttft_slo_base_ms: float,
    ttft_slo_per_prompt_token_ms: float,
    ttft_slo_jitter_min: float,
    ttft_slo_jitter_max: float,
    prompt_sampling_mode: str = "random",
    holdout_prompts_per_bucket: int = 0,
) -> list[ExperimentRequest]:
    if num_requests <= 0:
        raise ValueError("num_requests must be > 0")
    if slo_min_ms > slo_max_ms:
        raise ValueError("slo_min_ms must be <= slo_max_ms")
    if queue_slo_min_ms > queue_slo_max_ms:
        raise ValueError("queue_slo_min_ms must be <= queue_slo_max_ms")
    if ttft_slo_min_ms > ttft_slo_max_ms:
        raise ValueError("ttft_slo_min_ms must be <= ttft_slo_max_ms")
    if queue_slo_max_ms > ttft_slo_max_ms:
        raise ValueError(
            "queue_slo_max_ms must be <= ttft_slo_max_ms so TTFT SLO can remain "
            ">= queue SLO under the configured max bound"
        )
    if ttft_slo_jitter_min <= 0 or ttft_slo_jitter_max <= 0:
        raise ValueError("ttft_slo_jitter_{min,max} must be > 0")
    if ttft_slo_jitter_min > ttft_slo_jitter_max:
        raise ValueError("ttft_slo_jitter_min must be <= ttft_slo_jitter_max")

    rng = random.Random(seed)
    requests: list[ExperimentRequest] = []

    if prompt_sampling_mode == "mixed_then_shuffle":
        if holdout_prompts_per_bucket <= 0:
            raise ValueError(
                "holdout_prompts_per_bucket must be > 0 when prompt_sampling_mode is mixed_then_shuffle."
            )
        prompt_records = select_prompt_subset(
            iter_mixed_then_random_bucketed_prompts(
                bucket_dir,
                per_bucket_limit=holdout_prompts_per_bucket,
                limit=num_requests,
                seed=seed,
                include_complete_record=True,
            ),
            num_requests,
        )
    elif prompt_sampling_mode == "random":
        prompt_records = select_prompt_subset(
            iter_random_bucketed_prompts(
                bucket_dir,
                limit=num_requests,
                seed=seed,
                include_complete_record=True,
            ),
            num_requests,
        )
    else:
        raise ValueError(
            f"Unsupported prompt_sampling_mode '{prompt_sampling_mode}'. Supported: random, mixed_then_shuffle."
        )

    for idx, record in enumerate(prompt_records):
        prompt = record["prompt"]
        prompt_tokens = record["prompt_tokens"]
        bucket = _bucket_from_prompt_record(record)
        latency_slo_ms = _sample_latency_slo_ms(
            prompt_tokens,
            rng=rng,
            min_ms=slo_min_ms,
            max_ms=slo_max_ms,
        )
        queue_slo_ms = _sample_queue_slo_ms(
            rng=rng,
            min_ms=queue_slo_min_ms,
            max_ms=queue_slo_max_ms,
        )
        ttft_slo_ms = _sample_ttft_slo_ms(
            prompt_tokens,
            queue_slo_ms,
            rng=rng,
            min_ms=ttft_slo_min_ms,
            max_ms=ttft_slo_max_ms,
            base_ms=ttft_slo_base_ms,
            per_prompt_token_ms=ttft_slo_per_prompt_token_ms,
            jitter_min=ttft_slo_jitter_min,
            jitter_max=ttft_slo_jitter_max,
        )
        requests.append(
            ExperimentRequest(
                request_id=f"req-{idx}",
                prompt=prompt,
                prompt_tokens=prompt_tokens,
                bucket=bucket,
                latency_slo_ms=latency_slo_ms,
                queue_slo_ms=queue_slo_ms,
                ttft_slo_ms=ttft_slo_ms,
            )
        )
    return requests


def _predicted_cost(
    instance_id: str,
    *,
    prompt_tokens: int,
    output_lengths: Dict[str, float],
    instance_costs: Dict[str, Dict[str, float]],
) -> float:
    cost_info = instance_costs.get(instance_id) or {}
    prompt_rate = float(cost_info.get("prompt", 0.0))
    output_rate = float(cost_info.get("output", 0.0))
    predicted_output = float(output_lengths.get(instance_id, 0.0))
    return (prompt_rate * prompt_tokens) + (output_rate * predicted_output)


def _score_predicted_response_cost(
    instance_id: str,
    *,
    output_lengths: Dict[str, float],
    instance_costs: Dict[str, Dict[str, float]],
) -> float:
    """SCORE paper cost term: per-output-token cost times predicted length."""

    cost_info = instance_costs.get(instance_id) or {}
    output_rate = float(cost_info.get("output", 0.0))
    predicted_output = float(output_lengths.get(instance_id, 0.0))
    return output_rate * predicted_output


def _build_compact_score_candidate_terms(
    *,
    wait_results: Dict[str, WaitTimeResult],
    selected_instance_id: str,
    slo_ms: Optional[float],
    prompt_tokens: int,
    accuracy_scores: Dict[str, float],
    output_lengths: Dict[str, float],
    instance_costs: Dict[str, Dict[str, float]],
    lambda_weight: float,
) -> Dict[str, Any]:
    """Build auditable SCORE terms without duplicating calibration constants."""
    wait_ms_by_instance = {
        instance_id: float(wait_result.wait_ms)
        for instance_id, wait_result in wait_results.items()
    }
    if selected_instance_id not in wait_ms_by_instance:
        raise ValueError(
            "Selected SCORE instance is missing from candidate wait results: "
            f"{selected_instance_id!r}"
        )
    any_slo_feasible = (
        True
        if slo_ms is None
        else any(wait_ms <= float(slo_ms) for wait_ms in wait_ms_by_instance.values())
    )

    candidates: Dict[str, Dict[str, Any]] = {}
    for instance_id, wait_result in wait_results.items():
        diagnostics = wait_result.raw_payload.get(
            "_wait_estimator_diagnostics"
        )
        if not isinstance(diagnostics, dict) or diagnostics.get(
            "method"
        ) != SCORE_PROXY_TTFT_ESTIMATOR_NAME:
            raise ValueError(
                "SCORE candidate logging requires estimator diagnostics for "
                f"instance {instance_id!r}"
            )
        details = diagnostics.get("details")
        if not isinstance(details, dict):
            raise ValueError(
                "SCORE candidate logging requires diagnostic details for "
                f"instance {instance_id!r}"
            )

        def _required_float(mapping: Dict[str, Any], key: str) -> float:
            value = mapping.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"SCORE diagnostics for {instance_id!r} are missing {key!r}"
                )
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(
                    f"SCORE diagnostic {instance_id!r}.{key} must be finite"
                )
            return numeric

        estimated_ttft_ms = _required_float(
            diagnostics,
            "estimated_ttft_ms",
        )
        selected_wait_ms = float(wait_result.wait_ms)
        if not math.isclose(
            estimated_ttft_ms,
            selected_wait_ms,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "SCORE diagnostic TTFT differs from the value used for routing "
                f"on {instance_id!r}: diagnostics={estimated_ttft_ms}, "
                f"routing={selected_wait_ms}"
            )

        predicted_quality = float(accuracy_scores.get(instance_id, 0.0))
        predicted_output_tokens = float(output_lengths.get(instance_id, 0.0))
        if not math.isfinite(predicted_quality):
            raise ValueError(
                f"SCORE predicted quality must be finite for {instance_id!r}"
            )
        if (
            not math.isfinite(predicted_output_tokens)
            or predicted_output_tokens < 0.0
        ):
            raise ValueError(
                "SCORE predicted output tokens must be finite and nonnegative "
                f"for {instance_id!r}"
            )
        predicted_cost = _predicted_cost(
            instance_id,
            prompt_tokens=prompt_tokens,
            output_lengths=output_lengths,
            instance_costs=instance_costs,
        )
        if not math.isfinite(predicted_cost):
            raise ValueError(
                f"SCORE predicted cost must be finite for {instance_id!r}"
            )
        candidate_value = hard_slo_candidate_value(
            instance_id=instance_id,
            wait_ms_by_instance=wait_ms_by_instance,
            slo_ms=slo_ms,
            predicted_quality=predicted_quality,
            predicted_cost=predicted_cost,
            lambda_weight=lambda_weight,
        )
        if math.isnan(candidate_value) or candidate_value == float("inf"):
            raise ValueError(
                f"SCORE hard candidate value is invalid for {instance_id!r}"
            )
        pending_overlay_count = details.get("pending_overlay_count")
        if pending_overlay_count is not None and (
            isinstance(pending_overlay_count, bool)
            or not isinstance(pending_overlay_count, int)
            or pending_overlay_count < 0
        ):
            raise ValueError(
                "SCORE pending_overlay_count must be a nonnegative integer or null "
                f"for instance {instance_id!r}"
            )

        prefill_backlog_tokens = _required_float(
            details,
            "prefill_backlog_total_tokens",
        )
        decode_backlog_tokens = _required_float(
            details,
            "decode_backlog_total_tokens",
        )
        prefill_ms = _required_float(
            diagnostics,
            "estimated_prefill_ms",
        )
        decode_interference_ms = _required_float(
            diagnostics,
            "estimated_decode_interference_ms",
        )
        first_decode_batch_ms = _required_float(
            diagnostics,
            "estimated_first_decode_batch_ms",
        )
        if any(
            value < 0.0
            for value in (
                prefill_backlog_tokens,
                decode_backlog_tokens,
                prefill_ms,
                decode_interference_ms,
                first_decode_batch_ms,
            )
        ):
            raise ValueError(
                f"SCORE latency terms must be nonnegative for {instance_id!r}"
            )
        logged_term_sum_ms = (
            prefill_ms + decode_interference_ms + first_decode_batch_ms
        )
        if not math.isclose(
            logged_term_sum_ms,
            selected_wait_ms,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "SCORE logged terms do not sum to the routed TTFT for "
                f"{instance_id!r}: terms={logged_term_sum_ms}, "
                f"routing={selected_wait_ms}"
            )

        candidates[instance_id] = {
            "prefill_backlog_tokens": prefill_backlog_tokens,
            "decode_backlog_tokens": decode_backlog_tokens,
            "pending_overlay_count": pending_overlay_count,
            "prefill_ms": prefill_ms,
            "decode_interference_ms": decode_interference_ms,
            "first_decode_batch_ms": first_decode_batch_ms,
            "predicted_ttft_ms": selected_wait_ms,
            "slo_feasible": (
                True
                if slo_ms is None
                else selected_wait_ms <= float(slo_ms)
            ),
            "predicted_quality": predicted_quality,
            "predicted_output_tokens": predicted_output_tokens,
            "predicted_cost": float(predicted_cost),
            # JSON has no portable infinity. Null denotes a candidate excluded
            # by the common hard-SLO gate while another candidate is feasible.
            "hard_candidate_value": (
                None
                if candidate_value == float("-inf")
                else float(candidate_value)
            ),
        }

    return {
        "selected_instance_id": selected_instance_id,
        "slo_ms": float(slo_ms) if slo_ms is not None else None,
        "any_slo_feasible": bool(any_slo_feasible),
        "candidates": candidates,
    }


def _build_compact_published_score_candidate_terms(
    *,
    wait_results: Dict[str, WaitTimeResult],
    selected_instance_id: str,
    latency_limit_ms: float,
    accuracy_scores: Dict[str, float],
    output_lengths: Dict[str, float],
    instance_costs: Dict[str, Dict[str, float]],
    lambda_weight: float,
    cost_weight: float,
    latency_weight: float,
    state: ScorePolicyState,
) -> Dict[str, Any]:
    """Build auditable terms for the published SCORE decision equation."""

    if selected_instance_id not in wait_results:
        raise ValueError(
            "Selected published-SCORE instance is missing from candidate "
            f"latency results: {selected_instance_id!r}"
        )

    candidates: Dict[str, Dict[str, Any]] = {}
    for instance_id, wait_result in wait_results.items():
        diagnostics = wait_result.raw_payload.get("_wait_estimator_diagnostics")
        if not isinstance(diagnostics, dict) or diagnostics.get(
            "method"
        ) != SCORE_TOTAL_LATENCY_ESTIMATOR_NAME:
            raise ValueError(
                "Published SCORE logging requires total-latency diagnostics "
                f"for instance {instance_id!r}"
            )
        details = diagnostics.get("details")
        if not isinstance(details, dict):
            raise ValueError(
                "Published SCORE logging requires diagnostic details for "
                f"instance {instance_id!r}"
            )

        estimated_total_latency_ms = _as_nonnegative_float(
            diagnostics.get("estimated_total_latency_ms")
        )
        if estimated_total_latency_ms is None or not math.isclose(
            float(estimated_total_latency_ms),
            float(wait_result.wait_ms),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "Published SCORE total-latency diagnostic differs from the "
                f"routing value for {instance_id!r}"
            )

        predicted_quality = float(accuracy_scores.get(instance_id, 0.0))
        predicted_output_tokens = _as_nonnegative_float(
            output_lengths.get(instance_id)
        )
        if not math.isfinite(predicted_quality):
            raise ValueError(
                f"Published SCORE quality must be finite for {instance_id!r}"
            )
        if predicted_output_tokens is None:
            raise ValueError(
                "Published SCORE requires a predicted output length for "
                f"{instance_id!r}"
            )
        predicted_response_cost = _score_predicted_response_cost(
            instance_id,
            output_lengths=output_lengths,
            instance_costs=instance_costs,
        )
        objective_terms = published_score_candidate_terms(
            predicted_quality=predicted_quality,
            predicted_response_cost=predicted_response_cost,
            predicted_total_latency_ms=float(estimated_total_latency_ms),
            latency_limit_ms=float(latency_limit_ms),
            lambda_weight=float(lambda_weight),
            cost_weight=float(cost_weight),
            latency_weight=float(latency_weight),
            state=state,
        )
        candidates[instance_id] = {
            "prefill_backlog_tokens": details.get(
                "prefill_backlog_total_tokens"
            ),
            "decode_backlog_tokens": details.get(
                "decode_backlog_total_tokens"
            ),
            "pending_overlay_count": details.get("pending_overlay_count"),
            "prefill_wait_ms": details.get("prefill_wait_ms"),
            "decode_backlog_wait_ms": details.get("decode_backlog_wait_ms"),
            "waiting_time_ms": diagnostics.get("estimated_waiting_time_ms"),
            "predicted_runtime_ms": diagnostics.get("estimated_runtime_ms"),
            "predicted_total_latency_ms": float(estimated_total_latency_ms),
            "predicted_quality": predicted_quality,
            "predicted_output_tokens": float(predicted_output_tokens),
            "predicted_response_cost": float(predicted_response_cost),
            **objective_terms,
        }

    max_value = max(
        float(candidate["score_candidate_value"])
        for candidate in candidates.values()
    )
    selected_value = float(
        candidates[selected_instance_id]["score_candidate_value"]
    )
    if not math.isclose(
        selected_value,
        max_value,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Published SCORE selected a non-maximal candidate: "
            f"selected={selected_instance_id!r} value={selected_value} "
            f"max={max_value}"
        )

    return {
        "policy": "score",
        "published_fixed_lambda_argmax": True,
        "selected_instance_id": selected_instance_id,
        "request_index": state.next_request_index,
        "lambda_weight": float(lambda_weight),
        "cost_weight": float(cost_weight),
        "latency_weight": float(latency_weight),
        "latency_limit_ms": float(latency_limit_ms),
        "total_cost_budget": state.total_cost_budget,
        "cumulative_predicted_cost_before": state.cumulative_predicted_cost,
        "candidates": candidates,
    }


def build_utility_fn(
    utility_name: str,
    *,
    lambda_weight: float,
    delta_weight: float,
    instance_costs: Dict[str, Dict[str, float]],
    utility_state: UtilityState,
    score_cost_weight: float = 1.0,
    score_latency_weight: float = 1.0,
) -> UtilityCallable:
    utility = utility_name.strip().lower()

    def soft(
        instance_id: str,
        wait_results: Dict[str, WaitTimeResult],
        accuracy_scores: Dict[str, float],
        output_lengths: Dict[str, float],
        prompt_tokens: int,
    ) -> float:
        wait = wait_results[instance_id].wait_ms
        accuracy = float(accuracy_scores.get(instance_id, 0.0))
        cost = _predicted_cost(
            instance_id,
            prompt_tokens=prompt_tokens,
            output_lengths=output_lengths,
            instance_costs=instance_costs,
        )
        return accuracy - (lambda_weight * cost) - (delta_weight * wait)

    if utility in {"soft", "soft_prefill_tps", "soft_pk_mg1"}:
        return soft
    if utility == "min_wait":
        return lambda instance_id, wait_results, *_: -wait_results[instance_id].wait_ms
    if utility == "latency_and_cost_agnostic":
        return lambda instance_id, _wait, accuracy_scores, _out, _pt: float(
            accuracy_scores.get(instance_id, 0.0)
        )
    if utility == "latency_agnostic":
        return lambda instance_id, _wait, accuracy_scores, output_lengths, prompt_tokens: (
            float(accuracy_scores.get(instance_id, 0.0))
            - (
                lambda_weight
                * _predicted_cost(
                    instance_id,
                    prompt_tokens=prompt_tokens,
                    output_lengths=output_lengths,
                    instance_costs=instance_costs,
                )
            )
        )
    if utility == "score":
        score_state = utility_state.score_policy_state
        if score_state is None:
            raise ValueError("score utility requires ScorePolicyState")

        def score(
            instance_id: str,
            wait_results: Dict[str, WaitTimeResult],
            accuracy_scores: Dict[str, float],
            output_lengths: Dict[str, float],
            _prompt_tokens: int,
        ) -> float:
            latency_limit_ms = utility_state.score_latency_limit_ms
            if latency_limit_ms is None:
                raise ValueError(
                    "score utility requires the current total-latency limit"
                )
            predicted_response_cost = _score_predicted_response_cost(
                instance_id,
                output_lengths=output_lengths,
                instance_costs=instance_costs,
            )
            terms = published_score_candidate_terms(
                predicted_quality=float(
                    accuracy_scores.get(instance_id, 0.0)
                ),
                predicted_response_cost=predicted_response_cost,
                predicted_total_latency_ms=float(
                    wait_results[instance_id].wait_ms
                ),
                latency_limit_ms=float(latency_limit_ms),
                lambda_weight=float(lambda_weight),
                cost_weight=float(score_cost_weight),
                latency_weight=float(score_latency_weight),
                state=score_state,
            )
            return float(terms["score_candidate_value"])

        return score
    if utility in {
        "hard",
        "hard_prefill_tps",
        "hard_score_proxy",
        "hard_pk_mg1",
    }:

        def hard(
            instance_id: str,
            wait_results: Dict[str, WaitTimeResult],
            accuracy_scores: Dict[str, float],
            output_lengths: Dict[str, float],
            prompt_tokens: int,
        ) -> float:
            return hard_slo_candidate_value(
                instance_id=instance_id,
                wait_ms_by_instance={
                    candidate_id: float(wait_result.wait_ms)
                    for candidate_id, wait_result in wait_results.items()
                },
                slo_ms=utility_state.current_slo_ms,
                predicted_quality=float(accuracy_scores.get(instance_id, 0.0)),
                predicted_cost=_predicted_cost(
                    instance_id,
                    prompt_tokens=prompt_tokens,
                    output_lengths=output_lengths,
                    instance_costs=instance_costs,
                ),
                lambda_weight=lambda_weight,
            )

        return hard
    if utility == "slo_aware":

        def slo_aware(
            instance_id: str,
            wait_results: Dict[str, WaitTimeResult],
            accuracy_scores: Dict[str, float],
            output_lengths: Dict[str, float],
            prompt_tokens: int,
        ) -> float:
            wait = wait_results[instance_id].wait_ms
            slo_ms = utility_state.current_slo_ms
            if slo_ms is not None and wait > slo_ms:
                return -1e12 - wait
            return soft(
                instance_id,
                wait_results,
                accuracy_scores,
                output_lengths,
                prompt_tokens,
            )

        return slo_aware
    if utility == "round_robin":
        return lambda instance_id, wait_results, *_: -wait_results[instance_id].wait_ms
    if utility == "shortest_queue":
        return lambda instance_id, wait_results, *_: -wait_results[instance_id].wait_ms
    if utility == "instance_affinity":
        return lambda instance_id, wait_results, *_: -wait_results[instance_id].wait_ms
    raise ValueError(
        f"Unknown utility '{utility_name}'. "
        f"Supported: {', '.join(BUILTIN_UTILITIES)}."
    )


def build_default_instance_defs() -> list[dict[str, Any]]:
    user = os.environ.get("USER")
    job_id = os.environ.get("SLURM_JOB_ID")
    if not user or not job_id:
        raise EnvironmentError(
            "Missing USER or SLURM_JOB_ID. Provide --instances-config for a generic setup."
        )

    model_dir_qwen_0_6b = (
        f"/local/{user}/{job_id}/models--Qwen--Qwen3-0.6B/"
        "snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
    )
    model_dir_qwen3_8b = (
        f"/local/{user}/{job_id}/models--Qwen--Qwen3-8B/"
        "snapshots/b968826d9c46dd6066d109eabc6255188de91218"
    )
    model_dir_qwen3_32b = (
        f"/local/{user}/{job_id}/models--Qwen--Qwen3-32B/"
        "snapshots/9216db5781bf21249d130ec9da846c4624c16137"
    )
    return [
        {
            "instance_id": "vllm-0.6b",
            "address": "http://localhost:8002",
            "default_model": model_dir_qwen_0_6b,
            "model_id": "qwen3-0.6b",
            "max_num_batched_tokens": 32768,
            "max_num_seqs": 512,
            "chunked_prefill_enabled": True,
            "long_prefill_token_threshold": 0,
            "intercept": 0.002582076153059547,
            "prefill_coeff": 2.0335711505545034e-06,
            "prefill_sq_coeff": 1.8925400079684374e-10,
            "decode_coeff": 1.6889500804006565e-05,
            "sum_coeff": 3.5429418765562916e-08,
            "sum_sq_coeff": 1.5754738957656986e-13,
        },
        {
            "instance_id": "vllm-8b",
            "address": "http://localhost:8003",
            "default_model": model_dir_qwen3_8b,
            "model_id": "qwen3-8b",
            "max_num_batched_tokens": 32768,
            "max_num_seqs": 512,
            "chunked_prefill_enabled": True,
            "long_prefill_token_threshold": 0,
            "intercept": 0.007295929680901453,
            "prefill_coeff": 2.0789009922784725e-05,
            "prefill_sq_coeff": 5.119559033569263e-10,
            "decode_coeff": 2.5632010897991973e-05,
            "sum_coeff": 4.6490885513702363e-08,
            "sum_sq_coeff": 2.6851287984524727e-13,
        },
        {
            "instance_id": "vllm-32b",
            "address": "http://localhost:8001",
            "default_model": model_dir_qwen3_32b,
            "model_id": "qwen3-32b",
            "max_num_batched_tokens": 32768,
            "max_num_seqs": 512,
            "chunked_prefill_enabled": True,
            "long_prefill_token_threshold": 0,
            "intercept": 0.014347877296187458,
            "prefill_coeff": 5.294185118159464e-05,
            "prefill_sq_coeff": 8.774059443201359e-10,
            "decode_coeff": 6.304876078328351e-05,
            "sum_coeff": 3.6863941393981496e-08,
            "sum_sq_coeff": 4.519593317761812e-13,
        },
    ]


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if float(value) == 0.0:
            return False
        if float(value) == 1.0:
            return True
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return None


def _parse_instance_ttft_batch_params(
    raw_instance: Dict[str, Any],
    *,
    instance_id: str,
) -> Dict[str, Any]:
    source: Dict[str, Any] = dict(raw_instance)
    nested = raw_instance.get("ttft_batch_model")
    if isinstance(nested, dict):
        source.update(nested)

    parsed: Dict[str, Any] = {}

    int_fields = (
        "max_num_batched_tokens",
        "max_num_seqs",
        "long_prefill_token_threshold",
    )
    for field in int_fields:
        if field not in source:
            continue
        numeric = _as_nonnegative_int(source.get(field))
        if numeric is None:
            raise ValueError(
                f"instances-config: instance '{instance_id}' has invalid integer "
                f"field '{field}'."
            )
        parsed[field] = int(numeric)

    if "chunked_prefill_enabled" in source:
        parsed_bool = _coerce_bool(source.get("chunked_prefill_enabled"))
        if parsed_bool is None:
            raise ValueError(
                f"instances-config: instance '{instance_id}' has invalid boolean "
                "field 'chunked_prefill_enabled'."
            )
        parsed["chunked_prefill_enabled"] = parsed_bool

    float_fields = (
        "intercept",
        "prefill_coeff",
        "prefill_sq_coeff",
        "decode_coeff",
        "sum_coeff",
        "sum_sq_coeff",
    )
    for field in float_fields:
        if field not in source:
            continue
        value = source.get(field)
        if isinstance(value, bool):
            raise ValueError(
                f"instances-config: instance '{instance_id}' has invalid float "
                f"field '{field}'."
            )
        if not isinstance(value, (int, float, str)):
            raise ValueError(
                f"instances-config: instance '{instance_id}' has invalid float "
                f"field '{field}'."
            )
        try:
            numeric = float(value)
        except ValueError as exc:
            raise ValueError(
                f"instances-config: instance '{instance_id}' has non-numeric "
                f"field '{field}'."
            ) from exc
        if not math.isfinite(numeric):
            raise ValueError(
                f"instances-config: instance '{instance_id}' has non-finite "
                f"field '{field}'."
            )
        parsed[field] = numeric

    # Keep backwards compatibility with configs that predate quadratic terms.
    parsed.setdefault("prefill_sq_coeff", 0.0)
    parsed.setdefault("sum_sq_coeff", 0.0)

    return parsed


def _instance_ttft_params_from_metadata(
    instance_metadata: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    raw = instance_metadata.get("ttft_batch_params_by_instance")
    if not isinstance(raw, dict):
        return {}
    parsed: Dict[str, Dict[str, Any]] = {}
    for instance_id, payload in raw.items():
        if isinstance(payload, dict):
            params = dict(payload)
            params.setdefault("prefill_sq_coeff", 0.0)
            params.setdefault("sum_sq_coeff", 0.0)
            parsed[str(instance_id)] = params
    return parsed


def _validate_live_ttft_params(
    *,
    instance_ids: list[str],
    ttft_params_by_instance: Dict[str, Dict[str, Any]],
    context_label: str,
) -> None:
    missing_rows: list[str] = []
    for instance_id in instance_ids:
        params = ttft_params_by_instance.get(instance_id)
        if not isinstance(params, dict):
            missing_rows.append(
                f"{instance_id}: missing all required TTFT params {TTFT_BATCH_PARAM_KEYS}"
            )
            continue
        missing_keys = [key for key in TTFT_BATCH_PARAM_KEYS if key not in params]
        if missing_keys:
            missing_rows.append(f"{instance_id}: missing {missing_keys}")
    if missing_rows:
        joined = "; ".join(missing_rows)
        raise ValueError(
            f"{context_label}: live wait estimator requires TTFT batch-model params "
            f"for every active instance. {joined}"
        )


def load_instances(
    instances_config_path: Optional[Path],
) -> tuple[dict[str, InstanceClient], Dict[str, Dict[str, float]], dict[str, Any]]:
    if instances_config_path:
        payload = json.loads(instances_config_path.read_text(encoding="utf-8"))
        raw_instances = payload.get("instances")
        if not isinstance(raw_instances, list) or not raw_instances:
            raise ValueError("instances-config must contain a non-empty 'instances' list")
        instance_costs = payload.get("instance_costs", DEFAULT_INSTANCE_COSTS)
        # Preserve the serving/provenance contract consumed by stage validation
        # and result collation. Dropping these fields made valid pools fail only
        # after the servers had loaded, and also stripped results of provenance.
        metadata = {key: value for key, value in payload.items()
                    if key not in ("instances", "instance_costs")}
        if "serving_profile" in metadata and not isinstance(metadata["serving_profile"], dict):
            raise ValueError("instances-config 'serving_profile' must be an object")
        metadata["instances_config_path"] = str(instances_config_path)
    else:
        raw_instances = build_default_instance_defs()
        instance_costs = DEFAULT_INSTANCE_COSTS
        metadata = {"instances_config_path": None, "defaults": "qwen-localhost"}

    instances: dict[str, InstanceClient] = {}
    ttft_batch_params_by_instance: Dict[str, Dict[str, Any]] = {}
    for raw in raw_instances:
        instance_id = str(raw["instance_id"])
        snapshot_shm_name_raw = raw.get("snapshot_shm_name")
        snapshot_shm_name: Optional[str]
        if snapshot_shm_name_raw is None:
            snapshot_shm_name = None
        else:
            snapshot_shm_name = str(snapshot_shm_name_raw).strip()
            if not snapshot_shm_name:
                raise ValueError(
                    f"instances-config: instance '{instance_id}' has an empty "
                    "'snapshot_shm_name'."
                )

        snapshot_shm_size_bytes_raw = raw.get("snapshot_shm_size_bytes")
        snapshot_shm_size_bytes: Optional[int] = None
        if snapshot_shm_size_bytes_raw is not None:
            parsed_shm_size = _as_nonnegative_int(snapshot_shm_size_bytes_raw)
            if parsed_shm_size is None or parsed_shm_size <= 0:
                raise ValueError(
                    f"instances-config: instance '{instance_id}' has invalid "
                    "'snapshot_shm_size_bytes'."
                )
            snapshot_shm_size_bytes = int(parsed_shm_size)

        wait_time_http_fallback_enabled_raw = raw.get(
            "wait_time_http_fallback_enabled"
        )
        wait_time_http_fallback_enabled = False
        if wait_time_http_fallback_enabled_raw is not None:
            parsed_http_fallback = _coerce_bool(
                wait_time_http_fallback_enabled_raw
            )
            if parsed_http_fallback is None:
                raise ValueError(
                    f"instances-config: instance '{instance_id}' has invalid "
                    "'wait_time_http_fallback_enabled'."
                )
            wait_time_http_fallback_enabled = bool(parsed_http_fallback)

        client = InstanceClient(
            instance_id=instance_id,
            address=str(raw["address"]),
            default_model=str(raw["default_model"]),
            model_id=str(raw["model_id"]),
            wait_time_timeout_s=float(raw.get("wait_time_timeout_s", 5.0)),
            snapshot_shm_name=snapshot_shm_name,
            snapshot_shm_size_bytes=snapshot_shm_size_bytes,
            wait_time_http_fallback_enabled=wait_time_http_fallback_enabled,
        )
        instances[instance_id] = client
        ttft_batch_params_by_instance[instance_id] = _parse_instance_ttft_batch_params(
            raw,
            instance_id=instance_id,
        )

    metadata["ttft_batch_params_by_instance"] = ttft_batch_params_by_instance
    return instances, instance_costs, metadata

def _wait_estimator_live_queue(
    instance_id: str,
    instance: InstanceClient,
    live_wait_results: Dict[str, WaitTimeResult],
) -> Optional[float]:
    wait_record = live_wait_results.get(instance_id) or instance.last_wait
    if wait_record is None:
        return None
    return float(wait_record.wait_ms)


def _record_wait_estimator_diagnostics(
    context: Dict[str, Any],
    *,
    instance_id: str,
    diagnostics: Dict[str, Any],
) -> None:
    diagnostics_map = context.get("wait_estimator_diagnostics_by_instance")
    if isinstance(diagnostics_map, dict):
        diagnostics_map[instance_id] = diagnostics


def _extract_wait_metadata(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    reports = payload.get("reports")
    if not isinstance(reports, list) or not reports:
        return None
    report0 = reports[0]
    if not isinstance(report0, dict):
        return None
    metadata = report0.get("metadata")
    if not isinstance(metadata, dict):
        return None
    return metadata


def _extract_wait_num_requests(payload: Dict[str, Any]) -> Optional[int]:
    reports = payload.get("reports")
    if not isinstance(reports, list) or not reports:
        return None
    report0 = reports[0]
    if not isinstance(report0, dict):
        return None

    direct_num_requests = _as_nonnegative_int(report0.get("num_requests"))
    if direct_num_requests is not None:
        return int(direct_num_requests)

    metadata = report0.get("metadata")
    if not isinstance(metadata, dict):
        return None

    metadata_num_requests = _as_nonnegative_int(metadata.get("num_requests"))
    if metadata_num_requests is not None:
        return int(metadata_num_requests)

    running_at_snapshot = _as_nonnegative_int(metadata.get("running_at_snapshot"))
    queued_at_snapshot = _as_nonnegative_int(metadata.get("queued_at_snapshot"))
    if running_at_snapshot is not None and queued_at_snapshot is not None:
        return int(running_at_snapshot + queued_at_snapshot)

    num_running = _as_nonnegative_int(metadata.get("num_running"))
    num_waiting = _as_nonnegative_int(metadata.get("num_waiting"))
    if num_running is not None and num_waiting is not None:
        return int(num_running + num_waiting)

    return None


def _extract_pk_running_at_snapshot(payload: Dict[str, Any]) -> Optional[int]:
    metadata = _extract_wait_metadata(payload)
    if isinstance(metadata, dict):
        for key in ("running_at_snapshot", "num_running", "resident_set_size"):
            running_value = _as_nonnegative_int(metadata.get(key))
            if running_value is not None:
                return int(running_value)
    return _extract_wait_num_requests(payload)


def _extract_running_context_sum(metadata: Dict[str, Any]) -> Optional[float]:
    direct = _as_nonnegative_int(metadata.get("running_context_length_sum_snapshot"))
    if direct is not None:
        return float(direct)
    return None


def _extract_running_context_sq_sum(metadata: Dict[str, Any]) -> Optional[float]:
    direct = _as_nonnegative_int(metadata.get("running_context_length_sq_sum_snapshot"))
    if direct is not None:
        return float(direct)
    return None


def _extract_running_prefill_sq_sum(metadata: Dict[str, Any]) -> float:
    running = _as_nonnegative_int(metadata.get("prefill_backlog_running_sq_sum_tokens"))
    if running is not None:
        return float(running)
    total = _as_nonnegative_int(metadata.get("prefill_backlog_total_sq_sum_tokens"))
    if total is not None:
        return float(total)
    return 0.0


def _build_prefill_chunks(
    *,
    prompt_tokens: int,
    chunked_prefill_enabled: bool,
    long_prefill_token_threshold: int,
    max_num_batched_tokens: int,
) -> list[int]:
    if prompt_tokens <= 0:
        return []
    if not chunked_prefill_enabled:
        return [int(prompt_tokens)]

    chunk_size = int(long_prefill_token_threshold)
    if chunk_size <= 0:
        chunk_size = int(max_num_batched_tokens)
    chunk_size = max(chunk_size, 1)

    remaining = int(prompt_tokens)
    chunks: list[int] = []
    while remaining > 0:
        take = min(chunk_size, remaining)
        chunks.append(int(take))
        remaining -= take
    return chunks


def _resolve_live_ttft_params(
    *,
    instance_id: str,
    context: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    params_by_instance = context.get("ttft_instance_params")
    if not isinstance(params_by_instance, dict):
        return None
    raw_params = params_by_instance.get(instance_id)
    if not isinstance(raw_params, dict):
        return None

    max_num_batched_tokens = _as_nonnegative_int(raw_params.get("max_num_batched_tokens"))
    max_num_seqs = _as_nonnegative_int(raw_params.get("max_num_seqs"))
    chunked_prefill_enabled = _coerce_bool(raw_params.get("chunked_prefill_enabled"))
    long_prefill_token_threshold = _as_nonnegative_int(
        raw_params.get("long_prefill_token_threshold")
    )

    if (
        max_num_batched_tokens is None
        or max_num_batched_tokens <= 0
        or max_num_seqs is None
        or max_num_seqs <= 0
        or chunked_prefill_enabled is None
        or long_prefill_token_threshold is None
    ):
        return None

    float_params: Dict[str, float] = {}
    for key in (
        "intercept",
        "prefill_coeff",
        "prefill_sq_coeff",
        "decode_coeff",
        "sum_coeff",
        "sum_sq_coeff",
    ):
        value = raw_params.get(key)
        if value is None:
            if key in {"prefill_sq_coeff", "sum_sq_coeff"}:
                float_params[key] = 0.0
                continue
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            if key in {"prefill_sq_coeff", "sum_sq_coeff"}:
                float_params[key] = 0.0
                continue
            return None
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        float_params[key] = numeric

    return {
        "max_num_batched_tokens": int(max_num_batched_tokens),
        "max_num_seqs": int(max_num_seqs),
        "chunked_prefill_enabled": bool(chunked_prefill_enabled),
        "long_prefill_token_threshold": int(long_prefill_token_threshold),
        **float_params,
    }


def _estimate_live_prefill_term_ms(
    *,
    wait_payload: Dict[str, Any],
    prompt_tokens: int,
    ttft_params: Dict[str, Any],
) -> tuple[Optional[float], int, Optional[str], Dict[str, Any]]:
    metadata = _extract_wait_metadata(wait_payload)
    if metadata is None:
        return None, 0, "missing_wait_metadata", {}

    num_running = _as_nonnegative_int(metadata.get("resident_set_size"))
    if num_running is None:
        num_running = _as_nonnegative_int(metadata.get("running_at_snapshot"))
    if num_running is None:
        return None, 0, "missing_num_running", {}

    running_context_sum = _extract_running_context_sum(metadata)
    if running_context_sum is None:
        return None, 0, "missing_running_context_length_sum_snapshot", {}
    sum_sq_coeff = float(ttft_params.get("sum_sq_coeff", 0.0))
    running_context_sq_sum = _extract_running_context_sq_sum(metadata)
    if abs(sum_sq_coeff) > 0.0 and running_context_sq_sum is None:
        return None, 0, "missing_running_context_length_sq_sum_snapshot", {}
    if running_context_sq_sum is None:
        running_context_sq_sum = 0.0
    running_prefill_sq_sum = _extract_running_prefill_sq_sum(metadata)

    decode_other_per_batch = min(int(num_running), int(ttft_params["max_num_seqs"]))
    prefill_chunks = _build_prefill_chunks(
        prompt_tokens=int(prompt_tokens),
        chunked_prefill_enabled=bool(ttft_params["chunked_prefill_enabled"]),
        long_prefill_token_threshold=int(ttft_params["long_prefill_token_threshold"]),
        max_num_batched_tokens=int(ttft_params["max_num_batched_tokens"]),
    )

    prefill_term_ms = 0.0
    other_context_sum = float(running_context_sum)
    other_context_sq_sum = float(running_context_sq_sum)
    for prefill_tokens in prefill_chunks:
        sum_context_length = float(prompt_tokens) + other_context_sum
        sum_sq_tokens = float(prompt_tokens * prompt_tokens) + other_context_sq_sum
        prefill_sq_sum = (
            float(prefill_tokens * prefill_tokens) + float(running_prefill_sq_sum)
        )
        batch_ms = 1000.0 * (
            float(ttft_params["intercept"])
            + float(ttft_params["prefill_coeff"]) * float(prefill_tokens)
            + float(ttft_params["prefill_sq_coeff"]) * prefill_sq_sum
            + float(ttft_params["decode_coeff"]) * float(decode_other_per_batch)
            + float(ttft_params["sum_coeff"]) * float(sum_context_length)
            + float(ttft_params["sum_sq_coeff"]) * float(sum_sq_tokens)
        )
        if not math.isfinite(batch_ms):
            return None, 0, "non_finite_batch_ms", {}
        prefill_term_ms += max(batch_ms, 0.0)
        other_context_sq_sum += (
            2.0 * other_context_sum + float(decode_other_per_batch)
        )
        other_context_sum += float(decode_other_per_batch)

    details = {
        "num_running": int(num_running),
        "decode_other_per_batch": int(decode_other_per_batch),
        "running_context_sum": float(running_context_sum),
        "running_context_sq_sum": float(running_context_sq_sum),
        "running_prefill_sq_sum": float(running_prefill_sq_sum),
    }
    return float(prefill_term_ms), len(prefill_chunks), None, details


def _wait_estimator_live(
    instance_id: str,
    instance: InstanceClient,
    live_wait_results: Dict[str, WaitTimeResult],
    context: Dict[str, Any],
) -> Optional[float]:
    ttft_ms = _wait_estimator_live_queue(instance_id, instance, live_wait_results)
    if ttft_ms is None:
        return None

    prompt_tokens = _as_nonnegative_int(context.get("prompt_tokens"))
    diagnostics: Dict[str, Any] = {
        "method": "live_ttft",
        "estimated_queue_ms": None,
        "estimated_prefill_ms": None,
        "estimated_ttft_ms": float(ttft_ms),
        "estimated_prefill_batches": None,
        "estimated_fallback_reason": None,
        "details": {
            "prompt_tokens": int(prompt_tokens) if prompt_tokens is not None else None,
            "simulation_mode": "critical_path_prefill_done",
        },
    }
    _record_wait_estimator_diagnostics(
        context,
        instance_id=instance_id,
        diagnostics=diagnostics,
    )
    return float(ttft_ms)


def _wait_estimator_cached(
    instance_id: str,
    instance: InstanceClient,
    live_wait_results: Dict[str, WaitTimeResult],
    _context: Dict[str, Any],
) -> Optional[float]:
    if instance.last_wait is not None:
        return float(instance.last_wait.wait_ms)
    return _wait_estimator_live_queue(instance_id, instance, live_wait_results)


def _wait_estimator_zero(
    _instance_id: str,
    _instance: InstanceClient,
    _live_wait_results: Dict[str, WaitTimeResult],
    _context: Dict[str, Any],
) -> Optional[float]:
    return 0.0

def _as_nonnegative_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
    elif isinstance(value, str):
        try:
            numeric = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(numeric) or numeric < 0:
        return None
    return numeric


def _normalize_prefill_tps_key(value: Any) -> str:
    from sfs_core.shared.model_label_helpers import resolve_model_label
    raw = str(value or "").strip().lower()
    if not raw:
        return raw
    # Resolve the explicit family before the historical bare-size aliases.
    # Otherwise Ministral 8B aliases overwrite Qwen 8B metrics and can fall
    # back to Qwen defaults when family calibration is missing.
    family_label = resolve_model_label(raw, raw)
    if family_label is not None:
        return family_label
    if raw in PREFILL_TPS_ALIASES:
        return PREFILL_TPS_ALIASES[raw]
    if "qwen3-0.6b" in raw or "qwen3-0_6b" in raw or "0.6b" in raw or "0_6b" in raw:
        return "qwen3-0.6b"
    if "qwen3-32b" in raw or "32b" in raw:
        return "qwen3-32b"
    if "qwen3-8b" in raw or "8b" in raw:
        return "qwen3-8b"
    return raw


def _parse_prefill_tps_overrides(values: list[str]) -> Dict[str, float]:
    parsed: Dict[str, float] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(
                f"Invalid --prefill-tps override '{item}'. Use key=value format."
            )
        key, raw_value = item.split("=", 1)
        key = _normalize_prefill_tps_key(key)
        raw_value = raw_value.strip()
        if not key:
            raise ValueError(f"Invalid --prefill-tps override '{item}': empty key.")
        try:
            numeric = float(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --prefill-tps override '{item}': value is not numeric."
            ) from exc
        if not math.isfinite(numeric) or numeric <= 0:
            raise ValueError(
                f"Invalid --prefill-tps override '{item}': value must be > 0."
            )
        parsed[key] = numeric
    return parsed


def _first_positive_metric(*values: Any) -> Optional[float]:
    for value in values:
        numeric = _as_nonnegative_float(value)
        if numeric is not None and numeric > 0:
            return float(numeric)
    return None


def _load_calibrated_service_metrics(
    path: Optional[Path],
) -> Dict[str, Dict[str, float]]:
    if path is None:
        return {}
    resolved_path = path.expanduser().resolve()
    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(
            f"Service-metrics JSON does not exist: {resolved_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Service-metrics JSON is invalid: {resolved_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"Service-metrics JSON must contain an object: {resolved_path}"
        )

    model_rows = payload.get("models", payload)
    if not isinstance(model_rows, dict):
        raise ValueError(
            f"Service-metrics JSON 'models' entry must be an object: {resolved_path}"
        )

    calibrated: Dict[str, Dict[str, float]] = {}
    for raw_key, raw_row in model_rows.items():
        if not isinstance(raw_row, dict):
            continue
        score_row = raw_row.get("score_proxy")
        if not isinstance(score_row, dict):
            score_row = {}
        prefill_row = raw_row.get("prefill_theta")
        if not isinstance(prefill_row, dict):
            prefill_row = {}

        # Router backlog is an aggregate token count, so its service rate must
        # be the aggregate batch-stat throughput. Per-request prefill durations
        # repeat shared batch time and are not additive under batching.
        prefill_tps = _first_positive_metric(
            prefill_row.get("theta_p_tps_from_batch_stats"),
            score_row.get("prefill_tps"),
            raw_row.get("prefill_tps"),
            prefill_row.get("theta_p_tps_from_wait_logs"),
        )
        decode_tps = _first_positive_metric(
            score_row.get("decode_tps"),
            raw_row.get("decode_tps"),
        )
        mean_decode_batch_ms = _first_positive_metric(
            score_row.get("mean_decode_batch_ms"),
            score_row.get("first_decode_batch_ms"),
            raw_row.get("mean_decode_batch_ms"),
            raw_row.get("first_decode_batch_ms"),
        )

        metrics: Dict[str, float] = {}
        if prefill_tps is not None:
            metrics["prefill_tps"] = prefill_tps
        if decode_tps is not None:
            metrics["decode_tps"] = decode_tps
        if mean_decode_batch_ms is not None:
            metrics["mean_decode_batch_ms"] = mean_decode_batch_ms
        if metrics:
            calibrated[_normalize_prefill_tps_key(raw_key)] = metrics

    if not calibrated:
        raise ValueError(
            f"Service-metrics JSON contains no usable model metrics: {resolved_path}"
        )
    return calibrated


def _build_prefill_tps_lookup(
    *,
    overrides: Optional[Dict[str, float]] = None,
    calibrated_metrics: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, float]:
    lookup: Dict[str, float] = {
        _normalize_prefill_tps_key(key): float(value)
        for key, value in DEFAULT_PREFILL_TPS.items()
    }
    for key, metrics in (calibrated_metrics or {}).items():
        numeric = _first_positive_metric(metrics.get("prefill_tps"))
        if numeric is not None:
            lookup[_normalize_prefill_tps_key(key)] = numeric
    if overrides:
        for key, value in overrides.items():
            normalized_key = _normalize_prefill_tps_key(key)
            numeric = float(value)
            if numeric > 0 and math.isfinite(numeric):
                lookup[normalized_key] = numeric
    return lookup


def _resolve_prefill_tps_for_instance(
    *,
    instance_id: str,
    instance: InstanceClient,
    lookup: Dict[str, float],
) -> tuple[Optional[float], list[str]]:
    candidates = (
        instance_id,
        getattr(instance, "model_id", None),
        getattr(instance, "default_model", None),
    )
    candidate_keys: list[str] = []
    for candidate in candidates:
        key = _normalize_prefill_tps_key(candidate)
        if not key or key in candidate_keys:
            continue
        candidate_keys.append(key)
        numeric = _as_nonnegative_float(lookup.get(key))
        if numeric is not None and numeric > 0:
            return float(numeric), candidate_keys
    return None, candidate_keys


def _resolve_router_prefill_tps_by_instance(
    *,
    instances: Dict[str, InstanceClient],
    overrides: Optional[Dict[str, float]] = None,
    calibrated_metrics: Optional[Dict[str, Dict[str, float]]] = None,
) -> tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
    lookup = _build_prefill_tps_lookup(
        overrides=overrides,
        calibrated_metrics=calibrated_metrics,
    )
    resolved: Dict[str, float] = {}
    diagnostics: Dict[str, Dict[str, Any]] = {}
    missing: list[str] = []

    for instance_id, instance in instances.items():
        prefill_tps, candidate_keys = _resolve_prefill_tps_for_instance(
            instance_id=instance_id,
            instance=instance,
            lookup=lookup,
        )
        diagnostics[instance_id] = {
            "prefill_tps": prefill_tps,
            "candidate_keys": candidate_keys,
        }
        if prefill_tps is None:
            missing.append(f"{instance_id} candidates={candidate_keys}")
            continue
        resolved[instance_id] = float(prefill_tps)

    if missing:
        raise ValueError(
            "Prefill-TPS utility requires per-instance prefill TPS. Missing mappings for: "
            + "; ".join(missing)
            + ". Supply overrides via --prefill-tps key=value."
        )
    return resolved, diagnostics


def _resolve_score_proxy_metrics_by_instance(
    *,
    instances: Dict[str, InstanceClient],
    calibrated_metrics: Dict[str, Dict[str, float]],
) -> tuple[
    Dict[str, float],
    Dict[str, float],
    Dict[str, Dict[str, Any]],
]:
    decode_tps_by_instance: Dict[str, float] = {}
    mean_decode_batch_ms_by_instance: Dict[str, float] = {}
    diagnostics: Dict[str, Dict[str, Any]] = {}
    missing: list[str] = []

    for instance_id, instance in instances.items():
        candidates = (
            instance_id,
            getattr(instance, "model_id", None),
            getattr(instance, "default_model", None),
        )
        candidate_keys: list[str] = []
        matched_key: Optional[str] = None
        metrics: Optional[Dict[str, float]] = None
        for candidate in candidates:
            key = _normalize_prefill_tps_key(candidate)
            if not key or key in candidate_keys:
                continue
            candidate_keys.append(key)
            candidate_metrics = calibrated_metrics.get(key)
            if isinstance(candidate_metrics, dict):
                matched_key = key
                metrics = candidate_metrics
                break

        decode_tps = _first_positive_metric(
            metrics.get("decode_tps") if metrics is not None else None
        )
        mean_decode_batch_ms = _first_positive_metric(
            metrics.get("mean_decode_batch_ms") if metrics is not None else None
        )
        diagnostics[instance_id] = {
            "matched_key": matched_key,
            "candidate_keys": candidate_keys,
            "decode_tps": decode_tps,
            "mean_decode_batch_ms": mean_decode_batch_ms,
        }
        missing_fields: list[str] = []
        if decode_tps is None:
            missing_fields.append("decode_tps")
        if mean_decode_batch_ms is None:
            missing_fields.append("mean_decode_batch_ms")
        if missing_fields:
            missing.append(
                f"{instance_id} missing={missing_fields} candidates={candidate_keys}"
            )
            continue
        decode_tps_by_instance[instance_id] = decode_tps
        mean_decode_batch_ms_by_instance[instance_id] = mean_decode_batch_ms

    if missing:
        raise ValueError(
            "SCORE proxy requires calibrated decode TPS and mean decode-batch "
            "latency for every instance. Missing mappings for: "
            + "; ".join(missing)
            + ". Supply --service-metrics-json from the matching hardware and "
            "vLLM configuration."
        )
    return (
        decode_tps_by_instance,
        mean_decode_batch_ms_by_instance,
        diagnostics,
    )


def _build_service_rate_lookup() -> Dict[str, float]:
    return {
        _normalize_prefill_tps_key(key): float(value)
        for key, value in DEFAULT_SERVICE_RATES_RPS.items()
    }


def _resolve_service_rate_for_instance(
    *,
    instance_id: str,
    instance: InstanceClient,
    lookup: Dict[str, float],
) -> tuple[Optional[float], list[str]]:
    candidates = (
        instance_id,
        getattr(instance, "model_id", None),
        getattr(instance, "default_model", None),
    )
    candidate_keys: list[str] = []
    for candidate in candidates:
        key = _normalize_prefill_tps_key(candidate)
        if not key or key in candidate_keys:
            continue
        candidate_keys.append(key)
        numeric = _as_nonnegative_float(lookup.get(key))
        if numeric is not None and numeric > 0:
            return float(numeric), candidate_keys
    return None, candidate_keys


def _resolve_router_service_rates_by_instance(
    *,
    instances: Dict[str, InstanceClient],
) -> tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
    lookup = _build_service_rate_lookup()
    resolved: Dict[str, float] = {}
    diagnostics: Dict[str, Dict[str, Any]] = {}
    missing: list[str] = []

    for instance_id, instance in instances.items():
        service_rate_rps, candidate_keys = _resolve_service_rate_for_instance(
            instance_id=instance_id,
            instance=instance,
            lookup=lookup,
        )
        diagnostics[instance_id] = {
            "service_rate_rps": service_rate_rps,
            "candidate_keys": candidate_keys,
        }
        if service_rate_rps is None:
            missing.append(f"{instance_id} candidates={candidate_keys}")
            continue
        resolved[instance_id] = float(service_rate_rps)

    if missing:
        raise ValueError(
            "PK-MG1 requires per-instance service-rate priors. Missing mappings for: "
            + "; ".join(missing)
            + ". Update DEFAULT_SERVICE_RATES_RPS for these model aliases."
        )
    return resolved, diagnostics


def _prefill_backlog_tokens_from_wait_payload(payload: Dict[str, Any]) -> Optional[float]:
    reports = payload.get("reports")
    if not isinstance(reports, list) or not reports:
        return None
    report0 = reports[0]
    if not isinstance(report0, dict):
        return None
    metadata = report0.get("metadata")
    if not isinstance(metadata, dict):
        return None

    backlog_total = _as_nonnegative_float(metadata.get("prefill_backlog_total_tokens"))
    if backlog_total is not None:
        return backlog_total

    backlog_running = _as_nonnegative_float(metadata.get("prefill_backlog_running_tokens"))
    backlog_waiting = _as_nonnegative_float(metadata.get("prefill_backlog_waiting_tokens"))
    if backlog_running is not None and backlog_waiting is not None:
        return backlog_running + backlog_waiting
    return None


def _snapshot_metadata_from_wait_payload(
    payload: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    reports = payload.get("reports")
    if not isinstance(reports, list) or not reports:
        return None
    report0 = reports[0]
    if not isinstance(report0, dict):
        return None
    metadata = report0.get("metadata")
    return metadata if isinstance(metadata, dict) else None


def _wait_estimator_prefill_tps_ttft(
    instance_id: str,
    instance: InstanceClient,
    live_wait_results: Dict[str, WaitTimeResult],
    context: Dict[str, Any],
) -> Optional[float]:
    prefill_tps_by_instance = context.get("prefill_tps_by_instance")
    prefill_tps_value = None
    if isinstance(prefill_tps_by_instance, dict):
        prefill_tps_value = prefill_tps_by_instance.get(instance_id)
    prefill_tps = _as_nonnegative_float(prefill_tps_value)
    if prefill_tps is None or prefill_tps <= 0:
        raise ValueError(
            f"{PREFILL_TPS_TTFT_ESTIMATOR_NAME} estimator requires positive "
            f"prefill TPS for instance '{instance_id}'."
        )

    prompt_tokens = _as_nonnegative_int(context.get("prompt_tokens"))
    prompt_tokens_value = int(prompt_tokens) if prompt_tokens is not None else 0

    diagnostics: Dict[str, Any] = {
        "method": PREFILL_TPS_TTFT_ESTIMATOR_NAME,
        "estimated_prefill_ms": None,
        "estimated_ttft_ms": None,
        "estimated_fallback_reason": None,
    }

    wait_record = live_wait_results.get(instance_id) or instance.last_wait
    if wait_record is None or not isinstance(wait_record.raw_payload, dict):
        diagnostics["estimated_fallback_reason"] = "missing_wait_payload"
        _record_wait_estimator_diagnostics(
            context,
            instance_id=instance_id,
            diagnostics=diagnostics,
        )
        return None

    backlog_tokens = _prefill_backlog_tokens_from_wait_payload(wait_record.raw_payload)
    if backlog_tokens is None:
        diagnostics["estimated_fallback_reason"] = "missing_prefill_backlog_tokens"
        _record_wait_estimator_diagnostics(
            context,
            instance_id=instance_id,
            diagnostics=diagnostics,
        )
        return None

    total_prefill_tokens = float(backlog_tokens) + float(prompt_tokens_value)
    ttft_ms = float(total_prefill_tokens / float(prefill_tps) * 1000.0)
    diagnostics.update(
        {
            "estimated_prefill_ms": ttft_ms,
            "estimated_ttft_ms": ttft_ms,
            "estimated_fallback_reason": None,
            "details": {
                "prefill_backlog_total_tokens": float(backlog_tokens),
                "prompt_tokens": int(prompt_tokens_value),
                "prefill_tps": float(prefill_tps),
            },
        }
    )
    _record_wait_estimator_diagnostics(
        context,
        instance_id=instance_id,
        diagnostics=diagnostics,
    )
    return ttft_ms


def _wait_estimator_score_proxy_ttft(
    instance_id: str,
    instance: InstanceClient,
    live_wait_results: Dict[str, WaitTimeResult],
    context: Dict[str, Any],
) -> Optional[float]:
    prefill_tps_by_instance = context.get("prefill_tps_by_instance")
    decode_tps_by_instance = context.get("decode_tps_by_instance")
    mean_decode_batch_ms_by_instance = context.get(
        "mean_decode_batch_ms_by_instance"
    )
    prefill_tps = _first_positive_metric(
        prefill_tps_by_instance.get(instance_id)
        if isinstance(prefill_tps_by_instance, dict)
        else None
    )
    decode_tps = _first_positive_metric(
        decode_tps_by_instance.get(instance_id)
        if isinstance(decode_tps_by_instance, dict)
        else None
    )
    mean_decode_batch_ms = _first_positive_metric(
        mean_decode_batch_ms_by_instance.get(instance_id)
        if isinstance(mean_decode_batch_ms_by_instance, dict)
        else None
    )
    if (
        prefill_tps is None
        or decode_tps is None
        or mean_decode_batch_ms is None
    ):
        raise ValueError(
            f"{SCORE_PROXY_TTFT_ESTIMATOR_NAME} requires positive prefill TPS, "
            f"decode TPS, and mean decode-batch latency for instance "
            f"'{instance_id}'."
        )

    prompt_tokens = _as_nonnegative_int(context.get("prompt_tokens"))
    prompt_tokens_value = int(prompt_tokens) if prompt_tokens is not None else 0
    diagnostics: Dict[str, Any] = {
        "method": SCORE_PROXY_TTFT_ESTIMATOR_NAME,
        "score_exact_reproduction": False,
        "target": "ttft",
        "estimated_prefill_ms": None,
        "estimated_decode_interference_ms": None,
        "estimated_first_decode_batch_ms": float(mean_decode_batch_ms),
        "estimated_ttft_ms": None,
        "estimated_fallback_reason": None,
    }

    wait_record = live_wait_results.get(instance_id) or instance.last_wait
    if wait_record is None or not isinstance(wait_record.raw_payload, dict):
        diagnostics["estimated_fallback_reason"] = "missing_wait_payload"
        _record_wait_estimator_diagnostics(
            context,
            instance_id=instance_id,
            diagnostics=diagnostics,
        )
        raise RuntimeError(
            f"{SCORE_PROXY_TTFT_ESTIMATOR_NAME} requires snapshot metadata for "
            f"instance '{instance_id}'; no wait payload was available."
        )

    metadata = _snapshot_metadata_from_wait_payload(wait_record.raw_payload)
    prefill_backlog_tokens = (
        _prefill_backlog_tokens_from_wait_payload(wait_record.raw_payload)
    )
    decode_backlog_tokens = (
        _as_nonnegative_float(metadata.get("decode_backlog_total_tokens"))
        if metadata is not None
        else None
    )
    if prefill_backlog_tokens is None or decode_backlog_tokens is None:
        diagnostics["estimated_fallback_reason"] = "missing_effective_backlog_tokens"
        _record_wait_estimator_diagnostics(
            context,
            instance_id=instance_id,
            diagnostics=diagnostics,
        )
        raise RuntimeError(
            f"{SCORE_PROXY_TTFT_ESTIMATOR_NAME} requires effective prefill and "
            f"decode backlog metadata for instance '{instance_id}'."
        )

    ttft_ms, terms = estimate_score_proxy_ttft_ms(
        prompt_tokens=prompt_tokens_value,
        prefill_backlog_tokens=float(prefill_backlog_tokens),
        decode_backlog_tokens=float(decode_backlog_tokens),
        prefill_tps=prefill_tps,
        decode_tps=decode_tps,
        mean_decode_batch_ms=mean_decode_batch_ms,
    )
    diagnostics.update(
        {
            "estimated_prefill_ms": float(terms["prefill_ms"]),
            "estimated_decode_interference_ms": float(
                terms["decode_interference_ms"]
            ),
            "estimated_ttft_ms": float(ttft_ms),
            "details": {
                "prefill_backlog_total_tokens": float(prefill_backlog_tokens),
                "decode_backlog_total_tokens": float(decode_backlog_tokens),
                "prompt_tokens": int(prompt_tokens_value),
                "prefill_tps": float(prefill_tps),
                "decode_tps": float(decode_tps),
                "mean_decode_batch_ms": float(mean_decode_batch_ms),
                "pending_overlay_count": (
                    int(metadata["pending_overlay_count"])
                    if metadata is not None
                    and isinstance(metadata.get("pending_overlay_count"), int)
                    else None
                ),
            },
        }
    )
    _record_wait_estimator_diagnostics(
        context,
        instance_id=instance_id,
        diagnostics=diagnostics,
    )
    return float(ttft_ms)


def _wait_estimator_score_total_latency(
    instance_id: str,
    instance: InstanceClient,
    live_wait_results: Dict[str, WaitTimeResult],
    context: Dict[str, Any],
) -> Optional[float]:
    """Estimate SCORE's predicted total response latency for one candidate."""

    prefill_tps_by_instance = context.get("prefill_tps_by_instance")
    decode_tps_by_instance = context.get("decode_tps_by_instance")
    mean_decode_batch_ms_by_instance = context.get(
        "mean_decode_batch_ms_by_instance"
    )
    output_lengths = context.get("output_lengths")
    prefill_tps = _first_positive_metric(
        prefill_tps_by_instance.get(instance_id)
        if isinstance(prefill_tps_by_instance, dict)
        else None
    )
    decode_tps = _first_positive_metric(
        decode_tps_by_instance.get(instance_id)
        if isinstance(decode_tps_by_instance, dict)
        else None
    )
    mean_decode_batch_ms = _first_positive_metric(
        mean_decode_batch_ms_by_instance.get(instance_id)
        if isinstance(mean_decode_batch_ms_by_instance, dict)
        else None
    )
    predicted_output_tokens = _as_nonnegative_float(
        output_lengths.get(instance_id)
        if isinstance(output_lengths, dict)
        else None
    )
    if (
        prefill_tps is None
        or decode_tps is None
        or mean_decode_batch_ms is None
        or predicted_output_tokens is None
    ):
        raise ValueError(
            f"{SCORE_TOTAL_LATENCY_ESTIMATOR_NAME} requires positive prefill TPS, "
            "decode TPS, mean decode-batch latency, and a predicted output "
            f"length for instance '{instance_id}'."
        )

    prompt_tokens = _as_nonnegative_int(context.get("prompt_tokens"))
    prompt_tokens_value = int(prompt_tokens) if prompt_tokens is not None else 0
    diagnostics: Dict[str, Any] = {
        "method": SCORE_TOTAL_LATENCY_ESTIMATOR_NAME,
        "published_score_formula": True,
        "target": "total_response_latency",
        "estimated_waiting_time_ms": None,
        "estimated_runtime_ms": None,
        "estimated_total_latency_ms": None,
        "estimated_fallback_reason": None,
    }

    wait_record = live_wait_results.get(instance_id) or instance.last_wait
    if wait_record is None or not isinstance(wait_record.raw_payload, dict):
        diagnostics["estimated_fallback_reason"] = "missing_wait_payload"
        _record_wait_estimator_diagnostics(
            context,
            instance_id=instance_id,
            diagnostics=diagnostics,
        )
        raise RuntimeError(
            f"{SCORE_TOTAL_LATENCY_ESTIMATOR_NAME} requires snapshot metadata "
            f"for instance '{instance_id}'; no wait payload was available."
        )

    metadata = _snapshot_metadata_from_wait_payload(wait_record.raw_payload)
    prefill_backlog_tokens = _prefill_backlog_tokens_from_wait_payload(
        wait_record.raw_payload
    )
    decode_backlog_tokens = (
        _as_nonnegative_float(metadata.get("decode_backlog_total_tokens"))
        if metadata is not None
        else None
    )
    if prefill_backlog_tokens is None or decode_backlog_tokens is None:
        diagnostics["estimated_fallback_reason"] = "missing_effective_backlog_tokens"
        _record_wait_estimator_diagnostics(
            context,
            instance_id=instance_id,
            diagnostics=diagnostics,
        )
        raise RuntimeError(
            f"{SCORE_TOTAL_LATENCY_ESTIMATOR_NAME} requires effective prefill "
            f"and decode backlog metadata for instance '{instance_id}'."
        )

    total_latency_ms, terms = estimate_score_total_latency_ms(
        prompt_tokens=prompt_tokens_value,
        predicted_output_tokens=float(predicted_output_tokens),
        prefill_backlog_tokens=float(prefill_backlog_tokens),
        decode_backlog_tokens=float(decode_backlog_tokens),
        prefill_tps=float(prefill_tps),
        decode_tps=float(decode_tps),
        mean_decode_batch_ms=float(mean_decode_batch_ms),
    )
    diagnostics.update(
        {
            "estimated_waiting_time_ms": float(terms["waiting_time_ms"]),
            "estimated_runtime_ms": float(terms["predicted_runtime_ms"]),
            "estimated_total_latency_ms": float(total_latency_ms),
            "details": {
                "prefill_backlog_total_tokens": float(prefill_backlog_tokens),
                "decode_backlog_total_tokens": float(decode_backlog_tokens),
                "prompt_tokens": int(prompt_tokens_value),
                "predicted_output_tokens": float(predicted_output_tokens),
                "prefill_tps": float(prefill_tps),
                "decode_tps": float(decode_tps),
                "mean_decode_batch_ms": float(mean_decode_batch_ms),
                "prefill_wait_ms": float(terms["prefill_wait_ms"]),
                "decode_backlog_wait_ms": float(
                    terms["decode_backlog_wait_ms"]
                ),
                "pending_overlay_count": (
                    int(metadata["pending_overlay_count"])
                    if metadata is not None
                    and isinstance(metadata.get("pending_overlay_count"), int)
                    else None
                ),
            },
        }
    )
    _record_wait_estimator_diagnostics(
        context,
        instance_id=instance_id,
        diagnostics=diagnostics,
    )
    return float(total_latency_ms)


def _wait_estimator_pk_mg1(
    instance_id: str,
    instance: InstanceClient,
    live_wait_results: Dict[str, WaitTimeResult],
    context: Dict[str, Any],
) -> Optional[float]:
    tracker = context.get("pk_tracker")
    if not isinstance(tracker, PKOnlineStatsCollector):
        return _wait_estimator_live_queue(instance_id, instance, live_wait_results)
    prefill_tps_by_instance = context.get("prefill_tps_by_instance")
    prefill_tps_value = None
    if isinstance(prefill_tps_by_instance, dict):
        prefill_tps_value = prefill_tps_by_instance.get(instance_id)
    prefill_tps = _as_nonnegative_float(prefill_tps_value)
    if prefill_tps is None or prefill_tps <= 0:
        raise ValueError(
            f"pk_mg1 estimator requires positive prefill TPS for instance '{instance_id}'."
        )

    prompt_tokens = _as_nonnegative_int(context.get("prompt_tokens"))
    prompt_tokens_value = int(prompt_tokens) if prompt_tokens is not None else 0

    wait_record = live_wait_results.get(instance_id) or instance.last_wait
    running_at_snapshot = None
    if wait_record is not None and isinstance(wait_record.raw_payload, dict):
        running_at_snapshot = _extract_pk_running_at_snapshot(wait_record.raw_payload)

    estimate_ms, diagnostics = tracker.estimate_wait_ms(
        instance_id,
        prompt_tokens=prompt_tokens_value,
        prefill_tps=float(prefill_tps),
        running_at_snapshot=running_at_snapshot,
    )
    diagnostics_map = context.get("pk_last_diagnostics")
    if isinstance(diagnostics_map, dict):
        diagnostics_map[instance_id] = diagnostics
    return float(estimate_ms)

def _load_wait_estimator_callable(spec: str) -> WaitEstimatorCallable:
    module_spec, fn_name = spec.rsplit(":", 1)
    source_path = Path(module_spec).expanduser()
    if source_path.exists():
        loader_spec = importlib.util.spec_from_file_location(
            f"_wait_estimator_{source_path.stem}",
            str(source_path),
        )
        if loader_spec is None or loader_spec.loader is None:
            raise ImportError(f"Could not load estimator module from path: {source_path}")
        module = importlib.util.module_from_spec(loader_spec)
        loader_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_spec)
    fn = getattr(module, fn_name, None)
    if not callable(fn):
        raise TypeError(f"Estimator '{spec}' is not callable.")
    return fn


def resolve_wait_estimator(spec: str) -> tuple[str, WaitEstimatorCallable]:
    key = spec.strip().lower()
    if key == "live":
        return "live", _wait_estimator_live
    if key == "cached":
        return "cached", _wait_estimator_cached
    if key == "zero":
        return "zero", _wait_estimator_zero
    if key == PREFILL_TPS_TTFT_ESTIMATOR_NAME:
        return PREFILL_TPS_TTFT_ESTIMATOR_NAME, _wait_estimator_prefill_tps_ttft
    if key == SCORE_PROXY_TTFT_ESTIMATOR_NAME:
        return SCORE_PROXY_TTFT_ESTIMATOR_NAME, _wait_estimator_score_proxy_ttft
    if key == SCORE_TOTAL_LATENCY_ESTIMATOR_NAME:
        return SCORE_TOTAL_LATENCY_ESTIMATOR_NAME, _wait_estimator_score_total_latency
    if key == "pk_mg1":
        return "pk_mg1", _wait_estimator_pk_mg1
    if ":" not in spec:
        raise ValueError(
            "Custom wait estimator must use '<module_or_path>:<callable>' format."
        )
    return spec, _load_wait_estimator_callable(spec)


def _snapshot_log_offsets(paths: list[Path]) -> Dict[Path, int]:
    offsets: Dict[Path, int] = {}
    for path in paths:
        if path.exists():
            offsets[path] = path.stat().st_size
        else:
            offsets[path] = 0
    return offsets


def _read_latency_components_from_logs(
    paths: list[Path],
    *,
    offsets: Optional[Dict[Path, int]] = None,
    update_offsets: bool = False,
) -> dict[str, RequestLatencyComponents]:
    values: dict[str, RequestLatencyComponents] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as src:
            if offsets is not None:
                start = offsets.get(path, 0)
                if start > 0:
                    src.seek(start)
            for line in src:
                request_match = REQUEST_ID_PATTERN.search(line)
                if not request_match:
                    continue
                request_id = request_match.group(1)
                components = values.get(request_id)
                if components is None:
                    components = RequestLatencyComponents()
                    values[request_id] = components

                queue_match = QUEUE_PATTERN.search(line)
                if queue_match is not None:
                    try:
                        components.queue_ms = float(queue_match.group(1))
                    except ValueError:
                        LOGGER.warning(
                            "Skipping queue-log line with non-numeric queue_ms; "
                            "path=%s request_id=%s raw_queue_ms=%r",
                            path,
                            request_id,
                            queue_match.group(1),
                        )

                ttft_match = TTFT_PATTERN.search(line)
                if ttft_match is not None:
                    try:
                        components.frontend_ttft_ms = (
                            float(ttft_match.group(1)) * 1000.0
                        )
                    except ValueError:
                        LOGGER.warning(
                            "Skipping queue-log line with non-numeric ttft_s; "
                            "path=%s request_id=%s raw_ttft_s=%r",
                            path,
                            request_id,
                            ttft_match.group(1),
                        )

                prefill_match = PREFILL_PATTERN.search(line)
                if prefill_match is not None:
                    try:
                        components.prefill_ms = float(prefill_match.group(1)) * 1000.0
                    except ValueError:
                        LOGGER.warning(
                            "Skipping queue-log line with non-numeric prefill_s; "
                            "path=%s request_id=%s raw_prefill_s=%r",
                            path,
                            request_id,
                            prefill_match.group(1),
                        )

                queued_ts_match = QUEUED_TS_PATTERN.search(line)
                if queued_ts_match is not None:
                    try:
                        components.queued_ts_s = float(queued_ts_match.group(1))
                    except ValueError:
                        LOGGER.warning(
                            "Skipping queue-log line with non-numeric queued_ts_s; "
                            "path=%s request_id=%s raw_queued_ts_s=%r",
                            path,
                            request_id,
                            queued_ts_match.group(1),
                        )

                first_token_ts_match = FIRST_TOKEN_TS_PATTERN.search(line)
                if first_token_ts_match is not None:
                    try:
                        components.first_token_ts_s = float(
                            first_token_ts_match.group(1)
                        )
                    except ValueError:
                        LOGGER.warning(
                            "Skipping queue-log line with non-numeric "
                            "first_token_ts_s; path=%s request_id=%s "
                            "raw_first_token_ts_s=%r",
                            path,
                            request_id,
                            first_token_ts_match.group(1),
                        )
            if offsets is not None and update_offsets:
                offsets[path] = src.tell()
    return values


def _build_actual_wait_maps_from_logs(
    paths: list[Path],
    *,
    offsets: Optional[Dict[Path, int]] = None,
) -> tuple[dict[str, float], dict[str, float]]:
    components = _read_latency_components_from_logs(paths, offsets=offsets)
    queue_map: dict[str, float] = {}
    ttft_map: dict[str, float] = {}
    for request_id, component in components.items():
        if component.queue_ms is not None:
            queue_val = float(component.queue_ms)
            queue_map[request_id] = queue_val
        if (
            component.queued_ts_s is not None
            and component.first_token_ts_s is not None
        ):
            ttft_map[request_id] = float(
                (component.first_token_ts_s - component.queued_ts_s) * 1000.0
            )
        elif component.queue_ms is not None and component.prefill_ms is not None:
            ttft_map[request_id] = float(
                float(component.queue_ms) + float(component.prefill_ms)
            )
    return queue_map, ttft_map


def _read_queue_times_from_logs(
    paths: list[Path],
    *,
    offsets: Optional[Dict[Path, int]] = None,
) -> dict[str, float]:
    queue_map, _ = _build_actual_wait_maps_from_logs(paths, offsets=offsets)
    return queue_map


def _wait_gof_actual_metric_name(estimator_name: str) -> str:
    normalized = estimator_name.strip().lower()
    if normalized in TTFT_TARGET_ESTIMATORS:
        return "ttft_ms"
    return "queue_ms"


def _augment_with_actual_waits(
    per_request: list[Dict[str, Any]],
    actual_wait_map: dict[str, float],
    *,
    actual_metric_name: str,
) -> Dict[str, Any]:
    predicted: list[float] = []
    actual: list[float] = []
    matched = 0
    for item in per_request:
        response_id = item.get("response_id")
        actual_wait_ms = (
            actual_wait_map.get(str(response_id))
            if isinstance(response_id, str) and response_id
            else None
        )
        item["actual_wait_time_ms"] = actual_wait_ms
        item["actual_metric_name"] = actual_metric_name
        estimate = item.get("wait_time_ms")
        if actual_wait_ms is not None and isinstance(estimate, (int, float)):
            wait_error = float(estimate) - float(actual_wait_ms)
            item["wait_error_ms"] = wait_error
            predicted.append(float(estimate))
            actual.append(float(actual_wait_ms))
            matched += 1
        else:
            item["wait_error_ms"] = None
    return {
        "actual_metric_name": actual_metric_name,
        "matched_pairs": matched,
        "prediction_metrics_ms": _fit_error_metrics(actual, predicted),
    }

def run_batch_fit_experiment(
    *,
    csv_paths: list[Path],
    train_csv_paths: list[Path],
    test_csv_paths: list[Path],
    output_dir: Path,
    stall_percentile: float,
    huber_epsilon: float,
    max_plot_points: int,
    feature_set: str,
    nonnegative_coefficients: bool = False,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_train_paths = _resolve_batch_fit_paths(train_csv_paths)
    resolved_test_paths = _resolve_batch_fit_paths(test_csv_paths)
    if resolved_train_paths or resolved_test_paths:
        if csv_paths:
            raise ValueError(
                "Do not combine --batch-csv with held-out "
                "--batch-train-csv/--batch-test-csv."
            )
        if not resolved_train_paths or not resolved_test_paths:
            raise ValueError(
                "Held-out mode requires both --batch-train-csv and --batch-test-csv."
            )

        import pandas as pd

        train_frames = [_read_batch_fit_df(path) for path in resolved_train_paths]
        train_df = pd.concat(train_frames, ignore_index=True)
        fit_result = _fit_two_part_from_df(
            train_df,
            stall_percentile=stall_percentile,
            huber_epsilon=huber_epsilon,
            feature_set=feature_set,
            nonnegative_coefficients=nonnegative_coefficients,
        )

        coeff_path = output_dir / "heldout_two_part_coefficients.txt"
        coeff_path.write_text(
            _summarize_two_part_fit(fit_result, train_df),
            encoding="utf-8",
        )

        train_eval, _ = _evaluate_batch_fit_df(
            fit_result=fit_result,
            df=train_df,
            output_dir=output_dir,
            output_prefix="heldout_train",
            title_prefix="heldout_train",
            max_plot_points=max_plot_points,
        )

        test_files: list[Dict[str, Any]] = []
        agg_prefill: list[float] = []
        agg_decode: list[float] = []
        agg_actual: list[float] = []
        agg_typical: list[float] = []
        agg_expected: list[float] = []
        agg_inlier_mask: list[bool] = []

        for test_path in resolved_test_paths:
            test_df = _read_batch_fit_df(test_path)
            test_eval, series = _evaluate_batch_fit_df(
                fit_result=fit_result,
                df=test_df,
                output_dir=output_dir,
                output_prefix=f"heldout_test_{test_path.stem}",
                title_prefix=f"heldout_test_{test_path.stem}",
                max_plot_points=max_plot_points,
            )
            test_files.append(
                {
                    "csv_path": str(test_path),
                    **test_eval,
                }
            )
            agg_prefill.extend(series["prefill"])
            agg_decode.extend(series["decode"])
            agg_actual.extend(series["actual_exec"])
            agg_typical.extend(series["typical_exec"])
            agg_expected.extend(series["expected_exec"])
            agg_inlier_mask.extend(series["inlier_mask"])

        aggregate_typical_surface_path = _plot_batch_fit_3d(
            prefill=agg_prefill,
            decode=agg_decode,
            actual_exec=agg_actual,
            estimated_exec=agg_typical,
            title="heldout_test_aggregate: actual vs typical estimate",
            output_path=output_dir / "heldout_test_aggregate_actual_vs_typical_3d.png",
            max_points=max_plot_points,
        )
        aggregate_expected_surface_path = _plot_batch_fit_3d(
            prefill=agg_prefill,
            decode=agg_decode,
            actual_exec=agg_actual,
            estimated_exec=agg_expected,
            title="heldout_test_aggregate: actual vs expected estimate",
            output_path=output_dir / "heldout_test_aggregate_actual_vs_expected_3d.png",
            max_points=max_plot_points,
        )
        aggregate_expected_parity_path = _plot_batch_fit_parity(
            actual_exec=agg_actual,
            estimated_exec=agg_expected,
            title="heldout_test_aggregate: expected estimate parity",
            output_path=output_dir / "heldout_test_aggregate_expected_parity.png",
            max_points=max_plot_points,
        )
        aggregate_typical_parity_path = _plot_batch_fit_parity(
            actual_exec=agg_actual,
            estimated_exec=agg_typical,
            title="heldout_test_aggregate: typical estimate parity",
            output_path=output_dir / "heldout_test_aggregate_typical_parity.png",
            max_points=max_plot_points,
        )
        aggregate_predictions_path = _write_batch_fit_predictions(
            prefill=agg_prefill,
            decode=agg_decode,
            actual_exec=agg_actual,
            typical_exec=agg_typical,
            expected_exec=agg_expected,
            output_path=output_dir / "heldout_test_aggregate_predictions.csv",
        )

        aggregate_typical_metrics = _add_inlier_r2(
            metrics=_fit_error_metrics(agg_actual, agg_typical),
            actual=agg_actual,
            predicted=agg_typical,
            inlier_mask=agg_inlier_mask,
        )
        aggregate_expected_metrics = _add_inlier_r2(
            metrics=_fit_error_metrics(agg_actual, agg_expected),
            actual=agg_actual,
            predicted=agg_expected,
            inlier_mask=agg_inlier_mask,
        )

        return {
            "mode": "batch_fit",
            "evaluation": "heldout",
            "output_dir": str(output_dir),
            "fit_params": {
                "stall_percentile": stall_percentile,
                "huber_epsilon": huber_epsilon,
                "feature_set": feature_set,
                "nonnegative_coefficients": nonnegative_coefficients,
            },
            "train_csv_paths": [str(path) for path in resolved_train_paths],
            "test_csv_paths": [str(path) for path in resolved_test_paths],
            "model": {
                "stall_probability": fit_result.stall_probability,
                "mean_stall_delay": fit_result.mean_stall_delay,
                "residual_threshold": fit_result.residual_threshold,
                "coefficients_path": str(coeff_path),
            },
            "train": train_eval,
            "test": {
                "aggregate": {
                    "num_rows": len(agg_actual),
                    "goodness_of_fit": {
                        "typical": aggregate_typical_metrics,
                        "expected": aggregate_expected_metrics,
                    },
                    "artifacts": {
                        "actual_vs_typical_3d": aggregate_typical_surface_path,
                        "actual_vs_expected_3d": aggregate_expected_surface_path,
                        "typical_parity": aggregate_typical_parity_path,
                        "expected_parity": aggregate_expected_parity_path,
                        "predictions_csv": aggregate_predictions_path,
                    },
                },
                "files": test_files,
            },
        }

    resolved_csv_paths = csv_paths or sorted(
        DEFAULT_BATCH_FIT_SEARCH_DIR.glob(DEFAULT_BATCH_FIT_GLOB)
    )
    resolved_csv_paths = _resolve_batch_fit_paths(resolved_csv_paths)
    if not resolved_csv_paths:
        raise ValueError(
            "No CSV inputs found. Provide --batch-csv or ensure files match "
            f"{DEFAULT_BATCH_FIT_GLOB} under {DEFAULT_BATCH_FIT_SEARCH_DIR}."
        )

    files: list[Dict[str, Any]] = []
    for csv_path in resolved_csv_paths:
        result, df = _fit_two_part(
            csv_path,
            stall_percentile=stall_percentile,
            huber_epsilon=huber_epsilon,
            feature_set=feature_set,
            nonnegative_coefficients=nonnegative_coefficients,
        )

        coeff_path = output_dir / f"{csv_path.stem}_two_part_coefficients.txt"
        coeff_path.write_text(_summarize_two_part_fit(result, df), encoding="utf-8")

        eval_summary, _ = _evaluate_batch_fit_df(
            fit_result=result,
            df=df,
            output_dir=output_dir,
            output_prefix=csv_path.stem,
            title_prefix=csv_path.stem,
            max_plot_points=max_plot_points,
        )

        files.append(
            {
                "csv_path": str(csv_path),
                "stall_percentile": stall_percentile,
                "huber_epsilon": huber_epsilon,
                "stall_probability": result.stall_probability,
                "mean_stall_delay": result.mean_stall_delay,
                "residual_threshold": result.residual_threshold,
                "goodness_of_fit": eval_summary["goodness_of_fit"],
                "artifacts": {
                    "coefficients_path": str(coeff_path),
                    **eval_summary["artifacts"],
                },
            }
        )

    return {
        "mode": "batch_fit",
        "evaluation": "in_sample",
        "num_files": len(files),
        "output_dir": str(output_dir),
        "fit_params": {
            "stall_percentile": stall_percentile,
            "huber_epsilon": huber_epsilon,
            "feature_set": feature_set,
            "nonnegative_coefficients": nonnegative_coefficients,
        },
        "files": files,
    }


def _load_remaining_length_tables(
    remaining_length: Optional[Dict[str, Any]],
    instances: dict[str, InstanceClient],
) -> Optional[Dict[str, Any]]:
    """Attach the engines' remaining-length tables (instances.json block) per instance."""
    if not remaining_length or remaining_length.get("rule", "current") == "current":
        return None
    from vllm.v1.core.sched.remaining_length import RemainingLengthTable

    tables: Dict[str, Any] = {}
    for instance_id, client in instances.items():
        rule = remaining_length.get("models", {}).get(client.model_id) or {"mode": "off"}
        if rule["mode"] == "off":
            continue
        entry = rule.get("table")
        if entry is None:
            raise ValueError(f"remaining_length table missing for model {client.model_id}")
        table = RemainingLengthTable.load(entry["path"])
        if table.sha256 != entry.get("sha256"):
            raise ValueError(f"remaining_length table changed for model {client.model_id}")
        tables[instance_id] = table
    return tables or None


async def run_policy(
    *,
    utility_name: str,
    requests: list[ExperimentRequest],
    instances: dict[str, InstanceClient],
    instance_costs: Dict[str, Dict[str, float]],
    accuracy_model_path: Optional[str],
    output_length_model_path: Optional[str],
    lambda_weight: float,
    delta_weight: float,
    worker_count: int,
    max_queue_size: int,
    request_rate_qps: float,
    arrival_process: str,
    arrival_seed: int,
    max_completion_tokens: int,
    temperature: float,
    top_p: float,
    tokenizer_id: str = DEFAULT_TOKENIZER_ID,
    tokenizer_mode: str = "auto",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    chat_template_kwargs: dict[str, Any] | None = None,
    response_map_path: Optional[str],
    request_log_path: Optional[str],
    mmpp2_rate_ratio: float = DEFAULT_MMPP2_RATE_RATIO,
    mmpp2_high_fraction: float = DEFAULT_MMPP2_HIGH_FRACTION,
    mmpp2_correlation_time_s: float = DEFAULT_MMPP2_CORRELATION_TIME_S,
    request_id_prefix: Optional[str] = None,
    route_strategy: str = "utility",
    wait_estimator: Optional[WaitEstimatorCallable] = None,
    wait_estimator_name: str = "live",
    wait_estimator_context: Optional[Dict[str, Any]] = None,
    wait_estimators: Optional[list[tuple[str, WaitEstimatorCallable]]] = None,
    wait_estimator_contexts: Optional[Dict[str, Dict[str, Any]]] = None,
    run_label: Optional[str] = None,
    feasible_slo_mode: str = "queue",
    per_request_wait_logs: Optional[list[Path]] = None,
    wait_log_offsets: Optional[Dict[Path, int]] = None,
    skip_wait_result_build: bool = False,
    include_unconditional_live_fetch: bool = True,
    readiness_diagnostics: bool = False,
    readiness_predictor_path: Optional[str] = None,
    enable_wait_time_polling: bool = True,
    critical_wait_time_timeout_s: float = 0.01,
    route_random_seed: Optional[int] = None,
    affinity_bucket_to_instances: Optional[Dict[str, list[str]]] = None,
    affinity_global_fallback_instances: Optional[list[str]] = None,
    affinity_upgrade_margin: float = DEFAULT_AFFINITY_UPGRADE_MARGIN,
    score_cost_weight: float = 1.0,
    score_latency_weight: float = 1.0,
    score_total_cost_budget: Optional[float] = None,
    close_instances_on_stop: bool = True,
    decouple_arrivals: bool = True,
    methodology_calibration_path: Optional[str] = None,
    methodology_serving_profile: Optional[Dict[str, Any]] = None,
    routebalance_predictor_path: Optional[str] = None,
    routebalance_weights: tuple[float, float, float] = (1/3, 1/3, 1/3),
    routebalance_batch_max_size: int = 16,
    routebalance_batch_wait_ms: float = 25.0,
    methodology_snapshot_max_age_ms: float = 1000.0,
    trial_monitor: Any = None,
    latency_warmup_requests: Optional[str] = None,
    remaining_length: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    resolved_chat_template_kwargs = resolve_chat_template_kwargs(
        chat_template_kwargs
    )
    remaining_length_tables = _load_remaining_length_tables(
        remaining_length, instances
    )
    resolved_feasible_slo_mode = feasible_slo_mode.strip().lower()
    if resolved_feasible_slo_mode not in FEASIBLE_SLO_MODES:
        raise ValueError(
            f"Unsupported feasible_slo_mode '{feasible_slo_mode}'. "
            f"Choose from: {', '.join(FEASIBLE_SLO_MODES)}."
        )

    score_policy_state = (
        ScorePolicyState(
            total_requests=len(requests),
            total_cost_budget=score_total_cost_budget,
        )
        if utility_name.strip().lower() == "score"
        else None
    )
    utility_state = UtilityState(score_policy_state=score_policy_state)
    is_latency_history = utility_name == "vllm_sr_latency"
    is_methodology = utility_name in {"lmdeploy_proxy", "mooncake_prefill", "routebalance"}
    utility_fn = build_utility_fn(
        "hard" if (is_methodology or is_latency_history) else utility_name,
        lambda_weight=lambda_weight,
        delta_weight=delta_weight,
        instance_costs=instance_costs,
        utility_state=utility_state,
        score_cost_weight=score_cost_weight,
        score_latency_weight=score_latency_weight,
    )
    scheduler_kwargs = dict(
        defer_sidecar_writes=True,
        worker_count=worker_count,
        max_queue_size=max_queue_size,
        instance_costs=instance_costs,
        tokenizer_id=tokenizer_id,
        tokenizer_mode=tokenizer_mode,
        response_map_path=response_map_path,
        request_log_path=request_log_path,
    )
    if is_latency_history:
        from sfs_core.routing.latency_history_scheduler import LatencyHistoryScheduler
        from sfs_core.routing.latency_warmup import load_warmup
        scheduler = LatencyHistoryScheduler(instances, **scheduler_kwargs)
        warmup_rows = load_warmup(latency_warmup_requests)
        await scheduler.warmup([{
            "messages": build_messages(prompt=row["prompt"], system_prompt=system_prompt),
            "max_completion_tokens": max_completion_tokens, "temperature": temperature, "top_p": top_p,
            "extra_body": {"chat_template_kwargs": dict(resolved_chat_template_kwargs)},
        } for row in warmup_rows])
    elif is_methodology:
        from sfs_core.routing.methodology_calibration import MethodologyCalibration
        from sfs_core.routing.methodology_policies import RouteBalanceWeights
        from sfs_core.routing.methodology_scheduler import MethodologyScheduler
        if methodology_calibration_path is None:
            raise ValueError(f"{utility_name} requires --methodology-calibration-json")
        calibration = MethodologyCalibration.load(methodology_calibration_path)
        if methodology_serving_profile is not None:
            calibration.validate_runtime_profile(methodology_serving_profile)
        predictor = None
        if utility_name == "routebalance":
            from sfs_core.routing.routebalance_predictor import RouteBalancePredictor
            if routebalance_predictor_path is None:
                raise ValueError("RouteBalance requires --routebalance-predictor-path")
            predictor = RouteBalancePredictor.load(routebalance_predictor_path)
        scheduler = MethodologyScheduler(
            instances, policy=utility_name, calibration=calibration, predictor=predictor,
            weights=RouteBalanceWeights(*routebalance_weights),
            batch_max_size=routebalance_batch_max_size,
            batch_wait_ms=routebalance_batch_wait_ms,
            snapshot_max_age_ms=methodology_snapshot_max_age_ms,
            route_random_seed=69 if route_random_seed is None else route_random_seed,
            **scheduler_kwargs,
        )
    else:
        scheduler = CollectingWaitTimeScheduler(
            instances,
            defer_sidecar_writes=True,
            worker_count=worker_count,
            max_queue_size=max_queue_size,
            accuracy_model_path=accuracy_model_path,
            output_length_model_path=output_length_model_path,
            lambda_weight=lambda_weight,
            delta_weight=delta_weight,
            instance_costs=instance_costs,
            utility_fn=utility_fn,
            tokenizer_id=tokenizer_id,
            tokenizer_mode=tokenizer_mode,
            response_map_path=response_map_path,
            request_log_path=request_log_path,
            utility_state=utility_state,
            route_strategy=route_strategy,
            route_random_seed=route_random_seed,
            affinity_bucket_to_instances=affinity_bucket_to_instances,
            affinity_global_fallback_instances=affinity_global_fallback_instances,
            affinity_upgrade_margin=affinity_upgrade_margin,
            score_cost_weight=score_cost_weight,
            score_latency_weight=score_latency_weight,
            wait_estimator=wait_estimator,
            wait_estimator_name=wait_estimator_name,
            wait_estimator_context=wait_estimator_context,
            wait_estimators=wait_estimators,
            wait_estimator_contexts=wait_estimator_contexts,
            skip_wait_result_build=skip_wait_result_build,
            include_unconditional_live_fetch=include_unconditional_live_fetch,
            readiness_diagnostics=readiness_diagnostics,
            readiness_predictor_path=readiness_predictor_path,
            enable_wait_time_polling=enable_wait_time_polling,
            critical_wait_time_timeout_s=critical_wait_time_timeout_s,
            remaining_length_tables=remaining_length_tables,
        )

    loop = asyncio.get_running_loop()
    arrival_rng = random.Random(arrival_seed)
    resolved_arrival_process = str(arrival_process).strip().lower()
    mmpp2_params: Optional[MMPP2DerivedParams] = None
    mmpp2_state: Optional[Dict[str, Any]] = None
    if resolved_arrival_process == "mmpp2":
        mmpp2_params = _derive_mmpp2_params(
            request_rate_qps=float(request_rate_qps),
            rate_ratio=float(mmpp2_rate_ratio),
            high_fraction=float(mmpp2_high_fraction),
            correlation_time_s=float(mmpp2_correlation_time_s),
        )
        mmpp2_state = {"current_state": None}
    elif resolved_arrival_process not in ARRIVAL_PROCESSES:
        raise ValueError(
            f"Unsupported arrival_process '{arrival_process}'. "
            f"Supported: {', '.join(ARRIVAL_PROCESSES)}."
        )
    completion_futures: list[asyncio.Future] = []
    if not is_latency_history:
        await warm_up_instances(list(instances.values()))
    await scheduler.start()
    run_start_perf = time.perf_counter()
    if trial_monitor is not None:
        await trial_monitor.start(run_start_perf, scheduler, instances)

    def _selected_slo_ms(req: ExperimentRequest) -> float:
        if resolved_feasible_slo_mode == "queue":
            return float(req.queue_slo_ms)
        if resolved_feasible_slo_mode == "ttft":
            return float(req.ttft_slo_ms)
        return float(req.latency_slo_ms)

    def _scheduler_request_id(req: ExperimentRequest) -> str:
        if request_id_prefix:
            return f"{request_id_prefix}-{req.request_id}"
        return f"{utility_name}-{req.request_id}"

    async def _route_single_request(
        req: ExperimentRequest,
        *,
        completion_future: asyncio.Future,
        system_entry_perf: float,
        started_perf: Optional[float] = None,
    ) -> None:
        started_perf_value = (
            float(started_perf) if started_perf is not None else time.perf_counter()
        )
        payload: Dict[str, Any] = {
            "messages": build_messages(
                prompt=req.prompt,
                system_prompt=system_prompt,
            ),
            "max_completion_tokens": max_completion_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "_completion_future": completion_future,
            "_system_entry_perf": float(system_entry_perf),
            "_started_perf": started_perf_value,
            "_request_slo_ms": _selected_slo_ms(req),
            "_request_bucket": req.bucket,
            "_prompt_tokens": int(req.prompt_tokens),
        }
        if score_policy_state is not None:
            payload["_score_latency_limit_ms"] = float(req.latency_slo_ms)
        payload["extra_body"] = {
            "chat_template_kwargs": dict(resolved_chat_template_kwargs)
        }
        await scheduler.route_and_submit(
            request_id=_scheduler_request_id(req),
            **payload,
        )

    arrival_schedule = _ArrivalSchedule()

    async def _sleep_for_interarrival(idx: int) -> None:
        if idx <= 0:
            arrival_schedule.start()
            return
        interarrival_s = _sample_interarrival_s(
            request_rate_qps=request_rate_qps,
            arrival_process=resolved_arrival_process,
            rng=arrival_rng,
            mmpp2_params=mmpp2_params,
            mmpp2_state=mmpp2_state,
        )
        await arrival_schedule.wait(interarrival_s)

    try:
        if decouple_arrivals:
            ingress_queue: asyncio.Queue[
                Optional[tuple[ExperimentRequest, asyncio.Future, float]]
            ] = asyncio.Queue()

            def _admit(req: ExperimentRequest, system_entry_perf: float) -> None:
                # Loop thread: futures, the monitor and the queue are loop-owned.
                completion_future: asyncio.Future = loop.create_future()
                completion_futures.append(completion_future)
                if trial_monitor is not None:
                    trial_monitor.arrived(req, completion_future, system_entry_perf)
                ingress_queue.put_nowait((req, completion_future, system_entry_perf))

            def _finish_arrivals() -> None:
                if trial_monitor is not None:
                    trial_monitor.end_arrivals()
                ingress_queue.put_nowait(None)

            def _producer() -> None:
                # Runs on its own thread: the offered load and the system-entry
                # stamp are independent of event-loop load (thousands of SSE
                # streams saturate the loop; a starved timer under-delivers).
                # Same RNG object and draw order as the loop-side generator.
                for idx, req in enumerate(requests):
                    if idx <= 0:
                        arrival_schedule.start()
                    else:
                        arrival_schedule.wait_blocking(_sample_interarrival_s(
                            request_rate_qps=request_rate_qps,
                            arrival_process=resolved_arrival_process,
                            rng=arrival_rng,
                            mmpp2_params=mmpp2_params,
                            mmpp2_state=mmpp2_state,
                        ))
                    if trial_monitor is not None and trial_monitor.stop_reason():
                        break
                    loop.call_soon_threadsafe(_admit, req, time.perf_counter())
                loop.call_soon_threadsafe(_finish_arrivals)

            async def _submitter() -> None:
                while True:
                    ingress_item = await ingress_queue.get()
                    if ingress_item is None:
                        break
                    req, completion_future, system_entry_perf = ingress_item
                    await _route_single_request(
                        req,
                        completion_future=completion_future,
                        system_entry_perf=system_entry_perf,
                    )

            await asyncio.gather(asyncio.to_thread(_producer), _submitter())
        else:
            for idx, req in enumerate(requests):
                await _sleep_for_interarrival(idx)
                if trial_monitor is not None and trial_monitor.stop_reason():
                    break
                completion_future: asyncio.Future = loop.create_future()
                completion_futures.append(completion_future)
                system_entry_perf = time.perf_counter()
                if trial_monitor is not None:
                    trial_monitor.arrived(req, completion_future, system_entry_perf)
                await _route_single_request(
                    req,
                    completion_future=completion_future,
                    system_entry_perf=system_entry_perf,
                    started_perf=system_entry_perf,
                )
            if trial_monitor is not None:
                trial_monitor.end_arrivals()

        if is_methodology:
            scheduler.finish_arrivals()
        raw_results = await asyncio.gather(*completion_futures)
        await scheduler.drain()
        await scheduler.flush_sidecar_logs()
    finally:
        await scheduler.stop(close_instances=close_instances_on_stop)
        if trial_monitor is not None:
            await trial_monitor.close()

    response_latency_components: dict[str, RequestLatencyComponents] = {}
    if per_request_wait_logs:
        response_latency_components = _read_latency_components_from_logs(
            per_request_wait_logs,
            offsets=wait_log_offsets,
            update_offsets=wait_log_offsets is not None,
        )

    run_end_perf = max(
        [run_start_perf]
        + [float(item.get("completed_perf", run_start_perf)) for item in raw_results]
    )
    elapsed_s = max(run_end_perf - run_start_perf, 1e-9)

    per_request: list[Dict[str, Any]] = []
    instance_counts: Dict[str, int] = {}
    successes = 0
    e2e_slo_hits = 0
    latencies: list[float] = []
    predicted_accuracies: list[float] = []
    e2e_slo_violations_ms: list[float] = []
    costs: list[float] = []
    cost_total = 0.0
    cost_computable_requests = 0
    cost_source_counts: Dict[str, int] = {}
    actual_costs: list[float] = []
    actual_cost_total = 0.0
    actual_cost_computable_requests = 0
    actual_cost_source_counts: Dict[str, int] = {}
    predicted_costs: list[float] = []
    predicted_cost_total = 0.0
    predicted_cost_computable_requests = 0
    predicted_cost_source_counts: Dict[str, int] = {}
    arrival_to_dispatch_values: list[float] = []
    system_entry_to_dispatch_values: list[float] = []
    queue_slo_hits = 0
    queue_slo_violations_ms: list[float] = []
    queue_delay_values: list[float] = []
    queue_available = 0
    ttft_slo_hits = 0
    ttft_slo_violations_ms: list[float] = []
    ttft_values: list[float] = []
    ttft_available = 0
    e2e_ttft_slo_hits = 0
    e2e_ttft_slo_violations_ms: list[float] = []
    e2e_ttft_values: list[float] = []
    e2e_ttft_available = 0
    system_entry_e2e_ttft_slo_hits = 0
    system_entry_e2e_ttft_slo_violations_ms: list[float] = []
    system_entry_e2e_ttft_values: list[float] = []
    system_entry_e2e_ttft_available = 0

    for req, item in zip(requests, raw_results):
        latency_ms = item.get("latency_ms")
        error = item.get("error")
        instance_id = item.get("instance_id")
        predicted_accuracy = item.get("predicted_accuracy")
        usage_prompt_tokens = _as_nonnegative_int(item.get("usage_prompt_tokens"))
        usage_completion_tokens = _as_nonnegative_int(item.get("usage_completion_tokens"))
        usage_total_tokens = _as_nonnegative_int(item.get("usage_total_tokens"))
        cost_prompt_tokens = (
            int(usage_prompt_tokens)
            if usage_prompt_tokens is not None
            else int(req.prompt_tokens)
        )
        predicted_output_tokens = _as_nonnegative_float(item.get("predicted_output_tokens"))

        response_id = item.get("response_id")
        latency_components = (
            response_latency_components.get(str(response_id))
            if isinstance(response_id, str) and response_id
            else None
        )
        queue_delay_ms = (
            float(latency_components.queue_ms)
            if latency_components is not None and latency_components.queue_ms is not None
            else None
        )
        prefill_ms = (
            float(latency_components.prefill_ms)
            if latency_components is not None and latency_components.prefill_ms is not None
            else None
        )
        ttft_ms = (
            float(
                (
                    latency_components.first_token_ts_s
                    - latency_components.queued_ts_s
                )
                * 1000.0
            )
            if latency_components is not None
            and latency_components.queued_ts_s is not None
            and latency_components.first_token_ts_s is not None
            else (
                float(queue_delay_ms + prefill_ms)
                if queue_delay_ms is not None and prefill_ms is not None
                else None
            )
        )
        frontend_ttft_ms = (
            float(latency_components.frontend_ttft_ms)
            if latency_components is not None
            and latency_components.frontend_ttft_ms is not None
            else None
        )

        system_entry_perf = _as_nonnegative_float(item.get("system_entry_perf"))
        started_perf = _as_nonnegative_float(item.get("started_perf"))
        dispatch_perf = _as_nonnegative_float(item.get("dispatch_perf"))
        arrival_to_dispatch_ms = (
            float((dispatch_perf - started_perf) * 1000.0)
            if started_perf is not None
            and dispatch_perf is not None
            and dispatch_perf >= started_perf
            else None
        )
        system_entry_to_dispatch_ms = (
            float((dispatch_perf - system_entry_perf) * 1000.0)
            if system_entry_perf is not None
            and dispatch_perf is not None
            and dispatch_perf >= system_entry_perf
            else None
        )
        e2e_ttft_ms = (
            float(arrival_to_dispatch_ms + ttft_ms)
            if arrival_to_dispatch_ms is not None and ttft_ms is not None
            else None
        )
        system_entry_e2e_ttft_ms = (
            float(system_entry_to_dispatch_ms + ttft_ms)
            if system_entry_to_dispatch_ms is not None and ttft_ms is not None
            else None
        )

        actual_cost_completion_tokens: Optional[float] = None
        actual_cost_source = "unavailable"
        if usage_completion_tokens is not None:
            actual_cost_completion_tokens = float(usage_completion_tokens)
            actual_cost_source = "actual_usage"
        elif usage_total_tokens is not None:
            derived_completion_tokens = max(int(usage_total_tokens) - cost_prompt_tokens, 0)
            actual_cost_completion_tokens = float(derived_completion_tokens)
            actual_cost_source = "derived_from_total_tokens"

        predicted_cost_completion_tokens: Optional[float] = None
        predicted_cost_source = "unavailable"
        if predicted_output_tokens is not None:
            predicted_cost_completion_tokens = float(predicted_output_tokens)
            predicted_cost_source = "predicted_completion_fallback"

        actual_cost: Optional[float] = None
        predicted_cost: Optional[float] = None
        if instance_id is not None:
            cost_info = instance_costs.get(str(instance_id)) or {}
            prompt_rate = float(cost_info.get("prompt", 0.0))
            output_rate = float(cost_info.get("output", 0.0))
            if actual_cost_completion_tokens is not None:
                actual_cost = (prompt_rate * float(cost_prompt_tokens)) + (
                    output_rate * float(actual_cost_completion_tokens)
                )
                actual_costs.append(float(actual_cost))
                actual_cost_total += float(actual_cost)
                actual_cost_computable_requests += 1
            if predicted_cost_completion_tokens is not None:
                predicted_cost = (prompt_rate * float(cost_prompt_tokens)) + (
                    output_rate * float(predicted_cost_completion_tokens)
                )
                predicted_costs.append(float(predicted_cost))
                predicted_cost_total += float(predicted_cost)
                predicted_cost_computable_requests += 1

        actual_cost_source_counts[actual_cost_source] = (
            actual_cost_source_counts.get(actual_cost_source, 0) + 1
        )
        predicted_cost_source_counts[predicted_cost_source] = (
            predicted_cost_source_counts.get(predicted_cost_source, 0) + 1
        )

        cost_completion_tokens = actual_cost_completion_tokens
        cost = actual_cost
        cost_source = actual_cost_source
        if cost is None and predicted_cost is not None:
            cost_completion_tokens = predicted_cost_completion_tokens
            cost = predicted_cost
            cost_source = predicted_cost_source
        if cost is not None:
            costs.append(float(cost))
            cost_total += float(cost)
            cost_computable_requests += 1
        cost_source_counts[cost_source] = cost_source_counts.get(cost_source, 0) + 1

        if instance_id:
            instance_counts[instance_id] = instance_counts.get(instance_id, 0) + 1
        if latency_ms is not None:
            latencies.append(float(latency_ms))
        if arrival_to_dispatch_ms is not None:
            arrival_to_dispatch_values.append(float(arrival_to_dispatch_ms))
        if system_entry_to_dispatch_ms is not None:
            system_entry_to_dispatch_values.append(float(system_entry_to_dispatch_ms))
        if predicted_accuracy is not None and error is None:
            predicted_accuracies.append(float(predicted_accuracy))

        e2e_violation_ms = (
            max(float(latency_ms) - float(req.latency_slo_ms), 0.0)
            if latency_ms is not None
            else None
        )
        if e2e_violation_ms is not None:
            e2e_slo_violations_ms.append(e2e_violation_ms)
        e2e_slo_met = bool(
            error is None
            and latency_ms is not None
            and float(latency_ms) <= float(req.latency_slo_ms)
        )
        if e2e_slo_met:
            e2e_slo_hits += 1

        if queue_delay_ms is not None:
            queue_available += 1
            queue_delay_values.append(float(queue_delay_ms))
        queue_slo_violation_ms = (
            max(float(queue_delay_ms) - float(req.queue_slo_ms), 0.0)
            if queue_delay_ms is not None
            else None
        )
        if queue_slo_violation_ms is not None:
            queue_slo_violations_ms.append(queue_slo_violation_ms)
        queue_slo_met = bool(
            error is None
            and queue_delay_ms is not None
            and float(queue_delay_ms) <= float(req.queue_slo_ms)
        )
        if queue_slo_met:
            queue_slo_hits += 1

        if ttft_ms is not None:
            ttft_available += 1
            ttft_values.append(float(ttft_ms))
        ttft_slo_violation_ms = (
            max(float(ttft_ms) - float(req.ttft_slo_ms), 0.0)
            if ttft_ms is not None
            else None
        )
        if ttft_slo_violation_ms is not None:
            ttft_slo_violations_ms.append(ttft_slo_violation_ms)
        ttft_slo_met = bool(
            error is None
            and ttft_ms is not None
            and float(ttft_ms) <= float(req.ttft_slo_ms)
        )
        if ttft_slo_met:
            ttft_slo_hits += 1

        if e2e_ttft_ms is not None:
            e2e_ttft_available += 1
            e2e_ttft_values.append(float(e2e_ttft_ms))
        e2e_ttft_slo_violation_ms = (
            max(float(e2e_ttft_ms) - float(req.ttft_slo_ms), 0.0)
            if e2e_ttft_ms is not None
            else None
        )
        if e2e_ttft_slo_violation_ms is not None:
            e2e_ttft_slo_violations_ms.append(e2e_ttft_slo_violation_ms)
        e2e_ttft_slo_met = bool(
            error is None
            and e2e_ttft_ms is not None
            and float(e2e_ttft_ms) <= float(req.ttft_slo_ms)
        )
        if e2e_ttft_slo_met:
            e2e_ttft_slo_hits += 1

        if system_entry_e2e_ttft_ms is not None:
            system_entry_e2e_ttft_available += 1
            system_entry_e2e_ttft_values.append(float(system_entry_e2e_ttft_ms))
        system_entry_e2e_ttft_slo_violation_ms = (
            max(float(system_entry_e2e_ttft_ms) - float(req.ttft_slo_ms), 0.0)
            if system_entry_e2e_ttft_ms is not None
            else None
        )
        if system_entry_e2e_ttft_slo_violation_ms is not None:
            system_entry_e2e_ttft_slo_violations_ms.append(
                system_entry_e2e_ttft_slo_violation_ms
            )
        system_entry_e2e_ttft_slo_met = bool(
            error is None
            and system_entry_e2e_ttft_ms is not None
            and float(system_entry_e2e_ttft_ms) <= float(req.ttft_slo_ms)
        )
        if system_entry_e2e_ttft_slo_met:
            system_entry_e2e_ttft_slo_hits += 1

        if error is None:
            successes += 1

        record: Dict[str, Any] = {
            "request_id": req.request_id,
            "bucket": req.bucket,
            "prompt_tokens": req.prompt_tokens,
            "latency_slo_ms": req.latency_slo_ms,
            "queue_slo_ms": req.queue_slo_ms,
            "ttft_slo_ms": req.ttft_slo_ms,
            "latency_ms": latency_ms,
            "system_entry_offset_s": (
                system_entry_perf - run_start_perf if system_entry_perf is not None else None
            ),
            "arrival_to_dispatch_ms": arrival_to_dispatch_ms,
            "system_entry_to_dispatch_ms": system_entry_to_dispatch_ms,
            "queue_delay_ms": queue_delay_ms,
            "ttft_ms": ttft_ms,
            "frontend_ttft_ms": frontend_ttft_ms,
            "queued_ts_s": (
                latency_components.queued_ts_s
                if latency_components is not None
                else None
            ),
            "first_token_ts_s": (
                latency_components.first_token_ts_s
                if latency_components is not None
                else None
            ),
            "e2e_ttft_ms": e2e_ttft_ms,
            "system_entry_e2e_ttft_ms": system_entry_e2e_ttft_ms,
            "prefill_ms": prefill_ms,
            "slo_violation_ms": e2e_violation_ms,
            "slo_met": e2e_slo_met,
            "queue_slo_violation_ms": queue_slo_violation_ms,
            "queue_slo_met": queue_slo_met,
            "ttft_slo_violation_ms": ttft_slo_violation_ms,
            "ttft_slo_met": ttft_slo_met,
            "e2e_ttft_slo_violation_ms": e2e_ttft_slo_violation_ms,
            "e2e_ttft_slo_met": e2e_ttft_slo_met,
            "system_entry_e2e_ttft_slo_violation_ms": system_entry_e2e_ttft_slo_violation_ms,
            "system_entry_e2e_ttft_slo_met": system_entry_e2e_ttft_slo_met,
            "instance_id": instance_id,
            "wait_time_ms": item.get("wait_time_ms"),
            "live_wait_time_ms": item.get("live_wait_time_ms"),
            "wait_estimates_ms": item.get("wait_estimates_ms"),
            "score_candidate_terms": item.get("score_candidate_terms"),
            "score_policy_terms": item.get("score_policy_terms"),
            "methodology_terms": item.get("methodology_terms"),
            "wait_estimator": item.get("wait_estimator", wait_estimator_name),
            "route_strategy": item.get("route_strategy", route_strategy),
            "feasible_slo_mode": resolved_feasible_slo_mode,
            "wait_time_metadata": item.get("wait_time_metadata"),
            "shortest_queue_routing": item.get("shortest_queue_routing"),
            "affinity_routing": item.get("affinity_routing"),
            "request_bucket": item.get("request_bucket", req.bucket),
            "selected_effective_num_requests": item.get("selected_effective_num_requests"),
            "selected_num_requests_source": item.get("selected_num_requests_source"),
            "predicted_accuracy": predicted_accuracy,
            "predicted_output_tokens": item.get("predicted_output_tokens"),
            "usage_prompt_tokens": usage_prompt_tokens,
            "usage_completion_tokens": usage_completion_tokens,
            "usage_total_tokens": usage_total_tokens,
            "cost_prompt_tokens": int(cost_prompt_tokens),
            "actual_cost_completion_tokens": actual_cost_completion_tokens,
            "predicted_cost_completion_tokens": predicted_cost_completion_tokens,
            "cost_completion_tokens": cost_completion_tokens,
            "actual_cost": actual_cost,
            "predicted_cost": predicted_cost,
            "cost": cost,
            "actual_cost_source": actual_cost_source,
            "predicted_cost_source": predicted_cost_source,
            "cost_source": cost_source,
            "scheduler_request_id": item.get("request_id"),
            "response_id": item.get("response_id"),
            "response_model": item.get("response_model"),
            "error": error,
        }
        per_request.append(record)

    total = len(per_request)
    failures = total - successes
    throughput_qps_all = total / elapsed_s
    throughput_qps_success = successes / elapsed_s
    queue_missing_count = max(total - queue_available, 0)
    ttft_missing_count = max(total - ttft_available, 0)
    e2e_ttft_missing_count = max(total - e2e_ttft_available, 0)
    system_entry_e2e_ttft_missing_count = max(total - system_entry_e2e_ttft_available, 0)

    run_result: Dict[str, Any] = {
        "label": run_label or utility_name,
        "utility": utility_name,
        "arrival_process": resolved_arrival_process,
        "arrival_timing": ARRIVAL_TIMING_THREAD if decouple_arrivals else ARRIVAL_TIMING,
        "route_strategy": route_strategy,
        "wait_estimator": wait_estimator_name,
        "feasible_slo_mode": resolved_feasible_slo_mode,
        "response_map_path": response_map_path,
        "request_log_path": request_log_path,
        "summary": {
            "total_requests": total,
            "succeeded_requests": successes,
            "failed_requests": failures,
            "elapsed_s": elapsed_s,
            "throughput_qps_all": throughput_qps_all,
            "throughput_qps_success": throughput_qps_success,
            "throughput_rpm_all": throughput_qps_all * 60.0,
            "throughput_rpm_success": throughput_qps_success * 60.0,
            "slo_attainment_pct": (100.0 * e2e_slo_hits / total) if total else 0.0,
            "slo_violation_ms_total": float(sum(e2e_slo_violations_ms)),
            "slo_violation_ms": _metric_summary(e2e_slo_violations_ms),
            "latency_ms": _metric_summary(latencies),
            "arrival_to_dispatch_ms": _metric_summary(arrival_to_dispatch_values),
            "system_entry_to_dispatch_ms": _metric_summary(system_entry_to_dispatch_values),
            "queue_delay_ms": _metric_summary(queue_delay_values),
            "ttft_ms": _metric_summary(ttft_values),
            "e2e_ttft_ms": _metric_summary(e2e_ttft_values),
            "system_entry_e2e_ttft_ms": _metric_summary(system_entry_e2e_ttft_values),
            "queue_slo_attainment_pct": (100.0 * queue_slo_hits / total) if total else 0.0,
            "queue_slo_violation_ms_total": float(sum(queue_slo_violations_ms)),
            "queue_slo_violation_ms": _metric_summary(queue_slo_violations_ms),
            "queue_slo_coverage_pct": (100.0 * queue_available / total) if total else 0.0,
            "queue_slo_missing_count": queue_missing_count,
            "ttft_slo_attainment_pct": (100.0 * ttft_slo_hits / total) if total else 0.0,
            "ttft_slo_violation_ms_total": float(sum(ttft_slo_violations_ms)),
            "ttft_slo_violation_ms": _metric_summary(ttft_slo_violations_ms),
            "ttft_slo_coverage_pct": (100.0 * ttft_available / total) if total else 0.0,
            "ttft_slo_missing_count": ttft_missing_count,
            "e2e_ttft_slo_attainment_pct": (
                (100.0 * e2e_ttft_slo_hits / total) if total else 0.0
            ),
            "e2e_ttft_slo_violation_ms_total": float(sum(e2e_ttft_slo_violations_ms)),
            "e2e_ttft_slo_violation_ms": _metric_summary(e2e_ttft_slo_violations_ms),
            "e2e_ttft_slo_coverage_pct": (
                (100.0 * e2e_ttft_available / total) if total else 0.0
            ),
            "e2e_ttft_slo_missing_count": e2e_ttft_missing_count,
            "system_entry_e2e_ttft_slo_attainment_pct": (
                (100.0 * system_entry_e2e_ttft_slo_hits / total) if total else 0.0
            ),
            "system_entry_e2e_ttft_slo_violation_ms_total": float(
                sum(system_entry_e2e_ttft_slo_violations_ms)
            ),
            "system_entry_e2e_ttft_slo_violation_ms": _metric_summary(
                system_entry_e2e_ttft_slo_violations_ms
            ),
            "system_entry_e2e_ttft_slo_coverage_pct": (
                (100.0 * system_entry_e2e_ttft_available / total) if total else 0.0
            ),
            "system_entry_e2e_ttft_slo_missing_count": system_entry_e2e_ttft_missing_count,
            "predicted_accuracy": _metric_summary(predicted_accuracies),
            "actual_cost": _metric_summary(actual_costs),
            "actual_cost_total": float(actual_cost_total),
            "actual_cost_computable_requests": actual_cost_computable_requests,
            "actual_cost_source_counts": actual_cost_source_counts,
            "predicted_cost": _metric_summary(predicted_costs),
            "predicted_cost_total": float(predicted_cost_total),
            "predicted_cost_computable_requests": predicted_cost_computable_requests,
            "predicted_cost_source_counts": predicted_cost_source_counts,
            "cost": _metric_summary(costs),
            "cost_total": float(cost_total),
            "cost_computable_requests": cost_computable_requests,
            "cost_source_counts": cost_source_counts,
            "instance_route_counts": instance_counts,
            "accuracy_latency_tradeoff": build_tradeoff_summary(per_request),
        },
        "per_request": per_request,
    }
    if score_policy_state is not None:
        run_result["score_policy_state"] = score_policy_state.as_dict()
    if is_methodology or is_latency_history:
        run_result["methodology_config"] = scheduler.run_metadata()
    return run_result


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Router utility experiments: batch-fit GoF, wait-estimator GoF, "
            "and router baseline latency/SLO/throughput comparisons."
        )
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="router",
        choices=EXPERIMENT_MODES,
        help=(
            "Experiment mode: router (policy baselines), wait_gof "
            "(wait estimator goodness-of-fit), batch_fit "
            "(batch execution-time regression), or all."
        ),
    )
    parser.add_argument("--bucket-dir", type=Path, default=DEFAULT_PROMPT_BUCKET_DIR) ## Prompts should be the same across models since tokenizers are same
    parser.add_argument("--prompt-field", type=str, default="prompt")
    parser.add_argument("--num-requests", type=int, default=10000)
    parser.add_argument("--max-prompts-per-source", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--holdout-prompts-per-bucket",
        type=int,
        default=0,
        help=(
            "Enable holdout prompt mode and select this many prompts per bucket "
            "from raw bucketed_prompts sources after holdout offset."
        ),
    )
    parser.add_argument(
        "--holdout-bucket-dir",
        type=Path,
        default=DEFAULT_HOLDOUT_BUCKET_DIR,
        help="Raw bucketed_prompts directory used for holdout cache construction.",
    )
    parser.add_argument(
        "--holdout-start-index",
        type=int,
        default=2500,
        help="0-based index to start holdout slicing within each raw bucket JSONL.",
    )
    parser.add_argument(
        "--holdout-cache-dir",
        type=Path,
        default=DEFAULT_HOLDOUT_CACHE_DIR,
        help="Directory where converted holdout JSONL cache files are stored.",
    )
    parser.add_argument(
        "--holdout-context-length",
        type=int,
        default=65536,
        help="Context length used when computing holdout max_completion_tokens.",
    )
    parser.add_argument(
        "--rebuild-holdout-cache",
        action="store_true",
        help="Force rebuilding converted holdout cache files even when manifest matches.",
    )
    parser.add_argument("--require-existing-holdout-cache", action="store_true",
                        help="Fail instead of rebuilding an incompatible holdout cache.")
    parser.add_argument("--frozen-legacy-holdout-cache", action="store_true",
                        help="Use a checksummed derived cache retaining original paper token counts.")
    parser.add_argument(
        "--slo-min-ms",
        type=float,
        default=750.0,
        help="Minimum E2E latency SLO bound in milliseconds.",
    )
    parser.add_argument(
        "--slo-max-ms",
        type=float,
        default=12000.0,
        help="Maximum E2E latency SLO bound in milliseconds.",
    )
    parser.add_argument(
        "--queue-slo-min-ms",
        type=float,
        default=1.0,
        help="Minimum queue-delay SLO bound in milliseconds.",
    )
    parser.add_argument(
        "--queue-slo-max-ms",
        type=float,
        default=500.0,
        help="Maximum queue-delay SLO bound in milliseconds.",
    )
    parser.add_argument(
        "--ttft-slo-min-ms",
        type=float,
        default=150.0,
        help="Minimum TTFT SLO bound in milliseconds.",
    )
    parser.add_argument(
        "--ttft-slo-max-ms",
        type=float,
        default=3000.0,
        help="Maximum TTFT SLO bound in milliseconds.",
    )
    parser.add_argument(
        "--ttft-slo-base-ms",
        type=float,
        default=40.0,
        help=(
            "Base prefill-budget term in milliseconds added on top of queue SLO "
            "for TTFT SLO generation."
        ),
    )
    parser.add_argument(
        "--ttft-slo-per-prompt-token-ms",
        type=float,
        default=0.06,
        help=(
            "Per-prompt-token prefill-budget term in milliseconds added on top of "
            "queue SLO for TTFT SLO generation."
        ),
    )
    parser.add_argument(
        "--ttft-slo-jitter-min",
        type=float,
        default=0.9,
        help="Lower multiplicative jitter bound for TTFT prefill budget.",
    )
    parser.add_argument(
        "--ttft-slo-jitter-max",
        type=float,
        default=1.30,
        help="Upper multiplicative jitter bound for TTFT prefill budget.",
    )
    parser.add_argument(
        "--feasible-slo-mode",
        type=str,
        default="ttft",
        choices=FEASIBLE_SLO_MODES,
        help="SLO type used for hard/slo_aware feasibility checks.",
    )
    parser.add_argument("--request-rate-qps", type=float, default=0.0)
    parser.add_argument(
        "--arrival-process",
        type=str,
        default="poisson",
        choices=ARRIVAL_PROCESSES,
        help=(
            "Inter-arrival process at fixed mean request_rate_qps. "
            "Options: poisson (default), deterministic, mmpp2."
        ),
    )
    parser.add_argument(
        "--mmpp2-rate-ratio",
        type=float,
        default=DEFAULT_MMPP2_RATE_RATIO,
        help=(
            "Derived MMPP-2 knob: lambda_high / lambda_low (>= 1). "
            "Used only when --arrival-process=mmpp2."
        ),
    )
    parser.add_argument(
        "--mmpp2-high-fraction",
        type=float,
        default=DEFAULT_MMPP2_HIGH_FRACTION,
        help=(
            "Derived MMPP-2 knob: stationary fraction of time spent in the high-rate "
            "regime (strictly between 0 and 1). Used only with --arrival-process=mmpp2."
        ),
    )
    parser.add_argument(
        "--mmpp2-correlation-time-s",
        type=float,
        default=DEFAULT_MMPP2_CORRELATION_TIME_S,
        help=(
            "Derived MMPP-2 knob: correlation time tau in seconds where "
            "tau = 1 / (q_low_to_high + q_high_to_low). "
            "Used only when --arrival-process=mmpp2."
        ),
    )
    parser.add_argument(
        "--decouple-arrivals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When enabled, generate arrivals on the offered-load schedule and enqueue "
            "them into a separate ingress queue so system-entry timing is independent "
            "of scheduler queue backpressure. Use --no-decouple-arrivals for legacy "
            "coupled arrival generation."
        ),
    )
    parser.add_argument(
        "--enable-wait-time-polling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether the router polls each instance /wait_time endpoint before "
            "dispatch. Use --no-enable-wait-time-polling to disable polling."
        ),
    )
    parser.add_argument(
        "--critical-wait-time-timeout-s",
        type=float,
        default=0.05,
        help=(
            "Timeout in seconds for prompt-aware critical-path /wait_time polling "
            "before falling back to cached /wait_time."
        ),
    )
    parser.add_argument(
        "--readiness-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Record router-visible readiness-predictor inputs in the existing "
            "per-request wait log."
        ),
    )
    parser.add_argument(
        "--readiness-predictor-path",
        type=str,
        default=None,
        help=(
            "Optional fitted router-to-EngineCore readiness-delay predictor "
            "used by live scheduler simulation."
        ),
    )
    parser.add_argument(
        "--utilities",
        nargs="+",
        default=list(DEFAULT_ROUTER_POLICIES),
        help=f"Utility names to compare. Supported: {', '.join(BUILTIN_UTILITIES)}.",
    )
    parser.add_argument(
        "--affinity-scored-root",
        type=Path,
        default=DEFAULT_AFFINITY_SCORED_ROOT,
        help=(
            "Root containing per-model scored directories used by "
            "instance_affinity calibration (expects <root>/<model>/scored/*_scored.jsonl)."
        ),
    )
    parser.add_argument(
        "--affinity-quality-epsilon",
        type=float,
        default=DEFAULT_AFFINITY_QUALITY_EPSILON,
        help=(
            "Quality slack for bucket preference selection in instance_affinity. "
            "Models within epsilon of best quality are cost-ranked."
        ),
    )
    parser.add_argument(
        "--affinity-upgrade-margin",
        type=float,
        default=DEFAULT_AFFINITY_UPGRADE_MARGIN,
        help=(
            "Predicted-accuracy margin needed to override bucket preference "
            "and upgrade to global-best predicted accuracy."
        ),
    )
    parser.add_argument(
        "--wait-estimators",
        nargs="+",
        default=list(BUILTIN_WAIT_ESTIMATORS),
        help=(
            "Wait estimators for wait_gof. Built-ins: "
            f"{', '.join(BUILTIN_WAIT_ESTIMATORS)}. "
            "Custom format: <module_or_path>:<callable>."
        ),
    )
    parser.add_argument(
        "--wait-eval-instance-id",
        type=str,
        default=None,
        help=(
            "Optional instance_id for wait_gof single-instance evaluation. "
            "If omitted, the first configured instance is used."
        ),
    )
    parser.add_argument(
        "--pk-arrival-ewma-alpha",
        type=float,
        default=0.1,
        help="EWMA alpha for arrival-rate estimation in PK estimator.",
    )
    parser.add_argument(
        "--pk-arrival-rate-window-s",
        type=float,
        default=2.0,
        help=(
            "Window length (seconds) for arrival-rate counting in PK estimator "
            "before EWMA smoothing."
        ),
    )
    parser.add_argument(
        "--pk-service-rate-window-s",
        type=float,
        default=2.0,
        help=(
            "Window length (seconds) for service-rate counting in PK estimator "
            "before EWMA smoothing."
        ),
    )
    parser.add_argument(
        "--pk-min-service-samples",
        type=int,
        default=16,
        help="Minimum service samples diagnostic threshold for PK estimator.",
    )
    parser.add_argument(
        "--pk-service-staleness-s",
        type=float,
        default=120.0,
        help="Maximum age before PK marks service statistics as stale and reused.",
    )
    parser.add_argument(
        "--pk-rho-cap",
        type=float,
        default=0.98,
        help="Rho clip for PK denominator stability; still returns PK estimate.",
    )
    parser.add_argument(
        "--pk-max-wait-ms",
        type=float,
        default=60000.0,
        help="Upper clamp for PK wait estimate in milliseconds.",
    )
    parser.add_argument(
        "--pk-default-service-s",
        type=float,
        default=1e-4,
        help="Default service time used only when service moments are uninitialized.",
    )
    parser.add_argument(
        "--prefill-tps",
        action="append",
        default=[],
        help=(
            "Optional prefill TPS override(s) as key=value for prefill-TPS utilities "
            "(hard_prefill_tps, hard_score_proxy, score, soft_prefill_tps, "
            "hard_pk_mg1, soft_pk_mg1). Keys can be instance_id/model_id "
            "aliases (e.g., vllm-0.6b=120000)."
        ),
    )
    parser.add_argument(
        "--service-metrics-json",
        type=Path,
        default=None,
        help=(
            "Calibration JSON from compute_model_service_metrics.py. "
            "hard_score_proxy and score require per-model prefill TPS, decode "
            "TPS, and mean decode-batch latency measured under the same "
            "hardware and vLLM configuration."
        ),
    )
    parser.add_argument(
        "--per-request-wait-log",
        action="append",
        type=Path,
        default=[],
        help=(
            "Path to vLLM per-request wait log(s) produced via "
            "VLLM_PER_REQUEST_WAIT_LOG_PATH."
        ),
    )
    parser.add_argument(
        "--wait-gof-min-matched-pairs",
        type=int,
        default=1,
        help="Minimum matched predicted-vs-actual wait pairs required per estimator run.",
    )
    parser.add_argument("--instances-config", type=Path, default=None)
    parser.add_argument("--accuracy-model-path", type=str, default=None)
    parser.add_argument("--output-length-model-path", type=str, default=None)
    parser.add_argument("--latency-warmup-requests", type=str, default=None)
    parser.add_argument("--methodology-calibration-json", type=str, default=None,
                        help="Audited request-speed, Mooncake prefill, and RouteBalance TPOT artifact.")
    parser.add_argument("--routebalance-predictor-path", type=str, default=None,
                        help="Native CPU MiniLM/FAISS KNN artifact; no SFS predictor fallback.")
    parser.add_argument("--routebalance-weights", nargs=3, type=float,
                        default=(1/3, 1/3, 1/3), metavar=("QUALITY", "COST", "LATENCY"))
    parser.add_argument("--routebalance-batch-max-size", type=int, default=16)
    parser.add_argument("--routebalance-batch-wait-ms", type=float, default=25.0)
    parser.add_argument("--methodology-snapshot-max-age-ms", type=float, default=1000.0)
    parser.add_argument("--lambda-weight", type=float, default=0.3)
    parser.add_argument("--delta-weight", type=float, default=0.5)
    parser.add_argument(
        "--score-cost-weight",
        type=float,
        default=1.0,
        help="Published SCORE cost-constraint weight w_C.",
    )
    parser.add_argument(
        "--score-latency-weight",
        type=float,
        default=1.0,
        help="Published SCORE latency-constraint weight w_L.",
    )
    parser.add_argument(
        "--score-total-cost-budget",
        type=float,
        default=None,
        help=(
            "Optional SCORE total predicted response-token cost budget C_max. "
            "It is retained in the published cumulative Lagrangian expression; "
            "the paper does not specify an online lambda-update controller."
        ),
    )
    parser.add_argument("--worker-count", type=int, default=4)
    parser.add_argument("--max-queue-size", type=int, default=0)
    parser.add_argument("--max-completion-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--tokenizer-mode",
        choices=TOKENIZER_MODES,
        default="auto",
        help=(
            "Tokenizer backend used for prompt accounting and holdout-cache "
            "construction."
        ),
    )
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--chat-template-kwargs-json",
        dest="chat_template_kwargs",
        type=parse_chat_template_kwargs_json,
        default=None,
        help=(
            "JSON object passed to tokenizer and server chat templates. Defaults "
            "to Qwen's non-thinking mode; use '{}' for model families without "
            "enable_thinking."
        ),
    )
    parser.add_argument(
        "--response-map-path",
        type=Path,
        default=None,
        help="Optional response mapping log path. If omitted, a sidecar log is created next to the output JSON.",
    )
    parser.add_argument(
        "--request-log-path",
        type=Path,
        default=None,
        help=(
            "Optional predicted-wait request log path. If omitted, a sidecar log "
            "is created next to the output JSON."
        ),
    )
    parser.add_argument(
        "--batch-csv",
        action="append",
        type=Path,
        default=[],
        help=(
            "CSV file path for in-sample batch-fit regression/evaluation input. "
            "Repeatable. "
            f"Default glob: {DEFAULT_BATCH_FIT_GLOB} under {DEFAULT_BATCH_FIT_SEARCH_DIR}."
        ),
    )
    parser.add_argument(
        "--batch-train-csv",
        action="append",
        type=Path,
        default=[],
        help=(
            "CSV path(s) for held-out mode training data (set A). Repeatable. "
            "Requires --batch-test-csv and cannot be combined with --batch-csv."
        ),
    )
    parser.add_argument(
        "--batch-test-csv",
        action="append",
        type=Path,
        default=[],
        help=(
            "CSV path(s) for held-out mode test data (set B). Repeatable. "
            "Requires --batch-train-csv and cannot be combined with --batch-csv."
        ),
    )
    parser.add_argument(
        "--batch-fit-output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "batch_fit",
    )
    parser.add_argument("--batch-fit-stall-percentile", type=float, default=99.9)
    parser.add_argument("--batch-fit-huber-epsilon", type=float, default=1.35)
    parser.add_argument("--batch-fit-max-plot-points", type=int, default=20000)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--tokenizer-id", type=str, default=DEFAULT_TOKENIZER_ID)
    parser.add_argument(
        "--batch-fit-feature-set",
        type=str,
        default=_BATCH_FIT_DEFAULT_FEATURE_SET,
    )
    parser.add_argument(
        "--batch-fit-nonnegative",
        action="store_true",
        help=(
            "Constrain the batch-fit intercept and slopes to nonnegative values. "
            "Use this when evaluating the physical SFS calibration model."
        ),
    )

    args = parser.parse_args()
    try:
        from sfs_core.routing.methodology_policies import RouteBalanceWeights
        RouteBalanceWeights(*args.routebalance_weights)
    except ValueError as exc:
        parser.error(str(exc))
    if args.routebalance_batch_max_size < 1:
        parser.error("--routebalance-batch-max-size must be >= 1.")
    for name in ("routebalance_batch_wait_ms", "methodology_snapshot_max_age_ms"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative.")
    if "vllm_sr_latency" in args.utilities:
        from sfs_core.routing.latency_warmup import load_warmup
        try:
            load_warmup(args.latency_warmup_requests)
        except (ValueError, OSError, KeyError) as exc:
            parser.error(str(exc))
    methodology_selected = set(args.utilities) & {"lmdeploy_proxy", "mooncake_prefill", "routebalance"}
    if methodology_selected and not args.methodology_calibration_json:
        parser.error("Methodology baselines require --methodology-calibration-json.")
    if "routebalance" in methodology_selected and not args.routebalance_predictor_path:
        parser.error("RouteBalance requires --routebalance-predictor-path.")
    try:
        args.prefill_tps_overrides = _parse_prefill_tps_overrides(args.prefill_tps)
        args.calibrated_service_metrics = _load_calibrated_service_metrics(
            args.service_metrics_json
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.wait_gof_min_matched_pairs < 1:
        parser.error("--wait-gof-min-matched-pairs must be >= 1.")
    if args.pk_arrival_rate_window_s <= 0:
        parser.error("--pk-arrival-rate-window-s must be > 0.")
    if args.pk_service_rate_window_s <= 0:
        parser.error("--pk-service-rate-window-s must be > 0.")
    if args.slo_min_ms > args.slo_max_ms:
        parser.error("--slo-min-ms must be <= --slo-max-ms.")
    if args.queue_slo_min_ms > args.queue_slo_max_ms:
        parser.error("--queue-slo-min-ms must be <= --queue-slo-max-ms.")
    if args.ttft_slo_min_ms > args.ttft_slo_max_ms:
        parser.error("--ttft-slo-min-ms must be <= --ttft-slo-max-ms.")
    if args.queue_slo_max_ms > args.ttft_slo_max_ms:
        parser.error("--queue-slo-max-ms must be <= --ttft-slo-max-ms.")
    if args.ttft_slo_per_prompt_token_ms < 0:
        parser.error("--ttft-slo-per-prompt-token-ms must be >= 0.")
    if args.ttft_slo_jitter_min <= 0 or args.ttft_slo_jitter_max <= 0:
        parser.error("--ttft-slo-jitter-min/--ttft-slo-jitter-max must be > 0.")
    if args.ttft_slo_jitter_min > args.ttft_slo_jitter_max:
        parser.error("--ttft-slo-jitter-min must be <= --ttft-slo-jitter-max.")
    if args.affinity_quality_epsilon < 0:
        parser.error("--affinity-quality-epsilon must be >= 0.")
    if args.affinity_upgrade_margin < 0:
        parser.error("--affinity-upgrade-margin must be >= 0.")
    if not math.isfinite(args.score_cost_weight) or args.score_cost_weight < 0:
        parser.error("--score-cost-weight must be finite and >= 0.")
    if (
        not math.isfinite(args.score_latency_weight)
        or args.score_latency_weight < 0
    ):
        parser.error("--score-latency-weight must be finite and >= 0.")
    if args.score_total_cost_budget is not None and (
        not math.isfinite(args.score_total_cost_budget)
        or args.score_total_cost_budget < 0
    ):
        parser.error("--score-total-cost-budget must be finite and >= 0.")
    if args.arrival_process == "mmpp2":
        try:
            _derive_mmpp2_params(
                request_rate_qps=float(args.request_rate_qps),
                rate_ratio=float(args.mmpp2_rate_ratio),
                high_fraction=float(args.mmpp2_high_fraction),
                correlation_time_s=float(args.mmpp2_correlation_time_s),
            )
        except ValueError as exc:
            parser.error(str(exc))

    if args.holdout_prompts_per_bucket < 0:
        parser.error("--holdout-prompts-per-bucket must be >= 0.")
    if args.holdout_prompts_per_bucket > 0:
        if args.holdout_start_index < 0:
            parser.error("--holdout-start-index must be >= 0 when holdout mode is enabled.")
        if args.holdout_context_length <= 0:
            parser.error("--holdout-context-length must be > 0 when holdout mode is enabled.")

    args.chat_template_kwargs = resolve_chat_template_kwargs(
        args.chat_template_kwargs
    )

    return args


def _utility_response_map_path(base_path: Path, utility_name: str, multi_utility: bool) -> Path:
    if not multi_utility:
        return base_path
    suffix = base_path.suffix or ".log"
    return base_path.with_name(f"{base_path.stem}_{utility_name}{suffix}")


def _resolve_per_request_wait_logs(args: argparse.Namespace) -> list[Path]:
    if args.per_request_wait_log:
        return [Path(p).expanduser() for p in args.per_request_wait_log]
    env_path = os.environ.get("VLLM_PER_REQUEST_WAIT_LOG_PATH")
    if env_path:
        return [Path(env_path).expanduser()]
    return []


def _resolve_prompt_source(
    args: argparse.Namespace,
) -> tuple[Path, Dict[str, Any]]:
    if args.holdout_prompts_per_bucket <= 0:
        resolved_bucket_dir = args.bucket_dir.expanduser().resolve()
        source_info: Dict[str, Any] = {
            "mode": "default",
            "sampling_mode": "random",
            "effective_bucket_dir": str(resolved_bucket_dir),
            "source_bucket_dir": str(resolved_bucket_dir),
            "holdout_enabled": False,
        }
        return resolved_bucket_dir, source_info

    requested_cache_dir = args.holdout_cache_dir.expanduser().resolve()
    default_cache_root = DEFAULT_HOLDOUT_CACHE_DIR.expanduser().resolve()
    if requested_cache_dir == default_cache_root:
        cache_dir_for_holdout = PROMPTS_DATA_ROOT / f"holdout_cache_{int(args.holdout_prompts_per_bucket)}"
    else:
        cache_dir_for_holdout = requested_cache_dir

    cache_dir, manifest, cache_rebuilt = prepare_holdout_prompt_cache(
        source_bucket_dir=args.holdout_bucket_dir,
        cache_dir=cache_dir_for_holdout,
        tokenizer_id=args.tokenizer_id,
        tokenizer_mode=args.tokenizer_mode,
        holdout_start_index=args.holdout_start_index,
        holdout_prompts_per_bucket=args.holdout_prompts_per_bucket,
        holdout_context_length=args.holdout_context_length,
        max_completion_tokens=args.max_completion_tokens,
        rebuild=args.rebuild_holdout_cache,
        require_existing=getattr(args, "require_existing_holdout_cache", False),
        frozen_legacy=getattr(args, "frozen_legacy_holdout_cache", False),
        system_prompt=args.system_prompt,
        chat_template_kwargs=args.chat_template_kwargs,
    )

    source_info = {
        "mode": "holdout",
        "sampling_mode": "mixed_then_shuffle",
        "effective_bucket_dir": str(cache_dir),
        "source_bucket_dir": str(args.holdout_bucket_dir.expanduser().resolve()),
        "holdout_enabled": True,
        "holdout_cache_dir": str(cache_dir),
        "holdout_start_index": int(args.holdout_start_index),
        "holdout_prompts_per_bucket": int(args.holdout_prompts_per_bucket),
        "holdout_context_length": int(args.holdout_context_length),
        "cache_rebuilt": bool(cache_rebuilt),
        "cache_manifest": manifest,
    }
    return cache_dir, source_info


def _build_request_set(
    args: argparse.Namespace,
) -> tuple[list[ExperimentRequest], list[Dict[str, Any]], Dict[str, Any]]:
    effective_bucket_dir, prompt_source_info = _resolve_prompt_source(args)
    requests = build_experiment_requests(
        bucket_dir=effective_bucket_dir,
        num_requests=args.num_requests,
        seed=args.seed,
        slo_min_ms=args.slo_min_ms,
        slo_max_ms=args.slo_max_ms,
        queue_slo_min_ms=args.queue_slo_min_ms,
        queue_slo_max_ms=args.queue_slo_max_ms,
        ttft_slo_min_ms=args.ttft_slo_min_ms,
        ttft_slo_max_ms=args.ttft_slo_max_ms,
        ttft_slo_base_ms=args.ttft_slo_base_ms,
        ttft_slo_per_prompt_token_ms=args.ttft_slo_per_prompt_token_ms,
        ttft_slo_jitter_min=args.ttft_slo_jitter_min,
        ttft_slo_jitter_max=args.ttft_slo_jitter_max,
        prompt_sampling_mode=prompt_source_info["sampling_mode"],
        holdout_prompts_per_bucket=args.holdout_prompts_per_bucket,
    )
    request_manifest = [
        {
            "request_id": req.request_id,
            "bucket": req.bucket,
            "prompt_tokens": req.prompt_tokens,
            "latency_slo_ms": req.latency_slo_ms,
            "queue_slo_ms": req.queue_slo_ms,
            "ttft_slo_ms": req.ttft_slo_ms,
        }
        for req in requests
    ]
    prompt_source_info["effective_bucket_files"] = get_prompt_bucket_files(effective_bucket_dir)
    prompt_source_info["effective_bucket_count"] = len(prompt_source_info["effective_bucket_files"])
    if prompt_source_info.get("mode") == "holdout":
        prompt_source_info["effective_pool_size"] = int(
            sum(
                _as_nonnegative_int(
                    (prompt_source_info.get("cache_manifest", {}).get("bucket_counts", {}) or {}).get(file_name)
                )
                or 0
                for file_name in prompt_source_info["effective_bucket_files"]
            )
        )
    else:
        prompt_source_info["effective_pool_size"] = None
    return requests, request_manifest, prompt_source_info


def _bucket_from_scored_record(record: Dict[str, Any], *, scored_path: Path) -> str:
    bucket = record.get("bucket")
    if isinstance(bucket, str) and bucket.strip():
        return _normalize_bucket_name(bucket)
    metadata = record.get("prompt_metadata")
    if isinstance(metadata, dict):
        metadata_bucket = metadata.get("bucket")
        if isinstance(metadata_bucket, str) and metadata_bucket.strip():
            return _normalize_bucket_name(metadata_bucket)
    fallback = scored_path.stem
    if fallback.endswith("_scored"):
        fallback = fallback[: -len("_scored")]
    return _normalize_bucket_name(fallback)


def _prompt_tokens_from_scored_record(record: Dict[str, Any]) -> Optional[int]:
    prompt_tokens = _as_nonnegative_int(record.get("prompt_tokens"))
    if prompt_tokens is not None:
        return int(prompt_tokens)
    metadata = record.get("prompt_metadata")
    if isinstance(metadata, dict):
        metadata_prompt_tokens = _as_nonnegative_int(metadata.get("prompt_tokens"))
        if metadata_prompt_tokens is not None:
            return int(metadata_prompt_tokens)
    return None


def _output_tokens_from_scored_record(record: Dict[str, Any]) -> Optional[int]:
    response = record.get("response")
    if isinstance(response, dict):
        completion_tokens = _as_nonnegative_int(response.get("completion_tokens"))
        if completion_tokens is not None:
            return int(completion_tokens)
        output_tokens = _as_nonnegative_int(response.get("output_tokens"))
        if output_tokens is not None:
            return int(output_tokens)

    completion_tokens = _as_nonnegative_int(record.get("completion_tokens"))
    if completion_tokens is not None:
        return int(completion_tokens)
    output_tokens = _as_nonnegative_int(record.get("output_tokens"))
    if output_tokens is not None:
        return int(output_tokens)
    return None


def _affinity_alias_index(
    instances: Dict[str, InstanceClient],
) -> Dict[str, list[str]]:
    alias_to_instances: Dict[str, set[str]] = {}
    for instance_id, instance in instances.items():
        for raw_candidate in (
            instance_id,
            getattr(instance, "model_id", None),
            getattr(instance, "default_model", None),
        ):
            alias = _normalize_prefill_tps_key(raw_candidate)
            if not alias:
                continue
            alias_to_instances.setdefault(alias, set()).add(str(instance_id))
    return {
        alias: sorted(instance_ids)
        for alias, instance_ids in alias_to_instances.items()
        if instance_ids
    }


def _affinity_expected_cost(
    *,
    instance_id: str,
    mean_prompt_tokens: float,
    mean_output_tokens: float,
    instance_costs: Dict[str, Dict[str, float]],
) -> float:
    cost_info = instance_costs.get(instance_id) or {}
    prompt_rate = float(cost_info.get("prompt", 0.0))
    output_rate = float(cost_info.get("output", 0.0))
    return (prompt_rate * float(mean_prompt_tokens)) + (
        output_rate * float(mean_output_tokens)
    )


def _build_instance_affinity_calibration(
    *,
    scored_root: Path,
    instances: Dict[str, InstanceClient],
    instance_costs: Dict[str, Dict[str, float]],
    quality_epsilon: float,
) -> Dict[str, Any]:
    resolved_scored_root = scored_root.expanduser().resolve()
    if not resolved_scored_root.exists():
        raise FileNotFoundError(
            f"Affinity scored root does not exist: {resolved_scored_root}"
        )
    if not resolved_scored_root.is_dir():
        raise ValueError(
            f"Affinity scored root is not a directory: {resolved_scored_root}"
        )

    alias_to_instances = _affinity_alias_index(instances)
    if not alias_to_instances:
        raise ValueError("Cannot build instance-affinity calibration with no instances.")

    bucket_model_accum: Dict[str, Dict[str, Dict[str, float]]] = {}
    global_model_accum: Dict[str, Dict[str, float]] = {}
    discovered_models: set[str] = set()

    model_dirs = sorted(p for p in resolved_scored_root.iterdir() if p.is_dir())
    for model_dir in model_dirs:
        scored_dir = model_dir / "scored"
        if not scored_dir.is_dir():
            continue
        model_key = _normalize_prefill_tps_key(model_dir.name)
        scored_paths = sorted(scored_dir.glob("*_scored.jsonl"))
        if not scored_paths:
            continue
        discovered_models.add(model_key)
        for scored_path in scored_paths:
            with scored_path.open("r", encoding="utf-8") as fp:
                for line_number, line in enumerate(fp, start=1):
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        record = json.loads(text)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Malformed JSON in affinity scored file {scored_path} "
                            f"line {line_number}: {exc}"
                        ) from exc
                    if not isinstance(record, dict):
                        continue

                    quality = _as_nonnegative_float(record.get("quality"))
                    if quality is None:
                        continue

                    bucket = _bucket_from_scored_record(record, scored_path=scored_path)
                    prompt_tokens = _prompt_tokens_from_scored_record(record)
                    output_tokens = _output_tokens_from_scored_record(record)

                    model_bucket_accum = bucket_model_accum.setdefault(bucket, {}).setdefault(
                        model_key,
                        {
                            "count": 0.0,
                            "quality_sum": 0.0,
                            "prompt_sum": 0.0,
                            "prompt_count": 0.0,
                            "output_sum": 0.0,
                            "output_count": 0.0,
                        },
                    )
                    model_bucket_accum["count"] += 1.0
                    model_bucket_accum["quality_sum"] += float(quality)
                    if prompt_tokens is not None:
                        model_bucket_accum["prompt_sum"] += float(prompt_tokens)
                        model_bucket_accum["prompt_count"] += 1.0
                    if output_tokens is not None:
                        model_bucket_accum["output_sum"] += float(output_tokens)
                        model_bucket_accum["output_count"] += 1.0

                    model_global_accum = global_model_accum.setdefault(
                        model_key,
                        {
                            "count": 0.0,
                            "quality_sum": 0.0,
                        },
                    )
                    model_global_accum["count"] += 1.0
                    model_global_accum["quality_sum"] += float(quality)

    if not bucket_model_accum:
        raise ValueError(
            "No usable scored quality rows were found for instance_affinity "
            f"under {resolved_scored_root}."
        )

    model_to_instances: Dict[str, list[str]] = {
        model_key: list(alias_to_instances.get(model_key, []))
        for model_key in sorted(discovered_models)
    }

    bucket_to_preferred_instances: Dict[str, list[str]] = {}
    bucket_preference_models: Dict[str, list[str]] = {}
    bucket_summary: Dict[str, Any] = {}

    for bucket in sorted(bucket_model_accum.keys()):
        per_model_rows: list[Dict[str, Any]] = []
        for model_key, accum in sorted(bucket_model_accum[bucket].items()):
            count = int(accum.get("count", 0.0))
            if count <= 0:
                continue
            instance_ids = list(model_to_instances.get(model_key, []))
            if not instance_ids:
                continue

            mean_quality = float(accum["quality_sum"] / float(count))
            prompt_count = float(accum.get("prompt_count", 0.0))
            output_count = float(accum.get("output_count", 0.0))
            mean_prompt_tokens = (
                float(accum["prompt_sum"] / prompt_count) if prompt_count > 0 else 0.0
            )
            mean_output_tokens = (
                float(accum["output_sum"] / output_count) if output_count > 0 else 0.0
            )
            expected_cost_by_instance = {
                instance_id: _affinity_expected_cost(
                    instance_id=instance_id,
                    mean_prompt_tokens=mean_prompt_tokens,
                    mean_output_tokens=mean_output_tokens,
                    instance_costs=instance_costs,
                )
                for instance_id in instance_ids
            }
            expected_cost = min(expected_cost_by_instance.values())

            per_model_rows.append(
                {
                    "model_key": model_key,
                    "instance_ids": list(instance_ids),
                    "mean_quality": mean_quality,
                    "mean_prompt_tokens": mean_prompt_tokens,
                    "mean_output_tokens": mean_output_tokens,
                    "expected_cost": float(expected_cost),
                    "expected_cost_by_instance": {
                        instance_id: float(cost_value)
                        for instance_id, cost_value in expected_cost_by_instance.items()
                    },
                    "count": count,
                }
            )

        if not per_model_rows:
            raise ValueError(
                "No active instance coverage found for bucket "
                f"'{bucket}' while building instance_affinity calibration. "
                "Ensure active instances map to scored model directories."
            )

        best_quality = max(float(row["mean_quality"]) for row in per_model_rows)
        eligible_rows = [
            row
            for row in per_model_rows
            if (best_quality - float(row["mean_quality"])) <= (quality_epsilon + 1e-12)
        ]
        min_eligible_cost = min(float(row["expected_cost"]) for row in eligible_rows)
        selected_rows = [
            row
            for row in eligible_rows
            if math.isclose(
                float(row["expected_cost"]),
                min_eligible_cost,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ]

        preferred_models = sorted(
            {
                str(row["model_key"])
                for row in selected_rows
            }
        )
        preferred_instances = sorted(
            {
                str(instance_id)
                for row in selected_rows
                for instance_id in row["instance_ids"]
            }
        )
        if not preferred_instances:
            raise ValueError(
                f"Bucket '{bucket}' did not resolve to any preferred active instances."
            )

        bucket_to_preferred_instances[bucket] = list(preferred_instances)
        bucket_preference_models[bucket] = list(preferred_models)
        bucket_summary[bucket] = {
            "best_quality": float(best_quality),
            "selected_models": list(preferred_models),
            "selected_instances": list(preferred_instances),
            "eligible_models": sorted(
                {str(row["model_key"]) for row in eligible_rows}
            ),
            "quality_epsilon": float(quality_epsilon),
            "model_stats": per_model_rows,
        }

    active_global_rows: list[Dict[str, Any]] = []
    for model_key, accum in sorted(global_model_accum.items()):
        count = int(accum.get("count", 0.0))
        if count <= 0:
            continue
        instance_ids = list(model_to_instances.get(model_key, []))
        if not instance_ids:
            continue
        mean_quality = float(accum["quality_sum"] / float(count))
        active_global_rows.append(
            {
                "model_key": str(model_key),
                "mean_quality": mean_quality,
                "count": count,
                "instance_ids": list(instance_ids),
            }
        )

    if not active_global_rows:
        raise ValueError(
            "Failed to resolve global fallback for instance_affinity: no active "
            "instances matched any scored model directories."
        )

    global_best_quality = max(float(row["mean_quality"]) for row in active_global_rows)
    global_best_models = sorted(
        [
            str(row["model_key"])
            for row in active_global_rows
            if math.isclose(
                float(row["mean_quality"]),
                float(global_best_quality),
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ]
    )
    global_best_instances = sorted(
        {
            str(instance_id)
            for row in active_global_rows
            if row["model_key"] in global_best_models
            for instance_id in row["instance_ids"]
        }
    )
    if not global_best_instances:
        raise ValueError(
            "Failed to resolve global fallback instances for instance_affinity."
        )

    return {
        "score_field": "quality",
        "scored_root": str(resolved_scored_root),
        "quality_epsilon": float(quality_epsilon),
        "bucket_to_preferred_instances": bucket_to_preferred_instances,
        "bucket_preference_models": bucket_preference_models,
        "bucket_summary": bucket_summary,
        "model_to_instances": model_to_instances,
        "active_model_quality_summary": active_global_rows,
        "global_best_quality": float(global_best_quality),
        "global_best_quality_models": global_best_models,
        "global_best_quality_instances": global_best_instances,
        "buckets": sorted(bucket_to_preferred_instances.keys()),
    }


def _baseline_runtime_params(
    utility_name: str,
    *,
    lambda_weight: float,
    delta_weight: float,
) -> tuple[float, float, str, Optional[WaitEstimatorCallable], str, bool]:
    utility = utility_name.strip().lower()
    if utility in {"lmdeploy_proxy", "mooncake_prefill", "routebalance", "vllm_sr_latency"}:
        return lambda_weight, 0.0, utility, None, utility, True
    if utility == "round_robin":
        return (
            lambda_weight,
            delta_weight,
            "round_robin",
            _wait_estimator_zero,
            "zero",
            True,
        )
    if utility == "shortest_queue":
        return (
            lambda_weight,
            delta_weight,
            "shortest_queue",
            _wait_estimator_live,
            "live",
            False,
        )
    if utility == "instance_affinity":
        return (
            lambda_weight,
            delta_weight,
            "instance_affinity",
            _wait_estimator_live,
            "live",
            True,
        )
    if utility == "latency_agnostic":
        return (
            lambda_weight,
            0.0,
            "utility",
            _wait_estimator_zero,
            "zero",
            True,
        )
    if utility == "latency_and_cost_agnostic":
        return (
            0.0,
            0.0,
            "utility",
            _wait_estimator_zero,
            "zero",
            True,
        )
    if utility == "hard":
        return (
            lambda_weight,
            0.0,
            "utility",
            _wait_estimator_live,
            "live",
            False,
        )
    if utility == "hard_prefill_tps":
        return (
            lambda_weight,
            0.0,
            "utility",
            _wait_estimator_prefill_tps_ttft,
            PREFILL_TPS_TTFT_ESTIMATOR_NAME,
            False,
        )
    if utility == "hard_score_proxy":
        return (
            lambda_weight,
            0.0,
            "utility",
            _wait_estimator_score_proxy_ttft,
            SCORE_PROXY_TTFT_ESTIMATOR_NAME,
            False,
        )
    if utility == "score":
        return (
            lambda_weight,
            0.0,
            "utility",
            _wait_estimator_score_total_latency,
            SCORE_TOTAL_LATENCY_ESTIMATOR_NAME,
            False,
        )
    if utility == "hard_pk_mg1":
        return (
            lambda_weight,
            0.0,
            "utility",
            _wait_estimator_pk_mg1,
            "pk_mg1",
            False,
        )
    if utility == "soft_prefill_tps":
        return (
            lambda_weight,
            delta_weight,
            "utility",
            _wait_estimator_prefill_tps_ttft,
            PREFILL_TPS_TTFT_ESTIMATOR_NAME,
            False,
        )
    if utility == "soft_pk_mg1":
        return (
            lambda_weight,
            delta_weight,
            "utility",
            _wait_estimator_pk_mg1,
            "pk_mg1",
            False,
        )
    return (
        lambda_weight,
        delta_weight,
        "utility",
        _wait_estimator_live,
        "live",
        False,
    )


async def run_router_experiment(
    *,
    args: argparse.Namespace,
    requests: list[ExperimentRequest],
    instances: dict[str, InstanceClient],
    instance_costs: Dict[str, Dict[str, float]],
    instance_metadata: Dict[str, Any],
    response_map_base_path: Path,
    request_log_base_path: Path,
    trial_monitor: Any = None,
) -> Dict[str, Any]:
    normalized_utilities = [str(name).strip().lower() for name in args.utilities]
    for utility_name in args.utilities:
        if utility_name not in BUILTIN_UTILITIES:
            raise ValueError(
                f"Unsupported utility '{utility_name}'. "
                f"Choose from: {', '.join(BUILTIN_UTILITIES)}."
            )

    prefill_tps_by_instance: Optional[Dict[str, float]] = None
    prefill_tps_resolution: Optional[Dict[str, Dict[str, Any]]] = None
    service_rates_by_instance: Optional[Dict[str, float]] = None
    service_rates_resolution: Optional[Dict[str, Dict[str, Any]]] = None
    decode_tps_by_instance: Optional[Dict[str, float]] = None
    mean_decode_batch_ms_by_instance: Optional[Dict[str, float]] = None
    score_proxy_metrics_resolution: Optional[Dict[str, Dict[str, Any]]] = None
    prefill_tps_utilities = {
        "hard_prefill_tps",
        "hard_score_proxy",
        "score",
        "soft_prefill_tps",
        "hard_pk_mg1",
        "soft_pk_mg1",
    }
    if any(name in prefill_tps_utilities for name in normalized_utilities):
        prefill_tps_by_instance, prefill_tps_resolution = _resolve_router_prefill_tps_by_instance(
            instances=instances,
            overrides=getattr(args, "prefill_tps_overrides", None),
            calibrated_metrics=getattr(args, "calibrated_service_metrics", None),
        )
    if any(
        name in {"hard_score_proxy", "score"}
        for name in normalized_utilities
    ):
        (
            decode_tps_by_instance,
            mean_decode_batch_ms_by_instance,
            score_proxy_metrics_resolution,
        ) = _resolve_score_proxy_metrics_by_instance(
            instances=instances,
            calibrated_metrics=getattr(
                args,
                "calibrated_service_metrics",
                {},
            ),
        )
    if any(name in {"hard_pk_mg1", "soft_pk_mg1"} for name in normalized_utilities):
        service_rates_by_instance, service_rates_resolution = (
            _resolve_router_service_rates_by_instance(instances=instances)
        )
    affinity_calibration: Optional[Dict[str, Any]] = None
    if "instance_affinity" in normalized_utilities:
        affinity_calibration = _build_instance_affinity_calibration(
            scored_root=args.affinity_scored_root,
            instances=instances,
            instance_costs=instance_costs,
            quality_epsilon=float(args.affinity_quality_epsilon),
        )

    resolved_wait_logs = _resolve_per_request_wait_logs(args)
    if not resolved_wait_logs:
        raise ValueError(
            "router scoring requires per-request wait logs. Provide --per-request-wait-log "
            "or set VLLM_PER_REQUEST_WAIT_LOG_PATH."
        )
    per_request_wait_logs = [path for path in resolved_wait_logs if path.exists()]
    if not per_request_wait_logs:
        raise FileNotFoundError(
            "router scoring could not find per-request wait log files. Checked: "
            + ", ".join(str(path) for path in resolved_wait_logs)
        )
    offsets = _snapshot_log_offsets(per_request_wait_logs)

    runs: list[Dict[str, Any]] = []
    response_map_paths: Dict[str, str] = {}
    request_log_paths: Dict[str, str] = {}
    for utility_name in args.utilities:
        utility_response_map_path = _utility_response_map_path(
            response_map_base_path,
            utility_name,
            multi_utility=len(args.utilities) > 1,
        )
        utility_request_log_path = _utility_response_map_path(
            request_log_base_path,
            f"router_{utility_name}",
            multi_utility=True,
        )
        response_map_paths[utility_name] = str(utility_response_map_path)
        request_log_paths[utility_name] = str(utility_request_log_path)
        (
            run_lambda,
            run_delta,
            route_strategy,
            wait_estimator,
            wait_estimator_name,
            skip_wait_result_build,
        ) = _baseline_runtime_params(
            utility_name,
            lambda_weight=args.lambda_weight,
            delta_weight=args.delta_weight,
        )

        wait_estimator_context: Optional[Dict[str, Any]] = None
        if wait_estimator_name == PREFILL_TPS_TTFT_ESTIMATOR_NAME:
            if not prefill_tps_by_instance:
                raise ValueError(
                    "Prefill-TPS utility selected but no prefill TPS mapping resolved."
                )
            wait_estimator_context = {
                "prefill_tps_by_instance": dict(prefill_tps_by_instance),
            }
        elif wait_estimator_name == SCORE_PROXY_TTFT_ESTIMATOR_NAME:
            if (
                not prefill_tps_by_instance
                or not decode_tps_by_instance
                or not mean_decode_batch_ms_by_instance
            ):
                raise ValueError(
                    "SCORE proxy selected but its calibrated service metrics "
                    "were not resolved."
                )
            wait_estimator_context = {
                "prefill_tps_by_instance": dict(prefill_tps_by_instance),
                "decode_tps_by_instance": dict(decode_tps_by_instance),
                "mean_decode_batch_ms_by_instance": dict(
                    mean_decode_batch_ms_by_instance
                ),
            }
        elif wait_estimator_name == SCORE_TOTAL_LATENCY_ESTIMATOR_NAME:
            if (
                not prefill_tps_by_instance
                or not decode_tps_by_instance
                or not mean_decode_batch_ms_by_instance
            ):
                raise ValueError(
                    "Published SCORE selected but its calibrated service "
                    "metrics were not resolved."
                )
            wait_estimator_context = {
                "prefill_tps_by_instance": dict(prefill_tps_by_instance),
                "decode_tps_by_instance": dict(decode_tps_by_instance),
                "mean_decode_batch_ms_by_instance": dict(
                    mean_decode_batch_ms_by_instance
                ),
            }
        elif wait_estimator_name == "pk_mg1":
            if not prefill_tps_by_instance:
                raise ValueError(
                    "PK-MG1 utility selected but no prefill TPS mapping resolved."
                )
            if not service_rates_by_instance:
                raise ValueError(
                    "PK-MG1 utility selected but no service-rate priors resolved."
                )
            wait_estimator_context = {
                "prefill_tps_by_instance": dict(prefill_tps_by_instance),
                "pk_tracker": PKOnlineStatsCollector(
                    instance_ids=list(instances.keys()),
                    per_request_wait_logs=per_request_wait_logs,
                    arrival_ewma_alpha=float(args.pk_arrival_ewma_alpha),
                    arrival_rate_window_s=float(args.pk_arrival_rate_window_s),
                    service_rate_window_s=float(args.pk_service_rate_window_s),
                    service_mu_prior_rps_by_instance=dict(service_rates_by_instance),
                    min_service_samples=int(args.pk_min_service_samples),
                    service_staleness_s=float(args.pk_service_staleness_s),
                    rho_cap=float(args.pk_rho_cap),
                    max_wait_ms=float(args.pk_max_wait_ms),
                    default_service_s=float(args.pk_default_service_s),
                ),
                "pk_last_diagnostics": {},
            }
        include_unconditional_live_fetch = _should_include_unconditional_live_fetch(
            wait_estimator_name
        )
        affinity_bucket_to_instances: Optional[Dict[str, list[str]]] = None
        affinity_global_fallback_instances: Optional[list[str]] = None
        if route_strategy == "instance_affinity":
            if affinity_calibration is None:
                raise ValueError(
                    "instance_affinity selected but no affinity calibration data was built."
                )
            affinity_bucket_to_instances = dict(
                affinity_calibration.get("bucket_to_preferred_instances", {})
            )
            affinity_global_fallback_instances = list(
                affinity_calibration.get("global_best_quality_instances", [])
            )

        run = await run_policy(
            utility_name=utility_name,
            requests=requests,
            instances=instances,
            instance_costs=instance_costs,
            accuracy_model_path=args.accuracy_model_path,
            output_length_model_path=args.output_length_model_path,
            lambda_weight=run_lambda,
            delta_weight=run_delta,
            worker_count=args.worker_count,
            max_queue_size=args.max_queue_size,
            request_rate_qps=args.request_rate_qps,
            arrival_process=args.arrival_process,
            arrival_seed=args.seed,
            mmpp2_rate_ratio=float(args.mmpp2_rate_ratio),
            mmpp2_high_fraction=float(args.mmpp2_high_fraction),
            mmpp2_correlation_time_s=float(args.mmpp2_correlation_time_s),
            decouple_arrivals=bool(args.decouple_arrivals),
            max_completion_tokens=args.max_completion_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            tokenizer_id=args.tokenizer_id,
            tokenizer_mode=args.tokenizer_mode,
            system_prompt=args.system_prompt,
            chat_template_kwargs=args.chat_template_kwargs,
            response_map_path=str(utility_response_map_path),
            request_log_path=str(utility_request_log_path),
            route_strategy=route_strategy,
            wait_estimator=wait_estimator,
            wait_estimator_name=wait_estimator_name,
            wait_estimator_context=wait_estimator_context,
            run_label=utility_name,
            feasible_slo_mode=args.feasible_slo_mode,
            per_request_wait_logs=per_request_wait_logs,
            wait_log_offsets=offsets,
            skip_wait_result_build=skip_wait_result_build,
            include_unconditional_live_fetch=include_unconditional_live_fetch,
            enable_wait_time_polling=bool(args.enable_wait_time_polling),
            critical_wait_time_timeout_s=float(args.critical_wait_time_timeout_s),
            readiness_diagnostics=bool(args.readiness_diagnostics),
            readiness_predictor_path=args.readiness_predictor_path,
            route_random_seed=int(args.seed),
            affinity_bucket_to_instances=affinity_bucket_to_instances,
            affinity_global_fallback_instances=affinity_global_fallback_instances,
            affinity_upgrade_margin=float(args.affinity_upgrade_margin),
            score_cost_weight=float(args.score_cost_weight),
            score_latency_weight=float(args.score_latency_weight),
            score_total_cost_budget=args.score_total_cost_budget,
            latency_warmup_requests=getattr(args, "latency_warmup_requests", None),
            methodology_calibration_path=getattr(args, "methodology_calibration_json", None),
            methodology_serving_profile=instance_metadata.get("serving_profile", {}),
            routebalance_predictor_path=getattr(args, "routebalance_predictor_path", None),
            routebalance_weights=tuple(getattr(args, "routebalance_weights", (1/3, 1/3, 1/3))),
            routebalance_batch_max_size=getattr(args, "routebalance_batch_max_size", 16),
            routebalance_batch_wait_ms=getattr(args, "routebalance_batch_wait_ms", 25.0),
            methodology_snapshot_max_age_ms=getattr(args, "methodology_snapshot_max_age_ms", 1000.0),
            trial_monitor=trial_monitor,
            close_instances_on_stop=False,
            remaining_length=instance_metadata.get("remaining_length"),
        )
        run["remaining_length_rule"] = str(
            (instance_metadata.get("remaining_length") or {}).get("rule", "current")
        )
        run["baseline_config"] = {
            "lambda_weight": run_lambda,
            "delta_weight": run_delta,
            "route_strategy": route_strategy,
            "wait_estimator": wait_estimator_name,
            "feasible_slo_mode": args.feasible_slo_mode,
        }
        if "methodology_config" in run:
            run["baseline_config"]["methodology"] = run["methodology_config"]
        if utility_name == "score":
            run["baseline_config"]["score"] = {
                "published_fixed_lambda_argmax": True,
                "quality_term": "predicted_quality",
                "cost_term": "output_rate * predicted_output_tokens",
                "latency_term": "predicted_total_response_latency_s",
                "cost_weight": float(args.score_cost_weight),
                "latency_weight": float(args.score_latency_weight),
                "total_cost_budget": args.score_total_cost_budget,
                "latency_limit_source": "request_e2e_latency_slo_ms",
                "lambda_update": "fixed; paper does not specify controller",
            }
        if route_strategy == "instance_affinity" and affinity_calibration is not None:
            run["baseline_config"]["instance_affinity"] = {
                "scored_root": affinity_calibration.get("scored_root"),
                "score_field": affinity_calibration.get("score_field"),
                "quality_epsilon": affinity_calibration.get("quality_epsilon"),
                "upgrade_margin": float(args.affinity_upgrade_margin),
                "bucket_to_preferred_instances": affinity_calibration.get(
                    "bucket_to_preferred_instances", {}
                ),
                "bucket_preference_models": affinity_calibration.get(
                    "bucket_preference_models", {}
                ),
                "global_best_quality_models": affinity_calibration.get(
                    "global_best_quality_models", []
                ),
                "global_best_quality_instances": affinity_calibration.get(
                    "global_best_quality_instances", []
                ),
                "model_to_instances": affinity_calibration.get("model_to_instances", {}),
                "bucket_summary": affinity_calibration.get("bucket_summary", {}),
                "active_model_quality_summary": affinity_calibration.get(
                    "active_model_quality_summary", []
                ),
            }
        runs.append(run)

    result: Dict[str, Any] = {
        "mode": "router",
        "utilities": args.utilities,
        "feasible_slo_mode": args.feasible_slo_mode,
        "per_request_wait_logs": [str(path) for path in per_request_wait_logs],
        "response_map_paths": response_map_paths,
        "request_log_paths": request_log_paths,
        "runs": runs,
    }
    if prefill_tps_by_instance is not None:
        result["prefill_tps_by_instance"] = dict(prefill_tps_by_instance)
        result["prefill_tps_resolution"] = prefill_tps_resolution
    if service_rates_by_instance is not None:
        result["service_rates_rps_by_instance"] = dict(service_rates_by_instance)
        result["service_rates_rps_resolution"] = service_rates_resolution
    if (
        decode_tps_by_instance is not None
        and "hard_score_proxy" in normalized_utilities
    ):
        result["score_proxy"] = {
            "exact_score_reproduction": False,
            "target": "ttft",
            "formula": (
                "(effective_prefill_backlog_tokens + prompt_tokens) / "
                "prefill_tps + effective_decode_backlog_tokens / decode_tps + "
                "mean_decode_batch_ms"
            ),
            "decode_tps_by_instance": dict(decode_tps_by_instance),
            "mean_decode_batch_ms_by_instance": dict(
                mean_decode_batch_ms_by_instance or {}
            ),
            "metrics_resolution": score_proxy_metrics_resolution,
        }
    if decode_tps_by_instance is not None and "score" in normalized_utilities:
        result["score"] = {
            "published_fixed_lambda_argmax": True,
            "formula": (
                "quality_hat - lambda * (w_C * (c_output * output_hat) + "
                "w_L * (W + s * output_hat))"
            ),
            "waiting_time_adaptation": (
                "(effective_prefill_backlog_tokens + prompt_tokens) / "
                "prefill_tps + effective_decode_backlog_tokens / decode_tps"
            ),
            "decode_tps_by_instance": dict(decode_tps_by_instance),
            "mean_decode_batch_ms_by_instance": dict(
                mean_decode_batch_ms_by_instance or {}
            ),
            "metrics_resolution": score_proxy_metrics_resolution,
            "lambda_update": "fixed; paper does not specify controller",
        }
    if affinity_calibration is not None:
        result["instance_affinity_calibration"] = {
            "scored_root": affinity_calibration.get("scored_root"),
            "score_field": affinity_calibration.get("score_field"),
            "quality_epsilon": affinity_calibration.get("quality_epsilon"),
            "bucket_to_preferred_instances": affinity_calibration.get(
                "bucket_to_preferred_instances", {}
            ),
            "bucket_preference_models": affinity_calibration.get(
                "bucket_preference_models", {}
            ),
            "global_best_quality_models": affinity_calibration.get(
                "global_best_quality_models", []
            ),
            "global_best_quality_instances": affinity_calibration.get(
                "global_best_quality_instances", []
            ),
            "model_to_instances": affinity_calibration.get("model_to_instances", {}),
            "bucket_summary": affinity_calibration.get("bucket_summary", {}),
            "active_model_quality_summary": affinity_calibration.get(
                "active_model_quality_summary", []
            ),
        }
    return result

async def run_wait_gof_experiment(
    *,
    args: argparse.Namespace,
    requests: list[ExperimentRequest],
    instances: dict[str, InstanceClient],
    instance_costs: Dict[str, Dict[str, float]],
    instance_metadata: Dict[str, Any],
    response_map_base_path: Path,
    request_log_base_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    if not instances:
        raise ValueError("No instances provided for wait_gof.")
    if args.wait_eval_instance_id:
        if args.wait_eval_instance_id not in instances:
            raise ValueError(
                f"wait_eval_instance_id '{args.wait_eval_instance_id}' not found in instances."
            )
        wait_instances = {args.wait_eval_instance_id: instances[args.wait_eval_instance_id]}
    else:
        first_instance_id = next(iter(instances.keys()))
        wait_instances = {first_instance_id: instances[first_instance_id]}

    resolved_wait_logs = _resolve_per_request_wait_logs(args)
    if not resolved_wait_logs:
        raise ValueError(
            "wait_gof requires per-request wait logs. Provide --per-request-wait-log "
            "or set VLLM_PER_REQUEST_WAIT_LOG_PATH."
        )
    per_request_wait_logs = [path for path in resolved_wait_logs if path.exists()]
    if not per_request_wait_logs:
        raise FileNotFoundError(
            "wait_gof could not find per-request wait log files. Checked: "
            + ", ".join(str(path) for path in resolved_wait_logs)
        )

    resolved_estimators: list[tuple[str, WaitEstimatorCallable]] = []
    estimator_specs: list[tuple[str, str]] = []
    seen_estimator_names: set[str] = set()
    for estimator_spec in args.wait_estimators:
        estimator_name, estimator_fn = resolve_wait_estimator(estimator_spec)
        if estimator_name in seen_estimator_names:
            raise ValueError(
                f"Duplicate wait estimator name '{estimator_name}' after resolution. "
                "Estimator names must be unique in --wait-estimators."
            )
        seen_estimator_names.add(estimator_name)
        resolved_estimators.append((estimator_name, estimator_fn))
        estimator_specs.append((estimator_spec, estimator_name))

    if not resolved_estimators:
        raise ValueError("No wait estimators were resolved for wait_gof.")

    prefill_tps_by_instance: Optional[Dict[str, float]] = None
    prefill_tps_resolution: Optional[Dict[str, Dict[str, Any]]] = None
    service_rates_by_instance: Optional[Dict[str, float]] = None
    service_rates_resolution: Optional[Dict[str, Dict[str, Any]]] = None
    decode_tps_by_instance: Optional[Dict[str, float]] = None
    mean_decode_batch_ms_by_instance: Optional[Dict[str, float]] = None
    score_proxy_metrics_resolution: Optional[Dict[str, Dict[str, Any]]] = None
    if any(
        name
        in {
            PREFILL_TPS_TTFT_ESTIMATOR_NAME,
            SCORE_PROXY_TTFT_ESTIMATOR_NAME,
            "pk_mg1",
        }
        for name, _ in resolved_estimators
    ):
        prefill_tps_by_instance, prefill_tps_resolution = (
            _resolve_router_prefill_tps_by_instance(
                instances=wait_instances,
                overrides=getattr(args, "prefill_tps_overrides", None),
                calibrated_metrics=getattr(
                    args,
                    "calibrated_service_metrics",
                    None,
                ),
            )
        )
    if any(
        name == SCORE_PROXY_TTFT_ESTIMATOR_NAME
        for name, _ in resolved_estimators
    ):
        (
            decode_tps_by_instance,
            mean_decode_batch_ms_by_instance,
            score_proxy_metrics_resolution,
        ) = _resolve_score_proxy_metrics_by_instance(
            instances=wait_instances,
            calibrated_metrics=getattr(
                args,
                "calibrated_service_metrics",
                {},
            ),
        )
    if any(name == "pk_mg1" for name, _ in resolved_estimators):
        service_rates_by_instance, service_rates_resolution = (
            _resolve_router_service_rates_by_instance(instances=wait_instances)
        )

    offsets = _snapshot_log_offsets(per_request_wait_logs)
    primary_estimator_name, primary_estimator_fn = resolved_estimators[0]
    primary_key = primary_estimator_name.replace("/", "_").replace(":", "_")
    shared_response_map_path = _utility_response_map_path(
        response_map_base_path,
        primary_key if len(resolved_estimators) == 1 else "shared",
        multi_utility=len(resolved_estimators) > 1,
    )
    shared_request_log_path = _utility_response_map_path(
        request_log_base_path,
        "wait_gof_shared",
        multi_utility=True,
    )

    wait_estimator_contexts: Dict[str, Dict[str, Any]] = {}
    pk_trackers: Dict[str, PKOnlineStatsCollector] = {}
    for estimator_name, _ in resolved_estimators:
        estimator_context: Dict[str, Any] = {
            "wait_estimator_diagnostics_by_instance": {},
        }
        if estimator_name == PREFILL_TPS_TTFT_ESTIMATOR_NAME:
            if not prefill_tps_by_instance:
                raise ValueError(
                    f"{PREFILL_TPS_TTFT_ESTIMATOR_NAME} selected but no prefill TPS "
                    "mapping resolved."
                )
            estimator_context["prefill_tps_by_instance"] = dict(
                prefill_tps_by_instance
            )
        if estimator_name == SCORE_PROXY_TTFT_ESTIMATOR_NAME:
            if (
                not prefill_tps_by_instance
                or not decode_tps_by_instance
                or not mean_decode_batch_ms_by_instance
            ):
                raise ValueError(
                    f"{SCORE_PROXY_TTFT_ESTIMATOR_NAME} selected but calibrated "
                    "service metrics were not resolved."
                )
            estimator_context.update(
                {
                    "prefill_tps_by_instance": dict(prefill_tps_by_instance),
                    "decode_tps_by_instance": dict(decode_tps_by_instance),
                    "mean_decode_batch_ms_by_instance": dict(
                        mean_decode_batch_ms_by_instance
                    ),
                }
            )
        if estimator_name == "pk_mg1":
            if not prefill_tps_by_instance:
                raise ValueError(
                    "pk_mg1 selected but no prefill TPS mapping resolved."
                )
            if not service_rates_by_instance:
                raise ValueError(
                    "pk_mg1 selected but no service-rate priors resolved."
                )
            estimator_context["prefill_tps_by_instance"] = dict(
                prefill_tps_by_instance
            )
            pk_tracker = PKOnlineStatsCollector(
                instance_ids=list(wait_instances.keys()),
                per_request_wait_logs=per_request_wait_logs,
                arrival_ewma_alpha=float(args.pk_arrival_ewma_alpha),
                arrival_rate_window_s=float(args.pk_arrival_rate_window_s),
                service_rate_window_s=float(args.pk_service_rate_window_s),
                service_mu_prior_rps_by_instance=dict(service_rates_by_instance),
                min_service_samples=int(args.pk_min_service_samples),
                service_staleness_s=float(args.pk_service_staleness_s),
                rho_cap=float(args.pk_rho_cap),
                max_wait_ms=float(args.pk_max_wait_ms),
                default_service_s=float(args.pk_default_service_s),
            )
            estimator_context["pk_tracker"] = pk_tracker
            estimator_context["pk_last_diagnostics"] = {}
            pk_trackers[estimator_name] = pk_tracker
        wait_estimator_contexts[estimator_name] = estimator_context

    shared_run = await run_policy(
        utility_name="min_wait",
        requests=requests,
        instances=wait_instances,
        instance_costs=instance_costs,
        accuracy_model_path=args.accuracy_model_path,
        output_length_model_path=args.output_length_model_path,
        lambda_weight=0.0,
        delta_weight=1.0,
        worker_count=args.worker_count,
        max_queue_size=args.max_queue_size,
        request_rate_qps=args.request_rate_qps,
        arrival_process=args.arrival_process,
        arrival_seed=args.seed,
        mmpp2_rate_ratio=float(args.mmpp2_rate_ratio),
        mmpp2_high_fraction=float(args.mmpp2_high_fraction),
        mmpp2_correlation_time_s=float(args.mmpp2_correlation_time_s),
        decouple_arrivals=bool(args.decouple_arrivals),
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        tokenizer_id=args.tokenizer_id,
        tokenizer_mode=args.tokenizer_mode,
        system_prompt=args.system_prompt,
        chat_template_kwargs=args.chat_template_kwargs,
        response_map_path=str(shared_response_map_path),
        request_log_path=str(shared_request_log_path),
        request_id_prefix=f"wait-{primary_key}",
        route_strategy="utility",
        wait_estimator=primary_estimator_fn,
        wait_estimator_name=primary_estimator_name,
        wait_estimator_context=wait_estimator_contexts.get(primary_estimator_name),
        wait_estimators=resolved_estimators,
        wait_estimator_contexts=wait_estimator_contexts,
        include_unconditional_live_fetch=_should_include_unconditional_live_fetch(
            primary_estimator_name
        ),
        run_label=primary_estimator_name,
        enable_wait_time_polling=bool(args.enable_wait_time_polling),
        critical_wait_time_timeout_s=float(args.critical_wait_time_timeout_s),
        readiness_diagnostics=bool(args.readiness_diagnostics),
        readiness_predictor_path=args.readiness_predictor_path,
        close_instances_on_stop=False,
    )

    queue_map, ttft_map = _build_actual_wait_maps_from_logs(
        per_request_wait_logs,
        offsets=offsets,
    )

    runs: list[Dict[str, Any]] = []
    response_map_paths: Dict[str, str] = {}

    for estimator_spec, estimator_name in estimator_specs:
        estimator_key = estimator_name.replace("/", "_").replace(":", "_")
        estimator_response_map_path = shared_response_map_path
        response_map_paths[estimator_name] = str(estimator_response_map_path)

        estimator_per_request: list[Dict[str, Any]] = []
        for item in shared_run["per_request"]:
            projected = dict(item)
            estimate_ms: Optional[float] = None
            wait_estimates = item.get("wait_estimates_ms")
            if isinstance(wait_estimates, dict):
                candidate = wait_estimates.get(estimator_name)
                if isinstance(candidate, (int, float)):
                    estimate_ms = float(candidate)
            if estimate_ms is None and estimator_name == primary_estimator_name:
                candidate = item.get("wait_time_ms")
                if isinstance(candidate, (int, float)):
                    estimate_ms = float(candidate)

            projected["wait_time_ms"] = estimate_ms
            projected["wait_estimator"] = estimator_name

            estimator_per_request.append(projected)

        actual_metric_name = _wait_gof_actual_metric_name(estimator_name)
        actual_wait_map = ttft_map if actual_metric_name == "ttft_ms" else queue_map
        gof_summary = _augment_with_actual_waits(
            estimator_per_request,
            actual_wait_map,
            actual_metric_name=actual_metric_name,
        )
        matched_pairs = int(gof_summary.get("matched_pairs", 0))
        if matched_pairs < int(args.wait_gof_min_matched_pairs):
            raise RuntimeError(
                f"wait_gof matched_pairs={matched_pairs} for estimator "
                f"'{estimator_name}' is below --wait-gof-min-matched-pairs="
                f"{args.wait_gof_min_matched_pairs}."
            )

        run_summary = dict(shared_run["summary"])
        run_summary["actual_wait_fit"] = gof_summary
        run_summary["actual_metric_name"] = actual_metric_name
        actual_metric_definition = (
            "engine_core_first_token_ts_minus_queued_ts"
            if actual_metric_name == "ttft_ms"
            else "engine_core_scheduled_ts_minus_queued_ts"
        )
        run_summary["actual_metric_definition"] = actual_metric_definition
        run = {
            "label": estimator_name,
            "utility": shared_run["utility"],
            "route_strategy": shared_run["route_strategy"],
            "wait_estimator": estimator_name,
            "response_map_path": str(estimator_response_map_path),
            "request_log_path": shared_run.get("request_log_path"),
            "summary": run_summary,
            "per_request": estimator_per_request,
            "wait_estimator_spec": estimator_spec,
            "actual_wait_fit": gof_summary,
            "actual_metric_name": actual_metric_name,
            "actual_metric_definition": actual_metric_definition,
        }

        pk_tracker = pk_trackers.get(estimator_name)
        if pk_tracker is not None:
            run["pk_online_stats"] = pk_tracker.summary()

        predicted = [
            float(item["wait_time_ms"])
            for item in estimator_per_request
            if isinstance(item.get("wait_time_ms"), (int, float))
            and isinstance(item.get("actual_wait_time_ms"), (int, float))
        ]
        actual = [
            float(item["actual_wait_time_ms"])
            for item in estimator_per_request
            if isinstance(item.get("wait_time_ms"), (int, float))
            and isinstance(item.get("actual_wait_time_ms"), (int, float))
        ]
        plot_path = _plot_wait_fit(
            predicted=predicted,
            actual=actual,
            output_path=output_path.with_name(
                f"{output_path.stem}_wait_fit_{estimator_key}.png"
            ),
            title=(
                f"Wait estimator fit: {estimator_name} "
                f"(target={actual_metric_name})"
            ),
        )
        run["actual_wait_plot_path"] = plot_path
        runs.append(run)

    result = {
        "mode": "wait_gof",
        "wait_estimators": args.wait_estimators,
        "evaluated_instance_ids": list(wait_instances.keys()),
        "per_request_wait_logs": [str(path) for path in per_request_wait_logs],
        "response_map_paths": response_map_paths,
        "request_log_path": str(shared_request_log_path),
        "runs": runs,
    }
    if prefill_tps_by_instance is not None:
        result["prefill_tps_by_instance"] = dict(prefill_tps_by_instance)
        result["prefill_tps_resolution"] = prefill_tps_resolution
    if service_rates_by_instance is not None:
        result["service_rates_rps_by_instance"] = dict(service_rates_by_instance)
        result["service_rates_rps_resolution"] = service_rates_resolution
    if decode_tps_by_instance is not None:
        result["score_proxy_metrics"] = {
            "decode_tps_by_instance": dict(decode_tps_by_instance),
            "mean_decode_batch_ms_by_instance": dict(
                mean_decode_batch_ms_by_instance or {}
            ),
            "metrics_resolution": score_proxy_metrics_resolution,
        }
    return result

async def async_main(args: argparse.Namespace) -> None:
    output_path = args.output_path
    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_path = DEFAULT_OUTPUT_DIR / f"router_utility_experiment_{ts}.json"

    response_map_base_path = args.response_map_path
    if response_map_base_path is None:
        response_map_base_path = output_path.with_name(
            f"{output_path.stem}_response_map.log"
        )

    request_log_base_path = args.request_log_path
    if request_log_base_path is None:
        request_log_base_path = output_path.with_name(
            f"{output_path.stem}_predicted_waits.log"
        )

    selected_wait_estimators = {str(spec).strip().lower() for spec in args.wait_estimators}
    selected_router_utilities = {str(name).strip().lower() for name in args.utilities}
    include_pk_config = (
        (args.experiment in {"wait_gof", "all"} and "pk_mg1" in selected_wait_estimators)
        or (
            args.experiment in {"router", "all"}
            and bool({"hard_pk_mg1", "soft_pk_mg1"} & selected_router_utilities)
        )
    )
    include_prefill_tps_router_config = (
        args.experiment in {"router", "all"}
        and any(
            str(name).strip().lower()
            in {
                "hard_prefill_tps",
                "hard_score_proxy",
                "score",
                "soft_prefill_tps",
                "hard_pk_mg1",
                "soft_pk_mg1",
            }
            for name in args.utilities
        )
    )

    result: Dict[str, Any] = {
        "config": {
            "experiment": args.experiment,
            "num_requests": args.num_requests,
            "seed": args.seed,
            "request_rate_qps": args.request_rate_qps,
            "arrival_process": args.arrival_process,
            "mmpp2_rate_ratio": float(args.mmpp2_rate_ratio),
            "mmpp2_high_fraction": float(args.mmpp2_high_fraction),
            "mmpp2_correlation_time_s": float(args.mmpp2_correlation_time_s),
            "decouple_arrivals": bool(args.decouple_arrivals),
            "enable_wait_time_polling": bool(args.enable_wait_time_polling),
            "critical_wait_time_timeout_s": float(
                args.critical_wait_time_timeout_s
            ),
            "service_metrics_json": (
                str(args.service_metrics_json.expanduser().resolve())
                if args.service_metrics_json is not None
                else None
            ),
            "utilities": args.utilities,
            "wait_estimators": args.wait_estimators,
            "wait_gof_min_matched_pairs": args.wait_gof_min_matched_pairs,
            "slo_min_ms": args.slo_min_ms,
            "slo_max_ms": args.slo_max_ms,
            "queue_slo_min_ms": args.queue_slo_min_ms,
            "queue_slo_max_ms": args.queue_slo_max_ms,
            "ttft_slo_min_ms": args.ttft_slo_min_ms,
            "ttft_slo_max_ms": args.ttft_slo_max_ms,
            "ttft_slo_base_ms": args.ttft_slo_base_ms,
            "ttft_slo_per_prompt_token_ms": args.ttft_slo_per_prompt_token_ms,
            "ttft_slo_jitter_min": args.ttft_slo_jitter_min,
            "ttft_slo_jitter_max": args.ttft_slo_jitter_max,
            "feasible_slo_mode": args.feasible_slo_mode,
            "accuracy_model_path": args.accuracy_model_path,
            "output_length_model_path": args.output_length_model_path,
            "latency_warmup_requests": args.latency_warmup_requests,
            "methodology_calibration_json": args.methodology_calibration_json,
            "routebalance_predictor_path": args.routebalance_predictor_path,
            "routebalance_weights": list(args.routebalance_weights),
            "routebalance_batch_max_size": args.routebalance_batch_max_size,
            "routebalance_batch_wait_ms": args.routebalance_batch_wait_ms,
            "methodology_snapshot_max_age_ms": args.methodology_snapshot_max_age_ms,
            "lambda_weight": args.lambda_weight,
            "delta_weight": args.delta_weight,
            "score_cost_weight": args.score_cost_weight,
            "score_latency_weight": args.score_latency_weight,
            "score_total_cost_budget": args.score_total_cost_budget,
            "worker_count": args.worker_count,
            "max_queue_size": args.max_queue_size,
            "max_completion_tokens": args.max_completion_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "tokenizer_id": args.tokenizer_id,
            "tokenizer_mode": args.tokenizer_mode,
            "system_prompt": args.system_prompt,
            "chat_template_kwargs": args.chat_template_kwargs,
            "response_map_base_path": str(response_map_base_path),
            "request_log_base_path": str(request_log_base_path),
            "bucket_dir": str(args.bucket_dir.expanduser().resolve()),
            # Populated from the actual request source below. Batch-fit runs
            # have no prompt dependency; holdout caches may not exist yet.
            "bucket_files": [],
            "affinity_scored_root": str(args.affinity_scored_root.expanduser().resolve()),
            "affinity_quality_epsilon": float(args.affinity_quality_epsilon),
            "affinity_upgrade_margin": float(args.affinity_upgrade_margin),
            "holdout_prompts_per_bucket": args.holdout_prompts_per_bucket,
            "holdout_bucket_dir": str(args.holdout_bucket_dir.expanduser().resolve()),
            "holdout_start_index": args.holdout_start_index,
            "holdout_cache_dir": str(args.holdout_cache_dir.expanduser().resolve()),
            "holdout_context_length": args.holdout_context_length,
            "rebuild_holdout_cache": bool(args.rebuild_holdout_cache),
        },
    }
    if args.arrival_process == "mmpp2":
        mmpp2_derived = _derive_mmpp2_params(
            request_rate_qps=float(args.request_rate_qps),
            rate_ratio=float(args.mmpp2_rate_ratio),
            high_fraction=float(args.mmpp2_high_fraction),
            correlation_time_s=float(args.mmpp2_correlation_time_s),
        )
        result["config"]["mmpp2_derived"] = {
            "lambda_low_rps": float(mmpp2_derived.lambda_low_rps),
            "lambda_high_rps": float(mmpp2_derived.lambda_high_rps),
            "q_low_to_high_rps": float(mmpp2_derived.q_low_to_high_rps),
            "q_high_to_low_rps": float(mmpp2_derived.q_high_to_low_rps),
        }
    if include_prefill_tps_router_config:
        result["config"]["prefill_tps_defaults"] = dict(DEFAULT_PREFILL_TPS)
        result["config"]["prefill_tps_overrides"] = dict(
            getattr(args, "prefill_tps_overrides", {})
        )
    if include_pk_config:
        result["config"].update(
            {
                "pk_arrival_ewma_alpha": args.pk_arrival_ewma_alpha,
                "pk_arrival_rate_window_s": args.pk_arrival_rate_window_s,
                "pk_service_rate_window_s": args.pk_service_rate_window_s,
                "pk_min_service_samples": args.pk_min_service_samples,
                "pk_service_staleness_s": args.pk_service_staleness_s,
                "pk_rho_cap": args.pk_rho_cap,
                "pk_max_wait_ms": args.pk_max_wait_ms,
                "pk_default_service_s": args.pk_default_service_s,
                "pk_service_rate_rps_defaults": dict(DEFAULT_SERVICE_RATES_RPS),
            }
        )

    if args.experiment in {"router", "wait_gof", "all"}:
        requests, request_manifest, prompt_source_info = _build_request_set(args)
        result["config"]["prompt_source"] = prompt_source_info
        result["config"]["bucket_dir"] = prompt_source_info["effective_bucket_dir"]
        result["config"]["bucket_files"] = prompt_source_info.get("effective_bucket_files", [])

        instances, instance_costs, instance_metadata = load_instances(args.instances_config)
        result["config"]["instance_metadata"] = instance_metadata
        result["config"]["instance_costs"] = instance_costs
        result["request_set"] = {
            "num_requests": len(request_manifest),
            "slo_ms": _metric_summary([float(req.latency_slo_ms) for req in requests]),
            "latency_slo_ms": _metric_summary([float(req.latency_slo_ms) for req in requests]),
            "queue_slo_ms": _metric_summary([float(req.queue_slo_ms) for req in requests]),
            "ttft_slo_ms": _metric_summary([float(req.ttft_slo_ms) for req in requests]),
            "prompt_tokens": _metric_summary([float(req.prompt_tokens) for req in requests]),
            "requests": request_manifest,
        }

        try:
            if args.experiment in {"router", "all"}:
                result["router"] = await run_router_experiment(
                    args=args,
                    requests=requests,
                    instances=instances,
                    instance_costs=instance_costs,
                    instance_metadata=instance_metadata,
                    response_map_base_path=response_map_base_path,
                    request_log_base_path=request_log_base_path,
                )
            if args.experiment in {"wait_gof", "all"}:
                result["wait_gof"] = await run_wait_gof_experiment(
                    args=args,
                    requests=requests,
                    instances=instances,
                    instance_costs=instance_costs,
                    instance_metadata=instance_metadata,
                    response_map_base_path=response_map_base_path,
                    request_log_base_path=request_log_base_path,
                    output_path=output_path,
                )
        finally:
            close_tasks = [
                asyncio.to_thread(instance.close)
                for instance in instances.values()
            ]
            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)

    if args.experiment in {"batch_fit", "all"}:
        result["batch_fit"] = run_batch_fit_experiment(
            csv_paths=[path.expanduser() for path in args.batch_csv],
            train_csv_paths=[path.expanduser() for path in args.batch_train_csv],
            test_csv_paths=[path.expanduser() for path in args.batch_test_csv],
            output_dir=args.batch_fit_output_dir.expanduser(),
            stall_percentile=args.batch_fit_stall_percentile,
            huber_epsilon=args.batch_fit_huber_epsilon,
            max_plot_points=args.batch_fit_max_plot_points,
            feature_set=args.batch_fit_feature_set,
            nonnegative_coefficients=args.batch_fit_nonnegative,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    payload = {
        "output_path": str(output_path),
        "sections": sorted([k for k in result.keys() if k not in {"config", "request_set"}]),
    }
    print(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()

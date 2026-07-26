from __future__ import annotations

import io
from types import SimpleNamespace

import vllm.v1.engine

# The parser tests do not execute the native simulator. Source-only test
# environments do not have the optional extension built, but experiments.py
# imports the SHM client that references it.
try:
    from vllm.v1.engine import _scheduler_sim  # noqa: F401
except ImportError:
    vllm.v1.engine._scheduler_sim = SimpleNamespace()

from sfs_core.routing.pending_dispatch_ledger import PendingDispatch
from sfs_core.routing.wait_time_scheduler import WaitTimeResult
from scripts.eval import plot_router_wait_gof
from scripts.runs import experiments


class _MemoryPath:
    def __init__(self, text: str) -> None:
        self._text = text

    def exists(self) -> bool:
        return True

    def open(self, *args, **kwargs):
        return io.StringIO(self._text)

    def __str__(self) -> str:
        return "<memory>"


def test_wait_gof_uses_engine_core_queued_to_first_token():
    path = _MemoryPath(
        "vllm.per_request_latency request_id=chatcmpl-1 "
        "queue_ms=2.000000 ttft_s=0.125000 prefill_s=0.010000 "
        "decode_s=0.020000 inference_s=0.030000 e2e_s=0.150000 "
        "queued_ts_s=10.000000000 first_token_ts_s=10.012000000\n"
    )

    components = experiments._read_latency_components_from_logs([path])
    queue_map, ttft_map = experiments._build_actual_wait_maps_from_logs([path])

    assert components["chatcmpl-1"].frontend_ttft_ms == 125.0
    assert components["chatcmpl-1"].queued_ts_s == 10.0
    assert components["chatcmpl-1"].first_token_ts_s == 10.012
    assert queue_map["chatcmpl-1"] == 2.0
    assert abs(ttft_map["chatcmpl-1"] - 12.0) < 1e-9


def test_wait_gof_falls_back_to_queue_plus_prefill_for_legacy_logs():
    path = _MemoryPath(
        "vllm.per_request_latency request_id=chatcmpl-1 "
        "queue_ms=2.000000 prefill_s=0.010000 "
        "decode_s=0.020000 inference_s=0.030000 e2e_s=0.150000\n"
    )

    queue_map, ttft_map = experiments._build_actual_wait_maps_from_logs([path])
    plot_queue_map, plot_ttft_map, prefill_map = (
        plot_router_wait_gof._read_actual_wait_logs([path])
    )

    assert queue_map["chatcmpl-1"] == 2.0
    assert ttft_map["chatcmpl-1"] == 12.0
    assert plot_queue_map["chatcmpl-1"] == 2.0
    assert plot_ttft_map["chatcmpl-1"] == 12.0
    assert prefill_map["chatcmpl-1"] == 10.0


def test_wait_gof_plotting_uses_queue_plus_prefill():
    path = _MemoryPath(
        "vllm.per_request_latency request_id=chatcmpl-1 "
        "queue_ms=2.000000 ttft_s=0.125000 prefill_s=0.010000 "
        "decode_s=0.020000 inference_s=0.030000 e2e_s=0.150000\n"
    )

    _, ttft_map, _ = plot_router_wait_gof._read_actual_wait_logs([path])

    assert ttft_map["chatcmpl-1"] == 12.0


def test_readiness_diagnostics_record_only_router_visible_inputs():
    scheduler = experiments.CollectingWaitTimeScheduler.__new__(
        experiments.CollectingWaitTimeScheduler
    )
    scheduler._readiness_diagnostics = True
    wait_record = WaitTimeResult(
        instance_id="instance-a",
        wait_ms=12.0,
        fetched_at_s=1.0,
    )
    pending = (
        PendingDispatch(
            engine_request_id="earlier",
            prompt_tokens=100,
            predicted_output_tokens=10.0,
            completion_cap=20,
        ),
    )

    scheduler._attach_readiness_predictor_inputs(
        wait_record=wait_record,
        prompt_tokens=321,
        pending_dispatches=pending,
    )

    assert wait_record.raw_payload["_readiness_predictor_inputs"] == {
        "prompt_tokens": 321,
        "pending_dispatch_count": 1,
    }

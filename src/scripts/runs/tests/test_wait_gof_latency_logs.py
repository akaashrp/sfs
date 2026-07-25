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


def test_wait_gof_prefers_exact_ttft_over_queue_prefill_sum():
    path = _MemoryPath(
        "vllm.per_request_latency request_id=chatcmpl-1 "
        "queue_ms=2.000000 ttft_s=0.125000 prefill_s=0.010000 "
        "decode_s=0.020000 inference_s=0.030000 e2e_s=0.150000\n"
    )

    components = experiments._read_latency_components_from_logs([path])
    queue_map, ttft_map = experiments._build_actual_wait_maps_from_logs([path])

    assert components["chatcmpl-1"].ttft_ms == 125.0
    assert queue_map["chatcmpl-1"] == 2.0
    assert ttft_map["chatcmpl-1"] == 125.0


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


def test_wait_gof_plotting_prefers_exact_ttft():
    path = _MemoryPath(
        "vllm.per_request_latency request_id=chatcmpl-1 "
        "queue_ms=2.000000 ttft_s=0.125000 prefill_s=0.010000 "
        "decode_s=0.020000 inference_s=0.030000 e2e_s=0.150000\n"
    )

    _, ttft_map, _ = plot_router_wait_gof._read_actual_wait_logs([path])

    assert ttft_map["chatcmpl-1"] == 125.0

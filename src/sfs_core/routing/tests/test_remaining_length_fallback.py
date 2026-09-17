"""Router pending-dispatch overlay: missing prediction falls back to the table median only when tables are attached."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import vllm.v1.engine

try:
    from vllm.v1.engine import _scheduler_sim  # noqa: F401
except ImportError:
    vllm.v1.engine._scheduler_sim = SimpleNamespace()

from vllm.v1.core.sched.remaining_length import PROMPT_BIN_EDGES, RemainingLengthTable
from sfs_core.routing.pending_dispatch_ledger import PendingDispatchLedger
from sfs_core.routing.wait_time_scheduler import WaitTimeScheduler


def _table(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"schema_version": 1, "model": "m", "prompt_bin_edges": list(PROMPT_BIN_EDGES),
                                "min_support": 8, "cap": 8192,
                                "bins": {"0": [10, 20, 30, 40, 50, 60, 70, 80], "all": [1, 2, 3, 4, 5, 6, 7, 8, 9]}}))
    return RemainingLengthTable.load(path)


def _scheduler(tables):
    scheduler = WaitTimeScheduler.__new__(WaitTimeScheduler)
    scheduler._routing_state_lock = asyncio.Lock()
    scheduler._pending_dispatch_ledger = PendingDispatchLedger(("a", "b"))
    scheduler._remaining_length_tables = tables
    return scheduler


def _reserve(scheduler, predicted, prompt_tokens=16, cap=8192):
    scheduler._reserve_pending_dispatch(instance_id="a", engine_request_id="chatcmpl-1", prompt_tokens=prompt_tokens,
                                        predicted_output_tokens=predicted, completion_cap=cap)
    return scheduler._pending_dispatch_ledger.unobserved_for_instance("a")[0].predicted_output_tokens


def test_default_stays_one_token_without_tables():
    assert _reserve(_scheduler({}), 1.0) == 1.0
    assert _reserve(_scheduler({}), 37.5) == 37.5


def test_missing_prediction_uses_prompt_bin_median_with_tables(tmp_path):
    tables = {"a": _table(tmp_path)}
    assert _reserve(_scheduler(tables), 1.0) == 45.0            # median of bin 0, linear interpolation
    assert _reserve(_scheduler(tables), 1.0, prompt_tokens=300) == 5.0   # bin 1 absent -> model level
    assert _reserve(_scheduler(tables), 1.0, cap=12) == 12.0     # cap-aware
    assert _reserve(_scheduler(tables), 37.5) == 37.5            # real predictions untouched
    other = _scheduler({"b": _table(tmp_path)})
    assert _reserve(other, 1.0) == 1.0                            # no table for this instance

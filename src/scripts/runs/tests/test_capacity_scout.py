import asyncio
from dataclasses import dataclass
import json
from types import SimpleNamespace

import pytest

from scripts.runs.capacity_scout import TrialMonitor, classify_trial, next_bracket_rate, freeze_loads, plan_scout_trial
from scripts.runs.ministral3_methodology_stage import length_stratified_requests, smoke_requests


def events(arrival_rate=4, completion_rate=4, drain=True):
    rows = [{"event": "arrival", "t": i/arrival_rate} for i in range(int(arrival_rate*360))]
    rows += [{"event": "completion", "t": 10+i/completion_rate, "error": None}
             for i in range(int(completion_rate*350))]
    if drain:
        rows += [{"event": "completion", "t": 400+i/10, "error": None}
                 for i in range(max(0, int(arrival_rate*360-completion_rate*350)))]
    rows += [{"event": "arrivals_end", "t": 360, "reason": "duration_limit"},
             {"event": "sample", "t": 240, "engines": {"a": {}, "b": {}, "c": {}}}]
    return sorted(rows, key=lambda row: row["t"])


def test_classification_excludes_drain_and_initial_ramp():
    assert classify_trial(events(), requested_qps=4)["classification"] == "stable"
    result = classify_trial(events(completion_rate=2), requested_qps=4)
    assert result["classification"] == "unstable"
    assert result["arrival_window_throughput_qps"] == 2
    assert result["backlog_growth_qps"] == 2
    assert result["num_completions_total"] == result["num_arrivals"]


def test_shortfall_and_telemetry_failure_are_not_capacity():
    assert classify_trial(events(), requested_qps=8)["reason"] == "client_arrival_rate_mismatch"
    broken = events()+[{"event": "sample_error", "t": 250, "error": "unavailable"}]
    assert classify_trial(broken, requested_qps=4)["classification"] == "inconclusive"
    short = [e for e in events() if e["t"] < 150]
    short.append({"event": "arrivals_end", "t": 150, "reason": "backlog_limit"})
    assert classify_trial(short, requested_qps=4)["classification"] == "inconclusive"


def test_bracket_has_no_assumed_stable_endpoint_and_requires_narrowing():
    assert next_bracket_rate([]) == 2
    assert next_bracket_rate([{"classification": "unstable", "requested_qps": 2}]) == 1
    trials = [{"classification": "stable", "requested_qps": 4}]
    assert next_bracket_rate(trials) == 8
    trials.append({"classification": "unstable", "requested_qps": 8})
    assert next_bracket_rate(trials) == 6
    with pytest.raises(ValueError, match="narrow"):
        freeze_loads(trials)
    trials.append({"classification": "stable", "requested_qps": 7.75})
    result = freeze_loads(trials)
    assert result["K_M"] == 8
    assert result["qps_values"] == [5.2, 6.8, 7.6, 8.4]
    with pytest.raises(ValueError, match="Nonmonotonic"):
        next_bracket_rate(trials+[{"classification": "stable", "requested_qps": 9}])


def test_length_probes_are_unique_shuffled_and_span_the_pool():
    rows = [SimpleNamespace(prompt_tokens=i+1, request_id=str(i)) for i in range(100)]
    chosen = length_stratified_requests(rows)
    assert len({row.request_id for row in chosen}) == 64
    assert min(row.prompt_tokens for row in chosen) == 1
    assert max(row.prompt_tokens for row in chosen) == 100
    assert chosen != sorted(chosen, key=lambda row: row.prompt_tokens)


def test_monitor_counts_response_lifetime_and_preserves_failure(tmp_path):
    async def exercise():
        import time
        monitor = TrialMonitor(tmp_path/"events.jsonl", duration_s=360, max_outstanding=1)
        snapshot = SimpleNamespace(metadata=lambda: {"num_running": 0, "num_waiting": 0})
        async def state():
            return snapshot
        await monitor.start(time.perf_counter(), SimpleNamespace(_queue=asyncio.Queue()),
                            {"a": SimpleNamespace(refresh_baseline_state=state)})
        future = asyncio.get_running_loop().create_future()
        monitor.arrived(SimpleNamespace(request_id="r", bucket="alpaca"), future, time.perf_counter())
        assert monitor.stop_reason() == "backlog_limit"
        monitor.end_arrivals()
        future.set_result({"completed_perf": time.perf_counter(), "error": "failed"})
        await asyncio.sleep(0)
        assert monitor.stop_reason() == "request_failure"
        await monitor.close()
        rows = [json.loads(line) for line in (tmp_path/"events.jsonl").read_text().splitlines()]
        assert sum(row["event"] == "arrival" for row in rows) == 1
        assert next(row for row in rows if row["event"] == "completion")["error"] == "failed"
        assert rows[-1]["outstanding"] == 0
    asyncio.run(exercise())
def test_early_backlog_guard_refines_search_without_claiming_capacity():
    early = [e for e in events(arrival_rate=16, completion_rate=4) if e["t"] < 90]
    early.append({"event": "arrivals_end", "t": 90, "reason": "backlog_limit"})
    result = classify_trial(early, requested_qps=16)
    assert result["reason"] == "insufficient_sustained_observation"
    trials = [{"classification": "stable", "requested_qps": 4},
              {"classification": "inconclusive", "requested_qps": 8,
               "reason": "insufficient_sustained_observation", "stop_reason": "backlog_limit"}]
    assert next_bracket_rate(trials) == 6
    with pytest.raises(ValueError, match="measured transition bracket"):
        freeze_loads(trials)
    trials.append({"classification": "unstable", "requested_qps": 6})
    assert next_bracket_rate(trials) == 5


def test_resume_preserves_pending_longer_observation_and_five_percent_gate():
    trials = [{"classification": "stable", "requested_qps": 9},
              {"classification": "unstable", "requested_qps": 9.5},
              {"classification": "inconclusive", "requested_qps": 9.25,
               "reason": "borderline_or_nonstationary_windows", "stop_reason": "duration_limit"}]
    with pytest.raises(ValueError, match="narrow"):
        freeze_loads(trials)
    assert plan_scout_trial(trials) == {"status": "MEASURE", "qps": 9.25, "duration_s": 600}
    assert plan_scout_trial(trials+[trials[-1]])["status"] == "NEEDS_REVIEW"
    trials.append({"classification": "stable", "requested_qps": 9.25})
    assert plan_scout_trial(trials)["status"] == "BRACKET_READY"
    assert freeze_loads(trials)["stable_qps"] == 9.25

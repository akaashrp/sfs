"""Arrival-window accounting and bounded shortest-queue capacity bracketing.

This module deliberately does not interpret standalone service-rate sums as
router capacity. All classifications use actual arrivals and completions.
"""
from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
import time


def classify_trial(events, *, requested_qps, warmup_s=120, window_s=60,
                   min_windows=3, min_completions=100, arrival_tolerance=.10):
    arrivals = [e["t"] for e in events if e["event"] == "arrival"]
    completions = [e["t"] for e in events if e["event"] == "completion" and not e.get("error")]
    ends = [e for e in events if e["event"] == "arrivals_end"]
    errors = [e for e in events if e["event"] == "completion" and e.get("error")]
    out = {"classification": "inconclusive", "requested_qps": requested_qps,
           "window_s": window_s, "warmup_s": warmup_s, "windows": [],
           "num_arrivals": len(arrivals), "num_completions_total": len(completions)}
    if not ends or errors:
        return {**out, "reason": "missing_arrival_end_or_request_failure"}
    end = ends[-1]["t"]
    out["arrival_duration_s"] = end
    out["stop_reason"] = ends[-1]["reason"]
    if end <= warmup_s:
        return {**out, "reason": "insufficient_sustained_observation"}
    if any(e["event"] == "sample_error" and warmup_s <= e["t"] <= end for e in events):
        return {**out, "reason": "missing_engine_telemetry"}
    if not any(e["event"] == "sample" and warmup_s <= e["t"] <= end for e in events):
        return {**out, "reason": "missing_engine_telemetry"}
    for start in range(int(warmup_s), int(end - window_s) + 1, int(window_s)):
        stop = start + window_s
        a = sum(start <= t < stop for t in arrivals)
        c = sum(start <= t < stop for t in completions)
        b0 = sum(t < start for t in arrivals) - sum(t < start for t in completions)
        b1 = sum(t < stop for t in arrivals) - sum(t < stop for t in completions)
        out["windows"].append({"start_s": start, "end_s": stop,
            "arrivals": a, "completions": c, "arrival_qps": a / window_s,
            "completion_qps": c / window_s, "backlog_start": b0, "backlog_end": b1,
            "backlog_growth_qps": (b1-b0) / window_s})
    windows = out["windows"][-min_windows:]
    if len(windows) < min_windows or sum(w["completions"] for w in windows) < min_completions:
        return {**out, "reason": "insufficient_sustained_observation"}
    a = sum(w["arrivals"] for w in windows)
    c = sum(w["completions"] for w in windows)
    duration = window_s * len(windows)
    realized = a / duration
    ratio = c / max(a, 1)
    growth = (windows[-1]["backlog_end"] - windows[0]["backlog_start"]) / duration
    out.update(realized_qps=realized, arrival_window_throughput_qps=c / duration,
               completion_arrival_ratio=ratio, backlog_growth_qps=growth)
    if abs(realized / requested_qps - 1) > arrival_tolerance:
        return {**out, "reason": "client_arrival_rate_mismatch"}
    bad = [w["completions"] < .95 * w["arrivals"] and
           w["backlog_growth_qps"] > .05 * realized for w in windows]
    if all(bad) and ratio < .95 and growth > .05 * realized:
        return {**out, "classification": "unstable", "reason": "sustained_backlog_growth_during_arrivals"}
    if ratio >= .95 and growth <= .05 * realized and sum(bad) <= 1:
        return {**out, "classification": "stable", "reason": "arrival_window_flow_balance"}
    return {**out, "reason": "borderline_or_nonstationary_windows"}


def next_bracket_rate(trials, *, initial_qps=2.0, max_qps=32.0, relative_width=.05):
    """Double/halve absolute rates, then bisect; no invented stable endpoints."""
    valid = [t for t in trials if t["classification"] in {"stable", "unstable"}]
    stable = [t["requested_qps"] for t in valid if t["classification"] == "stable"]
    unstable = [t["requested_qps"] for t in valid if t["classification"] == "unstable"]
    # Hitting the outstanding-work cap before enough windows have elapsed is
    # a search bound, never a measured unstable endpoint. Refine below that
    # guarded rate instead of repeating an identically capped trial.
    guarded = [t["requested_qps"] for t in trials if
               t.get("classification") == "inconclusive" and t.get("stop_reason") == "backlog_limit"
               and t.get("reason") == "insufficient_sustained_observation"
               and (not stable or t["requested_qps"] > max(stable))]
    if stable and unstable:
        low, high = max(stable), min(unstable)
        if low >= high:
            raise ValueError("Nonmonotonic scout results: stable rate overlaps unstable bracket")
        return None if (high-low)/low <= relative_width else (low+high)/2
    if stable:
        if guarded:
            return (max(stable)+min(guarded))/2
        value = max(stable)*2
        if value > max_qps:
            raise ValueError("No unstable endpoint within the declared scout maximum")
        return value
    if unstable or guarded:
        value = min(unstable or guarded)/2
        if value < .125:
            raise ValueError("No stable endpoint above the declared scout minimum")
        return value
    return float(initial_qps)


def freeze_loads(trials):
    if next_bracket_rate(trials) is not None:
        raise ValueError("A sufficiently narrow measured transition bracket is required")
    low = max(t["requested_qps"] for t in trials if t["classification"] == "stable")
    high = min(t["requested_qps"] for t in trials if t["classification"] == "unstable")
    # The upper bracket edge provides an observed overload reference. With a
    # <=5% bracket, 1.05*K is at most 10.25% above its stable lower endpoint.
    return {"stable_qps": low, "unstable_qps": high, "K_M": high,
            "reference_rule": "smallest_observed_unstable_rate_in_5_percent_bracket",
            "fractions": [.65, .85, .95, 1.05],
            "qps_values": [round(f * high, 6) for f in (.65, .85, .95, 1.05)]}


def plan_scout_trial(trials, *, duration_s=360):
    """Resume the measured bracket, including a pending longer observation."""
    rate = next_bracket_rate(trials)
    if rate is None:
        return {"status": "BRACKET_READY"}
    inconclusive = [t for t in trials if t["requested_qps"] == rate and
                    t["classification"] == "inconclusive" and
                    t.get("stop_reason") != "backlog_limit"]
    if any(t.get("reason") in {"client_arrival_rate_mismatch", "missing_engine_telemetry",
                               "missing_arrival_end_or_request_failure"} for t in inconclusive):
        return {"status": "NEEDS_REVIEW", "reason": "instrumentation_or_request_failure", "qps": rate}
    if len(inconclusive) >= 2:
        return {"status": "NEEDS_REVIEW", "reason": "repeated_inconclusive_observation", "qps": rate}
    return {"status": "MEASURE", "qps": rate,
            "duration_s": max(duration_s, 600) if inconclusive else duration_s}


class TrialMonitor:
    """Keep original arrival timestamps, cap buildup, and sample observed state."""
    def __init__(self, path, *, duration_s=360, max_outstanding=1024, sample_s=2):
        self.path = Path(path)
        self.duration_s = float(duration_s)
        self.max_outstanding = int(max_outstanding)
        self.sample_s = float(sample_s)
        self.events = []
        self.arrivals = self.completed = self.errors = 0
        self.task = None
        self.origin = None
        self.ended = False

    def emit(self, event, *, perf=None, **fields):
        row = {"event": event, "t": (time.perf_counter() if perf is None else perf)-self.origin, **fields}
        self.events.append(row)
        # Match the experiment harness's deferred sidecars. Lustre writes in
        # this callback can throttle the very arrival stream we are measuring.

    async def start(self, origin, scheduler, instances):
        self.origin = origin
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("x")
        self.emit("start", perf=origin, duration_s=self.duration_s,
                  max_outstanding=self.max_outstanding)

        async def sample():
            while True:
                try:
                    snapshots = await asyncio.gather(*(x.refresh_baseline_state() for x in instances.values()))
                    engines = {key: value.metadata() for key, value in zip(instances, snapshots)}
                    self.emit("sample", outstanding=self.arrivals-self.completed,
                              router_queue=scheduler._queue.qsize(), engines=engines)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.emit("sample_error", error=f"{type(exc).__name__}: {exc}")
                await asyncio.sleep(self.sample_s)
        self.task = asyncio.create_task(sample())

    def stop_reason(self):
        if self.errors:
            return "request_failure"
        if self.arrivals-self.completed >= self.max_outstanding:
            return "backlog_limit"
        if time.perf_counter()-self.origin >= self.duration_s:
            return "duration_limit"
        return None

    def arrived(self, request, future, perf):
        self.arrivals += 1
        self.emit("arrival", perf=perf, request_id=request.request_id, bucket=request.bucket)
        def done(result):
            self.completed += 1
            if result.cancelled():
                self.errors += 1
                self.emit("completion", request_id=request.request_id, error="cancelled")
                return
            try:
                row = result.result()
            except Exception as exc:
                row = {"error": str(exc)}
            self.errors += bool(row.get("error"))
            self.emit("completion", perf=row.get("completed_perf"), request_id=request.request_id,
                      instance_id=row.get("instance_id"), dispatch_perf=row.get("dispatch_perf"),
                      error=row.get("error"))
        future.add_done_callback(done)

    def end_arrivals(self):
        if not self.ended:
            self.ended = True
            self.emit("arrivals_end", reason=self.stop_reason() or "request_budget")

    async def close(self):
        self.end_arrivals()
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        self.emit("drain_end", outstanding=self.arrivals-self.completed, failures=self.errors)
        def save():
            self.stream.writelines(json.dumps(row, allow_nan=False)+"\n" for row in self.events)
            self.stream.close()
        await asyncio.to_thread(save)

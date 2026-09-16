"""Drift-free arrival generation: absolute schedule, unchanged seeded draws."""

import asyncio
import random

import pytest

from scripts.runs import experiments
from scripts.runs.tests.test_methodology_baselines import (  # noqa: F401 (fixtures)
    FakeClient, _run_kwargs, calibration_path, harness,
)


class LaggingLoop:
    """Fake clock whose every sleep wakes up `lag_s` late, like a busy event loop."""

    def __init__(self, lag_s):
        self.now = 100.0
        self.lag_s = lag_s
        self.requested = []

    async def sleep(self, delay_s):
        self.requested.append(delay_s)
        self.now += delay_s + self.lag_s
        await asyncio.sleep(0)


def test_late_wakeups_do_not_accumulate_into_the_timeline():
    loop = LaggingLoop(lag_s=0.05)
    schedule = experiments._ArrivalSchedule(clock=lambda: loop.now, sleep=loop.sleep)
    gaps = [0.1] * 40

    async def arrivals():
        schedule.start()
        origin = loop.now
        for gap in gaps:
            await schedule.wait(gap)
            yield loop.now - origin

    realized = asyncio.run(_collect(arrivals()))
    # Relative sleeps would realize sum(gaps) + 40 lags (6.0 s); the absolute
    # schedule stays within a single wakeup lag of every target.
    assert realized[-1] == pytest.approx(sum(gaps) + loop.lag_s)
    for index, offset in enumerate(realized):
        assert 0 <= offset - (index + 1) * 0.1 <= loop.lag_s + 1e-9
    # Once behind, the next sleep is shortened (never negative) to catch up.
    assert loop.requested[0] == pytest.approx(0.1)
    assert all(r == pytest.approx(0.05) for r in loop.requested[1:])


def test_nonpositive_gaps_do_not_sleep_or_move_the_schedule():
    loop = LaggingLoop(lag_s=0.05)
    schedule = experiments._ArrivalSchedule(clock=lambda: loop.now, sleep=loop.sleep)

    async def run():
        await schedule.wait(0.0)
        await schedule.wait(-1.0)
        await schedule.wait(0.2)

    asyncio.run(run())
    assert loop.requested == [pytest.approx(0.2)]
    assert schedule.offset_s == pytest.approx(0.2)


async def _collect(iterator):
    return [item async for item in iterator]


@pytest.mark.parametrize("decouple_arrivals", [False, True])
def test_run_policy_uses_absolute_schedule_with_unchanged_seeded_gaps(
    decouple_arrivals, harness, calibration_path, tmp_path, monkeypatch,
):
    loop = LaggingLoop(lag_s=0.01)
    offsets = []

    class RecordingSchedule(experiments._ArrivalSchedule):
        def __init__(self):
            super().__init__(clock=lambda: loop.now, sleep=loop.sleep)

        async def wait(self, interarrival_s):
            await super().wait(interarrival_s)
            offsets.append(self.offset_s)

    monkeypatch.setattr(experiments, "_ArrivalSchedule", RecordingSchedule)
    clients = {client.instance_id: client for client in (FakeClient(0), FakeClient(1))}
    kwargs = _run_kwargs("round_robin", clients, calibration_path, tmp_path)
    kwargs.update(accuracy_model_path=None, output_length_model_path=None,
                  enable_wait_time_polling=False, decouple_arrivals=decouple_arrivals,
                  request_rate_qps=50.0, arrival_process="poisson", arrival_seed=11)
    result = asyncio.run(asyncio.wait_for(experiments.run_policy(**kwargs), timeout=10))
    assert result["summary"]["succeeded_requests"] == 3
    assert result["arrival_process"] == "poisson"
    assert result["arrival_timing"] == experiments.ARRIVAL_TIMING == "absolute_schedule"
    # Same RNG object, same draw order as the relative-sleep generator.
    rng = random.Random(11)
    expected, total = [], 0.0
    for _ in range(2):
        total += rng.expovariate(50.0)
        expected.append(total)
    assert offsets == pytest.approx(expected)
    assert loop.requested == pytest.approx([expected[0], expected[1] - expected[0] - loop.lag_s])

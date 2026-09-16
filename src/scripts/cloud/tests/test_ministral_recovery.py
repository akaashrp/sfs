import asyncio
from copy import deepcopy
import time

import pytest

from sfs_core.routing.ministral_methodology_scheduler import MinistralMethodologyScheduler
from sfs_core.routing.methodology_scheduler import MethodologyScheduler
from sfs_core.routing.tests.test_methodology_scheduler import (
    FakeClient, FakeCalibration, FakePredictor, queued, snapshot_payload, request_state)
from sfs_core.routing.methodology_snapshot import parse_baseline_snapshot
from sfs_core.routing import wait_time_scheduler
from scripts.runs.measured_audit import require_complete_ttft


class Client(FakeClient):
    async def refresh_baseline_state(self):
        self.snapshot_calls += 1
        return parse_baseline_snapshot(self.raw, observed_at=time.monotonic())


def state(*, busy=False, age=0, version=1):
    result = snapshot_payload(requests={'old': request_state('old')} if busy else {}, version=version)
    result['created_at'] = time.monotonic()-age
    return result


def scheduler(monkeypatch, tmp_path, policy='mooncake_prefill', **kwargs):
    monkeypatch.setattr(wait_time_scheduler, 'load_tokenizer', lambda *a, **k: None)
    clients = {m: Client(m) for m in ('ministral3-3b', 'ministral3-8b')}
    for c in clients.values(): c.raw = state()
    calibration = FakeCalibration(); calibration.speeds = {m: 1. for m in clients}
    predictor = FakePredictor() if policy == 'routebalance' else None
    if predictor: predictor.model_labels = tuple(clients)
    return MinistralMethodologyScheduler(clients, policy=policy, calibration=calibration,
        predictor=predictor, snapshot_retry_interval_s=.005,
        request_log_path=str(tmp_path/'requests.jsonl'), response_map_path=str(tmp_path/'responses.jsonl'),
        instance_costs={m: {'prompt': 1., 'output': 2.} for m in clients}, **kwargs), clients


@pytest.mark.parametrize('policy', ['mooncake_prefill', 'routebalance'])
def test_long_busy_iteration_dispatches_once_without_freshness_wait(monkeypatch, tmp_path, policy):
    async def run():
        sched, clients = scheduler(monkeypatch, tmp_path, policy)
        for key, c in clients.items():
            c.raw = state(busy=True, age=30)
            sched.lifetime.reserve('old-'+key, key, prompt_tokens=8, predicted_output_tokens=20)
            sched._engine_ids['old-'+key] = 'old'
        req = queued('new')
        await sched._dispatch_batch([req])
        await sched.drain()
        result = await req.payload['_completion_future']
        assert 'error' not in result
        assert sum(len(c.submissions) for c in clients.values()) == 1
        assert all(c.snapshot_calls == 1 for c in clients.values())
        assert sched._batch_id == 1 and sched._batch_sizes == {1: 1}
        assert all(x['snapshot']['snapshot_version'] == 1 for x in result['methodology_terms']['candidates'].values())
        assert sched.run_metadata()['snapshot_age_gate_ms'] is None
        assert sched.run_metadata()['snapshot_dispatch_wait_s'] == 0
        # Old work remains owned until completion, irrespective of its age.
        assert sum(sched.lifetime.unfinished_counts().values()) == 2
    asyncio.run(run())


def test_missing_publication_fails_without_dispatch(monkeypatch, tmp_path):
    async def run():
        sched, clients = scheduler(monkeypatch, tmp_path)
        async def broken():
            raise RuntimeError('Failed to read consistent baseline scheduler snapshot')
        next(iter(clients.values())).refresh_baseline_state = broken
        with pytest.raises(RuntimeError, match='consistent'):
            await sched._dispatch_batch([queued('new')])
        assert all(not c.submissions for c in clients.values())
        assert not sched._routing_state_lock.locked()
    asyncio.run(run())


def test_version_regression_is_not_retried(monkeypatch, tmp_path):
    async def run():
        sched, clients = scheduler(monkeypatch, tmp_path)
        sched._versions = {m: 5 for m in clients}
        with pytest.raises(RuntimeError, match='version regressed'):
            await sched._dispatch_batch([queued('new')])
        assert all(c.snapshot_calls == 1 and not c.submissions for c in clients.values())
    asyncio.run(run())


def test_cancellation_of_pending_transport_releases_lock(monkeypatch, tmp_path):
    async def run():
        sched, clients = scheduler(monkeypatch, tmp_path)
        gate = asyncio.Event()
        async def hung_transport():
            await gate.wait()
        for c in clients.values(): c.refresh_baseline_state = hung_transport
        task = asyncio.create_task(sched._dispatch_batch([queued('new')]))
        await asyncio.sleep(.015)
        task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
        assert not sched._routing_state_lock.locked()
        assert all(not c.submissions for c in clients.values())
    asyncio.run(run())


def test_qwen_cannot_enter_ministral_adapter():
    with pytest.raises(ValueError,match='only Ministral'):
        MinistralMethodologyScheduler({'qwen':FakeClient('qwen3-8b')},policy='mooncake_prefill',calibration=FakeCalibration())


@pytest.mark.parametrize('value', [None, float('nan'), -1, True])
def test_ttft_row_failure_cannot_be_hidden_by_summary(value):
    with pytest.raises(ValueError,match='TTFT'):
        require_complete_ttft({'summary':{'system_entry_e2e_ttft_slo_missing_count':0},
                               'per_request':[{'system_entry_e2e_ttft_ms':value}]})


def test_old_misspelled_summary_key_cannot_pass():
    with pytest.raises(ValueError,match='TTFT'):
        require_complete_ttft({'summary':{'system_entry_e2e_ttft_missing_count':0},
                               'per_request':[{'system_entry_e2e_ttft_ms':12.}]})


def test_ministral_wrapper_restores_shared_scheduler_after_error(monkeypatch):
    from types import SimpleNamespace
    from scripts.runs import ministral3_reliable as reliable
    from sfs_core.routing import methodology_scheduler as shared
    async def failing(**kwargs):
        assert shared.MethodologyScheduler is MinistralMethodologyScheduler
        raise RuntimeError('GPU error fixture')
    with pytest.raises(RuntimeError,match='GPU error fixture'):
        asyncio.run(reliable.run_router_experiment(args=SimpleNamespace(),requests=[],
            instances={'a':SimpleNamespace(model_id='ministral3-3b')},_original=failing))
    assert shared.MethodologyScheduler is MethodologyScheduler


def test_checkpoint_contains_the_quality_consumer_request_map_key(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from scripts.cloud import worker
    from scripts.runs import ministral3_reliable, ministral3_methodology_stage
    from scripts.eval.augment_router_actual_accuracy import _extract_holdout_prompts_per_bucket
    async def drained(*a,**k):pass
    async def router(**kwargs):
        monitor=kwargs['trial_monitor']
        assert monitor is not None and monitor.max_outstanding == len(kwargs['requests'])+1
        return {'runs':[]}
    monkeypatch.setattr(ministral3_methodology_stage,'wait_drained',drained)
    monkeypatch.setattr(ministral3_reliable,'run_router_experiment',router)
    args=SimpleNamespace(utilities=['mooncake_prefill'],holdout_prompts_per_bucket=2000,
        holdout_start_index=2500,holdout_cache_dir='/workspace/holdout',tokenizer_id='ministral')
    point=asyncio.run(worker.run_point('ministral',args,[],{}, {}, {},tmp_path/'evaluation'))
    assert _extract_holdout_prompts_per_bucket(point)==2000
    smoke=asyncio.run(worker.run_point('ministral',args,[],{}, {}, {},tmp_path/'smoke',data_role='calibration'))
    assert _extract_holdout_prompts_per_bucket(smoke) is None
    assert smoke['config']['prompt_source']['data_role']=='calibration'

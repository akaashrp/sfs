import json
from pathlib import Path

import pytest

from scripts.prep.prepare_methodology_service import PROFILE
from scripts.runs import ministral3_figure5 as fig
from scripts.runs import ministral3_hard_prefill_tps as side
from scripts.runs import ministral3_latency as latency
from scripts.runs import ministral3_methodology_stage as stage

ROOT = Path(__file__).resolve().parents[4]
TPS = {'ministral3-3b': 61731.2, 'ministral3-8b': 33112.4, 'ministral3-14b': 21767.7}


def test_cells_match_campaign_fallback_cells_and_leave_legacy_policies_alone():
    campaign = json.loads((ROOT/'scripts/cloud/baseline-campaign-20260916.json').read_text())
    expected = [c for c in campaign['fallback_cells'] if c['family'] == 'ministral']
    keys = ('id', 'family', 'variant', 'policy', 'qps', 'requests')
    assert [tuple(c[k] for k in keys) for c in expected] == [tuple(c[k] for k in keys) for c in side.cells()]
    assert side.QPS == (6.0125, 7.8625, 8.7875) == latency.QPS[:3] and side.REQUESTS == 8000
    assert side.POLICY == 'hard_prefill_tps' and side.POLICY not in stage.POLICIES and len(stage.POLICIES) == 8
    assert side.POLICY not in latency.POLICIES and side.POLICY not in fig.GROUPS['all']
    assert side.option(['--a', '1', '--b', '2', '--a', '3'], '--a') == '3' and side.option(['--a'], '--a') is None


def _legacy(argv=('--x', '1')):
    return {'policies': list(stage.POLICIES), 'loads': {'qps_values': list(latency.QPS)}, 'requests_per_cell': 8000,
            'serving_profile': PROFILE, 'matrix_cells': 32, 'seed': 69, 'experiment_argv': list(argv),
            'configuration': {}, 'file_sha256': {}, 'stage_dir': 'stage'}


def _manifest(tmp_path, **overrides):
    frozen = tmp_path/'frozen.txt'; frozen.write_text('frozen')
    legacy = tmp_path/'legacy.json'; legacy.write_text(json.dumps(_legacy()))
    data = {'status': side.STATUS, 'family': 'ministral', 'provider': 'bridges', 'policy': side.POLICY,
            'wait_estimator': side.ESTIMATOR, 'arrival_timing': side.ARRIVAL_TIMING, 'qps_values': list(side.QPS),
            'requests_per_cell': 8000, 'matrix_cells': 3, 'cells': side.cells(), 'serving_profile': PROFILE,
            'experiment_argv': ['--x', '1'], 'legacy_manifest': str(legacy), 'legacy_manifest_sha256': side.digest(legacy),
            'legacy_policies': list(stage.POLICIES), 'request_identity_slo_sha256': side.REFERENCE,
            'calibration_reuse': {'prefill_tps_by_model': TPS},
            'file_sha256': {str(frozen): side.digest(frozen)}, 'source_sha256': {'src/a.py': 'a'},
            'side_source_sha256': side.digest(Path(side.__file__))}
    data.update(overrides)
    path = tmp_path/'manifest.json'; path.write_text(json.dumps(data))
    return path, frozen, legacy


def test_validate_restricts_policy_grid_and_rejects_manifest_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(side, 'source_hashes', lambda: {'src/a.py': 'a'})
    monkeypatch.setattr(fig, 'load_manifest', lambda path, verify_files=True: json.loads(Path(path).read_text()))
    path, frozen, legacy = _manifest(tmp_path)
    assert side.validate(path)['policy'] == 'hard_prefill_tps'
    for bad in ({'policy': 'hard'}, {'qps_values': [6.0125]}, {'qps_values': [6., 7., 8.]}, {'requests_per_cell': 16000},
                {'matrix_cells': 4}, {'legacy_policies': [*stage.POLICIES, side.POLICY]}, {'provider': 'vast'},
                {'request_identity_slo_sha256': 'x'}, {'wait_estimator': 'live'}, {'arrival_timing': 'event_loop'},
                {'cells': side.cells()[:2]}):
        p, _, _ = _manifest(tmp_path, **bad)
        with pytest.raises(ValueError, match='Invalid frozen'): side.validate(p)
    p, _, _ = _manifest(tmp_path, side_source_sha256='0'*64)
    with pytest.raises(ValueError, match='Side module changed'): side.validate(p)
    p, _, _ = _manifest(tmp_path, source_sha256={'src/a.py': 'b'})
    with pytest.raises(ValueError, match='Source changed'): side.validate(p)
    path, frozen, legacy = _manifest(tmp_path)
    frozen.write_text('changed')
    with pytest.raises(ValueError, match='input changed'): side.validate(path)
    path, frozen, legacy = _manifest(tmp_path)
    legacy.write_text(json.dumps(_legacy(argv=('--x', '2'))))
    with pytest.raises(ValueError, match='drifted'): side.validate(path)
    path, frozen, legacy = _manifest(tmp_path)
    monkeypatch.setattr(fig, 'load_manifest', lambda path, verify_files=True: {**_legacy(), 'policies': [*stage.POLICIES, side.POLICY]})
    with pytest.raises(ValueError, match='policy tuple'): side.validate(path)
    assert side.validate(path, verify_files=False)['policy'] == side.POLICY


def test_prepare_rejects_legacy_drift_before_touching_anything(tmp_path, monkeypatch):
    for base, match in ((_legacy() | {'policies': [*stage.POLICIES, side.POLICY]}, 'policy tuple'),
                        (_legacy() | {'loads': {'qps_values': [6., 7., 8., 8.3]}}, 'load grid'),
                        (_legacy() | {'requests_per_cell': 16000}, 'load grid'),
                        (_legacy() | {'serving_profile': {**PROFILE, 'max_num_seqs': 256}}, 'load grid')):
        monkeypatch.setattr(fig, 'load_manifest', lambda path, verify_files=True, base=base: base)
        with pytest.raises(ValueError, match=match):
            side.prepare(tmp_path/'legacy.json', tmp_path/'calibration.jsonl', tmp_path/'side')
    assert not (tmp_path/'side').exists()


def _payload(rows=3):
    per_request = [{'request_id': f'req-{i}', 'response_id': f'r{i}', 'wait_estimator': side.ESTIMATOR, 'queue_delay_ms': 1.,
                    'ttft_ms': 2., 'system_entry_to_dispatch_ms': .5, 'actual_cost': .1, 'usage_completion_tokens': 4,
                    'system_entry_e2e_ttft_ms': 3., 'system_entry_offset_s': i/2.} for i in range(rows)]
    run = {'utility': side.POLICY, 'wait_estimator': side.ESTIMATOR, 'arrival_timing': side.ARRIVAL_TIMING,
           'per_request': per_request, 'summary': {'system_entry_e2e_ttft_slo_missing_count': 0, 'failed_requests': 0,
                                                    'succeeded_requests': rows}, 'methodology_config': {}}
    return {'config': {'request_rate_qps': 2., 'instance_metadata': {'serving_profile': PROFILE}},
            'router': {'runs': [run], 'prefill_tps_by_instance': {f'vllm-{m}': v for m, v in TPS.items()}}}


def test_estimator_audit_requires_prefill_tps_path_on_thread_arrivals():
    assert side.audit_estimator(_payload(), 3, TPS)['prefill_tps_by_instance'] == {f'vllm-{m}': v for m, v in TPS.items()}
    mutations = [
        (lambda p: p['router']['runs'][0].update(utility='hard'), 'prefill-TPS estimator'),
        (lambda p: p['router']['runs'][0].update(wait_estimator='live'), 'prefill-TPS estimator'),
        (lambda p: p['router']['runs'][0].update(arrival_timing='absolute_schedule'), 'dedicated-thread'),
        (lambda p: p['config']['instance_metadata'].update(serving_profile={**PROFILE, 'max_num_seqs': 256}), 'Ministral profile'),
        (lambda p: p['router']['prefill_tps_by_instance'].update({'vllm-ministral3-3b': 1.}), 'service metrics'),
        (lambda p: p['router']['prefill_tps_by_instance'].pop('vllm-ministral3-3b'), 'service metrics'),
        (lambda p: p['router']['runs'][0]['per_request'][1].update(wait_estimator='live'), 'without the prefill-TPS'),
        (lambda p: p['router']['runs'][0]['per_request'][1].update(system_entry_e2e_ttft_ms=None), 'TTFT'),
        (lambda p: p['router']['runs'][0]['per_request'].pop(), 'smoke request coverage'),
        (lambda p: p['router']['runs'].append({}), 'exactly one policy'),
    ]
    for mutate, match in mutations:
        payload = _payload(); mutate(payload)
        with pytest.raises(ValueError, match=match): side.audit_estimator(payload, 3, TPS)


def test_point_audit_labels_provider_bridges_and_runs_every_gate(tmp_path, monkeypatch):
    import scripts.cloud.worker as worker
    calls = []
    monkeypatch.setattr(side.subprocess, 'check_output', lambda *a, **k: (_ for _ in ()).throw(OSError('no nvidia-smi')))
    monkeypatch.setattr(worker, 'audit_cell', lambda payload, cell: calls.append(('cell', cell['id'])) or {'status': 'PASS_CELL', 'requests': 3, 'realized_qps': cell['qps']})
    monkeypatch.setattr(fig, 'audit_points', lambda paths, manifest, policies: calls.append(('fig', manifest['loads']['qps_values'], policies)) or {'status': 'PASS'})
    monkeypatch.setenv('SLURM_JOB_ID', '4242')
    cell = {**side.cells()[1], 'requests': 3}
    entry = side.audit_point(_payload(), tmp_path/'point.json', cell, {'calibration_reuse': {'prefill_tps_by_model': TPS}}, _legacy())
    assert entry['provider'] == 'bridges' and entry['status'] == 'PASS_CELL' and entry['cell'] == cell
    assert entry['serving_profile'] == PROFILE and entry['slurm_job_id'] == '4242' and entry['gpus'][0].startswith('unavailable')
    assert entry['wait_estimator'] == 'prefill_tps_ttft' and entry['arrival_timing'] == 'absolute_schedule_thread'
    assert calls == [('cell', 'ministral-hard_prefill_tps-7.8625-rerun'), ('fig', [7.8625], ['hard_prefill_tps'])]
    monkeypatch.setattr(worker, 'audit_cell', lambda payload, cell: (_ for _ in ()).throw(ValueError('Arrival generator missed the requested rate by more than 10%')))
    with pytest.raises(ValueError, match='Arrival generator'):
        side.audit_point(_payload(), tmp_path/'point.json', cell, {'calibration_reuse': {'prefill_tps_by_model': TPS}}, _legacy())


def test_smoke_gate_rejects_stale_or_foreign_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(side, 'source_hashes', lambda: {'src/a.py': 'a'})
    manifest = tmp_path/'manifest.json'; manifest.write_text('{}')
    smoke = tmp_path/'smoke_dir'; (smoke/'smoke').mkdir(parents=True)
    (smoke/'smoke/point.json').write_text(json.dumps(_payload(side.SMOKE_REQUESTS)))
    good = {'status': 'PASS_GPU_SMOKE', 'policy': side.POLICY, 'provider': 'bridges', 'manifest_sha256': side.digest(manifest),
            'source_sha256': {'src/a.py': 'a'}, 'point_sha256': side.digest(smoke/'smoke/point.json')}
    (smoke/'audit.json').write_text(json.dumps(good))
    assert side.smoke_gate(smoke, manifest, {'calibration_reuse': {'prefill_tps_by_model': TPS}})['status'] == 'PASS_GPU_SMOKE'
    for bad in ({'policy': 'hard'}, {'provider': 'vast'}, {'manifest_sha256': '0'*64}, {'source_sha256': {'src/a.py': 'b'}},
                {'point_sha256': '0'*64}, {'status': 'PASS'}):
        (smoke/'audit.json').write_text(json.dumps({**good, **bad}))
        with pytest.raises(ValueError, match='stale'):
            side.smoke_gate(smoke, manifest, {'calibration_reuse': {'prefill_tps_by_model': TPS}})


def test_prefill_tps_input_must_be_positive_and_distinct_per_model(tmp_path):
    metrics = tmp_path/'model_metrics.json'
    metrics.write_text(json.dumps({'models': {m: {'score_proxy': {'prefill_tps': v}} for m, v in TPS.items()}}))
    assert side.prefill_tps_by_model(metrics) == TPS
    for broken in ({'ministral3-8b': {'score_proxy': {'prefill_tps': 0}}}, {'ministral3-8b': {'score_proxy': {}}},
                   {'ministral3-8b': {'score_proxy': {'prefill_tps': TPS['ministral3-3b']}}}):
        metrics.write_text(json.dumps({'models': {**{m: {'score_proxy': {'prefill_tps': v}} for m, v in TPS.items()}, **broken}}))
        with pytest.raises(ValueError, match='prefill_tps'): side.prefill_tps_by_model(metrics)

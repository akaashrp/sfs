import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.runs import qwen_baselines as qwen
from scripts.runs import qwen_predictor_variants as variants
from scripts.runs import qwen_hard_prefill_tps as side

ROOT = Path(__file__).resolve().parents[4]


def test_canonical_cells_match_baseline_campaign_fallback_cells():
    campaign = json.loads((ROOT/'scripts/cloud/baseline-campaign-20260916.json').read_text())
    expected = [c for c in campaign['fallback_cells'] if c['family'] == 'qwen']
    actual = side.cells('canonical')
    assert [(c['id'], c['policy'], c['qps'], c['requests']) for c in expected] == \
        [(c['id'], c['policy'], c['qps'], c['requests']) for c in actual]
    assert side.QPS == (6., 7., 8., 8.3) and side.POLICY == 'hard_prefill_tps'
    assert side.POLICY not in qwen.NEW_POLICIES and set(side.SMOKE_POLICIES) == set(qwen.POLICIES) | {side.POLICY}
    with pytest.raises(ValueError): side.cells('score')


def test_side_sweep_argv_restricted_to_policy_and_grid(tmp_path, monkeypatch):
    stage = tmp_path/'stage'; (stage/'timing_models').mkdir(parents=True)
    (stage/'timing_models/methodology_calibration.json').write_text('{}'); (stage/'model_metrics.json').write_text('{}')
    mp = tmp_path/'manifest.json'; mp.write_text('{}')
    manifest = {'sfs_root': str(tmp_path), 'manifest_path': str(mp), 'experiment_argv': ['--num-requests', '16000']}
    output = tmp_path/'out'; output.mkdir()
    for bad in ({'policies': ['hard']}, {'qps_values': [5.]}, {'qps_values': [6., 6.]}, {'timing_gate': lambda *a: 'x'}):
        with pytest.raises(ValueError):
            side.run_sweep(manifest, stage, output, tmp_path/'instances.json', tmp_path/'predictor', **bad)
    assert not (output/'sweep_started.json').exists()
    captured = []
    monkeypatch.setattr(side.subprocess, 'run', lambda argv, **kw: captured.append(argv))
    side.run_sweep(manifest, stage, output, tmp_path/'instances.json', tmp_path/'predictor')
    argv = captured[0]
    assert argv[argv.index('--qps-values')+1:argv.index('--qps-utilities')] == ['6.0', '7.0', '8.0', '8.3']
    assert argv[argv.index('--qps-utilities')+1:argv.index('--output-dir')] == ['hard_prefill_tps']
    assert argv[argv.index('--num-requests')+1] == '16000'
    assert argv[argv.index('--service-metrics-json')+1] == str(stage/'model_metrics.json')
    started = json.loads((output/'sweep_started.json').read_text())
    assert started['provider'] == 'bridges' and started['policies'] == ['hard_prefill_tps'] and started['requests_per_cell'] == 16000
    assert json.loads((output/'sweep_completed.json').read_text())['status'] == 'COMPLETE_UNCOLLATED'


def test_canonical_modes_select_smoke_policies_and_scoped_sweep(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(qwen, 'validate_manifest', lambda path: {'manifest': str(path)})
    monkeypatch.setattr(qwen, 'start_and_run', lambda ns, manifest: seen.append((ns, manifest, qwen.run_sweep)))
    original = qwen.run_sweep
    options = SimpleNamespace(manifest=tmp_path/'m.json', predictor='p', output_root='o', cpu_validation='c', stage_dir='s')
    side.canonical('smoke', options); side.canonical('sweep', options)
    with pytest.raises(ValueError): side.canonical('calibrate', options)
    smoke, sweep = seen
    assert smoke[0].mode == 'smoke' and smoke[0].policies == list(side.SMOKE_POLICIES) and len(smoke[0].policies) == 10
    assert sweep[0].mode == 'sweep' and sweep[0].policies == ['hard_prefill_tps'] and sweep[0].qps_values == [6., 7., 8., 8.3]
    assert smoke[2] is side.run_sweep and sweep[2] is side.run_sweep
    assert qwen.run_sweep is original


def _side_manifest(tmp_path, **overrides):
    base = tmp_path/'base.json'; base.write_text(json.dumps({'arm': 'mlp_length'}))
    data = {'status': 'PASS_CPU_VARIANT_SIDE', 'provider': 'bridges', 'arm': 'mlp_length', 'policy': side.POLICY,
            'qps_values': list(side.QPS), 'requests_per_cell': 16000, 'base_manifest': str(base), 'base_sha256': side.sha256(base),
            'stage': str(tmp_path/'stage'), 'paths': {'x': 'a'}, 'side_source_sha256': side.sha256(Path(side.__file__))}
    data.update(overrides)
    path = tmp_path/'side.json'; path.write_text(json.dumps(data)); return path, base


def test_side_variant_contract_requires_base_validation_and_policy_smoke(tmp_path, monkeypatch):
    path, base = _side_manifest(tmp_path)
    base_record = {'arm': 'mlp_length', 'paths': {'x': 'a'}, 'stage': str(tmp_path/'stage'), 'canonical_manifest': 'c', 'sfs_root': str(tmp_path)}
    monkeypatch.setattr(variants, 'validate', lambda p: dict(base_record))
    monkeypatch.setattr(qwen, 'validate_manifest', lambda p: {'manifest_path': p})
    calls = []
    monkeypatch.setattr(qwen, 'validate_smoke', lambda stage, manifest, predictor, policies=None: calls.append((stage, policies)))
    assert side.variant_validate(path)[1]['arm'] == 'mlp_length'
    assert calls == [(tmp_path/'stage', ['hard_prefill_tps'])]
    for bad in ({'policy': 'hard'}, {'qps_values': [7., 8.3, 8.6, 8.9]}, {'status': 'PASS_CPU_VARIANT'}, {'arm': 'canonical'}):
        p, _ = _side_manifest(tmp_path, **bad)
        with pytest.raises(ValueError, match='contract'): side.variant_validate(p)
    base.write_text(json.dumps({'arm': 'mlp_length', 'changed': True}))
    with pytest.raises(ValueError, match='contract'): side.variant_validate(path)
    path, base = _side_manifest(tmp_path, paths={'x': 'b'})
    with pytest.raises(ValueError, match='differs'): side.variant_validate(path)


def test_variant_smoke_gate_requires_side_policy_result(tmp_path):
    manifest = tmp_path/'side.json'; manifest.write_text('{}')
    result = tmp_path/'smoke_hard_prefill_tps.json'; result.write_text('{}')
    audit = {'status': 'PASS_GPU_VARIANT_SMOKE', 'policy': 'hard', 'arm': 'mlp_length', 'paths': {'x': 'a'},
             'manifest_sha256': side.sha256(manifest), 'result_sha256': side.sha256(result)}
    (tmp_path/'smoke_audit.json').write_text(json.dumps(audit))
    with pytest.raises(ValueError, match='stale'):
        side.variant_validate_smoke({'arm': 'mlp_length', 'paths': {'x': 'a'}}, manifest, tmp_path)


def _points(tmp_path, rates):
    out = tmp_path/'sweep'; (out/'outputs').mkdir(parents=True)
    for i, rate in enumerate(rates):
        (out/'outputs'/f'x_qps{rate:g}_point{i:02d}.json').write_text(json.dumps(
            {'config': {'request_rate_qps': rate}, 'router': {'runs': [{'utility': 'hard_prefill_tps', 'wait_estimator': 'prefill_tps_ttft'}]}}))
    return out


def test_audit_labels_provider_bridges_and_reports_failed_or_missing_cells(tmp_path, monkeypatch):
    import scripts.cloud.worker as worker
    manifest = tmp_path/'m.json'; manifest.write_text('{}')
    monkeypatch.setattr(side.subprocess, 'check_output', lambda *a, **k: (_ for _ in ()).throw(OSError('no nvidia-smi')))
    monkeypatch.setattr(worker, 'audit_cell', lambda payload, cell: {'status': 'PASS_CELL', 'requests': 16000, 'realized_qps': cell['qps']})
    out = _points(tmp_path, side.QPS)
    results = side.audit('canonical', out, manifest, tmp_path)
    assert set(results) == {c['id'] for c in side.cells('canonical')}
    assert all(r['provider'] == 'bridges' and r['status'] == 'PASS_CELL' and r['serving_profile'] == qwen.PROFILE for r in results.values())
    summary = json.loads((out/'cells_audit.json').read_text())
    assert summary['status'] == 'PASS_ALL_CELLS' and summary['provider'] == 'bridges'
    assert (out/'audits'/'qwen-hard_prefill_tps-8.3-rerun.json').exists()

    def failing(payload, cell):
        if cell['qps'] == 8.: raise ValueError('Arrival generator missed the requested rate by more than 10%')
        return {'status': 'PASS_CELL'}
    monkeypatch.setattr(worker, 'audit_cell', failing)
    out = _points(tmp_path/'failed', side.QPS)
    with pytest.raises(ValueError, match='audit failed'): side.audit('mlp_length', out, manifest, tmp_path)
    summary = json.loads((out/'cells_audit.json').read_text())
    assert summary['status'] == 'FAILED_CELLS' and summary['failed'] == ['qwen-mlp_length-hard_prefill_tps-8']
    assert summary['cells']['qwen-mlp_length-hard_prefill_tps-8']['status'] == 'FAIL_CELL'
    out = _points(tmp_path/'missing', side.QPS[:3])
    monkeypatch.setattr(worker, 'audit_cell', lambda payload, cell: {'status': 'PASS_CELL'})
    with pytest.raises(ValueError, match='missing'): side.audit('canonical', out, manifest, tmp_path)

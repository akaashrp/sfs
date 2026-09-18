"""Predictor-variant overlay: grid, no-op reuse pointers, kind dispatch, per-variant argv, matrix, status."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts.cloud.baseline_campaign import RATES
from scripts.cloud.common import ROOT, read, digest, write
from scripts.cloud.variant_campaign import ABLATION_RATES, apply_variant_campaign, VARIANTS, ABLATED_POLICIES, NOOP_POLICIES, variant_cell_id
from scripts.cloud.tests.test_sfs_score_campaign import _bundle, OVERLAY as SFS_OVERLAY, SHA

OVERLAY = ROOT/'scripts/cloud/predictor-variants-campaign-20260917.json'
INVENTORY = ROOT/'scripts/cloud/reports/baseline-campaign-20260916/qwen-reference-inventory.json'
RULE = 'qwen3-0.6b=running_all_prompt_bin_q50'


def _variant_bundle():
    bundle = _bundle(); bundle['variants'] = {v: f'@BUNDLE@/variants/{v}' for v in VARIANTS}
    return bundle


def test_overlay_is_accepted_and_preserves_the_bundle():
    campaign = read(OVERLAY); bundle = _variant_bundle(); saved = deepcopy(bundle)
    active = apply_variant_campaign(bundle, campaign)
    assert bundle == saved and active['files'] == saved['files'] and active['families']['ministral'] == saved['families']['ministral']
    assert len(active['cells']) == 27 and active['requests_total'] == 432000 == campaign['requests_total']
    assert {c['id'] for c in active['cells']} == {variant_cell_id(v, p, q) for v in VARIANTS for p in ABLATED_POLICIES for q in ABLATION_RATES}
    assert all(c['family'] == 'qwen' and c['requests'] == 16000 and c['variant'] != 'canonical' for c in active['cells'])
    assert active['families']['qwen']['policies'] == ['hard', 'score', 'latency_agnostic'] and active['families']['qwen']['qps'] == [7., 8., 8.3]
    assert active['kind'] == 'predictor_variants'
    assert set(active['noop_policies']) == set(NOOP_POLICIES)
    assert all(v['label'] == 'reused_canonical' and v['reuse'] and v['reason'] for v in active['noop_policies'].values())
    assert 'bridges' in active['noop_policies']['round_robin']['reuse'] and 'baseline-campaign' in active['noop_policies']['lmdeploy_proxy']['reuse']
    assert active['comparators']['latency_agnostic']['provider'] == 'bridges'
    reference = {c['qps']: c['point_sha256'] for c in read(INVENTORY)['cells']
                 if c['policy'] == 'latency_agnostic' and c['qps'] in ABLATION_RATES}
    assert {float(k): v['point_sha256'] for k, v in active['comparators']['latency_agnostic']['points'].items()} == reference
    assert active['comparators']['hard']['cells'] == ['qwen-hard-7', 'qwen-hard-8', 'qwen-hard-8.3']
    assert active['comparators']['score']['provider'] == 'vast'
    sfs = read(SFS_OVERLAY)['remaining_length']
    assert campaign['remaining_length']['files'] == sfs['files'] == SHA and campaign['remaining_length']['rules'] == sfs['rules']
    assert campaign['remaining_length']['off_models'] == sfs['off_models']
    assert active['remaining_length']['rules'] == {'qwen3-0.6b': {'mode': 'running_all', 'quantile': 0.5, 'conditioning': 'prompt_bin'}}
    assert 'latency_agnostic' in campaign['remaining_length']['note'] and 'per pool' in campaign['remaining_length']['note']
    assert active['accepted_prior_source_digests'] == {}


def _noop_cell(c): c['cells'][0]['policy'] = 'round_robin'; c['cells'][0]['id'] = 'qwen-flash_quality-round_robin-6'
def _ministral(c): c['cells'][0]['family'] = 'ministral'
def _unknown_variant(c): c['cells'][0]['variant'] = 'mlp_both'; c['cells'][0]['id'] = 'qwen-mlp_both-hard-6'
def _rate(c): c['cells'][0]['qps'] = 8.6; c['cells'][0]['id'] = 'qwen-flash_quality-hard-8.6'
def _budget(c): c['cells'][0]['requests'] = 8000
def _dup(c): c['cells'][1]['id'] = c['cells'][0]['id']
def _drop(c): c['cells'].pop()
def _id(c): c['cells'][0]['id'] = 'qwen-flash_quality-6'
def _kind(c): c['kind'] = 'sfs_score'
def _comparator_missing(c): del c['comparators']['score']
def _comparator_provider(c): c['comparators']['latency_agnostic']['provider'] = 'vast'
def _noop_missing(c): del c['noop_policies']['round_robin']
def _noop_pointer(c): c['noop_policies']['routebalance']['reuse'] = ''
def _smoke(c): c['policies']['qwen'] = ['hard']
def _hash(c): c['remaining_length']['files']['qwen3-0.6b.json'] = '0'*64
def _ministral_rule(c): c['remaining_length']['rules']['ministral3-8b'] = 'running_all'; c['remaining_length']['off_models'].remove('ministral3-8b')
def _total(c): c['requests_total'] -= 16000


@pytest.mark.parametrize('mutate', [_noop_cell, _ministral, _unknown_variant, _rate, _budget, _dup, _drop, _id, _kind, _comparator_missing,
                                    _comparator_provider, _noop_missing, _noop_pointer, _smoke, _hash, _ministral_rule, _total])
def test_overlay_rejections(mutate):
    bad = read(OVERLAY); mutate(bad)
    with pytest.raises(ValueError):
        apply_variant_campaign(_variant_bundle(), bad)
    with pytest.raises(ValueError):
        apply_variant_campaign(_variant_bundle(), read(SFS_OVERLAY))


def test_kind_dispatch_and_policy_derivation():
    from scripts.cloud.campaigns import apply_any_campaign
    from scripts.cloud.worker import policies_for
    active = apply_any_campaign(_variant_bundle(), read(OVERLAY))
    assert active['kind'] == 'predictor_variants'
    for variant in VARIANTS:
        assert policies_for(active, 'qwen', variant, active['families']['qwen']) == ['hard', 'score', 'latency_agnostic']
    from scripts.cloud.sfs_score_campaign import apply_sfs_score_campaign
    sfs = apply_sfs_score_campaign(_variant_bundle(), read(SFS_OVERLAY))
    for variant in VARIANTS:
        assert policies_for(sfs, 'qwen', variant, sfs['families']['qwen']) == ['hard']   # frozen-bundle fallback, no cells


def _main(tmp_path, monkeypatch, *extra):
    from scripts.cloud import worker
    captured = {}
    async def execute(options, manifest, definition, model_paths, output): captured.update(manifest=manifest, options=options)
    monkeypatch.setattr(worker, 'validate_bundle', lambda bundle: _variant_bundle())
    monkeypatch.setattr(worker, 'execute', execute)
    monkeypatch.setattr(worker, 'read', lambda path: {} if str(path).endswith('models.json') else read(path))
    output = tmp_path/f'out{len(list(tmp_path.iterdir()))}'
    monkeypatch.setattr(sys, 'argv', ['worker', 'campaign', '--bundle', str(tmp_path), '--models', str(tmp_path/'models.json'),
                                      '--state', str(tmp_path/'state'), '--output', str(output), '--gpus', '0,1,2,3',
                                      '--campaign', str(OVERLAY), *extra])
    worker.main()
    return captured


def test_worker_dispatches_per_variant_and_refuses_pools_without_cells(tmp_path, monkeypatch):
    for variant in VARIANTS:
        captured = _main(tmp_path, monkeypatch, '--family', 'qwen', '--variant', variant)
        assert captured['manifest']['kind'] == 'predictor_variants' and captured['options'].variant == variant
        assert captured['options'].qualification == str((tmp_path/captured['options'].output).resolve())
    with pytest.raises(SystemExit):   # canonical Qwen has no cells in the variant overlay
        _main(tmp_path, monkeypatch, '--family', 'qwen')
    with pytest.raises(SystemExit):   # Ministral never runs predictor ablations
        _main(tmp_path, monkeypatch, '--family', 'ministral', '--variant', 'mlp_length')
    with pytest.raises(SystemExit):   # Ministral has no cells in this overlay
        _main(tmp_path, monkeypatch, '--family', 'ministral')


def test_each_variant_changes_exactly_one_router_flag_and_only_mlp_length_reaches_servers(tmp_path, monkeypatch):
    from scripts.cloud import worker
    from scripts.cloud.pool import server_argv
    from scripts.runs.qwen_baselines import pool_config
    monkeypatch.setattr(worker, 'portable_cache', lambda bundle, family: Path('/cache'))
    monkeypatch.setattr(worker, 'portable_routebalance', lambda bundle, family: Path('/rb'))
    definition = {'experiment_argv': ['--experiment', 'router', '--seed', '69', '--accuracy-model-path', '@BUNDLE@/qwen/quality',
                                      '--output-length-model-path', '@BUNDLE@/qwen/length', '--bucket-dir', '@BUNDLE@/qwen/holdout',
                                      '--holdout-cache-dir', '@BUNDLE@/qwen/holdout', '--holdout-bucket-dir', '@BUNDLE@/qwen/holdout',
                                      '--routebalance-predictor-path', '@BUNDLE@/qwen/routebalance',
                                      '--methodology-calibration-json', '@QUALIFIED_TIMING@'],
                  'models': ['qwen3-0.6b', 'qwen3-8b', 'qwen3-32b']}
    manifest = {'variants': {v: f'/b/variants/{v}' for v in VARIANTS}}
    pairs = lambda argv: dict(zip(argv[::2], argv[1::2]))
    canonical = pairs(worker.arguments(definition, '/b', 'canonical', manifest))
    assert canonical['--accuracy-model-path'] == '/b/qwen/quality' and canonical['--output-length-model-path'] == '/b/qwen/length'
    for variant in VARIANTS:
        current = pairs(worker.arguments(definition, '/b', variant, manifest))
        flag = '--output-length-model-path' if variant == 'mlp_length' else '--accuracy-model-path'
        assert set(current) == set(canonical) and {k for k in canonical if canonical[k] != current[k]} == {flag}
        assert current[flag] == f'/b/variants/{variant}'
    rows = pool_config({}, (1, 2, 3), 't')['instances']
    def servers(variant):
        argv = worker.arguments(definition, '/b', variant, manifest)
        length = argv[argv.index('--output-length-model-path')+1]
        return [server_argv('qwen', '/m', row, i, tmp_path, length) for i, row in enumerate(rows)]
    base = servers('canonical')
    assert servers('mlp_quality') == base == servers('flash_quality')
    assert all(a[a.index('--output-length-model-path')+1] == '/b/variants/mlp_length' for a in servers('mlp_length'))
    assert all(a.count('--output-length-model-path') == 1 for a in servers('mlp_length'))


def test_noop_schedulers_never_load_router_predictors(monkeypatch, tmp_path):
    from sfs_core.routing.tests.test_methodology_scheduler import make_scheduler
    for policy in ('lmdeploy_proxy', 'mooncake_prefill', 'routebalance'):
        scheduler = make_scheduler(monkeypatch, tmp_path, policy, accuracy_model_path='/quality', output_length_model_path='/length')[0]
        assert scheduler._accuracy_predictor is None and scheduler._output_length_predictor is None
    from sfs_core.routing import wait_time_scheduler
    from sfs_core.routing.latency_history_scheduler import LatencyHistoryScheduler
    from sfs_core.routing.tests.test_latency_scheduler import Client
    monkeypatch.setattr(wait_time_scheduler, 'load_tokenizer', lambda *a, **k: object())
    scheduler = LatencyHistoryScheduler({'a': Client('a')}, accuracy_model_path='/quality', output_length_model_path='/length')
    assert scheduler._accuracy_predictor is None and scheduler._output_length_predictor is None


def test_noop_route_strategies_skip_wait_building():
    from scripts.runs import experiments as exp
    for policy, route in (('round_robin', 'round_robin'), ('latency_agnostic', 'utility')):
        _, _, strategy, estimator, label, skip = exp._baseline_runtime_params(policy, lambda_weight=5e-4, delta_weight=0.)
        assert strategy == route and estimator is exp._wait_estimator_zero and label == 'zero' and skip is True
    for policy in ('lmdeploy_proxy', 'mooncake_prefill', 'routebalance', 'vllm_sr_latency'):
        assert exp._baseline_runtime_params(policy, lambda_weight=5e-4, delta_weight=0.)[3] is None
    _, _, _, estimator, label, skip = exp._baseline_runtime_params('shortest_queue', lambda_weight=5e-4, delta_weight=0.)
    assert estimator is exp._wait_estimator_live and label == 'live' and skip is False


def test_snapshot_parser_and_request_counts_ignore_output_targets():
    from sfs_core.routing.tests.test_methodology_snapshot import _payload
    from sfs_core.routing.methodology_snapshot import parse_baseline_snapshot
    raw = _payload(); other = deepcopy(raw)
    for request in other['requests'].values(): request['num_output_target_tokens'] = 1
    other['decode_backlog_total_tokens'] = 1
    assert parse_baseline_snapshot(raw, observed_at=10) == parse_baseline_snapshot(other, observed_at=10)
    from scripts.runs.experiments import CollectingWaitTimeScheduler
    from sfs_core.routing.wait_time_scheduler import WaitTimeResult
    scheduler = CollectingWaitTimeScheduler.__new__(CollectingWaitTimeScheduler)
    scheduler._last_seen_num_requests_by_instance = {}
    def result(target):
        return WaitTimeResult(instance_id='a', wait_ms=1., fetched_at_s=0., raw_payload={'reports': [
            {'num_requests': 3, 'metadata': {'num_output_target_tokens': target, 'decode_backlog_total_tokens': target}}]})
    first = scheduler._resolve_effective_num_requests(instance_ids=['a'], wait_results={'a': result(1)})
    second = scheduler._resolve_effective_num_requests(instance_ids=['a'], wait_results={'a': result(900000)})
    assert first == second and first[0] == {'a': 3} and first[1] == {'a': 'snapshot'}


def test_collate_binds_records_to_the_overlay_and_labels_status():
    from scripts.cloud.collate import check_record, audit_status, expected_rules
    active = apply_variant_campaign(_variant_bundle(), read(OVERLAY)); expected = {c['id']: c for c in active['cells']}
    rules = expected_rules(active)
    assert rules == {'qwen': RULE, 'ministral': 'current'}
    cell = expected['qwen-mlp_length-hard-7']
    record = {'cell': cell, 'campaign_sha256': digest(OVERLAY), 'remaining_length_rule': RULE}
    assert check_record(record, expected, digest(OVERLAY), 'predictor_variants', rules) == cell['id']
    with pytest.raises(ValueError, match='Unexpected'):
        check_record({'cell': dict(cell, qps=8.6)}, expected, digest(OVERLAY), 'predictor_variants', rules)
    with pytest.raises(ValueError, match='Unexpected'):
        check_record({'cell': dict(cell, id='qwen-hard-7')}, expected, digest(OVERLAY), 'predictor_variants', rules)
    with pytest.raises(ValueError, match='different campaign'):
        check_record(dict(record, campaign_sha256='0'*64), expected, digest(OVERLAY), 'predictor_variants', rules)
    with pytest.raises(ValueError, match='remaining-length'):
        check_record(dict(record, remaining_length_rule='current'), expected, digest(OVERLAY), 'predictor_variants', rules)
    assert check_record({'cell': cell}, expected) == cell['id']    # the baseline overlay has no kind binding
    assert audit_status(expected, []) == 'PASS_27_CELLS' and audit_status(expected, ['x']) == 'PARTIAL'
    from scripts.cloud.sfs_score_campaign import apply_sfs_score_campaign
    sfs = apply_sfs_score_campaign(_variant_bundle(), read(SFS_OVERLAY))
    assert audit_status({c['id']: c for c in sfs['cells']}, []) == 'PASS_14_CELLS' and expected_rules(sfs) == rules


def _observed(pro, flash):
    return {'ontimeutility': {'pro': pro, 'flash': flash}, 'ttft_slo_attainment_pct': 90., 'observed_scored_queries': 15996}


def _record(cid, rule=RULE):
    return {'cell': {'id': cid, 'family': 'qwen'}, 'hardware': {'host': 'vast', 'gpus': [['0', 'uuid', 'NVIDIA H100 80GB HBM3', '81559', '575']]},
            'source_sha256': {'a': '1'}, 'qualification_sha256': 'q', 'campaign_sha256': 'c', 'remaining_length_rule': rule,
            'point': '/p', 'point_sha256': 'p'}


def test_variant_matrix_merges_measured_and_reused_rows():
    from scripts.cloud.collate import variant_matrix
    active = apply_variant_campaign(_variant_bundle(), read(OVERLAY))
    measured = {'qwen-flash_quality-hard-7': (_observed(.4, .5), _record('qwen-flash_quality-hard-7'))}
    reused = {'qwen-hard-7': (_observed(.3, .35), _record('qwen-hard-7'), '/collation/sfs'),
              'qwen-lmdeploy_proxy-7': (_observed(.2, .25), _record('qwen-lmdeploy_proxy-7', 'current'), '/collation/baseline')}
    bridges = {('latency_agnostic', 7.): {'provider': 'bridges', 'cell_id': None, 'point': '/bridges/la7.json', 'point_sha256': 'b',
                                          'status': 'EXISTING_REFERENCE_RETAINED', 'ttft_slo_attainment_pct': 36.9,
                                          'ontimeutility': {'pro': .1, 'flash': .12}, 'remaining_length_rule': 'current',
                                          'hardware': {'host': 'bridges'}, 'reused_from': str(INVENTORY)},
               ('round_robin', 7.): {'provider': 'bridges', 'cell_id': None, 'point': '/bridges/rr7.json', 'point_sha256': 'r',
                                     'status': 'EXISTING_REFERENCE_RETAINED', 'ttft_slo_attainment_pct': 67.4, 'ontimeutility': None,
                                     'ontimeutility_note': 'Bridges point not readable on this host', 'remaining_length_rule': 'current',
                                     'hardware': {'host': 'bridges'}, 'reused_from': str(INVENTORY)}}
    matrix = variant_matrix(active, measured, reused, bridges)
    rows = {(r['variant'], r['policy'], r['qps']): r for r in matrix['rows']}
    assert len(rows) == len(matrix['rows']) == (3 + 3*9) * len(ABLATION_RATES)
    m = rows[('flash_quality', 'hard', 7.)]
    assert m['source'] == 'measured' and m['provider'] == 'vast' and m['primary_judge'] == 'flash' and m['primary_ontimeutility'] == .5
    assert m['hardware'] == {'host': 'vast', 'gpus': ['NVIDIA H100 80GB HBM3']} and m['remaining_length_rule'] == RULE
    assert m['campaign_sha256'] == 'c' and m['qualification_sha256'] == 'q' and len(m['source_digest']) == 64
    assert rows[('mlp_length', 'hard', 7.)] == {'variant': 'mlp_length', 'policy': 'hard', 'qps': 7., 'primary_judge': 'pro',
                                                'source': 'missing', 'cell_id': 'qwen-mlp_length-hard-7'}
    c = rows[('canonical', 'hard', 7.)]
    assert c['source'] == 'reused_canonical' and c['provider'] == 'vast' and c['primary_judge'] == 'pro' and c['primary_ontimeutility'] == .3
    assert c['reused_from'] == '/collation/sfs' and c['cell_id'] == 'qwen-hard-7'
    f = rows[('flash_quality', 'lmdeploy_proxy', 7.)]
    assert f['source'] == 'reused_canonical' and f['primary_judge'] == 'flash' and f['primary_ontimeutility'] == .25
    assert f['cell_id'] == 'qwen-lmdeploy_proxy-7' and f['remaining_length_rule'] == 'current'
    assert rows[('mlp_quality', 'lmdeploy_proxy', 7.)]['primary_ontimeutility'] == .2
    la = rows[('canonical', 'latency_agnostic', 7.)]
    assert la['provider'] == 'bridges' and la['source'] == 'reused_canonical' and la['point'] == '/bridges/la7.json' and la['primary_ontimeutility'] == .1
    assert la['reused_from'] == str(INVENTORY) and la['cell_id'] is None
    rr = rows[('mlp_quality', 'round_robin', 7.)]
    assert rr['provider'] == 'bridges' and rr['primary_ontimeutility'] is None and 'not readable' in rr['ontimeutility_note']
    assert rows[('canonical', 'score', 7.)]['source'] == 'missing_canonical'
    assert rows[('mlp_length', 'vllm_sr_latency', 8.3)] == {'variant': 'mlp_length', 'policy': 'vllm_sr_latency', 'qps': 8.3,
                                                             'primary_judge': 'pro', 'source': 'missing_canonical', 'cell_id': 'qwen-vllm_sr_latency-8.3'}
    assert not any(r['variant'] == 'canonical' and r['policy'] in NOOP_POLICIES for r in matrix['rows'])
    assert matrix['comparators'] == active['comparators'] and matrix['noop_policies'] == active['noop_policies']


def test_bridges_reference_rows_recompute_utility_from_readable_points(tmp_path):
    from scripts.cloud.collate import bridges_reference_rows
    per_request = [{'request_id': 'req-0', 'bucket': 'alpaca', 'system_entry_e2e_ttft_slo_met': True, 'system_entry_e2e_ttft_ms': 100.,
                    'ttft_slo_ms': 200., 'response_model': 'qwen3-8b', 'actual_cost': 1.},
                   {'request_id': 'req-1', 'bucket': 'alpaca', 'system_entry_e2e_ttft_slo_met': False, 'system_entry_e2e_ttft_ms': 300.,
                    'ttft_slo_ms': 200., 'response_model': 'qwen3-8b', 'actual_cost': 1.}]
    summary = {'system_entry_e2e_ttft_slo_attainment_pct': 50.}
    point = tmp_path/'point.json'
    write(point, {'config': {'lambda_weight': .5, 'request_rate_qps': 7.}, 'router': {'runs': [
        {'utility': 'round_robin', 'per_request': per_request, 'summary': summary},
        {'utility': 'latency_agnostic', 'per_request': per_request, 'summary': summary}]}})
    reference = tmp_path/'inventory.json'
    cells = [{'family': 'qwen', 'policy': 'latency_agnostic', 'qps': 7., 'requests': 2, 'point': str(point), 'point_sha256': digest(point),
              'summary': summary, 'status': 'EXISTING_REFERENCE_RETAINED'},
             {'family': 'qwen', 'policy': 'round_robin', 'qps': 7., 'requests': 2, 'point': str(tmp_path/'missing.json'), 'point_sha256': '0'*64,
              'summary': {'system_entry_e2e_ttft_slo_attainment_pct': 40.}, 'status': 'EXISTING_REFERENCE_RETAINED'}]
    write(reference, {'cells': cells})
    mapping = {'req-0': SimpleNamespace(bucket='alpaca', example_id='e0'), 'req-1': SimpleNamespace(bucket='alpaca', example_id='e1')}
    quality = {'pro': {('qwen3-8b', 'alpaca', 'e0'): 1., ('qwen3-8b', 'alpaca', 'e1'): 1.},
               'flash': {('qwen3-8b', 'alpaca', 'e0'): .8, ('qwen3-8b', 'alpaca', 'e1'): .8}}
    common = {('alpaca', 'e0'), ('alpaca', 'e1')}
    rows = bridges_reference_rows(reference, mapping, quality, common)
    la = rows[('latency_agnostic', 7.)]
    assert la['provider'] == 'bridges' and la['ontimeutility'] == {'pro': pytest.approx(.25), 'flash': pytest.approx(.15)}
    assert la['ttft_slo_attainment_pct'] == 50. and la['point_sha256'] == digest(point) and la['remaining_length_rule'] == 'current'
    assert la['observed_scored_queries'] == 2 and 'Recomputed' in la['ontimeutility_note']
    rr = rows[('round_robin', 7.)]
    assert rr['ontimeutility'] is None and 'not readable' in rr['ontimeutility_note'] and rr['ttft_slo_attainment_pct'] == 40.
    write(reference, {'cells': [dict(cells[0], point_sha256='0'*64)]})
    with pytest.raises(ValueError, match='changed'):
        bridges_reference_rows(reference, mapping, quality, common)
    write(reference, {'cells': [dict(cells[0], policy='hard')]})
    with pytest.raises(ValueError, match='unique hard run'):
        bridges_reference_rows(reference, mapping, quality, common)


def test_control_status_accounts_for_overlay_cells(tmp_path, capsys):
    from scripts.cloud.control import status
    state = tmp_path/'state'; (state/'jobs').mkdir(parents=True)
    point = tmp_path/'point.json'; write(point, {'x': 1})
    write(state/'completed'/'qwen-mlp_length-hard-7.json', {'cell': {'id': 'qwen-mlp_length-hard-7'}, 'point': str(point), 'point_sha256': digest(point)})
    bundle = tmp_path/'bundle'; bundle.mkdir(); write(bundle/'bundle.json', _variant_bundle())
    status(str(state), str(bundle), str(OVERLAY))
    out = json.loads(capsys.readouterr().out)
    assert out['expected'] == 27 and out['completed'] == 1 and out['campaign_kind'] == 'predictor_variants'
    assert out['campaign_sha256'] == digest(OVERLAY) and len(out['remaining']) == 26 and 'qwen-mlp_length-hard-7' not in out['remaining']
    status(str(state), str(bundle), str(SFS_OVERLAY))
    out = json.loads(capsys.readouterr().out)
    assert out['expected'] == 14 and out['completed'] == 0 and out['campaign_kind'] == 'sfs_score' and len(out['remaining']) == 14
    assert out['completed_outside_campaign'] == ['qwen-mlp_length-hard-7']
    status(str(state), str(bundle))
    out = json.loads(capsys.readouterr().out)
    assert out['expected'] == 0 and out['campaign_kind'] is None and out['campaign'] is None

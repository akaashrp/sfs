"""SFS/SCORE overlay: exact grid, bound tables/rules, worker dispatch, rule reaches only the 0.6B engine."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

from scripts.cloud.baseline_campaign import RATES
from scripts.cloud.common import ROOT, read, digest
from scripts.cloud.sfs_score_campaign import apply_sfs_score_campaign, POLICIES

OVERLAY = ROOT/'scripts/cloud/sfs-score-campaign-20260917.json'
BASELINE = ROOT/'scripts/cloud/baseline-campaign-20260916.json'
TABLES = ROOT/'scripts/cloud/remaining-length-tables-20260917'
QWEN = ['qwen3-0.6b', 'qwen3-8b', 'qwen3-32b']
MINISTRAL = ['ministral3-3b', 'ministral3-8b', 'ministral3-14b']
SHA = {'qwen3-0.6b.json': '50a14ca76681895a622cbf2d69ba362e37792b32b76c5967b7b9b67f3f6ed87d',
       'qwen3-8b.json': '2392950a25c04003e945124554fd33654da1c552ccdecf7e59d20770f9c04ce6',
       'qwen3-32b.json': '74e3a4ef30c8be24e01983b2609c4d76c8641422f26def2a899ec47ec88cf599',
       'manifest.json': '2eac8a79de5650982d998c7ded87f8cf48e19f3a53d1005afb734d7c4c491fbb'}


def _bundle():
    return {'families': {'qwen': {'qps': [99], 'policies': ['x'], 'requests': 16000, 'models': list(QWEN)},
                         'ministral': {'qps': [99], 'policies': ['y'], 'requests': 8000, 'models': list(MINISTRAL)}},
            'files': {'quality': 'immutable'}, 'cells': [], 'variants': {'mlp_length': '@BUNDLE@/variants/mlp_length'}}


def test_committed_tables_match_their_manifest_and_the_overlay():
    assert {p.name: digest(p) for p in TABLES.iterdir()} == SHA
    manifest = read(TABLES/'manifest.json')
    assert {m: e['sha256'] for m, e in manifest['tables'].items()} == {m: SHA[f'{m}.json'] for m in QWEN}
    assert manifest['evaluation_ids'] == 16000 and manifest['min_support'] == 8 and manifest['cap'] == 8192
    assert read(OVERLAY)['remaining_length']['files'] == SHA


def test_overlay_is_accepted_and_preserves_the_bundle():
    campaign = read(OVERLAY); bundle = _bundle(); saved = deepcopy(bundle)
    active = apply_sfs_score_campaign(bundle, campaign)
    assert bundle == saved and active['files'] == saved['files'] and active['variants'] == saved['variants']
    assert len(active['cells']) == 14 and active['requests_total'] == 176000 == campaign['requests_total']
    assert {c['id'] for c in active['cells']} == {f'{f}-{p}-{q:g}' for f in RATES for p in POLICIES for q in RATES[f]}
    assert all(c['variant'] == 'canonical' and c['requests'] == (16000 if c['family'] == 'qwen' else 8000) for c in active['cells'])
    for family in RATES:
        assert active['families'][family]['policies'] == ['hard', 'score'] and active['families'][family]['qps'] == list(RATES[family])
    assert active['kind'] == 'sfs_score'
    rl = active['remaining_length']
    assert Path(rl['tables']) == TABLES.resolve() and rl['files'] == SHA
    assert rl['rules'] == {'qwen3-0.6b': {'mode': 'running_all', 'quantile': 0.5, 'conditioning': 'prompt_bin'}}
    assert rl['rule_specs'] == {'qwen3-0.6b': 'running_all:0.5:prompt_bin'}
    assert rl['off_models'] == sorted(set(QWEN + MINISTRAL) - {'qwen3-0.6b'})
    assert rl['evidence']['on']['ttft_slo_attainment_pct'] == pytest.approx(90.89375)
    assert rl['evidence']['on']['pro_ontimeutility'] == pytest.approx(0.4732, abs=1e-4)
    assert {k: v['ttft_slo_attainment_pct'] for k, v in rl['evidence']['off_controls'].items()} == pytest.approx(
        {'qps8p6': 49.9625, 'qps8p6-repeat1': 63.71}, abs=0.01)
    assert active['accepted_prior_source_digests'] == read(BASELINE)['accepted_prior_source_digests']
    assert 'qwen-hard-8' in campaign['reuse'] and set(campaign['reuse']['qwen-hard-8']['candidates']) == {'qps8-lane-a', 'qps8-lane-b'}


def _drop(c): c['cells'].pop()
def _rate(c): c['cells'][0]['qps'] = 8.6
def _budget(c): c['cells'][-1]['requests'] = 16000
def _dup(c): c['cells'][1]['id'] = c['cells'][0]['id']
def _variant(c): c['cells'][0]['variant'] = 'mlp_length'
def _id(c): c['cells'][0]['id'] = 'qwen-hard-six'
def _policy(c): c['cells'][0]['policy'] = 'lmdeploy_proxy'
def _kind(c): c['kind'] = 'baseline'
def _smoke(c): c['policies']['qwen'] = ['hard']
def _total(c): c['requests_total'] += 1
def _ministral_rule(c): c['remaining_length']['rules']['ministral3-3b'] = 'running_all:0.5:prompt_bin'; c['remaining_length']['off_models'].remove('ministral3-3b')
def _explicit_off(c): c['remaining_length']['rules']['qwen3-8b'] = 'off'; c['remaining_length']['off_models'].remove('qwen3-8b')
def _no_rules(c): c['remaining_length']['rules'] = {}; c['remaining_length']['off_models'].append('qwen3-0.6b')
def _bad_spec(c): c['remaining_length']['rules']['qwen3-0.6b'] = 'always:0.5'
def _hash(c): c['remaining_length']['files']['qwen3-0.6b.json'] = '0'*64
def _unlisted(c): del c['remaining_length']['files']['qwen3-32b.json']
def _off(c): c['remaining_length']['off_models'].remove('qwen3-32b')
def _absolute(c): c['remaining_length']['tables'] = str(TABLES)
def _outside(c): c['remaining_length']['tables'] = '../sfs_reserve_20260916'
def _evidence(c): c['remaining_length']['evidence'] = {}
def _digests(c): c['accepted_prior_source_digests'] = {'short': 'x'}
def _extra_field(c): c['cells'][0]['priority'] = 'high'


@pytest.mark.parametrize('mutate', [_drop, _rate, _budget, _dup, _variant, _id, _policy, _kind, _smoke, _total, _ministral_rule,
                                    _explicit_off, _no_rules, _bad_spec, _hash, _unlisted, _off, _absolute, _outside, _evidence,
                                    _digests, _extra_field])
def test_overlay_rejections(mutate):
    bad = read(OVERLAY); mutate(bad)
    with pytest.raises(ValueError):
        apply_sfs_score_campaign(_bundle(), bad)
    with pytest.raises(ValueError):
        apply_sfs_score_campaign(_bundle(), read(BASELINE))


def test_kind_dispatch_selects_the_validator():
    from scripts.cloud.campaigns import apply_any_campaign
    assert apply_any_campaign(_bundle(), read(OVERLAY))['kind'] == 'sfs_score'
    baseline = apply_any_campaign(_bundle(), read(BASELINE))
    assert len(baseline['cells']) == 28 and 'kind' not in baseline and 'remaining_length' not in baseline
    with pytest.raises(ValueError, match='Unknown campaign kind'):
        apply_any_campaign(_bundle(), {'kind': 'mystery', 'cells': []})


def test_worker_main_dispatches_on_campaign_kind(tmp_path, monkeypatch):
    from scripts.cloud import worker
    captured = {}
    async def execute(options, manifest, definition, model_paths, output): captured.update(manifest=manifest, definition=definition)
    monkeypatch.setattr(worker, 'validate_bundle', lambda bundle: _bundle())
    monkeypatch.setattr(worker, 'execute', execute)
    monkeypatch.setattr(worker, 'read', lambda path: {} if str(path).endswith('models.json') else read(path))
    monkeypatch.setattr(sys, 'argv', ['worker', 'campaign', '--bundle', str(tmp_path), '--models', str(tmp_path/'models.json'),
                                      '--state', str(tmp_path/'state'), '--output', str(tmp_path/'out'), '--family', 'ministral',
                                      '--gpus', '4,5,6', '--campaign', str(OVERLAY)])
    worker.main()
    assert captured['manifest']['kind'] == 'sfs_score' and captured['definition']['policies'] == ['hard', 'score']
    assert read(tmp_path/'out/status.json')['state'] == 'COMPLETE'


def test_policies_for_follows_the_manifest_cells():
    from scripts.cloud.worker import policies_for
    from scripts.runs import qwen_baselines as qwen
    frozen = {'cells': [{'family': 'qwen', 'variant': 'canonical', 'policy': p} for p in qwen.NEW_POLICIES for _ in range(4)]
                       + [{'family': 'qwen', 'variant': v, 'policy': 'hard'} for v in ('mlp_quality', 'mlp_length', 'flash_quality')]
                       + [{'family': 'ministral', 'variant': 'canonical', 'policy': p} for p in ('hard', 'score', 'round_robin')]}
    definition = {'policies': list(qwen.NEW_POLICIES)}
    assert policies_for(frozen, 'qwen', 'canonical', definition) == list(qwen.NEW_POLICIES)
    for variant in ('mlp_quality', 'mlp_length', 'flash_quality'):
        assert policies_for(frozen, 'qwen', variant, definition) == ['hard']
    assert policies_for(frozen, 'ministral', 'canonical', {'policies': ['round_robin', 'hard', 'score']}) == ['round_robin', 'hard', 'score']
    active = apply_sfs_score_campaign(_bundle(), read(OVERLAY))
    for family in ('qwen', 'ministral'):
        assert policies_for(active, family, 'canonical', active['families'][family]) == ['hard', 'score']
    assert policies_for({'cells': []}, 'qwen', 'canonical', definition) == list(qwen.NEW_POLICIES)
    assert policies_for({'cells': []}, 'qwen', 'mlp_length', definition) == ['hard']


def test_rule_reaches_only_the_small_qwen_engine_and_the_instances_block(tmp_path):
    from scripts.cloud.worker import family_remaining_length, remaining_length_record
    from scripts.cloud.pool import remaining_length_provenance, server_argv
    from scripts.runs.qwen_baselines import pool_config
    active = apply_sfs_score_campaign(_bundle(), read(OVERLAY))
    qwen = active['families']['qwen']
    block = family_remaining_length(active, qwen)
    assert block == {'tables': str(TABLES.resolve()), 'rules': active['remaining_length']['rules']}
    provenance = remaining_length_provenance(qwen, block)
    assert provenance['rule'] == 'qwen3-0.6b=running_all_prompt_bin_q50'
    small = provenance['models']['qwen3-0.6b']
    assert small['table']['sha256'] == SHA['qwen3-0.6b.json'] and small['table']['path'] == str(TABLES.resolve()/'qwen3-0.6b.json')
    assert small['table']['support']['all'] == 10000
    for model in ('qwen3-8b', 'qwen3-32b'):
        assert provenance['models'][model]['rule'] == 'current' and 'table' not in provenance['models'][model]
    rows = pool_config({}, (1, 2, 3), 't')['instances']
    argv = [server_argv('qwen', '/m', row, i, tmp_path, '/len', provenance) for i, row in enumerate(rows)]
    off = [server_argv('qwen', '/m', row, i, tmp_path, '/len') for i, row in enumerate(rows)]
    assert argv[0] != off[0] and argv[1:] == off[1:]
    assert argv[0][argv[0].index('--remaining-length-table')+1] == small['table']['path']
    assert argv[0][argv[0].index('--remaining-length-mode')+1] == 'running_all'
    assert not any('--remaining-length-mode' in a for a in argv[1:])
    assert remaining_length_record(provenance) == {'rule': 'qwen3-0.6b=running_all_prompt_bin_q50', 'models': {
        'qwen3-0.6b': {'rule': 'running_all_prompt_bin_q50', 'table_sha256': SHA['qwen3-0.6b.json']},
        'qwen3-8b': {'rule': 'current', 'table_sha256': None}, 'qwen3-32b': {'rule': 'current', 'table_sha256': None}}}
    # Router side: the fill table is attached to the 0.6B instance only.
    from types import SimpleNamespace
    from scripts.runs import experiments as exp
    instances = {f'vllm-{m}': SimpleNamespace(model_id=m) for m in QWEN}
    attached = exp._load_remaining_length_tables(provenance, instances)
    assert set(attached) == {'vllm-qwen3-0.6b'} and attached['vllm-qwen3-0.6b'].sha256 == SHA['qwen3-0.6b.json']


def test_ministral_pool_never_receives_and_always_refuses_rules(tmp_path):
    from scripts.cloud.worker import family_remaining_length
    from scripts.cloud.pool import remaining_length_provenance, server_argv
    active = apply_sfs_score_campaign(_bundle(), read(OVERLAY))
    ministral = active['families']['ministral']
    assert family_remaining_length(active, ministral) is None
    assert remaining_length_provenance(ministral, None)['rule'] == 'current'
    forced = dict(active, remaining_length={'tables': str(TABLES), 'rules': {'ministral3-3b': {'mode': 'running_all', 'quantile': .5, 'conditioning': 'prompt_bin'}}})
    block = family_remaining_length(forced, ministral)
    assert block['rules'] == forced['remaining_length']['rules']
    with pytest.raises((ValueError, OSError)):   # no Ministral table exists; the pool refuses before any server starts
        remaining_length_provenance(ministral, block)
    row = {'default_model': 'x', 'model_id': 'ministral3-3b', 'address': 'http://h:1', 'snapshot_shm_name': 's',
           'snapshot_shm_size_bytes': 1, 'ttft_batch_model': {}}
    with pytest.raises(ValueError, match='Qwen family'):
        server_argv('ministral', '/m', row, 0, tmp_path, '/len', {'rule': 'x', 'models': {'ministral3-3b': {'mode': 'running_all'}}})


def test_release_binds_the_remaining_length_rule(tmp_path, monkeypatch):
    from scripts.cloud import worker
    from scripts.cloud.common import write
    from types import SimpleNamespace
    monkeypatch.setattr(worker, 'hardware', lambda gpus: {'gpu': gpus})
    bundle = tmp_path/'bundle'; bundle.mkdir(); write(bundle/'bundle.json', {'cells': []})
    write(tmp_path/'smoke.json', {'ok': True})
    report = {'source_sha256': {'a': '1'}, 'family': 'qwen', 'variant': 'canonical', 'bundle_sha256': digest(bundle/'bundle.json'),
              'hardware': {'gpu': ['0']}, 'files': {'smoke.json': digest(tmp_path/'smoke.json')},
              'campaign_sha256': digest(OVERLAY), 'remaining_length_rule': 'qwen3-0.6b=running_all_prompt_bin_q50'}
    write(tmp_path/'qualification.json', report)
    write(tmp_path/'release.json', {'status': 'RELEASED', 'qualification_sha256': digest(tmp_path/'qualification.json'),
                                    'timing_review': 'x'*40, 'load_review': 'y'*40})
    options = SimpleNamespace(family='qwen', variant='canonical', bundle=str(bundle), gpus='0', campaign=str(OVERLAY))
    worker.validate_release(tmp_path, options, {'a': '1'}, 'qwen3-0.6b=running_all_prompt_bin_q50')
    with pytest.raises(ValueError, match='remaining-length'):
        worker.validate_release(tmp_path, options, {'a': '1'}, 'current')
    with pytest.raises(ValueError, match='remaining-length'):
        worker.validate_release(tmp_path, options, {'a': '1'})
    report.pop('remaining_length_rule'); write(tmp_path/'qualification.json', report)
    write(tmp_path/'release.json', {'status': 'RELEASED', 'qualification_sha256': digest(tmp_path/'qualification.json'),
                                    'timing_review': 'x'*40, 'load_review': 'y'*40})
    worker.validate_release(tmp_path, options, {'a': '1'})   # pre-rule qualifications default to the current rule

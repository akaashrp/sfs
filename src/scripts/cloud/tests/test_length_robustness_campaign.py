"""The length-robustness overlay: authorized grid, the fill it exists to test, and the comparison it records."""
from copy import deepcopy

import pytest

from scripts.cloud.common import ROOT, read
from scripts.cloud.campaigns import apply_any_campaign
from scripts.cloud.length_robustness_campaign import (ARMS, FILLED_MODELS, KIND, POLICY, RATES, REQUESTS, SUFFIX,
                                                      VARIANTS, arm_rates, cell_id, declared_arms, FAMILY_ARMS)
from scripts.cloud.tests.test_sfs_score_campaign import _bundle

OVERLAY = ROOT/'scripts/cloud/length-robustness-campaign-20260920.json'


@pytest.fixture
def bundle():
    # The same synthetic manifest the other overlay tests use: no artifact tree on the test host.
    return _bundle()


def overlay():
    return deepcopy(read(OVERLAY))


def test_the_committed_overlay_is_the_authorized_grid(bundle):
    m = apply_any_campaign(bundle, overlay())
    assert m['kind'] == KIND
    # The grid is whatever the overlay declares, validated against the arms this family knows.
    declared = declared_arms(overlay(), 'qwen')
    assert set(declared) <= set(FAMILY_ARMS['qwen']) and 'flash_quality' in declared
    expected = {cell_id(v, q): None for v in declared for q in arm_rates(v, 'qwen', overlay())}
    assert {c['id'] for c in m['cells']} == set(expected) and len(m['cells']) == len(expected)
    assert all(c['policy'] == POLICY and c['requests'] == REQUESTS and c['family'] == 'qwen' for c in m['cells'])
    assert m['families']['qwen']['policies'] == [POLICY]
    assert m['families']['qwen']['qps'] == sorted({q for v in declared for q in arm_rates(v, 'qwen', overlay())})
    assert m['requests_total'] == len(expected)*REQUESTS


def test_every_qwen_engine_fills_from_its_table(bundle):
    m = apply_any_campaign(bundle, overlay())
    rules = m['remaining_length']['rules']
    assert set(rules) == set(FILLED_MODELS)
    # The canonical rule itself is unchanged; only the set of engines it covers is wider.
    canonical = apply_any_campaign(_bundle(), read(ROOT/'scripts/cloud/sfs-score-campaign-20260917.json'))['remaining_length']
    assert list(rules.values()) == [canonical['rules']['qwen3-0.6b']]*len(FILLED_MODELS)
    assert m['remaining_length']['tables'] == canonical['tables'] and m['remaining_length']['files'] == canonical['files']
    assert not [model for model in m['remaining_length']['off_models'] if model.startswith('qwen3-')]


def test_an_overlay_that_leaves_an_engine_on_the_predictor_is_refused(bundle):
    bad = overlay(); bad['remaining_length']['rules'].pop('qwen3-32b')
    bad['remaining_length']['off_models'] = sorted(bad['remaining_length']['off_models'] + ['qwen3-32b'])
    with pytest.raises(ValueError, match='fills every qwen engine'):
        apply_any_campaign(bundle, bad)


def test_each_cell_records_the_measured_numbers_it_is_compared_against(bundle):
    m = apply_any_campaign(bundle, overlay())
    for cid, entry in m['repeats'].items():
        assert entry['cell'] == cid[:-len(SUFFIX)]
        assert 0 < entry['sfs_ontimeutility'] < 1 and 0 < entry['prefill_tps_ontimeutility'] < 1
    # The 8.3 cells are the ones this experiment exists for: SFS below prefill-TPS when only the 0.6B fills.
    for variant in VARIANTS:
        entry = m['repeats'][cell_id(variant, 8.3)]
        assert entry['sfs_ontimeutility'] < entry['prefill_tps_ontimeutility']


@pytest.mark.parametrize('mutate, match', [
    (lambda o: o.update(diagnosis='  '), 'diagnosis'),
    (lambda o: o.update(policies={'qwen': ['hard', 'score']}), 'hard smoke policy'),
    (lambda o: o['cells'].pop(), 'exactly one cell'),
    (lambda o: o['cells'].append(deepcopy(o['cells'][0])), 'duplicate'),
    (lambda o: o['cells'][0].update(qps=6.0), 'authorized'),
    (lambda o: o['cells'][0].update(id='qwen-mlp_length-hard-7'), 'Cell id must be'),
    (lambda o: o['cells'][0].update(requests=8000), 'repeat a qwen'),
    (lambda o: o['cells'][0].update(policy='score'), 'repeat a qwen'),
    (lambda o: o['repeats'].popitem(), 'repeats must name'),
    (lambda o: o['repeats'][cell_id('mlp_length', 8.3)].update(prefill_tps_ontimeutility='0.48'), 'measured OnTimeUtility'),
    (lambda o: o['repeats'][cell_id('mlp_length', 8.3)].update(cell='qwen-hard-8.3'), 'must repeat'),
])
def test_overlay_rejections(bundle, mutate, match):
    bad = overlay(); mutate(bad)
    with pytest.raises(ValueError, match=match):
        apply_any_campaign(bundle, bad)


def test_the_canonical_arm_carries_no_variant_segment_and_its_own_rates(bundle):
    m = apply_any_campaign(bundle, overlay())
    canonical = [c for c in m['cells'] if c['variant'] == 'canonical']
    assert {c['id'] for c in canonical} == {f'qwen-hard-{q:g}{SUFFIX}' for q in (6., 7., 8., 8.3)}
    assert arm_rates('canonical') == (6., 7., 8., 8.3) and arm_rates('mlp_length') == RATES
    # It is read against the headline grid it would replace.
    for c in canonical:
        entry = m['repeats'][c['id']]
        assert entry['cell'] == f"qwen-hard-{c['qps']:g}" and 0 < entry['prefill_tps_ontimeutility'] < 1


def test_the_measured_cells_keep_their_earlier_overlay_and_source(bundle):
    m = apply_any_campaign(bundle, overlay())
    assert m['accepted_prior_campaign_digests'] and m['accepted_prior_source_digests']
    with pytest.raises(ValueError, match='accepted_prior_campaign_digests'):
        bad = overlay(); bad['accepted_prior_campaign_digests'] = {'nope': 'reason'}
        apply_any_campaign(bundle, bad)


def test_a_ministral_engine_receives_the_fill_flags(tmp_path):
    """The rule reaches a Ministral server exactly as it reaches a Qwen one."""
    from scripts.cloud.pool import server_argv
    table = tmp_path/'ministral3-3b.json'; table.write_text('{}')
    row = {'model_id': 'ministral3-3b', 'default_model': 'ministral3-3b-instruct', 'address': '127.0.0.1:9100',
           'snapshot_shm_name': 'sfs_test_0', 'snapshot_shm_size_bytes': 1024,
           'ttft_batch_model': {'intercept': 0.0, 'prefill_coeff': 0.0, 'prefill_sq_coeff': 0.0,
                                'decode_coeff': 0.0, 'sum_coeff': 0.0, 'sum_sq_coeff': 0.0},
           'batch_time_feature_set': 'legacy'}
    block = {'models': {'ministral3-3b': {'mode': 'running_all', 'quantile': 0.5, 'conditioning': 'prompt_bin',
                                          'table': {'path': str(table)}}}}
    argv = server_argv('ministral', tmp_path/'model', row, 0, tmp_path, tmp_path/'length.txt', block)
    for flag, value in (('--remaining-length-mode', 'running_all'), ('--remaining-length-table', str(table)),
                        ('--remaining-length-quantile', '0.5'), ('--remaining-length-conditioning', 'prompt_bin')):
        assert flag in argv and argv[argv.index(flag)+1] == value
    # A model without a rule still gets none of them.
    bare = server_argv('ministral', tmp_path/'model', row, 0, tmp_path, tmp_path/'length.txt', {'models': {}})
    assert not [a for a in bare if a.startswith('--remaining-length')]

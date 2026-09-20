"""The length-robustness overlay: authorized grid, the fill it exists to test, and the comparison it records."""
from copy import deepcopy

import pytest

from scripts.cloud.common import ROOT, read
from scripts.cloud.campaigns import apply_any_campaign
from scripts.cloud.length_robustness_campaign import (FILLED_MODELS, KIND, POLICY, RATES, REQUESTS, SUFFIX,
                                                      VARIANTS, cell_id)
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
    assert {c['id'] for c in m['cells']} == {cell_id(v, q) for v in VARIANTS for q in RATES}
    assert len(m['cells']) == len(VARIANTS)*len(RATES)
    assert all(c['policy'] == POLICY and c['requests'] == REQUESTS and c['family'] == 'qwen' for c in m['cells'])
    assert m['families']['qwen']['policies'] == [POLICY] and m['families']['qwen']['qps'] == sorted(RATES)
    assert m['requests_total'] == len(VARIANTS)*len(RATES)*REQUESTS


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
    with pytest.raises(ValueError, match='fills every Qwen engine'):
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
    (lambda o: o['cells'][0].update(requests=8000), 'repeat a Qwen'),
    (lambda o: o['cells'][0].update(policy='score'), 'repeat a Qwen'),
    (lambda o: o['repeats'].popitem(), 'repeats must name'),
    (lambda o: o['repeats'][cell_id('mlp_length', 8.3)].update(prefill_tps_ontimeutility='0.48'), 'measured OnTimeUtility'),
    (lambda o: o['repeats'][cell_id('mlp_length', 8.3)].update(cell='qwen-hard-8.3'), 'must repeat'),
])
def test_overlay_rejections(bundle, mutate, match):
    bad = overlay(); mutate(bad)
    with pytest.raises(ValueError, match=match):
        apply_any_campaign(bundle, bad)

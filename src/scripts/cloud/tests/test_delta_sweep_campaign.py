"""The delta sweep overlay: authorized levels, one cell per level, and the per-cell router flag."""
from copy import deepcopy

import pytest

from scripts.cloud.common import ROOT, read
from scripts.cloud.campaigns import apply_any_campaign
from scripts.cloud.delta_sweep_campaign import KIND, LEVELS, POLICY, QPS, REQUESTS, delta_cell_id
from scripts.cloud.tests.test_sfs_score_campaign import _bundle
from scripts.cloud.worker import cell_delta, delta_argv, delta_levels

OVERLAY = ROOT/'scripts/cloud/delta-sweep-campaign-20260920.json'


def overlay():
    return deepcopy(read(OVERLAY))


def test_the_committed_overlay_measures_every_authorized_level():
    m = apply_any_campaign(_bundle(), overlay())
    assert m['kind'] == KIND and m['delta_levels'] == sorted(LEVELS)
    assert {c['id'] for c in m['cells']} == {delta_cell_id(d) for d in LEVELS}
    assert all(c['policy'] == POLICY and c['qps'] == QPS and c['requests'] == REQUESTS for c in m['cells'])
    assert m['families']['qwen']['policies'] == [POLICY] and m['families']['qwen']['qps'] == [QPS]
    # Every delta of the published figure, so the curve is redrawn rather than extended.
    assert set(m['delta_levels']) == set(LEVELS) and len(LEVELS) == 15
    assert {0.0, 1e-5, 5e-3, 1e9} <= set(m['delta_levels'])


def test_the_router_receives_the_cell_delta_only_under_this_kind():
    cell = {'delta_weight': 1e-3}
    assert cell_delta(cell) == 1e-3 and cell_delta({}) is None
    argv = delta_argv(['x'], {'kind': KIND}, 1e-3)
    assert argv[argv.index('--delta-weight')+1] == '0.001'
    assert delta_argv(['x'], {'kind': KIND}, None) == ['x']          # no cell weight, no flag
    with pytest.raises(ValueError, match='delta_sweep overlay'):
        delta_argv(['x'], {'kind': 'sfs_score'}, 1e-3)
    for bad in (-1.0, float('inf')):
        with pytest.raises(ValueError, match='finite and non-negative'):
            delta_argv(['x'], {'kind': KIND}, bad)
    assert delta_levels({'cells': [{'delta_weight': 5e-4}, {'delta_weight': 1e-3}, {}]}) == [5e-4, 1e-3]


@pytest.mark.parametrize('mutate, match', [
    (lambda o: o['cells'][0].update(delta_weight=7e-4), 'must be one of'),
    (lambda o: o['cells'][0].update(requests=16000), 'at 8 QPS'),
    (lambda o: o['cells'][0].update(qps=7.0), 'at 8 QPS'),
    (lambda o: o['cells'][0].update(policy='hard'), 'at 8 QPS'),
    (lambda o: o['cells'][0].update(id='qwen-soft-8-delta9e9'), 'Cell id must be'),
    (lambda o: o['cells'].append(deepcopy(o['cells'][0])), 'at most one cell per level'),
    (lambda o: o.update(policies={'qwen': ['soft', 'hard']}), 'soft smoke policy'),
    (lambda o: o.update(reading='  '), 'how its curve is to be read'),
    (lambda o: o['cells'][0].pop('delta_weight'), 'Unexpected cell fields'),
])
def test_overlay_rejections(mutate, match):
    bad = overlay(); mutate(bad)
    with pytest.raises(ValueError, match=match):
        apply_any_campaign(_bundle(), bad)

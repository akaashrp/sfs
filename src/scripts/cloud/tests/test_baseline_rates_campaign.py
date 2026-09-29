"""The rate-fill overlay must stay the authorized grid: five policies, seven rates, two budgets."""
import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.cloud.baseline_rates_campaign import (BUDGETS, KIND, POLICIES, RATES,
                                                   apply_baseline_rates_campaign, rate_cell_id)
from scripts.cloud.campaigns import apply_any_campaign

OVERLAY = Path(__file__).resolve().parents[3].parent / 'scripts/cloud/baseline-rates-campaign-20260928.json'


def overlay():
    return json.loads(OVERLAY.read_text())


def bundle():
    # Both families, because the overlay's remaining-length block names every model the rule is off for.
    return {'families': {'qwen': {'policies': [], 'qps': [], 'models': ['qwen3-0.6b', 'qwen3-8b', 'qwen3-32b']},
                         'ministral': {'policies': [], 'qps': [],
                                       'models': ['ministral3-3b', 'ministral3-8b', 'ministral3-14b']}},
            'cells': [], 'requests_total': 0}


def test_overlay_matches_the_authorized_grid():
    applied = apply_baseline_rates_campaign(bundle(), overlay())
    assert len(applied['cells']) == len(POLICIES) * len(RATES) == 35
    assert applied['requests_total'] == sum(BUDGETS[float(c['qps'])] for c in applied['cells'])
    assert applied['families']['qwen']['policies'] == list(POLICIES)
    assert applied['families']['qwen']['qps'] == list(RATES)
    assert applied['kind'] == KIND
    # SCORE is in the grid, so the tuned multiplier must travel with the overlay.
    assert applied['score_lambda_weight'] > 0
    assert applied['score_lambda_evidence']['sweep']


def test_dispatcher_routes_the_kind():
    assert apply_any_campaign(bundle(), overlay())['kind'] == KIND


def test_unsaturated_rates_carry_the_short_budget():
    assert {r: BUDGETS[r] for r in (3.0, 4.0, 5.0)} == {3.0: 8000, 4.0: 8000, 5.0: 8000}
    assert all(BUDGETS[r] == 16000 for r in RATES if r >= 8.6)


@pytest.mark.parametrize('mutate, message', [
    (lambda d: d['cells'].pop(), 'every policy at every rate'),
    (lambda d: d['cells'][0].update(requests=16000), 'requests'),
    (lambda d: d['cells'][0].update(qps=7.0), 'authorized rates'),
    (lambda d: d['cells'][0].update(policy='hard'), 'rate-fill policies'),
    (lambda d: d['cells'][0].update(variant='mlp_length'), 'canonical'),
    (lambda d: d['cells'][0].update(id='wrong-id'), 'Cell id must be'),
    (lambda d: d.update(policies={'qwen': ['score']}), 'must declare exactly'),
    (lambda d: d.update(reading=''), 'how its cells join'),
    (lambda d: d.update(requests_total=1), 'disagrees with the cells'),
])
def test_rejects_a_changed_grid(mutate, message):
    doc = deepcopy(overlay())
    mutate(doc)
    with pytest.raises(ValueError, match=message):
        apply_baseline_rates_campaign(bundle(), doc)


def test_duplicate_cell_is_rejected():
    doc = deepcopy(overlay())
    doc['cells'].append(deepcopy(doc['cells'][0]))
    with pytest.raises(ValueError, match='Duplicate'):
        apply_baseline_rates_campaign(bundle(), doc)


def test_cell_ids_follow_the_canonical_pattern():
    assert rate_cell_id('mooncake_prefill', 8.6) == 'qwen-mooncake_prefill-8.6'
    assert rate_cell_id('score', 9.0) == 'qwen-score-9'

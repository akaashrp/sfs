from copy import deepcopy

from scripts.cloud.common import ROOT, read
from scripts.cloud.schedule import QWEN_QPS, update_rates


def test_rate_migration_preserves_artifacts_ministral_and_request_budget():
    campaign = read(ROOT/'scripts/cloud/campaign.json')
    before = deepcopy(campaign)
    for cell in before['cells']:
        if cell['family'] == 'qwen':
            cell['qps'] = {8.: 8.3, 8.75: 8.9}.get(cell['qps'], cell['qps'])
            cell['id'] = cell['id'].rsplit('-', 1)[0] + f'-{cell["qps"]:g}'
    before.update(files={'predictor': 'frozen-hash'}, families={
        'qwen': {'qps': [7., 8.3, 8.6, 8.9], 'experiment_argv': ['--seed', '69']},
        'ministral': {'qps': [6.0125, 7.8625, 8.7875, 9.7125]}})
    saved = deepcopy(before)
    after = update_rates(before)
    assert before == saved
    assert after['cells'] == campaign['cells']
    assert after['files'] == before['files']
    assert after['families']['ministral'] == before['families']['ministral']
    assert after['families']['qwen']['experiment_argv'] == before['families']['qwen']['experiment_argv']
    assert after['families']['qwen']['qps'] == list(QWEN_QPS)
    assert len(after['cells']) == 68 and sum(c['requests'] for c in after['cells']) == 800000
    groups = {}
    for cell in after['cells']:
        if cell['family'] == 'qwen':
            groups.setdefault((cell['variant'], cell['policy']), set()).add(cell['qps'])
    assert len(groups) == 8 and all(rates == set(QWEN_QPS) for rates in groups.values())
    assert update_rates(after) == after

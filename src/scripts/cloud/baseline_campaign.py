"""Explicit authorized cells layered over immutable workload/model artifacts."""
from copy import deepcopy

RATES = {'qwen': (6., 7., 8., 8.3), 'ministral': (6.0125, 7.8625, 8.7875)}
POLICIES = {'qwen': ('lmdeploy_proxy', 'vllm_sr_latency', 'mooncake_prefill', 'routebalance'),
            'ministral': ('shortest_queue', 'vllm_sr_latency', 'mooncake_prefill', 'routebalance')}


def apply_campaign(bundle, campaign):
    result = deepcopy(bundle)
    cells = campaign['cells']
    expected = {(family, policy, rate) for family in RATES for policy in POLICIES[family] for rate in RATES[family]}
    actual = {(c['family'], c['policy'], c['qps']) for c in cells}
    if actual != expected or len(cells) != len(expected):
        raise ValueError('Baseline campaign does not match the authorized grid and policies')
    if len({c['id'] for c in cells}) != len(cells):
        raise ValueError('Duplicate baseline cell IDs')
    for c in cells:
        if c['variant'] != 'canonical' or c['requests'] != (16000 if c['family'] == 'qwen' else 8000):
            raise ValueError('Baseline campaign changed predictor or request budget')
    for family, definition in result['families'].items():
        definition['policies'] = list(POLICIES[family])
        definition['qps'] = list(RATES[family])
    result['cells'] = deepcopy(cells)
    result['requests_total'] = sum(c['requests'] for c in cells)
    return result

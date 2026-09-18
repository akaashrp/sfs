"""Explicit authorized cells layered over immutable workload/model artifacts."""
from copy import deepcopy

# Ministral rates are fractions of K_M = 9.25: 0.65, 0.85, 0.876 and 0.95. 8.1 was authorized by the
# user on 18 September 2026 to locate where SFS stops meeting SLOs, after it led both lower rates and
# collapsed at 8.7875 (scripts/cloud/reports/sfs-saturation-fallback-20260918).
RATES = {'qwen': (6., 7., 8., 8.3), 'ministral': (6.0125, 7.8625, 8.1, 8.7875)}
# LMDeploy, latency-agnostic and round robin are measured for Ministral on Bridges
# (scripts/cloud/bridges-reuse-review-20260916.json) at 6.0125/7.8625/8.7875/9.7125. Only 8.1 postdates
# that campaign, so exactly those three cells are measured here instead, completing the 8.1 row. The
# user accepted the host mix on 18 September 2026; nothing else is promoted from Bridges to Vast.
SUPPLEMENTARY = {'ministral': (('lmdeploy_proxy', 8.1), ('latency_agnostic', 8.1), ('round_robin', 8.1))}
POLICIES = {'qwen': ('lmdeploy_proxy', 'vllm_sr_latency', 'mooncake_prefill', 'routebalance'),
            'ministral': ('shortest_queue', 'vllm_sr_latency', 'mooncake_prefill', 'routebalance')}


def apply_campaign(bundle, campaign):
    result = deepcopy(bundle)
    cells = campaign['cells']
    expected = {(family, policy, rate) for family in RATES for policy in POLICIES[family] for rate in RATES[family]}
    expected |= {(family, policy, rate) for family, extra in SUPPLEMENTARY.items() for policy, rate in extra}
    actual = {(c['family'], c['policy'], c['qps']) for c in cells}
    if actual != expected or len(cells) != len(expected):
        raise ValueError('Baseline campaign does not match the authorized grid and policies')
    if len({c['id'] for c in cells}) != len(cells):
        raise ValueError('Duplicate baseline cell IDs')
    for c in cells:
        if c['variant'] != 'canonical' or c['requests'] != (16000 if c['family'] == 'qwen' else 8000):
            raise ValueError('Baseline campaign changed predictor or request budget')
    for family, definition in result['families'].items():
        # Smoke has to cover the supplementary policies too, or the pool would evaluate a policy it
        # never proved on this host.
        supplementary = tuple(dict.fromkeys(p for p, _ in SUPPLEMENTARY.get(family, ())))
        definition['policies'] = [*POLICIES[family], *(p for p in supplementary if p not in POLICIES[family])]
        definition['qps'] = list(RATES[family])
    result['cells'] = deepcopy(cells)
    result['requests_total'] = sum(c['requests'] for c in cells)
    # Completed cells audited under an earlier, explicitly accepted source pin
    # are skipped rather than rerun; the digest and its reason stay auditable.
    accepted = campaign.get('accepted_prior_source_digests', {})
    if not isinstance(accepted, dict) or not all(isinstance(k, str) and len(k) == 64 and isinstance(v, str) and v for k, v in accepted.items()):
        raise ValueError('accepted_prior_source_digests must map 64-hex source digests to reasons')
    result['accepted_prior_source_digests'] = dict(accepted)
    return result

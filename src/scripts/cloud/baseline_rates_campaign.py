"""Baseline routers at the arrival rates the canonical Qwen grid never measured them at.

Figure 5 plots OnTimeUtility against offered load. SFS, the prefill-throughput estimator,
shortest queue, round robin, latency-agnostic and instance affinity span 3 to 9.5 QPS, but
Mooncake, LMDeploy, RouteBalance, SCORE and vLLM-SR were only ever run at 6, 7, 8 and 8.3,
so every other rate shows a partial baseline set. This overlay measures exactly those five
policies at the missing rates of the published grid.

Request budgets follow the regime. At 3, 4 and 5 QPS no instance saturates and the queue is
stationary, so 8,000 requests measure the same steady state that 16,000 would; at 8.6 and
above the run length does matter and the cells keep the canonical 16,000 so they sit beside
the existing points at those rates.

Everything else is the canonical campaign's own: frozen holdout, explicit TTFT SLOs, the
fixed arrival generator, the 0.6B remaining-length rule and SCORE's tuned multiplier.
"""
from copy import deepcopy

from scripts.cloud.campaigns import apply_tuned_score_lambda
from scripts.cloud.sfs_score_campaign import (accepted_prior_campaign_digests, accepted_prior_source_digests,
                                              resolve_remaining_length)

KIND = 'baseline_rates'
FAMILY = 'qwen'
# The five policies whose canonical rows stop at 8.3.
POLICIES = ('lmdeploy_proxy', 'mooncake_prefill', 'routebalance', 'score', 'vllm_sr_latency')
# Rate -> requests. 3/4/5 are unsaturated, so they carry the shorter budget; 8.6 upward keep
# the canonical 16,000 because run length changes a saturated cell.
BUDGETS = {3.0: 8000, 4.0: 8000, 5.0: 8000, 8.6: 16000, 8.9: 16000, 9.0: 16000, 9.2: 16000}
RATES = tuple(sorted(BUDGETS))
CELL_FIELDS = {'id', 'family', 'variant', 'policy', 'qps', 'requests'}


def rate_cell_id(policy, qps):
    return f'{FAMILY}-{policy}-{qps:g}'


def declared_grid():
    return {(policy, rate) for policy in POLICIES for rate in RATES}


def apply_baseline_rates_campaign(bundle, campaign):
    if campaign.get('kind') != KIND:
        raise ValueError('Not a baseline rate-fill overlay')
    result = deepcopy(bundle)
    cells = campaign.get('cells')
    if not isinstance(cells, list) or not cells:
        raise ValueError('Baseline rate fill needs cells')
    seen = set()
    for c in cells:
        if set(c) != CELL_FIELDS:
            raise ValueError(f'Unexpected cell fields: {sorted(c)}')
        if (c['family'], c['variant']) != (FAMILY, 'canonical'):
            raise ValueError(f'Rate-fill cells run on the canonical {FAMILY} pool: {c["id"]}')
        policy, qps = c['policy'], float(c['qps'])
        if policy not in POLICIES:
            raise ValueError(f'{policy} is not one of the rate-fill policies: {c["id"]}')
        if qps not in BUDGETS:
            raise ValueError(f'{qps:g} QPS is outside the authorized rates {RATES}: {c["id"]}')
        if c['requests'] != BUDGETS[qps]:
            raise ValueError(f'{qps:g} QPS carries {BUDGETS[qps]} requests, not {c["requests"]}: {c["id"]}')
        if c['id'] != rate_cell_id(policy, qps):
            raise ValueError(f'Cell id must be {rate_cell_id(policy, qps)}: {c["id"]}')
        if (policy, qps) in seen:
            raise ValueError(f'Duplicate rate-fill cell: {c["id"]}')
        seen.add((policy, qps))
    # A partial grid would put a baseline on some rates of the figure and not others, which is
    # the defect this overlay exists to remove.
    if seen != declared_grid():
        missing = sorted(declared_grid() - seen)
        raise ValueError(f'Rate fill must cover every policy at every rate; missing {missing}')
    if campaign.get('policies') != {FAMILY: list(POLICIES)}:
        raise ValueError(f'Overlay must declare exactly {list(POLICIES)} for {FAMILY}')
    if not str(campaign.get('reading') or '').strip():
        raise ValueError('The overlay must record how its cells join the published figure')
    definition = result['families'][FAMILY]
    definition['policies'] = list(POLICIES)
    definition['qps'] = list(RATES)
    result['cells'] = deepcopy(cells)
    result['requests_total'] = sum(c['requests'] for c in cells)
    if campaign.get('requests_total') not in (None, result['requests_total']):
        raise ValueError('requests_total disagrees with the cells')
    result['kind'] = KIND
    result['reading'] = campaign['reading']
    result['remaining_length'] = resolve_remaining_length(campaign.get('remaining_length'), result['families'])
    result['accepted_prior_source_digests'] = accepted_prior_source_digests(campaign)
    result['accepted_prior_campaign_digests'] = accepted_prior_campaign_digests(campaign)
    # SCORE is one of the five, so the tuned multiplier has to travel with the overlay or the
    # cells would run a differently-parameterised SCORE from the canonical grid.
    return apply_tuned_score_lambda(result, campaign)

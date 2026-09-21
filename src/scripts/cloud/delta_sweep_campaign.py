"""Utility-latency tradeoff sweep: the soft objective at 8 QPS with the latency penalty delta varied.

The soft policy scores a candidate as accuracy - lambda*cost - delta*wait, so delta buys latency with
utility and traces a frontier. The published curve (Bridges, April) sampled fourteen deltas from 0 to
1e-4 at 8,000 requests; its own data past that cutoff shows the frontier keeps improving to delta 1e-3
(0.5573 utility at 222 ms mean TTFT, against 0.5216 at 4,204 ms for 1e-5) and only turns back beyond
it. This overlay re-measures the region that matters on the current runtime: the two folds, the
interior optimum between them, and the asymptote.

Every other setting is the campaign's own -- frozen holdout, explicit TTFT SLOs, the fixed arrival
generator, refitted coefficients and the 0.6B remaining-length rule -- so the curve is comparable to
the canonical grid whose baselines it is plotted against.
"""
from copy import deepcopy

from scripts.cloud.sfs_score_campaign import (accepted_prior_campaign_digests, accepted_prior_source_digests,
                                              resolve_remaining_length)

KIND = 'delta_sweep'
POLICY = 'soft'
QPS = 8.0
REQUESTS = 8000          # the published figure's point size
FAMILY = 'qwen'
CELL_FIELDS = {'id', 'family', 'variant', 'policy', 'qps', 'requests', 'delta_weight'}
# Exactly the deltas of the published figure (vllm_utils/experiments/delta_plots/
# router_delta_sweep_metrics_summary.json): re-measured on the current runtime so the curve can be
# redrawn rather than extended. delta 0 is the latency-agnostic limit of the soft objective.
LEVELS = (0.0, 1e-7, 3e-7, 4e-7, 5e-7, 6e-7, 8e-7, 1e-6, 3e-6, 4e-6, 5e-6, 8e-6, 1e-5, 5e-3, 1e9)


def delta_cell_id(delta, qps=QPS, policy=POLICY):
    return f'qwen-{policy}-{qps:g}-delta{float(delta):g}'


def apply_delta_sweep_campaign(bundle, campaign):
    if campaign.get('kind') != KIND:
        raise ValueError('Not a delta sweep overlay')
    result = deepcopy(bundle)
    cells = campaign.get('cells')
    if not isinstance(cells, list) or not cells:
        raise ValueError('Delta sweep needs cells')
    seen = []
    for c in cells:
        if set(c) != CELL_FIELDS:
            raise ValueError(f'Unexpected cell fields: {sorted(c)}')
        if (c['family'], c['variant'], c['policy'], c['qps'], c['requests']) != (FAMILY, 'canonical', POLICY, QPS, REQUESTS):
            raise ValueError(f'Delta cells run {POLICY} on the canonical {FAMILY} pool at {QPS:g} QPS '
                             f'and {REQUESTS} requests: {c["id"]}')
        delta = c['delta_weight']
        if isinstance(delta, bool) or not isinstance(delta, (int, float)) or float(delta) not in LEVELS:
            raise ValueError(f'delta_weight must be one of {LEVELS}: {c["id"]}')
        if c['id'] != delta_cell_id(delta):
            raise ValueError(f'Cell id must be {delta_cell_id(delta)}: {c["id"]}')
        seen.append(float(delta))
    if len(set(seen)) != len(seen):
        raise ValueError('Delta sweep holds at most one cell per level')
    if set(seen) - set(LEVELS):
        raise ValueError(f'Delta levels outside the authorized set: {sorted(set(seen) - set(LEVELS))}')
    if campaign.get('policies') != {FAMILY: [POLICY]}:
        raise ValueError(f'Delta overlay must set exactly the {POLICY} smoke policy for {FAMILY}')
    if not str(campaign.get('reading') or '').strip():
        raise ValueError('The overlay must record how its curve is to be read')
    definition = result['families'][FAMILY]
    definition['policies'] = [POLICY]
    definition['qps'] = [QPS]
    result['cells'] = deepcopy(cells)
    result['requests_total'] = sum(c['requests'] for c in cells)
    if campaign.get('requests_total') not in (None, result['requests_total']):
        raise ValueError('requests_total disagrees with the cells')
    result['kind'] = KIND
    result['delta_levels'] = sorted(seen)
    result['reading'] = campaign['reading']
    result['remaining_length'] = resolve_remaining_length(campaign.get('remaining_length'), result['families'])
    result['accepted_prior_source_digests'] = accepted_prior_source_digests(campaign)
    result['accepted_prior_campaign_digests'] = accepted_prior_campaign_digests(campaign)
    return result

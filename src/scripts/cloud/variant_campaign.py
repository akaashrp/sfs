"""Predictor/judge ablations for the predictor-consuming Qwen policies on the cloud grid.

Only hard, hard_prefill_tps, score and latency_agnostic consume a routing-time predictor
(scripts/cloud/reports/variants-20260916/PLAN.md section 1). The other six policies are pure
no-ops for every variant: their canonical cells are reused, labelled `reused_canonical` in the
variant matrix (for flash_quality reported under the Flash judge that collation already computes).
The same per-engine remaining-length rule as the SFS/SCORE overlay applies to every Qwen pool.
"""
from copy import deepcopy

from scripts.cloud.baseline_campaign import RATES
from scripts.cloud.campaigns import apply_tuned_score_lambda

# The ablations drop 6 QPS: the canonical grid shows almost no policy differentiation there,
# so the predictor arms are measured at 7, 8 and 8.3 only (user decision, 18 September 2026).
ABLATION_RATES = (7.0, 8.0, 8.3)
from scripts.cloud.sfs_score_campaign import accepted_prior_source_digests, check_cells, resolve_remaining_length

KIND = 'predictor_variants'
VARIANTS = ('flash_quality', 'mlp_quality', 'mlp_length')
CONSUMING_POLICIES = ('hard', 'hard_prefill_tps', 'score', 'latency_agnostic')
ABLATED_POLICIES = ('hard', 'score', 'latency_agnostic')          # hard_prefill_tps is deferred
NOOP_POLICIES = ('round_robin', 'shortest_queue', 'lmdeploy_proxy', 'mooncake_prefill', 'routebalance', 'vllm_sr_latency')
VARIANT_SENSITIVITY = {
    'hard': 'routing changes (quality/cost terms, pending-dispatch overlay); snapshot targets change under mlp_length',
    'hard_prefill_tps': 'routing changes (quality/cost terms, pending-dispatch overlay); deferred to the fallback backlog',
    'score': 'routing changes (quality/cost/latency terms); snapshot decode backlog changes under mlp_length',
    'latency_agnostic': 'routing changes (quality and cost terms only); reads no snapshots',
    'round_robin': 'predictions logged only; cursor routing with the zero estimator',
    'shortest_queue': 'predictions logged only; selection uses snapshot request counts only',
    'lmdeploy_proxy': 'predictors never loaded (MethodologyScheduler); reads no snapshots',
    'mooncake_prefill': 'predictors never loaded; the snapshot parser ignores num_output_target_tokens',
    'routebalance': 'predictors never loaded (own MiniLM/KNN artifact); snapshot target field ignored',
    'vllm_sr_latency': 'predictors never loaded (LatencyHistoryScheduler); reads no snapshots',
}
COMPARATOR_PROVIDERS = {'hard': 'vast', 'score': 'vast', 'latency_agnostic': 'bridges'}


def variant_cell_id(variant, policy, qps):
    return f'qwen-{variant}-{policy}-{qps:g}'


def apply_variant_campaign(bundle, campaign):
    if campaign.get('kind') != KIND:
        raise ValueError('Not a predictor-variant campaign overlay')
    result = deepcopy(bundle)
    cells = campaign['cells']
    if any(c['family'] != 'qwen' for c in cells):
        raise ValueError('Predictor ablations are Qwen only')
    unknown = {c['variant'] for c in cells} - set(bundle['variants'])
    if unknown:
        raise ValueError(f'Unknown predictor variants: {sorted(unknown)}')
    noop = {c['policy'] for c in cells} & set(NOOP_POLICIES)
    if noop:
        raise ValueError(f'No-op policies belong in noop_policies with reuse pointers, not in cells: {sorted(noop)}')
    expected = {('qwen', variant, policy, rate) for variant in VARIANTS for policy in ABLATED_POLICIES for rate in ABLATION_RATES}
    check_cells(cells, expected, {'qwen': 16000}, KIND)
    noop_policies = campaign.get('noop_policies')
    if (not isinstance(noop_policies, dict) or set(noop_policies) != set(NOOP_POLICIES)
            or not all(isinstance(v, dict) and v.get('reuse') and v.get('reason') for v in noop_policies.values())):
        raise ValueError('noop_policies must give a reuse pointer and reason for each of the six no-op policies')
    comparators = campaign.get('comparators')
    if (not isinstance(comparators, dict) or set(comparators) != set(ABLATED_POLICIES)
            or any(not isinstance(comparators[p], dict) or comparators[p].get('provider') != COMPARATOR_PROVIDERS[p]
                   for p in ABLATED_POLICIES)):
        raise ValueError('comparators must name the canonical source per ablated policy (Vast for hard/score, Bridges for latency_agnostic)')
    if campaign.get('policies') != {'qwen': list(ABLATED_POLICIES)}:
        raise ValueError('Variant overlay must set exactly the hard, score and latency_agnostic smoke policies for Qwen')
    qwen = result['families']['qwen']
    qwen['policies'] = list(ABLATED_POLICIES)
    qwen['qps'] = list(ABLATION_RATES)
    result['cells'] = deepcopy(cells)
    result['requests_total'] = sum(c['requests'] for c in cells)
    if campaign.get('requests_total') not in (None, result['requests_total']):
        raise ValueError('requests_total disagrees with the cells')
    result['kind'] = KIND
    result['noop_policies'] = deepcopy(noop_policies)
    result['comparators'] = deepcopy(comparators)
    result['remaining_length'] = resolve_remaining_length(campaign.get('remaining_length'), result['families'])
    result['accepted_prior_source_digests'] = accepted_prior_source_digests(campaign)
    result = apply_tuned_score_lambda(result, campaign)
    return result

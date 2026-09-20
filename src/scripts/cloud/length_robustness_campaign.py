"""Does SFS's advantage survive a worse output-length predictor?

At 8.3 QPS under the MLP length and MLP quality predictor arms, SFS lost to the external
prefill-TPS estimator (0.4599 against 0.4853, and 0.4661 against 0.4865) while winning at 7 and 8
QPS in both arms. The diagnosis is in the cells themselves: SFS admits a request by simulating each
candidate's wait, and that simulation consumes the router's predicted output lengths, so a worse
length predictor degrades the simulated decode backlog. The prefill-TPS estimator reads prompt
tokens and measured prefill throughput only, and is untouched by the swap. Measured on the same
cells, SFS's wait prediction error rose from 119 ms under the canonical predictors to 338 ms under
mlp_length, and it routed 34.8% of traffic to the 8B (77% attainment) instead of 27%.

This overlay repeats those cells with one change: the engine-side conditional remaining-length fill
is enabled on every Qwen engine instead of the 0.6B alone, so a running request's remaining length
comes from the measured survival table for its prompt bin rather than from the predictor. The
mechanism, the tables and the rule are the ones the canonical overlay already ships; only the set of
engines they apply to changes. If SFS's advantage is restored, the sensitivity is a configuration
gap; if it is not, the sensitivity is intrinsic and belongs in the paper as a limitation.
"""
from copy import deepcopy

from scripts.cloud.sfs_score_campaign import accepted_prior_source_digests, resolve_remaining_length

KIND = 'length_robustness'
POLICY = 'hard'
VARIANTS = ('mlp_length', 'mlp_quality')
RATES = (7.0, 8.0, 8.3)
REQUESTS = 16000
SUFFIX = '-lenfill'
CELL_FIELDS = {'id', 'family', 'variant', 'policy', 'qps', 'requests'}
# The whole point of the overlay: every Qwen engine fills a running request's remaining length from
# its measured table. An overlay that leaves one of them on the predictor is not this experiment.
FILLED_MODELS = ('qwen3-0.6b', 'qwen3-8b', 'qwen3-32b')


def cell_id(variant, qps, policy=POLICY):
    return f'qwen-{variant}-{policy}-{qps:g}{SUFFIX}'


def check_repeats(campaign):
    """Each cell must name the measured cell it repeats and that cell's two published numbers.

    The experiment only means something against what it is compared with, so the overlay carries the
    comparison it will be read against rather than leaving it to a later note.
    """
    repeats = campaign.get('repeats')
    if not isinstance(repeats, dict) or set(repeats) != {cell_id(v, q) for v in VARIANTS for q in RATES}:
        raise ValueError('repeats must name the measured cell each cell of this overlay repeats')
    for cid, entry in repeats.items():
        if not isinstance(entry, dict):
            raise ValueError(f'repeats.{cid} must record the measured cell and its numbers')
        if entry.get('cell') != cid[:-len(SUFFIX)]:
            raise ValueError(f'repeats.{cid} must repeat {cid[:-len(SUFFIX)]}')
        for field in ('sfs_ontimeutility', 'prefill_tps_ontimeutility'):
            value = entry.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f'repeats.{cid}.{field} must be the measured OnTimeUtility it is compared against')
    return deepcopy(repeats)


def apply_length_robustness_campaign(bundle, campaign):
    if campaign.get('kind') != KIND:
        raise ValueError('Not a length-robustness overlay')
    result = deepcopy(bundle)
    cells = campaign.get('cells')
    if not isinstance(cells, list) or not cells:
        raise ValueError('Length-robustness sweep needs cells')
    expected = {(v, q) for v in VARIANTS for q in RATES}
    seen = set()
    for c in cells:
        if set(c) != CELL_FIELDS:
            raise ValueError(f'Unexpected cell fields: {sorted(c)}')
        if (c['family'], c['policy'], c['requests']) != ('qwen', POLICY, REQUESTS):
            raise ValueError(f'Length-robustness cells repeat a Qwen {POLICY} cell at {REQUESTS} requests: {c["id"]}')
        key = (c['variant'], float(c['qps']))
        if key not in expected or key in seen:
            raise ValueError(f'Cell outside the authorized {VARIANTS} x {RATES} grid, or a duplicate: {c["id"]}')
        if c['id'] != cell_id(c['variant'], c['qps']):
            raise ValueError(f'Cell id must be {cell_id(c["variant"], c["qps"])}: {c["id"]}')
        seen.add(key)
    if seen != expected:
        raise ValueError('Length-robustness sweep must hold exactly one cell per variant and rate')
    if campaign.get('policies') != {'qwen': [POLICY]}:
        raise ValueError('Length-robustness overlay must set exactly the hard smoke policy for Qwen')
    if not str(campaign.get('diagnosis') or '').strip():
        raise ValueError('The overlay must record the diagnosis it tests')
    qwen = result['families']['qwen']
    qwen['policies'] = [POLICY]
    qwen['qps'] = sorted(RATES)
    result['cells'] = deepcopy(cells)
    result['requests_total'] = sum(c['requests'] for c in cells)
    if campaign.get('requests_total') not in (None, result['requests_total']):
        raise ValueError('requests_total disagrees with the cells')
    result['kind'] = KIND
    result['repeats'] = check_repeats(campaign)
    result['diagnosis'] = campaign['diagnosis']
    result['remaining_length'] = resolve_remaining_length(campaign.get('remaining_length'), result['families'])
    rules = result['remaining_length']['rules']
    missing = [m for m in FILLED_MODELS if m not in rules]
    if missing:
        raise ValueError(f'This overlay fills every Qwen engine; no rule for {missing}')
    result['accepted_prior_source_digests'] = accepted_prior_source_digests(campaign)
    return result

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

from scripts.cloud.sfs_score_campaign import (accepted_prior_campaign_digests, accepted_prior_source_digests,
                                               resolve_remaining_length)

KIND = 'length_robustness'
POLICY = 'hard'
VARIANTS = ('mlp_length', 'mlp_quality')
# The canonical predictors under the same fill: adopting it campaign-wide would re-measure the headline
# grid, so what it does there has to be known before, not after. The canonical grid runs 6 QPS too.
ARMS = ('canonical', *VARIANTS)
RATES = (7.0, 8.0, 8.3)
CANONICAL_RATES = (6.0, 7.0, 8.0, 8.3)
# Ministral carries the same question at its own rates, with its own engines and its own tables. An
# overlay holds exactly one family; the arms above are Qwen's.
# Only the rate where the external prefill-TPS estimator leads; the user authorized this cell alone
# (2026-09-20) to save GPU time.
MINISTRAL_RATES = (7.8625,)
FAMILY_ARMS = {'qwen': ('canonical', 'mlp_length', 'mlp_quality', 'flash_quality'), 'ministral': ('canonical',)}
# The rates each arm may be measured at; an overlay declares the subset it runs.
FAMILY_RATES = {('qwen', 'canonical'): CANONICAL_RATES, ('qwen', 'mlp_length'): RATES,
                ('qwen', 'mlp_quality'): RATES, ('qwen', 'flash_quality'): RATES,
                ('ministral', 'canonical'): (6.0125, 7.8625, 8.1, 8.7875)}
FAMILY_FILLED = {'qwen': ('qwen3-0.6b', 'qwen3-8b', 'qwen3-32b'),
                 'ministral': ('ministral3-3b', 'ministral3-8b', 'ministral3-14b')}
REQUESTS = 16000
REQUESTS_BY_FAMILY = {'qwen': 16000, 'ministral': 8000}
SUFFIX = '-lenfill'
CELL_FIELDS = {'id', 'family', 'variant', 'policy', 'qps', 'requests'}
# The whole point of the overlay: every Qwen engine fills a running request's remaining length from
# its measured table. An overlay that leaves one of them on the predictor is not this experiment.
FILLED_MODELS = ('qwen3-0.6b', 'qwen3-8b', 'qwen3-32b')


def arm_rates(variant, family='qwen', campaign=None):
    """The rates this overlay measures for one arm: what it declares, else the arm's full grid."""
    allowed = FAMILY_RATES[(family, variant)]
    if campaign is None:
        return MINISTRAL_RATES if family == 'ministral' else (CANONICAL_RATES if variant == 'canonical' else RATES)
    declared = campaign.get('rates')
    values = declared.get(variant) if isinstance(declared, dict) else declared
    if not isinstance(values, list) or not values:
        raise ValueError(f'rates must declare the {variant} rates this overlay measures')
    chosen = tuple(float(v) for v in values)
    if set(chosen) - set(allowed) or len(set(chosen)) != len(chosen):
        raise ValueError(f'{variant} rates must be distinct and within {allowed}: {chosen}')
    return chosen


def declared_arms(campaign, family):
    """The predictor arms this overlay measures; a subset of the family's known arms."""
    arms = campaign.get('predictor_arms')
    if (not isinstance(arms, list) or not arms or len(set(arms)) != len(arms)
            or set(arms) - set(FAMILY_ARMS[family])):
        raise ValueError(f'predictor_arms must be a distinct subset of {FAMILY_ARMS[family]}')
    return tuple(arms)


def cell_id(variant, qps, policy=POLICY, family='qwen'):
    # Canonical cells carry no variant segment, as everywhere else in the campaign.
    prefix = family if variant == 'canonical' else f'{family}-{variant}'
    return f'{prefix}-{policy}-{qps:g}{SUFFIX}'


def campaign_family(campaign):
    """The single family an overlay measures; mixing them would mix engines, tables and rates."""
    families = {c.get('family') for c in campaign.get('cells') or []}
    if len(families) != 1 or not families <= set(FAMILY_ARMS):
        raise ValueError(f'A length-robustness overlay measures exactly one known family: {sorted(families)}')
    return families.pop()


def check_repeats(campaign):
    """Each cell must name the measured cell it repeats and that cell's two published numbers.

    The experiment only means something against what it is compared with, so the overlay carries the
    comparison it will be read against rather than leaving it to a later note.
    """
    family = campaign_family(campaign)
    repeats = campaign.get('repeats')
    expected_ids = {cell_id(v, q, family=family)
                    for v in declared_arms(campaign, family) for q in arm_rates(v, family, campaign)}
    if not isinstance(repeats, dict) or set(repeats) != expected_ids:
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
    family = campaign_family(campaign)
    expected = {(v, q) for v in declared_arms(campaign, family) for q in arm_rates(v, family, campaign)}
    seen = set()
    for c in cells:
        if set(c) != CELL_FIELDS:
            raise ValueError(f'Unexpected cell fields: {sorted(c)}')
        if (c['family'], c['policy']) != (family, POLICY) or c['requests'] != REQUESTS_BY_FAMILY[family]:
            raise ValueError(f'Length-robustness cells repeat a {family} {POLICY} cell at '
                             f'{REQUESTS_BY_FAMILY[family]} requests: {c["id"]}')
        key = (c['variant'], float(c['qps']))
        if key not in expected or key in seen:
            raise ValueError(f'Cell outside the authorized {ARMS} grid, or a duplicate: {c["id"]}')
        if c['id'] != cell_id(c['variant'], c['qps'], family=family):
            raise ValueError(f'Cell id must be {cell_id(c["variant"], c["qps"], family=family)}: {c["id"]}')
        seen.add(key)
    if seen != expected:
        raise ValueError('Length-robustness sweep must hold exactly one cell per variant and rate')
    if campaign.get('policies') != {family: [POLICY]}:
        raise ValueError('Length-robustness overlay must set exactly the hard smoke policy for Qwen')
    if not str(campaign.get('diagnosis') or '').strip():
        raise ValueError('The overlay must record the diagnosis it tests')
    definition = result['families'][family]
    definition['policies'] = [POLICY]
    definition['qps'] = sorted({q for v in declared_arms(campaign, family) for q in arm_rates(v, family, campaign)})
    result['cells'] = deepcopy(cells)
    result['requests_total'] = sum(c['requests'] for c in cells)
    if campaign.get('requests_total') not in (None, result['requests_total']):
        raise ValueError('requests_total disagrees with the cells')
    result['kind'] = KIND
    result['repeats'] = check_repeats(campaign)
    result['diagnosis'] = campaign['diagnosis']
    result['remaining_length'] = resolve_remaining_length(campaign.get('remaining_length'), result['families'], family)
    rules = result['remaining_length']['rules']
    missing = [m for m in FAMILY_FILLED[family] if m not in rules]
    if missing:
        raise ValueError(f'This overlay fills every {family} engine; no rule for {missing}')
    result['accepted_prior_source_digests'] = accepted_prior_source_digests(campaign)
    # Adding the canonical arm leaves the six cells already measured under the earlier overlay untouched.
    result['accepted_prior_campaign_digests'] = accepted_prior_campaign_digests(campaign)
    return result

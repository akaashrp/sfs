"""Canonical SFS (hard) and SCORE cells on the cloud grid, with the flag-gated remaining-length rule.

Mirrors baseline_campaign.apply_campaign strictness: exact grid, budgets, unique canonical ids,
plus a verified `remaining_length` block (committed survival tables with per-file sha256, rules
naming Qwen models only, an explicit list of models kept on the current rule, and the evidence).
"""
from copy import deepcopy
from pathlib import Path

from scripts.cloud.baseline_campaign import RATES
from scripts.cloud.campaigns import apply_tuned_score_lambda
from scripts.cloud.common import ROOT, digest, read

KIND = 'sfs_score'
POLICIES = ('hard', 'score')
BUDGETS = {'qwen': 16000, 'ministral': 8000}
CELL_FIELDS = {'id', 'family', 'variant', 'policy', 'qps', 'requests'}


def cell_id(family, policy, qps):
    return f'{family}-{policy}-{qps:g}'


def accepted_prior_source_digests(campaign):
    accepted = campaign.get('accepted_prior_source_digests', {})
    if not isinstance(accepted, dict) or not all(isinstance(k, str) and len(k) == 64 and isinstance(v, str) and v
                                                 for k, v in accepted.items()):
        raise ValueError('accepted_prior_source_digests must map 64-hex source digests to reasons')
    return dict(accepted)


def resolve_remaining_length(block, families):
    """Verify the committed tables and per-model rules; return the pool-ready block.

    Overlay form: {"tables": repo-relative dir, "files": {name: sha256}, "rules": {model:
    "MODE:QUANTILE:CONDITIONING"}, "off_models": [...], "evidence": {...}}. Every file of the
    directory must be listed with its current sha256, the table manifest must agree, rules may
    only name Qwen models with a committed table, and off_models must be exactly the other models.
    """
    from scripts.cloud.pool import parse_remaining_length_rules
    if not isinstance(block, dict):
        raise ValueError('remaining_length block is required')
    for key in ('tables', 'files', 'rules', 'off_models', 'evidence'):
        if key not in block:
            raise ValueError(f'remaining_length block lacks {key}')
    if Path(block['tables']).is_absolute():
        raise ValueError('remaining_length.tables must be repo-relative')
    tables = (ROOT / block['tables']).resolve()
    if not tables.is_relative_to(ROOT) or not tables.is_dir():
        raise ValueError('remaining_length.tables must be a committed directory')
    files = block['files']
    if not isinstance(files, dict) or 'manifest.json' not in files or {p.name for p in tables.iterdir()} != set(files):
        raise ValueError('remaining_length.files must list every file of the tables directory')
    for name, expected in files.items():
        if digest(tables / name) != expected:
            raise ValueError(f'Remaining-length table changed: {name}')
    manifest = read(tables / 'manifest.json')
    for model, entry in manifest['tables'].items():
        if files.get(f'{model}.json') != entry['sha256']:
            raise ValueError(f'Table manifest disagrees with the overlay hash for {model}')
    qwen = set(families['qwen']['models'])
    rules = block['rules']
    if not isinstance(rules, dict) or not rules:
        raise ValueError('remaining_length.rules must name at least one Qwen model')
    if set(rules) - qwen:
        raise ValueError(f'Remaining-length rules are Qwen only: {sorted(set(rules) - qwen)}')
    parsed = parse_remaining_length_rules([f'{model}={spec}' for model, spec in rules.items()])
    for model, rule in parsed.items():
        if rule['mode'] == 'off':
            raise ValueError(f'Omit {model} from rules instead of an explicit off rule')
        if model not in manifest['tables']:
            raise ValueError(f'No committed table for {model}')
    expected_off = {m for f in families.values() for m in f['models']} - set(rules)
    if not isinstance(block['off_models'], list) or set(block['off_models']) != expected_off or len(block['off_models']) != len(expected_off):
        raise ValueError('off_models must list exactly the models without a rule')
    if not isinstance(block['evidence'], dict) or not block['evidence']:
        raise ValueError('remaining_length.evidence must record the paired-run basis')
    return {'tables': str(tables), 'rules': parsed, 'files': dict(files), 'rule_specs': dict(rules),
            'off_models': sorted(expected_off), 'evidence': deepcopy(block['evidence'])}


def check_cells(cells, expected, budgets, kind):
    """Exact grid membership, unique canonical ids of the <family>[-<variant>]-<policy>-<qps> form, fixed budgets."""
    actual = {(c['family'], c['variant'], c['policy'], c['qps']) for c in cells}
    if actual != expected or len(cells) != len(expected):
        raise ValueError(f'{kind} campaign does not match the authorized grid and policies')
    if len({c['id'] for c in cells}) != len(cells):
        raise ValueError(f'Duplicate {kind} cell IDs')
    for c in cells:
        if set(c) != CELL_FIELDS:
            raise ValueError(f'Unexpected cell fields: {sorted(c)}')
        if c['requests'] != budgets[c['family']]:
            raise ValueError(f'{kind} campaign changed the request budget')
        prefix = c['family'] if c['variant'] == 'canonical' else f"{c['family']}-{c['variant']}"
        if c['id'] != f"{prefix}-{c['policy']}-{c['qps']:g}":
            raise ValueError(f'Cell id must be {prefix}-<policy>-<qps>: {c["id"]}')


def apply_sfs_score_campaign(bundle, campaign):
    if campaign.get('kind') != KIND:
        raise ValueError('Not an SFS/SCORE campaign overlay')
    result = deepcopy(bundle)
    cells = campaign['cells']
    expected = {(family, 'canonical', policy, rate) for family in RATES for policy in POLICIES for rate in RATES[family]}
    check_cells(cells, expected, BUDGETS, KIND)
    if campaign.get('policies') != {family: list(POLICIES) for family in RATES}:
        raise ValueError('SFS/SCORE overlay must set exactly the hard and score smoke policies per family')
    for family, definition in result['families'].items():
        definition['policies'] = list(POLICIES)
        definition['qps'] = list(RATES[family])
    result['cells'] = deepcopy(cells)
    result['requests_total'] = sum(c['requests'] for c in cells)
    if campaign.get('requests_total') not in (None, result['requests_total']):
        raise ValueError('requests_total disagrees with the cells')
    result['kind'] = KIND
    result['remaining_length'] = resolve_remaining_length(campaign.get('remaining_length'), result['families'])
    result['accepted_prior_source_digests'] = accepted_prior_source_digests(campaign)
    result = apply_tuned_score_lambda(result, campaign)
    return result

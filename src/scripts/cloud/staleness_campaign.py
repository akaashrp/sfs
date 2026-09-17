"""Snapshot-staleness sweep: canonical SFS (hard) at 8 QPS with router-side snapshot delays.

Each cell repeats the canonical qwen-hard-8 cell of the SFS/SCORE overlay while the router acts on
engine snapshots delayed by a fixed one-way delay D (``--snapshot-staleness-ms``, injected in the
shared SHM transport). The D = 0 comparator is the canonical cell itself, run under the SFS/SCORE
overlay with the same remaining-length rule; this overlay therefore has no D = 0 cell.
"""
from copy import deepcopy
from pathlib import Path

from scripts.cloud.common import ROOT, read
from scripts.cloud.sfs_score_campaign import accepted_prior_source_digests, resolve_remaining_length

KIND = 'staleness_sweep'
POLICY = 'hard'
QPS = 8.0
REQUESTS = 16000
LEVELS_MS = (25.0, 100.0, 400.0, 1000.0)
CELL_FIELDS = {'id', 'family', 'variant', 'policy', 'qps', 'requests', 'snapshot_staleness_ms'}
COMPARATOR_CELL = 'qwen-hard-8'
COMPARATOR_KIND = 'sfs_score'


def staleness_cell_id(staleness_ms, qps=QPS, policy=POLICY):
    return f'qwen-{policy}-{qps:g}-stale{float(staleness_ms):g}'


def check_comparator(campaign, remaining_length):
    """The D = 0 comparator must be the canonical cell of a committed SFS/SCORE overlay sharing this rule block."""
    comparator = campaign.get('comparator')
    if not isinstance(comparator, dict) or comparator.get('cell') != COMPARATOR_CELL or comparator.get('snapshot_staleness_ms') != 0:
        raise ValueError(f'comparator must name {COMPARATOR_CELL} at snapshot_staleness_ms 0')
    overlay = comparator.get('overlay')
    if not isinstance(overlay, str) or Path(overlay).is_absolute():
        raise ValueError('comparator.overlay must be a repo-relative overlay path')
    path = (ROOT / overlay).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError('comparator.overlay must be a committed overlay')
    source = read(path)
    if source.get('kind') != COMPARATOR_KIND:
        raise ValueError(f'comparator.overlay must be a {COMPARATOR_KIND} overlay')
    cells = [c for c in source['cells'] if c['id'] == COMPARATOR_CELL]
    if len(cells) != 1 or cells[0]['policy'] != POLICY or cells[0]['qps'] != QPS or cells[0]['requests'] != REQUESTS:
        raise ValueError(f'comparator overlay lacks the canonical {COMPARATOR_CELL} cell')
    theirs = source.get('remaining_length', {})
    for key in ('tables', 'files', 'rules', 'off_models'):
        if remaining_length.get(key) != theirs.get(key):
            raise ValueError(f'remaining_length.{key} differs from the comparator overlay')
    if not comparator.get('note'):
        raise ValueError('comparator.note must explain the D = 0 comparison')
    return deepcopy(comparator)


def apply_staleness_campaign(bundle, campaign):
    if campaign.get('kind') != KIND:
        raise ValueError('Not a snapshot-staleness sweep overlay')
    result = deepcopy(bundle)
    cells = campaign['cells']
    if not isinstance(cells, list) or not cells:
        raise ValueError('Staleness sweep needs cells')
    levels = []
    for c in cells:
        if set(c) != CELL_FIELDS:
            raise ValueError(f'Unexpected cell fields: {sorted(c)}')
        if (c['family'], c['variant'], c['policy'], c['qps'], c['requests']) != ('qwen', 'canonical', POLICY, QPS, REQUESTS):
            raise ValueError(f'Staleness cells repeat the canonical qwen {POLICY} {QPS:g} QPS cell: {c["id"]}')
        delay = c['snapshot_staleness_ms']
        if isinstance(delay, bool) or not isinstance(delay, (int, float)) or float(delay) not in LEVELS_MS:
            raise ValueError(f'snapshot_staleness_ms must be one of {LEVELS_MS}: {c["id"]}')
        if c['id'] != staleness_cell_id(delay):
            raise ValueError(f'Cell id must be {staleness_cell_id(delay)}: {c["id"]}')
        levels.append(float(delay))
    if sorted(levels) != sorted(LEVELS_MS) or len(levels) != len(LEVELS_MS):
        raise ValueError(f'Staleness sweep must hold exactly one cell per level {LEVELS_MS}')
    if campaign.get('policies') != {'qwen': [POLICY]}:
        raise ValueError('Staleness overlay must set exactly the hard smoke policy for Qwen')
    qwen = result['families']['qwen']
    qwen['policies'] = [POLICY]
    qwen['qps'] = [QPS]
    result['cells'] = deepcopy(cells)
    result['requests_total'] = sum(c['requests'] for c in cells)
    if campaign.get('requests_total') not in (None, result['requests_total']):
        raise ValueError('requests_total disagrees with the cells')
    result['kind'] = KIND
    result['snapshot_staleness_levels_ms'] = sorted(levels)
    remaining_length = campaign.get('remaining_length')
    result['comparator'] = check_comparator(campaign, remaining_length if isinstance(remaining_length, dict) else {})
    result['remaining_length'] = resolve_remaining_length(remaining_length, result['families'])
    result['accepted_prior_source_digests'] = accepted_prior_source_digests(campaign)
    return result

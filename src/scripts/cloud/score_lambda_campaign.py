"""SCORE Lagrange-multiplier (lambda) tuning probe: short SCORE cells at 7 QPS, one per lambda.

SCORE (Lakha, Yu, Shahout, ICLR 2025 SLLM workshop) picks the destination maximizing
``Q_i - lambda * (w_C * c_i * S_i + w_L * W_i + w_L * s_i * S_i)`` and states that lambda is the knob
that enforces the cost and latency constraints, while w_C and w_L only set their relative importance.
The paper never fixes lambda and its scales differ from ours (quality 1..5 and outputs up to 512 tokens
at 1 request/second with two models, against quality in [0, 1], outputs up to 8192 tokens and 6-8.3 QPS
over three models here), so lambda is the one quantity this overlay varies: w_C = w_L = 1 and every
other SCORE input stays exactly as in the SFS/SCORE overlay.

These cells are a tuning probe, never a reportable point. Each carries ``data_role`` ``tuning_probe``
and a 2,000-request budget (the validator refuses any other budget), each id is prefixed so it can
never collide with a reportable cell id of a committed overlay, and ``scripts.cloud.worker`` writes the
receipts to ``<state>/completed-probes`` instead of the canonical completed ledger.
"""
from copy import deepcopy
from pathlib import Path

from scripts.cloud.common import ROOT, read
from scripts.cloud.sfs_score_campaign import accepted_prior_source_digests, resolve_remaining_length

KIND = 'score_lambda_sweep'
FAMILY = 'qwen'
POLICY = 'score'
QPS = 7.0
REQUESTS = 2000
DATA_ROLE = 'tuning_probe'
ID_PREFIX = 'probe-'
LEVELS = (0.0005, 0.005, 0.05, 0.5)
CONTROL_LAMBDA = 0.0005
CELL_FIELDS = {'id', 'family', 'variant', 'policy', 'qps', 'requests', 'lambda_weight', 'data_role'}
REFERENCE_CELL = 'qwen-score-7'
REFERENCE_KIND = 'sfs_score'
REFERENCE_REQUESTS = 16000
OVERLAY_DIR = ROOT / 'scripts/cloud'


def lambda_tag(value):
    """Filesystem-safe cell-id tag for a lambda: 0.0005 -> 5em4, 0.05 -> 5em2, 1.5 -> 1p5ep0."""
    mantissa, exponent = f'{float(value):e}'.split('e')
    return f"{f'{float(mantissa):g}'.replace('.', 'p')}{'em' if int(exponent) < 0 else 'ep'}{abs(int(exponent))}"


def probe_cell_id(lambda_weight, qps=QPS, policy=POLICY, family=FAMILY):
    return f'{ID_PREFIX}{family}-{policy}-{qps:g}-lambda{lambda_tag(lambda_weight)}'


def reportable_cell_ids():
    """Every cell id named by a committed document of this repo's cloud directory, except this kind's own."""
    ids = set()
    for path in sorted(OVERLAY_DIR.glob('*.json')):
        document = read(path)
        if not isinstance(document, dict) or document.get('kind') == KIND:
            continue
        cells = document.get('cells')
        entries = list(cells.values()) if isinstance(cells, dict) else cells if isinstance(cells, list) else []
        for entry in entries:
            if isinstance(entry, str):
                ids.add(entry)
            elif isinstance(entry, dict) and isinstance(entry.get('id'), str):
                ids.add(entry['id'])
    return ids


def check_probe_ids(cells):
    """No probe may land in the canonical completed ledger under a reportable id."""
    reportable = reportable_cell_ids()
    for cell in cells:
        if not cell['id'].startswith(ID_PREFIX):
            raise ValueError(f'Tuning-probe cell ids must start with {ID_PREFIX!r}: {cell["id"]}')
        if cell['id'] in reportable:
            raise ValueError(f'Probe id collides with a reportable campaign cell: {cell["id"]}')


def check_reference(campaign, remaining_length):
    """The probes stand in for one reportable SCORE cell of a committed SFS/SCORE overlay, sharing its rule block."""
    reference = campaign.get('reference')
    if not isinstance(reference, dict) or reference.get('cell') != REFERENCE_CELL:
        raise ValueError(f'reference must name the {REFERENCE_CELL} cell these probes stand in for')
    if reference.get('lambda_weight') != CONTROL_LAMBDA:
        raise ValueError(f'reference.lambda_weight must be the current campaign value {CONTROL_LAMBDA}')
    overlay = reference.get('overlay')
    if not isinstance(overlay, str) or Path(overlay).is_absolute():
        raise ValueError('reference.overlay must be a repo-relative overlay path')
    path = (ROOT / overlay).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError('reference.overlay must be a committed overlay')
    source = read(path)
    if source.get('kind') != REFERENCE_KIND:
        raise ValueError(f'reference.overlay must be a {REFERENCE_KIND} overlay')
    cells = [c for c in source['cells'] if c['id'] == REFERENCE_CELL]
    if (len(cells) != 1 or cells[0]['policy'] != POLICY or cells[0]['qps'] != QPS
            or cells[0]['requests'] != REFERENCE_REQUESTS or cells[0]['family'] != FAMILY):
        raise ValueError(f'reference overlay lacks the canonical {REFERENCE_CELL} cell')
    theirs = source.get('remaining_length', {})
    for key in ('tables', 'files', 'rules', 'off_models'):
        if remaining_length.get(key) != theirs.get(key):
            raise ValueError(f'remaining_length.{key} differs from the reference overlay')
    if not reference.get('note'):
        raise ValueError('reference.note must explain that these cells are a tuning probe, not a reportable point')
    return deepcopy(reference)


def apply_score_lambda_campaign(bundle, campaign):
    if campaign.get('kind') != KIND:
        raise ValueError('Not a SCORE lambda-sweep overlay')
    result = deepcopy(bundle)
    cells = campaign['cells']
    if not isinstance(cells, list) or not cells:
        raise ValueError('SCORE lambda sweep needs cells')
    levels = []
    for c in cells:
        if set(c) != CELL_FIELDS:
            raise ValueError(f'Unexpected cell fields: {sorted(c)}')
        if (c['family'], c['variant'], c['policy'], c['qps']) != (FAMILY, 'canonical', POLICY, QPS):
            raise ValueError(f'Lambda probes repeat the canonical qwen {POLICY} {QPS:g} QPS cell: {c["id"]}')
        if c['requests'] != REQUESTS or isinstance(c['requests'], bool):
            raise ValueError(f'A lambda probe is exactly {REQUESTS} requests, never a reportable budget: {c["id"]}')
        if c['data_role'] != DATA_ROLE:
            raise ValueError(f'Lambda probe cells must carry data_role {DATA_ROLE!r}: {c["id"]}')
        weight = c['lambda_weight']
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or float(weight) not in LEVELS:
            raise ValueError(f'lambda_weight must be one of {LEVELS}: {c["id"]}')
        if c['id'] != probe_cell_id(weight):
            raise ValueError(f'Cell id must be {probe_cell_id(weight)}: {c["id"]}')
        levels.append(float(weight))
    if sorted(levels) != sorted(LEVELS) or len(levels) != len(LEVELS):
        raise ValueError(f'The sweep must hold exactly one cell per lambda {LEVELS}')
    if CONTROL_LAMBDA not in levels:
        raise ValueError(f'The sweep must keep the current {CONTROL_LAMBDA} value as its control')
    if campaign.get('policies') != {FAMILY: [POLICY]}:
        raise ValueError('Lambda overlay must set exactly the score smoke policy for Qwen')
    check_probe_ids(cells)
    qwen = result['families'][FAMILY]
    qwen['policies'] = [POLICY]
    qwen['qps'] = [QPS]
    result['cells'] = deepcopy(cells)
    result['requests_total'] = sum(c['requests'] for c in cells)
    if campaign.get('requests_total') not in (None, result['requests_total']):
        raise ValueError('requests_total disagrees with the cells')
    result['kind'] = KIND
    result['data_role'] = DATA_ROLE
    result['lambda_weights'] = sorted(levels)
    remaining_length = campaign.get('remaining_length')
    result['reference'] = check_reference(campaign, remaining_length if isinstance(remaining_length, dict) else {})
    result['remaining_length'] = resolve_remaining_length(remaining_length, result['families'])
    result['accepted_prior_source_digests'] = accepted_prior_source_digests(campaign)
    return result

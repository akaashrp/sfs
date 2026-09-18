"""SCORE lambda tuning probe: cells and levels, probe/reportable separation, kind dispatch, router argv, provenance."""
from copy import deepcopy
import sys

import pytest

from scripts.cloud.common import ROOT, read, digest
from scripts.cloud.score_lambda_campaign import (apply_score_lambda_campaign, LEVELS, REQUESTS, DATA_ROLE,
                                                 ID_PREFIX, lambda_tag, probe_cell_id, reportable_cell_ids)
from scripts.cloud.sfs_score_campaign import apply_sfs_score_campaign
from scripts.cloud.staleness_campaign import apply_staleness_campaign
from scripts.cloud.tests.test_sfs_score_campaign import _bundle, OVERLAY as SFS_OVERLAY, BASELINE, SHA

OVERLAY = ROOT/'scripts/cloud/score-lambda-sweep-20260917.json'
STALENESS = ROOT/'scripts/cloud/staleness-sweep-campaign-20260917.json'
IDS = ['probe-qwen-score-7-lambda5em4', 'probe-qwen-score-7-lambda5em3',
       'probe-qwen-score-7-lambda5em2', 'probe-qwen-score-7-lambda5em1']
RULE = 'qwen3-0.6b=running_all_prompt_bin_q50'


def test_lambda_tags_are_unique_and_filesystem_safe():
    assert [lambda_tag(v) for v in LEVELS] == ['5em4', '5em3', '5em2', '5em1']
    assert lambda_tag(1.5) == '1p5ep0' and lambda_tag(2) == '2ep0' and lambda_tag(0.00025) == '2p5em4'
    assert [probe_cell_id(v) for v in LEVELS] == IDS
    assert all(set(i) <= set('abcdefghijklmnopqrstuvwxyz0123456789-.') for i in IDS)


def test_overlay_is_accepted_and_preserves_the_bundle():
    campaign = read(OVERLAY); bundle = _bundle(); saved = deepcopy(bundle)
    active = apply_score_lambda_campaign(bundle, campaign)
    assert bundle == saved and active['files'] == saved['files'] and active['families']['ministral'] == saved['families']['ministral']
    assert [c['id'] for c in active['cells']] == IDS == [probe_cell_id(v) for v in LEVELS]
    assert [c['lambda_weight'] for c in active['cells']] == [0.0005, 0.005, 0.05, 0.5]
    assert all((c['family'], c['variant'], c['policy'], c['qps'], c['requests'], c['data_role'])
               == ('qwen', 'canonical', 'score', 7.0, 2000, DATA_ROLE) for c in active['cells'])
    assert REQUESTS == 2000 and active['requests_total'] == 8000 == campaign['requests_total']
    assert active['kind'] == 'score_lambda_sweep' and active['data_role'] == DATA_ROLE
    assert active['lambda_weights'] == [0.0005, 0.005, 0.05, 0.5] and 0.0005 in active['lambda_weights']
    assert active['families']['qwen']['policies'] == ['score'] and active['families']['qwen']['qps'] == [7.0]
    assert active['reference']['cell'] == 'qwen-score-7' and active['reference']['lambda_weight'] == 0.0005
    assert active['reference']['overlay'] == 'scripts/cloud/sfs-score-campaign-20260917.json'
    sfs = apply_sfs_score_campaign(_bundle(), read(SFS_OVERLAY))
    for key in ('tables', 'files', 'rules', 'rule_specs', 'off_models', 'evidence'):
        assert active['remaining_length'][key] == sfs['remaining_length'][key]
    assert active['remaining_length']['files'] == SHA and active['accepted_prior_source_digests'] == {}


def test_probe_ids_can_never_collide_with_a_reportable_cell():
    reportable = reportable_cell_ids()
    assert 'qwen-score-7' in reportable and 'qwen-hard-8' in reportable and 'qwen-hard-8-stale400' in reportable
    assert all(i.startswith(ID_PREFIX) for i in IDS) and not any(i.startswith(ID_PREFIX) for i in reportable)
    assert not (set(IDS) & reportable)
    for overlay in (SFS_OVERLAY, BASELINE, STALENESS, ROOT/'scripts/cloud/predictor-variants-campaign-20260917.json'):
        assert not ({c['id'] for c in read(overlay)['cells']} & set(IDS))


def _drop(c): c['cells'].pop()
def _dup(c): c['cells'][1] = deepcopy(c['cells'][0])
def _level(c): c['cells'][0]['lambda_weight'] = 0.001; c['cells'][0]['id'] = 'probe-qwen-score-7-lambda1em3'
def _no_control(c): c['cells'][0] = dict(c['cells'][0], lambda_weight=5.0, id='probe-qwen-score-7-lambda5ep0')
def _bool_level(c): c['cells'][0]['lambda_weight'] = True
def _string_level(c): c['cells'][0]['lambda_weight'] = '0.0005'
def _missing_level(c): del c['cells'][0]['lambda_weight']
def _id(c): c['cells'][0]['id'] = 'probe-qwen-score-7-lambda0.0005'
def _unprefixed_id(c): c['cells'][0]['id'] = 'qwen-score-7-lambda5em4'
def _reportable_id(c): c['cells'][0]['id'] = 'qwen-score-7'
def _reportable_budget(c): c['cells'][0]['requests'] = 16000
def _short_budget(c): c['cells'][0]['requests'] = 1000
def _off_by_one_budget(c): c['cells'][0]['requests'] = 1999
def _policy(c): c['cells'][0]['policy'] = 'hard'
def _qps(c): c['cells'][0]['qps'] = 8.0
def _family(c): c['cells'][0]['family'] = 'ministral'
def _variant(c): c['cells'][0]['variant'] = 'mlp_length'
def _role(c): c['cells'][0]['data_role'] = 'evaluation'
def _missing_role(c): del c['cells'][0]['data_role']
def _field(c): c['cells'][0]['seed'] = 1
def _kind(c): c['kind'] = 'sfs_score'
def _smoke(c): c['policies'] = {'qwen': ['hard', 'score']}
def _total(c): c['requests_total'] += 1
def _reference_cell(c): c['reference']['cell'] = 'qwen-score-8'
def _reference_lambda(c): c['reference']['lambda_weight'] = 0.05
def _reference_absolute(c): c['reference']['overlay'] = str(SFS_OVERLAY)
def _reference_baseline(c): c['reference']['overlay'] = 'scripts/cloud/baseline-campaign-20260916.json'
def _reference_missing(c): del c['reference']
def _reference_note(c): c['reference']['note'] = ''
def _rule_differs(c): c['remaining_length']['rules'] = {'qwen3-0.6b': 'running_all:0.9:prompt_bin'}
def _off_differs(c): c['remaining_length']['off_models'].remove('qwen3-32b')
def _hash(c): c['remaining_length']['files']['qwen3-0.6b.json'] = '0'*64
def _no_rl(c): del c['remaining_length']


@pytest.mark.parametrize('mutate', [_drop, _dup, _level, _no_control, _bool_level, _string_level, _missing_level, _id,
                                    _unprefixed_id, _reportable_id, _reportable_budget, _short_budget, _off_by_one_budget,
                                    _policy, _qps, _family, _variant, _role, _missing_role, _field, _kind, _smoke, _total,
                                    _reference_cell, _reference_lambda, _reference_absolute, _reference_baseline,
                                    _reference_missing, _reference_note, _rule_differs, _off_differs, _hash, _no_rl])
def test_overlay_rejections(mutate):
    bad = read(OVERLAY); mutate(bad)
    with pytest.raises(ValueError):
        apply_score_lambda_campaign(_bundle(), bad)
    for other in (SFS_OVERLAY, BASELINE, STALENESS):
        with pytest.raises(ValueError):
            apply_score_lambda_campaign(_bundle(), read(other))


def test_every_budget_but_the_probe_budget_is_refused():
    for budget in (1, 200, 1999, 2001, 8000, 16000, 2000.5):
        bad = read(OVERLAY)
        for cell in bad['cells']:
            cell['requests'] = budget
        bad['requests_total'] = 4*budget if isinstance(budget, int) else None
        with pytest.raises(ValueError, match='never a reportable budget'):
            apply_score_lambda_campaign(_bundle(), bad)
    assert apply_score_lambda_campaign(_bundle(), read(OVERLAY))['cells'][0]['requests'] == 2000


def test_kind_dispatch_selects_the_validator():
    from scripts.cloud.campaigns import apply_any_campaign
    assert apply_any_campaign(_bundle(), read(OVERLAY))['kind'] == 'score_lambda_sweep'
    assert apply_any_campaign(_bundle(), read(SFS_OVERLAY))['kind'] == 'sfs_score'
    assert apply_any_campaign(_bundle(), read(STALENESS))['kind'] == 'staleness_sweep'
    with pytest.raises(ValueError):
        apply_any_campaign(_bundle(), dict(read(SFS_OVERLAY), kind='score_lambda_sweep'))


def test_router_argv_carries_the_cell_lambda_only_under_the_sweep():
    from scripts.cloud.worker import lambda_argv, cell_lambda, lambda_levels, staleness_argv, parse
    active = apply_score_lambda_campaign(_bundle(), read(OVERLAY))
    sfs = apply_sfs_score_campaign(_bundle(), read(SFS_OVERLAY))
    base = ['--utilities', 'score', '--request-rate-qps', '7', '--lambda-weight', '5e-4']
    assert lambda_levels(active) == [0.0005, 0.005, 0.05, 0.5] and lambda_levels(sfs) == []
    assert [cell_lambda(c) for c in active['cells']] == [0.0005, 0.005, 0.05, 0.5] and cell_lambda(sfs['cells'][0]) is None
    for cell in active['cells']:
        argv = lambda_argv(base, active, cell_lambda(cell))
        # The swept value is SCORE's own Lagrange multiplier and moves the routing flag only: the
        # objective every cell is scored on stays at the bundle's --lambda-weight, so probes routed
        # under different multipliers remain comparable on one OnTimeUtility.
        assert argv.count('--score-lambda-weight') == 1 and argv[-2] == '--score-lambda-weight'
        assert parse(argv).score_lambda_weight == float(cell['lambda_weight'])
        assert parse(argv).lambda_weight == 5e-4
        assert parse(argv).utilities == ['score'] and parse(argv).request_rate_qps == 7.0
    # The per-pool smoke and every non-probe overlay keep the bundle's own multiplier.
    assert lambda_argv(base, active, None) == base and parse(lambda_argv(base, active, None)).lambda_weight == 5e-4
    assert parse(base).score_lambda_weight is None
    assert lambda_argv(base, sfs, None) == base
    for bad_manifest in (sfs, {'kind': None, 'cells': []}):
        with pytest.raises(ValueError, match='sweep overlay or an overlay-wide tuned value'):
            lambda_argv(base, bad_manifest, 0.05)
    # A tuned multiplier reaches SCORE cells of an ordinary overlay, and only those cells.
    from scripts.cloud.worker import routing_lambda
    tuned = dict(sfs, score_lambda_weight=0.05)
    score_cell = next(c for c in tuned['cells'] if c['policy'] == 'score')
    hard_cell = next(c for c in tuned['cells'] if c['policy'] == 'hard')
    assert routing_lambda(tuned, score_cell) == 0.05 and routing_lambda(tuned, hard_cell) is None
    assert routing_lambda(sfs, score_cell) is None
    tuned_argv = lambda_argv(base, tuned, routing_lambda(tuned, score_cell))
    assert parse(tuned_argv).score_lambda_weight == 0.05 and parse(tuned_argv).lambda_weight == 5e-4
    for bad in (-1.0, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            lambda_argv(base, active, bad)
    # The lambda flag and the staleness flag stay independent: neither overlay leaks the other's argv.
    assert staleness_argv(lambda_argv(base, active, 0.05), active, None) == lambda_argv(base, active, 0.05)
    with pytest.raises(ValueError, match='staleness_sweep'):
        staleness_argv(base, active, 25.0)


def test_probe_receipts_stay_out_of_the_canonical_ledger(tmp_path):
    from scripts.cloud.worker import ledger_dir, campaign_data_role
    active = apply_score_lambda_campaign(_bundle(), read(OVERLAY))
    sfs = apply_sfs_score_campaign(_bundle(), read(SFS_OVERLAY))
    stale = apply_staleness_campaign(_bundle(), read(STALENESS))
    assert campaign_data_role(active) == DATA_ROLE
    assert ledger_dir(tmp_path, active) == tmp_path/'completed-probes'
    for manifest in (sfs, stale, _bundle()):
        assert campaign_data_role(manifest) == 'evaluation' and ledger_dir(tmp_path, manifest) == tmp_path/'completed'
    assert ledger_dir(tmp_path, active) != ledger_dir(tmp_path, sfs)


def test_completed_records_are_bound_to_their_cell_lambda():
    from scripts.cloud.collate import check_record, expected_rules
    active = apply_score_lambda_campaign(_bundle(), read(OVERLAY))
    expected = {c['id']: c for c in active['cells']}
    rules = expected_rules(active)
    assert rules['qwen'] == RULE
    cell = expected['probe-qwen-score-7-lambda5em2']
    record = {'cell': cell, 'campaign_sha256': digest(OVERLAY), 'remaining_length_rule': RULE,
              'lambda_weight': 0.05, 'data_role': DATA_ROLE}
    assert check_record(record, expected, digest(OVERLAY), 'score_lambda_sweep', rules) == 'probe-qwen-score-7-lambda5em2'
    with pytest.raises(ValueError, match='lambda weight'):
        check_record(dict(record, lambda_weight=0.0005), expected, digest(OVERLAY), 'score_lambda_sweep', rules)
    with pytest.raises(ValueError, match='lambda weight'):
        check_record({k: v for k, v in record.items() if k != 'lambda_weight'}, expected, digest(OVERLAY), 'score_lambda_sweep', rules)
    with pytest.raises(ValueError, match='Unexpected campaign cell'):
        check_record(dict(record, cell=dict(cell, lambda_weight=0.5)), expected, digest(OVERLAY), 'score_lambda_sweep', rules)
    with pytest.raises(ValueError, match='different campaign overlay'):
        check_record(dict(record, campaign_sha256='0'*64), expected, digest(OVERLAY), 'score_lambda_sweep', rules)


def test_worker_main_dispatches_the_sweep_and_refuses_other_pools(tmp_path, monkeypatch):
    from scripts.cloud import worker
    captured = {}
    async def execute(options, manifest, definition, model_paths, output): captured.update(manifest=manifest, definition=definition, options=options)
    monkeypatch.setattr(worker, 'validate_bundle', lambda bundle: _bundle())
    monkeypatch.setattr(worker, 'execute', execute)
    monkeypatch.setattr(worker, 'read', lambda path: {} if str(path).endswith('models.json') else read(path))
    argv = ['worker', 'campaign', '--bundle', str(tmp_path), '--models', str(tmp_path/'models.json'), '--state', str(tmp_path/'state'),
            '--output', str(tmp_path/'out'), '--family', 'qwen', '--gpus', '0,1,2,3', '--campaign', str(OVERLAY), '--cells', ','.join(IDS)]
    monkeypatch.setattr(sys, 'argv', argv)
    worker.main()
    assert captured['manifest']['kind'] == 'score_lambda_sweep' and captured['definition']['policies'] == ['score']
    assert captured['manifest']['data_role'] == DATA_ROLE and captured['manifest']['lambda_weights'] == [0.0005, 0.005, 0.05, 0.5]
    assert captured['options'].cells == ','.join(IDS) and captured['options'].qualification == str((tmp_path/'out').resolve())
    assert read(tmp_path/'out/status.json')['state'] == 'COMPLETE'
    monkeypatch.setattr(sys, 'argv', [a if a != 'qwen' else 'ministral' for a in argv[:-2]] + ['--output', str(tmp_path/'out2')])
    with pytest.raises(SystemExit):
        worker.main()

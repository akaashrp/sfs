"""Staleness-sweep overlay: cells and levels, comparator binding, kind dispatch, router argv, provenance."""
from copy import deepcopy
import sys
from types import SimpleNamespace

import pytest

from scripts.cloud.common import ROOT, read, digest
from scripts.cloud.staleness_campaign import apply_staleness_campaign, LEVELS_MS, staleness_cell_id
from scripts.cloud.sfs_score_campaign import apply_sfs_score_campaign
from scripts.cloud.tests.test_sfs_score_campaign import _bundle, OVERLAY as SFS_OVERLAY, BASELINE, SHA

OVERLAY = ROOT/'scripts/cloud/staleness-sweep-campaign-20260917.json'
IDS = ['qwen-hard-8-stale25', 'qwen-hard-8-stale100', 'qwen-hard-8-stale400', 'qwen-hard-8-stale1000']
RULE = 'qwen3-0.6b=running_all_prompt_bin_q50'


def test_overlay_is_accepted_and_preserves_the_bundle():
    campaign = read(OVERLAY); bundle = _bundle(); saved = deepcopy(bundle)
    active = apply_staleness_campaign(bundle, campaign)
    assert bundle == saved and active['files'] == saved['files'] and active['families']['ministral'] == saved['families']['ministral']
    assert [c['id'] for c in active['cells']] == IDS == [staleness_cell_id(d) for d in LEVELS_MS]
    assert [c['snapshot_staleness_ms'] for c in active['cells']] == [25, 100, 400, 1000]
    assert all((c['family'], c['variant'], c['policy'], c['qps'], c['requests']) == ('qwen', 'canonical', 'hard', 8.0, 16000)
               for c in active['cells'])
    assert active['requests_total'] == 64000 == campaign['requests_total'] and active['kind'] == 'staleness_sweep'
    assert active['snapshot_staleness_levels_ms'] == [25.0, 100.0, 400.0, 1000.0]
    assert active['families']['qwen']['policies'] == ['hard'] and active['families']['qwen']['qps'] == [8.0]
    assert active['comparator']['cell'] == 'qwen-hard-8' and active['comparator']['snapshot_staleness_ms'] == 0
    assert active['comparator']['overlay'] == 'scripts/cloud/sfs-score-campaign-20260917.json'
    sfs = apply_sfs_score_campaign(_bundle(), read(SFS_OVERLAY))
    for key in ('tables', 'files', 'rules', 'rule_specs', 'off_models', 'evidence'):
        assert active['remaining_length'][key] == sfs['remaining_length'][key]
    assert active['remaining_length']['files'] == SHA and active['accepted_prior_source_digests'] == {}
    assert 'stale25' not in {c['id'] for c in sfs['cells']} and 'qwen-hard-8' in {c['id'] for c in sfs['cells']}


def _drop(c): c['cells'].pop()
def _dup(c): c['cells'][1] = deepcopy(c['cells'][0])
def _level(c): c['cells'][0]['snapshot_staleness_ms'] = 50; c['cells'][0]['id'] = 'qwen-hard-8-stale50'
def _bool_level(c): c['cells'][0]['snapshot_staleness_ms'] = True
def _string_level(c): c['cells'][0]['snapshot_staleness_ms'] = '25'
def _zero_cell(c): c['cells'].append(dict(c['cells'][0], id='qwen-hard-8-stale0', snapshot_staleness_ms=0))
def _id(c): c['cells'][0]['id'] = 'qwen-hard-8-stale-25'
def _policy(c): c['cells'][0]['policy'] = 'score'
def _qps(c): c['cells'][0]['qps'] = 8.3
def _budget(c): c['cells'][0]['requests'] = 8000
def _family(c): c['cells'][0]['family'] = 'ministral'
def _variant(c): c['cells'][0]['variant'] = 'mlp_length'
def _field(c): c['cells'][0]['seed'] = 1
def _missing_field(c): del c['cells'][0]['snapshot_staleness_ms']
def _kind(c): c['kind'] = 'sfs_score'
def _smoke(c): c['policies'] = {'qwen': ['hard', 'score']}
def _total(c): c['requests_total'] += 1
def _comparator_cell(c): c['comparator']['cell'] = 'qwen-hard-7'
def _comparator_delay(c): c['comparator']['snapshot_staleness_ms'] = 25
def _comparator_absolute(c): c['comparator']['overlay'] = str(SFS_OVERLAY)
def _comparator_baseline(c): c['comparator']['overlay'] = 'scripts/cloud/baseline-campaign-20260916.json'
def _comparator_missing(c): del c['comparator']
def _comparator_note(c): c['comparator']['note'] = ''
def _rule_differs(c): c['remaining_length']['rules'] = {'qwen3-0.6b': 'running_all:0.9:prompt_bin'}
def _off_differs(c): c['remaining_length']['off_models'].remove('qwen3-32b')
def _hash(c): c['remaining_length']['files']['qwen3-0.6b.json'] = '0'*64
def _no_rl(c): del c['remaining_length']


@pytest.mark.parametrize('mutate', [_drop, _dup, _level, _bool_level, _string_level, _zero_cell, _id, _policy, _qps, _budget, _family,
                                    _variant, _field, _missing_field, _kind, _smoke, _total, _comparator_cell, _comparator_delay,
                                    _comparator_absolute, _comparator_baseline, _comparator_missing, _comparator_note, _rule_differs,
                                    _off_differs, _hash, _no_rl])
def test_overlay_rejections(mutate):
    bad = read(OVERLAY); mutate(bad)
    with pytest.raises(ValueError):
        apply_staleness_campaign(_bundle(), bad)
    for other in (SFS_OVERLAY, BASELINE):
        with pytest.raises(ValueError):
            apply_staleness_campaign(_bundle(), read(other))


def test_kind_dispatch_selects_the_validator():
    from scripts.cloud.campaigns import apply_any_campaign
    assert apply_any_campaign(_bundle(), read(OVERLAY))['kind'] == 'staleness_sweep'
    assert apply_any_campaign(_bundle(), read(SFS_OVERLAY))['kind'] == 'sfs_score'
    with pytest.raises(ValueError):
        apply_any_campaign(_bundle(), dict(read(SFS_OVERLAY), kind='staleness_sweep'))


def test_router_argv_carries_the_cell_delay_only_under_the_sweep():
    from scripts.cloud.worker import staleness_argv, cell_staleness, staleness_levels, smoke_staleness, parse
    active = apply_staleness_campaign(_bundle(), read(OVERLAY))
    sfs = apply_sfs_score_campaign(_bundle(), read(SFS_OVERLAY))
    base = ['--utilities', 'hard', '--request-rate-qps', '8', '--snapshot-staleness-ms', '7']
    assert staleness_levels(active) == [25.0, 100.0, 400.0, 1000.0] and staleness_levels(sfs) == [0.0]
    assert smoke_staleness(active) == [0.0, 1000.0] and smoke_staleness(sfs) == [None]
    assert [cell_staleness(c) for c in active['cells']] == [25.0, 100.0, 400.0, 1000.0] and cell_staleness(sfs['cells'][0]) == 0.0
    for cell in active['cells']:
        argv = staleness_argv(base, active, cell_staleness(cell))
        assert argv.count('--snapshot-staleness-ms') == 1 and argv[-2:] == ['--snapshot-staleness-ms', f"{cell['snapshot_staleness_ms']:g}"]
        assert parse(argv).snapshot_staleness_ms == float(cell['snapshot_staleness_ms'])
    assert staleness_argv(base, active, 0.0)[-2:] == ['--snapshot-staleness-ms', '0'] and parse(staleness_argv(base, active, None)).snapshot_staleness_ms == 0.0
    assert staleness_argv(base, sfs, None) == base and staleness_argv(base, sfs, 0) == base     # other overlays: argv untouched
    with pytest.raises(ValueError, match='staleness_sweep'):
        staleness_argv(base, sfs, 25.0)
    assert parse(['--utilities', 'hard']).snapshot_staleness_ms == 0.0                             # default: injection off
    for bad in ('-1', 'nan', 'inf'):
        with pytest.raises(SystemExit):
            parse(['--utilities', 'hard', '--snapshot-staleness-ms', bad])


def test_run_applies_the_delay_to_every_reused_instance_client():
    from scripts.runs.experiments import _apply_snapshot_staleness
    seen = {}
    clients = {name: SimpleNamespace(set_snapshot_staleness_ms=lambda v, name=name: seen.__setitem__(name, v)) for name in ('a', 'b')}
    _apply_snapshot_staleness(clients, 400)
    assert seen == {'a': 400.0, 'b': 400.0}
    _apply_snapshot_staleness(clients, 0)
    assert seen == {'a': 0.0, 'b': 0.0}
    _apply_snapshot_staleness({'legacy': SimpleNamespace()}, 0)
    with pytest.raises(ValueError, match='cannot inject'):
        _apply_snapshot_staleness({'legacy': SimpleNamespace()}, 25)
    with pytest.raises(ValueError):
        _apply_snapshot_staleness(clients, -5)


def test_completed_records_are_bound_to_their_cell_delay():
    from scripts.cloud.collate import check_record, expected_rules
    active = apply_staleness_campaign(_bundle(), read(OVERLAY))
    expected = {c['id']: c for c in active['cells']}
    rules = expected_rules(active)
    assert rules['qwen'] == RULE
    cell = expected['qwen-hard-8-stale400']
    record = {'cell': cell, 'campaign_sha256': digest(OVERLAY), 'remaining_length_rule': RULE, 'snapshot_staleness_ms': 400.0}
    assert check_record(record, expected, digest(OVERLAY), 'staleness_sweep', rules) == 'qwen-hard-8-stale400'
    with pytest.raises(ValueError, match='snapshot staleness'):
        check_record(dict(record, snapshot_staleness_ms=100.0), expected, digest(OVERLAY), 'staleness_sweep', rules)
    with pytest.raises(ValueError, match='snapshot staleness'):
        check_record({k: v for k, v in record.items() if k != 'snapshot_staleness_ms'}, expected, digest(OVERLAY), 'staleness_sweep', rules)
    with pytest.raises(ValueError, match='Unexpected campaign cell'):
        check_record(dict(record, cell=dict(cell, snapshot_staleness_ms=100)), expected, digest(OVERLAY), 'staleness_sweep', rules)


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
    assert captured['manifest']['kind'] == 'staleness_sweep' and captured['definition']['policies'] == ['hard']
    assert captured['options'].cells == ','.join(IDS) and captured['options'].qualification == str((tmp_path/'out').resolve())
    assert read(tmp_path/'out/status.json')['state'] == 'COMPLETE'
    monkeypatch.setattr(sys, 'argv', [a if a != 'qwen' else 'ministral' for a in argv[:-2]] + ['--output', str(tmp_path/'out2')])
    with pytest.raises(SystemExit):
        worker.main()

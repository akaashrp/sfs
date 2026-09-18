"""FCFS unchunked SFS/SCORE overlay: explicit hard/score sub-grid with the 0.6B remaining-length rule under --profile fcfs."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts.cloud.common import ROOT, read, write, digest
from scripts.cloud.campaigns import apply_any_campaign
from scripts.cloud.fcfs import config as fcfs
from scripts.cloud.fcfs.campaign import apply_campaign, build_sfs_score, AUTHORIZATION_20260917, SFS_SCORE_KIND
from scripts.cloud.pool import instance_argv, instance_config, remaining_length_provenance
from scripts.cloud.sfs_score_campaign import apply_sfs_score_campaign
from scripts.cloud.worker import family_remaining_length, provenance, remaining_length_record
from scripts.runs import qwen_baselines as qwen

OVERLAY = read(ROOT/'scripts/cloud/fcfs/campaign-sfs-score-20260917.json')
MATRIX = read(ROOT/'scripts/cloud/fcfs/campaign-20260916.json')
CANONICAL = read(ROOT/'scripts/cloud/sfs-score-campaign-20260917.json')
TABLES = ROOT/'scripts/cloud/remaining-length-tables-20260917'
RULE = 'qwen3-0.6b=running_all_prompt_bin_q50'
IDS = {f'{fcfs.CONFIG_ID}-{p}-{q:g}' for p in ('hard', 'score') for q in (6, 7, 8, 8.3)}


@pytest.fixture
def bundle(tmp_path):
    write(tmp_path/'bundle.json', {'schema_version': 1, 'models': deepcopy(MATRIX['models']), 'files': {}, 'cells': [{'id': f'c{i}'} for i in range(68)],
        'families': {'qwen': {'profile': deepcopy(qwen.PROFILE), 'models': list(MATRIX['models']), 'qps': [7., 8., 8.6, 8.75], 'policies': ['mooncake_prefill'], 'requests': 16000},
                     'ministral': {'profile': {'max_num_batched_tokens': 32768}, 'models': ['ministral3-3b', 'ministral3-8b', 'ministral3-14b'], 'qps': [6.0125], 'policies': ['hard'], 'requests': 8000}}})
    write(tmp_path/'qwen/bridges_metrics.json', {})
    return tmp_path


def flags(argv):
    out = {}
    for i, token in enumerate(argv):
        if not token.startswith('--'):
            continue
        # Simulation options arrive as a single --option=value token (scripts.runs.service_metrics_config).
        if '=' in token:
            option, _, value = token.partition('=')
            out[option] = value
        else:
            out[token] = argv[i+1] if i+1 < len(argv) and not argv[i+1].startswith('--') else True
    return out


def test_committed_overlay_is_the_generated_authorized_sub_grid(bundle):
    assert build_sfs_score(bundle) == OVERLAY   # the committed overlay is exactly what the generator emits
    assert OVERLAY['kind'] == SFS_SCORE_KIND and OVERLAY['configuration_id'] == fcfs.CONFIG_ID and OVERLAY['profile'] == fcfs.SETTINGS
    assert OVERLAY['full_matrix_authorized'] is True and OVERLAY['authorization'] == AUTHORIZATION_20260917 and OVERLAY['decision']
    assert {c['id'] for c in OVERLAY['cells']} == IDS and len(OVERLAY['cells']) == 8 and OVERLAY['requests_total'] == 128000
    assert all(c['status'] == 'runnable' and c['requests'] == 16000 and c['variant'] == 'canonical' for c in OVERLAY['cells'])
    assert OVERLAY['policies'] == ['hard', 'score'] and OVERLAY['lanes'] == {
        'a': [f'{fcfs.CONFIG_ID}-hard-{q:g}' for q in (6, 7, 8, 8.3)], 'b': [f'{fcfs.CONFIG_ID}-score-{q:g}' for q in (6, 7, 8, 8.3)]}
    # The same tables, hashes and rule as the canonical SFS/SCORE overlay; only Qwen models are listed as off.
    rl, canonical = OVERLAY['remaining_length'], CANONICAL['remaining_length']
    assert (rl['tables'], rl['files'], rl['rules'], rl['evidence']) == (canonical['tables'], canonical['files'], canonical['rules'], canonical['evidence'])
    assert rl['rules'] == {'qwen3-0.6b': 'running_all:0.5:prompt_bin'} and rl['off_models'] == ['qwen3-8b', 'qwen3-32b']
    assert not Path(rl['tables']).is_absolute() and {p.name: digest(p) for p in TABLES.iterdir()} == rl['files']
    # The eight cells are exactly the matrix overlay's blocked cells; the matrix overlay itself is untouched.
    assert IDS == {c['id'] for c in MATRIX['cells'] if c['status'].startswith('BLOCKED')} and MATRIX['full_matrix_authorized'] is True
    assert 'remaining_length' not in MATRIX and len(MATRIX['cells']) == 36


def test_overlay_accepted_in_every_mode_and_resolves_the_rule(bundle):
    m = read(bundle/'bundle.json'); saved = deepcopy(m)
    for mode in ('calibrate', 'qualify', 'run', 'campaign', 'inspect'):
        active = apply_campaign(m, OVERLAY, mode)
        assert m == saved and active['files'] == saved['files'] and active['families']['ministral'] == saved['families']['ministral']
        assert {c['id'] for c in active['cells']} == IDS and active['blocked_cells'] == [] and active['requests_total'] == 128000
        assert active['kind'] == SFS_SCORE_KIND and active['configuration_id'] == fcfs.CONFIG_ID
        q = active['families']['qwen']
        assert q['policies'] == ['hard', 'score'] and q['qps'] == [6., 7., 8., 8.3] and q['configuration_id'] == fcfs.CONFIG_ID
        assert all(q['profile'][k] == v for k, v in fcfs.SETTINGS.items())
        rl = active['remaining_length']
        assert Path(rl['tables']) == TABLES.resolve() and rl['files'] == OVERLAY['remaining_length']['files']
        assert rl['rules'] == {'qwen3-0.6b': {'mode': 'running_all', 'quantile': 0.5, 'conditioning': 'prompt_bin'}}
        assert rl['off_models'] == ['qwen3-32b', 'qwen3-8b'] and rl['evidence'] == CANONICAL['remaining_length']['evidence']
        assert active['accepted_prior_source_digests'] == {}
    assert apply_any_campaign(m, OVERLAY, inspect=True)['kind'] == SFS_SCORE_KIND   # control status / collate inspection path
    assert len(apply_any_campaign(m, MATRIX, inspect=True)['cells']) == 28
    for overlay in (OVERLAY, MATRIX):   # a worker without --profile fcfs never applies an FCFS overlay to a canonical pool
        with pytest.raises(ValueError, match='profile fcfs'):
            apply_any_campaign(m, overlay)
    with pytest.raises(ValueError):   # the canonical validator never accepts the FCFS overlay
        apply_sfs_score_campaign(m, OVERLAY)


def _unauthorized(o): o['full_matrix_authorized'] = False
def _other_policy(o): o['cells'][0].update(policy='round_robin', id=f'{fcfs.CONFIG_ID}-round_robin-6')
def _extra_policy(o): o['cells'].append(dict(o['cells'][0], policy='latency_agnostic', id=f'{fcfs.CONFIG_ID}-latency_agnostic-6')); o['requests_total'] += 16000
def _drop(o): o['cells'].pop(); o['requests_total'] -= 16000
def _rate(o): o['cells'][0]['qps'] = 8.6
def _blocked(o): o['cells'][0]['status'] = 'BLOCKED_TOKEN_LENGTH_VALIDATION'
def _budget(o): o['cells'][0]['requests'] = 8000
def _variant(o): o['cells'][0]['variant'] = 'mlp_length'
def _canonical_id(o): o['cells'][0]['id'] = 'qwen-hard-6'
def _smoke(o): o['policies'] = ['hard']
def _total(o): o['requests_total'] += 1
def _kind(o): o['kind'] = 'sfs_score'
def _profile(o): o['profile'] = dict(fcfs.SETTINGS, chunked_prefill=True)
def _no_rule(o): del o['remaining_length']
def _hash(o): o['remaining_length']['files']['qwen3-0.6b.json'] = '0'*64
def _absolute(o): o['remaining_length']['tables'] = str(TABLES)
def _eight_b(o): o['remaining_length']['rules']['qwen3-8b'] = 'running_all:0.5:prompt_bin'
def _ministral(o): o['remaining_length']['rules']['ministral3-3b'] = 'running_all:0.5:prompt_bin'
def _off(o): o['remaining_length']['off_models'] = ['qwen3-8b']
def _no_rules(o): o['remaining_length']['rules'] = {}; o['remaining_length']['off_models'].append('qwen3-0.6b')
def _evidence(o): o['remaining_length']['evidence'] = {}


@pytest.mark.parametrize('mutate', [_unauthorized, _other_policy, _extra_policy, _drop, _rate, _blocked, _budget, _variant, _canonical_id,
                                    _smoke, _total, _kind, _profile, _no_rule, _hash, _absolute, _eight_b, _ministral, _off, _no_rules, _evidence])
def test_overlay_rejections(bundle, mutate):
    m = read(bundle/'bundle.json'); bad = deepcopy(OVERLAY); mutate(bad)
    with pytest.raises(ValueError):
        apply_campaign(m, bad, 'run')
    if mutate is _unauthorized:   # only the authorization flag is mode dependent: qualify still works, run/campaign refuse
        assert len(apply_campaign(m, bad, 'qualify')['cells']) == 8
        with pytest.raises(ValueError, match='not authorized'):
            apply_campaign(m, bad, 'campaign')
    else:
        with pytest.raises(ValueError):
            apply_campaign(m, bad, 'qualify')


def test_matrix_overlay_still_blocks_hard_and_score_and_carries_no_rule(bundle):
    m = read(bundle/'bundle.json')
    active = apply_campaign(m, MATRIX, 'run')
    assert len(active['cells']) == 28 and set(active['blocked_cells']) == IDS and 'remaining_length' not in active and 'kind' not in active
    assert not any(c['policy'] in fcfs.BLOCKED for c in active['cells']) and active['families']['qwen']['policies'] == list(fcfs.METHODS)
    assert family_remaining_length(active, active['families']['qwen']) is None
    assert remaining_length_provenance(active['families']['qwen'], None)['rule'] == 'current'
    unblocked = deepcopy(MATRIX); unblocked['cells'][0]['status'] = 'runnable'
    with pytest.raises(ValueError, match='stay blocked'):
        apply_campaign(m, unblocked, 'run')
    with_rule = deepcopy(MATRIX); with_rule['remaining_length'] = deepcopy(OVERLAY['remaining_length'])
    with pytest.raises(ValueError, match='no remaining-length rule'):
        apply_campaign(m, with_rule, 'qualify')
    only_eight = deepcopy(MATRIX); only_eight['cells'] = [c for c in MATRIX['cells'] if c['policy'] in fcfs.BLOCKED]; only_eight['requests_total'] = 128000
    with pytest.raises(ValueError, match='nine-policy'):
        apply_campaign(m, only_eight, 'qualify')
    with pytest.raises(ValueError, match='Unknown FCFS overlay kind'):
        apply_campaign(m, dict(MATRIX, kind='mystery'), 'qualify')


def test_rule_reaches_only_the_small_engine_under_the_fcfs_profile(bundle, tmp_path):
    active = apply_campaign(read(bundle/'bundle.json'), OVERLAY, 'qualify')
    q = active['families']['qwen']
    block = family_remaining_length(active, q)
    assert block == {'tables': str(TABLES.resolve()), 'rules': active['remaining_length']['rules']}
    prov = remaining_length_provenance(q, block)
    assert prov['rule'] == RULE and prov['models']['qwen3-0.6b']['table']['sha256'] == OVERLAY['remaining_length']['files']['qwen3-0.6b.json']
    assert all(prov['models'][m]['rule'] == 'current' and 'table' not in prov['models'][m] for m in ('qwen3-8b', 'qwen3-32b'))
    fitted = {m: dict(zip(qwen.COEFFICIENT_NAMES, (1., 2., 3., 4., 5., 6.))) for m in MATRIX['models']}
    cfg = instance_config('qwen', None, bundle, (9100, 9101, 9102), 'iso', 'fcfs', fitted)
    assert cfg['configuration_id'] == fcfs.CONFIG_ID and cfg['coefficient_status'] == 'FITTED_FOR_CONFIGURATION'
    for i, row in enumerate(cfg['instances']):
        argv = instance_argv('qwen', tmp_path/'model', row, i, tmp_path, bundle/'qwen/length', bundle, 'fcfs', prov)
        off = instance_argv('qwen', tmp_path/'model', row, i, tmp_path, bundle/'qwen/length', bundle, 'fcfs')
        got, base = flags(argv), flags(off)
        assert base['--no-enable-chunked-prefill'] is True and base['--scheduling-policy'] == 'fcfs' and base['--max-num-batched-tokens'] == '65536'
        assert float(got['--simulation-intercept']) == 1.0 and len([t for t in argv if t.startswith('--')]) == len(got)
        if row['model_id'] == 'qwen3-0.6b':
            assert {k: v for k, v in got.items() if not k.startswith('--remaining-length')} == base
            assert got['--remaining-length-mode'] == 'running_all' and got['--remaining-length-table'] == str(TABLES.resolve()/'qwen3-0.6b.json')
            assert got['--remaining-length-quantile'] == '0.5' and got['--remaining-length-conditioning'] == 'prompt_bin'
        else:
            assert argv == off and not any(t.startswith('--remaining-length') for t in argv)
    assert remaining_length_record(prov)['models'] == {'qwen3-0.6b': {'rule': 'running_all_prompt_bin_q50', 'table_sha256': OVERLAY['remaining_length']['files']['qwen3-0.6b.json']},
                                                        'qwen3-8b': {'rule': 'current', 'table_sha256': None}, 'qwen3-32b': {'rule': 'current', 'table_sha256': None}}


def test_worker_main_applies_the_overlay_under_the_fcfs_profile(bundle, tmp_path, monkeypatch):
    from scripts.cloud import worker
    captured = {}
    async def execute(options, manifest, definition, model_paths, output): captured.update(manifest=manifest, definition=definition, options=options)
    monkeypatch.setattr(worker, 'execute', execute)
    monkeypatch.setattr(worker, 'read', lambda path: {} if str(path).endswith('models.json') else read(path))
    overlay = ROOT/'scripts/cloud/fcfs/campaign-sfs-score-20260917.json'
    common = ['--bundle', str(bundle), '--models', str(tmp_path/'models.json'), '--state', str(tmp_path/'state'), '--family', 'qwen', '--gpus', '0,1,2,3',
              '--profile', 'fcfs', '--coefficients', str(tmp_path/'coefficients.json'), '--campaign', str(overlay)]
    for mode, extra in (('qualify', []), ('run', ['--qualification', str(tmp_path/'q')])):
        monkeypatch.setattr(sys, 'argv', ['worker', mode, '--output', str(tmp_path/mode), *extra, *common])
        worker.main()
        manifest, definition = captured['manifest'], captured['definition']
        assert manifest['kind'] == SFS_SCORE_KIND and {c['id'] for c in manifest['cells']} == IDS and definition['policies'] == ['hard', 'score']
        assert family_remaining_length(manifest, definition)['rules'] == {'qwen3-0.6b': {'mode': 'running_all', 'quantile': 0.5, 'conditioning': 'prompt_bin'}}
        assert read(tmp_path/mode/'status.json')['state'] == {'qualify': 'QUALIFIED_AWAITING_REVIEW', 'run': 'COMPLETE'}[mode]
    # Without the profile the FCFS overlay would land on a canonical chunked pool; it is refused before any output exists.
    monkeypatch.setattr(sys, 'argv', ['worker', 'qualify', '--output', str(tmp_path/'plain'), '--bundle', str(bundle), '--models', str(tmp_path/'models.json'),
                                      '--state', str(tmp_path/'state'), '--family', 'qwen', '--gpus', '0,1,2,3', '--campaign', str(overlay)])
    with pytest.raises(ValueError, match='profile fcfs'):
        worker.main()
    assert not (tmp_path/'plain').exists()
    unauthorized = deepcopy(OVERLAY); unauthorized['full_matrix_authorized'] = False; write(tmp_path/'unauthorized.json', unauthorized)
    monkeypatch.setattr(sys, 'argv', ['worker', 'run', '--output', str(tmp_path/'blocked'), '--qualification', str(tmp_path/'q'),
                                      *[a if a != str(overlay) else str(tmp_path/'unauthorized.json') for a in common]])
    with pytest.raises(ValueError, match='not authorized'):
        worker.main()
    assert not (tmp_path/'blocked').exists()


def test_provenance_and_release_bind_configuration_coefficients_overlay_and_rule(bundle, tmp_path, monkeypatch):
    from scripts.cloud import worker
    monkeypatch.setattr(worker, 'hardware', lambda gpus: {'gpus': gpus})
    write(tmp_path/'coefficients.json', {'fitted': True}); write(tmp_path/'other.json', {'fitted': False})
    overlay = ROOT/'scripts/cloud/fcfs/campaign-sfs-score-20260917.json'
    active = apply_campaign(read(bundle/'bundle.json'), OVERLAY, 'run')
    q = active['families']['qwen']
    rule = remaining_length_provenance(q, family_remaining_length(active, q))
    options = SimpleNamespace(family='qwen', variant='canonical', bundle=str(bundle), gpus='0,1,2,3', profile='fcfs',
                              coefficients=str(tmp_path/'coefficients.json'), campaign=str(overlay))
    record = provenance(options, active, rule)
    assert record == {'configuration_id': fcfs.CONFIG_ID, 'coefficient_policy': 'refit', 'coefficients_sha256': digest(tmp_path/'coefficients.json'), 'campaign_sha256': digest(overlay),
                      'campaign_kind': SFS_SCORE_KIND, 'remaining_length_rule': RULE, 'remaining_length': remaining_length_record(rule)}
    assert record['remaining_length']['models']['qwen3-0.6b']['table_sha256'] == OVERLAY['remaining_length']['files']['qwen3-0.6b.json']
    qdir = tmp_path/'q'; write(qdir/'evidence.json', {'ok': True})
    report = {'status': 'GPU_MEASURED_REVIEW_REQUIRED', 'family': 'qwen', 'variant': 'canonical', 'source_sha256': 's', 'bundle_sha256': digest(bundle/'bundle.json'),
              'hardware': {'gpus': ['0', '1', '2', '3']}, 'files': {'evidence.json': digest(qdir/'evidence.json')}, **record}
    def publish(extra):
        write(qdir/'qualification.json', {**report, **extra})
        write(qdir/'release.json', {'status': 'RELEASED', 'qualification_sha256': digest(qdir/'qualification.json'), 'timing_review': 't'*40, 'load_review': 'l'*40})
    publish({})
    worker.validate_release(qdir, options, 's', RULE)
    with pytest.raises(ValueError, match='remaining-length'):   # a pool on the current rule may not run cells qualified with the rule
        worker.validate_release(qdir, options, 's', 'current')
    for change in ({'profile': 'canonical', 'coefficients': None}, {'coefficients': str(tmp_path/'other.json')}, {'gpus': '4,5,6,7'}):
        with pytest.raises(ValueError, match='qualification'):
            worker.validate_release(qdir, SimpleNamespace(**dict(vars(options), **change)), 's', RULE)
    with pytest.raises(ValueError, match='campaign changed'):   # the matrix overlay cannot reuse an SFS/SCORE qualification
        worker.validate_release(qdir, SimpleNamespace(**dict(vars(options), campaign=str(ROOT/'scripts/cloud/fcfs/campaign-20260916.json'))), 's', RULE)
    publish({'remaining_length_rule': 'current'})
    with pytest.raises(ValueError, match='remaining-length'):
        worker.validate_release(qdir, options, 's', RULE)

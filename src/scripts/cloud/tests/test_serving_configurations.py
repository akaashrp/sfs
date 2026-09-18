"""Additional Qwen serving configurations (chunk8192, prefix_cache): profile table, argv deltas, overlays, coefficients policy, worker CLI."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.cloud.common import ROOT, read, write, digest
from scripts.cloud.campaigns import apply_any_campaign
from scripts.cloud.fcfs import config as fcfs
from scripts.cloud.fcfs.campaign import apply_campaign as apply_fcfs
from scripts.cloud.pool import instance_argv, instance_config, remaining_length_provenance
from scripts.cloud.serving import campaign, coefficients, profiles
from scripts.cloud.serving.gpu_smoke import evidence, run as smoke_run
from scripts.cloud.serving.profiles import CHUNK8192, PREFIX_CACHE, FCFS, PROFILES, NAMES, CANONICAL
from scripts.cloud.sfs_score_campaign import apply_sfs_score_campaign
from scripts.cloud.worker import configuration, coefficient_policy, family_remaining_length, provenance, remaining_length_record
from scripts.runs import qwen_baselines as qwen

OVERLAYS = {p: read(campaign.OVERLAYS[p]) for p in ('chunk8192', 'prefix_cache')}
FCFS_MATRIX = read(ROOT/'scripts/cloud/fcfs/campaign-20260916.json')
FCFS_SFS_SCORE = read(ROOT/'scripts/cloud/fcfs/campaign-sfs-score-20260917.json')
CANONICAL_SFS_SCORE = read(ROOT/'scripts/cloud/sfs-score-campaign-20260917.json')
TABLES = ROOT/'scripts/cloud/remaining-length-tables-20260917'
RULE = 'qwen3-0.6b=running_all_prompt_bin_q50'
RATES = (6, 7, 8, 8.3)
PORTS = (9100, 9101, 9102)


def ids(profile, policies=campaign.POLICIES):
    return [f'{profile.configuration_id}-{p}-{q:g}' for p in policies for q in RATES]


@pytest.fixture
def bundle(tmp_path):
    write(tmp_path/'bundle.json', {'schema_version': 1, 'models': deepcopy(FCFS_MATRIX['models']), 'files': {}, 'cells': [{'id': f'c{i}'} for i in range(68)],
        'families': {'qwen': {'profile': deepcopy(qwen.PROFILE), 'models': list(FCFS_MATRIX['models']), 'qps': [7., 8., 8.6, 8.75], 'policies': ['mooncake_prefill'], 'requests': 16000},
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


def trace(path, c, rows=400, noise=2e-5, seed=1, start=1e9):
    rng = np.random.default_rng(seed)
    with path.open('w') as f:
        f.write('ts,engine,prefill,prefill_sq_sum,decode,decode_sq_sum,total,sched,exec,interval,num_seqs,sum_tokens,sum_sq_tokens,avg_tokens,max_tokens,prefill_x_processed_ctx_sum\n')
        for i in range(rows):
            p = int(rng.choice([0, 0, 64, 512, 2048, 4096, 8192])); d = int(rng.integers(0, 128)) if rng.random() < .9 else 0
            d = d or (0 if p else 1); n = d+(p > 0); s = p+d*int(rng.integers(200, 4000)); ssq = int(s*s/n)
            y = c['intercept']+c['prefill_coeff']*p+c['prefill_sq_coeff']*p*p+c['decode_coeff']*d+c['sum_coeff']*s+c['sum_sq_coeff']*ssq+rng.normal(0, noise)
            f.write(f'{start+i},0,{p},{p*p},{d},{d},{p+d},0.0004,{max(y,1e-4):.6f},0,{n},{s},{ssq},{s/n:.3f},{s},0\n')


def test_profile_table():
    assert NAMES == ('canonical', 'fcfs', 'chunk8192', 'prefix_cache') and set(PROFILES) == {'fcfs', 'chunk8192', 'prefix_cache'}
    assert FCFS.configuration_id == fcfs.CONFIG_ID and FCFS.settings == fcfs.SETTINGS and FCFS.coefficient_policy == 'refit' and FCFS.admission_audit
    # Canonical reference row agrees with the canonical Qwen profile; each ablation changes exactly the named setting.
    assert all(CANONICAL[k] == qwen.PROFILE[k] for k in ('chunked_prefill', 'max_num_batched_tokens', 'max_model_len', 'max_num_seqs', 'prefix_caching', 'gpu_memory_utilization'))
    assert CHUNK8192.configuration_id == 'qwen-chunk8192' and CHUNK8192.changed == {'max_num_batched_tokens': 8192} and CHUNK8192.settings['chunked_prefill'] is True
    assert CHUNK8192.coefficient_policy == 'refit' and not CHUNK8192.admission_audit and CHUNK8192.bounded_smoke == 'chunk_bound'
    assert PREFIX_CACHE.configuration_id == 'qwen-prefix-cache' and PREFIX_CACHE.changed == {'prefix_caching': True} and PREFIX_CACHE.settings['max_num_batched_tokens'] == 32768
    assert PREFIX_CACHE.coefficient_policy == 'canonical' and not PREFIX_CACHE.admission_audit and PREFIX_CACHE.bounded_smoke is None
    assert CHUNK8192.snapshot_config == {'chunked_prefill_enabled': True, 'max_num_batched_tokens': 8192, 'max_num_seqs': 512, 'max_model_len': 131072, 'long_prefill_token_threshold': 0, 'policy': 'fcfs'}
    assert FCFS.snapshot_config['chunked_prefill_enabled'] is False and FCFS.snapshot_config['max_num_batched_tokens'] == 65536
    assert profiles.for_configuration('qwen-chunk8192') is CHUNK8192 and profiles.profile('prefix_cache') is PREFIX_CACHE
    for bad in (lambda: profiles.profile('canonical'), lambda: profiles.profile('chunk4096'), lambda: profiles.for_configuration('canonical')):
        with pytest.raises(ValueError):
            bad()
    for name, cid, policy in (('canonical', 'canonical', 'canonical'), ('fcfs', fcfs.CONFIG_ID, 'refit'), ('chunk8192', 'qwen-chunk8192', 'refit'), ('prefix_cache', 'qwen-prefix-cache', 'canonical')):
        options = SimpleNamespace(profile=name)
        assert configuration(options) == cid and coefficient_policy(options) == policy
    assert configuration(SimpleNamespace()) == 'canonical'


def test_argv_deltas_reach_every_qwen_server(bundle, tmp_path):
    reference = instance_config('qwen', {'profile': qwen.PROFILE}, bundle, PORTS, 'iso')
    for profile in (CHUNK8192, PREFIX_CACHE):
        cfg = instance_config('qwen', None, bundle, PORTS, 'iso', profile.name)
        assert cfg['configuration_id'] == profile.configuration_id and cfg['coefficient_policy'] == profile.coefficient_policy
        assert cfg['instance_costs'] == reference['instance_costs'] and all(cfg['serving_profile'][k] == v for k, v in profile.settings.items())
        assert cfg['serving_profile']['rope_scaling'] == qwen.PROFILE['rope_scaling']
        for i, (row, base) in enumerate(zip(cfg['instances'], reference['instances'])):
            assert row['ttft_batch_model'] == base['ttft_batch_model'] and row['address'] == base['address'] and row['snapshot_shm_name'] == base['snapshot_shm_name']
            assert {k: v for k, v in row.items() if k not in profile.rows} == {k: v for k, v in base.items() if k not in profile.rows}
            assert all(row[k] == v for k, v in profile.rows.items())
            argv = instance_argv('qwen', tmp_path/'model', row, i, tmp_path, bundle/'qwen/length', bundle, profile.name)
            canonical = instance_argv('qwen', tmp_path/'model', base, i, tmp_path, bundle/'qwen/length', bundle)
            assert argv[0] == canonical[0] == sys.executable and len([t for t in argv if t.startswith('--')]) == len(flags(argv))  # every option exactly once
            got, expected = flags(argv), flags(canonical)
            if profile is CHUNK8192:
                expected['--max-num-batched-tokens'] = '8192'
                assert row['max_num_batched_tokens'] == 8192 and row['chunked_prefill_enabled'] is True and row['long_prefill_token_threshold'] == 0
                assert got['--enable-chunked-prefill'] is True and got['--no-enable-prefix-caching'] is True and '--long-prefill-token-threshold' not in got
            else:
                del expected['--no-enable-prefix-caching']; expected['--enable-prefix-caching'] = True; expected['--enable-prompt-tokens-details'] = True
                assert row['max_num_batched_tokens'] == 32768 and row['prefix_caching_enabled'] is True and got['--max-num-batched-tokens'] == '32768'
                assert '--no-enable-prefix-caching' not in argv and got['--enable-chunked-prefill'] is True
            assert got == expected and got['--tensor-parallel-size'] == str((1, 1, 2)[i]) and got['--max-model-len'] == '131072'
            assert float(got['--simulation-intercept']) == base['ttft_batch_model']['intercept']   # canonical coefficients until a refit is passed
        # The FCFS delta is untouched by the generalization.
        fcfs_argv = flags(instance_argv('qwen', tmp_path/'model', cfg['instances'][0], 0, tmp_path, bundle/'qwen/length', bundle, 'fcfs'))
        assert fcfs_argv['--no-enable-chunked-prefill'] is True and fcfs_argv['--max-num-batched-tokens'] == '65536' and fcfs_argv['--scheduling-policy'] == 'fcfs'


def test_remaining_length_rule_reaches_only_the_small_engine_under_each_profile(bundle, tmp_path):
    for profile in (CHUNK8192, PREFIX_CACHE):
        active = campaign.apply_campaign(profile, read(bundle/'bundle.json'), OVERLAYS[profile.name], 'qualify')
        q = active['families']['qwen']
        prov = remaining_length_provenance(q, family_remaining_length(active, q))
        assert prov['rule'] == RULE and prov['models']['qwen3-0.6b']['table']['sha256'] == CANONICAL_SFS_SCORE['remaining_length']['files']['qwen3-0.6b.json']
        cfg = instance_config('qwen', None, bundle, PORTS, 'iso', profile.name)
        for i, row in enumerate(cfg['instances']):
            argv = instance_argv('qwen', tmp_path/'model', row, i, tmp_path, bundle/'qwen/length', bundle, profile.name, prov)
            off = instance_argv('qwen', tmp_path/'model', row, i, tmp_path, bundle/'qwen/length', bundle, profile.name)
            got, base = flags(argv), flags(off)
            assert len([t for t in argv if t.startswith('--')]) == len(got)
            if row['model_id'] == 'qwen3-0.6b':
                assert {k: v for k, v in got.items() if not k.startswith('--remaining-length')} == base
                assert got['--remaining-length-mode'] == 'running_all' and got['--remaining-length-table'] == str(TABLES.resolve()/'qwen3-0.6b.json')
                assert got['--remaining-length-quantile'] == '0.5' and got['--remaining-length-conditioning'] == 'prompt_bin'
            else:
                assert argv == off and not any(t.startswith('--remaining-length') for t in argv)
        assert remaining_length_record(prov)['models']['qwen3-8b'] == {'rule': 'current', 'table_sha256': None}


def test_committed_overlays_are_the_generated_reduced_grids(bundle):
    for name, overlay in OVERLAYS.items():
        profile = PROFILES[name]
        assert campaign.build(profile, bundle) == overlay   # the committed overlay is exactly what the generator emits
        assert overlay['kind'] == campaign.KIND and overlay['configuration_id'] == profile.configuration_id and overlay['profile'] == profile.settings
        assert overlay['coefficient_policy'] == profile.coefficient_policy and overlay['changed_from_canonical'] == profile.changed
        assert overlay['full_matrix_authorized'] is True and overlay['authorization'] == campaign.AUTHORIZATION
        assert overlay['authorization'] == ('User authorized on 17 September 2026: reduced grids (SFS + two strongest baselines) for the prefix-caching '
                                            'and fine-chunk serving configurations')
        assert overlay['policies'] == ['hard', 'mooncake_prefill', 'lmdeploy_proxy'] and overlay['policy_rule'] == campaign.POLICY_RULE
        assert overlay['rates'] == [6., 7., 8., 8.3] and overlay['requests_per_cell'] == 16000 and 'cells' not in overlay and 'requests_total' not in overlay
        assert overlay['models'] == FCFS_MATRIX['models'] and overlay['evaluation'] == FCFS_MATRIX['evaluation'] and overlay['restrictions'] == FCFS_MATRIX['restrictions']
        rl, canonical = overlay['remaining_length'], CANONICAL_SFS_SCORE['remaining_length']
        assert (rl['tables'], rl['files'], rl['rules'], rl['evidence']) == (canonical['tables'], canonical['files'], canonical['rules'], canonical['evidence'])
        assert rl['rules'] == {'qwen3-0.6b': 'running_all:0.5:prompt_bin'} and rl['off_models'] == ['qwen3-8b', 'qwen3-32b']
        assert {p.name: digest(p) for p in TABLES.iterdir()} == rl['files'] and overlay['accepted_prior_source_digests'] == {}
        delta = overlay['server_argv_delta']
        assert delta == {'boolean_swaps': [list(s) for s in profile.boolean_swaps], 'options': [[f, v] for f, v in profile.options],
                         'extra_argv': list(profile.extra_argv), 'instance_rows': profile.rows}
    assert OVERLAYS['chunk8192']['server_argv_delta'] == {'boolean_swaps': [], 'options': [['--max-num-batched-tokens', 8192]], 'extra_argv': [], 'instance_rows': {'max_num_batched_tokens': 8192}}
    assert OVERLAYS['prefix_cache']['server_argv_delta'] == {'boolean_swaps': [['--no-enable-prefix-caching', '--enable-prefix-caching']], 'options': [],
                                                             'extra_argv': ['--enable-prompt-tokens-details'], 'instance_rows': {'prefix_caching_enabled': True}}
    assert 'cache-unaware' in OVERLAYS['prefix_cache']['estimators'] and 'point of this ablation' in OVERLAYS['prefix_cache']['estimators']
    assert 'estimators' not in OVERLAYS['chunk8192'] and 'refitted' in OVERLAYS['chunk8192']['coefficients'] and 'retained' in OVERLAYS['prefix_cache']['coefficients']
    # Distinct identities: nothing collides with canonical, FCFS or each other.
    all_ids = {i for p in (CHUNK8192, PREFIX_CACHE) for i in ids(p)}
    assert len(all_ids) == 24 and all_ids.isdisjoint({c['id'] for c in FCFS_MATRIX['cells']} | {c['id'] for c in CANONICAL_SFS_SCORE['cells']} | {c['id'] for c in read(ROOT/'scripts/cloud/campaign.json')['cells']})
    with pytest.raises(ValueError, match='No reduced-grid overlay'):
        campaign.build(FCFS, bundle)


def test_overlay_accepted_in_every_mode_and_policies_is_the_single_editable_field(bundle):
    m = read(bundle/'bundle.json'); saved = deepcopy(m)
    for profile in (CHUNK8192, PREFIX_CACHE):
        overlay = OVERLAYS[profile.name]
        for mode in ('calibrate', 'qualify', 'run', 'campaign', 'inspect'):
            active = campaign.apply_campaign(profile, m, overlay, mode)
            assert m == saved and active['files'] == saved['files'] and active['families']['ministral'] == saved['families']['ministral']
            assert [c['id'] for c in active['cells']] == ids(profile) and active['blocked_cells'] == [] and active['requests_total'] == 12*16000
            assert all(set(c) == set(campaign.KEYS) and c['requests'] == 16000 and c['variant'] == 'canonical' and c['family'] == 'qwen' for c in active['cells'])
            assert active['kind'] == campaign.KIND and active['configuration_id'] == profile.configuration_id and active['coefficient_policy'] == profile.coefficient_policy
            q = active['families']['qwen']
            assert q['policies'] == ['hard', 'mooncake_prefill', 'lmdeploy_proxy'] and q['qps'] == [6., 7., 8., 8.3] and q['configuration_id'] == profile.configuration_id
            assert all(q['profile'][k] == v for k, v in profile.settings.items()) and q['profile']['rope_scaling'] == qwen.PROFILE['rope_scaling']
            rl = active['remaining_length']
            assert Path(rl['tables']) == TABLES.resolve() and rl['rules'] == {'qwen3-0.6b': {'mode': 'running_all', 'quantile': 0.5, 'conditioning': 'prompt_bin'}}
            assert rl['off_models'] == ['qwen3-32b', 'qwen3-8b'] and active['accepted_prior_source_digests'] == {}
        # Editing only `policies` re-derives the grid (the hash changes, so re-qualification follows).
        edited = deepcopy(overlay); edited['policies'] = ['hard', 'score']
        active = campaign.apply_campaign(profile, m, edited, 'run')
        assert [c['id'] for c in active['cells']] == ids(profile, ['hard', 'score']) and active['families']['qwen']['policies'] == ['hard', 'score'] and active['requests_total'] == 8*16000
        write(tmp_edit := bundle/f'{profile.name}-edited.json', edited); assert digest(tmp_edit) != digest(campaign.OVERLAYS[profile.name])   # the bound overlay hash changes
        # Read-only inspection (control status, collate) accepts both; running without the profile is refused.
        assert [c['id'] for c in apply_any_campaign(m, overlay, inspect=True)['cells']] == ids(profile)
        with pytest.raises(ValueError, match=f'profile {profile.name}'):
            apply_any_campaign(m, overlay)
        # Neither the FCFS validators nor the canonical one accept these overlays, and vice versa.
        for reject in (lambda: apply_fcfs(m, overlay, 'qualify'), lambda: apply_sfs_score_campaign(m, overlay),
                       lambda: campaign.apply_campaign(profile, m, FCFS_MATRIX, 'qualify'), lambda: campaign.apply_campaign(profile, m, FCFS_SFS_SCORE, 'qualify'),
                       lambda: campaign.apply_profile_campaign('fcfs', m, overlay, 'qualify'), lambda: campaign.apply_profile_campaign(profile.name, m, FCFS_SFS_SCORE, 'qualify')):
            with pytest.raises(ValueError):
                reject()
    # An overlay of one configuration never applies under the other profile.
    for name, other in (('chunk8192', PREFIX_CACHE), ('prefix_cache', CHUNK8192)):
        with pytest.raises(ValueError, match='not the'):
            campaign.apply_campaign(other, m, OVERLAYS[name], 'qualify')
        with pytest.raises(ValueError, match='not the'):
            campaign.apply_profile_campaign(other.name, m, OVERLAYS[name], 'qualify')
    # The FCFS overlays still dispatch to the FCFS validator through the profile dispatcher.
    assert len(campaign.apply_profile_campaign('fcfs', m, FCFS_SFS_SCORE, 'run')['cells']) == 8
    assert campaign.apply_configuration_campaign(m, FCFS_MATRIX)['configuration_id'] == fcfs.CONFIG_ID


def _unauthorized(o): o['full_matrix_authorized'] = False
def _no_authorization(o): o['authorization'] = ''
def _unknown_policy(o): o['policies'] = ['hard', 'mystery']
def _duplicate_policy(o): o['policies'] = ['hard', 'hard']
def _no_policies(o): o['policies'] = []
def _policies_string(o): o['policies'] = 'hard'
def _rates(o): o['rates'] = [6., 7., 8., 8.6]
def _budget(o): o['requests_per_cell'] = 8000
def _explicit_cells(o): o['cells'] = campaign.cells(CHUNK8192, o['policies'])
def _kind(o): o['kind'] = 'fcfs_sfs_score'
def _no_kind(o): del o['kind']
def _configuration(o): o['configuration_id'] = 'qwen-fcfs-unchunked-65536'
def _profile(o): o['profile'] = dict(o['profile'], max_num_batched_tokens=4096)
def _profile_key(o): o['profile'] = dict(o['profile'], prefix_caching=not o['profile']['prefix_caching'])
def _coefficient_policy(o): o['coefficient_policy'] = 'canonical' if o['coefficient_policy'] == 'refit' else 'refit'
def _no_rule(o): del o['remaining_length']
def _hash(o): o['remaining_length']['files']['qwen3-0.6b.json'] = '0'*64
def _absolute(o): o['remaining_length']['tables'] = str(TABLES)
def _eight_b(o): o['remaining_length']['rules']['qwen3-8b'] = 'running_all:0.5:prompt_bin'
def _off(o): o['remaining_length']['off_models'] = ['qwen3-8b']
def _evidence(o): o['remaining_length']['evidence'] = {}
def _prior(o): o['accepted_prior_source_digests'] = {'abc': 'reason'}


@pytest.mark.parametrize('name', ['chunk8192', 'prefix_cache'])
@pytest.mark.parametrize('mutate', [_unauthorized, _no_authorization, _unknown_policy, _duplicate_policy, _no_policies, _policies_string, _rates, _budget, _explicit_cells,
                                    _kind, _no_kind, _configuration, _profile, _profile_key, _coefficient_policy, _no_rule, _hash, _absolute, _eight_b, _off, _evidence, _prior])
def test_overlay_rejections(bundle, name, mutate):
    profile = PROFILES[name]; m = read(bundle/'bundle.json'); bad = deepcopy(OVERLAYS[name]); mutate(bad)
    with pytest.raises(ValueError):
        campaign.apply_campaign(profile, m, bad, 'run')
    if mutate in (_unauthorized, _no_authorization):   # only the authorization is mode dependent: qualify still works, run/campaign refuse
        assert len(campaign.apply_campaign(profile, m, bad, 'qualify')['cells']) == 12
        with pytest.raises(ValueError, match='not authorized' if mutate is _unauthorized else 'authorization text'):
            campaign.apply_campaign(profile, m, bad, 'campaign')
    else:
        with pytest.raises(ValueError):
            campaign.apply_campaign(profile, m, bad, 'qualify')


def test_chunk8192_refits_coefficients_and_prefix_cache_refuses_them(bundle, tmp_path):
    truth = dict(zip(qwen.COEFFICIENT_NAMES, qwen.COEFFICIENTS[1]))
    calibration = tmp_path/'calibration'; calibration.mkdir()
    for model in FCFS_MATRIX['models']: trace(calibration/f'calibration_trace_{model}.csv', truth)
    coefficients.fit(calibration, tmp_path/'chunk.json', CHUNK8192)
    payload = read(tmp_path/'chunk.json')
    assert payload['configuration_id'] == 'qwen-chunk8192' and payload['profile'] == CHUNK8192.settings and payload['status'].startswith('FITTED_FOR_CONFIGURATION')
    fitted = coefficients.load(tmp_path/'chunk.json', CHUNK8192)
    assert set(fitted) == set(FCFS_MATRIX['models']) and all(set(f) == set(coefficients.NAMES) and all(v >= 0 for v in f.values()) for f in fitted.values())
    assert all(row['fit_prediction_diagnostics']['r2_all_rows'] > .99 for row in payload['models'].values())
    # The fitted file is bound to its configuration: neither the FCFS profile nor the FCFS file crosses over.
    with pytest.raises(ValueError, match='not fitted'):
        coefficients.load(tmp_path/'chunk.json', FCFS)
    coefficients.fit(calibration, tmp_path/'fcfs.json', FCFS)
    with pytest.raises(ValueError, match='not fitted'):
        coefficients.load(tmp_path/'fcfs.json', CHUNK8192)
    # Fitted coefficients reach the router rows and the servers under chunk8192.
    cfg = instance_config('qwen', None, bundle, PORTS, 'iso', 'chunk8192', fitted)
    assert cfg['coefficient_status'] == 'FITTED_FOR_CONFIGURATION' and cfg['coefficient_policy'] == 'refit'
    for i, row in enumerate(cfg['instances']):
        assert row['ttft_batch_model'] == fitted[row['model_id']] and row['max_num_batched_tokens'] == 8192
        argv = flags(instance_argv('qwen', tmp_path/'model', row, i, tmp_path, bundle/'qwen/length', bundle, 'chunk8192'))
        assert float(argv['--simulation-intercept']) == fitted[row['model_id']]['intercept'] and argv['--max-num-batched-tokens'] == '8192'
    assert instance_config('qwen', None, bundle, PORTS, 'iso', 'chunk8192')['coefficient_status'] == 'CANONICAL_PLACEHOLDER_FOR_TRACE_COLLECTION_ONLY'
    for model in FCFS_MATRIX['models']: trace(calibration/f'batch_stats_{model}.csv', truth, seed=3, start=2e9)
    coefficients.validate(tmp_path/'chunk.json', calibration, tmp_path/'audit.json', CHUNK8192)
    audit = read(tmp_path/'audit.json')
    assert audit['configuration_id'] == 'qwen-chunk8192' and audit['models']['qwen3-8b']['independent']['rows'] == 400 and audit['models']['qwen3-8b']['independent']['r2_all_rows'] > .99
    # The prefix-cache profile retains the canonical coefficients: nothing is fitted, loaded or accepted.
    cfg = instance_config('qwen', None, bundle, PORTS, 'iso', 'prefix_cache')
    assert cfg['coefficient_status'] == 'CANONICAL_RETAINED_BY_CONFIGURATION_POLICY' and cfg['coefficient_policy'] == 'canonical'
    assert [r['ttft_batch_model'] for r in cfg['instances']] == [dict(zip(qwen.COEFFICIENT_NAMES, c)) for c in qwen.COEFFICIENTS]
    for refuse in (lambda: instance_config('qwen', None, bundle, PORTS, 'iso', 'prefix_cache', fitted), lambda: coefficients.load(tmp_path/'chunk.json', PREFIX_CACHE),
                   lambda: coefficients.fit(calibration, tmp_path/'prefix.json', PREFIX_CACHE), lambda: coefficients.validate(tmp_path/'chunk.json', calibration, tmp_path/'x.json', PREFIX_CACHE)):
        with pytest.raises(ValueError, match='retains the canonical'):
            refuse()
    assert not (tmp_path/'prefix.json').exists()


def test_worker_cli_requires_coefficients_for_chunk8192_and_refuses_them_for_prefix_cache(bundle, tmp_path, monkeypatch):
    from scripts.cloud import worker
    captured = {}
    async def execute(options, manifest, definition, model_paths, output): captured.update(manifest=manifest, definition=definition, options=options)
    monkeypatch.setattr(worker, 'execute', execute)
    monkeypatch.setattr(worker, 'read', lambda path: {} if str(path).endswith('models.json') else read(path))
    write(tmp_path/'coefficients.json', {'fitted': True})
    common = ['--bundle', str(bundle), '--models', str(tmp_path/'models.json'), '--state', str(tmp_path/'state'), '--family', 'qwen', '--gpus', '0,1,2,3']
    def run(*argv):
        monkeypatch.setattr(sys, 'argv', ['worker', *argv, *common]); worker.main()
    chunk, prefix = str(campaign.OVERLAYS['chunk8192']), str(campaign.OVERLAYS['prefix_cache'])
    rejected = [('qualify', '--profile', 'chunk8192', '--campaign', chunk),                                            # refit profile without coefficients
                ('run', '--profile', 'chunk8192', '--campaign', chunk, '--qualification', 'q'),
                ('qualify', '--profile', 'prefix_cache', '--campaign', prefix, '--coefficients', str(tmp_path/'coefficients.json')),   # canonical policy refuses a refit
                ('calibrate', '--profile', 'prefix_cache', '--campaign', prefix),
                ('qualify', '--profile', 'chunk8192', '--coefficients', str(tmp_path/'coefficients.json')),           # no overlay
                ('qualify', '--profile', 'prefix_cache'),
                ('qualify', '--profile', 'chunk8192', '--campaign', chunk, '--coefficients', str(tmp_path/'coefficients.json'), '--variant', 'mlp_length'),
                ('qualify', '--profile', 'chunk4096', '--campaign', chunk)]
    for argv in rejected:
        with pytest.raises(SystemExit):
            run('--output', str(tmp_path/'rejected'), *argv)
        assert not (tmp_path/'rejected').exists()
    accepted = [('calibrate', 'chunk8192', chunk, [], 'CALIBRATION_TRACES_COLLECTED'),
                ('qualify', 'chunk8192', chunk, ['--coefficients', str(tmp_path/'coefficients.json')], 'QUALIFIED_AWAITING_REVIEW'),
                ('run', 'chunk8192', chunk, ['--coefficients', str(tmp_path/'coefficients.json'), '--qualification', str(tmp_path/'q')], 'COMPLETE'),
                ('qualify', 'prefix_cache', prefix, [], 'QUALIFIED_AWAITING_REVIEW'),
                ('run', 'prefix_cache', prefix, ['--qualification', str(tmp_path/'q')], 'COMPLETE')]
    for n, (mode, name, overlay, extra, state) in enumerate(accepted):
        run(mode, '--output', str(tmp_path/f'ok{n}'), '--profile', name, '--campaign', overlay, *extra)
        manifest, definition = captured['manifest'], captured['definition']
        assert manifest['kind'] == campaign.KIND and manifest['configuration_id'] == PROFILES[name].configuration_id and [c['id'] for c in manifest['cells']] == ids(PROFILES[name])
        assert definition['policies'] == ['hard', 'mooncake_prefill', 'lmdeploy_proxy'] and definition['profile']['max_num_batched_tokens'] == PROFILES[name].settings['max_num_batched_tokens']
        assert family_remaining_length(manifest, definition)['rules'] == {'qwen3-0.6b': {'mode': 'running_all', 'quantile': 0.5, 'conditioning': 'prompt_bin'}}
        assert read(tmp_path/f'ok{n}'/'status.json')['state'] == state
    # Cross-profile and unauthorized overlays are refused before any output exists.
    for argv, match in ((('qualify', '--profile', 'fcfs', '--campaign', chunk, '--coefficients', str(tmp_path/'coefficients.json')), 'not the FCFS'),
                        (('qualify', '--profile', 'prefix_cache', '--campaign', chunk), 'not the qwen-prefix-cache'),
                        (('qualify', '--campaign', prefix), 'profile prefix_cache')):
        with pytest.raises(ValueError, match=match):
            run('--output', str(tmp_path/'refused'), *argv)
        assert not (tmp_path/'refused').exists()
    unauthorized = deepcopy(OVERLAYS['prefix_cache']); unauthorized['full_matrix_authorized'] = False; write(tmp_path/'unauthorized.json', unauthorized)
    with pytest.raises(ValueError, match='not authorized'):
        run('run', '--output', str(tmp_path/'blocked'), '--profile', 'prefix_cache', '--campaign', str(tmp_path/'unauthorized.json'), '--qualification', 'q')
    assert not (tmp_path/'blocked').exists()
    run('qualify', '--output', str(tmp_path/'unauth-qualify'), '--profile', 'prefix_cache', '--campaign', str(tmp_path/'unauthorized.json'))
    assert read(tmp_path/'unauth-qualify'/'status.json')['state'] == 'QUALIFIED_AWAITING_REVIEW'


def test_provenance_and_release_bind_configuration_and_coefficient_policy(bundle, tmp_path, monkeypatch):
    from scripts.cloud import worker
    monkeypatch.setattr(worker, 'hardware', lambda gpus: {'gpus': gpus})
    write(tmp_path/'chunk.json', {'fitted': True})
    m = read(bundle/'bundle.json')
    cases = {'prefix_cache': SimpleNamespace(family='qwen', variant='canonical', bundle=str(bundle), gpus='0,1,2,3', profile='prefix_cache', coefficients=None,
                                             campaign=str(campaign.OVERLAYS['prefix_cache'])),
             'chunk8192': SimpleNamespace(family='qwen', variant='canonical', bundle=str(bundle), gpus='0,1,2,3', profile='chunk8192', coefficients=str(tmp_path/'chunk.json'),
                                          campaign=str(campaign.OVERLAYS['chunk8192']))}
    for name, options in cases.items():
        profile = PROFILES[name]
        active = campaign.apply_campaign(profile, m, OVERLAYS[name], 'run'); q = active['families']['qwen']
        rule = remaining_length_provenance(q, family_remaining_length(active, q))
        record = provenance(options, active, rule)
        assert record == {'configuration_id': profile.configuration_id, 'coefficient_policy': profile.coefficient_policy,
                          'coefficients_sha256': digest(tmp_path/'chunk.json') if name == 'chunk8192' else None, 'campaign_sha256': digest(options.campaign),
                          'campaign_kind': campaign.KIND, 'remaining_length_rule': RULE, 'remaining_length': remaining_length_record(rule)}
        qdir = tmp_path/name; write(qdir/'evidence.json', {'ok': True})
        report = {'status': 'GPU_MEASURED_REVIEW_REQUIRED', 'family': 'qwen', 'variant': 'canonical', 'source_sha256': 's', 'bundle_sha256': digest(bundle/'bundle.json'),
                  'hardware': {'gpus': ['0', '1', '2', '3']}, 'files': {'evidence.json': digest(qdir/'evidence.json')}, **record}
        write(qdir/'qualification.json', report)
        write(qdir/'release.json', {'status': 'RELEASED', 'qualification_sha256': digest(qdir/'qualification.json'), 'timing_review': 't'*40, 'load_review': 'l'*40})
        worker.validate_release(qdir, options, 's', RULE)
        with pytest.raises(ValueError, match='remaining-length'):
            worker.validate_release(qdir, options, 's', 'current')
        other = 'chunk8192' if name == 'prefix_cache' else 'prefix_cache'
        changes = [{'profile': 'canonical', 'coefficients': None, 'campaign': None}, {'profile': other, 'coefficients': cases[other].coefficients, 'campaign': cases[other].campaign},
                   {'profile': 'fcfs', 'coefficients': str(tmp_path/'chunk.json')}, {'gpus': '4,5,6,7'},
                   {'coefficients': None if name == 'chunk8192' else str(tmp_path/'chunk.json')}]
        for change in changes:
            with pytest.raises(ValueError, match='qualification'):
                worker.validate_release(qdir, SimpleNamespace(**dict(vars(options), **change)), 's', RULE)
        with pytest.raises(ValueError, match='campaign changed'):
            worker.validate_release(qdir, SimpleNamespace(**dict(vars(options), campaign=str(ROOT/'scripts/cloud/fcfs/campaign-20260916.json'))), 's', RULE)


def _observation(prompt, planned, scheduled, generated=0, extra=()):
    requests = {'r': {'num_prompt_tokens': prompt, 'num_computed_tokens': planned, 'num_output_processed_tokens': generated}}
    scheduled_by = {'r': scheduled}
    for k, (p, pl, s) in dict(extra).items():
        requests[k] = {'num_prompt_tokens': p, 'num_computed_tokens': pl, 'num_output_processed_tokens': 0}; scheduled_by[k] = s
    return {'inflight': {'scheduled_tokens_by_request': scheduled_by}, 'requests': requests}


def test_bounded_smoke_evidence_rules(tmp_path):
    chunked = [_observation(20000, 8192, 8192), _observation(20000, 16384, 8192), _observation(20000, 20000, 3616), _observation(500, 500, 500)]
    assert evidence(CHUNK8192, chunked) == 2   # two partial prefill steps of the long prompt
    with pytest.raises(AssertionError):   # a single step above the budget
        evidence(CHUNK8192, [_observation(20000, 9000, 9000)])
    with pytest.raises(AssertionError):   # a batch above the budget
        evidence(CHUNK8192, [_observation(6000, 6000, 6000, extra={'s': (3000, 3000, 3000)})])
    with pytest.raises(ValueError, match='longer than the step budget'):
        evidence(CHUNK8192, [_observation(500, 500, 500)])
    with pytest.raises(ValueError, match='longer than the step budget'):
        evidence(CHUNK8192, [])
    with pytest.raises(ValueError, match='No observed'):
        evidence(FCFS, [])
    assert evidence(FCFS, [_observation(20000, 20000, 20000), _observation(30000, 30000, 30000)]) == 2
    with pytest.raises(AssertionError):
        evidence(FCFS, [_observation(20000, 8192, 8192)])
    with pytest.raises(ValueError, match='No bounded smoke rule'):
        evidence(PREFIX_CACHE, chunked)
    with pytest.raises(ValueError, match='no bounded GPU smoke rule'):
        smoke_run(PREFIX_CACHE, tmp_path, {}, tmp_path/'smoke', ['0', '1'], (2,))
    assert not (tmp_path/'smoke').exists()

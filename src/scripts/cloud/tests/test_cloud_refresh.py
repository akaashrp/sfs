import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.cloud.collate import evaluation_judge, observed_qwen_queries, observed_utilities
from scripts.cloud.model_downloads import download_models


def test_primary_judge_tracks_quality_training_and_allows_explicit_comparison():
    for variant in ('canonical', 'mlp_quality', 'mlp_length'):
        assert evaluation_judge({'family': 'qwen', 'variant': variant}) == 'pro'
    flash = {'family': 'qwen', 'variant': 'flash_quality'}
    assert evaluation_judge(flash) == 'flash'
    assert evaluation_judge(flash, 'pro') == 'pro'
    assert evaluation_judge({'family': 'ministral', 'variant': 'canonical'}, 'flash') == 'pro'


def test_observed_cohort_excludes_imputed_unselected_candidate(tmp_path):
    models = ['qwen3-0.6b', 'qwen3-8b', 'qwen3-32b']
    scores = {judge: {(model, 'alpaca', example): .7 for model in models for example in ('a', 'b')}
              for judge in ('pro', 'flash')}
    path = tmp_path/'qwen/scores_flash/qwen3-32b/alpaca_scored.jsonl'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'bucket': 'alpaca', 'prompt_metadata': {'example_id': 'b'},
                                'quality_imputed': True}) + '\n')
    assert observed_qwen_queries(tmp_path, models, scores) == {('alpaca', 'a')}


def test_ontimeutility_uses_matching_judge_common_cohort_and_full_ttft_denominator():
    mapping = {f'req-{i}': SimpleNamespace(bucket='alpaca', example_id=str(i)) for i in range(3)}
    quality = {judge: {('qwen3-8b', 'alpaca', str(i)): score for i in range(3)}
               for judge, score in [('pro', .4), ('flash', .8)]}
    rows = [{'request_id': f'req-{i}', 'bucket': 'alpaca', 'response_model': 'qwen3-8b',
             'actual_cost': 2., 'system_entry_e2e_ttft_slo_met': i != 1,
             'system_entry_e2e_ttft_ms': 200. if i == 1 else 50., 'ttft_slo_ms': 100.}
            for i in range(3)]
    point = {'config': {'lambda_weight': .01}, 'router': {'runs': [{'per_request': rows,
             'summary': {'system_entry_e2e_ttft_slo_attainment_pct': 200/3}}]}}
    result = observed_utilities(point, mapping, quality, {('alpaca', '0'), ('alpaca', '1')}, 'flash')
    assert result['ontimeutility'] == pytest.approx({'pro': .19, 'flash': .39})
    assert result['primary_ontimeutility'] == pytest.approx(.39)
    assert result['observed_scored_queries'] == 2 and result['excluded_queries'] == 1
    assert result['ttft_denominator'] == 3
    rows[0]['system_entry_e2e_ttft_slo_met'] = False
    with pytest.raises(ValueError, match='TTFT attainment'):
        observed_utilities(point, mapping, quality, {('alpaca', '0')}, 'flash')


def test_pinned_downloads_overlap_and_publish_only_complete_map(tmp_path, monkeypatch):
    import huggingface_hub
    import threading
    barrier = threading.Barrier(2)
    calls = []
    output = tmp_path/'models.json'
    def fetch(**kwargs):
        calls.append(kwargs)
        assert not output.exists()
        barrier.wait(timeout=10)
        directory = tmp_path/kwargs['repo_id']/'snapshots'/kwargs['revision']
        directory.mkdir(parents=True)
        filename = 'consolidated.safetensors' if 'Ministral' in kwargs['repo_id'] else 'model.safetensors'
        (directory/filename).write_bytes(b'fixture weights')
        return str(directory)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', fetch)
    models = {'qwen': {'repo': 'Qwen/test', 'revision': 'a'*40, 'family': 'qwen'},
              'ministral': {'repo': 'Ministral/test', 'revision': 'b'*40, 'family': 'ministral'}}
    download_models(models, tmp_path/'cache', output)
    assert set(json.loads(output.read_text())) == set(models)
    assert {c['revision'] for c in calls} == {'a'*40, 'b'*40}
    ministral = next(c for c in calls if c['repo_id'].startswith('Ministral'))
    assert 'consolidated.safetensors' in ministral['allow_patterns']
    assert '*.safetensors' not in ministral['allow_patterns']
    assert json.loads(Path(str(output)+'.progress.json').read_text())['remaining'] == []


def test_incomplete_shard_set_does_not_publish_model_map(tmp_path, monkeypatch):
    import huggingface_hub
    snapshot = tmp_path/'snapshot'
    snapshot.mkdir()
    (snapshot/'model-1.safetensors').write_bytes(b'partial')
    (snapshot/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {
        'a': 'model-1.safetensors', 'b': 'model-2.safetensors'}}))
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', lambda **kw: str(snapshot))
    with pytest.raises(ValueError, match='Missing shard'):
        download_models({'qwen': {'repo': 'Qwen/test', 'revision': 'a'*40, 'family': 'qwen'}},
                        tmp_path/'cache', tmp_path/'models.json')
    assert not (tmp_path/'models.json').exists()

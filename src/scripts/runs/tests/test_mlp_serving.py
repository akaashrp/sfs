"""Exercise exported MLPs through production predictor and serving interfaces."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from scripts.prep.train_serving_mlp import export_pipeline
from vllm.v1.engine.accuracy_predictor import AccuracyPredictor
from vllm.v1.engine.output_length_predictor import OutputLengthPredictor, AdmissionFeatures

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(scope='module')
def artifacts(tmp_path_factory):
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    dest = tmp_path_factory.mktemp('mlp_artifacts')
    admissions = [AdmissionFeatures(m, f'Explain topic {i} in a short paragraph.', 32+i)
                  for i in range(12) for m in ('qwen3-0.6b', 'qwen3-8b', 'qwen3-32b')]
    result = {}
    for target, name, cls, attr in (
        ('quality', 'accuracy_predictor', AccuracyPredictor, '_feature_builder'),
        ('output_tokens', 'output_length_predictor', OutputLengthPredictor, '_feature_extractor')):
        source = ROOT/'src/assets/predictors'/name
        original = cls(str(source))
        x = np.asarray([getattr(original, attr).build_feature_row(a) for a in admissions], dtype=float)
        y = np.linspace(.1, .9, len(x)) if target == 'quality' else np.log1p(np.linspace(20, 400, len(x)))
        model = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(8,4),
                solver='lbfgs', max_iter=1000, random_state=69, tol=1e-5))
        with threadpool_limits(limits=1):
            model.fit(x, y)
        path = dest/name
        export_pipeline(model, json.loads((source/'metadata.json').read_text()), target, path)
        result[target] = (path, cls(str(path)), model, x)
    return result, admissions


@pytest.mark.parametrize('target', ['quality', 'output_tokens'])
def test_batch_single_shared_context_and_reload_parity(artifacts, target):
    data, admissions = artifacts
    path, loaded, pipeline, x = data[target]
    expected = pipeline.predict(x)
    expected = (np.clip(expected,0,1) if target=='quality' else
                np.clip(np.expm1(np.clip(expected,0,np.log1p(8192))),1,8192))
    actual = loaded.predict_batch(admissions)
    if target=='output_tokens': actual=[p.mean_tokens for p in actual]
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-8)
    extractor = loaded._feature_builder._prompt_extractor if target=='quality' else loaded._feature_extractor
    context = extractor.build_prompt_context(prompt_text=admissions[0].prompt_text,
                                              prompt_token_count=admissions[0].prompt_token_count)
    shared = loaded.predict_batch(admissions[:3], prompt_context=context)
    if target=='output_tokens': shared=[p.mean_tokens for p in shared]
    np.testing.assert_allclose(shared,actual[:3],rtol=1e-10,atol=1e-8)
    single=loaded.predict(admissions[0])
    assert (single if target=='quality' else single.mean_tokens)==pytest.approx(actual[0])
    assert loaded.predict_batch([])==[]


def test_router_and_server_consume_same_real_length_artifact(artifacts):
    from sfs_core.routing.wait_time_scheduler import WaitTimeScheduler
    from vllm.platforms.cpu import CpuPlatform
    data, admissions=artifacts
    scheduler=WaitTimeScheduler.__new__(WaitTimeScheduler)
    scheduler._output_length_predictor=data['output_tokens'][1]
    values=scheduler._predict_output_lengths_for_admissions(instance_ids=['instance'],
        admissions=admissions[:1],completion_cap=8192,prompt_context=None)
    with patch('vllm.platforms._current_platform',CpuPlatform()):
        from vllm.v1.engine.async_llm import AsyncLLM
        server=AsyncLLM.__new__(AsyncLLM)
        server.output_length_predictor=OutputLengthPredictor(str(data['output_tokens'][0]))
        server.vllm_config=SimpleNamespace(scheduler_config=SimpleNamespace(
            enable_snapshot_shm_publishing=True,enable_wait_time_simulation=False))
        server.model_config=SimpleNamespace(model=admissions[0].model_id)
        request=SimpleNamespace(prompt_token_ids=list(range(admissions[0].prompt_token_count)),
                                predicted_output_tokens_mean=None)
        server._attach_output_length_prediction(request,admissions[0].prompt_text)
    assert request.predicted_output_tokens_mean==pytest.approx(values['instance'])
    # This field is the input copied into resident scheduler snapshots.
    assert 1 <= request.predicted_output_tokens_mean <= 8192
    scheduler._accuracy_predictor=data['quality'][1]
    quality=scheduler._predict_accuracy_for_admissions(instance_ids=['instance'],
                            admissions=admissions[:1],prompt_context=None)
    assert quality['instance']==pytest.approx(data['quality'][1].predict(admissions[0]))


def test_corrupt_or_wrong_target_artifacts_fail_closed(artifacts,tmp_path):
    import shutil
    data,_=artifacts
    source=data['quality'][0]
    dest=tmp_path/'corrupt'
    shutil.copytree(source,dest)
    p=dest/'mlp_weights.npz'
    p.write_bytes(p.read_bytes()+b'corrupt')
    with pytest.raises(ValueError,match='checksum'):
        AccuracyPredictor(str(dest))
    meta=json.loads((source/'metadata.json').read_text())
    meta['mlp']['target']='output_tokens'
    (dest/'metadata.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='target'):
        AccuracyPredictor(str(dest))


def test_length_cap_invalid_features_and_original_lightgbm(artifacts):
    data,admissions=artifacts
    predictor=data['output_tokens'][1]
    invalid=AdmissionFeatures('qwen3-0.6b','text',0)
    assert predictor.predict_batch([invalid])==[None]
    with patch.object(predictor,'_mean_model',Mock(predict=Mock(return_value=[1e6]))):
        assert predictor.predict(admissions[0]).mean_tokens==pytest.approx(8192)
    with pytest.raises(ValueError,match='input features'):
        data['quality'][1]._booster.predict(np.full_like(data['quality'][3],np.nan))
    old=AccuracyPredictor(str(ROOT/'src/assets/predictors/accuracy_predictor'))
    x=np.vstack([old._feature_builder.build_feature_row(a) for a in admissions[:3]])
    np.testing.assert_array_equal(old.predict_batch(admissions[:3]),
                                  [old._inverse_link(float(v)) for v in old._booster.predict(x)])

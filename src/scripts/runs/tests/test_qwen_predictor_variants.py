from pathlib import Path
from unittest.mock import patch
import json
import pytest
from scripts.runs import qwen_predictor_variants as v
from scripts.runs import qwen_baselines as q


@pytest.mark.parametrize('arm',v.ARMS)
def test_one_predictor_change_and_server_router_agreement(tmp_path,arm):
    paths=v.variant_paths(tmp_path,arm,tmp_path/'flash')
    canonical={'experiment_argv':q.base_argv(tmp_path,tmp_path/'cache')}
    args=v.experiment_argv(canonical,paths);before=canonical['experiment_argv']
    changed=[before[i-1]for i,(a,b)in enumerate(zip(before,args))if a!=b]
    assert changed==['--output-length-model-path'if arm=='mlp_length'else'--accuracy-model-path']
    for index,row in enumerate(q.pool_config(canonical)['instances']):
        old=q.server_argv(tmp_path,tmp_path/'model',row,index,tmp_path/'out')
        new=v.server_argv(tmp_path,tmp_path/'model',row,index,tmp_path/'out',paths)
        assert new[new.index('--output-length-model-path')+1]==paths['router_output_length_model_path']
        for i,(a,b)in enumerate(zip(old,new)):
            if a!=b:assert old[i-1]=='--output-length-model-path'and arm=='mlp_length'
        assert new[new.index('--tensor-parallel-size')+1]==str((1,1,2)[index])


def test_mismatched_server_artifact_rejected(tmp_path):
    paths=v.variant_paths(tmp_path,'mlp_length');paths['server_output_length_model_path']='wrong'
    with pytest.raises(ValueError,match='differ'):v.server_argv(tmp_path,tmp_path,{},0,tmp_path,paths)


@pytest.mark.parametrize('args',[[],['--x','a','--x','b']])
def test_ambiguous_option_rejected(args):
    with pytest.raises(ValueError):v.replace_option(args,'--x','c')


def test_artifact_drift_rejected(tmp_path):
    p=tmp_path/'weights';p.write_text('before');files={str(p):v.sha256(p)};v.verify_files(files)
    p.write_text('after')
    with pytest.raises(ValueError,match='changed'):v.verify_files(files)


def test_heavy_sweep_rejects_missing_smoke_before_output_or_gpu(tmp_path):
    output=tmp_path/'output'
    with patch.object(v,'validate',return_value={'arm':'mlp_quality'}),patch.object(v,'pool')as pool:
        with pytest.raises(FileNotFoundError):v.run(tmp_path/'manifest','sweep',output,tmp_path/'missing')
        pool.assert_not_called();assert not output.exists()


def test_smoke_artifact_identity_cannot_be_reused_for_other_variant(tmp_path):
    manifest=tmp_path/'manifest';manifest.write_text('{}');result=tmp_path/'smoke_hard.json';result.write_text('{}')
    audit={'status':'PASS_GPU_VARIANT_SMOKE','manifest_sha256':v.sha256(manifest),'result_sha256':v.sha256(result),'paths':{'x':'a'},'arm':'mlp_quality'}
    (tmp_path/'smoke_audit.json').write_text(json.dumps(audit))
    with pytest.raises(ValueError,match='stale'):v.validate_smoke({'paths':{'x':'b'},'arm':'mlp_length'},manifest,tmp_path)


def test_prediction_gate_matches_canonical_links_and_float_endpoint():
    import math
    from types import SimpleNamespace
    ls=[SimpleNamespace(mean_tokens=math.expm1(math.log1p(8192)))]*576
    v.validate_predictions([1.001]*576,ls,quality_backend='lightgbm',length_backend='numpy_mlp')
    with pytest.raises(ValueError,match='quality clipping'):v.validate_predictions([1.001]*576,ls,quality_backend='numpy_mlp',length_backend='numpy_mlp')
    for bad in [float('nan'),float('inf'),0.,8192.001]:
        with pytest.raises(ValueError):v.validate_predictions([.5]*576,[SimpleNamespace(mean_tokens=bad)]*576,quality_backend='numpy_mlp',length_backend='numpy_mlp')

import copy
import pytest
from scripts.prep.flash_calibration import fresh,validate_labels


def records():
    return [dict(bucket='alpaca',example_id='a',prompt_metadata={'example_id':'a'},model_label=m,prompt='q',response={'output_text':'answer','completion_tokens':2},quality=.5,judge_alias_to_model={'A':'qwen3-0.6b','B':'qwen3-8b','C':'qwen3-32b'})for m in ['qwen3-0.6b','qwen3-8b','qwen3-32b']]


def test_fresh_removes_labels_without_mutating_source():
    r=records()[0];out=fresh(r)
    assert 'quality'not in out and 'judge_alias_to_model'not in out
    assert out['response']==r['response']and r['quality']==.5


def test_grouped_canary_and_pair_identity():
    a=records();b=copy.deepcopy(a);b[0]['quality']=.9
    assert validate_labels(a,b,require_grouped=True)['candidates']==3
    b[0]['response']['output_text']='changed'
    with pytest.raises(ValueError,match='changed canonical'):validate_labels(a,b)


@pytest.mark.parametrize('change',['duplicate','missing','nan','imputed','alias'])
def test_bad_canary_rejected(change):
    a=records();b=copy.deepcopy(a)
    if change=='duplicate':b.append(b[0])
    elif change=='missing':b.pop()
    elif change=='nan':b[0]['quality']=float('nan')
    elif change=='imputed':b[0]['quality_imputed']=True
    else:b[0]['judge_alias_to_model']={}
    with pytest.raises(ValueError):validate_labels(a,b,require_grouped=True)


def test_full_label_audit_keeps_fallback_accounting():
    a=records();b=copy.deepcopy(a);b[0]['quality_imputed']=True;b[1].pop('judge_alias_to_model')
    result=validate_labels(a,b)
    assert result['imputed']==1 and result['individual_without_group_alias']==1

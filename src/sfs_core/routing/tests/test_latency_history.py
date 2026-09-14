import json, random, subprocess, os
from pathlib import Path
import pytest
from sfs_core.routing.latency_history import LatencyHistory

ROOT=Path(__file__).resolve().parents[4]

def test_differential_upstream(tmp_path):
    import importlib.util
    spec=importlib.util.spec_from_file_location('oracle_build',ROOT/'src/scripts/runs/tests/latency_oracle/build.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    go=os.environ.get('SFS_TEST_GO')
    if not go: pytest.fail('Set SFS_TEST_GO to a local Go compiler for mandatory upstream parity')
    executable=module.build(ROOT,tmp_path/'oracle',Path(go))
    rng=random.Random(69);models=['b','a','c'];history=LatencyHistory(models);events=[];expected=[]
    def select():
        events.append({'Kind':'select','Models':models,'TTFT':95,'TPOT':90});expected.append(history.select())
    select()
    for i in range(3300):
        model=models[i%3];metric='ttft' if i%7 else 'tpot';value=rng.choice([.2,1.,10.,rng.random()*10])
        events.append({'Kind':'update','Model':model,'Metric':metric,'Value':value})
        history.update(model,metric,value,request_id=str(i),generation=history.generation)
        if i<20 or i%37==0: select()
    # Force eviction of both metric windows plus ties/reset behavior.
    for i in range(1005):
        for m in models:
            for metric in ('ttft','tpot'):
                value=1. if i%2 else 2.
                events.append({'Kind':'update','Model':m,'Metric':metric,'Value':value})
                history.update(m,metric,value,request_id=f'evict:{m}:{i}',generation=history.generation)
    select();events.append({'Kind':'reset'});history.reset();select()
    for model,metric,value in [('a','ttft',1e300),('a','tpot',1e-300),('b','ttft',1e-300),('b','tpot',1e300)]:
        events.append({'Kind':'update','Model':model,'Metric':metric,'Value':value})
        history.update(model,metric,value,request_id=f'extreme:{model}:{metric}',generation=history.generation)
    select()
    actual=json.loads(subprocess.check_output([str(executable)],input=json.dumps(events),text=True))
    assert len(actual)==len(expected)
    for row,ref in zip(actual,expected):
        assert row['SelectedModel']==ref['selected_model']
        assert row['AllScores']==pytest.approx(ref['scores'],rel=1e-12,abs=1e-12)


def test_guards_reset_fallback_and_warmth():
    h=LatencyHistory(['a','b'])
    assert h.select()['selected_model']=='a'
    for v in [float('nan'),float('inf'),-1,0,True]:
        assert not h.update('a','ttft',v,request_id='x',generation=0)
    assert h.update('a','ttft',2.,request_id='x',generation=0)
    assert not h.update('a','ttft',3.,request_id='x',generation=0)
    assert h.metric('a','ttft')==2.
    h.reset();assert not h.update('a','tpot',1.,request_id='late',generation=0)
    with pytest.raises(ValueError,match='three'):h.require_warm()
    for i in range(3):
        for m in h.models:
            for metric in h.percentiles:h.update(m,metric,1.,request_id=f'{m}:{i}',generation=1)
    h.require_warm();assert h.select()['selected_model']=='a'
    assert LatencyHistory(['a'],ttft_percentile=0,tpot_percentile=0).select()['fallback']=='missing_config'
    with pytest.raises(ValueError):LatencyHistory(['a'],ttft_percentile=101)

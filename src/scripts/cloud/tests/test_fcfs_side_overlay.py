"""CPU checks for the user-authorized FCFS side overlay (hard_prefill_tps) and model-level hardware matching."""
from copy import deepcopy
import sys
from types import SimpleNamespace
import pytest

from scripts.cloud.common import ROOT, read, write, digest
from scripts.cloud.fcfs import config as fcfs
from scripts.cloud.fcfs.campaign import build, build_allowance, apply_campaign, allowed_policies
from scripts.runs import qwen_baselines as qwen

MAIN=read(ROOT/'scripts/cloud/fcfs/campaign-20260916.json')
SIDE=read(ROOT/'scripts/cloud/fcfs/campaign-ptps-bridges-20260917.json')
POLICY='hard_prefill_tps'


@pytest.fixture
def bundle(tmp_path):
    write(tmp_path/'bundle.json',{'schema_version':1,'models':deepcopy(MAIN['models']),'files':{},'cells':[{'id':f'c{i}'} for i in range(68)],
        'families':{'qwen':{'profile':deepcopy(qwen.PROFILE),'models':list(MAIN['models']),'qps':[7.,8.,8.6,8.75],'policies':['mooncake_prefill'],'requests':16000},
                    'ministral':{'profile':{'max_num_batched_tokens':32768},'qps':[6.0125],'policies':['hard'],'requests':8000}}})
    return tmp_path


def test_side_overlay_is_exactly_the_authorized_four_cells(bundle):
    m=read(bundle/'bundle.json');saved=deepcopy(m)
    assert build_allowance(bundle,[POLICY],SIDE['policy_allowance']['authorization'],SIDE['gpu_allocation'])==SIDE
    assert SIDE['configuration_id']==fcfs.CONFIG_ID and SIDE['profile']==fcfs.SETTINGS and SIDE['full_matrix_authorized'] is True
    assert allowed_policies(SIDE)==(POLICY,) and allowed_policies(MAIN)==tuple(fcfs.METHODS)
    assert [c['id'] for c in SIDE['cells']]==[f'{fcfs.CONFIG_ID}-{POLICY}-{q:g}' for q in (6,7,8,8.3)]
    assert SIDE['requests_total']==64000 and all(c['requests']==16000 and c['variant']=='canonical' for c in SIDE['cells'])
    assert {c['id'] for c in SIDE['cells']}.isdisjoint({c['id'] for c in MAIN['cells']})
    assert POLICY not in fcfs.METHODS and POLICY not in {c['policy'] for c in MAIN['cells']}
    for mode in ('qualify','run'):
        applied=apply_campaign(m,SIDE,mode)
        assert m==saved and applied['families']['qwen']['policies']==[POLICY] and applied['families']['qwen']['qps']==[6.,7.,8.,8.3]
        assert [c['id'] for c in applied['cells']]==[c['id'] for c in SIDE['cells']] and applied['blocked_cells']==[] and applied['requests_total']==64000
        assert all(applied['families']['qwen']['profile'][k]==v for k,v in fcfs.SETTINGS.items())
    # The main overlay keeps its nine-policy grid: the allowance never leaks into it.
    assert {**build(bundle),'status':MAIN['status'],'full_matrix_authorized':True,'authorization':MAIN['authorization']}==MAIN
    assert apply_campaign(m,MAIN,'qualify')['families']['qwen']['policies']==list(fcfs.METHODS)
    leaked=deepcopy(MAIN);leaked['full_matrix_authorized']=True;leaked['cells'].append(dict(SIDE['cells'][0]))
    with pytest.raises(ValueError,match='nine-policy'):apply_campaign(m,leaked,'run')


def test_side_overlay_rejections(bundle):
    m=read(bundle/'bundle.json')
    def broken(change):
        bad=deepcopy(SIDE);change(bad)
        with pytest.raises(ValueError):apply_campaign(m,bad,'run')
    broken(lambda o:o.update(full_matrix_authorized=False))
    broken(lambda o:o['policy_allowance'].update(policies=['hard']))              # inside the nine: belongs to the main overlay
    broken(lambda o:o['policy_allowance'].update(policies=[POLICY,POLICY]))
    broken(lambda o:o['policy_allowance'].update(policies=[]))
    broken(lambda o:o['policy_allowance'].update(authorization='short'))
    broken(lambda o:o['policy_allowance'].pop('authorization'))
    broken(lambda o:o['policy_allowance'].update(extra=True))
    broken(lambda o:o.update(policy_allowance=[POLICY]))
    broken(lambda o:o['cells'].pop())                                           # missing a rate
    broken(lambda o:o['cells'].append(dict(o['cells'][0],id=f'{fcfs.CONFIG_ID}-{POLICY}-9',qps=9.)))
    broken(lambda o:o['cells'][0].update(policy='hard',id=f'{fcfs.CONFIG_ID}-hard-6'))
    broken(lambda o:o['cells'][0].update(id=f'qwen-{POLICY}-6'))
    broken(lambda o:o['cells'][0].update(requests=8000))
    broken(lambda o:o['cells'][0].update(variant='mlp_length'))
    broken(lambda o:o.update(profile=dict(fcfs.SETTINGS,chunked_prefill=True)))
    with pytest.raises(ValueError,match='outside the nine'):build_allowance(bundle,['score'],SIDE['policy_allowance']['authorization'])
    with pytest.raises(ValueError,match='authorization'):build_allowance(bundle,[POLICY],'')


def test_model_level_hardware_match_is_explicit_and_recorded(bundle,tmp_path,monkeypatch):
    from scripts.cloud import worker
    fingerprint=lambda host,name='NVIDIA H100 80GB HBM3',memory='81559',driver='550.90.07',uuids=('GPU-a','GPU-b','GPU-c','GPU-d'):{
        'host':host,'boot_id':host+'-boot','topology':host+'-topo','gpus':[[str(i),u,name,memory,driver] for i,u in enumerate(uuids)]}
    qualified,other=fingerprint('vast-a'),fingerprint('bridges-b',uuids=('GPU-e','GPU-f','GPU-g','GPU-h'))
    assert worker.hardware_class(qualified)==worker.hardware_class(other)
    for changed in (fingerprint('bridges-b',name='NVIDIA A100-SXM4-80GB'),fingerprint('bridges-b',memory='40960'),fingerprint('bridges-b',driver='535.1'),
                    {**other,'gpus':other['gpus'][:3]}):
        assert worker.hardware_class(qualified)!=worker.hardware_class(changed)
    q=tmp_path/'q';write(q/'evidence.json',{'ok':True});write(tmp_path/'coefficients.json',{'fitted':True})
    write(q/'qualification.json',{'status':'GPU_MEASURED_REVIEW_REQUIRED','family':'qwen','variant':'canonical','source_sha256':'s','bundle_sha256':digest(bundle/'bundle.json'),
        'hardware':qualified,'files':{'evidence.json':digest(q/'evidence.json')},'configuration_id':fcfs.CONFIG_ID,'coefficients_sha256':digest(tmp_path/'coefficients.json')})
    write(q/'release.json',{'status':'RELEASED','qualification_sha256':digest(q/'qualification.json'),'timing_review':'t'*40,'load_review':'l'*40})
    base=dict(family='qwen',variant='canonical',bundle=str(bundle),gpus='0,1,2,3',profile='fcfs',coefficients=str(tmp_path/'coefficients.json'),campaign=None)
    monkeypatch.setattr(worker,'hardware',lambda gpus:other)
    with pytest.raises(ValueError,match='qualification'):worker.validate_release(q,SimpleNamespace(**base),'s')          # exact is the default
    with pytest.raises(ValueError,match='qualification'):worker.validate_release(q,SimpleNamespace(**base,hardware_match='exact'),'s')
    worker.validate_release(q,SimpleNamespace(**base,hardware_match='model'),'s')
    monkeypatch.setattr(worker,'hardware',lambda gpus:fingerprint('bridges-b',name='NVIDIA A100-SXM4-80GB'))
    with pytest.raises(ValueError,match='qualification'):worker.validate_release(q,SimpleNamespace(**base,hardware_match='model'),'s')
    monkeypatch.setattr(worker,'hardware',lambda gpus:qualified)
    worker.validate_release(q,SimpleNamespace(**base),'s')
    # The CLI only accepts relaxed matching for run mode, and never silently.
    overlay=ROOT/'scripts/cloud/fcfs/campaign-ptps-bridges-20260917.json'
    common=['--bundle',str(bundle),'--models','m','--state',str(tmp_path/'state'),'--output',str(tmp_path/'out'),'--gpus','0,1,2,3',
            '--profile','fcfs','--family','qwen','--campaign',str(overlay),'--coefficients',str(tmp_path/'coefficients.json')]
    for argv in (['qualify','--hardware-match','model'],['calibrate','--hardware-match','model'],['run','--qualification',str(q),'--hardware-match','uuid']):
        monkeypatch.setattr(sys,'argv',['worker',*argv,*common])
        with pytest.raises(SystemExit):worker.main()
    assert not (tmp_path/'out').exists()

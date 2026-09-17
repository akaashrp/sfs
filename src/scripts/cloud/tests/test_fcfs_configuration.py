"""CPU checks for the separately qualified Qwen FCFS unchunked serving configuration."""
from copy import deepcopy
import sys
from types import SimpleNamespace
import numpy as np
import pytest

from scripts.cloud.common import ROOT, read, write, digest
from scripts.cloud.baseline_campaign import apply_campaign as apply_baseline
from scripts.cloud.fcfs import config as fcfs, coefficients
from scripts.cloud.fcfs.campaign import build, apply_campaign
from scripts.cloud.pool import instance_argv, instance_config
from scripts.runs import qwen_baselines as qwen

OVERLAY=read(ROOT/'scripts/cloud/fcfs/campaign-20260916.json')
BASELINE=read(ROOT/'scripts/cloud/baseline-campaign-20260916.json')
CHANGED={'--no-enable-chunked-prefill':True,'--scheduling-policy':'fcfs','--max-num-batched-tokens':'65536','--max-model-len':'65536',
         '--max-num-seqs':'512','--long-prefill-token-threshold':'0','--gpu-memory-utilization':'0.9'}


@pytest.fixture
def bundle(tmp_path):
    write(tmp_path/'bundle.json',{'schema_version':1,'models':deepcopy(OVERLAY['models']),'files':{},'cells':[{'id':f'c{i}'} for i in range(68)],
        'families':{'qwen':{'profile':deepcopy(qwen.PROFILE),'models':list(OVERLAY['models']),'qps':[7.,8.,8.6,8.75],'policies':['mooncake_prefill'],'requests':16000},
                    'ministral':{'profile':{'max_num_batched_tokens':32768},'qps':[6.0125],'policies':['hard'],'requests':8000}}})
    write(tmp_path/'qwen/bridges_metrics.json',{})
    return tmp_path


def flags(argv):
    out={}
    for i,token in enumerate(argv):
        if token.startswith('--'):out[token]=argv[i+1] if i+1<len(argv) and not argv[i+1].startswith('--') else True
    return out


def trace(path, c, rows=400, noise=2e-5, seed=1, start=1e9):
    rng=np.random.default_rng(seed)
    with path.open('w') as f:
        f.write('ts,engine,prefill,prefill_sq_sum,decode,decode_sq_sum,total,sched,exec,interval,num_seqs,sum_tokens,sum_sq_tokens,avg_tokens,max_tokens,prefill_x_processed_ctx_sum\n')
        for i in range(rows):
            p=int(rng.choice([0,0,64,512,2048,8192,32768]));d=int(rng.integers(0,128)) if rng.random()<.9 else 0
            d=d or (0 if p else 1);n=d+(p>0);s=p+d*int(rng.integers(200,4000));ssq=int(s*s/n)
            y=c['intercept']+c['prefill_coeff']*p+c['prefill_sq_coeff']*p*p+c['decode_coeff']*d+c['sum_coeff']*s+c['sum_sq_coeff']*ssq+rng.normal(0,noise)
            f.write(f'{start+i},0,{p},{p*p},{d},{d},{p+d},0.0004,{max(y,1e-4):.6f},0,{n},{s},{ssq},{s/n:.3f},{s},0\n')


def test_overlay_is_separate_and_not_launchable(bundle):
    m=read(bundle/'bundle.json');saved=deepcopy(m)
    # The committed overlay carries the recorded 16 September user authorization on top of the generated preparation overlay.
    assert len(OVERLAY['cells'])==36 and OVERLAY['requests_total']==576000 and OVERLAY['full_matrix_authorized'] is True
    assert OVERLAY['configuration_id']==fcfs.CONFIG_ID and OVERLAY['profile']==fcfs.SETTINGS and OVERLAY['status'].startswith('FULL_MATRIX_AUTHORIZED')
    assert len(OVERLAY['authorization'])>30 and 'authorized' in OVERLAY['authorization']
    ids={c['id'] for c in OVERLAY['cells']}
    assert all(i.startswith(fcfs.CONFIG_ID+'-') for i in ids)
    assert ids.isdisjoint({c['id'] for c in BASELINE['cells']+BASELINE['fallback_cells']+read(ROOT/'scripts/cloud/campaign.json')['cells']})
    generated=build(bundle)  # The committed overlay is exactly what the code generates plus the authorization flip.
    assert generated['full_matrix_authorized'] is False and generated['status'].startswith('PREPARATION_ONLY')
    assert {**generated,'status':OVERLAY['status'],'full_matrix_authorized':True,'authorization':OVERLAY['authorization']}==OVERLAY
    with pytest.raises(ValueError):apply_baseline(m,OVERLAY)
    with pytest.raises(ValueError):apply_campaign(m,BASELINE,'qualify')
    unauthorized=deepcopy(OVERLAY);unauthorized['full_matrix_authorized']=False
    for mode in ('run','campaign'):
        with pytest.raises(ValueError,match='not authorized'):apply_campaign(m,unauthorized,mode)
    applied=apply_campaign(m,OVERLAY,'qualify')
    assert m==saved and applied['files']==saved['files'] and applied['families']['ministral']==saved['families']['ministral']
    q=applied['families']['qwen']
    assert q['qps']==[6.,7.,8.,8.3] and q['policies']==list(fcfs.METHODS) and q['configuration_id']==fcfs.CONFIG_ID
    assert all(q['profile'][k]==v for k,v in fcfs.SETTINGS.items()) and q['profile']['rope_scaling']==qwen.PROFILE['rope_scaling']
    assert len(applied['cells'])==28 and len(applied['blocked_cells'])==8 and applied['requests_total']==28*16000
    assert not any(c['policy'] in fcfs.BLOCKED for c in applied['cells']) and set(applied['cells'][0])==set(BASELINE['cells'][0])
    authorized=deepcopy(OVERLAY)
    assert len(apply_campaign(m,authorized,'run')['cells'])==28
    for change in (lambda o:o['cells'][0].update(qps=8.6),lambda o:o['cells'][0].update(id='qwen-hard-6'),
                   lambda o:o['cells'][0].update(requests=8000),lambda o:o.update(profile=dict(fcfs.SETTINGS,chunked_prefill=True))):
        bad=deepcopy(authorized);change(bad)
        with pytest.raises(ValueError):apply_campaign(m,bad,'run')


def test_fcfs_server_argv_changes_only_scheduler_settings(bundle,tmp_path):
    ports=(9100,9101,9102)
    cfg=instance_config('qwen',None,bundle,ports,'iso','fcfs')
    reference=instance_config('qwen',{'profile':qwen.PROFILE},bundle,ports,'iso')
    assert cfg['configuration_id']==fcfs.CONFIG_ID and cfg['coefficient_status'].startswith('CANONICAL_PLACEHOLDER')
    assert all(cfg['serving_profile'][k]==v for k,v in fcfs.SETTINGS.items()) and cfg['instance_costs']==reference['instance_costs']
    for i,(row,base) in enumerate(zip(cfg['instances'],reference['instances'])):
        assert row['chunked_prefill_enabled'] is False and row['long_prefill_token_threshold']==0 and row['scheduling_policy']=='fcfs'
        assert row['max_num_batched_tokens']==row['max_model_len']==65536 and row['max_num_seqs']==512
        assert row['ttft_batch_model']==base['ttft_batch_model'] and row['address']==base['address'] and row['snapshot_shm_name']==base['snapshot_shm_name']
        argv=instance_argv('qwen',tmp_path/'model',row,i,tmp_path,bundle/'qwen/length',bundle,'fcfs')
        canonical=instance_argv('qwen',tmp_path/'model',base,i,tmp_path,bundle/'qwen/length',bundle)
        assert argv[0]==canonical[0]==sys.executable and '--enable-chunked-prefill' not in argv
        assert len([t for t in argv if t.startswith('--')])==len(flags(argv))  # every option exactly once
        expected={k:v for k,v in flags(canonical).items() if k!='--enable-chunked-prefill'};expected.update(CHANGED)
        assert flags(argv)==expected
        assert flags(argv)['--tensor-parallel-size']==str((1,1,2)[i]) and flags(argv)['--no-enable-prefix-caching'] is True


def test_fitted_coefficients_reach_router_and_servers(bundle,tmp_path):
    truth=dict(zip(qwen.COEFFICIENT_NAMES,qwen.COEFFICIENTS[1]))
    calibration=tmp_path/'calibration';calibration.mkdir()
    for model in OVERLAY['models']:trace(calibration/f'calibration_trace_{model}.csv',truth)
    coefficients.fit(calibration,tmp_path/'coefficients.json')
    fitted=coefficients.load(tmp_path/'coefficients.json');payload=read(tmp_path/'coefficients.json')
    assert set(fitted)==set(OVERLAY['models']) and payload['configuration_id']==fcfs.CONFIG_ID and payload['profile']==fcfs.SETTINGS
    for model,row in payload['models'].items():
        assert set(fitted[model])==set(coefficients.NAMES) and all(v>=0 for v in fitted[model].values())
        assert row['fit_prediction_diagnostics']['r2_all_rows']>.99 and row['trace_sha256']==digest(calibration/f'calibration_trace_{model}.csv')
    cfg=instance_config('qwen',None,bundle,(9100,9101,9102),'iso','fcfs',fitted)
    assert cfg['coefficient_status']=='FITTED_FOR_CONFIGURATION'
    for i,row in enumerate(cfg['instances']):
        assert row['ttft_batch_model']==fitted[row['model_id']]
        argv=flags(instance_argv('qwen',tmp_path/'model',row,i,tmp_path,bundle/'qwen/length',bundle,'fcfs'))
        assert argv['--simulation-intercept']==str(fitted[row['model_id']]['intercept']) and argv['--no-enable-chunked-prefill'] is True
    # The independent audit only scores batches recorded after calibration ended.
    for model in OVERLAY['models']:trace(calibration/f'batch_stats_{model}.csv',truth,seed=3,start=2e9)
    coefficients.validate(tmp_path/'coefficients.json',calibration,tmp_path/'audit.json')
    audit=read(tmp_path/'audit.json')['models']['qwen3-8b']
    assert audit['independent']['rows']==400 and audit['independent']['r2_all_rows']>.99 and audit['calibration']['rows']==400
    wrong=deepcopy(payload);wrong['configuration_id']='canonical';write(tmp_path/'wrong.json',wrong)
    with pytest.raises(ValueError,match='not fitted'):coefficients.load(tmp_path/'wrong.json')
    noisy=tmp_path/'noisy';noisy.mkdir();trace(noisy/'calibration_trace_qwen3-8b.csv',truth,noise=1.,seed=4)
    with pytest.raises(ValueError,match='below'):coefficients.fit(noisy,tmp_path/'noisy.json')


def test_release_and_cli_bind_configuration_and_coefficients(bundle,tmp_path,monkeypatch):
    from scripts.cloud import worker
    monkeypatch.setattr(worker,'hardware',lambda gpus:{'gpus':gpus})
    q=tmp_path/'q';write(q/'evidence.json',{'ok':True});write(tmp_path/'coefficients.json',{'fitted':True});write(tmp_path/'other.json',{'fitted':False})
    report={'status':'GPU_MEASURED_REVIEW_REQUIRED','family':'qwen','variant':'canonical','source_sha256':'s','bundle_sha256':digest(bundle/'bundle.json'),
        'hardware':{'gpus':['0','1','2','3']},'files':{'evidence.json':digest(q/'evidence.json')}}
    def publish(extra):
        write(q/'qualification.json',{**report,**extra})
        write(q/'release.json',{'status':'RELEASED','qualification_sha256':digest(q/'qualification.json'),'timing_review':'t'*40,'load_review':'l'*40})
    opts=SimpleNamespace(family='qwen',variant='canonical',bundle=str(bundle),gpus='0,1,2,3',profile='fcfs',coefficients=str(tmp_path/'coefficients.json'),campaign=None)
    publish({'configuration_id':fcfs.CONFIG_ID,'coefficients_sha256':digest(tmp_path/'coefficients.json')})
    worker.validate_release(q,opts,'s')
    for change in ({'profile':'canonical','coefficients':None},{'coefficients':str(tmp_path/'other.json')}):
        with pytest.raises(ValueError,match='qualification'):worker.validate_release(q,SimpleNamespace(**dict(vars(opts),**change)),'s')
    publish({})  # Qualifications recorded before this field remain canonical.
    worker.validate_release(q,SimpleNamespace(family='qwen',variant='canonical',bundle=str(bundle),gpus='0,1,2,3'),'s')
    with pytest.raises(ValueError,match='qualification'):worker.validate_release(q,opts,'s')
    assert worker.configuration(SimpleNamespace())=='canonical' and worker.configuration(opts)==fcfs.CONFIG_ID
    common=['--bundle',str(bundle),'--models','m','--state',str(tmp_path/'state'),'--output',str(tmp_path/'out'),'--gpus','0,1,2,3']
    for argv in ('qualify --profile fcfs --family ministral --campaign x --coefficients c','qualify --profile fcfs --family qwen --coefficients c',
                 'run --profile fcfs --family qwen --campaign x --qualification q','calibrate --family qwen','qualify --family qwen --coefficients c'):
        monkeypatch.setattr(sys,'argv',['worker',*argv.split(),*common])
        with pytest.raises(SystemExit):worker.main()
    unauthorized=deepcopy(read(ROOT/'scripts/cloud/fcfs/campaign-20260916.json'));unauthorized['full_matrix_authorized']=False
    overlay=tmp_path/'unauthorized.json';write(overlay,unauthorized)
    monkeypatch.setattr(sys,'argv',['worker','run','--profile','fcfs','--family','qwen','--campaign',str(overlay),'--coefficients','c','--qualification','q',*common])
    with pytest.raises(ValueError,match='not authorized'):worker.main()
    assert not (tmp_path/'out').exists()

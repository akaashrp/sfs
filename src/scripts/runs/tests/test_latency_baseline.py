import asyncio,json,sys
from pathlib import Path
import pytest
from scripts.runs import experiments,experiments_sweep,qwen_baselines as qwen
from scripts.runs.tests.test_methodology_baselines import _run_kwargs
from sfs_core.routing.tests.test_latency_scheduler import Client
from sfs_core.routing import wait_time_scheduler
from sfs_core.routing.latency_history import audit_history
from scripts.runs.latency_validation import validate_cpu,source_hashes

def warmup_file(tmp_path):
    path=tmp_path/'warm.json'
    path.write_text(json.dumps({'data_role':'calibration','requests':[{'request_id':str(i),'bucket':qwen.BUCKETS[i%4],'prompt':f'calibration {i}','prompt_tokens':8} for i in range(32)]}))
    return path

def test_actual_run_policy_loads_no_sfs_or_methodology_artifacts(tmp_path,monkeypatch):
    monkeypatch.setattr(wait_time_scheduler,'load_tokenizer',lambda *a,**k:object())
    def forbidden(*a,**k):raise AssertionError('Predictor must not load')
    monkeypatch.setattr(wait_time_scheduler,'AccuracyPredictor',forbidden)
    monkeypatch.setattr(wait_time_scheduler,'OutputLengthPredictor',forbidden)
    clients={m:Client(m) for m in ['a','b','c']}
    kwargs=_run_kwargs('vllm_sr_latency',clients,tmp_path/'nonexistent',tmp_path)
    kwargs['latency_warmup_requests']=str(warmup_file(tmp_path))
    result=asyncio.run(experiments.run_policy(**kwargs))
    assert len(result['per_request'])==3
    assert result['summary']['failed_requests']==0
    assert all(r['usage_completion_tokens']==2 for r in result['per_request'])
    assert result['methodology_config']['warmup_completions']==96
    assert audit_history(result['methodology_config'])['selections']==3
    assert sum(c.counter for c in clients.values())==99

def test_cpu_gate_rejects_missing_or_stale_source(tmp_path):
    root=Path(__file__).resolve().parents[4]
    with pytest.raises(ValueError):validate_cpu(None,root)
    p=tmp_path/'audit.json';p.write_text(json.dumps({'status':'PASS_CPU','upstream_differential_passed':True,'source_sha256':{}}))
    with pytest.raises(ValueError,match='stale'):validate_cpu(p,root)
    data={'status':'PASS_CPU','upstream_differential_passed':True,'source_sha256':source_hashes(root)}
    p.write_text(json.dumps(data));assert validate_cpu(p,root)==data

def test_sweep_parser_registers_selector_and_requires_warmup(tmp_path,monkeypatch):
    monkeypatch.setattr(sys,'argv',['experiments','--experiment','router','--utilities','vllm_sr_latency'])
    with pytest.raises(SystemExit):experiments.parse_args()
    wrapper,forwarded=experiments_sweep._parse_wrapper_args(['--sweep','qps','--qps-values','7','8.3','8.6','8.9',
        '--qps-utilities','vllm_sr_latency','--output-dir',str(tmp_path/'out'),'--experiment','router',
        '--utilities','vllm_sr_latency','--latency-warmup-requests',str(warmup_file(tmp_path))])
    base=experiments_sweep._parse_experiment_args(forwarded)
    assert base.latency_warmup_requests
    assert wrapper.qps_utilities==['vllm_sr_latency']


def test_selector_figure_aggregation_preserves_quality_cost_tradeoff(tmp_path):
    from scripts.reporting.router_qps_sweep_summary import aggregate_jsons, DISPLAY_LABELS
    for rate in qwen.QPS:
        path=tmp_path/f'{rate}.json'
        path.write_text(json.dumps({'config':{'request_rate_qps':rate,'lambda_weight':.1},'router':{'runs':[
            {'utility':policy,'summary':{'actual_cost':{'mean':2.}},'per_request':[
                {'actual_accuracy':.8,'actual_cost':2.,'system_entry_e2e_ttft_slo_met':True},
                {'actual_accuracy':.4,'actual_cost':2.,'system_entry_e2e_ttft_slo_met':False}]} for policy in qwen.POLICIES]}}))
    before={p:p.read_bytes() for p in tmp_path.glob('*.json')}
    summary,rates=aggregate_jsons([tmp_path])
    assert len(rates)==4 and len(summary['utilities'])==9
    for rate in rates:
        assert summary['qps'][rate]['vllm_sr_latency']['actual_slo_gated_utility_mean']==pytest.approx(.3)
    assert 'adaptation' in DISPLAY_LABELS['vllm_sr_latency']
    assert all(p.read_bytes()==b for p,b in before.items())


def test_qwen_sweep_subset_and_prefill_gate(tmp_path,monkeypatch):
    from scripts.prep.paper_ablation_data import write_json
    stage=tmp_path/'stage';(stage/'timing_models').mkdir(parents=True)
    (stage/'timing_models/methodology_calibration.json').write_text('{}')
    mp=tmp_path/'manifest.json';mp.write_text('{}')
    manifest={'sfs_root':str(tmp_path),'manifest_path':str(mp),'experiment_argv':['--num-requests','16000']}
    output=tmp_path/'out';output.mkdir()
    with pytest.raises(ValueError,match='bootstrap'):
        qwen.run_sweep(manifest,stage,output,tmp_path/'instances.json',tmp_path/'predictor')
    captured=[]
    monkeypatch.setattr(qwen.subprocess,'run',lambda argv,**kw:captured.append(argv))
    qwen.run_sweep(manifest,stage,output,tmp_path/'instances.json',tmp_path/'predictor',policies=['vllm_sr_latency'])
    argv=captured[0]
    assert argv[argv.index('--qps-values')+1:argv.index('--qps-utilities')]==['7.0','8.3','8.6','8.9']
    assert argv[argv.index('--qps-utilities')+1:argv.index('--output-dir')]==['vllm_sr_latency']
    assert argv[argv.index('--num-requests')+1]=='16000'

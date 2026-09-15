"""Small synthetic cells exercise real joins, audits and plot generation.

Only manifest input loading and the fixture's request count are substituted;
canonical 16k ingestion is validated separately against the saved request map.
"""
import json,sys
from pathlib import Path
from types import SimpleNamespace
from scripts.runs import qwen_baselines as qwen
from scripts.reporting import collate_qwen_baselines as col
from scripts.reporting import router_qps_sweep_summary as report
from scripts.eval import augment_router_actual_accuracy as aug
from sfs_core.routing.latency_history import LatencyHistory


def test_multiple_raw_roots_join_selector_and_preserve_originals(tmp_path,monkeypatch):
    roots=[tmp_path/'old_new',tmp_path/'selector'];refs=tmp_path/'refs';refs.mkdir()
    for root in roots:(root/'outputs').mkdir(parents=True)
    manifest={'qps_values':list(qwen.QPS),'sfs_root':str(tmp_path),'request_map':'fixture.csv','canonical_full_curve_cells':{}}
    real_contract=col.contract
    fixture_contract=lambda m:{**real_contract(m),'requests_per_cell':3}
    monkeypatch.setattr(col,'contract',fixture_contract)
    monkeypatch.setattr(qwen,'validate_manifest',lambda _:manifest)
    real_audit=qwen.audit_run
    def fixture_audit(run,expected):
        assert expected==16000
        return real_audit(run,3)
    monkeypatch.setattr(qwen,'audit_run',fixture_audit)
    monkeypatch.setattr(aug,'load_req_map',lambda _: {f'req-{i}':aug.ReqMapEntry(f'req-{i}','alpaca',str(i)) for i in range(3)})
    monkeypatch.setattr(aug,'load_quality_index',lambda _: {(m,'alpaca',str(i)):.8 for m in qwen.MODELS for i in range(3)})
    for rate in qwen.QPS:
        for policy in qwen.POLICIES:
            metadata=None;terms=[]
            if policy=='vllm_sr_latency':
                h=LatencyHistory(qwen.MODELS);h.reset()
                for j in range(32):
                    for model in qwen.MODELS:
                        for metric in h.percentiles:h.update(model,metric,.1,request_id=f'warmup:{model}:{j}',generation=1)
                for i in range(3):
                    terms.append(h.select())
                    for metric in h.percentiles:h.update(qwen.MODELS[0],metric,.1,request_id=f'req-{i}',generation=1)
                metadata={**h.metadata(),'warmup_completions':96,'instance_models':{'vllm-0.6b':qwen.MODELS[0]}}
            rows=[{'request_id':f'req-{i}','scheduler_request_id':f'req-{i}','response_id':f'{rate}:{policy}:{i}',
                   'instance_id':'vllm-0.6b','response_model':qwen.MODELS[0],'bucket':'alpaca',
                   'system_entry_offset_s':i/rate,'system_entry_to_dispatch_ms':1.,'queue_delay_ms':1.,'ttft_ms':2.,
                   'system_entry_e2e_ttft_ms':3.,'system_entry_e2e_ttft_slo_met':True,'actual_cost':.01,'usage_completion_tokens':2,
                   'methodology_terms':terms[i] if terms else {}} for i in range(3)]
            run={'utility':policy,'per_request':rows,'summary':{'succeeded_requests':3,'failed_requests':0,'system_entry_e2e_ttft_slo_missing_count':0,'system_entry_e2e_ttft_slo_attainment_pct':100.,'system_entry_e2e_ttft_ms':{'mean':3.,'p50':3.,'p90':3.}}}
            if metadata:run['methodology_config']=metadata
            payload={'config':{**real_contract(manifest)['configuration'],'request_rate_qps':rate,'seed':69,'arrival_process':'poisson',
                               'prompt_source':{'holdout_prompts_per_bucket':4000},'instance_metadata':{'serving_profile':qwen.PROFILE}},
                     'request_set':{'num_requests':3},'router':{'runs':[run]}}
            folder=roots[1]/'outputs' if policy=='vllm_sr_latency' else roots[0]/'outputs' if policy in qwen.NEW_POLICIES else refs
            path=folder/f'{rate}_{policy}_point0.json';path.write_text(json.dumps(payload))
            if folder==refs:manifest['canonical_full_curve_cells'][f'{rate:g}:{policy}']=str(path)
    manifest['canonical_reference_cells']=manifest['canonical_full_curve_cells']
    mp=tmp_path/'manifest.json';mp.write_text(json.dumps(manifest))
    originals={p:p.read_bytes() for p in tmp_path.rglob('*.json')}
    def plot(argv,**kwargs):
        monkeypatch.setattr(sys,'argv',['router_qps_sweep_summary',*argv[3:]])
        report.main()
    monkeypatch.setattr(col,'subprocess',SimpleNamespace(run=plot))
    result=col.collate(mp,roots,tmp_path/'derived_result')
    assert result['matrix_cells']==36 and result['raw_modified'] is False
    summary=json.loads((tmp_path/'derived_result/figures/router_qps_sweep_summary.json').read_text())
    assert len(summary['utilities'])==9
    for values in summary['qps'].values():
        for row in values.values():
            assert row['average_system_entry_e2e_ttft_ms']==3.
            assert row['system_entry_e2e_ttft_ms_slo_attainment_pct']==100.
    assert all('vllm_sr_latency' in values for values in summary['qps'].values())
    assert len(list((tmp_path/'derived_result/figures').glob('*.png')))==4
    assert all(p.read_bytes()==b for p,b in originals.items())

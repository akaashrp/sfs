import asyncio
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from scripts.cloud.common import locks, set_option, expand
from scripts.cloud.pool import config, server_argv
from scripts.runs import qwen_baselines as qwen
from scripts.runs import ministral3_latency as ministral
from scripts.runs.ministral3_methodology_stage import POLICIES as CORE
from sfs_core.routing.latency_stream import submit_latency_stream


class Stream:
    def __init__(self, model): self.model=model;self.closed=False
    def __aiter__(self): return self.chunks()
    async def chunks(self):
        for row in [dict(choices=[{'index':0,'finish_reason':None}]),
                    dict(choices=[{'index':0,'finish_reason':'stop'}]),
                    dict(choices=[],usage={'completion_tokens':2})]:
            yield {'id':'response','model':self.model,**row}
    async def close(self):self.closed=True


@pytest.mark.parametrize('canonical,alias',ministral.ALIASES.items())
def test_ministral_alias_is_validated_before_normalization(canonical,alias):
    async def run(model):
        stream=Stream(model)
        async def submit(**kwargs):return stream
        client=ministral.AliasClient(SimpleNamespace(model_id=canonical,submit_request=submit))
        try:
            result=await submit_latency_stream(client,{},started_perf=0.,on_first=lambda _:None)
            return result
        finally:assert stream.closed
    assert asyncio.run(run(alias))['model']==canonical
    with pytest.raises(ValueError,match='served alias'):asyncio.run(run('wrong-model'))


def test_qwen_transport_and_policy_set_remain_strict():
    assert len(CORE)==8 and len(qwen.POLICIES)==9 and len(set(qwen.POLICIES))==9
    assert ministral.POLICIES==(*CORE,'vllm_sr_latency')
    assert qwen.QPS==(7.,8.3,8.6,8.9)
    assert len(qwen.NEW_POLICIES)*len(qwen.QPS)+len(ministral.POLICIES)*len(ministral.QPS)+12==68
    with pytest.raises(ValueError,match='Ministral candidate'):
        ministral.AliasClient(SimpleNamespace(model_id='qwen3-8b'))
    async def run():
        async def submit(**kwargs):return Stream('qwen-alias')
        return await submit_latency_stream(SimpleNamespace(model_id='qwen3-8b',submit_request=submit),{},started_perf=0,on_first=lambda _:None)
    with pytest.raises(ValueError,match='selected candidate'):asyncio.run(run())


def test_gpu_ownership_cannot_overlap(tmp_path):
    with locks(tmp_path,['gpu-A','gpu-B']):
        with pytest.raises(RuntimeError,match='Already owned'):
            with locks(tmp_path,['gpu-B','gpu-C']):pass
        with locks(tmp_path,['gpu-C']):pass
    with locks(tmp_path,['gpu-A']):pass


def test_cli_replacement_preserves_effective_settings():
    assert set_option(['--count','10000','--seed','69','--count','8000'],'--count',8000)==['--seed','69','--count','8000']
    assert expand(['@BUNDLE@/qwen/holdout'], '/data/bundle',{})==['/data/bundle/qwen/holdout']


def test_cloud_qwen_server_preserves_all_canonical_arguments(tmp_path):
    cfg=qwen.pool_config({},(9100,9101,9102),'isolation')
    from scripts.cloud.common import ROOT
    for i,row in enumerate(cfg['instances']):
        canonical=qwen.server_argv(ROOT,tmp_path/'model',row,i,tmp_path)
        cloud=server_argv('qwen',tmp_path/'model',row,i,tmp_path,ROOT/'src/assets/predictors/output_length_predictor')
        # Only placement of the unchanged predictor flag and loopback binding differ.
        normalized=set_option(canonical,'--output-length-model-path',ROOT/'src/assets/predictors/output_length_predictor')
        assert cloud==normalized+['--host','127.0.0.1']
        assert cloud[cloud.index('--tensor-parallel-size')+1]==str((1,1,2)[i])


def test_mlp_length_reaches_every_qwen_server(tmp_path):
    for i,row in enumerate(qwen.pool_config({})['instances']):
        argv=server_argv('qwen',tmp_path/'model',row,i,tmp_path,tmp_path/'mlp_length')
        assert argv.count('--output-length-model-path')==1
        assert argv[argv.index('--output-length-model-path')+1]==str(tmp_path/'mlp_length')


def test_changed_or_incomplete_cell_cannot_be_completed(tmp_path):
    from scripts.cloud.worker import audit_cell
    with pytest.raises(ValueError,match='exactly one'):
        audit_cell({'router':{'runs':[]}}, {'requests':8000})


def test_release_is_bound_to_evidence(tmp_path):
    from scripts.cloud.common import write,digest
    from scripts.cloud.control import release
    evidence=tmp_path/'smoke.json';write(evidence,{'passed':True})
    write(tmp_path/'qualification.json',{'status':'GPU_MEASURED_REVIEW_REQUIRED','files':{'smoke.json':digest(evidence)}})
    write(evidence,{'passed':False})
    with pytest.raises(ValueError,match='Changed qualification'):
        release(tmp_path,'x'*40,'y'*40)


def test_real_ministral_openai_sse_alias():
    import httpx
    from openai import AsyncOpenAI
    from sfs_core.routing.wait_time_scheduler import InstanceClient
    async def run():
        async def handler(request):
            assert json.loads(request.content)['model']=='ministral3-3b-instruct'
            stream=Stream('ministral3-3b-instruct');body=[]
            async for row in stream:
                row.update(object='chat.completion.chunk',created=1)
                body.append('data: '+json.dumps(row)+'\n\n')
            return httpx.Response(200,headers={'content-type':'text/event-stream'},text=''.join(body)+'data: [DONE]\n\n')
        sdk=AsyncOpenAI(api_key='test',base_url='http://test/v1',http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        client=InstanceClient.__new__(InstanceClient);client._client=sdk;client.default_model='ministral3-3b-instruct';client.model_id='ministral3-3b'
        try:
            result=await submit_latency_stream(ministral.AliasClient(client),{'messages':[{'role':'user','content':'test'}]},started_perf=0,on_first=lambda _:None)
            assert result['model']=='ministral3-3b' and result['usage'].completion_tokens==2
        finally:await sdk.close()
    asyncio.run(run())


def test_protected_qwen_and_shared_source_bytes():
    from scripts.cloud.common import ROOT,digest,read
    for name,expected in read(ROOT/'scripts/cloud/protected-source.json').items():
        assert digest(ROOT/name)==expected, name


def test_ministral_alias_runs_through_shared_scheduler(tmp_path,monkeypatch):
    from scripts.runs.tests.test_methodology_baselines import _run_kwargs
    from scripts.runs.tests.test_latency_baseline import warmup_file
    from sfs_core.routing.tests.test_latency_scheduler import Client
    from sfs_core.routing import wait_time_scheduler
    from scripts.runs import experiments
    from sfs_core.routing.latency_history import audit_history
    monkeypatch.setattr(wait_time_scheduler,'load_tokenizer',lambda *a,**k:object())
    class ServedClient(Client):
        async def submit_request(self,**payload):
            stream=await super().submit_request(**payload)
            # Server fixture emits public aliases; adapter must verify and normalize.
            outer=self
            class PublicStream:
                def __aiter__(self):return self.events()
                async def events(self):
                    async for chunk in stream:
                        yield {**chunk,'model':ministral.ALIASES[outer.model_id]}
                async def close(self):await stream.close()
            return PublicStream()
    clients={m:ServedClient(m) for m in ministral.ALIASES}
    kwargs=_run_kwargs('vllm_sr_latency',ministral.selector_clients(clients),tmp_path/'missing',tmp_path)
    kwargs['latency_warmup_requests']=str(warmup_file(tmp_path))
    result=asyncio.run(experiments.run_policy(**kwargs))
    assert result['summary']['failed_requests']==0
    assert result['methodology_config']['warmup_completions']==96
    assert audit_history(result['methodology_config'])['selections']==3
    assert all(r['response_model'] in ministral.ALIASES for r in result['per_request'])
    assert sum(c.counter for c in clients.values())==99


def test_fresh_calibration_orchestration_writes_measured_inputs(tmp_path,monkeypatch):
    from scripts.cloud.worker import calibrate
    from scripts.runs import ministral3_methodology_stage as stage
    from scripts.prep import fit_methodology_calibration as fitting
    from sfs_core.shared import trace_theta
    from scripts.cloud.common import read,write
    async def idle(*args,**kwargs):pass
    monkeypatch.setattr(stage,'wait_drained',idle)
    monkeypatch.setattr(asyncio,'sleep',idle)
    req=SimpleNamespace(prompt='Explain queues',request_id='cal-1',prompt_tokens=10)
    monkeypatch.setattr(stage,'length_stratified_requests',lambda requests:[req])
    monkeypatch.setattr(stage,'smoke_requests',lambda requests,**kw:[req]*2)
    monkeypatch.setattr(trace_theta,'estimate_score_proxy_metrics_from_batch_stats',lambda **kw:{'decode_tps':1.})
    captured=[]
    def fit(path,out):captured.append(read(path));out.mkdir();write(out/'methodology_calibration.json',{})
    monkeypatch.setattr(fitting,'fit_manifest',fit)
    trace=tmp_path/'batch_stats_ministral3-3b.csv'
    trace.write_text('prefill,prefill_sq_sum,decode,num_seqs,sum_tokens,prefill_x_processed_ctx_sum,exec\n')
    class Client:
        model_id=instance_id='ministral3-3b'
        async def submit_request(self,**kwargs):
            assert kwargs['extra_body']['chat_template_kwargs']=={}
            with trace.open('a') as stream:stream.write('10,100,0,1,10,0,0.01\n')
            return SimpleNamespace(id=kwargs['extra_body']['request_id'],usage=SimpleNamespace(model_dump=lambda:{'completion_tokens':1}))
    asyncio.run(calibrate('ministral',{'profile':{'max_num_batched_tokens':32768}},[req],{'m':Client()},
        SimpleNamespace(system_prompt='You are helpful.',chat_template_kwargs={}),tmp_path))
    assert captured[0]['data_role']=='calibration'
    assert captured[0]['models']['ministral3-3b']['num_queries']==2
    assert len((tmp_path/'calibration_trace_ministral3-3b.csv').read_text().splitlines())==4

import asyncio
from types import SimpleNamespace
import pytest
from sfs_core.routing.latency_stream import submit_latency_stream, RawChunks, field

class Stream:
    def __init__(self,chunks):self.chunks=chunks;self.closed=False
    def __aiter__(self):return self.iterator()
    async def iterator(self):
        for c in self.chunks:
            if isinstance(c,BaseException):raise c
            yield c
    async def close(self):self.closed=True

class Client:
    model_id='a'
    def __init__(self,chunks):self.stream=Stream(chunks)
    async def submit_request(self,**payload):
        assert payload['stream'] and payload['stream_options']=={'include_usage':True}
        return self.stream

def chunks(count=2):
    return [{'id':'x','model':'a','choices':[{'index':0,'delta':{'role':'assistant'},'finish_reason':None}]},
            {'id':'x','model':'a','choices':[{'index':0,'delta':{},'finish_reason':'stop'}]},
            {'id':'x','model':'a','choices':[],'usage':{'completion_tokens':count}}]

@pytest.mark.parametrize('count',[0,1,2])
def test_live_first_event_actual_usage_and_total_duration(count):
    c=Client(chunks(count));first=[];ticks=iter([12.,14.,15.,16.])
    r=asyncio.run(submit_latency_stream(c,{},started_perf=10.,on_first=first.append,clock=lambda:next(ticks)))
    assert first==[2.];assert r['tpot_s']==(6/count if count else None);assert c.stream.closed

@pytest.mark.parametrize('bad',[[],chunks()[:-1],chunks()[:1]+[RuntimeError('disconnect')],chunks()[:1]+[asyncio.CancelledError()],
                                [{'id':'x','model':'wrong'}], [{'id':'x','model':'a'},{'id':'different','model':'a'}]])
def test_bad_streams_close_and_fail(bad):
    c=Client(bad)
    with pytest.raises((ValueError,RuntimeError,asyncio.CancelledError)):
        asyncio.run(submit_latency_stream(c,{},started_perf=0.,on_first=lambda _:None))
    assert c.stream.closed


def test_real_openai_sse_client(monkeypatch):
    import httpx,json
    from openai import AsyncOpenAI
    from sfs_core.routing.wait_time_scheduler import InstanceClient
    async def run():
        async def handler(request):
            body=json.loads(request.content)
            assert body['model']=='a' and body['stream_options']['include_usage']
            payload=[]
            for chunk in chunks():
                chunk.update(object='chat.completion.chunk',created=1)
                payload.append('data: '+json.dumps(chunk)+'\n\n')
            return httpx.Response(200,headers={'content-type':'text/event-stream'},text=''.join(payload)+'data: [DONE]\n\n')
        sdk=AsyncOpenAI(api_key='test',base_url='http://test/v1',http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        client=InstanceClient.__new__(InstanceClient);client._client=sdk;client.default_model=client.model_id='a'
        first=[]
        result=await submit_latency_stream(client,{'messages':[{'role':'user','content':'x'}]},started_perf=0.,on_first=first.append)
        raw=await client.submit_request(messages=[],stream=True,stream_options={'include_usage':True});assert isinstance(raw,RawChunks);await raw.close()
        assert field(result['usage'],'completion_tokens')==2 and len(first)==1
        await sdk.close()
    asyncio.run(run())


# Raw SSE consumer over the SDK stream's httpx body (compact vLLM chunk format).
def vllm_chunks(model='a', rid='chatcmpl-1'):
    def chunk(delta, finish=None, **extra):
        return {'id':rid,'object':'chat.completion.chunk','created':1,'model':model,
                'choices':[{'index':0,'delta':delta,'logprobs':None,'finish_reason':finish}],'usage':None,**extra}
    return [chunk({'role':'assistant','content':''}),
            chunk({'content':'Hello'}), chunk({'content':' "finish_reason":"stop" "usage":{'}), chunk({'content':'\u2028x\r\n'}),
            chunk({}, 'stop'),
            {'id':rid,'object':'chat.completion.chunk','created':1,'model':model,'choices':[],'usage':{'prompt_tokens':3,'completion_tokens':3,'total_tokens':6}}]

def sse(chunks, done=True):
    import json
    body=b''.join(b'data: '+json.dumps(c,separators=(',',':'),ensure_ascii=False).encode()+b'\n\n' for c in chunks)
    return body+(b'data: [DONE]\n\n' if done else b'')

class RawClient:
    model_id='a'
    def __init__(self,body,block=7):
        self.blocks=[body[i:i+block] for i in range(0,len(body),block)];self.closed=False;self.decoded=[]
    async def aiter_bytes(self):
        for b in self.blocks: yield b
    async def close(self):self.closed=True
    async def submit_request(self,**payload):
        assert payload['stream'] and payload['stream_options']=={'include_usage':True}
        return RawChunks(SimpleNamespace(response=self,close=self.close))

def run_raw(body,monkeypatch=None,**kw):
    import itertools
    c=RawClient(body);first=[]
    if monkeypatch:
        import json;real=json.loads
        monkeypatch.setattr('sfs_core.routing.latency_stream.json.loads',lambda d:(c.decoded.append(d),real(d))[1])
    r=asyncio.run(submit_latency_stream(c,{},started_perf=10.,on_first=first.append,clock=lambda t=itertools.count(12.):next(t),**kw))
    return c,first,r

def test_raw_consumer_decodes_only_first_finish_and_usage_chunks(monkeypatch):
    c,first,r=run_raw(sse(vllm_chunks()),monkeypatch)
    assert first==[2.] and r['id']=='chatcmpl-1' and r['model']=='a' and c.closed
    assert r['usage']=={'prompt_tokens':3,'completion_tokens':3,'total_tokens':6}
    assert r['first_chunk_perf']==12. and r['completed_perf']==18. and r['tpot_s']==pytest.approx(8/3)
    assert len(c.decoded)==3 and not any(b'Hello' in d for d in c.decoded)
    assert 'Hello' not in repr(r)

def test_raw_consumer_usage_and_finish_in_one_chunk_and_generic_spacing():
    chunks=vllm_chunks()[:2]+[{'id':'chatcmpl-1','model':'a','choices':[{'index':0,'delta':{},'finish_reason':'length'}],'usage':{'completion_tokens':1}}]
    import json
    body=b''.join(b'data: '+json.dumps(c).encode()+b'\n\n' for c in chunks)+b': keepalive\n\ndata: [DONE]\n\n'
    c,first,r=run_raw(body)
    assert first==[2.] and r['tpot_s']==pytest.approx(5.) and c.closed

@pytest.mark.parametrize('body,match',[
    (sse(vllm_chunks(model='wrong')),'selected candidate'),
    (sse(vllm_chunks()[:2]+vllm_chunks(rid='chatcmpl-2')[2:]),'ID changed'),
    (sse(vllm_chunks()[:-1]),'Incomplete'),
    (sse(vllm_chunks()[:3],done=False),'Incomplete'),
    (sse([{'id':'x','model':'a','choices':[{'index':0,'delta':{},'finish_reason':'stop'}]}]),'Incomplete'),
    (sse([{'id':'x','model':'a','choices':[{'index':1,'delta':{},'finish_reason':None}]}]),'Unexpected stream choice'),
    (b'data: [1,2]\n\n','Malformed'),
])
def test_raw_consumer_guards(body,match):
    c=RawClient(body)
    with pytest.raises(ValueError,match=match):
        asyncio.run(submit_latency_stream(c,{},started_perf=0.,on_first=lambda _:None))
    assert c.closed

def test_raw_consumer_error_chunk_supplies_no_ttft():
    c=RawClient(sse([{'error':{'message':'boom','type':'InternalServerError'}}]));first=[]
    with pytest.raises(ValueError,match='Stream error'):
        asyncio.run(submit_latency_stream(c,{},started_perf=0.,on_first=first.append))
    assert first==[] and c.closed

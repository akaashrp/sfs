import asyncio
from types import SimpleNamespace
import pytest
from sfs_core.routing.latency_stream import submit_latency_stream

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
        assert result['usage'].completion_tokens==2 and len(first)==1
        await sdk.close()
    asyncio.run(run())

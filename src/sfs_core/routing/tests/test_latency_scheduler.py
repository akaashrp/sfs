import asyncio,json,time
from types import SimpleNamespace
import pytest
from sfs_core.routing import wait_time_scheduler
from sfs_core.routing.latency_history_scheduler import LatencyHistoryScheduler
from sfs_core.routing.latency_history import audit_history

class Client:
    def __init__(self,model):self.model_id=self.instance_id=model;self.counter=0;self.gate=None;self.streams=[]
    async def submit_request(self,**payload):
        assert payload['stream'] is True
        self.counter+=1;rid=f'{self.model_id}:{self.counter}';model=self.model_id;gate=self.gate
        class Stream:
            closed=False
            def __aiter__(self):return self.iterator()
            async def iterator(self):
                yield {'id':rid,'model':model,'choices':[{'delta':{'role':'assistant'}}]}
                if gate:await gate.wait()
                yield {'id':rid,'model':model,'choices':[{'finish_reason':'stop'}],'usage':{'prompt_tokens':8,'completion_tokens':2,'total_tokens':10}}
            async def close(self):self.closed=True
        stream=Stream();self.streams.append(stream);return stream
    def close(self):pass

@pytest.fixture
def fake_tokenizer(monkeypatch):
    monkeypatch.setattr(wait_time_scheduler,'load_tokenizer',lambda *a,**k:None)

def test_scheduler_live_feedback_before_completion_and_cancellation(fake_tokenizer):
    async def run():
        clients={m:Client(m) for m in ['a','b']};s=LatencyHistoryScheduler(clients)
        await s.warmup([{'messages':[]}]*3);await s.start()
        for c in clients.values():c.gate=asyncio.Event()
        done=asyncio.get_running_loop().create_future()
        await s.route_and_submit('test',await_dispatch=True,messages=[],_completion_future=done,_system_entry_perf=time.perf_counter())
        await asyncio.sleep(.01)
        observed=[e for e in s.history.events if e.get('request_id')=='test']
        assert [e['metric'] for e in observed]==['ttft'];assert not done.done()
        done.cancel();await asyncio.sleep(.01)
        await s.drain();await s.stop()
        assert not s._active
        assert all(st.closed for c in clients.values() for st in c.streams)
        audit_history(s.run_metadata())
    asyncio.run(run())

def test_scheduler_real_completion_sidecars_and_order(fake_tokenizer,tmp_path):
    async def run():
        s=LatencyHistoryScheduler({'a':Client('a')},request_log_path=str(tmp_path/'wait.log'),response_map_path=str(tmp_path/'map.log'))
        await s.warmup([{'messages':[]}]*3);await s.start();futures=[]
        for i in range(6):
            f=asyncio.get_running_loop().create_future();futures.append(f)
            await s.route_and_submit(str(i),messages=[],_completion_future=f,_system_entry_perf=time.perf_counter())
        rows=await asyncio.gather(*futures);await s.drain();await s.stop()
        assert all(r['usage_completion_tokens']==2 and not r.get('error') for r in rows)
        assert len({r['response_id'] for r in rows})==6
        assert audit_history(s.run_metadata())['selections']==6
        assert (tmp_path/'map.log').stat().st_size>0
    # Warm-up reset requires no workers; test the active guard before stopping.
    async def checked():
        s=LatencyHistoryScheduler({'a':Client('a')});await s.warmup([{'messages':[]}]*3);await s.start()
        with pytest.raises(ValueError,match='idle'):await s.warmup([{}])
        await s.stop()
    # remove no-op guard from main lifecycle scenario
    asyncio.run(run())
    asyncio.run(checked())

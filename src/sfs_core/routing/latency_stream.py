"""Online OpenAI SSE feedback, without buffering generated text.

First parsed SSE chunk (including role-only) approximates upstream first body
chunk. TPOT intentionally includes initial wait: total duration/output tokens.
"""
import time


def field(value, name, default=None):
    return value.get(name,default) if isinstance(value,dict) else getattr(value,name,default)


async def submit_latency_stream(client, payload, *, started_perf, on_first, clock=time.perf_counter):
    if payload.get('n',1) != 1: raise ValueError('Latency feedback requires n=1')
    args = dict(payload, stream=True, stream_options={'include_usage':True})
    stream = await client.submit_request(**args)
    first = None; usage = None; response_id = None; model = None; finished = False
    try:
        async for chunk in stream:
            now = clock()
            if first is None:
                first = now
                on_first(now-started_perf)
            cid = field(chunk,'id'); cm = field(chunk,'model')
            if cid:
                if response_id and response_id != cid: raise ValueError('Stream response ID changed')
                response_id = cid
            if cm:
                if cm != client.model_id: raise ValueError('Stream model differs from selected candidate')
                model = cm
            if field(chunk,'usage') is not None: usage = field(chunk,'usage')
            for choice in field(chunk,'choices',[]) or []:
                if field(choice,'index',0) != 0: raise ValueError('Unexpected stream choice')
                if field(choice,'finish_reason') is not None: finished = True
        completed = clock()
        count = field(usage,'completion_tokens')
        if first is None or not finished or not response_id or not model or type(count) is not int or count < 0:
            raise ValueError('Incomplete stream or missing actual completion usage')
        return {'id':response_id,'model':model,'usage':usage,'completed_perf':completed,
                'first_chunk_perf':first,'tpot_s':(completed-started_perf)/count if count>0 else None}
    finally:
        await stream.close()

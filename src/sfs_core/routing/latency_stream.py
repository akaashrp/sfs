"""Online OpenAI SSE feedback, without buffering generated text.

First parsed SSE chunk (including role-only) approximates upstream first body
chunk. TPOT intentionally includes initial wait: total duration/output tokens.
"""
import json
import time


def field(value, name, default=None):
    return value.get(name,default) if isinstance(value,dict) else getattr(value,name,default)


async def sse_data(response):
    """Yield each SSE event's `data` payload (bytes) from a streaming httpx response."""
    buffer = b''; parts = []
    async for block in response.aiter_bytes():
        buffer += block
        while (cut := buffer.find(b'\n')) >= 0:
            line, buffer = buffer[:cut].rstrip(b'\r'), buffer[cut+1:]
            if line.startswith(b'data:'): parts.append(line[5:].strip())
            elif not line and parts: yield b'\n'.join(parts); parts = []
    if parts: yield b'\n'.join(parts)


class RawChunks:
    """Lightweight consumer of the SDK stream's raw httpx SSE body.

    Yields dict chunks for `submit_latency_stream`. JSON is decoded only for the
    first chunk and for chunks that may carry new facts: an unknown response
    ID/model, a non-null finish_reason, or a usage object. Other compact vLLM
    chunks are recognized by substring checks and yielded as a cached ID/model
    stub, so generated text is never decoded or retained; any other formatting
    falls back to full decoding, never to skipped guards.
    """
    def __init__(self, stream): self.stream = stream; self.stub = None; self.tokens = ()

    def __aiter__(self): return self.chunks()

    async def chunks(self):
        async for data in sse_data(self.stream.response):
            if data == b'[DONE]': return
            if (self.stub and all(t in data for t in self.tokens) and b'"index":0' in data
                    and b'"finish_reason":null' in data and b'"finish_reason":"' not in data
                    and (b'"usage"' not in data or b'"usage":null' in data)):
                yield self.stub; continue
            chunk = json.loads(data)
            if not isinstance(chunk, dict): raise ValueError('Malformed stream chunk')
            if chunk.get('error') is not None: raise ValueError(f"Stream error: {chunk['error']}")
            if self.stub is None and chunk.get('id') and chunk.get('model'):
                self.stub = {'id': chunk['id'], 'model': chunk['model']}
                self.tokens = tuple(json.dumps({k: v}, separators=(',', ':'), ensure_ascii=False)[1:-1].encode()
                                    for k, v in self.stub.items())
            yield chunk

    async def close(self): await self.stream.close()


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

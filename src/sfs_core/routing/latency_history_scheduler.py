"""Event-loop-owned latency history routing with isolated calibration warm-up."""
import asyncio
import time
from .latency_history import LatencyHistory, POLICY
from .latency_stream import submit_latency_stream, field
from .wait_time_scheduler import WaitTimeScheduler, RoutedRequest, WaitTimeResult
from .methodology_scheduler import MethodologyScheduler


class LatencyHistoryScheduler(WaitTimeScheduler):
    def __init__(self, instances, **kwargs):
        kwargs.update(worker_count=1, accuracy_model_path=None, output_length_model_path=None,
                      readiness_predictor_path=None, enable_wait_time_polling=False)
        super().__init__(instances, **kwargs)
        self.history = LatencyHistory([c.model_id for c in instances.values()])
        self.by_model = {c.model_id:k for k,c in instances.items()}
        self._active = set()
        self._request_ids = set()
        self.warmup_count = 0

    async def start(self):
        if not self._workers:
            self.history.require_warm()
            self._workers.append(asyncio.create_task(self._worker_loop()))

    async def warmup(self, payloads):
        if self._active or self._workers: raise ValueError('Warm-up requires idle fresh scheduler')
        self.history.reset()
        for index,payload in enumerate(payloads):
            async def one(model, key):
                rid = f'warmup:{self.history.generation}:{index}:{model}'
                start = time.perf_counter()
                await self._observe(key,payload,rid,start)
                self.warmup_count += 1
            results = await asyncio.gather(*(one(model,key) for model,key in self.by_model.items()), return_exceptions=True)
            for result in results:
                if isinstance(result,BaseException): raise result
        self.history.require_warm()

    async def _observe(self,key,payload,rid,start):
        if rid in self._request_ids: raise ValueError('Duplicate latency request ID')
        self._request_ids.add(rid); self._active.add(rid)
        generation = self.history.generation
        model = self._instances[key].model_id
        def first(value):
            self.history.update(model,'ttft',value,request_id=rid,generation=generation)
        try:
            response = await submit_latency_stream(self._instances[key],payload,started_perf=start,on_first=first)
            if response['tpot_s'] is not None:
                self.history.update(model,'tpot',response['tpot_s'],request_id=rid,generation=generation)
            return response
        finally:
            self._active.discard(rid)

    async def _dispatch(self,queued):
        completion = queued.payload.get('_completion_future')
        if completion is not None and completion.cancelled(): return
        selection_start = time.perf_counter()
        decision = self.history.select()
        selection_ms = (time.perf_counter()-selection_start)*1000
        selected = self.by_model[decision['selected_model']]
        record = MethodologyScheduler._base_record(queued)
        record.update(instance_id=selected, route_strategy=POLICY, wait_estimator=POLICY,
                      methodology_terms={'policy':POLICY, 'selection_ms':selection_ms, **decision}, dispatch_perf=time.perf_counter())
        payload = {k:v for k,v in queued.payload.items() if not k.startswith('_')}
        engine_id = self._attach_engine_request_id(payload,queued.request_id)
        record['methodology_terms']['engine_request_id'] = engine_id
        task = asyncio.create_task(self._complete(queued,payload,record))
        self._submit_tasks.add(task)
        task.add_done_callback(self._submit_tasks.discard)
        if completion is not None:
            completion.add_done_callback(lambda future: task.cancel() if future.cancelled() and not task.done() else None)
        if not queued.result_future.done():
            queued.result_future.set_result(RoutedRequest(queued.request_id,selected,None,decision))

    async def _complete(self,queued,payload,record):
        try:
            response = await self._observe(record['instance_id'],payload,queued.request_id,record['system_entry_perf'])
            record.update(completed_perf=response['completed_perf'], response_id=response['id'], response_model=response['model'],
                          feedback_first_chunk_perf=response['first_chunk_perf'])
            for name in ('prompt_tokens','completion_tokens','total_tokens'):
                record['usage_'+name] = field(response['usage'],name)
            await self._log_response_mapping(queued.request_id,record['instance_id'],response['id'],response['model'])
            wait = WaitTimeResult(record['instance_id'],0.,time.time(),{'methodology_terms':record['methodology_terms']})
            self._request_log[queued.request_id] = wait
            await self._log_to_file(queued.request_id,wait)
        except BaseException as exc:
            record.update(completed_perf=time.perf_counter(),error=f'{type(exc).__name__}: {exc}')
            if isinstance(exc,asyncio.CancelledError): raise
        finally:
            record.setdefault('completed_perf',time.perf_counter())
            record['latency_ms'] = (record['completed_perf']-record['started_perf'])*1000
            completion = queued.payload.get('_completion_future')
            if completion is not None and not completion.done(): completion.set_result(record)

    def run_metadata(self):
        return {**self.history.metadata(), 'warmup_completions':self.warmup_count, 'instance_models':{k:c.model_id for k,c in self._instances.items()}}

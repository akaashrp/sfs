"""Verify every frozen prompt through the actual OpenAI chat admission path."""
import argparse
import asyncio
from array import array
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from scripts.cloud.common import ROOT, read, write, digest, validate_bundle
from scripts.cloud.worker import arguments, parse
from scripts.cloud.fcfs.config import manifest,SETTINGS


async def audit(bundle, models, output):
    from scripts.runs import experiments as exp
    from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
    from vllm.entrypoints.openai.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.serving_engine import OpenAIServing
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.processor import Processor
    m=validate_bundle(bundle);d=m['families']['qwen']
    args=parse(arguments(d,bundle,'canonical',m))
    evaluation,_,_=exp._build_request_set(args)
    calibration=[exp.ExperimentRequest(**json.loads(line)) for line in (bundle/'qwen/calibration_requests.jsonl').read_text().splitlines()]
    warmup=[exp.ExperimentRequest(**r) for r in read(bundle/'qwen/warmup.json')['requests']]
    assert len(evaluation)==16000 and len(calibration)==10000 and len(warmup)==32
    with (bundle/'qwen/request_map.csv').open() as stream:mapped={r['req_id']:r for r in csv.DictReader(stream)}
    for r in evaluation:
        assert r.prompt_tokens==int(mapped[r.request_id]['prompt_tokens']) and r.bucket==mapped[r.request_id]['bucket']
    calibration_keys={(r.request_id,r.bucket,r.prompt) for r in calibration}
    assert all((r.request_id,r.bucket,r.prompt) in calibration_keys for r in warmup)
    template_path=ROOT/'src/assets/templates/chat_template_qwen3.jinja';template=template_path.read_text()
    reports={};output.mkdir(parents=True,exist_ok=False)
    write(output/'manifest.json',manifest(bundle))
    for model_id in d['models']:
        path=Path(models[model_id])
        if path.name!=m['models'][model_id]['revision']:raise ValueError('Model revision changed')
        model=ModelConfig(model=str(path),tokenizer=str(path),dtype='auto',max_model_len=65536,
            hf_overrides={'rope_scaling':d['profile']['rope_scaling']})
        config=VllmConfig(model_config=model,cache_config=CacheConfig(block_size=16,enable_prefix_caching=False,gpu_memory_utilization=.90),
            scheduler_config=SchedulerConfig(max_model_len=65536,max_num_seqs=512,max_num_batched_tokens=65536,
                enable_chunked_prefill=False,long_prefill_token_threshold=0,policy='fcfs',async_scheduling=False))
        assert config.scheduler_config.chunked_prefill_enabled is False
        assert config.scheduler_config.long_prefill_token_threshold==0
        processor=Processor(config)
        engine=AsyncLLM.__new__(AsyncLLM);engine.model_config=model;engine.vllm_config=config;engine.processor=processor
        serving=OpenAIServing(engine_client=engine,model_config=model,models=None,request_logger=None);serving._processor=processor
        roles={}
        try:
            for role,requests in [('evaluation',evaluation),('calibration',calibration),('latency_warmup',warmup)]:
                maximum=0;difference=0;token_digest=hashlib.sha256();identity=hashlib.sha256()
                with (output/f'{model_id}-{role}-lengths.csv').open('x') as f:
                    writer=csv.writer(f);writer.writerow(['request_id','bucket','frozen_prompt_tokens','formatted_prompt_tokens','max_completion_tokens','remaining_context_after_allowance'])
                    for i,r in enumerate(requests):
                        messages=[{'role':'system','content':args.system_prompt},{'role':'user','content':r.prompt}]
                        req=ChatCompletionRequest(model=model_id,messages=messages,temperature=0,top_p=1,max_completion_tokens=8192)
                        _,_,prompts=await serving._preprocess_chat(req,processor.tokenizer,messages,template,'auto',chat_template_kwargs={'enable_thinking':False})
                        ids=prompts[0]['prompt_token_ids'];n=len(ids)
                        if n+8192>65536:raise ValueError(f'Formatted request exceeds FCFS context: {model_id}/{role}/{r.request_id}: {n}+8192')
                        maximum=max(maximum,n);difference+=n!=r.prompt_tokens
                        writer.writerow([r.request_id,r.bucket,r.prompt_tokens,n,8192,65536-n-8192])
                        token_digest.update(array('I',ids).tobytes());identity.update(json.dumps(asdict(r),sort_keys=True).encode())
                        if i%2000==0:print(model_id,role,i,'max_formatted_tokens',maximum,flush=True)
                roles[role]={'requests':len(requests),'maximum_formatted_prompt_tokens':maximum,'output_allowance':8192,
                    'minimum_context_headroom':65536-maximum-8192,'different_from_frozen_token_count':difference,
                    'token_stream_sha256':token_digest.hexdigest(),'request_identity_prompt_and_slo_sha256':identity.hexdigest(),
                    'lengths_csv_sha256':digest(output/f'{model_id}-{role}-lengths.csv')}
        finally:serving._tokenizer_executor.shutdown(wait=True)
        reports[model_id]={'roles':roles,'resolved_dtype':str(model.dtype),'resolved_max_model_len':model.max_model_len,
            'resolved_rope_scaling':model.hf_config.rope_scaling,'scheduler':{k:getattr(config.scheduler_config,k) for k in ['max_num_batched_tokens','max_num_seqs','long_prefill_token_threshold','chunked_prefill_enabled','policy','async_scheduling']},
            'model_config_sha256':digest(path/'config.json'),'tokenizer_config_sha256':digest(path/'tokenizer_config.json')}
    write(output/'audit.json',{'status':'PASS_ACTUAL_CHAT_INPUTS','models':reports,'bundle_sha256':digest(bundle/'bundle.json'),
        'chat_template_sha256':digest(template_path),'gpu_feasibility_verified':False,'frozen_max_prompt_tokens':max(r.prompt_tokens for r in evaluation),
        'request_map_sha256':digest(bundle/'qwen/request_map.csv'),'calibration_requests_sha256':digest(bundle/'qwen/calibration_requests.jsonl'),
        'evaluation_data_modified':False,'profile':SETTINGS,'roles_verified':['evaluation','calibration','latency_warmup']})


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--models',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from vllm.platforms.cpu import CpuPlatform
    with patch('vllm.platforms._current_platform',CpuPlatform()):asyncio.run(audit(a.bundle,read(a.models),a.output))

if __name__=='__main__':main()

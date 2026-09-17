"""Immutable workload with an explicit, separately qualified serving profile."""
from copy import deepcopy
from scripts.cloud.common import ROOT, read, set_option
from scripts.cloud.pool import config as canonical_config, server_argv as canonical_server

CONFIG_ID='qwen-fcfs-unchunked-65536'
RATES=(6.,7.,8.,8.3)
METHODS=('hard','round_robin','latency_agnostic','shortest_queue','score','lmdeploy_proxy','mooncake_prefill','routebalance','vllm_sr_latency')
BLOCKED=('hard','score')
SETTINGS={'scheduling_policy':'fcfs','chunked_prefill':False,'max_num_batched_tokens':65536,
          'max_model_len':65536,'max_num_seqs':512,'long_prefill_token_threshold':0,
          'gpu_memory_utilization':.90,'prefix_caching':False}


def family(manifest):
    d=deepcopy(manifest['families']['qwen'])
    d['profile'].update(SETTINGS)
    d['qps']=list(RATES);d['policies']=list(METHODS);d['configuration_id']=CONFIG_ID
    return d


def definition(bundle):return family(read(bundle/'bundle.json'))


def instances(bundle, ports, tag, coefficients=None):
    d=definition(bundle)
    cfg=canonical_config('qwen',d,{},ports,tag)
    cfg['serving_profile']=d['profile']
    for row in cfg['instances']:
        row.update(max_num_batched_tokens=65536,max_num_seqs=512,chunked_prefill_enabled=False,
                   long_prefill_token_threshold=0,max_model_len=65536,scheduling_policy='fcfs')
        if coefficients is not None:row['ttft_batch_model']=coefficients[row['model_id']]
    cfg['configuration_id']=CONFIG_ID
    cfg['coefficient_status']='FITTED_FOR_CONFIGURATION' if coefficients else 'CANONICAL_PLACEHOLDER_FOR_TRACE_COLLECTION_ONLY'
    return cfg


def server_argv(bundle, model_path, row, index, output, remaining_length=None):
    """Canonical Qwen argv (including any per-engine remaining-length rule) with only the scheduler settings changed."""
    argv=canonical_server('qwen',model_path,row,index,output,bundle/'qwen/length',remaining_length)
    argv[argv.index('--enable-chunked-prefill')]='--no-enable-chunked-prefill'
    for flag,value in [('--scheduling-policy','fcfs'),('--max-num-batched-tokens',65536),('--max-model-len',65536),
                       ('--max-num-seqs',512),('--long-prefill-token-threshold',0),('--gpu-memory-utilization',.90)]:
        argv=set_option(argv,flag,value)
    return argv


def manifest(bundle):
    b=read(bundle/'bundle.json')
    return {'schema_version':1,'id':CONFIG_ID,'status':'PREPARATION_ONLY_FULL_MATRIX_NOT_LAUNCHED',
        'profile':definition(bundle)['profile'],'models':{m:b['models'][m] for m in b['families']['qwen']['models']},
        'cells':[{'id':f'{CONFIG_ID}-{p}-{q:g}','family':'qwen','variant':'canonical','policy':p,'qps':q,'requests':16000,
                  'status':'BLOCKED_TOKEN_LENGTH_VALIDATION' if p in BLOCKED else 'AWAITING_CONFIGURATION_QUALIFICATION'} for p in METHODS for q in RATES],
        'evaluation':{'arrival_process':'poisson','seed':69,'slo_rule':'Frozen canonical Qwen SLOs without refitting',
                      'primary_judge':'pro','quality_scores':'Frozen canonical scores; common 15996 observed queries for Pro/Flash comparisons',
                      'max_completion_tokens':8192,'temperature':0,'top_p':1,'enable_thinking':False},
        'restrictions':['No full-matrix launch without subsequent user authorization','Do not deploy an unvalidated output-length change','Preserve canonical kernels, precision, chat template, YaRN/RoPE and TP1/1/2'],
        'second_serving_configuration':None,'profiling_reuse':'Only hash/profile-compatible artifacts from this configuration'}

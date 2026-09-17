"""Immutable workload with an explicit, separately qualified serving profile (the FCFS unchunked configuration).

The configuration itself is the `fcfs` row of scripts.cloud.serving.profiles; this module keeps the
FCFS names (CONFIG_ID, SETTINGS, METHODS, BLOCKED, RATES) and the 36-cell matrix manifest.
"""
from scripts.cloud.common import read
from scripts.cloud.serving import profiles
from scripts.cloud.serving.profiles import FCFS, RATES, METHODS, EVALUATION, RESTRICTIONS

CONFIG_ID=FCFS.configuration_id
BLOCKED=('hard','score')
SETTINGS=FCFS.settings


def family(manifest):return profiles.family(FCFS,manifest)


def definition(bundle):return family(read(bundle/'bundle.json'))


def instances(bundle, ports, tag, coefficients=None):return profiles.instances(FCFS,bundle,ports,tag,coefficients)


def server_argv(bundle, model_path, row, index, output, remaining_length=None):
    """Canonical Qwen argv (including any per-engine remaining-length rule) with only the scheduler settings changed."""
    return profiles.server_argv(FCFS,bundle,model_path,row,index,output,remaining_length)


def manifest(bundle):
    b=read(bundle/'bundle.json')
    return {'schema_version':1,'id':CONFIG_ID,'status':'PREPARATION_ONLY_FULL_MATRIX_NOT_LAUNCHED',
        'profile':definition(bundle)['profile'],'models':{m:b['models'][m] for m in b['families']['qwen']['models']},
        'cells':[{'id':f'{CONFIG_ID}-{p}-{q:g}','family':'qwen','variant':'canonical','policy':p,'qps':q,'requests':16000,
                  'status':'BLOCKED_TOKEN_LENGTH_VALIDATION' if p in BLOCKED else 'AWAITING_CONFIGURATION_QUALIFICATION'} for p in METHODS for q in RATES],
        'evaluation':dict(EVALUATION),'restrictions':list(RESTRICTIONS),
        'second_serving_configuration':None,'profiling_reuse':'Only hash/profile-compatible artifacts from this configuration'}

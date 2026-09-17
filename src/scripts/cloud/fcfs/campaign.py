"""Explicit 36-cell FCFS overlay with its own ledger identity; unlaunchable by default."""
import argparse
from copy import deepcopy
from pathlib import Path

from scripts.cloud.common import ROOT, read, write
from scripts.cloud.fcfs.config import CONFIG_ID, RATES, METHODS, BLOCKED, SETTINGS, family, manifest

PATH=ROOT/'scripts/cloud/fcfs/campaign-20260916.json'
KEYS=('id','family','variant','policy','qps','requests')


def build(bundle):
    m=manifest(bundle)
    return {'schema_version':1,'configuration_id':CONFIG_ID,'status':m['status'],'full_matrix_authorized':False,
        'profile':SETTINGS,'cells':m['cells'],'requests_total':sum(c['requests'] for c in m['cells']),
        'blocked_policies':list(BLOCKED),'models':m['models'],'evaluation':m['evaluation'],'restrictions':m['restrictions'],
        'gpu_allocation':{'qwen':[0,1,2,3],'debug':[7]},'ledger':'Separate FCFS completion ledger; cell IDs carry the configuration prefix and never collide with canonical cells',
        'qualification':['Actual chat admission audit PASS_ACTUAL_CHAT_INPUTS for all three models','Bounded GPU FCFS smoke for every model including 32B TP=2',
                         'worker calibrate: destination FCFS traces with placeholder coefficients','scripts.cloud.fcfs.coefficients fit: nonnegative SFS batch coefficients for this configuration',
                         'worker qualify with fitted coefficients: timing heads, nine policy smokes, load probes','coefficients validate on independent post-calibration batches; control release with reviews'],
        'launch_rule':'worker run/campaign refuse this overlay until full_matrix_authorized is true; flipping it requires explicit user authorization'}


def apply_campaign(bundle, campaign, mode):
    if campaign.get('configuration_id')!=CONFIG_ID or campaign.get('profile')!=SETTINGS:
        raise ValueError('Overlay is not the FCFS unchunked configuration')
    cells=campaign['cells']
    expected={('qwen',p,q) for p in METHODS for q in RATES}
    if {(c['family'],c['policy'],c['qps']) for c in cells}!=expected or len(cells)!=len(expected):
        raise ValueError('FCFS overlay does not match the nine-policy 6/7/8/8.3 grid')
    if len({c['id'] for c in cells})!=len(cells) or any(c['id']!=f"{CONFIG_ID}-{c['policy']}-{c['qps']:g}" for c in cells):
        raise ValueError('FCFS cell IDs must carry the configuration prefix')
    if any(c['variant']!='canonical' or c['requests']!=16000 for c in cells):
        raise ValueError('FCFS overlay changed predictor or request budget')
    if mode not in ('calibrate','qualify') and campaign.get('full_matrix_authorized') is not True:
        raise ValueError('FCFS full matrix is not authorized; only calibrate/qualify may use this overlay')
    result=deepcopy(bundle);result['families']['qwen']=family(bundle);result['configuration_id']=CONFIG_ID
    result['cells']=[{k:c[k] for k in KEYS} for c in cells if not c['status'].startswith('BLOCKED')]
    result['blocked_cells']=[c['id'] for c in cells if c['status'].startswith('BLOCKED')]
    result['requests_total']=sum(c['requests'] for c in result['cells'])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,default=PATH)
    a=p.parse_args();write(a.output,build(a.bundle));print(a.output)

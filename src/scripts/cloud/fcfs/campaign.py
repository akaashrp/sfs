"""Explicit 36-cell FCFS overlay with its own ledger identity; unlaunchable by default.

A side overlay may carry an explicit, user-authorized policy allowance for a
policy outside the nine (for example the prefill-throughput-estimator baseline
``hard_prefill_tps``); the nine-policy grid check stays in force for the main
overlay and every allowance is bound to the same configuration, rates, request
budget and cell-id prefix.
"""
import argparse
from copy import deepcopy
from pathlib import Path

from scripts.cloud.common import ROOT, read, write
from scripts.cloud.fcfs.config import CONFIG_ID, RATES, METHODS, BLOCKED, SETTINGS, family, manifest

PATH=ROOT/'scripts/cloud/fcfs/campaign-20260916.json'
KEYS=('id','family','variant','policy','qps','requests')
ALLOWANCE_KEYS={'policies','authorization'}


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


def allowed_policies(campaign):
    """Policies the overlay may schedule: the nine, or an explicit user-authorized allowance outside them."""
    allowance=campaign.get('policy_allowance')
    if allowance is None:return tuple(METHODS)
    if not isinstance(allowance,dict) or set(allowance)!=ALLOWANCE_KEYS:
        raise ValueError('policy_allowance must carry exactly policies and authorization')
    policies=allowance['policies']
    if (not isinstance(policies,list) or not policies or len(set(policies))!=len(policies)
            or any(not isinstance(p,str) or not p or p in METHODS for p in policies)):
        raise ValueError('policy_allowance.policies must name distinct policies outside the nine FCFS policies')
    if not isinstance(allowance['authorization'],str) or len(allowance['authorization'].strip())<30:
        raise ValueError('policy_allowance.authorization must record the explicit user authorization')
    return tuple(policies)


def build_allowance(bundle, policies, authorization, gpu_allocation=None):
    """Side overlay for user-authorized policies outside the nine under the same configuration."""
    m=manifest(bundle)
    overlay={'schema_version':1,'configuration_id':CONFIG_ID,'status':'USER_AUTHORIZED_SIDE_OVERLAY_QUALIFICATION_REQUIRED','full_matrix_authorized':True,
        'profile':SETTINGS,'policy_allowance':{'policies':list(policies),'authorization':authorization},
        'cells':[{'id':f'{CONFIG_ID}-{p}-{q:g}','family':'qwen','variant':'canonical','policy':p,'qps':q,'requests':16000,
                  'status':'AWAITING_CONFIGURATION_QUALIFICATION'} for p in policies for q in RATES],
        'blocked_policies':[],'models':m['models'],'evaluation':m['evaluation'],'restrictions':m['restrictions'],
        'gpu_allocation':gpu_allocation or {'qwen':[0,1,2,3]},
        'ledger':'Separate FCFS completion ledger; cell IDs carry the configuration prefix and never collide with canonical cells',
        'qualification':['Actual chat admission audit PASS_ACTUAL_CHAT_INPUTS for all three models','Bounded GPU FCFS smoke for every model including 32B TP=2',
                         'worker calibrate: destination FCFS traces with placeholder coefficients','scripts.cloud.fcfs.coefficients fit: nonnegative SFS batch coefficients for this configuration',
                         'worker qualify with fitted coefficients and this overlay: timing heads, allowed-policy smokes, load probes','coefficients validate on independent post-calibration batches; control release with reviews'],
        'launch_rule':'worker run refuses this overlay without a released qualification recorded against these overlay bytes; the allowance never extends the main 36-cell overlay'}
    overlay['requests_total']=sum(c['requests'] for c in overlay['cells'])
    allowed_policies(overlay)
    return overlay


def apply_campaign(bundle, campaign, mode):
    if campaign.get('configuration_id')!=CONFIG_ID or campaign.get('profile')!=SETTINGS:
        raise ValueError('Overlay is not the FCFS unchunked configuration')
    policies=allowed_policies(campaign)
    cells=campaign['cells']
    expected={('qwen',p,q) for p in policies for q in RATES}
    if {(c['family'],c['policy'],c['qps']) for c in cells}!=expected or len(cells)!=len(expected):
        raise ValueError('FCFS overlay does not match the nine-policy 6/7/8/8.3 grid' if policies==tuple(METHODS)
                         else 'FCFS side overlay does not match its allowed policies on the 6/7/8/8.3 grid')
    if len({c['id'] for c in cells})!=len(cells) or any(c['id']!=f"{CONFIG_ID}-{c['policy']}-{c['qps']:g}" for c in cells):
        raise ValueError('FCFS cell IDs must carry the configuration prefix')
    if any(c['variant']!='canonical' or c['requests']!=16000 for c in cells):
        raise ValueError('FCFS overlay changed predictor or request budget')
    if mode not in ('calibrate','qualify') and campaign.get('full_matrix_authorized') is not True:
        raise ValueError('FCFS full matrix is not authorized; only calibrate/qualify may use this overlay')
    result=deepcopy(bundle);result['families']['qwen']=family(bundle);result['configuration_id']=CONFIG_ID
    # Qualification smokes exactly the policies this overlay may schedule.
    result['families']['qwen']['policies']=list(policies)
    result['cells']=[{k:c[k] for k in KEYS} for c in cells if not c['status'].startswith('BLOCKED')]
    result['blocked_cells']=[c['id'] for c in cells if c['status'].startswith('BLOCKED')]
    result['requests_total']=sum(c['requests'] for c in result['cells'])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,default=PATH)
    p.add_argument('--policies',help='Comma-separated user-authorized policies outside the nine; writes a side overlay instead of the main one')
    p.add_argument('--authorization',default='',help='Explicit user authorization text recorded in the side overlay')
    p.add_argument('--gpus',help='Comma-separated Qwen GPU allocation recorded in the side overlay')
    a=p.parse_args()
    if a.policies:
        allocation={'qwen':[int(g) for g in a.gpus.split(',')]} if a.gpus else None
        overlay=build_allowance(a.bundle,a.policies.split(','),a.authorization,allocation)
    else:overlay=build(a.bundle)
    write(a.output,overlay);print(a.output)

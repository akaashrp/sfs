"""Explicit FCFS overlays with their own ledger identity: the 36-cell matrix and the SFS/SCORE sub-grid.

The matrix overlay (campaign-20260916.json) keeps its `hard`/`score` cells BLOCKED_* and carries no
remaining-length rule. The SFS/SCORE overlay (campaign-sfs-score-20260917.json, kind `fcfs_sfs_score`)
is exactly those eight cells, runnable, with the same flag-gated 0.6B remaining-length rule as the
canonical SFS/SCORE overlay; it is validated by the helper the canonical overlay uses. Both refuse
run/campaign until `full_matrix_authorized` is true.
"""
import argparse
from copy import deepcopy
from pathlib import Path

from scripts.cloud.common import ROOT, read, write
from scripts.cloud.fcfs.config import CONFIG_ID, RATES, METHODS, BLOCKED, SETTINGS, family, manifest

PATH=ROOT/'scripts/cloud/fcfs/campaign-20260916.json'
SFS_SCORE_PATH=ROOT/'scripts/cloud/fcfs/campaign-sfs-score-20260917.json'
CANONICAL_SFS_SCORE=ROOT/'scripts/cloud/sfs-score-campaign-20260917.json'
SFS_SCORE_KIND='fcfs_sfs_score'
KINDS=(None,SFS_SCORE_KIND)
INSPECT_MODES=('calibrate','qualify','inspect')   # modes that never launch a cell and so need no authorization
KEYS=('id','family','variant','policy','qps','requests')
AUTHORIZATION_20260917=('User authorized on 17 September 2026: unchunked SFS and SCORE cells run before the predictor ablations, '
                        'with the 0.6B remaining-length rule')


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


def build_sfs_score(bundle):
    """The eight unchunked SFS/SCORE cells with the canonical overlay's remaining-length block (Qwen models only)."""
    m=manifest(bundle);canonical_overlay=read(CANONICAL_SFS_SCORE);canonical=canonical_overlay['remaining_length']
    cells=[dict({k:c[k] for k in KEYS},status='runnable') for c in m['cells'] if c['policy'] in BLOCKED]
    off=[model for model in m['models'] if model not in canonical['rules']]
    return {'schema_version':1,'kind':SFS_SCORE_KIND,'configuration_id':CONFIG_ID,'status':'AUTHORIZED_PENDING_LANE_QUALIFICATION',
        'full_matrix_authorized':True,'authorization':AUTHORIZATION_20260917,
        'decision':'User decision 2026-09-17: the FCFS unchunked SFS (hard) and SCORE cells at 6/7/8/8.3 QPS x 16,000 requests run before the '
                   'predictor ablations, after the canonical SFS/SCORE cells, with the flag-gated remaining-length rule ON for qwen3-0.6b only '
                   '(running_all, q0.5, prompt_bin; the rule adopted from the canonical paired run) and qwen3-8b/qwen3-32b on the current rule. '
                   'Each lane is qualified separately under --profile fcfs with the fitted FCFS coefficients; the configuration id, coefficient '
                   'file hash, this overlay hash, the rule and the table sha256 are bound to every qualification and completed-ledger entry.',
        'profile':SETTINGS,'policies':list(BLOCKED),'cells':cells,'requests_total':sum(c['requests'] for c in cells),
        # SCORE's tuned Lagrange multiplier is inherited from the canonical overlay rather than restated, so one
        # tuned value covers every serving configuration by construction and cannot drift between them.
        'score_lambda_weight':canonical_overlay['score_lambda_weight'],
        'score_lambda_evidence':dict(canonical_overlay['score_lambda_evidence']),
        'models':m['models'],'evaluation':m['evaluation'],'restrictions':m['restrictions'],
        'gpu_allocation':{'lane_a':[0,1,2,3],'lane_b':[4,5,6,7]},
        'lanes':{'a':[c['id'] for c in cells if c['policy']=='hard'],'b':[c['id'] for c in cells if c['policy']=='score']},
        'ledger':'Same FCFS completion ledger and cell IDs as the matrix overlay, whose hard/score cells stay BLOCKED; nothing collides with canonical cells',
        'qualification':['Gates of the FCFS state root re-established under the merged source (test.sh, prepare cpu/serving, fcfs-inputs.json)',
                         'worker qualify --profile fcfs --coefficients <fitted FCFS coefficients> --campaign <this overlay> per lane: calibration, timing heads, hard and score smokes, load probes',
                         'qualification.json records configuration_id, coefficients_sha256, campaign_sha256, remaining_length_rule and the 0.6B table sha256',
                         'server_argv_qwen3-0.6b.json carries the --remaining-length-* flags; the 8B and 32B argv do not',
                         'control release with timing and load reviews per lane; worker run binds the qualification to the lane hardware'],
        'launch_rule':'worker run/campaign refuse this overlay until full_matrix_authorized is true; flipping it requires explicit user authorization',
        'remaining_length':{'tables':canonical['tables'],'files':dict(canonical['files']),'rules':dict(canonical['rules']),'off_models':off,
            'note':'Same rule and tables as the canonical SFS/SCORE overlay: only the qwen3-0.6b server receives --remaining-length-mode running_all '
                   '--remaining-length-table qwen3-0.6b.json --remaining-length-quantile 0.5 --remaining-length-conditioning prompt_bin, and the router '
                   'attaches that table for its missing-prediction fill on that instance only; qwen3-8b and qwen3-32b stay on the current rule. The '
                   'tables were built from canonical (chunked) calibration outputs; the evidence is the canonical paired run, adopted by the user for '
                   'every Qwen SFS/SCORE cell including this unchunked configuration. No unchunked paired run exists.',
            'evidence':deepcopy(canonical['evidence'])},
        'accepted_prior_source_digests':{}}


def apply_campaign(bundle, campaign, mode):
    if campaign.get('configuration_id')!=CONFIG_ID or campaign.get('profile')!=SETTINGS:
        raise ValueError('Overlay is not the FCFS unchunked configuration')
    kind=campaign.get('kind')
    if kind not in KINDS:raise ValueError(f'Unknown FCFS overlay kind: {kind!r}')
    policies=BLOCKED if kind==SFS_SCORE_KIND else METHODS
    cells=campaign['cells']
    expected={('qwen',p,q) for p in policies for q in RATES}
    if {(c['family'],c['policy'],c['qps']) for c in cells}!=expected or len(cells)!=len(expected):
        raise ValueError('FCFS overlay does not match the nine-policy 6/7/8/8.3 grid' if kind is None
                         else 'FCFS SFS/SCORE overlay does not match the hard/score 6/7/8/8.3 sub-grid')
    if len({c['id'] for c in cells})!=len(cells) or any(c['id']!=f"{CONFIG_ID}-{c['policy']}-{c['qps']:g}" for c in cells):
        raise ValueError('FCFS cell IDs must carry the configuration prefix')
    if any(c['variant']!='canonical' or c['requests']!=16000 for c in cells):
        raise ValueError('FCFS overlay changed predictor or request budget')
    if kind is None:
        if any(c['policy'] in BLOCKED and not c['status'].startswith('BLOCKED') for c in cells):
            raise ValueError('SFS and SCORE cells of the FCFS matrix stay blocked; they run only through the FCFS SFS/SCORE overlay')
        if 'remaining_length' in campaign:
            raise ValueError('The FCFS matrix overlay carries no remaining-length rule')
    else:
        if campaign.get('policies')!=list(BLOCKED):
            raise ValueError('FCFS SFS/SCORE overlay must set exactly the hard and score smoke policies')
        if any(c['status']!='runnable' for c in cells):
            raise ValueError('Every FCFS SFS/SCORE cell must be runnable')
    if mode not in INSPECT_MODES and campaign.get('full_matrix_authorized') is not True:
        raise ValueError('FCFS full matrix is not authorized; only calibrate/qualify may use this overlay')
    result=deepcopy(bundle);result['families']['qwen']=family(bundle);result['configuration_id']=CONFIG_ID
    if kind==SFS_SCORE_KIND:
        from scripts.cloud.sfs_score_campaign import accepted_prior_source_digests, resolve_remaining_length
        result['kind']=kind;result['families']['qwen']['policies']=list(BLOCKED)
        # Qwen-only families: off_models must be exactly the FCFS models without a rule (8B, 32B).
        result['remaining_length']=resolve_remaining_length(campaign.get('remaining_length'),{'qwen':result['families']['qwen']})
        result['accepted_prior_source_digests']=accepted_prior_source_digests(campaign)
    result['cells']=[{k:c[k] for k in KEYS} for c in cells if not c['status'].startswith('BLOCKED')]
    result['blocked_cells']=[c['id'] for c in cells if c['status'].startswith('BLOCKED')]
    result['requests_total']=sum(c['requests'] for c in result['cells'])
    if campaign.get('requests_total')!=sum(c['requests'] for c in cells):
        raise ValueError('requests_total disagrees with the cells')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path)
    p.add_argument('--kind',choices=['matrix','sfs_score'],default='matrix')
    a=p.parse_args()
    if a.kind=='sfs_score':output=a.output or SFS_SCORE_PATH;write(output,build_sfs_score(a.bundle))
    else:output=a.output or PATH;write(output,build(a.bundle))
    print(output)

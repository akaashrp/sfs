"""Reduced-grid overlays for the additional Qwen serving configurations (kind `serving_config`).

One overlay per profile (chunk8192, prefix_cache) with its own ledger identity: cells derive from the
single editable `policies` list times the 6/7/8/8.3 grid at 16,000 requests, ids
`<configuration_id>-<policy>-<qps>`; the overlay carries the same flag-gated 0.6B remaining-length
block as the canonical SFS/SCORE overlay and is validated by the helpers that overlay uses. Editing
`policies` changes the overlay hash, so every lane re-qualifies against the edited overlay. Run and
campaign modes refuse the overlay until `full_matrix_authorized` is true.
"""
import argparse
from copy import deepcopy
from pathlib import Path

from scripts.cloud.common import ROOT, read, write
from scripts.cloud.serving.profiles import RATES, METHODS, EVALUATION, RESTRICTIONS, family, profile as load_profile, for_configuration

KIND='serving_config'
POLICIES=('hard','mooncake_prefill','lmdeploy_proxy')
POLICY_RULE='SFS plus the two highest-utility external baselines on the canonical Vast grid; revisit when canonical SCORE lands'
AUTHORIZATION=('User authorized on 17 September 2026: reduced grids (SFS + two strongest baselines) for the prefix-caching and '
               'fine-chunk serving configurations')
INSPECT_MODES=('calibrate','qualify','inspect')   # modes that never launch a cell and so need no authorization
KEYS=('id','family','variant','policy','qps','requests')
REQUESTS=16000
MAX_RATE=12.0                 # above any rate this hardware sustains; a typo, not a grid
OVERLAYS={'chunk8192':ROOT/'scripts/cloud/fcfs/campaign-chunk8192-20260917.json',
          'prefix_cache':ROOT/'scripts/cloud/fcfs/campaign-prefix-cache-20260917.json',
          'kv_constrained':ROOT/'scripts/cloud/fcfs/campaign-kv-constrained-20260918.json'}
CANONICAL_SFS_SCORE=ROOT/'scripts/cloud/sfs-score-campaign-20260917.json'
QUALIFICATION={
    'refit':['Gates of the configuration state root under this source (test.sh, prepare cpu/serving); no FCFS admission audit (canonical context length)',
             'Bounded GPU smoke for 32B at TP=2 (scripts.cloud.serving.gpu_smoke --profile <profile> --indices 2): startup, memory, observed chunk-bound scheduling',
             'worker calibrate --profile <profile>: destination traces with placeholder coefficients',
             'scripts.cloud.serving.coefficients fit --profile <profile>: nonnegative SFS batch coefficients for this configuration (R^2 >= 0.95)',
             'worker qualify --profile <profile> --coefficients <fitted> --campaign <this overlay>: calibration, timing heads, one smoke per listed policy, load probes',
             'scripts.cloud.serving.coefficients validate on independent post-calibration batches; TPOT/timing-head review; control release with reviews',
             'qualification.json records configuration_id, coefficient_policy, coefficients_sha256, campaign_sha256, remaining_length_rule and the 0.6B table sha256'],
    'canonical':['Gates of the configuration state root under this source (test.sh, prepare cpu/serving); no FCFS admission audit (canonical context length)',
                 'worker qualify --profile <profile> --campaign <this overlay> (no --coefficients: the canonical SFS batch coefficients are retained): calibration, timing heads, one smoke per listed policy, load probes',
                 'Timing-head and destination-residual review on the canonical coefficients, prefix-cache hit evidence from usage.prompt_tokens_details.cached_tokens; control release with reviews',
                 'qualification.json records configuration_id, coefficient_policy canonical, coefficients_sha256 null, campaign_sha256, remaining_length_rule and the 0.6B table sha256']}
NOTES={
    'chunk8192':{'decision':'User decision 2026-09-17: fine-chunk serving configuration (max_num_batched_tokens 8192, chunked prefill on, long_prefill_token_threshold 0) '
                            'as a reduced grid of SFS (hard) plus the two strongest external baselines at 6/7/8/8.3 QPS x 16,000 requests, with the 0.6B '
                            'remaining-length rule. The 8192-token step changes the batch timing regime, so the SFS batch coefficients are refitted from '
                            'destination traces of this configuration and bound to every qualification and completed-ledger entry.'},
    'kv_constrained':{'decision':'User decision 2026-09-17: a fourth serving configuration combining KV cache size (gpu_memory_utilization 0.70), '
                                 'a concurrency cap (max_num_seqs 128) and a long-prefill threshold (2048 tokens), as a reduced grid of SFS (hard) plus '
                                 'the two strongest external baselines at 6/7/8/8.3 QPS x 16,000 requests, with the 0.6B remaining-length rule. '
                                 'Attribution between the three knobs is not the point: the configuration is one coherent constrained-capacity regime, '
                                 'chosen because all three are settings vLLM actually schedules on and the SFS simulator models.',
                      'estimators':'Every knob here is one the simulator already reads: max_num_seqs and long_prefill_token_threshold arrive in the '
                                   'published snapshot config and bound its scheduling loop, and the KV cache size arrives as free blocks, which is what '
                                   'drives its preemption path. Nothing about the estimator changes; what changes is that the engine now spends most of '
                                   'its time in the regime where those terms bind.',
                      'coefficients':'The 128-sequence cap truncates the decode-batch range the canonical fit was measured over, so the SFS batch '
                                     'coefficients are refitted from destination traces of this configuration and validated by the batch-residual audit '
                                     'before any cell runs.'},
    'prefix_cache':{'decision':'User decision 2026-09-17: prefix-caching serving configuration (vLLM automatic prefix caching on, canonical 32768-token step) as a '
                               'reduced grid of SFS (hard) plus the two strongest external baselines at 6/7/8/8.3 QPS x 16,000 requests, with the 0.6B '
                               'remaining-length rule. Per-token step costs are unchanged, so the canonical SFS batch coefficients are retained '
                               '(--coefficients is refused); qualification is a worker qualify pass plus review and release.',
                    'estimators':'The SFS batch simulator (vllm.v1.engine.scheduler_simulator: get_computed_blocks returns no cached blocks), the Mooncake '
                                 'prefill estimator and the RouteBalance TPOT head are cache-unaware by design: waiting requests are costed as full prompt '
                                 'prefills even when the engine will serve part of the prompt from the prefix cache; running requests are read from the '
                                 'snapshot after the hit (num_computed_tokens includes cached tokens). Measuring the routers under that estimator/engine '
                                 'mismatch is the point of this ablation; no estimator is changed.',
                    'observability':'--enable-prompt-tokens-details is added so every response usage carries prompt_tokens_details.cached_tokens; it has no scheduling effect.'}}


def cells(profile, policies, rates=RATES):
    return [{'id':f'{profile.configuration_id}-{p}-{q:g}','family':'qwen','variant':'canonical','policy':p,'qps':q,'requests':REQUESTS}
            for p in policies for q in rates]


def check_rates(campaign):
    """The overlay's request-rate grid: the default 6/7/8/8.3, or a grid that records what chose it.

    A configuration whose capacity sits below the default grid measures nothing there -- every policy
    is equally past the knee -- so a different grid is allowed, but only with the evidence for it
    written into the overlay, the way a tuned SCORE multiplier must cite its sweep.
    """
    rates=campaign.get('rates')
    # Types first: a string or a bool in the list must be refused, never compared.
    if (not isinstance(rates,list) or not rates
            or any(isinstance(r,bool) or not isinstance(r,(int,float)) or not 0<r<=MAX_RATE for r in rates)
            or len(set(rates))!=len(rates) or sorted(rates)!=list(rates)):
        raise ValueError('rates must be an ascending list of distinct positive request rates')
    if [float(r) for r in rates]!=[float(r) for r in RATES] and not str(campaign.get('rate_rule') or '').strip():
        raise ValueError('A grid other than the default must record the evidence that chose it in rate_rule')
    return [float(r) for r in rates]


def build(profile, bundle, rates=RATES, rate_rule=None, accepted_prior_campaigns=None, accepted_prior_sources=None):
    """The reduced-grid overlay for a profile, with the canonical overlay's remaining-length block (Qwen models only)."""
    if profile.name not in OVERLAYS:raise ValueError(f'No reduced-grid overlay is defined for the {profile.name} profile')
    b=read(bundle/'bundle.json');models={m:b['models'][m] for m in b['families']['qwen']['models']}
    canonical=read(CANONICAL_SFS_SCORE)['remaining_length'];off=[model for model in models if model not in canonical['rules']]
    return {'schema_version':1,'kind':KIND,'configuration_id':profile.configuration_id,'status':'AUTHORIZED_PENDING_QUALIFICATION',
        'full_matrix_authorized':True,'authorization':AUTHORIZATION,'decision':NOTES[profile.name]['decision'],
        'summary':profile.summary,'profile':dict(profile.settings),'changed_from_canonical':profile.changed,
        'server_argv_delta':{'boolean_swaps':[list(s) for s in profile.boolean_swaps],'options':[[f,v] for f,v in profile.options],
                             'extra_argv':list(profile.extra_argv),'instance_rows':dict(profile.rows)},
        'coefficient_policy':profile.coefficient_policy,
        'coefficients':('SFS batch coefficients refitted from destination traces of this configuration; the fitted file hash is bound to every qualification and ledger entry'
                        if profile.coefficient_policy=='refit' else 'Canonical SFS batch coefficients retained (per-token step costs unchanged); coefficients_sha256 is null'),
        'policies':list(POLICIES),'policy_rule':POLICY_RULE,
        'policy_edit_note':'Edit `policies` only: cells derive from policies x rates when the overlay is applied (no cells list is stored); any edit changes this '
                           'overlay hash, so every lane re-qualifies against the edited overlay before running',
        'rates':[float(r) for r in rates],**({'rate_rule':rate_rule} if rate_rule else {}),'requests_per_cell':REQUESTS,'cell_id_form':f'{profile.configuration_id}-<policy>-<qps>',
        'models':models,'evaluation':dict(EVALUATION),'restrictions':list(RESTRICTIONS),
        'gpu_allocation':{'lane':[0,1,2,3],'note':'One four-GPU lane (any free slot: 0-3 or 4-7); the qualification binds the exact hardware and the run must use it'},
        'ledger':f'Separate completion ledger under the configuration state root; cell IDs carry the {profile.configuration_id} prefix and never collide with canonical or FCFS cells',
        'qualification':[s.replace('<profile>',profile.name) for s in QUALIFICATION[profile.coefficient_policy]],
        'launch_rule':'worker run/campaign refuse this overlay until full_matrix_authorized is true; flipping it requires explicit user authorization',
        'remaining_length':{'tables':canonical['tables'],'files':dict(canonical['files']),'rules':dict(canonical['rules']),'off_models':off,
            'note':'Same rule and tables as the canonical SFS/SCORE overlay: only the qwen3-0.6b server receives --remaining-length-mode running_all '
                   '--remaining-length-table qwen3-0.6b.json --remaining-length-quantile 0.5 --remaining-length-conditioning prompt_bin, and the router '
                   'attaches that table for its missing-prediction fill on that instance only; qwen3-8b and qwen3-32b stay on the current rule. The '
                   'tables were built from canonical calibration outputs; the evidence is the canonical paired run, adopted by the user for every Qwen '
                   'SFS/SCORE cell including this serving configuration. No paired run exists under this configuration.',
            'evidence':deepcopy(canonical['evidence'])},
        'accepted_prior_source_digests':dict(accepted_prior_sources or {}),
        **({'accepted_prior_campaign_digests':dict(accepted_prior_campaigns)} if accepted_prior_campaigns else {}),
        **{k:v for k,v in NOTES[profile.name].items() if k!='decision'}}


def apply_campaign(profile, bundle, campaign, mode):
    if campaign.get('kind')!=KIND:raise ValueError('Not a serving-configuration overlay')
    if (campaign.get('configuration_id')!=profile.configuration_id or campaign.get('profile')!=profile.settings
            or campaign.get('coefficient_policy')!=profile.coefficient_policy):
        raise ValueError(f'Overlay is not the {profile.configuration_id} configuration')
    policies=campaign.get('policies')
    if not isinstance(policies,list) or not policies or len(set(policies))!=len(policies) or set(policies)-set(METHODS):
        raise ValueError('policies must be a non-empty list of distinct known router policies')
    rates=check_rates(campaign)
    if campaign.get('requests_per_cell')!=REQUESTS:
        raise ValueError('Serving-configuration overlays keep 16,000 requests per cell')
    if 'cells' in campaign:raise ValueError('Cells derive from policies; remove the explicit cells list')
    if mode not in INSPECT_MODES:
        if campaign.get('full_matrix_authorized') is not True:
            raise ValueError(f'{profile.configuration_id} grid is not authorized; only calibrate/qualify may use this overlay')
        if not isinstance(campaign.get('authorization'),str) or not campaign['authorization'].strip():
            raise ValueError('An authorized overlay must record the authorization text')
    from scripts.cloud.sfs_score_campaign import (accepted_prior_campaign_digests, accepted_prior_source_digests,
                                                   resolve_remaining_length)
    result=deepcopy(bundle);result['families']['qwen']=family(profile,bundle,policies,rates)
    result['kind']=KIND;result['configuration_id']=profile.configuration_id;result['coefficient_policy']=profile.coefficient_policy
    # Qwen-only family: off_models must be exactly the Qwen models without a rule (8B, 32B).
    result['remaining_length']=resolve_remaining_length(campaign.get('remaining_length'),{'qwen':result['families']['qwen']})
    result['accepted_prior_source_digests']=accepted_prior_source_digests(campaign)
    # Adding rates to the grid leaves every cell already measured exactly as it was: its overlay digest is
    # accepted by name, with the reason, instead of burning GPU hours reproducing identical numbers.
    result['accepted_prior_campaign_digests']=accepted_prior_campaign_digests(campaign)
    result['cells']=cells(profile,policies,rates);result['blocked_cells']=[]
    result['requests_total']=sum(c['requests'] for c in result['cells'])
    return result


def apply_profile_campaign(name, bundle, campaign, mode):
    """Validate an overlay for the worker --profile it runs under (fcfs keeps its own validator)."""
    if name=='fcfs':
        from scripts.cloud.fcfs.campaign import apply_campaign as apply_fcfs
        return apply_fcfs(bundle,campaign,mode)
    return apply_campaign(load_profile(name),bundle,campaign,mode)


def apply_configuration_campaign(bundle, campaign, mode='inspect'):
    """Dispatch by the overlay's configuration_id (read-only inspection by control status and collate)."""
    return apply_profile_campaign(for_configuration(campaign.get('configuration_id')).name,bundle,campaign,mode)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path)
    p.add_argument('--profile',choices=sorted(OVERLAYS),required=True)
    p.add_argument('--rates',help='Comma-separated request rates in place of the default grid (requires --rate-rule)')
    p.add_argument('--rate-rule',help='The evidence that chose a non-default grid')
    p.add_argument('--accept-prior',action='append',default=[],metavar='SHA256=REASON',
                   help='An earlier overlay digest whose completed cells stay valid under this grid')
    p.add_argument('--accept-prior-source',action='append',default=[],metavar='SHA256=REASON',
                   help='An earlier source digest whose completed cells stay valid under this grid')
    a=p.parse_args();output=a.output or OVERLAYS[a.profile]
    rates=tuple(float(r) for r in a.rates.split(',')) if a.rates else RATES
    prior=dict(entry.split('=',1) for entry in a.accept_prior)
    sources=dict(entry.split('=',1) for entry in a.accept_prior_source)
    write(output,build(load_profile(a.profile),a.bundle,rates,a.rate_rule,prior,sources));print(output)

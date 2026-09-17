"""Table of Qwen serving configurations selectable with worker --profile.

Every profile keeps the canonical Qwen workload, models, kernels, precision, chat template, YaRN/RoPE,
TP 1/1/2, predictors and SLOs; only the scheduler/cache settings named in `settings` differ from
`scripts.runs.qwen_baselines.PROFILE`. A profile carries its argv delta (boolean flag swaps and
scalar options applied over the canonical server argv, remaining-length rule included), the
instances.json row overrides the router reads, the coefficient policy (`refit`: SFS batch
coefficients are fitted from destination traces of this configuration and must be passed with
--coefficients; `canonical`: the canonical coefficients are retained and --coefficients is refused),
whether the FCFS actual-chat admission audit gate applies, the scheduler config every published
snapshot must report, and the bounded GPU smoke evidence rule.
"""
from copy import deepcopy
from dataclasses import dataclass, field

from scripts.cloud.common import read, set_option
from scripts.cloud.pool import config as canonical_config, server_argv as canonical_server

RATES = (6., 7., 8., 8.3)
METHODS = ('hard', 'round_robin', 'latency_agnostic', 'shortest_queue', 'score', 'lmdeploy_proxy', 'mooncake_prefill',
           'routebalance', 'vllm_sr_latency')
COEFFICIENT_POLICIES = ('refit', 'canonical')
EVALUATION = {'arrival_process': 'poisson', 'seed': 69, 'slo_rule': 'Frozen canonical Qwen SLOs without refitting',
              'primary_judge': 'pro', 'quality_scores': 'Frozen canonical scores; common 15996 observed queries for Pro/Flash comparisons',
              'max_completion_tokens': 8192, 'temperature': 0, 'top_p': 1, 'enable_thinking': False}
RESTRICTIONS = ['No full-matrix launch without subsequent user authorization', 'Do not deploy an unvalidated output-length change',
                'Preserve canonical kernels, precision, chat template, YaRN/RoPE and TP1/1/2']
CANONICAL = {'scheduling_policy': 'fcfs', 'chunked_prefill': True, 'max_num_batched_tokens': 32768, 'max_model_len': 131072,
             'max_num_seqs': 512, 'long_prefill_token_threshold': 0, 'gpu_memory_utilization': .90, 'prefix_caching': False}


@dataclass(frozen=True)
class Profile:
    name: str
    configuration_id: str
    settings: dict
    coefficient_policy: str
    summary: str
    boolean_swaps: tuple = ()      # (canonical flag, replacement flag) pairs swapped in place
    options: tuple = ()            # (flag, value) pairs applied with set_option
    rows: dict = field(default_factory=dict)   # instances.json per-row overrides
    extra_argv: tuple = ()         # observability flags appended verbatim (no scheduling effect)
    admission_audit: bool = False  # worker requires setup/fcfs-inputs.json (actual chat admission audit)
    bounded_smoke: str = None      # gpu_smoke evidence rule: 'full_prefill', 'chunk_bound' or None

    @property
    def snapshot_config(self):
        """Scheduler config every published snapshot of this configuration must report."""
        s = self.settings
        return {'chunked_prefill_enabled': s['chunked_prefill'], 'max_num_batched_tokens': s['max_num_batched_tokens'],
                'max_num_seqs': s['max_num_seqs'], 'max_model_len': s['max_model_len'],
                'long_prefill_token_threshold': s['long_prefill_token_threshold'], 'policy': s['scheduling_policy']}

    @property
    def changed(self):
        return {k: v for k, v in self.settings.items() if CANONICAL[k] != v}


FCFS = Profile('fcfs', 'qwen-fcfs-unchunked-65536',
    {'scheduling_policy': 'fcfs', 'chunked_prefill': False, 'max_num_batched_tokens': 65536, 'max_model_len': 65536, 'max_num_seqs': 512,
     'long_prefill_token_threshold': 0, 'gpu_memory_utilization': .90, 'prefix_caching': False},
    'refit', 'FCFS without chunked prefill: whole prompts per step, 65536-token step budget and context',
    boolean_swaps=(('--enable-chunked-prefill', '--no-enable-chunked-prefill'),),
    options=(('--scheduling-policy', 'fcfs'), ('--max-num-batched-tokens', 65536), ('--max-model-len', 65536), ('--max-num-seqs', 512),
             ('--long-prefill-token-threshold', 0), ('--gpu-memory-utilization', .90)),
    rows={'max_num_batched_tokens': 65536, 'max_num_seqs': 512, 'chunked_prefill_enabled': False, 'long_prefill_token_threshold': 0,
          'max_model_len': 65536, 'scheduling_policy': 'fcfs'},
    admission_audit=True, bounded_smoke='full_prefill')

CHUNK8192 = Profile('chunk8192', 'qwen-chunk8192', dict(CANONICAL, max_num_batched_tokens=8192),
    'refit', 'Canonical chunked prefill with a 8192-token step budget (long_prefill_token_threshold stays 0, so prompts are chunked at 8192)',
    options=(('--max-num-batched-tokens', 8192),), rows={'max_num_batched_tokens': 8192}, bounded_smoke='chunk_bound')

PREFIX_CACHE = Profile('prefix_cache', 'qwen-prefix-cache', dict(CANONICAL, prefix_caching=True),
    'canonical', 'Canonical scheduler with vLLM automatic prefix caching enabled; per-token step costs unchanged, canonical SFS coefficients retained',
    boolean_swaps=(('--no-enable-prefix-caching', '--enable-prefix-caching'),), rows={'prefix_caching_enabled': True},
    extra_argv=('--enable-prompt-tokens-details',))

PROFILES = {p.name: p for p in (FCFS, CHUNK8192, PREFIX_CACHE)}
NAMES = ('canonical', *PROFILES)


def profile(name):
    if name not in PROFILES:
        raise ValueError(f'Unknown serving profile: {name!r}')
    return PROFILES[name]


def for_configuration(configuration_id):
    found = [p for p in PROFILES.values() if p.configuration_id == configuration_id]
    if not found:
        raise ValueError(f'Unknown serving configuration: {configuration_id!r}')
    return found[0]


def family(profile, manifest, policies=None):
    """The bundle's Qwen family with this configuration's profile, rates, policies and identity."""
    d = deepcopy(manifest['families']['qwen'])
    d['profile'].update(profile.settings)
    d['qps'] = list(RATES); d['policies'] = list(policies if policies is not None else METHODS); d['configuration_id'] = profile.configuration_id
    return d


def coefficient_status(profile, coefficients):
    if profile.coefficient_policy == 'canonical':
        if coefficients is not None:
            raise ValueError(f'The {profile.name} profile retains the canonical SFS coefficients; no fitted coefficients are accepted')
        return 'CANONICAL_RETAINED_BY_CONFIGURATION_POLICY'
    return 'FITTED_FOR_CONFIGURATION' if coefficients else 'CANONICAL_PLACEHOLDER_FOR_TRACE_COLLECTION_ONLY'


def instances(profile, bundle, ports, tag, coefficients=None):
    """Canonical Qwen instances.json with this configuration's row overrides, coefficients and identity."""
    d = family(profile, read(bundle/'bundle.json'))
    cfg = canonical_config('qwen', d, {}, ports, tag)
    cfg['serving_profile'] = d['profile']
    status = coefficient_status(profile, coefficients)
    for row in cfg['instances']:
        row.update(deepcopy(profile.rows))
        if coefficients is not None:
            row['ttft_batch_model'] = coefficients[row['model_id']]
    cfg['configuration_id'] = profile.configuration_id
    cfg['coefficient_policy'] = profile.coefficient_policy
    cfg['coefficient_status'] = status
    return cfg


def server_argv(profile, bundle, model_path, row, index, output, remaining_length=None):
    """Canonical Qwen argv (including any per-engine remaining-length rule) with only this configuration's delta applied."""
    argv = canonical_server('qwen', model_path, row, index, output, bundle/'qwen/length', remaining_length)
    for old, new in profile.boolean_swaps:
        argv[argv.index(old)] = new
    for flag, value in profile.options:
        argv = set_option(argv, flag, value)
    return [*argv, *profile.extra_argv]

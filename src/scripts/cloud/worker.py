"""Qualify and execute auditable cells on a locally owned cloud GPU pool."""
import argparse
import asyncio
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time
import uuid

from scripts.cloud.common import ROOT, digest, read, write, expand, set_option, source_hashes, source_digest, validate_bundle, locks, portable_cache, portable_routebalance
from scripts.cloud.batch_residual import audit_pool
from scripts.cloud.pool import pool, hardware, remaining_length_provenance


def arguments(definition, bundle, variant, manifest, qualification=None):
    argv = list(definition['experiment_argv'])
    if variant != 'canonical':
        flag = '--output-length-model-path' if variant == 'mlp_length' else '--accuracy-model-path'
        argv = set_option(argv, flag, manifest['variants'][variant])
    argv = expand(argv, bundle, {})
    family = 'qwen' if definition['models'][0].startswith('qwen') else 'ministral'
    cache = portable_cache(bundle, family)
    for flag in ('--bucket-dir', '--holdout-cache-dir', '--holdout-bucket-dir'):
        argv = set_option(argv, flag, cache)
    argv = set_option(argv, '--routebalance-predictor-path', portable_routebalance(bundle, family))
    if qualification:
        argv = set_option(argv, '--service-metrics-json', Path(qualification)/'model_metrics.json')
        argv = set_option(argv, '--methodology-calibration-json', Path(qualification)/'timing_models/methodology_calibration.json')
    else:
        # CPU ingestion does not load runtime timing heads.
        while '--methodology-calibration-json' in argv:
            i = argv.index('--methodology-calibration-json'); del argv[i:i+2]
    return argv


def parse(argv):
    from scripts.runs.experiments_sweep import _parse_experiment_args
    return _parse_experiment_args(argv)


def policies_for(manifest, family, variant, definition):
    """Policies this pool will run, and therefore must smoke: those of the manifest's cells for the family/variant.

    Order follows the family definition; with the frozen bundle this yields the canonical lists and
    ['hard'] for the frozen predictor variants, with an overlay the campaign's own policies.
    """
    found = {c['policy'] for c in manifest['cells'] if c['family'] == family and c['variant'] == variant}
    if not found:
        return list(definition['policies']) if variant == 'canonical' else ['hard']
    return [p for p in definition['policies'] if p in found] + sorted(found - set(definition['policies']))


def staleness_levels(manifest):
    """Sorted distinct router snapshot delays (ms) of the manifest cells; [0.0] for overlays without injection."""
    return sorted({cell_staleness(c) for c in manifest['cells']}) or [0.0]


def cell_staleness(cell):
    return float(cell.get('snapshot_staleness_ms', 0) or 0)


def smoke_staleness(manifest):
    """Delays every new pool smokes: none under other overlays, D = 0 and the largest D under the sweep."""
    if manifest.get('kind') != 'staleness_sweep':
        return [None]
    return sorted({0.0, max(staleness_levels(manifest))})


def staleness_argv(argv, manifest, staleness):
    """Router argv for one run: the staleness flag reaches the router only under a staleness_sweep overlay."""
    delay = float(staleness or 0)
    if manifest.get('kind') != 'staleness_sweep':
        if delay:
            raise ValueError('Snapshot staleness is only authorized under a staleness_sweep overlay')
        return list(argv)
    return set_option(argv, '--snapshot-staleness-ms', f'{delay:g}')


PROBE_KINDS = ('score_lambda_sweep',)


def campaign_data_role(manifest):
    """'tuning_probe' for an overlay whose points may never be reported; 'evaluation' otherwise."""
    return 'tuning_probe' if manifest.get('kind') in PROBE_KINDS else 'evaluation'


def ledger_dir(state, manifest):
    """Completed-cell receipts of this overlay: probe receipts stay out of the canonical completed ledger."""
    return Path(state)/('completed-probes' if campaign_data_role(manifest) == 'tuning_probe' else 'completed')


def cell_lambda(cell):
    """The cell's SCORE Lagrange multiplier, or None when the cell keeps the bundle's own --lambda-weight."""
    weight = cell.get('lambda_weight')
    return None if weight is None else float(weight)


def routing_lambda(manifest, cell):
    """The SCORE routing multiplier for one cell: per-cell under a sweep, else the overlay's tuned value.

    SCORE's lambda is the multiplier of its own constraint formulation and is a tuning knob of that
    method; the campaign's --lambda-weight is the cost weight of the objective every method is scored
    on.  Only the former moves here, so a tuned SCORE still competes on the same OnTimeUtility as the
    policies that carry no multiplier at all.
    """
    weight = cell_lambda(cell)
    if weight is None and str(cell.get('policy')) == 'score':
        overlay = manifest.get('score_lambda_weight')
        weight = None if overlay is None else float(overlay)
    return weight


def lambda_levels(manifest):
    """Sorted distinct per-cell lambda weights of the manifest cells; [] when no cell overrides the bundle."""
    return sorted({w for w in (cell_lambda(c) for c in manifest['cells']) if w is not None})


def lambda_argv(argv, manifest, weight):
    """Router argv for one run: a SCORE routing multiplier, which never touches the evaluation lambda.

    A per-cell weight is authorized only under a score_lambda_sweep overlay; any other overlay carries
    at most one tuned value for all of its SCORE cells.  Either way it is passed as
    --score-lambda-weight, so --lambda-weight stays at the bundle's value and every cell of the
    campaign, SCORE included, is scored on one objective.
    """
    if weight is None:
        return list(argv)
    if manifest.get('kind') != 'score_lambda_sweep' and manifest.get('score_lambda_weight') is None:
        raise ValueError('A SCORE routing multiplier requires a sweep overlay or an overlay-wide tuned value')
    weight = float(weight)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('A SCORE routing multiplier must be finite and non-negative')
    return set_option(argv, '--score-lambda-weight', f'{weight:.12g}')


def cell_delta(cell):
    """The cell's latency penalty delta, or None when the run keeps the bundle's own --delta-weight."""
    weight = cell.get('delta_weight')
    return None if weight is None else float(weight)


def delta_argv(argv, manifest, weight):
    """Router argv for one run: the latency penalty of the soft objective (accuracy - lambda*cost - delta*wait).

    Only a delta_sweep overlay varies it per cell; every other overlay leaves the bundle's value alone,
    so the utility-latency tradeoff curve is the one experiment that moves this knob.
    """
    if weight is None:
        return list(argv)
    if manifest.get('kind') != 'delta_sweep':
        raise ValueError('A per-cell delta weight requires a delta_sweep overlay')
    weight = float(weight)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('A latency penalty must be finite and non-negative')
    return set_option(argv, '--delta-weight', f'{weight:.12g}')


def delta_levels(manifest):
    """Sorted distinct per-cell delta weights of the manifest cells; [] when no cell overrides the bundle."""
    return sorted({w for w in (cell_delta(c) for c in manifest['cells']) if w is not None})


def family_remaining_length(manifest, definition):
    """Pool-ready {'tables', 'rules'} for this family's models, or None (current rule everywhere)."""
    block = manifest.get('remaining_length')
    if not block:
        return None
    rules = {model: rule for model, rule in block['rules'].items() if model in definition['models']}
    return {'tables': block['tables'], 'rules': rules} if rules else None


def remaining_length_record(block):
    """Ledger/qualification summary of a pool's remaining-length provenance (the instances.json block)."""
    return {'rule': block['rule'], 'models': {model: {'rule': rule['rule'], 'table_sha256': (rule.get('table') or {}).get('sha256')}
                                              for model, rule in block['models'].items()}}


def completed_source_accepted(previous, source, manifest):
    """A completed cell is reusable under the current source or an explicitly accepted prior pin."""
    prior = previous['source_sha256']
    return prior == source or source_digest(prior) in manifest.get('accepted_prior_source_digests', {})


def audit_cell(payload, cell):
    from scripts.runs.ministral3_methodology_stage import audit_run
    from scripts.runs.measured_audit import require_complete_ttft
    if len(payload['router']['runs']) != 1:
        raise ValueError('Expected exactly one policy per checkpoint')
    run = payload['router']['runs'][0]
    audit_run(run, cell['requests'])
    if run['utility'] != cell['policy'] or payload['config']['request_rate_qps'] != cell['qps']:
        raise ValueError('Cell identity mismatch')
    rows = run['per_request']
    if {r['request_id'] for r in rows} != {f'req-{i}' for i in range(cell['requests'])}:
        raise ValueError('Missing or duplicate evaluation request identities')
    summary = run['summary']
    # A failed request stops this pool, whatever its cause. The bounded infrastructure-fault
    # salvage of scripts.cloud.salvage is an explicit, after-the-fact, user-authorised step and is
    # deliberately not consulted here; salvage_rule/salvage_classification below only expose it.
    if summary.get('failed_requests') != 0 or summary.get('succeeded_requests') != len(rows):
        raise ValueError('Incomplete requests or end-to-end TTFT')
    # Transient snapshot-read faults degrade one candidate; a decision that lost
    # every candidate is the one case that still fails a request. Older points
    # predate the counters and are audited by the checks above alone.
    faults = summary.get('snapshot_read_faults')
    if isinstance(faults, dict) and (faults.get('totals') or {}).get('requests_without_estimate'):
        raise ValueError('Routing decisions lost every candidate to snapshot read faults')
    require_complete_ttft(run)
    arrivals = [r.get('system_entry_offset_s') for r in rows]
    if any(not isinstance(t, (float, int)) or not math.isfinite(t) or t < 0 for t in arrivals):
        raise ValueError('Missing arrival telemetry')
    realized = (len(rows)-1)/(max(arrivals)-min(arrivals))
    if abs(realized/cell['qps'] - 1) > .1:
        raise ValueError('Arrival generator missed the requested rate by more than 10%')
    return {'status': 'PASS_CELL', 'requests': len(rows), 'realized_qps': realized}


def salvage_rule():
    """Read-only view of the infrastructure-fault salvage allowlist, cap and penalty.

    Exposed next to audit_cell so reviewers and tools import one bound rule; audit_cell itself
    never calls it and the worker's live behaviour is unchanged.
    """
    from scripts.cloud import salvage
    return {'causes': {name: list(needles) for name, needles in salvage.SALVAGEABLE_CAUSES.items()},
            'max_requests': salvage.MAX_SALVAGEABLE_REQUESTS,
            'max_fraction_pct': salvage.MAX_SALVAGEABLE_FRACTION * 100,
            'penalty': salvage.PENALTY, 'authorization': salvage.AUTHORIZATION}


def salvage_classification(payload, cell):
    """Read-only report of a completed point's failed rows under that rule. Never admits a cell."""
    from scripts.cloud.salvage import classify_failures
    return classify_failures(payload, cell)


async def run_point(family, args, requests, clients, costs, metadata, folder, monitor=None, *, data_role='evaluation'):
    from scripts.runs import experiments as exp
    from scripts.runs.ministral3_methodology_stage import wait_drained
    folder.mkdir(parents=True, exist_ok=False)
    await wait_drained(clients, timeout_s=120)
    if family == 'ministral' and monitor is None:
        from scripts.runs.capacity_scout import TrialMonitor
        # Stop new arrivals after a terminal routing failure, retain the partial
        # point, and let its completeness audit stop the job. Do not cap a
        # successful full evaluation's outstanding request budget.
        monitor = TrialMonitor(folder/'events.jsonl', duration_s=10800,
                               max_outstanding=len(requests)+1)
    function = exp.run_router_experiment
    if family == 'ministral' and args.utilities == ['vllm_sr_latency']:
        from scripts.runs.ministral3_latency import run_selector
        function = run_selector
    elif family == 'ministral':
        from scripts.runs.ministral3_reliable import run_router_experiment
        function = run_router_experiment
    result = await asyncio.wait_for(function(args=args, requests=requests, instances=clients,
        instance_costs=costs, instance_metadata=metadata, response_map_base_path=folder/'responses.log',
        request_log_base_path=folder/'predicted_waits.log', trial_monitor=monitor), timeout=10800)
    await wait_drained(clients, timeout_s=120)
    config = {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
    config['instance_metadata'] = metadata
    config['prompt_source'] = {'data_role': data_role}
    if data_role == 'evaluation':
        # The existing quality join resolves its request map through this
        # nested field, as written by the original experiment sweep producer.
        config['prompt_source'].update({name: getattr(args, name, None) for name in
            ('holdout_prompts_per_bucket', 'holdout_start_index', 'holdout_cache_dir', 'tokenizer_id')})
    payload = {'config': config, 'request_set': {'num_requests': len(requests)}, 'router': result}
    # Some argparse metadata contains Paths; preserve them as strings.
    payload = json.loads(json.dumps(payload, default=str))
    write(folder/'point.json', payload)
    return payload


LOADED_MAX_COMPLETION_TOKENS = 8192


def capped_output_summary(responses, model):
    """Count loaded-phase calibration outputs of one model that hit the token cap."""
    loaded = [r['usage'].get('completion_tokens') or 0 for r in responses
              if r['model'] == model and r['probe_id'].startswith(f'loaded-{model}-')]
    return {'loaded_outputs': len(loaded), 'loaded_max_completion_tokens': LOADED_MAX_COMPLETION_TOKENS,
            'capped_loaded_outputs': sum(tokens >= LOADED_MAX_COMPLETION_TOKENS for tokens in loaded)}


def calibration_capped_outputs(metrics):
    """Reviewer summary of token-capped calibration outputs and the SCORE decode window."""
    keys = ('loaded_outputs', 'capped_loaded_outputs', 'loaded_max_completion_tokens',
            'decode_window_rule', 'decode_rows_excluded_by_window', 'decode_batch_stats_rows_used')
    return {model: {key: row['score_proxy'].get(key) for key in keys} for model, row in metrics.items()}


async def calibrate(family, definition, requests, clients, base_args, output):
    """Warm shapes first, measure singleton prefill and loaded service/decode next."""
    from scripts.runs.ministral3_methodology_stage import length_stratified_requests, smoke_requests, wait_drained
    from scripts.prep.fit_methodology_calibration import fit_manifest, load_trace_rows
    from sfs_core.shared.shared_experiment_helpers import build_messages
    from sfs_core.shared.trace_theta import (SCORE_PROXY_MULTI_SEQUENCE_PURE_DECODE,
        estimate_score_proxy_metrics_from_batch_stats)
    probes, sample = length_stratified_requests(requests), smoke_requests(requests, per_bucket=128)
    recorded, services = [], {}

    async def measure(client):
        async def submit(req, rid, tokens):
            reply = await client.submit_request(messages=build_messages(req.prompt, base_args.system_prompt),
                temperature=0, top_p=1, max_completion_tokens=tokens,
                extra_body={'chat_template_kwargs': base_args.chat_template_kwargs, 'request_id': rid})
            if reply.usage is None or not reply.id:
                raise ValueError('Missing measured calibration response/usage')
            recorded.append({'model': client.model_id, 'request_id': req.request_id,
                             'probe_id': rid, 'response_id': reply.id, 'usage': reply.usage.model_dump()})
        for i, req in enumerate(probes):
            await submit(req, f'warm-shape-{client.model_id}-{i}', 1)
            await wait_drained({client.instance_id: client}, timeout_s=120)
        trace = output/f'batch_stats_{client.model_id}.csv'
        await asyncio.sleep(2)
        with trace.open() as stream:
            header = stream.readline(); skip = sum(1 for _ in stream)
        for i, req in enumerate(probes):
            await submit(req, f'prefill-probe-{client.model_id}-{i}', 1)
            await wait_drained({client.instance_id: client}, timeout_s=120)
        semaphore = asyncio.Semaphore(128)
        async def loaded(i, req):
            async with semaphore:
                await submit(req, f'loaded-{client.model_id}-{i}', LOADED_MAX_COMPLETION_TOKENS)
        start = time.monotonic()
        await asyncio.gather(*(loaded(i, r) for i, r in enumerate(sample)))
        elapsed = time.monotonic()-start
        await wait_drained({client.instance_id: client}, timeout_s=120)
        await asyncio.sleep(2)
        frozen = output/f'calibration_trace_{client.model_id}.csv'
        with trace.open() as source, frozen.open('x') as dest:
            next(source)
            for _ in range(skip): next(source)
            dest.write(header)
            shutil.copyfileobj(source, dest)
        rows, _ = load_trace_rows([frozen])
        positive = [r for r in rows if r['prefill'] > 0]
        # A single token-capped output draining alone is not loaded decode; exclude
        # single-sequence iterations (scripts/cloud/reports/score-proxy-window-20260917).
        proxy = estimate_score_proxy_metrics_from_batch_stats(batch_stats_csv_path=frozen, batch_stats_offset=0,
            decode_window=SCORE_PROXY_MULTI_SEQUENCE_PURE_DECODE)
        proxy['prefill_tps'] = sum(r['prefill'] for r in positive)/sum(r['exec'] for r in positive)
        proxy.update(capped_output_summary(recorded, client.model_id))
        services[client.model_id] = {'service_rate_qps': len(sample)/elapsed, 'num_queries': len(sample),
            'succeeded': len(sample), 'failed': 0, 'elapsed_s': elapsed, 'score_proxy': proxy,
            'service_rate_definition': '512 calibration requests at concurrency 128 / whole-run elapsed; not router capacity',
            'traces': [str(frozen)]}
    await asyncio.gather(*(measure(client) for client in clients.values()))
    write(output/'calibration_responses.json', {'data_role': 'calibration', 'responses': recorded})
    write(output/'model_metrics.json', services)
    write(output/'service_manifest.json', {'data_role': 'calibration', 'serving_profile_verified': True,
        'serving_profile': definition['profile'], 'models': services})
    await asyncio.to_thread(fit_manifest, output/'service_manifest.json', output/'timing_models')


async def execute(options, manifest, definition, model_paths, output):
    from scripts.runs import experiments as exp
    from scripts.runs.ministral3_methodology_stage import smoke_requests, audit_run, wait_drained
    from sfs_core.shared.shared_experiment_helpers import warm_up_instances
    gpus = options.gpus.split(',')
    base = arguments(definition, options.bundle, options.variant, manifest)
    base_args = parse(base)
    source = source_hashes()
    tests_dir = Path(options.state).parent/'setup/tests'
    tests = read(tests_dir/'gate.json')
    if (tests.get('status') != 'PASS_CPU_REGRESSION' or tests.get('source_sha256') != source
            or tests.get('xml_sha256') != digest(tests_dir/'results.xml')):
        raise ValueError('Missing/stale source-bound CPU regression gate; rerun scripts/cloud/test.sh')
    cpu = read(Path(options.state).parent/'setup/cpu-inputs.json')
    if (cpu.get('status') != 'PASS_CPU_INPUTS' or cpu.get('source_sha256') != source
            or cpu.get('bundle_sha256') != digest(Path(options.bundle)/'bundle.json')):
        raise ValueError('Missing/stale destination CPU input verification; rerun prepare cpu')
    serving = read(Path(options.state).parent/'setup/cpu-serving.json')
    if (serving.get('status') != 'PASS_CPU_SERVING' or serving.get('source_sha256') != source
            or serving.get('bundle_sha256') != digest(Path(options.bundle)/'bundle.json')):
        raise ValueError('Missing/stale CPU chat, prediction and scheduler gate; rerun prepare serving')
    calibration = [exp.ExperimentRequest(**json.loads(line)) for line in
        Path(expand(definition['calibration_requests'], options.bundle, {})).read_text().splitlines()]
    length = base[base.index('--output-length-model-path')+1]
    for model in definition['models']:
        path = Path(model_paths[model])
        if path.name != manifest['models'][model]['revision']:
            raise ValueError(f'Unpinned model snapshot: {model}')
        for name in ('config.json','params.json','tokenizer_config.json','tekken.json'):
            frozen = Path(options.bundle)/'tokenizers'/model/name
            if frozen.exists() and digest(path/name) != digest(frozen):
                raise ValueError(f'Model/tokenizer configuration differs from the frozen checkpoint: {model}/{name}')
    qualification = Path(options.qualification).resolve() if options.qualification else output
    # Per-engine remaining-length rule (SFS/SCORE and predictor-variant overlays): tables and rules
    # come from the validated campaign; the provenance below is what the pool writes to instances.json.
    remaining_length = family_remaining_length(manifest, definition)
    expected_rule = remaining_length_provenance(definition, remaining_length)
    if options.mode == 'run':
        validate_release(qualification, options, source, expected_rule['rule'])
    with pool(options.family, definition, model_paths, options.bundle, output, gpus, options.state, length,
              remaining_length) as (instances_path, machine, processes):
        clients, costs, metadata = exp.load_instances(instances_path)
        active_rule = metadata.get('remaining_length') or {'rule': 'current', 'models': {}}
        if active_rule != expected_rule:
            raise ValueError('Pool remaining-length provenance differs from the campaign manifest')
        async def heartbeat():
            while True:
                if any(p.poll() is not None for p in processes):
                    raise RuntimeError('A serving process exited during the workload')
                if shutil.disk_usage(output).free < 20*1024**3:
                    raise RuntimeError('Less than 20 GiB free; stopped to preserve artifacts')
                write(output/'heartbeat.json', {'time': time.time(), 'pid': os.getpid(), 'state': 'RUNNING'})
                await asyncio.sleep(15)
        async def workload():
            await warm_up_instances(list(clients.values()))
            await wait_drained(clients, timeout_s=120)
            if options.mode in ('qualify', 'campaign'):
                await calibrate(options.family, definition, calibration, clients, base_args, output)
            argv = arguments(definition, options.bundle, options.variant, manifest, qualification)
            def args_for(policy, rate, count, staleness=None, lambda_weight=None, delta_weight=None):
                args = parse(delta_argv(lambda_argv(staleness_argv(argv, manifest, staleness), manifest, lambda_weight),
                                        manifest, delta_weight))
                args.utilities, args.num_requests, args.request_rate_qps = [policy], count, rate
                args.per_request_wait_log = [str(output/f'wait_{m}.log') for m in definition['models']]
                return args
            policies = policies_for(manifest, options.family, options.variant, definition)
            # Every new pool passes matching smoke, including resumption on the same host.
            smoke = smoke_requests(calibration)
            for policy in policies:
                smoke_rate = definition['qps'][-1] if options.family == 'ministral' else 2.
                for level in smoke_staleness(manifest):
                    folder = output/'smoke'/(policy if level is None else f'{policy}-stale{level:g}')
                    payload = await run_point(options.family, args_for(policy, smoke_rate, 192, level), smoke, clients, costs,
                                              metadata, folder, data_role='calibration')
                    audit_run(payload['router']['runs'][0], 192)
                    from scripts.runs.measured_audit import require_complete_ttft
                    require_complete_ttft(payload['router']['runs'][0])
            # Smoke has now driven enough batches through every engine to replay that engine's own
            # batch-latency coefficients against what it actually did.  A coefficient consumed under the
            # wrong feature set still runs, still audits clean, and makes every wait estimate meaningless
            # (scripts/cloud/reports/ministral-sfs-gap-20260918), so no pool evaluates before this agrees.
            write(output/'batch_residual_audit.json',
                  audit_pool(read(instances_path)['instances'], output))
            if options.family == 'ministral' and options.variant == 'canonical':
                # The former 192-request/2-QPS smoke missed long busy iterations.
                # Exercise both snapshot baselines with 512 balanced calibration
                # prompts at the actual maximum offered load before evaluation.
                stress = smoke_requests(calibration, per_bucket=128)
                for policy in ('mooncake_prefill', 'routebalance'):
                    if policy not in policies:
                        continue
                    payload = await run_point('ministral', args_for(policy, definition['qps'][-1], len(stress)),
                        stress, clients, costs, metadata, output/'snapshot_stress'/policy, data_role='calibration')
                    audit_run(payload['router']['runs'][0], len(stress))
                    require_complete_ttft(payload['router']['runs'][0])
            if options.mode in ('qualify', 'campaign'):
                from scripts.runs.capacity_scout import TrialMonitor, classify_trial
                load_probes = []
                # Campaign qualification uses the policy smoke and 512-request
                # stress at its top load. No extra capacity search is needed
                # for the user-specified non-SFS grid.
                for rate in (() if options.mode == 'campaign' else (definition['qps'][0], definition['qps'][-1])):
                    folder = output/'load_probes'/f'{rate:g}'
                    monitor = TrialMonitor(folder/'events.jsonl', duration_s=360, max_outstanding=1024)
                    payload = await run_point(options.family, args_for('shortest_queue', rate, len(calibration)), calibration,
                        clients, costs, metadata, folder, monitor, data_role='calibration')
                    audit_run(payload['router']['runs'][0])
                    probe = classify_trial(monitor.events, requested_qps=rate)
                    load_probes.append(probe)
                evidence = {str(p.relative_to(output)): digest(p) for p in output.rglob('*')
                            if p.is_file() and (p.suffix in ('.json', '.jsonl') or p.name.startswith('calibration_trace_'))
                            and p.name not in ('heartbeat.json', 'status.json')}
                write(output/'qualification.json', {'status': 'GPU_MEASURED_REVIEW_REQUIRED',
                    'family': options.family, 'variant': options.variant, 'hardware': machine,
                    'source_sha256': source, 'bundle_sha256': digest(Path(options.bundle)/'bundle.json'),
                    'load_probes': load_probes, 'files': evidence,
                    'campaign_sha256': digest(options.campaign) if getattr(options, 'campaign', None) else None,
                    'campaign_kind': manifest.get('kind'), 'policy_smoke': policies,
                    'remaining_length_rule': active_rule['rule'], 'remaining_length': remaining_length_record(active_rule),
                    'snapshot_staleness_levels_ms': staleness_levels(manifest),
                    # lambda_weights lists the per-cell multipliers a sweep overlay varies; an ordinary
                    # overlay instead carries one tuned SCORE multiplier for all of its SCORE cells, and a
                    # release is reviewed against the qualification alone, so it is recorded here too.
                    'lambda_weights': lambda_levels(manifest),
                    'score_lambda_weight': manifest.get('score_lambda_weight'),
                    'data_role': campaign_data_role(manifest),
                    'calibration_capped_outputs': calibration_capped_outputs(read(output/'model_metrics.json')),
                    'serving_coefficients': 'Canonical SFS batch coefficients retained; destination residuals require review',
                    'evaluation_started': False})
                if options.mode == 'qualify':
                    return
                write(output/'phase.json', {'state': 'QUALIFIED_AWAITING_REVIEW', 'time': time.time()})
                # Keep this pool warm while the controller reviews calibration
                # residuals and stress results. No evaluation before release.
                while not (qualification/'release.json').exists():
                    await asyncio.sleep(5)
                validate_release(qualification, options, source, active_rule['rule'])
            cells = [c for c in manifest['cells'] if c['family'] == options.family and c['variant'] == options.variant]
            if options.cells:
                requested = set(options.cells.split(','))
                if not requested.issubset({c['id'] for c in cells}):
                    raise ValueError('Requested cells do not match this family/variant')
                cells = [c for c in cells if c['id'] in requested]
            ledger = ledger_dir(options.state, manifest)
            for cell in cells:
                with locks(Path(options.state)/'cell-locks', [cell['id']]):
                    done = ledger/(cell['id']+'.json')
                    if done.exists():
                        previous = read(done)
                        if previous['bundle_sha256'] != digest(Path(options.bundle)/'bundle.json'):
                            raise ValueError('Completed cell belongs to a different bundle')
                        if not completed_source_accepted(previous, source, manifest):
                            raise ValueError('Completed cell used different source; review before mixing implementations')
                        if digest(previous['point']) != previous['point_sha256']:
                            raise ValueError('Completed point checksum changed')
                        continue
                    validate_release(qualification, options, source_hashes(), active_rule['rule'])
                    staleness, weight, delta = cell_staleness(cell), routing_lambda(manifest, cell), cell_delta(cell)
                    args = args_for(cell['policy'], cell['qps'], cell['requests'], staleness, weight, delta)
                    requests, _, _ = exp._build_request_set(args)
                    if len(requests) != cell['requests']:
                        raise ValueError('Evaluation ingestion budget changed')
                    folder = output/'cells'/cell['id']
                    write(output/'active_cell.json', {'cell': cell, 'started': time.time()})
                    payload = await run_point(options.family, args, requests, clients, costs, metadata, folder)
                    audit = audit_cell(payload, cell)
                    if payload['router']['runs'][0].get('remaining_length_rule', 'current') != active_rule['rule']:
                        raise ValueError('Router did not record the pool remaining-length rule')
                    if float(payload['router']['runs'][0].get('snapshot_staleness_ms', 0) or 0) != staleness:
                        raise ValueError('Router did not record the cell snapshot staleness')
                    if weight is not None and float(payload['config']['score_lambda_weight']) != weight:
                        raise ValueError('Router did not record the cell SCORE routing multiplier')
                    if delta is not None and float(payload['config']['delta_weight']) != delta:
                        raise ValueError('Router did not record the cell latency penalty')
                    if float(payload['config']['lambda_weight']) != float(parse(argv).lambda_weight):
                        raise ValueError('Cell was scored on a different objective lambda than the bundle')
                    if source_hashes() != source:
                        raise ValueError('Runtime source changed during evaluation')
                    point = folder/'point.json'
                    entry = {**audit, 'cell': cell, 'point': str(point), 'point_sha256': digest(point),
                        'source_sha256': source, 'bundle_sha256': digest(Path(options.bundle)/'bundle.json'),
                        'qualification_sha256': digest(qualification/'qualification.json'), 'hardware': machine,
                        'campaign_sha256': digest(options.campaign) if getattr(options, 'campaign', None) else None,
                        'campaign_kind': manifest.get('kind'), 'remaining_length_rule': active_rule['rule'],
                        'remaining_length': remaining_length_record(active_rule),
                        'snapshot_staleness_ms': staleness, 'snapshot_staleness_levels_ms': staleness_levels(manifest),
                        'lambda_weight': weight, 'lambda_weights': lambda_levels(manifest),
                        'score_routing_lambda_weight': weight,
                        'delta_weight': delta, 'delta_weights': delta_levels(manifest),
                        'evaluation_lambda_weight': float(parse(argv).lambda_weight),
                        'data_role': cell.get('data_role', campaign_data_role(manifest))}
                    write(folder/'audit.json', entry)
                    write(done, entry)
                    write(output/'phase.json', {'state': 'CELL_COMPLETE', 'cell': cell['id'], 'time': time.time(),
                                                'remaining_length_rule': active_rule['rule']})
        watcher = asyncio.create_task(heartbeat())
        task = asyncio.create_task(workload())
        try:
            completed, _ = await asyncio.wait((watcher, task), return_when=asyncio.FIRST_COMPLETED)
            for future in completed: await future
        finally:
            for future in (watcher, task):
                future.cancel()
            await asyncio.gather(watcher, task, return_exceptions=True)
            for client in clients.values(): client.close()


def validate_release(qualification, options, source, remaining_length_rule='current'):
    report, release = read(qualification/'qualification.json'), read(qualification/'release.json')
    if (release.get('status') != 'RELEASED' or release.get('qualification_sha256') != digest(qualification/'qualification.json')
            or not release.get('timing_review') or not release.get('load_review')
            or report['source_sha256'] != source or report['family'] != options.family or report['variant'] != options.variant
            or report['bundle_sha256'] != digest(Path(options.bundle)/'bundle.json')
            or report['hardware'] != hardware(options.gpus.split(','))):
        raise ValueError('Missing/stale destination qualification and reviewed release')
    if getattr(options, 'campaign', None) and report.get('campaign_sha256') != digest(options.campaign):
        raise ValueError('Active campaign changed after qualification')
    if report.get('remaining_length_rule', 'current') != remaining_length_rule:
        raise ValueError('Pool remaining-length rule differs from the qualification')
    for name, expected in report['files'].items():
        if digest(qualification/name) != expected:
            raise ValueError(f'Qualification evidence changed: {name}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['qualify', 'run', 'campaign'])
    p.add_argument('--bundle', required=True); p.add_argument('--models', required=True)
    p.add_argument('--state', required=True); p.add_argument('--output', required=True)
    p.add_argument('--family', choices=['qwen', 'ministral'], required=True)
    p.add_argument('--variant', choices=['canonical', 'mlp_quality', 'mlp_length', 'flash_quality'], default='canonical')
    p.add_argument('--gpus', required=True); p.add_argument('--cpus'); p.add_argument('--qualification'); p.add_argument('--cells')
    p.add_argument('--campaign', help='Explicit overlay on the frozen artifact bundle (baseline, sfs_score, predictor_variants, staleness_sweep or score_lambda_sweep kind)')
    options = p.parse_args()
    if options.family == 'ministral' and options.variant != 'canonical':
        p.error('Predictor ablations are Qwen only')
    if options.mode == 'run' and not options.qualification: p.error('Run requires destination qualification')
    if options.cpus:
        os.sched_setaffinity(0, {int(c) for c in options.cpus.split(',')})
    manifest = validate_bundle(options.bundle)
    if options.mode == 'campaign' and not options.campaign:
        p.error('Campaign mode requires its explicit manifest')
    if options.campaign:
        from scripts.cloud.campaigns import apply_any_campaign
        manifest = apply_any_campaign(manifest, read(options.campaign))
        if not any(c['family'] == options.family and c['variant'] == options.variant for c in manifest['cells']):
            p.error('The campaign overlay has no cells for this family/variant')
    if options.mode == 'campaign':
        options.qualification = str(Path(options.output).resolve())
    output = Path(options.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    def stop(signum, frame): raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    write(output/'status.json', {'state': 'RUNNING', 'pid': os.getpid(), 'started': time.time(), 'options': vars(options)})
    try:
        asyncio.run(execute(options, manifest, manifest['families'][options.family], read(options.models), output))
    except BaseException as error:
        write(output/'status.json', {'state': 'FAILED', 'error': f'{type(error).__name__}: {error}', 'ended': time.time()})
        raise
    else:
        write(output/'status.json', {'state': 'QUALIFIED_AWAITING_REVIEW' if options.mode == 'qualify' else 'COMPLETE', 'ended': time.time()})


if __name__ == '__main__': main()

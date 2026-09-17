"""Join frozen judge scores on derived copies, preserving provider provenance."""
import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import statistics

from scripts.cloud.common import read, write, digest, validate_bundle, source_digest
from scripts.cloud.worker import audit_cell, family_remaining_length
from scripts.cloud.pool import remaining_length_provenance


def evaluation_judge(cell, requested='auto'):
    if requested not in ('auto', 'pro', 'flash'):
        raise ValueError('Unknown evaluation judge')
    if cell['family'] != 'qwen':
        return 'pro'
    if requested != 'auto':
        return requested
    return 'flash' if cell['variant'] == 'flash_quality' else 'pro'


def observed_qwen_queries(bundle, models, quality):
    """Use the same complete, observed candidate groups for both judge views."""
    excluded = set()
    for directory in ('scores', 'scores_flash'):
        for path in (Path(bundle)/'qwen'/directory).glob('*/*_scored.jsonl'):
            with path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if row.get('quality_imputed') or row.get('quality_metric') == 'judge_default_bucket_mean':
                        excluded.add((row['bucket'], str(row['prompt_metadata']['example_id'])))
    candidates = {key[1:] for key in quality['pro']}
    return {key for key in candidates if key not in excluded and
            all((model, *key) in quality[judge] for model in models for judge in ('pro', 'flash'))}


def observed_utilities(payload, mapping, quality, common, primary):
    from sfs_core.shared.model_label_helpers import normalize_model_label
    run = payload['router']['runs'][0]
    rows = run['per_request']
    if len(rows) != len({r['request_id'] for r in rows}) or {r['request_id'] for r in rows} != set(mapping):
        raise ValueError('Evaluation request map does not match the routed requests')
    utilities = {'pro': [], 'flash': []}
    for row in rows:
        mapped = mapping[row['request_id']]
        if row['bucket'] != mapped.bucket:
            raise ValueError('Request bucket disagrees with frozen map')
        gate = row['system_entry_e2e_ttft_slo_met']
        if gate != (row['system_entry_e2e_ttft_ms'] <= row['ttft_slo_ms']):
            raise ValueError('TTFT attainment flag disagrees with measured latency')
        key = (mapped.bucket, mapped.example_id)
        if key not in common:
            continue
        model = normalize_model_label(row['response_model'])
        cost = row['actual_cost']
        if not math.isfinite(cost):
            raise ValueError('Nonfinite realized cost')
        for judge in utilities:
            utilities[judge].append((quality[judge][(model, *key)] - payload['config']['lambda_weight'] * cost) * gate)
    if not utilities['pro']:
        raise ValueError('No observed scored queries')
    scores = {judge: statistics.mean(values) for judge, values in utilities.items()}
    return {'requests': len(rows), 'observed_scored_queries': len(utilities['pro']),
            'excluded_queries': len(rows)-len(utilities['pro']), 'primary_judge': primary,
            'primary_ontimeutility': scores[primary], 'ontimeutility': scores,
            'ttft_slo_attainment_pct': run['summary']['system_entry_e2e_ttft_slo_attainment_pct'],
            'ttft_denominator': len(rows),
            'quality_definition': 'Saved candidate scores joined to actual routing decisions; common observed query groups for both judges'}


def campaign_manifest(bundle, campaign=None):
    """The frozen bundle manifest, overlaid by the campaign document when one is given."""
    m = validate_bundle(bundle)
    if campaign:
        from scripts.cloud.campaigns import apply_any_campaign
        m = apply_any_campaign(m, read(campaign), inspect=True)
    return m


def expected_rules(manifest):
    """Remaining-length rule label every completed cell of each family must carry under this overlay."""
    return {family: remaining_length_provenance(definition, family_remaining_length(manifest, definition))['rule']
            for family, definition in manifest['families'].items()}


def check_record(record, expected, campaign_sha=None, kind=None, rules=None):
    """A completed cell must belong to the overlay; under a kinded overlay also to this exact overlay and rule."""
    cid = record['cell']['id']
    if cid not in expected or record['cell'] != expected[cid]: raise ValueError('Unexpected campaign cell')
    if kind:
        if record.get('campaign_sha256') != campaign_sha: raise ValueError(f'Cell completed under a different campaign overlay: {cid}')
        if record.get('remaining_length_rule') != rules[record['cell']['family']]:
            raise ValueError(f'Cell ran under a different remaining-length rule: {cid}')
        if kind == 'staleness_sweep' and record.get('snapshot_staleness_ms') != record['cell']['snapshot_staleness_ms']:
            raise ValueError(f'Cell ran under a different snapshot staleness: {cid}')
    return cid


def audit_status(expected, missing):
    return 'PARTIAL' if missing else f'PASS_{len(expected)}_CELLS'


def reused_canonical(reuse_roots, bundle_sha256):
    """Canonical Qwen rows from earlier collations of this bundle: cell id -> (observed summary, audit record, root)."""
    rows = {}
    for root in reuse_roots:
        root = Path(root); audit = read(root/'audit.json'); observed = read(root/'observed_judge_summary.json')
        if audit['bundle_sha256'] != bundle_sha256: raise ValueError(f'Reused collation belongs to a different bundle: {root}')
        for cid, record in audit['provenance'].items():
            if record['cell']['family'] != 'qwen' or record['cell']['variant'] != 'canonical': continue
            if cid in rows: raise ValueError(f'Canonical cell reused from two collations: {cid}')
            rows[cid] = (observed[cid], record, str(root))
    return rows


def bridges_reference_rows(reference, mapping, quality, common):
    """April Bridges references by (policy, qps), labelled provider=bridges; OnTimeUtility is recomputed on the
    observed judge cohort from the referenced point when it is readable here and matches its recorded sha256."""
    rows = {}
    for cell in read(reference)['cells']:
        key = (cell['policy'], float(cell['qps']))
        if key in rows: raise ValueError(f'Duplicate Bridges reference: {key}')
        row = {'provider': 'bridges', 'cell_id': None, 'point': cell['point'], 'point_sha256': cell['point_sha256'],
               'status': cell['status'], 'ttft_slo_attainment_pct': cell['summary']['system_entry_e2e_ttft_slo_attainment_pct'],
               'ontimeutility': None, 'remaining_length_rule': 'current', 'hardware': {'host': 'bridges'}, 'reused_from': str(reference)}
        point = Path(cell['point'])
        if not point.is_file():
            row['ontimeutility_note'] = 'Bridges point not readable on this host; attainment copied from the reference inventory'
        elif digest(point) != cell['point_sha256']:
            raise ValueError(f'Bridges reference point changed: {point}')
        else:
            payload = read(point)
            runs = [r for r in payload['router']['runs'] if r['utility'] == cell['policy']]
            if len(runs) != 1: raise ValueError(f'Bridges point lacks a unique {cell["policy"]} run: {point}')
            summary = observed_utilities({'config': payload['config'], 'router': {'runs': runs}}, mapping, quality, common, 'pro')
            row.update(ontimeutility=summary['ontimeutility'], observed_scored_queries=summary['observed_scored_queries'],
                       ontimeutility_note='Recomputed from the Bridges point on the observed judge cohort')
        rows[key] = row
    return rows


def _fill(row, observed, record, source, provider, reused_from=None):
    hardware = record.get('hardware')
    row.update(source=source, provider=provider, cell_id=record['cell']['id'], ontimeutility=observed['ontimeutility'],
               primary_ontimeutility=observed['ontimeutility'][row['primary_judge']],
               ttft_slo_attainment_pct=observed['ttft_slo_attainment_pct'], observed_scored_queries=observed.get('observed_scored_queries'),
               hardware={'host': hardware['host'], 'gpus': [g[2] for g in hardware['gpus']]} if hardware else None,
               source_digest=source_digest(record['source_sha256']), qualification_sha256=record.get('qualification_sha256'),
               campaign_sha256=record.get('campaign_sha256'), remaining_length_rule=record.get('remaining_length_rule', 'current'),
               point=record['point'], point_sha256=record['point_sha256'], reused_from=reused_from)


def variant_matrix(manifest, measured, reused, bridges, judge='auto'):
    """One row per (variant, policy, qps): measured ablation cells, reused canonical cells for the no-op policies,
    and the canonical comparators (Vast for hard/score, the Bridges reference for latency_agnostic).

    measured: cell id -> (observed summary, audit record); reused: canonical cell id -> (observed, record, root);
    bridges: (policy, qps) -> bridges_reference_rows entry.
    """
    from scripts.cloud.variant_campaign import VARIANTS, ABLATED_POLICIES, NOOP_POLICIES, variant_cell_id
    rows = []
    for variant in ('canonical', *VARIANTS):
        policies = ABLATED_POLICIES if variant == 'canonical' else (*ABLATED_POLICIES, *NOOP_POLICIES)
        for policy in policies:
            for qps in manifest['families']['qwen']['qps']:
                row = {'variant': variant, 'policy': policy, 'qps': qps,
                       'primary_judge': evaluation_judge({'family': 'qwen', 'variant': variant}, judge)}
                if variant != 'canonical' and policy in ABLATED_POLICIES:
                    cid = variant_cell_id(variant, policy, qps)
                    if cid in measured: _fill(row, *measured[cid], 'measured', 'vast')
                    else: row.update(source='missing', cell_id=cid)
                else:
                    cid = f'qwen-{policy}-{qps:g}'
                    if cid in reused:
                        observed, record, root = reused[cid]
                        _fill(row, observed, record, 'reused_canonical', 'vast', root)
                    elif (policy, float(qps)) in bridges:
                        entry = bridges[(policy, float(qps))]
                        row.update(source='reused_canonical', **entry,
                                   primary_ontimeutility=entry['ontimeutility'][row['primary_judge']] if entry['ontimeutility'] else None)
                    else: row.update(source='missing_canonical', cell_id=cid)
                rows.append(row)
    return {'rows': rows, 'noop_policies': manifest.get('noop_policies'), 'comparators': manifest.get('comparators'),
            'judge_rule': 'flash_quality rows and their reused comparators report the Flash judge as primary; other rows Pro',
            'sources': {'measured': 'fresh cell of this overlay on Vast', 'reused_canonical': 'canonical cell reused (provider vast: an '
                        'earlier collation of this bundle; provider bridges: the April reference inventory)', 'missing': 'ablation cell '
                        'not yet completed', 'missing_canonical': 'no canonical comparator available'}}


def collate(bundle, roots, output, allow_partial=False, judge='auto', campaign=None, reuse_roots=(), bridges_reference=None):
    from scripts.eval.augment_router_actual_accuracy import load_req_map, load_quality_index, augment_file
    from scripts.reporting.router_qps_sweep_summary import aggregate_jsons
    bundle, output = Path(bundle), Path(output)
    m = campaign_manifest(bundle, campaign)
    kind, campaign_sha = m.get('kind'), (digest(campaign) if campaign else None)
    rules = expected_rules(m) if kind else None
    expected = {c['id']:c for c in m['cells']}
    points = {}
    for root in roots:
        for audit in Path(root).rglob('cells/*/audit.json'):
            record = read(audit); point = audit.parent/'point.json'
            cid = check_record(record, expected, campaign_sha, kind, rules)
            if cid in points: raise ValueError(f'Duplicate completed cell across hosts/attempts: {cid}')
            if digest(point) != record['point_sha256'] or record['bundle_sha256'] != digest(bundle/'bundle.json'):
                raise ValueError('Result or input bundle checksum mismatch')
            audit_cell(read(point), expected[cid])
            points[cid] = (point, record)
    missing = sorted(set(expected)-set(points))
    if missing and not allow_partial: raise ValueError(f'Missing {len(missing)} cells: {missing}')
    output.mkdir(parents=True, exist_ok=False)
    maps = {f:load_req_map(bundle/f/'request_map.csv') for f in m['families']}
    quality = {f:{'pro':load_quality_index(bundle/f/'scores')} for f in m['families']}
    quality['qwen']['flash'] = load_quality_index(bundle/'qwen/scores_flash')
    common = observed_qwen_queries(bundle, m['families']['qwen']['models'], quality['qwen'])
    expected_observed = read(bundle/'provenance/flash_holdout_comparison.json')['complete_queries']
    if len(common) != expected_observed:
        raise ValueError('Observed judge coverage disagrees with frozen comparison audit')
    provenance, observed, judges = {}, {}, {}
    for cid, (point, record) in points.items():
        cell = expected[cid]; family, variant = cell['family'], cell['variant']
        selected_judge = evaluation_judge(cell, judge)
        judges[cid] = selected_judge
        # Keep each family/variant and destination hardware visible in reporting.
        dest = output/'derived'/family/variant/(cid+'.json');dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(point,dest)
        stats=augment_file(json_path=dest,req_maps_by_holdout={cell['requests']//4:maps[family]},quality_index=quality[family][selected_judge],dry_run=False)
        if stats.skipped_reason or stats.missing_req_map or stats.missing_example_id or stats.unresolved_model or stats.missing_quality:
            raise ValueError(f'Incomplete quality join: {stats}')
        provenance[cid] = record
        if family == 'qwen':
            observed[cid] = observed_utilities(read(point), maps[family], quality[family], common, selected_judge)
    summaries = {}
    for family in m['families']:
        for directory in (output/'derived'/family).glob('*'):
            summary, rates = aggregate_jsons([directory])
            summaries[f'{family}/{directory.name}'] = summary
    write(output/'figure5_13_summary.json', summaries)
    write(output/'observed_judge_summary.json', observed)
    write(output/'audit.json', {'status':audit_status(expected, missing), 'cells':len(points), 'missing':missing,
        'campaign':str(campaign) if campaign else None, 'campaign_kind':kind, 'campaign_sha256':campaign_sha,
        'expected_remaining_length_rules':rules, 'qwen_evaluation_judge':judge, 'evaluation_judge_by_cell':judges,
        'flash_judge_imputation_provenance':read(bundle/'provenance/flash_holdout_comparison.json'),
        'primary_qwen_quality_report':'observed_judge_summary.json',
        'full_request_report':'figure5_13_summary.json includes the frozen imputed scores; observed_judge_summary.json excludes them consistently',
        'bundle_sha256':digest(bundle/'bundle.json'),'provenance':provenance,
        'comparison_boundary':'Cloud hardware and fresh timing calibration are recorded per cell; historical Bridges curves are separate evidence'})
    if kind == 'predictor_variants':
        measured = {cid: (observed[cid], provenance[cid]) for cid in points}
        reused = reused_canonical(reuse_roots, digest(bundle/'bundle.json'))
        bridges = bridges_reference_rows(bridges_reference, maps['qwen'], quality['qwen'], common) if bridges_reference else {}
        write(output/'variant_matrix.json', variant_matrix(m, measured, reused, bridges, judge))
    print(json.dumps({'cells':len(points),'missing':missing,'output':str(output)}))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True);p.add_argument('--raw-root',action='append',required=True)
    p.add_argument('--output',required=True);p.add_argument('--allow-partial',action='store_true')
    p.add_argument('--judge',choices=['auto','pro','flash'],default='auto')
    p.add_argument('--campaign',help='Overlay whose cells are collated (baseline, sfs_score or predictor_variants kind)')
    p.add_argument('--reuse-root',action='append',default=[],help='Earlier collation output of this bundle whose canonical Qwen cells the variant matrix reuses')
    p.add_argument('--bridges-reference',help='qwen-reference-inventory.json with the April Bridges canonical references (provider=bridges rows)')
    a=p.parse_args();collate(a.bundle,a.raw_root,a.output,a.allow_partial,a.judge,a.campaign,a.reuse_root,a.bridges_reference)

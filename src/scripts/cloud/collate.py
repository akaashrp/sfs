"""Join frozen judge scores on derived copies, preserving provider provenance."""
import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import statistics

from scripts.cloud.common import read, write, digest, validate_bundle
from scripts.cloud.worker import audit_cell


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


def collate(bundle, roots, output, allow_partial=False, judge='auto'):
    from scripts.eval.augment_router_actual_accuracy import load_req_map, load_quality_index, augment_file
    from scripts.reporting.router_qps_sweep_summary import aggregate_jsons
    bundle, output = Path(bundle), Path(output)
    m = validate_bundle(bundle)
    expected = {c['id']:c for c in m['cells']}
    points = {}
    for root in roots:
        for audit in Path(root).rglob('cells/*/audit.json'):
            record = read(audit); cid = record['cell']['id']; point = audit.parent/'point.json'
            if cid not in expected or record['cell'] != expected[cid]: raise ValueError('Unexpected campaign cell')
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
    write(output/'audit.json', {'status':'PARTIAL' if missing else 'PASS_68_CELLS', 'cells':len(points), 'missing':missing,
        'qwen_evaluation_judge':judge, 'evaluation_judge_by_cell':judges,
        'flash_judge_imputation_provenance':read(bundle/'provenance/flash_holdout_comparison.json'),
        'primary_qwen_quality_report':'observed_judge_summary.json',
        'full_request_report':'figure5_13_summary.json includes the frozen imputed scores; observed_judge_summary.json excludes them consistently',
        'bundle_sha256':digest(bundle/'bundle.json'),'provenance':provenance,
        'comparison_boundary':'Cloud hardware and fresh timing calibration are recorded per cell; historical Bridges curves are separate evidence'})
    print(json.dumps({'cells':len(points),'missing':missing,'output':str(output)}))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True);p.add_argument('--raw-root',action='append',required=True)
    p.add_argument('--output',required=True);p.add_argument('--allow-partial',action='store_true')
    p.add_argument('--judge',choices=['auto','pro','flash'],default='auto')
    a=p.parse_args();collate(a.bundle,a.raw_root,a.output,a.allow_partial,a.judge)

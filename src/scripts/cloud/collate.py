"""Join frozen judge scores on derived copies, preserving provider provenance."""
import argparse
import csv
import json
from pathlib import Path
import shutil

from scripts.cloud.common import read, write, digest, validate_bundle
from scripts.cloud.worker import audit_cell


def collate(bundle, roots, output, allow_partial=False, judge='pro'):
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
    quality = {f:load_quality_index(bundle/f/('scores_flash' if f=='qwen' and judge=='flash' else 'scores')) for f in m['families']}
    provenance = {}
    for cid, (point, record) in points.items():
        cell = expected[cid]; family, variant = cell['family'], cell['variant']
        # Keep each family/variant and destination hardware visible in reporting.
        dest = output/'derived'/family/variant/(cid+'.json');dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(point,dest)
        stats=augment_file(json_path=dest,req_maps_by_holdout={cell['requests']//4:maps[family]},quality_index=quality[family],dry_run=False)
        if stats.skipped_reason or stats.missing_req_map or stats.missing_example_id or stats.unresolved_model or stats.missing_quality:
            raise ValueError(f'Incomplete quality join: {stats}')
        provenance[cid] = record
    summaries = {}
    for family in m['families']:
        for directory in (output/'derived'/family).glob('*'):
            summary, rates = aggregate_jsons([directory])
            summaries[f'{family}/{directory.name}'] = summary
    write(output/'figure5_13_summary.json', summaries)
    write(output/'audit.json', {'status':'PARTIAL' if missing else 'PASS_68_CELLS', 'cells':len(points), 'missing':missing,
        'qwen_evaluation_judge':judge, 'flash_judge_imputation_provenance':read(bundle/'provenance/flash_holdout_comparison.json') if judge=='flash' else None,
        'bundle_sha256':digest(bundle/'bundle.json'),'provenance':provenance,
        'comparison_boundary':'Cloud hardware and fresh timing calibration are recorded per cell; historical Bridges curves are separate evidence'})
    print(json.dumps({'cells':len(points),'missing':missing,'output':str(output)}))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True);p.add_argument('--raw-root',action='append',required=True)
    p.add_argument('--output',required=True);p.add_argument('--allow-partial',action='store_true')
    p.add_argument('--judge',choices=['pro','flash'],default='pro')
    a=p.parse_args();collate(a.bundle,a.raw_root,a.output,a.allow_partial,a.judge)

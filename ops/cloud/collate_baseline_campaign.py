"""Incrementally audit and score the explicit baseline campaign on derived copies."""
import json
from pathlib import Path
import shutil
import time

from scripts.cloud.common import digest, read, write, validate_bundle
from scripts.cloud.worker import audit_cell
from scripts.cloud.collate import observed_qwen_queries, observed_utilities
from scripts.eval.augment_router_actual_accuracy import load_req_map, load_quality_index, augment_file
from scripts.reporting.router_qps_sweep_summary import aggregate_jsons

ROOT=Path('/workspace/sfs/state')
BUNDLE=Path('/workspace/sfs/bundle')
OUTPUT=ROOT/'baseline-results'


def main():
    m=validate_bundle(BUNDLE)
    campaign=read('/workspace/sfs/repo/scripts/cloud/baseline-campaign-20260916.json')
    expected={c['id']:c for c in campaign['cells']+campaign.get('fallback_cells',[])}
    maps={f:load_req_map(BUNDLE/f/'request_map.csv') for f in m['families']}
    quality={f:{'pro':load_quality_index(BUNDLE/f/'scores')} for f in m['families']}
    quality['qwen']['flash']=load_quality_index(BUNDLE/'qwen/scores_flash')
    common=observed_qwen_queries(BUNDLE,m['families']['qwen']['models'],quality['qwen'])
    if len(common)!=15996:raise ValueError('Qwen common judge coverage changed')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    while True:
        try:
            for audit in sorted((ROOT/'baselines').glob('*/cells/*/audit.json')):
                record=read(audit);cell=record['cell'];cid=cell['id']
                expected_cell=expected.get(cid)
                if expected_cell is None or any(cell[k]!=expected_cell[k] for k in ('id','family','variant','policy','qps','requests')):
                    raise ValueError('Result does not belong to explicit campaign')
                point=audit.parent/'point.json'
                if digest(point)!=record['point_sha256'] or record['bundle_sha256']!=digest(BUNDLE/'bundle.json'):
                    raise ValueError('Raw point/bundle changed')
                receipt=OUTPUT/'audits'/(cid+'.json')
                if receipt.exists():
                    done=read(receipt)
                    if done['point_sha256']!=record['point_sha256'] or digest(done['derived_point'])!=done['derived_sha256']:
                        raise ValueError('Duplicate changed cell or modified scored result')
                    continue
                payload=read(point);audit_cell(payload,cell)
                family=cell['family'];dest=OUTPUT/'derived'/family/(cid+'.json')
                dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists():raise ValueError('Unreceipted scored point; inspect previous attempt before resuming')
                shutil.copyfile(point,dest)
                stats=augment_file(json_path=dest,req_maps_by_holdout={cell['requests']//4:maps[family]},quality_index=quality[family]['pro'],dry_run=False)
                if any(getattr(stats,k) for k in ('skipped_reason','missing_req_map','missing_example_id','unresolved_model','missing_quality')):
                    raise ValueError(f'Incomplete judge join: {stats}')
                derived=read(dest)
                scores=observed_utilities(payload,maps[family],quality[family],common,'pro') if family=='qwen' else {'primary_judge':'pro','summary':derived['router']['runs'][0]['summary']}
                write(receipt,{**record,'derived_point':str(dest),'derived_sha256':digest(dest),'scoring':scores})
                print(json.dumps({'cell':cid,'status':'AUDITED_AND_SCORED','scores':scores}),flush=True)
            receipts={p.stem:read(p) for p in (OUTPUT/'audits').glob('*.json')}
            write(OUTPUT/'progress.json',{'time':time.time(),'status':'MONITORING','completed':len(receipts),'cells':{c:r['scoring'] for c,r in receipts.items()},'missing_primary':[c['id'] for c in campaign['cells'] if c['id'] not in receipts]})
            summaries={f:aggregate_jsons([OUTPUT/'derived'/f])[0] for f in m['families'] if (OUTPUT/'derived'/f).exists()}
            write(OUTPUT/'figure5-summary.json',summaries)
        except Exception as error:
            write(OUTPUT/'progress.json',{'time':time.time(),'status':'SCORING_FAILED','error':str(error)})
            raise
        time.sleep(60)

if __name__=='__main__':main()

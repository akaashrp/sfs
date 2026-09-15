"""Recover valid cells and run only the missing Ministral matrix on Bridges.

Original job sources, manifests and result files remain immutable. A new
source-bound plan records the explicit audit/freshness corrections and requires
GPU smoke at the actual maximum load before any new evaluation cell.
"""
import argparse
import asyncio
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

from scripts.cloud.common import ROOT, digest, read, write, source_hashes
from scripts.runs import ministral3_figure5 as fig
from scripts.runs.measured_audit import require_complete_ttft

REUSE = ['lmdeploy_proxy', 'latency_agnostic', 'round_robin']
RERUN = ['mooncake_prefill', 'hard', 'score', 'routebalance', 'shortest_queue']


def salvage(manifest_path, raw_root, output):
    manifest = fig.load_manifest(manifest_path)
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    raw_paths = fig.point_paths(Path(raw_root)/'outputs')
    before = {str(p): digest(p) for p in raw_paths}
    paths = []; rejected = []
    for raw in raw_paths:
        payload = read(raw)
        for run in payload['router']['runs']:
            if run['utility'] not in REUSE:
                rejected.append({'qps':payload['config']['request_rate_qps'], 'policy':run['utility'],
                    'failed_requests':run['summary']['failed_requests']})
                continue
            path = output/'outputs'/f'point_{len(paths):03d}.json'
            derived = {**payload, 'router':{**payload['router'], 'runs':[run]},
                       'recovery_source':{'path':str(raw), 'sha256':before[str(raw)]}}
            write(path, derived);paths.append(path)
    audit = fig.audit_points(paths, manifest, REUSE)
    # Quality is joined on separate derived copies, never the saved raw points.
    from scripts.eval.augment_router_actual_accuracy import load_req_map, load_quality_index, augment_file
    request_map = load_req_map(Path(manifest['request_map']))
    quality = load_quality_index(Path(manifest['scored_root']))
    derived_paths = []
    for source in paths:
        path = output/'quality'/source.name;path.parent.mkdir(exist_ok=True)
        shutil.copyfile(source,path)
        stats=augment_file(json_path=path,req_maps_by_holdout={2000:request_map},quality_index=quality,dry_run=False)
        if any((stats.skipped_reason,stats.missing_req_map,stats.missing_example_id,stats.unresolved_model,stats.missing_quality)):
            raise ValueError(f'Quality join failed: {stats}')
        derived_paths.append(path)
    quality_audit = fig.audit_points(derived_paths,manifest,REUSE,augmented=True)
    assert all(digest(p)==value for p,value in before.items()), 'Original results changed'
    audit.update(manifest_sha256=digest(manifest_path), point_sha256={str(p):digest(p) for p in paths},
                 source_sha256=source_hashes(), original_results_sha256=before, original_results_modified=False,
                 quality_audit=quality_audit, rejected=rejected, correction='Use producer TTFT summary key and validate measured rows')
    write(output/'audit.json',audit)
    print(json.dumps({'status':'PASS_REUSED_12_CELLS','requests':audit['records'],'output':str(output)}))


def prepare(manifest_path, calibration, salvaged, gate, output):
    manifest=fig.load_manifest(manifest_path)
    source=source_hashes();tests=read(gate)
    if tests['status']!='PASS_CPU_REGRESSION' or tests['source_sha256']!=source:
        raise ValueError('Run the regression gate against these exact sources')
    audit=read(Path(salvaged)/'audit.json')
    if audit['status']!='PASS' or audit['matrix_cells']!=12 or audit['manifest_sha256']!=digest(manifest_path):
        raise ValueError('Expected the audited twelve-cell recovery')
    rows=[json.loads(line) for line in Path(calibration).read_text().splitlines()]
    from scripts.runs.ministral3_methodology_stage import BUCKETS
    if Counter(r['bucket'] for r in rows)!=Counter({b:2500 for b in BUCKETS}):
        raise ValueError('Expected the frozen balanced calibration pool')
    stage = read(Path(manifest['stage_dir'])/'stage_started.json')
    # The stage records the authoritative calibration request bytes.
    metadata_path = Path(stage['prepared_dir'])/'metadata.json'
    meta = read(metadata_path)
    if (stage['prepared_metadata_sha256'] != digest(metadata_path)
            or meta.get('data_role') != 'calibration' or meta.get('holdout_start_index') != 0
            or meta.get('requests_sha256') != digest(calibration)):
        raise ValueError('Calibration requests differ from the measured stage')
    plan={'schema_version':1,'manifest':str(Path(manifest_path).resolve()),'manifest_sha256':digest(manifest_path),
        'source_sha256':source,'test_gate':str(Path(gate).resolve()),'test_gate_sha256':digest(gate),
        'calibration_requests':str(Path(calibration).resolve()),'calibration_sha256':digest(calibration),
        'salvaged':str(Path(salvaged).resolve()),'salvage_audit_sha256':digest(Path(salvaged)/'audit.json'),
        'policies':RERUN,'cells':[{'policy':p,'qps':q,'requests':8000} for p in RERUN for q in manifest['loads']['qps_values']],
        'required_gpu_gate':'192 calibration requests per policy plus 512 each for Mooncake and RouteBalance at maximum evaluation QPS',
        'scope':'12 audited cells reused; 4 failed Mooncake cells rerun; 16 previously pending snapshot-policy cells run',
        'created_at':time.time()}
    write(output,plan)
    print(json.dumps({'status':'PREPARED_GPU_GATE_REQUIRED','cells':len(plan['cells']),'plan':str(output)}))


def validate(plan_path):
    plan=read(plan_path)
    if plan['source_sha256']!=source_hashes():raise ValueError('Recovery source changed')
    for path,sha in [(plan['manifest'],plan['manifest_sha256']),(plan['calibration_requests'],plan['calibration_sha256']),
                     (plan['test_gate'],plan['test_gate_sha256']), (str(Path(plan['salvaged'])/'audit.json'),plan['salvage_audit_sha256'])]:
        if digest(path)!=sha:raise ValueError(f'Recovery evidence changed: {path}')
    for path,sha in read(Path(plan['salvaged'])/'audit.json')['point_sha256'].items():
        if digest(path)!=sha:raise ValueError(f'Recovered point changed: {path}')
    manifest=fig.load_manifest(plan['manifest'])
    return plan,manifest


async def run(plan_path, instances_path, wait_logs, output):
    from scripts.runs import experiments as exp
    from scripts.runs.experiments_sweep import _parse_experiment_args
    from scripts.runs.ministral3_methodology_stage import smoke_requests,audit_run,wait_drained
    from sfs_core.shared.shared_experiment_helpers import warm_up_instances
    from scripts.cloud.worker import run_point,audit_cell
    plan,manifest=validate(plan_path);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    args=_parse_experiment_args(manifest['experiment_argv'])
    args.per_request_wait_log=list(wait_logs)
    requests,_,_=exp._build_request_set(args)
    if len(requests)!=8000 or {r.request_id for r in requests}!={f'req-{i}' for i in range(8000)}:
        raise ValueError('Full frozen evaluation request set required')
    calibration=[exp.ExperimentRequest(**json.loads(line)) for line in Path(plan['calibration_requests']).read_text().splitlines()]
    clients,costs,metadata=exp.load_instances(instances_path)
    if set(c.model_id for c in clients.values())!={'ministral3-3b','ministral3-8b','ministral3-14b'}:
        raise ValueError('Unexpected recovery pool')
    write(output/'provenance.json',{'plan_sha256':digest(plan_path),'source_sha256':plan['source_sha256'],
        'slurm_job_id':os.getenv('SLURM_JOB_ID'),'hostname':os.uname().nodename,'instance_metadata':metadata,
        'instances_sha256':digest(instances_path),'freshness_gate_ms':1000,'snapshot_stall_timeout_s':10})
    async def heartbeat():
        while True:
            write(output/'heartbeat.json',{'time':time.time(),'pid':os.getpid()})
            await asyncio.sleep(15)
    pulse=asyncio.create_task(heartbeat())
    try:
        write(output/'status.json',{'state':'GPU_SMOKE'})
        await warm_up_instances(list(clients.values()));await wait_drained(clients,timeout_s=120)
        for policy in plan['policies']:
            selected=smoke_requests(calibration);args.utilities=[policy];args.num_requests=len(selected)
            args.request_rate_qps=manifest['loads']['qps_values'][-1]
            payload=await run_point('ministral',args,selected,clients,costs,metadata,output/'smoke'/policy,data_role='calibration')
            audit_run(payload['router']['runs'][0],len(selected));require_complete_ttft(payload['router']['runs'][0])
        for policy in ('mooncake_prefill','routebalance'):
            selected=smoke_requests(calibration,per_bucket=128);args.utilities=[policy];args.num_requests=len(selected)
            payload=await run_point('ministral',args,selected,clients,costs,metadata,output/'stress'/policy,data_role='calibration')
            audit_run(payload['router']['runs'][0],len(selected));require_complete_ttft(payload['router']['runs'][0])
        write(output/'gpu_gate.json',{'status':'PASS','source_sha256':plan['source_sha256'],
            'point_sha256':{str(p):digest(p) for directory in ('smoke','stress') for p in (output/directory).rglob('point.json')}})
        paths=[]
        for i,cell in enumerate(plan['cells']):
            write(output/'status.json',{'state':'EVALUATING','completed':len(paths),'current_cell':cell})
            args.utilities=[cell['policy']];args.request_rate_qps=cell['qps'];args.num_requests=8000
            directory=output/'cells'/f'{cell["policy"]}-{cell["qps"]:g}'
            payload=await run_point('ministral',args,requests,clients,costs,metadata,directory)
            audit=audit_cell(payload,cell)
            # Use the same complete Figure 5 consumer contract after every cell.
            one={**manifest,'loads':{**manifest['loads'],'qps_values':[cell['qps']]}}
            fig.audit_points([directory/'point.json'],one,[cell['policy']])
            path=output/'outputs'/f'point_{i:03d}.json';path.parent.mkdir(exist_ok=True)
            shutil.copyfile(directory/'point.json',path);paths.append(path)
            write(directory/'audit.json',{**audit,'cell':cell,'point_sha256':digest(directory/'point.json')})
        audit=fig.audit_points(paths,manifest,plan['policies'])
        audit.update(manifest_sha256=plan['manifest_sha256'],point_sha256={str(p):digest(p) for p in paths},
                     source_sha256=plan['source_sha256'],plan_sha256=digest(plan_path))
        write(output/'audit.json',audit)
        fig.collate(SimpleNamespace(output_root=output/'collated_32',raw_runs=[plan['salvaged'],str(output)],
                                    manifest=plan['manifest']),manifest)
        write(output/'status.json',{'state':'PASS_32_CELLS','new_cells':20,'reused_cells':12})
    except BaseException as exc:
        write(output/'status.json',{'state':'FAILED','error':f'{type(exc).__name__}: {exc}'})
        raise
    finally:
        pulse.cancel();await asyncio.gather(pulse,return_exceptions=True)
        for client in clients.values():client.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['salvage','prepare','validate','run'])
    for arg in ('manifest','raw-root','output','calibration','salvaged','gate','plan','instances'):
        p.add_argument('--'+arg)
    p.add_argument('--wait-log',action='append',default=[])
    a=p.parse_args()
    if a.mode=='salvage':salvage(a.manifest,a.raw_root,a.output)
    elif a.mode=='prepare':prepare(a.manifest,a.calibration,a.salvaged,a.gate,a.output)
    elif a.mode=='validate':validate(a.plan);print('PASS_RECOVERY_PREFLIGHT')
    else:asyncio.run(run(a.plan,a.instances,a.wait_log,a.output))


if __name__=='__main__':main()

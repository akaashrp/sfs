"""Isolated predictor ablations using the canonical Qwen serving pool and SFS.

This module does not alter the source-bound baseline campaign. Each arm changes
one predictor, validates real CPU admission, and requires its own GPU smoke.
"""
from __future__ import annotations
import argparse,asyncio,contextlib,hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
from collections import Counter
from scripts.runs import qwen_baselines as qwen
from scripts.prep.paper_ablation_data import sha256,write_json

ARMS=('mlp_quality','mlp_length','flash_quality')


def replace_option(argv, key, value):
    args=list(argv)
    if args.count(key)!=1:raise ValueError(f'Expected one {key}')
    args[args.index(key)+1]=str(value)
    return args


def variant_paths(root,arm,flash_root=None):
    root=Path(root);canonical=root/'src/assets/predictors';mlp=root/'experiments/mlp_serving_20260911/trained'
    if arm not in ARMS:raise ValueError('Unknown predictor arm')
    accuracy=canonical/'accuracy_predictor';length=canonical/'output_length_predictor'
    if arm=='mlp_quality':accuracy=mlp/'accuracy_predictor'
    elif arm=='mlp_length':length=mlp/'output_length_predictor'
    else:
        if flash_root is None:raise ValueError('Flash export required')
        accuracy=Path(flash_root)/'accuracy_predictor'
    return {'router_accuracy_model_path':str(accuracy),'router_output_length_model_path':str(length),'server_output_length_model_path':str(length)}


def experiment_argv(canonical, paths):
    args=replace_option(canonical['experiment_argv'],'--accuracy-model-path',paths['router_accuracy_model_path'])
    return replace_option(args,'--output-length-model-path',paths['router_output_length_model_path'])


def server_argv(root,model,row,index,output,paths):
    if paths['server_output_length_model_path']!=paths['router_output_length_model_path']:
        raise ValueError('Router/server length predictors differ')
    return replace_option(qwen.server_argv(root,model,row,index,output),'--output-length-model-path',paths['server_output_length_model_path'])


def verify_files(files):
    for p,digest in files.items():
        if sha256(p)!=digest:raise ValueError(f'Frozen variant input changed: {p}')


def source_files(root):
    from scripts.runs.latency_validation import source_hashes
    files={str(root/p):h for p,h in source_hashes(root).items()}
    for p in [Path(__file__),root/'src/scripts/runs/serving_contract_preflight.py',root/'src/scripts/prep/train_serving_mlp.py',root/'vllm/vllm/v1/engine/accuracy_predictor.py',root/'vllm/vllm/v1/engine/mlp_predictor.py']:
        files[str(p.resolve())]=sha256(p)
    return files


def validate_predictions(qs,ls,*,quality_backend,length_backend):
    if len(qs)!=576 or len(ls)!=576 or any(not math.isfinite(float(q))for q in qs):
        raise ValueError('Invalid real consumer predictions')
    if quality_backend=='numpy_mlp' and any(not 0<=q<=1 for q in qs):
        raise ValueError('MLP quality clipping contract violated')
    limit=8192+8*math.ulp(8192.)
    if any(not math.isfinite(p.mean_tokens)or p.mean_tokens<=0 or
           (length_backend=='numpy_mlp'and p.mean_tokens>limit)for p in ls):
        raise ValueError('Invalid real consumer length predictions')


def prepare(root,arm,canonical_path,stage,cpu_report,output,flash_root=None):
    from scripts.runs.latency_validation import validate_cpu
    from scripts.runs.serving_contract_preflight import validate as serving_preflight
    from scripts.runs.experiments_sweep import _parse_experiment_args
    from scripts.runs import experiments as exp
    from vllm.v1.engine.accuracy_predictor import AccuracyPredictor
    from vllm.v1.engine.output_length_predictor import OutputLengthPredictor,AdmissionFeatures
    root=Path(root).resolve();output=Path(output).resolve();stage=Path(stage).resolve()
    if output.exists():raise ValueError('Preserve prior variant manifest')
    canonical=qwen.validate_manifest(canonical_path);validate_cpu(cpu_report,root)
    rb=root/'experiments/paper_ablation_20260907/qwen_routebalance_predictor'
    qwen.validate_smoke(stage,canonical,rb)
    paths=variant_paths(root,arm,flash_root)
    training=Path(flash_root)/'training_audit.json'if arm=='flash_quality'else root/'experiments/mlp_serving_20260911/trained/serving_mlp_audit.json'
    audit=json.loads(training.read_text())
    if audit['status']!='PASS_CPU_EXPORT':raise ValueError('Training/export did not pass')
    verify_files(audit['source_sha256'])
    if arm=='flash_quality':
        if audit.get('judge_model')!='gemini-2.5-flash':raise ValueError('Flash labels required')
        verify_files(audit['artifact_sha256'])
    sources=source_files(root)
    files={**sources,**canonical['file_sha256'],str(Path(canonical_path).resolve()):sha256(canonical_path),str(training):sha256(training)}
    for path in paths.values():
        for p in Path(path).iterdir():
            if p.is_file():files[str(p.resolve())]=sha256(p)
    for p in [stage/'model_metrics.json',stage/'smoke_audit.json',Path(cpu_report)]:files[str(p.resolve())]=sha256(p)
    args=experiment_argv(canonical,paths)
    # Same complete holdout; compare both construction orders, prompts and SLOs.
    reference,_,_=exp._build_request_set(_parse_experiment_args(canonical['experiment_argv']))
    actual,_,_=exp._build_request_set(_parse_experiment_args(args))
    if actual!=reference or len(actual)!=16000 or Counter(r.bucket for r in actual)!=Counter({b:4000 for b in qwen.BUCKETS}):
        raise ValueError('Variant changed canonical 16000-request workload')
    sample=qwen.smoke_requests([exp.ExperimentRequest(**r)for r in qwen.rows(canonical['calibration_requests'])])
    admissions=[AdmissionFeatures(m,r.prompt,r.prompt_tokens)for r in sample for m in qwen.MODELS]
    quality=AccuracyPredictor(paths['router_accuracy_model_path']);length=OutputLengthPredictor(paths['router_output_length_model_path'])
    qs=quality.predict_batch(admissions);ls=length.predict_batch(admissions)
    # Canonical LightGBM quality uses an identity link (unbounded regression).
    # The MLP log1p inverse can exceed 8192 by a few floating-point ULPs.
    validate_predictions(qs,ls,quality_backend=quality._backend,length_backend=length._backend)
    contract=serving_preflight(root,'qwen',Path(paths['server_output_length_model_path']))
    for index,row in enumerate(qwen.pool_config(canonical)['instances']):
        argv=server_argv(root,Path(canonical['model_paths'][index]),row,index,root/'experiments/predictor_prereqs_20260912/argv_only',paths)
        if argv[argv.index('--output-length-model-path')+1]!=paths['router_output_length_model_path']:raise ValueError('Wrong server artifact')
    verify_files(files)
    write_json(output,{'status':'PASS_CPU_VARIANT','arm':arm,'sfs_root':str(root),'canonical_manifest':str(Path(canonical_path).resolve()),'stage':str(stage),'requests_per_cell':16000,'qps_values':list(qwen.QPS),'policy':'hard','paths':paths,'experiment_argv':args,'file_sha256':files,'source_sha256':sources,'serving_contract':contract,'calibration_prediction_candidates':576,'gpu_smoke_passed':False,'canonical_ingestion_identical':True})


def validate(path):
    m=json.loads(Path(path).read_text())
    if m.get('status')!='PASS_CPU_VARIANT'or m['arm']not in ARMS or m['requests_per_cell']!=16000 or m['qps_values']!=list(qwen.QPS)or m['policy']!='hard':raise ValueError('Invalid frozen variant contract')
    verify_files(m['file_sha256'])
    if source_files(Path(m['sfs_root']))!=m['source_sha256']:raise ValueError('Variant sources changed')
    if m['paths']['router_output_length_model_path']!=m['paths']['server_output_length_model_path']:raise ValueError('Router/server length mismatch')
    return m


@contextlib.contextmanager
def pool(m,output):
    from scripts.runs.serving_ipc import qwen_ipc_environment
    import urllib.request
    root=Path(m['sfs_root']);canonical=qwen.validate_manifest(m['canonical_manifest']);job=os.environ.get('SLURM_JOB_ID','');visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if not job.isdecimal()or len(visible)!=4:raise ValueError('Requires four allocated GPUs')
    local=Path('/local')/os.environ['USER']/('sfs-variants-'+job);local.mkdir(parents=True,exist_ok=True)
    env=qwen_ipc_environment(local,os.environ);ports=[20000+(int(job)%10000)*3+i for i in range(3)]
    config=qwen.pool_config(canonical,ports,job+'_'+m['arm']);config['predictor_variant']={'arm':m['arm'],'paths':m['paths']}
    config_path=output/'instances.json';write_json(config_path,config)
    processes=[];logs=[]
    try:
        for i,row in enumerate(config['instances']):
            dest=local/qwen.HF_NAMES[i];dest.mkdir(exist_ok=True)
            subprocess.run(['rsync','-aL',canonical['model_paths'][i]+'/',str(dest)+'/'],check=True)
            argv=server_argv(root,dest,row,i,output,m['paths']);write_json(output/f'server_argv_{row["model_id"]}.json',argv)
            server_env=dict(env,CUDA_VISIBLE_DEVICES=visible[i]if i<2 else ','.join(visible[2:]),VLLM_ATTENTION_BACKEND='FLASH_ATTN',VLLM_USE_FLASHINFER_SAMPLER='0',VLLM_PER_REQUEST_WAIT_LOG_PATH=str(output/f'wait_{row["model_id"]}.log'),XDG_CACHE_HOME=str(local/'.cache'),TRITON_CACHE_DIR=str(local/'.triton'),CUDA_CACHE_PATH=str(local/'.nv'))
            log=(output/f'server_{row["model_id"]}.log').open('x');logs.append(log)
            processes.append(subprocess.Popen(argv,cwd=root/'src',env=server_env,stdout=log,stderr=subprocess.STDOUT))
        deadline=time.monotonic()+1200
        for row in config['instances']:
            while True:
                if any(p.poll()is not None for p in processes):raise RuntimeError('Variant server exited before readiness')
                try:
                    with urllib.request.urlopen(row['address']+'/v1/models',timeout=2)as response:models=json.load(response)
                    if row['model_id']in {x['id']for x in models['data']}:break
                except(OSError,ValueError):pass
                if time.monotonic()>deadline:raise TimeoutError('Variant readiness deadline')
                time.sleep(2)
        yield config_path
    finally:
        for p in processes:
            if p.poll()is None:p.terminate()
        for p in processes:
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for log in logs:log.close()


async def smoke(m,manifest_path,instances_path,output):
    from scripts.runs import experiments as exp
    from scripts.runs.experiments_sweep import _parse_experiment_args
    from sfs_core.shared.shared_experiment_helpers import warm_up_instances
    canonical=json.loads(Path(m['canonical_manifest']).read_text());args=_parse_experiment_args(m['experiment_argv']+['--service-metrics-json',str(Path(m['stage'])/'model_metrics.json')])
    args.utilities=['hard'];args.num_requests=192;args.request_rate_qps=2.
    args.per_request_wait_log=[str(output/f'wait_{model}.log')for model in qwen.MODELS]
    requests=qwen.smoke_requests([exp.ExperimentRequest(**r)for r in qwen.rows(canonical['calibration_requests'])])
    instances,costs,metadata=exp.load_instances(instances_path)
    try:
        await warm_up_instances(list(instances.values()));await qwen.wait_drained(instances)
        result=await asyncio.wait_for(exp.run_router_experiment(args=args,requests=requests,instances=instances,instance_costs=costs,instance_metadata=metadata,response_map_base_path=output/'responses.log',request_log_base_path=output/'router_waits.log'),timeout=1800)
        write_json(output/'smoke_hard.json',result);qwen.audit_run(result['runs'][0],192);await qwen.wait_drained(instances)
        validate(manifest_path)
        write_json(output/'smoke_audit.json',{'status':'PASS_GPU_VARIANT_SMOKE','arm':m['arm'],'requests':192,'evaluation_started':False,'manifest_sha256':sha256(manifest_path),'result_sha256':sha256(output/'smoke_hard.json'),'paths':m['paths'],'artifact_and_source_hashes_verified':True,'evidence':'Exact launched server argv and frozen artifact hashes; CPU real chat/transport/scheduler checks; GPU SFS request and snapshot execution'})
    finally:
        for c in instances.values():c.close()


def validate_smoke(m,manifest_path,stage):
    stage=Path(stage);a=json.loads((stage/'smoke_audit.json').read_text())
    if a.get('status')!='PASS_GPU_VARIANT_SMOKE'or a.get('manifest_sha256')!=sha256(manifest_path)or a.get('result_sha256')!=sha256(stage/'smoke_hard.json')or a.get('paths')!=m['paths']or a.get('arm')!=m['arm']:raise ValueError('Missing or stale variant GPU smoke')
    result=json.loads((stage/'smoke_hard.json').read_text());qwen.audit_run(result['runs'][0],192)


def run(path,mode,output,smoke_stage=None):
    m=validate(path);output=Path(output)
    if mode=='sweep':validate_smoke(m,path,smoke_stage)
    output.mkdir(parents=True,exist_ok=False);write_json(output/'run_started.json',{'arm':m['arm'],'mode':mode,'manifest_sha256':sha256(path),'paths':m['paths']})
    with pool(m,output)as instances:
        if mode=='smoke':asyncio.run(smoke(m,path,instances,output))
        else:
            argv=[sys.executable,'-m','scripts.runs.experiments_sweep','--sweep','qps','--qps-values',*map(str,m['qps_values']),'--qps-utilities','hard','--output-dir',str(output/'outputs'),'--output-prefix',m['arm'],*m['experiment_argv'],'--instances-config',str(instances),'--service-metrics-json',str(Path(m['stage'])/'model_metrics.json')]
            for model in qwen.MODELS:argv+=['--per-request-wait-log',str(output/f'wait_{model}.log')]
            with (output/'driver.log').open('x')as log:subprocess.run(argv,cwd=Path(m['sfs_root'])/'src',stdout=log,stderr=subprocess.STDOUT,check=True)
            validate(path);write_json(output/'sweep_completed.json',{'status':'COMPLETE_UNCOLLATED','arm':m['arm'],'manifest_sha256':sha256(path)})


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','validate','smoke','sweep']);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--root',type=Path);p.add_argument('--arm',choices=ARMS);p.add_argument('--canonical',type=Path);p.add_argument('--stage',type=Path);p.add_argument('--cpu-report',type=Path);p.add_argument('--flash-root',type=Path);p.add_argument('--output',type=Path);p.add_argument('--smoke-stage',type=Path);a=p.parse_args()
    if a.mode=='prepare':prepare(a.root,a.arm,a.canonical,a.stage,a.cpu_report,a.manifest,a.flash_root)
    elif a.mode=='validate':validate(a.manifest);print('PASS_CPU_VARIANT')
    else:run(a.manifest,a.mode,a.output,a.smoke_stage)
if __name__=='__main__':main()

"""Calibration-only Flash labels and canonical-feature LightGBM export."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, random
from collections import Counter
from pathlib import Path
from scripts.prep.paper_ablation_data import BUCKETS, MODELS, rows, identity, sha256, write_json
from scripts.eval.judge_ablation import is_imputed


def verify_files(files):
    for path, digest in files.items():
        if sha256(path) != digest:
            raise ValueError(f'Frozen input changed: {path}')


def fresh(row):
    return {k:v for k,v in row.items() if not k.startswith(('quality','judge_'))}


def prepare(prepared, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    files={};counts={};keys={}
    for model in MODELS:
        keys[model]=set()
        for bucket in BUCKETS:
            source=Path(prepared)/'calibration'/model/f'{bucket}_scored.jsonl'
            records=list(rows(source));files[str(source.resolve())]=sha256(source)
            if len(records)!=2500 or any(identity(r)[0]!=bucket or r['model_label']!=model for r in records):
                raise ValueError('Incomplete calibration candidates')
            ids={identity(r) for r in records}
            if len(ids)!=2500:raise ValueError('Duplicate calibration identity')
            keys[model]|=ids
            dest=output/'inputs'/model/f'{bucket}.jsonl';dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('x')as f:
                for row in records:f.write(json.dumps(fresh(row),allow_nan=False)+'\n')
            files[str(dest.resolve())]=sha256(dest);counts[f'{model}:{bucket}']=len(records)
    if any(k!=keys[MODELS[0]] for k in keys.values()):raise ValueError('Unpaired calibration groups')
    write_json(output/'prepared.json',{'status':'PASS_CPU','data_role':'calibration','groups':10000,'counts':counts,'file_sha256':files,'prepared_source':str(Path(prepared).resolve()),'judge_model':'gemini-2.5-flash'})


def validate_labels(reference, candidate, *, require_grouped=False):
    left={(*identity(r),r['model_label']):r for r in reference}
    right={(*identity(r),r['model_label']):r for r in candidate}
    if len(left)!=len(reference) or len(right)!=len(candidate) or left.keys()!=right.keys():
        raise ValueError('Incomplete or duplicate Flash candidates')
    imputed=0;individual=0
    for key,a in left.items():
        b=right[key];q=b.get('quality')
        if a['prompt']!=b['prompt'] or a['response']!=b['response']:
            raise ValueError('Flash changed canonical prompt or response')
        if not isinstance(q,(float,int)) or not math.isfinite(q) or not 0<=q<=1:
            raise ValueError('Invalid Flash label')
        aliases=b.get('judge_alias_to_model');grouped=isinstance(aliases,dict)and set(aliases)=={'A','B','C'}and set(aliases.values())==set(MODELS)
        if is_imputed(b):imputed+=1
        elif not grouped:individual+=1
        if require_grouped and (not grouped or is_imputed(b)):raise ValueError('Canary requires observed grouped scores')
    return {'candidates':len(right),'imputed':imputed,'individual_without_group_alias':individual}


def judge(output, *, canary_only=False):
    from scripts.prep import quality_metrics as qm
    root=Path(output);manifest=json.loads((root/'prepared.json').read_text());verify_files(manifest['file_sha256'])
    qm.DEFAULT_JUDGE_MODEL=manifest['judge_model'];model=qm._get_gemini_client().models.get(model=qm.DEFAULT_JUDGE_MODEL)
    random.seed(69)
    report={};rubric=hashlib.sha256(qm.JUDGE_GROUP_SYSTEM_PROMPT.encode()).hexdigest()
    for bucket in BUCKETS:
        inputs={m:root/'inputs'/m/f'{bucket}.jsonl'for m in MODELS}
        outputs={m:root/'labels'/m/f'{bucket}_scored.jsonl'for m in MODELS}
        canaries={m:root/'canaries'/m/f'{bucket}.jsonl'for m in MODELS}
        for m in MODELS:
            outputs[m].parent.mkdir(parents=True,exist_ok=True);canaries[m].parent.mkdir(parents=True,exist_ok=True)
            if not canaries[m].exists():canaries[m].write_text(json.dumps(next(rows(inputs[m])))+'\n')
        if canary_only:
            summary=qm.annotate_bucket_group_with_quality(canaries,outputs,judge_concurrency=1,judge_retries=3,individual_retries=3)
            reference=[r for p in canaries.values()for r in rows(p)];candidate=[r for p in outputs.values()for r in rows(p)]
        else:
            if not (root/'canary_audit.json').exists():raise ValueError('API canary must pass before full judging')
            summary=qm.annotate_bucket_group_with_quality(inputs,outputs,judge_concurrency=20,judge_retries=3,individual_retries=3)
            reference=[r for p in inputs.values()for r in rows(p)];candidate=[r for p in outputs.values()for r in rows(p)]
        report[bucket]={'scorer':summary,**validate_labels(reference,candidate,require_grouped=canary_only)}
    verify_files(manifest['file_sha256'])
    result={'status':'PASS_API_CANARY'if canary_only else'PASS_CALIBRATION_LABELS','judge_model':qm.DEFAULT_JUDGE_MODEL,'model_resource':model.name,'rubric_sha256':rubric,'buckets':report,'data_role':'calibration','seed':69,'holdout_used':False,'file_sha256':{str(p.resolve()):sha256(p)for p in (root/'labels').glob('*/*_scored.jsonl')}}
    write_json(root/('canary_audit.json'if canary_only else'label_audit.json'),result)


def train(output, predictor_root):
    import lightgbm as lgb
    import numpy as np
    from vllm.v1.engine.accuracy_predictor import AccuracyPredictor
    from vllm.v1.engine.output_length_predictor import AdmissionFeatures
    from scripts.prep.train_serving_mlp import calibration_split
    root=Path(output);audit=json.loads((root/'label_audit.json').read_text());verify_files(audit['file_sha256'])
    prepared=json.loads((root/'prepared.json').read_text());verify_files(prepared['file_sha256'])
    canonical_train,canonical_test,split_sources=calibration_split(Path(prepared['prepared_source']),Path(predictor_root))
    labels={(*identity(r),r['model_label']):r for p in (root/'labels').glob('*/*_scored.jsonl')for r in rows(p)}
    # A common observed-label mask; imputed Pro or Flash scores never become training targets.
    def select(records):return [labels[(*identity(r),r['model_label'])]for r in records if not is_imputed(r)and not is_imputed(labels[(*identity(r),r['model_label'])])]
    train_rows,test_rows=select(canonical_train),select(canonical_test)
    if len(train_rows)<25000 or len(test_rows)<2500:raise ValueError('Excessive missing observed calibration labels; review before training')
    source=Path(predictor_root)/'accuracy_predictor';original=AccuracyPredictor(str(source));builder=original._feature_builder
    def features(rs):return np.asarray([builder.build_feature_row(AdmissionFeatures(r['model_label'],r['prompt'],r['prompt_tokens']))for r in rs],dtype=np.float32)
    x,xt=features(train_rows),features(test_rows);y=np.array([r['quality']for r in train_rows]);yt=np.array([r['quality']for r in test_rows])
    params={'objective':'huber','metric':['l2','l1'],'learning_rate':.05,'num_leaves':256,'min_data_in_leaf':50,'feature_fraction':.9,'bagging_fraction':.8,'bagging_freq':1,'max_depth':-1,'max_bin':63,'device_type':'cpu','num_threads':4,'verbosity':-1}
    booster=lgb.train(params,lgb.Dataset(x,label=y,feature_name=builder.feature_names,categorical_feature=[0]),num_boost_round=500)
    dest=root/'accuracy_predictor';dest.mkdir(exist_ok=False)
    booster.save_model(str(dest/'accuracy_model.txt'))
    metadata=copy.deepcopy(json.loads((source/'metadata.json').read_text()));metadata.update(train_examples=len(train_rows),test_examples=len(test_rows),predictor_backend='lightgbm')
    write_json(dest/'metadata.json',metadata)
    write_json(dest/'test_example_ids.json',json.loads((source/'test_example_ids.json').read_text()))
    loaded=AccuracyPredictor(str(dest));adm=[AdmissionFeatures(r['model_label'],r['prompt'],r['prompt_tokens'])for r in test_rows]
    expected=np.asarray(booster.predict(xt));actual=np.asarray(loaded.predict_batch(adm));np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=1e-7)
    verify_files(audit['file_sha256']);verify_files(split_sources)
    write_json(root/'training_audit.json',{'status':'PASS_CPU_EXPORT','judge_model':audit['judge_model'],'holdout_used':False,'params':params,'num_boost_round':500,'canonical_gpu_training_replaced_by_cpu':True,'preprocessing':'Frozen canonical feature metadata','train_candidates':len(train_rows),'validation_candidates':len(test_rows),'excluded_imputed_train':len(canonical_train)-len(train_rows),'split_policy':'Canonical reserved IDs and duplicate-text exclusion; common observed Pro/Flash labels','validation_mae':float(abs(actual-yt).mean()),'validation_rmse':float(np.sqrt(((actual-yt)**2).mean())),'reload_max_abs_difference':float(abs(actual-expected).max()),'source_sha256':{**split_sources,**audit['file_sha256'],str(source/'metadata.json'):sha256(source/'metadata.json'),str(Path(__file__).resolve()):sha256(__file__)},'artifact_sha256':{str(p.resolve()):sha256(p)for p in dest.iterdir()}})


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','canary','run','train']);p.add_argument('--output',type=Path,required=True);p.add_argument('--prepared',type=Path);p.add_argument('--predictor-root',type=Path)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.prepared,a.output)
    elif a.mode in ['canary','run']:judge(a.output,canary_only=a.mode=='canary')
    else:train(a.output,a.predictor_root)
if __name__=='__main__':main()

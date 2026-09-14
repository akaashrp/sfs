"""Verify portable inputs and workload identity, and materialize pinned models."""
import argparse
from collections import Counter
import csv
import importlib.util
import json
from pathlib import Path
import shutil

from scripts.cloud.common import ROOT, digest, read, write, expand, validate_bundle, source_hashes


def cpu(bundle, output):
    from scripts.cloud.worker import arguments, parse
    from scripts.runs import experiments as exp
    from scripts.runs.ministral3_methodology_stage import BUCKETS, smoke_requests
    from vllm.v1.engine.accuracy_predictor import AccuracyPredictor
    from vllm.v1.engine.output_length_predictor import OutputLengthPredictor, AdmissionFeatures
    from scripts.runs.qwen_predictor_variants import validate_predictions
    from sfs_core.routing.latency_warmup import load_warmup
    from sfs_core.routing.routebalance_predictor import RouteBalancePredictor
    bundle = Path(bundle).resolve()
    m = validate_bundle(bundle)
    if Path(importlib.util.find_spec('vllm').origin).resolve().parent.parent != ROOT/'vllm':
        raise ValueError('Active vLLM import is not the selected submodule')
    results = {}
    for family, f in m['families'].items():
        print(f'Checking {family}: frozen holdout, calibration, warm-up, and native predictors', flush=True)
        base = arguments(f, bundle, 'canonical', m)
        args = parse(base)
        requests, _, _ = exp._build_request_set(args)
        if len(requests) != f['requests'] or Counter(r.bucket for r in requests) != Counter({b: f['requests']//4 for b in BUCKETS}):
            raise ValueError('Incorrect holdout count or bucket balance')
        with Path(expand(f['request_map'], bundle, {})).open() as stream:
            mapped = {r['req_id']: r for r in csv.DictReader(stream)}
        cached = {(b, row['prompt_metadata']['example_id']): row for b in BUCKETS for row in
                  (json.loads(line) for line in (bundle/family/'holdout'/f'{b}.jsonl').read_text().splitlines())}
        if len(mapped) != len(requests): raise ValueError('Request map size mismatch')
        for req in requests:
            row = mapped[req.request_id]
            if (row['bucket'] != req.bucket or int(row['prompt_tokens']) != req.prompt_tokens
                    or cached[(row['bucket'], row['example_id'])]['prompt'] != req.prompt):
                raise ValueError('Workload differs from the canonical request map')
        calibration = [exp.ExperimentRequest(**json.loads(line)) for line in
                       (bundle/family/'calibration_requests.jsonl').read_text().splitlines()]
        if len(calibration) != 10000 or Counter(r.bucket for r in calibration) != Counter({b:2500 for b in BUCKETS}):
            raise ValueError('Incomplete calibration pool')
        warmup = load_warmup(bundle/family/'warmup.json')
        cal_identity = {(r.request_id, r.bucket, r.prompt) for r in calibration}
        if any((r['request_id'], r['bucket'], r['prompt']) not in cal_identity for r in warmup):
            raise ValueError('Warm-up is not selected from calibration')
        rb = RouteBalancePredictor.load(args.routebalance_predictor_path)
        rb.predict_batch(['Summarize why the sky appears blue.'])
        arms = ['canonical', *m['variants']] if family == 'qwen' else ['canonical']
        for arm in arms:
            argv = arguments(f, bundle, arm, m)
            other = parse(argv)
            actual, _, _ = exp._build_request_set(other)
            if actual != requests: raise ValueError('Predictor variant changed evaluation requests')
            quality = AccuracyPredictor(other.accuracy_model_path)
            length = OutputLengthPredictor(other.output_length_model_path)
            admissions = [AdmissionFeatures(model, r.prompt, r.prompt_tokens)
                          for r in smoke_requests(calibration) for model in f['models']]
            validate_predictions(quality.predict_batch(admissions), length.predict_batch(admissions),
                quality_backend=quality._backend, length_backend=length._backend)
        results[family] = {'requests': len(requests), 'calibration': len(calibration), 'warmup': len(warmup), 'variants': arms}
    write(output, {'status': 'PASS_CPU_INPUTS', 'bundle_sha256': digest(bundle/'bundle.json'), 'families': results,
        'source_sha256': source_hashes(), 'gpu_executed': False, 'cells': len(m['cells']), 'requests_total': m['requests_total']})


def serving(bundle, output):
    import asyncio
    from unittest.mock import patch
    from vllm.platforms.cpu import CpuPlatform
    from scripts.runs.serving_contract_preflight import validate_model
    bundle=Path(bundle).resolve();m=validate_bundle(bundle);reports=[]
    with patch('vllm.platforms._current_platform',CpuPlatform()):
        for family,f in m['families'].items():
            lengths=[bundle/family/'length']
            if family=='qwen':lengths.append(bundle/'variants/mlp_length')
            for length in lengths:
                for model in f['models']:
                    print(f'Checking real chat admission and scheduler: {model}, {length.name}',flush=True)
                    reports.append(asyncio.run(validate_model(ROOT,bundle/'tokenizers'/model,length,
                        'qwen' if family=='qwen' else 'ministral3')))
    write(output,{'status':'PASS_CPU_SERVING','models':reports,'gpu_executed':False,
        'bundle_sha256':digest(bundle/'bundle.json'),'source_sha256':source_hashes()})


def seed_encoder(bundle, cache):
    m = read(Path(bundle)/'bundle.json'); encoder = m['encoder']
    directory = Path(cache)/('models--'+encoder['model_id'].replace('/', '--'))/'snapshots'/encoder['revision']
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.resolve() == (Path(bundle)/'encoder').resolve(): return
    shutil.copytree(Path(bundle)/'encoder', directory, dirs_exist_ok=True)


def download(bundle, cache, output, family):
    from huggingface_hub import snapshot_download
    m = validate_bundle(bundle)
    seed_encoder(bundle, cache)
    paths = {}
    for key, model in m['models'].items():
        if family != 'all' and model['family'] != family: continue
        patterns = ['*.json', '*.jinja', '*.model', '*.tiktoken', 'tekken.json']
        patterns += ['consolidated.safetensors'] if model['family'] == 'ministral' else ['*.safetensors']
        path = Path(snapshot_download(repo_id=model['repo'], revision=model['revision'],
            cache_dir=cache, allow_patterns=patterns))
        if not list(path.glob('*.safetensors')): raise ValueError('Missing model weights')
        index = path/'model.safetensors.index.json'
        if model['family'] == 'qwen' and index.exists():
            for shard in set(read(index)['weight_map'].values()):
                if not (path/shard).is_file(): raise ValueError('Incomplete model shard set')
        paths[key] = str(path)
    write(output, paths)
    write(str(output)+'.audit.json', {'revisions': {k:m['models'][k] for k in paths},
        'file_sha256': {str(p): digest(p) for root in paths.values() for p in Path(root).iterdir() if p.is_file()}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['cpu', 'models', 'encoder', 'serving'])
    parser.add_argument('--bundle', required=True); parser.add_argument('--output')
    parser.add_argument('--cache'); parser.add_argument('--family', choices=['all', 'qwen', 'ministral'], default='all')
    args = parser.parse_args()
    if args.mode == 'cpu': cpu(args.bundle, args.output)
    elif args.mode == 'serving': serving(args.bundle,args.output)
    elif args.mode == 'models': download(args.bundle, args.cache, args.output, args.family)
    else: seed_encoder(args.bundle, args.cache)

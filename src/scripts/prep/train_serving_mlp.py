"""Train calibration-only MLPs and export portable serving artifacts on CPU."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np

from scripts.prep.paper_ablation_data import BUCKETS, MODELS, identity, rows, sha256, write_json


def export_pipeline(pipeline, metadata, target, output):
    """Export numeric scaler/MLP arrays; no Python pickle is used in serving."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    scaler, model = pipeline.steps[0][1], pipeline.steps[1][1]
    if model.activation != 'relu' or model.out_activation_ != 'identity':
        raise ValueError('Only ReLU regression MLPs are supported')
    arrays = {'scaler_mean': scaler.mean_, 'scaler_scale': scaler.scale_}
    for i, (weight, bias) in enumerate(zip(model.coefs_, model.intercepts_)):
        arrays[f'weight_{i}'] = weight
        arrays[f'bias_{i}'] = bias
    path = output / 'mlp_weights.npz'
    np.savez(path, **arrays)
    meta = copy.deepcopy(metadata)
    meta.update(predictor_backend='numpy_mlp', mlp={
        'format_version': 1, 'activation': 'relu', 'output_activation': 'identity',
        'file': path.name, 'sha256': sha256(path), 'target': target,
        'layer_sizes': [model.coefs_[0].shape[0]] + [w.shape[1] for w in model.coefs_]})
    if target == 'quality':
        meta['target_link'] = 'identity'
    elif target == 'output_tokens':
        meta['mean_model_target'] = 'log1p_tokens'
        meta['mean_model_file'] = path.name
    else:
        raise ValueError('Unknown target')
    write_json(output / 'metadata.json', meta)


def calibration_split(prepared, predictor_root):
    reserved_ids = set()
    sources = {}
    for name in ('accuracy_predictor', 'output_length_predictor'):
        p = predictor_root / name / 'test_example_ids.json'
        sources[str(p)] = sha256(p)
        reserved_ids.update(json.loads(p.read_text())['example_ids'])
    records = []
    for model in MODELS:
        for bucket in BUCKETS:
            p = prepared / 'calibration' / model / f'{bucket}_scored.jsonl'
            sources[str(p)] = sha256(p)
            records.extend(rows(p))
    def digest(r):
        return hashlib.sha256(r['prompt'].encode()).hexdigest()
    reserved_text = {digest(r) for r in records if identity(r)[1] in reserved_ids}
    train = [r for r in records if identity(r)[1] not in reserved_ids and digest(r) not in reserved_text]
    validation = [r for r in records if identity(r)[1] in reserved_ids]
    if len(records) != 30000 or not train or not validation:
        raise ValueError('Incomplete canonical calibration split')
    if len({(*identity(r), r['model_label']) for r in records}) != len(records):
        raise ValueError('Duplicate calibration candidate')
    return train, validation, sources


def run(prepared, predictor_root, output):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    from vllm.v1.engine.accuracy_predictor import AccuracyPredictor
    from vllm.v1.engine.output_length_predictor import OutputLengthPredictor, AdmissionFeatures
    import vllm
    expected = Path(__file__).resolve().parents[3] / 'vllm/vllm'
    if Path(vllm.__file__).resolve().parent != expected:
        raise ValueError('Wrong editable vLLM import: ' + str(vllm.__file__))
    if output.exists():
        raise ValueError('Preserve prior artifacts; choose a new output root')
    train, validation, sources = calibration_split(prepared, predictor_root)
    output.mkdir(parents=True)
    metrics = {}
    for target, name, cls, attr in (
        ('quality', 'accuracy_predictor', AccuracyPredictor, '_feature_builder'),
        ('output_tokens', 'output_length_predictor', OutputLengthPredictor, '_feature_extractor')):
        source = predictor_root / name
        metadata_path = source / 'metadata.json'
        metadata = json.loads(metadata_path.read_text())
        sources[str(metadata_path)] = sha256(metadata_path)
        metadata.update(train_examples=len(train), test_examples=len(validation),
                        test_example_ids_file='test_example_ids.json',
                        feature_source_metadata_sha256=sources[str(metadata_path)])
        original = cls(str(source))
        builder = getattr(original, attr)
        def features(records):
            admissions = [AdmissionFeatures(r['model_label'], r['prompt'], r['prompt_tokens']) for r in records]
            # Explicit float64 ensures training/export/runtime share scaler arithmetic.
            return admissions, np.asarray([builder.build_feature_row(a) for a in admissions], dtype=np.float64)
        _, x = features(train)
        admissions, xv = features(validation)
        def targets(records):
            y = np.asarray([r['quality'] if target == 'quality' else r['response']['completion_tokens'] for r in records], float)
            if not np.isfinite(y).all() or np.any(y < 0) or np.any(y > (1 if target == 'quality' else 8192)):
                raise ValueError('Invalid training/validation target')
            return y
        y, yv = targets(train), targets(validation)
        model = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32),
            max_iter=500, batch_size=256, random_state=69, early_stopping=False, tol=1e-4))
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
            warnings.simplefilter('always', ConvergenceWarning)
            start = time.perf_counter()
            model.fit(x, np.log1p(y) if target == 'output_tokens' else y)
            training_s = time.perf_counter() - start
            pred = model.predict(xv)
        expected_prediction = (np.clip(pred, 0, 1) if target == 'quality'
                               else np.clip(np.expm1(np.clip(pred, 0, np.log1p(8192))), 1, 8192))
        artifact = output / name
        export_pipeline(model, metadata, target, artifact)
        write_json(artifact / 'test_example_ids.json', {
            'example_ids': sorted({identity(r)[1] for r in validation}),
            'source': 'Union of canonical reserved calibration IDs'})
        loaded = cls(str(artifact))
        with threadpool_limits(limits=1):
            actual = loaded.predict_batch(admissions)
            if target == 'output_tokens':
                actual = [r.mean_tokens for r in actual]
            np.testing.assert_allclose(actual, expected_prediction, rtol=1e-10, atol=1e-8)
            # Real feature extraction and loaded batch-of-three inference, on CPU.
            loaded.predict_batch(admissions[:3])
            start = time.perf_counter()
            for _ in range(100):
                loaded.predict_batch(admissions[:3])
            batch3_ms = (time.perf_counter() - start) * 10
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        error = np.asarray(actual) - yv
        metrics[target] = {'train_candidates': len(train), 'validation_candidates': len(validation),
            'iterations': model.steps[-1][1].n_iter_, 'converged': converged,
            'training_seconds': training_s, 'calibration_validation_mae': float(abs(error).mean()),
            'calibration_validation_rmse': float(np.sqrt((error**2).mean())),
            'export_reload_max_abs_difference': float(np.max(abs(np.asarray(actual)-expected_prediction))),
            'cpu_batch3_features_and_inference_mean_ms': batch3_ms,
            'validation_used_for_tuning': False}
        print(json.dumps({target: metrics[target]}), flush=True)
    for path, digest in sources.items():
        if sha256(path) != digest:
            raise ValueError('Calibration source changed during training')
    write_json(output / 'serving_mlp_audit.json', {
        'status': 'PASS_CPU_EXPORT' if all(m['converged'] for m in metrics.values()) else 'CPU_EXPORT_PASS_CONVERGENCE_REVIEW_REQUIRED',
        'targets': metrics, 'source_sha256': sources, 'training_script_sha256': sha256(__file__),
        'holdout_read_or_used': False, 'gpu_smoke_passed': False,
        'training_imputed_candidates': sum(bool(r.get('quality_imputed') or r.get('quality_metric')=='judge_default_bucket_mean') for r in train),
        'split_policy': 'Canonical reserved calibration IDs; exclude their duplicate prompt texts from training; no final holdout access',
        'vllm_import': vllm.__file__})
    write_json(output / 'serving_variants.json', {
        'mlp_quality': {'router_accuracy_model_path': str(output / 'accuracy_predictor'),
                        'router_output_length_model_path': str(predictor_root / 'output_length_predictor'),
                        'server_output_length_model_path': str(predictor_root / 'output_length_predictor')},
        'mlp_output_length': {'router_accuracy_model_path': str(predictor_root / 'accuracy_predictor'),
                              'router_output_length_model_path': str(output / 'output_length_predictor'),
                              'server_output_length_model_path': str(output / 'output_length_predictor')},
        'gpu_gate': 'All server instances and router must report matching selected artifact hashes; GPU smoke required'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--predictor-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    run(args.prepared_dir.resolve(), args.predictor_root.resolve(), args.output_root.resolve())

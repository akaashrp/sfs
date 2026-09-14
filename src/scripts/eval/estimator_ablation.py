"""CPU scalar estimator comparison using canonical Qwen features and holdouts."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from scripts.prep.paper_ablation_data import BUCKETS, MODELS, identity, rows, sha256, validate, write_json

PRICES = {"qwen3-0.6b": (.044, .173), "qwen3-8b": (.072, .287), "qwen3-32b": (.287, .64)}


def split_data(prepared, predictor_root):
    train, test = [], []
    excluded_ids = set()
    for name in ("accuracy_predictor", "output_length_predictor"):
        data = json.loads((Path(predictor_root)/name/"test_example_ids.json").read_text())
        excluded_ids.update(data["example_ids"])
    for role, dest in (("calibration", train), ("holdout", test)):
        for model in MODELS:
            for bucket in BUCKETS:
                dest.extend(rows(Path(prepared)/role/model/f"{bucket}_scored.jsonl"))
    text_hash = lambda row: hashlib.sha256(row["prompt"].encode()).hexdigest()
    reserved_text = {text_hash(row) for row in train if identity(row)[1] in excluded_ids}
    calibration_text = {text_hash(row) for row in train}
    filtered_train = [r for r in train if identity(r)[1] not in excluded_ids and text_hash(r) not in reserved_text]
    # Even a new example_id can contain duplicated calibration text. Exclude
    # those evaluation groups for every estimator, including the saved SFS model.
    filtered_test = [r for r in test if text_hash(r) not in calibration_text]
    if not filtered_train or not filtered_test:
        raise ValueError("Empty estimator train/test split")
    audit = {"train_candidates": len(filtered_train), "evaluation_candidates": len(filtered_test),
             "evaluation_queries": len(filtered_test)//3, "requested_evaluation_queries": len(test)//3,
             "excluded_duplicate_text_queries": (len(test)-len(filtered_test))//3,
             "calibration_validation_ids_excluded": len(excluded_ids), "test_used_for_tuning": False}
    return filtered_train, filtered_test, audit


def errors(target, prediction):
    import numpy as np
    a, b = np.asarray(target, dtype=float), np.asarray(prediction, dtype=float)
    if a.shape != b.shape or a.size == 0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Missing or nonfinite estimator predictions")
    delta = a-b
    return {"n": len(a), "mae": float(np.abs(delta).mean()), "rmse": float(np.sqrt((delta**2).mean()))}


def grouped_errors(records, target, prediction):
    groups = defaultdict(list)
    for i, row in enumerate(records):
        for group in ("overall", "model:"+row["model_label"], "bucket:"+row["bucket"],
                      row["model_label"]+":"+row["bucket"]):
            groups[group].append(i)
    return {group: errors(target[index], prediction[index]) for group, index in groups.items()}


def run(prepared, predictor_root, output, *, validate_only=False):
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from vllm.v1.engine.accuracy_predictor import AccuracyPredictor
    from vllm.v1.engine.output_length_predictor import OutputLengthPredictor, AdmissionFeatures

    data_audit = validate(prepared)
    train, test, split_audit = split_data(prepared, predictor_root)
    if validate_only:
        def small(records):
            return [r for model in MODELS for bucket in BUCKETS for r in
                    [row for row in records if row["model_label"] == model and row["bucket"] == bucket][:4]]
        train, test = small(train), small(test)
    root = Path(output)
    if root.exists():
        raise ValueError("Choose a new estimator output root")
    quality = AccuracyPredictor(str(Path(predictor_root)/"accuracy_predictor"))
    length = OutputLengthPredictor(str(Path(predictor_root)/"output_length_predictor"))
    predictors = {"quality": quality, "output_tokens": length}
    results, predictions = {}, {}
    for target in ("quality", "output_tokens"):
        predictor = predictors[target]
        builder = predictor._feature_builder if target == "quality" else predictor._feature_extractor
        def features(records):
            # Match the deployed router's AdmissionFeatures, including chat
            # template tokens. Every alternative receives these same features.
            return [AdmissionFeatures(r["model_label"], r["prompt"], r["prompt_tokens"])
                    for r in records]
        train_adm, test_adm = features(train), features(test)
        x = np.vstack([builder.build_feature_row(a) for a in train_adm])
        xt = np.vstack([builder.build_feature_row(a) for a in test_adm])
        records_train, records_test = train[:len(x)], test[:len(xt)]
        y = np.asarray([r["quality"] if target == "quality" else r["response"]["completion_tokens"] for r in records_train])
        yt = np.asarray([r["quality"] if target == "quality" else r["response"]["completion_tokens"] for r in records_test])
        if target == "quality":
            existing = np.asarray(predictor.predict_batch(test_adm))
        else:
            existing = np.asarray([r.mean_tokens for r in predictor.predict_batch(test_adm)])
        alternatives = {
            "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            "mlp": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32),
                max_iter=500, random_state=69, early_stopping=False, tol=1e-4, batch_size=256)),
            "xgboost": XGBRegressor(n_estimators=300, max_depth=6, learning_rate=.05,
                objective="reg:squarederror", tree_method="hist", n_jobs=4, random_state=69),
            "lightgbm_refit": LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=.05,
                n_jobs=4, random_state=69, verbosity=-1),
        }
        if validate_only:
            alternatives["mlp"].steps[-1][1].set_params(max_iter=2, batch_size=16)
            alternatives["xgboost"].set_params(n_estimators=2)
            alternatives["lightgbm_refit"].set_params(n_estimators=2)
        raw_predictions = {"sfs_existing": existing}
        # Mean/median baselines condition only on known model/bucket identifiers.
        groups = defaultdict(list)
        for i, r in enumerate(records_train):
            groups[(r["model_label"], r["bucket"])].append(i)
        for method, reducer in (("mean", np.mean), ("median", np.median)):
            raw_predictions[method] = np.asarray([reducer(y[groups[(r["model_label"], r["bucket"])]])
                for r in records_test])
        for name, model in alternatives.items():
            fit_target = np.log1p(y) if target == "output_tokens" else y
            model.fit(x, fit_target)
            value = model.predict(xt)
            raw_predictions[name] = np.expm1(np.clip(value, 0, np.log1p(8192))) if target == "output_tokens" else value
        predictions[target], results[target] = {}, {}
        for name, values in raw_predictions.items():
            values = np.clip(values, 0, 1) if target == "quality" else np.clip(values, 1, 8192)
            predictions[target][name] = values
            results[target][name] = grouped_errors(records_test, yt, values)
        if target == "output_tokens":
            prompt = np.asarray([r["prompt_tokens"] for r in records_test])
            prices = np.asarray([PRICES[r["model_label"]] for r in records_test])
            actual_cost = (prompt*prices[:, 0]+yt*prices[:, 1])/1e6
            results["cost_usd"] = {name: grouped_errors(records_test, actual_cost,
                (prompt*prices[:, 0]+values*prices[:, 1])/1e6)
                for name, values in predictions[target].items()}
    report = {"status": "PASS", "validation_only": validate_only,
        "split": split_audit, "metrics": results, "training_imputed_candidates": data_audit["counts"].get("calibration_imputed", 0),
        "features": "Saved canonical SFS feature builders, shared by all alternatives within each target",
        "length_training_target": "log1p(tokens) for refitted learners; mean/median and existing SFS use their native targets",
        "clipping": "quality [0,1], output length [1,8192] for deployment comparison",
        "cost_units": "USD per request; canonical model input/output token prices",
        "hyperparameter_selection": "Fixed before evaluation; no holdout tuning",
        "prepared_data_audit_sha256": sha256(Path(prepared)/"data_audit.json"),
        "predictor_files_sha256": {str(p): sha256(p) for name in predictors for p in
             (Path(predictor_root)/("accuracy_predictor" if name == "quality" else "output_length_predictor")).glob("*") if p.is_file()},
        "gpu_executed": False}
    root.mkdir(parents=True)
    write_json(root/"estimator_metrics.json", report)
    import csv
    with (root/"estimator_metrics.csv").open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=("target", "estimator", "group", "n", "mae", "rmse"))
        writer.writeheader()
        for target, methods in results.items():
            for method, groups in methods.items():
                for group, values in groups.items():
                    writer.writerow(dict(target=target, estimator=method, group=group, **values))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared-dir", type=Path, required=True)
    p.add_argument("--predictor-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--validate-only", action="store_true")
    a = p.parse_args()
    result = run(a.prepared_dir, a.predictor_root, a.output_root, validate_only=a.validate_only)
    print(json.dumps({k: result[k] for k in ("status", "validation_only", "split")}))


if __name__ == "__main__":
    main()

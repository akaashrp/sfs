"""Freeze portable inputs from the reviewed Bridges campaign, without editing it."""
import argparse
import json
from pathlib import Path
import shutil

from scripts.cloud.common import digest, read, write, set_option
from scripts.cloud.schedule import QWEN_QPS


def export(source, output, wheel):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Choose a new bundle output")
    output.mkdir(parents=True)
    provenance = {}

    def copy(src, dest):
        src, dest = Path(src), output / dest
        if src.is_dir():
            for child in sorted(src.rglob("*")):
                if child.is_file() and "__pycache__" not in child.parts:
                    copy(child, dest.relative_to(output) / child.relative_to(src))
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            provenance[str(dest.relative_to(output))] = {"source": str(src), "sha256": digest(src)}
            if digest(dest) != provenance[str(dest.relative_to(output))]["sha256"]:
                raise ValueError(f"Copy changed: {src}")
        return "@BUNDLE@/" + str(dest.relative_to(output))

    qpath = source / "experiments/vllm_sr_latency_20260911/qwen/qwen_manifest.json"
    mpath = source / "experiments/campaign_cold_start_fix_20260912/ministral_figure5_manifest.json"
    q, m = read(qpath), read(mpath)
    families = {}
    models = {}
    from scripts.runs import qwen_baselines as qw
    from scripts.runs import ministral3_latency as ml
    from scripts.prep.prepare_methodology_service import PROFILE as MPROFILE
    for family, manifest in (("qwen", q), ("ministral", m)):
        copy(qpath if family == "qwen" else mpath, f"provenance/{family}_bridges_manifest.json")
        argv = list(manifest["experiment_argv"])
        def option(flag):
            return argv[len(argv) - 1 - argv[::-1].index(flag) + 1]
        cache = copy(option("--holdout-cache-dir"), f"{family}/holdout")
        argv = set_option(argv, "--bucket-dir", cache)
        argv = set_option(argv, "--holdout-cache-dir", cache)
        # Existing cache is mandatory; never rebuild final prompts on a cloud host.
        argv = set_option(argv, "--holdout-bucket-dir", cache)
        for flag, name in (("--accuracy-model-path", "quality"), ("--output-length-model-path", "length")):
            argv = set_option(argv, flag, copy(option(flag), f"{family}/{name}"))
        if family == "qwen":
            calibration = q["calibration_requests"]
            scored = Path(q["prepared_data"]) / "holdout"
            rb = source / "experiments/paper_ablation_20260907/qwen_routebalance_predictor"
            metrics = source / "experiments/campaign_cold_start_fix_20260912/qwen_smoke/model_metrics.json"
            model_ids, pins = list(qw.MODELS), qw.PINS
            repos = ["Qwen/" + name for name in qw.HF_NAMES]
            profile, rates, policies = qw.PROFILE, list(QWEN_QPS), list(qw.NEW_POLICIES)
            warmup = q["latency_warmup"]
        else:
            stage = Path(m["stage_dir"])
            prepared = Path(read(stage / "stage_started.json")["prepared_dir"])
            calibration = prepared / "requests.jsonl"
            scored, rb = Path(m["scored_root"]), Path(option("--routebalance-predictor-path"))
            metrics = Path(m["service_run_dir"]) / "model_metrics.json"
            model_ids = list(ml.ALIASES)
            pins = ["b6d637bef2393152b3da2b2fde72eecdee30557e", "f6fae9795746f63c9be8344932f01275f3c63734", "3cea74c1ebaf5ce5f5a2553de470e2ceab825142"]
            repos = [f"mistralai/Ministral-3-{n}B-Instruct-2512-BF16" for n in (3, 8, 14)]
            profile, rates, policies = MPROFILE, list(ml.QPS), list(ml.POLICIES)
            from scripts.runs.experiments import ExperimentRequest
            requests = [ExperimentRequest(**json.loads(line)) for line in Path(calibration).read_text().splitlines()]
            warmup = output / "ministral/latency_warmup.json"
            ml.write_warmup(requests, warmup)
        for index, (mid, pin, repo) in enumerate(zip(model_ids, pins, repos)):
            snapshot = source.parent / ".cache/huggingface/hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / pin
            models[mid] = {"repo": repo, "revision": pin, "family": family}
            # Tokenizers are available for offline CPU rehearsal before weights download.
            for path in snapshot.iterdir():
                if path.is_file() and not path.name.endswith((".safetensors", ".bin", ".pt")):
                    copy(path, f"tokenizers/{mid}/{path.name}")
        argv = set_option(argv, "--tokenizer-id", f"@BUNDLE@/tokenizers/{model_ids[1]}")
        argv = set_option(argv, "--latency-warmup-requests", copy(warmup, f"{family}/warmup.json"))
        argv = set_option(argv, "--service-metrics-json", copy(metrics, f"{family}/bridges_metrics.json"))
        argv = set_option(argv, "--routebalance-predictor-path", copy(rb, f"{family}/routebalance"))
        if "--methodology-calibration-json" in argv:
            # This placeholder can only be supplied by destination qualification.
            argv = set_option(argv, "--methodology-calibration-json", "@QUALIFIED_TIMING@")
        copy(calibration, f"{family}/calibration_requests.jsonl")
        copy(manifest["request_map"], f"{family}/request_map.csv")
        for mid in model_ids:
            for bucket in qw.BUCKETS:
                copy(scored / mid / (bucket + "_scored.jsonl"), f"{family}/scores/{mid}/{bucket}_scored.jsonl")
        families[family] = {"experiment_argv": argv, "models": model_ids, "profile": profile,
            "qps": rates, "policies": policies, "requests": 16000 if family == "qwen" else 8000,
            "calibration_requests": f"@BUNDLE@/{family}/calibration_requests.jsonl",
            "request_map": f"@BUNDLE@/{family}/request_map.csv", "scored_root": f"@BUNDLE@/{family}/scores",
            "serving_calibration_origin": "Bridges; must qualify on destination before evaluation"}
    variants = {}
    for arm, path in {
        "mlp_quality": source / "experiments/mlp_serving_20260911/trained/accuracy_predictor",
        "mlp_length": source / "experiments/mlp_serving_20260911/trained/output_length_predictor",
        "flash_quality": source / "experiments/predictor_prereqs_20260912/flash/accuracy_predictor",
    }.items():
        variants[arm] = copy(path, f"variants/{arm}")
    for rel in ("experiments/mlp_serving_20260911/trained/serving_mlp_audit.json",
                "experiments/predictor_prereqs_20260912/flash/training_audit.json"):
        copy(source / rel, "provenance/" + Path(rel).name)
    flash_scores = source/'experiments/paper_ablation_20260907/judge_flash'
    for model in qw.MODELS:
        for bucket in qw.BUCKETS:
            copy(flash_scores/model/f'{bucket}_scored.jsonl', f'qwen/scores_flash/{model}/{bucket}_scored.jsonl')
    for name in ('judge_run_summary.json', 'comparison.json'):
        copy(flash_scores/name, 'provenance/flash_holdout_'+name)
    encoder = read(output / "qwen/routebalance/metadata.json")["encoder"]
    encoder_cache = source.parent / ".cache/huggingface/hub" / ("models--" + encoder["model_id"].replace("/", "--"))
    copy(encoder_cache / "snapshots" / encoder["revision"], "encoder")
    copy(wheel, "runtime/vllm.whl")
    cells = []
    for family, f in families.items():
        for policy in f["policies"]:
            for rate in f["qps"]:
                cells.append({"id": f"{family}-{policy}-{rate:g}", "family": family, "variant": "canonical",
                    "policy": policy, "qps": rate, "requests": f["requests"]})
    for arm in variants:
        for rate in QWEN_QPS:
            cells.append({"id": f"qwen-{arm}-{rate:g}", "family": "qwen", "variant": arm,
                "policy": "hard", "qps": rate, "requests": 16000})
    files = {str(p.relative_to(output)): digest(p) for p in output.rglob("*") if p.is_file()}
    write(output / "bundle.json", {"schema_version": 1, "families": families, "models": models,
        "variants": variants, "cells": cells, "files": files, "source_provenance": provenance,
        "encoder": encoder, "requests_total": sum(c["requests"] for c in cells),
        "qualification": "fresh destination calibration, all-policy smoke, load probes, reviewed release"})
    print(json.dumps({"bundle": str(output), "cells": len(cells), "bytes": sum(p.stat().st_size for p in output.rglob('*') if p.is_file())}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wheel", required=True)
    args = parser.parse_args()
    export(args.source, args.output, args.wheel)

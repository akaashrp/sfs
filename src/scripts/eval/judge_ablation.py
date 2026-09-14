"""Rejudge saved Qwen candidates and compare strictly paired, observed scores."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

from scripts.prep.paper_ablation_data import BUCKETS, MODELS, identity, rows, sha256, validate, write_json


def is_imputed(row):
    return bool(row.get("quality_imputed") or row.get("quality_metric") == "judge_default_bucket_mean")


def paired_summary(reference, candidate):
    """Average candidate disagreement within each complete query, then queries."""
    indexes = []
    for records in (reference, candidate):
        index = {}
        for row in records:
            key = (*identity(row), row["model_label"])
            if key in index:
                raise ValueError(f"Duplicate judged candidate: {key}")
            if row["model_label"] not in MODELS:
                raise ValueError("Unexpected model label")
            value = row.get("quality")
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Invalid normalized judge score")
            index[key] = row
        indexes.append(index)
    left, right = indexes
    if set(left) != set(right):
        raise ValueError("Judge candidates must have exactly matching canonical keys")
    differences, distributions = defaultdict(dict), defaultdict(lambda: {"pro": [], "flash": []})
    excluded = Counter()
    pairs = []
    for key in sorted(left):
        a, b = left[key], right[key]
        if a["prompt"] != b["prompt"] or a["response"]["output_text"] != b["response"]["output_text"]:
            raise ValueError(f"Judge comparison changed the prompt or generated response: {key}")
        if is_imputed(a) or is_imputed(b):
            excluded["imputed_candidate_pairs"] += 1
            continue
        aliases = b.get("judge_alias_to_model")
        if not isinstance(aliases, dict) or set(aliases) != {"A", "B", "C"} or set(aliases.values()) != set(MODELS):
            raise ValueError("Flash result lacks a complete realized alias mapping")
        delta = abs(a["quality"]-b["quality"])
        differences[key[:2]][key[2]] = delta
        pairs.append({"bucket": key[0], "example_id": key[1], "model_label": key[2],
                      "pro": a["quality"], "flash": b["quality"], "absolute_disagreement": delta,
                      "flash_alias_to_model": aliases, "pro_alias_to_model": a.get("judge_alias_to_model")})
        for grouping in ("overall", f"model:{key[2]}", f"bucket:{key[0]}"):
            distributions[grouping]["pro"].append(a["quality"])
            distributions[grouping]["flash"].append(b["quality"])
    complete = {key: statistics.mean(values.values()) for key, values in differences.items()
                if set(values) == set(MODELS)}
    if not complete:
        raise ValueError("No complete non-imputed query groups for judge comparison")
    def describe(values):
        hist = [0]*11
        for value in values:
            hist[min(10, int(math.floor(value*10+0.5))) ] += 1
        return {"count": len(values), "mean": statistics.mean(values),
                "std": statistics.pstdev(values), "histogram_scores_0_to_1_step_0p1": hist}
    summary = {"status": "PASS", "paired_candidates": len(pairs),
        "complete_queries": len(complete), "expected_queries": len(left)//3,
        "excluded_queries": len(left)//3-len(complete), "exclusions": dict(excluded),
        "mean_per_query_absolute_disagreement": statistics.mean(complete.values()),
        "per_bucket_mean_per_query_disagreement": {b: statistics.mean(
            v for (bucket, _), v in complete.items() if bucket == b)
            for b in BUCKETS if any(key[0] == b for key in complete)},
        "distributions": {g: {judge: describe(v) for judge, v in values.items()}
                          for g, values in distributions.items()},
        "interpretation": "Observed disagreement includes candidate presentation order and judge sampling variance; imputed pairs are excluded."}
    return summary, pairs


def compare(prepared, output):
    reference, candidate = [], []
    files = {}
    for model in MODELS:
        for bucket in BUCKETS:
            a = Path(prepared)/"holdout"/model/f"{bucket}_scored.jsonl"
            b = Path(output)/model/f"{bucket}_scored.jsonl"
            reference.extend(rows(a)); candidate.extend(rows(b))
            files[str(a)] = sha256(a); files[str(b)] = sha256(b)
    summary, pairs = paired_summary(reference, candidate)
    summary["file_sha256"] = files
    write_json(Path(output)/"comparison.json", summary)
    with (Path(output)/"paired_scores.jsonl").open("x") as stream:
        for pair in pairs:
            stream.write(json.dumps(pair)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, ax = plt.subplots(figsize=(6, 4))
    for label in ("pro", "flash"):
        d = summary["distributions"]["overall"][label]
        ax.plot([i/10 for i in range(11)], [n/d["count"] for n in d["histogram_scores_0_to_1_step_0p1"]],
                marker="o", label=label.title())
    ax.set(xlabel="Judge quality score", ylabel="Fraction of paired candidates")
    ax.legend(); figure.tight_layout()
    figure.savefig(Path(output)/"judge_distributions.png", dpi=180)
    plt.close(figure)
    return summary


def run(prepared, output, model="gemini-2.5-flash", concurrency=20):
    from scripts.prep import quality_metrics as qm
    audit = validate(prepared)
    root = Path(output).resolve()
    if root.exists():
        raise ValueError("Choose a new judge output root; preserve existing scores")
    # Model lookup and four real grouped calls precede the full campaign. The
    # successful canary scores are reused by the existing resume-aware scorer.
    qm.DEFAULT_JUDGE_MODEL = model
    available = qm._get_gemini_client().models.get(model=model)
    root.mkdir(parents=True)
    write_json(root/"run_started.json", {"judge_model": model, "model_resource": available.name,
        "prepared_audit_sha256": sha256(Path(prepared)/"data_audit.json"),
        "expected_groups": audit["holdout_prompt_groups"], "concurrency": concurrency,
        "group_retries": 3, "individual_retries": 3,
        "rubric_sha256": __import__("hashlib").sha256(qm.JUDGE_GROUP_SYSTEM_PROMPT.encode()).hexdigest()})
    summaries, bucket_io = {}, {}
    for bucket in BUCKETS:
        inputs, outputs, canaries = {}, {}, {}
        for candidate in MODELS:
            inputs[candidate] = Path(prepared)/"judge_inputs"/candidate/f"{bucket}.jsonl"
            outputs[candidate] = root/candidate/f"{bucket}_scored.jsonl"
            outputs[candidate].parent.mkdir(parents=True, exist_ok=True)
            canaries[candidate] = root/"canary_inputs"/candidate/f"{bucket}.jsonl"
            canaries[candidate].parent.mkdir(parents=True, exist_ok=True)
            with canaries[candidate].open("x") as stream:
                stream.write(json.dumps(next(rows(inputs[candidate])))+"\n")
        qm.annotate_bucket_group_with_quality(canaries, outputs, judge_concurrency=1,
                                              judge_retries=3, individual_retries=3)
        for path in outputs.values():
            canary = list(rows(path))
            if len(canary) != 1 or is_imputed(canary[0]) or not canary[0].get("judge_alias_to_model"):
                raise ValueError("Judge API canary did not produce observed grouped scores")
        bucket_io[bucket] = (inputs, outputs)
    for bucket, (inputs, outputs) in bucket_io.items():
        summaries[bucket] = qm.annotate_bucket_group_with_quality(inputs, outputs,
            judge_concurrency=concurrency, judge_retries=3, individual_retries=3)
    write_json(root/"judge_run_summary.json", {"judge_model": model, "buckets": summaries})
    return compare(prepared, root)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("validate", "run", "compare"))
    p.add_argument("--prepared-dir", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--judge-model", default="gemini-2.5-flash")
    p.add_argument("--concurrency", type=int, default=20)
    a = p.parse_args()
    if a.concurrency < 1:
        p.error("Concurrency must be positive")
    if a.mode == "validate":
        audit = validate(a.prepared_dir)
        if a.output_root.exists():
            raise ValueError("Judge output already exists")
        print(json.dumps({"status": "PASS", "groups": audit["holdout_prompt_groups"], "api_executed": False}))
    else:
        result = run(a.prepared_dir, a.output_root, a.judge_model, a.concurrency) if a.mode == "run" else compare(a.prepared_dir, a.output_root)
        print(json.dumps({k: result[k] for k in ("status", "complete_queries", "mean_per_query_absolute_disagreement")}))


if __name__ == "__main__":
    main()

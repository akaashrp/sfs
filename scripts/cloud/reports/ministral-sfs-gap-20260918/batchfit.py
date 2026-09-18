import csv, json, math, sys
from statistics import median

path, coefs_json, label = sys.argv[1], sys.argv[2], sys.argv[3]
C = json.loads(coefs_json)
rows = []
with open(path) as fh:
    for r in csv.DictReader(fh):
        try:
            rows.append({k: float(r[k]) for k in ("ts","prefill","prefill_sq_sum","decode","exec","num_seqs","sum_tokens","sum_sq_tokens","prefill_x_processed_ctx_sum")})
        except Exception:
            pass

def pred(r, ctx_feature):
    return (C["intercept"] + C["prefill_coeff"]*r["prefill"] + C["prefill_sq_coeff"]*r["prefill_sq_sum"]
            + C["decode_coeff"]*r["decode"] + C["sum_coeff"]*r["sum_tokens"]
            + C["sum_sq_coeff"]*r[ctx_feature])

def bucket_stats(rs, name):
    if not rs: return None
    out = {"name": name, "rows": len(rs)}
    for mode, feat in (("legacy_as_shipped","sum_sq_tokens"), ("cross_term_intended","prefill_x_processed_ctx_sum")):
        ratios, preds, acts = [], [], []
        for r in rs:
            p = pred(r, feat); a = r["exec"]
            preds.append(p); acts.append(a)
            if a > 0: ratios.append(p/a)
        ratios.sort()
        out[mode] = {
            "median_pred_ms": round(median(preds)*1000, 4),
            "median_actual_ms": round(median(acts)*1000, 4),
            "median_ratio_pred_over_actual": round(median(ratios), 4) if ratios else None,
            "p90_ratio": round(ratios[int(.9*len(ratios))], 4) if ratios else None,
            "max_ratio": round(ratios[-1], 4) if ratios else None,
            "mean_pred_ms": round(sum(preds)/len(preds)*1000, 4),
            "mean_actual_ms": round(sum(acts)/len(acts)*1000, 4),
        }
    # contribution of the context-squared term alone
    t = [C["sum_sq_coeff"]*r["sum_sq_tokens"] for r in rs]
    out["sum_sq_term_ms"] = {"median": round(median(t)*1000,4), "max": round(max(t)*1000,4)}
    out["num_seqs"] = {"median": median([r["num_seqs"] for r in rs]), "max": max(r["num_seqs"] for r in rs)}
    out["sum_tokens"] = {"median": median([r["sum_tokens"] for r in rs]), "max": max(r["sum_tokens"] for r in rs)}
    out["actual_exec_ms"] = {"median": round(median([r["exec"] for r in rs])*1000,4), "p99": round(sorted(r["exec"] for r in rs)[int(.99*len(rs))]*1000,4)}
    return out

res = {"label": label, "path": path, "total_rows": len(rows), "coefficients": C, "buckets": []}
decode_only = [r for r in rows if r["prefill"] == 0 and r["decode"] > 0]
for name, sel in (
    ("all", rows),
    ("decode_only", decode_only),
    ("decode_only_num_seqs_ge_20", [r for r in decode_only if r["num_seqs"] >= 20]),
    ("decode_only_num_seqs_ge_50", [r for r in decode_only if r["num_seqs"] >= 50]),
    ("prefill_containing", [r for r in rows if r["prefill"] > 0]),
):
    b = bucket_stats(sel, name)
    if b: res["buckets"].append(b)
print(json.dumps(res))

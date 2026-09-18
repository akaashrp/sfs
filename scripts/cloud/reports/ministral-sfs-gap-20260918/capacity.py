import json, sys, os, glob
from statistics import median as _med
def med(xs):
    xs=list(xs); return _med(xs) if xs else None
def r(x,n=3): return None if x is None else round(x,n)

rows=[]
for path in sys.argv[1:]:
    if not os.path.exists(path): continue
    d=json.load(open(path))
    for run in d["router"]["runs"]:
        pr=run["per_request"]; s=run["summary"]
        el=s.get("elapsed_s")
        byi={}
        for x in pr: byi.setdefault(x["instance_id"],[]).append(x)
        for inst,xs in sorted(byi.items()):
            met=sum(1 for x in xs if x.get("ttft_slo_met"))
            rows.append({
                "cell": os.path.basename(os.path.dirname(path)),
                "policy": run.get("label"),
                "qps": d["config"]["request_rate_qps"],
                "elapsed_s": r(el,1),
                "instance": inst,
                "routed": len(xs),
                "engine_qps": r(len(xs)/el,4) if el else None,
                "ttft_attain_pct": r(100*met/len(xs),2),
                "ttft_p50_ms": r(med([x["ttft_ms"] for x in xs if x.get("ttft_ms") is not None]),1),
                "ttft_p90_ms": r(sorted([x["ttft_ms"] for x in xs if x.get("ttft_ms") is not None])[int(.9*len(xs))],1) if xs else None,
                "queue_p50_ms": r(med([x["queue_delay_ms"] for x in xs if x.get("queue_delay_ms") is not None]),1),
                "decode_tok_per_s": r(sum(x.get("usage_completion_tokens") or 0 for x in xs)/el,1) if el else None,
                "prompt_tok_per_s": r(sum(x.get("usage_prompt_tokens") or 0 for x in xs)/el,1) if el else None,
                "prompt_p50": med([x["prompt_tokens"] for x in xs]),
                "completion_p50": med([x.get("usage_completion_tokens") or 0 for x in xs]),
            })
print(json.dumps(rows))

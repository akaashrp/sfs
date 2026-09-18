import json, sys, math
from statistics import median as _median

def median(xs):
    xs=list(xs)
    return _median(xs) if xs else None


def q(xs, p):
    xs = sorted(xs)
    if not xs: return None
    return xs[min(len(xs)-1, int(p*len(xs)))]

def r(x, n=2):
    return None if x is None else round(x, n)

out = {}
for spec in sys.argv[1:]:
    label, path = spec.split("=", 1)
    d = json.load(open(path))
    run = d["router"]["runs"][0]
    pr = run["per_request"]
    cfg = d["config"]
    pr = sorted(pr, key=lambda x: x.get("system_entry_offset_s") or 0.0)
    n = len(pr)
    cell = {
        "qps": cfg["request_rate_qps"],
        "n": n,
        "remaining_length_rule": run.get("remaining_length_rule"),
        "ttft_slo_attainment_pct": run["summary"].get("ttft_slo_attainment_pct"),
        "elapsed_s": run["summary"].get("elapsed_s"),
    }
    # per instance
    byinst = {}
    for x in pr:
        byinst.setdefault(x["instance_id"], []).append(x)
    cell["per_instance"] = {}
    for inst, xs in sorted(byinst.items()):
        met = [x for x in xs if x.get("ttft_slo_met")]
        cell["per_instance"][inst] = {
            "routed": len(xs),
            "ttft_attain_pct": r(100.0*len(met)/len(xs)),
            "ttft_ms_p50": r(median([x["ttft_ms"] for x in xs if x.get("ttft_ms") is not None])),
            "ttft_ms_p90": r(q([x["ttft_ms"] for x in xs if x.get("ttft_ms") is not None], .9)),
            "queue_delay_ms_p50": r(median([x["queue_delay_ms"] for x in xs if x.get("queue_delay_ms") is not None])),
            "pred_wait_ms_p50": r(median([x["wait_time_ms"] for x in xs if x.get("wait_time_ms") is not None])),
            "prompt_tokens_p50": r(median([x["prompt_tokens"] for x in xs])),
            "completion_tokens_sum": sum(x.get("usage_completion_tokens") or 0 for x in xs),
            "prompt_tokens_sum": sum(x.get("usage_prompt_tokens") or 0 for x in xs),
        }
    # quintiles by arrival
    cell["quintiles"] = []
    k = n // 5
    for i in range(5):
        xs = pr[i*k:(i+1)*k if i < 4 else n]
        pw = [x["wait_time_ms"] for x in xs if x.get("wait_time_ms") is not None]
        qd = [x["queue_delay_ms"] for x in xs if x.get("queue_delay_ms") is not None]
        tt = [x["ttft_ms"] for x in xs if x.get("ttft_ms") is not None]
        md = [x.get("wait_time_metadata") or {} for x in xs]
        nb = [m.get("num_batches") for m in md if m.get("num_batches")]
        ras = [m.get("running_at_snapshot") for m in md if m.get("running_at_snapshot") is not None]
        qas = [m.get("queued_at_snapshot") for m in md if m.get("queued_at_snapshot") is not None]
        perbatch = [x["wait_time_ms"]/m["num_batches"] for x, m in zip(xs, md)
                    if x.get("wait_time_ms") is not None and m.get("num_batches")]
        infeas = [x for x in xs if x.get("wait_time_ms") is not None and x.get("ttft_slo_ms") is not None
                  and x["wait_time_ms"] > x["ttft_slo_ms"]]
        met = [x for x in xs if x.get("ttft_slo_met")]
        cell["quintiles"].append({
            "i": i+1, "n": len(xs),
            "pred_wait_ms_p50": r(median(pw)) if pw else None,
            "queue_delay_ms_p50": r(median(qd)) if qd else None,
            "ttft_ms_p50": r(median(tt)) if tt else None,
            "ttft_slo_ms_p50": r(median([x["ttft_slo_ms"] for x in xs])),
            "over_pred_ratio_p50": r(median([a/b for a, b in zip(pw, qd) if b and b > 1.0]), 1),
            "num_batches_p50": median(nb) if nb else None,
            "implied_ms_per_batch_p50": r(median(perbatch), 3) if perbatch else None,
            "running_at_snapshot_p50": median(ras) if ras else None,
            "queued_at_snapshot_p50": median(qas) if qas else None,
            "selected_pred_infeasible_pct": r(100.0*len(infeas)/len(xs)),
            "ttft_attain_pct": r(100.0*len(met)/len(xs)),
        })
    # whole-cell aggregates
    pw = [x["wait_time_ms"] for x in pr if x.get("wait_time_ms") is not None]
    qd = [x["queue_delay_ms"] for x in pr if x.get("queue_delay_ms") is not None]
    infeas = [x for x in pr if x.get("wait_time_ms") is not None and x.get("ttft_slo_ms") is not None
              and x["wait_time_ms"] > x["ttft_slo_ms"]]
    cell["overall"] = {
        "pred_wait_ms_p50": r(median(pw)), "pred_wait_ms_p90": r(q(pw, .9)),
        "queue_delay_ms_p50": r(median(qd)), "queue_delay_ms_p90": r(q(qd, .9)),
        "ttft_slo_ms_p50": r(median([x["ttft_slo_ms"] for x in pr])),
        "selected_pred_infeasible_pct": r(100.0*len(infeas)/len(pr)),
        "implied_ms_per_batch_p50": r(median([x["wait_time_ms"]/(x.get("wait_time_metadata") or {}).get("num_batches", 1)
                                              for x in pr if x.get("wait_time_ms") is not None
                                              and (x.get("wait_time_metadata") or {}).get("num_batches")]), 3),
    }
    out[label] = cell
print(json.dumps(out))

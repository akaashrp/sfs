"""Corrected-estimate check: is the simulator's BATCH COUNT right and only its clock wrong?

corrected_ms = num_batches * (measured median engine step time at the concurrency the
snapshot reported).  Compared against the request's measured queue_delay_ms.
"""
import csv, json, sys, os
from statistics import median as _med
def med(xs):
    xs = list(xs)
    return _med(xs) if xs else None
def r(x, n=2): return None if x is None else round(x, n)

point_paths = json.loads(sys.argv[1])   # {cell: point.json}
stats_paths = json.loads(sys.argv[2])   # {instance_id: batch_stats.csv}

# measured step time by concurrency bucket, per engine
BUCKETS = [(0,5),(5,10),(10,20),(20,35),(35,50),(50,70),(70,90),(90,120),(120,10**9)]
lookup = {}
for inst, path in stats_paths.items():
    by = {b: [] for b in BUCKETS}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            try:
                ns = float(row["num_seqs"]); ex = float(row["exec"])
            except Exception:
                continue
            for b in BUCKETS:
                if b[0] <= ns < b[1]:
                    by[b].append(ex); break
    lookup[inst] = {b: med(v) for b, v in by.items() if v}

def step_ms(inst, running):
    tbl = lookup.get(inst) or {}
    for b in BUCKETS:
        if b[0] <= running < b[1] and tbl.get(b) is not None:
            return tbl[b]*1000.0
    vals = [v for v in tbl.values() if v is not None]
    return med(vals)*1000.0 if vals else None

out = {"measured_step_ms_by_concurrency": {
        i: {f"{b[0]}-{b[1] if b[1] < 10**9 else 'inf'}": r(v*1000.0, 3)
            for b, v in sorted(t.items())} for i, t in lookup.items()},
       "cells": {}}

for cell, path in point_paths.items():
    if not os.path.exists(path): continue
    d = json.load(open(path))
    run = d["router"]["runs"][0]
    pr = sorted(run["per_request"], key=lambda x: x.get("system_entry_offset_s") or 0.0)
    n = len(pr); k = n//5
    quint = []
    for i in range(5):
        xs = pr[i*k:(i+1)*k if i < 4 else n]
        raw, corr, act = [], [], []
        for x in xs:
            m = x.get("wait_time_metadata") or {}
            nb, ras = m.get("num_batches"), m.get("running_at_snapshot")
            qd, pw = x.get("queue_delay_ms"), x.get("wait_time_ms")
            if not nb or ras is None or qd is None or pw is None: continue
            s = step_ms(x["instance_id"], ras)
            if s is None: continue
            raw.append(pw); corr.append(nb*s); act.append(qd)
        pairs_r = [a/b for a, b in zip(raw, act) if b and b > 100.0]
        pairs_c = [a/b for a, b in zip(corr, act) if b and b > 100.0]
        quint.append({"i": i+1, "n": len(raw),
            "raw_pred_ms_p50": r(med(raw)), "corrected_pred_ms_p50": r(med(corr)),
            "measured_queue_delay_ms_p50": r(med(act)),
            "raw_over_actual_p50": r(med(pairs_r), 1), "corrected_over_actual_p50": r(med(pairs_c), 2),
            "pairs_used": len(pairs_c)})
    out["cells"][cell] = quint
print(json.dumps(out))

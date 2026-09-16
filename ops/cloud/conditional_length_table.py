"""Build and time model-only conditional length tables from held-out calibration.

This is a diagnostic artifact builder; it does not modify the serving estimator.
"""
import argparse
from array import array
from bisect import bisect_right
import hashlib
import json
import math
import statistics
from pathlib import Path
import time


def tables(lengths, cap=8192):
    values=sorted(min(cap,int(x)) for x in lengths)
    suffix=[0]*(len(values)+1)
    for i in range(len(values)-1,-1,-1):suffix[i]=suffix[i+1]+values[i]
    median,mean,support=array('I'),array('I'),array('I')
    for generated in range(cap):
        start=bisect_right(values,generated); n=len(values)-start
        if n:
            j=start+n//2
            med=values[j] if n%2 else (values[j-1]+values[j])/2
            median.append(math.ceil(med));mean.append(math.ceil(suffix[start]/n))
        else:
            median.append(generated+1);mean.append(generated+1)
        support.append(n)
    return {'median_total':median,'mean_total':mean,'survivors':support}


def main():
    p=argparse.ArgumentParser();p.add_argument('--calibration',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    raw=a.calibration.read_bytes();cal=json.loads(raw)
    models=cal['models']
    start=time.perf_counter();result={m:tables(v if isinstance(v,list) else v['lengths']) for m,v in models.items()};build=time.perf_counter()-start
    check=[]
    for model,tab in result.items():
        for g in range(8192):
            assert g<tab['median_total'][g]<=8192 and g<tab['mean_total'][g]<=8192
        for g in (0, 1, 128, 512, 2048, 4096, 8191):
            survivors = [min(8192,int(x)) for x in models[model] if min(8192,int(x)) > g]
            assert tab['survivors'][g] == len(survivors)
            assert tab['median_total'][g] == (math.ceil(statistics.median(survivors)) if survivors else g+1)
            assert tab['mean_total'][g] == (math.ceil(statistics.mean(survivors)) if survivors else g+1)
        check.append({'model':model,'table_bytes':sum(x.itemsize*len(x) for x in tab.values()),'min_survivors':min(tab['survivors'])})
    first=next(iter(result.values()))['median_total'];total=0;t=time.perf_counter()
    for i in range(1000000):total+=first[i%8192]-(i%8192)
    latency=(time.perf_counter()-t)*1e9/1000000
    a.output.mkdir(parents=True,exist_ok=False)
    for model,tab in result.items():
        for key,data in tab.items():(a.output/f'{model}-{key}.bin').write_bytes(data.tobytes())
    (a.output/'summary.json').write_text(json.dumps({'status':'PASS_OFFLINE_LOOKUP','source_sha256':hashlib.sha256(raw).hexdigest(),'build_seconds':build,'python_lookup_nanoseconds':latency,'tables':check,'cap':8192,'neural_training_required':False,'runtime_deployed':False,'limitations':['Model-only conditioning; no prompt/initial-prediction features','Empty calibration tails fall back to one remaining token; support counts expose sparse tails','Latency benefit must be tested; low lookup cost does not establish predictor quality'],'checksum':total},indent=2)+'\n')
    print((a.output/'summary.json').read_text())

if __name__=='__main__':main()

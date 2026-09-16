#!/usr/bin/env python3
"""Read-only comparison of saved Qwen control requests and batch traces."""
import argparse
import ast
import collections
import csv
import json
import math
from pathlib import Path
import statistics as st


def stats(values):
    a=sorted(values)
    return {'n':len(a),'mean':st.mean(a),'p50':st.median(a),'p90':a[math.ceil(.9*len(a))-1]} if a else None


def analyze(root):
    point=next((root/'outputs').glob('*point01.json'))
    data=json.loads(point.read_text()); rows=sorted(data['router']['runs'][0]['per_request'],key=lambda r:r['system_entry_offset_s'])
    log=next((root/'outputs').glob('*predicted_waits*'))
    with log.open() as f:start=ast.literal_eval(next(f))['fetched_at_s']
    requests=[]
    for i in range(4):
        group=rows[i*4000:(i+1)*4000]; models={}
        for model in sorted(set(r['response_model'] for r in group)):
            selected=[r for r in group if r['response_model']==model]
            models[model]={'n':len(selected),'buckets':dict(collections.Counter(r['bucket'] for r in selected)),
                'attainment_pct':100*st.mean(r['system_entry_e2e_ttft_slo_met'] for r in selected),
                **{k:stats([r[k] for r in selected]) for k in ('ttft_ms','wait_time_ms','system_entry_to_dispatch_ms','usage_completion_tokens','usage_prompt_tokens')}}
        requests.append({'arrival_start_s':group[0]['system_entry_offset_s'],'arrival_end_s':group[-1]['system_entry_offset_s'],'attainment_pct':100*st.mean(r['system_entry_e2e_ttft_slo_met'] for r in group),'models':models})
    batches={}
    for path in root.glob('batch_stats_*.csv'):
        model=path.stem.removeprefix('batch_stats_'); groups=collections.defaultdict(list);prev=None
        with path.open() as f:
            for row in csv.DictReader(f):
                row={k:float(v) for k,v in row.items()}; ts=row['ts']-start
                # interval is between schedule starts: subtract PREVIOUS batch work.
                if prev is not None and 0<=ts<1900 and row['prefill']==0 and prev['prefill']==0 and prev['total']>0 and 0<row['interval']<.2:
                    key=(int(row['num_seqs'])//8,int(row['sum_tokens'])//25000)
                    groups[key].append((row['exec']*1000,row['interval']*1000,(row['interval']-prev['exec']-prev['sched'])*1000,row['sched']*1000))
                prev=row
        batches[model]={str(key):{'n':len(a),**{k:st.median(r[j] for r in a) for j,k in enumerate(('exec_ms','interval_ms','outside_sched_exec_ms','sched_ms'))}} for key,a in groups.items() if len(a)>=30}
    return {'point':str(point),'request_quarters':requests,'decode_shape_bins':batches}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bridges',type=Path,required=True);p.add_argument('--vast',type=Path,required=True);p.add_argument('--output',type=Path,required=True);o=p.parse_args()
    result={n:analyze(r) for n,r in [('bridges',o.bridges),('vast',o.vast)]};matched={}
    for model,base in result['bridges']['decode_shape_bins'].items():
        other=result['vast']['decode_shape_bins'][model]; pairs=[(min(row['n'],other[key]['n']),row,other[key]) for key,row in base.items() if key in other]
        weight=sum(a[0] for a in pairs)
        matched[model]={'common_shape_bins':len(pairs),'matched_weight':weight,'caveat':'8-sequence and 25k-context-token bins; not exact tensor-shape pairing',**{metric:sum(w*(b[metric]/a[metric]) for w,a,b in pairs)/weight for metric in ('exec_ms','interval_ms','sched_ms')}}
    result['matched_vast_over_bridges']=matched;o.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(matched,indent=2))

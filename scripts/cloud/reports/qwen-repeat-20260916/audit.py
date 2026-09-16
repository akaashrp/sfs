"""Independently audit final repeat counts, TTFT gates and both saved-score utilities."""
import csv,hashlib,json,math,statistics
from pathlib import Path
base=Path('/ocean/projects/cis250162p/aparthas')
root=base/'sfs_cloud_results_20260916/sfs-vast/controls/qps8p6-repeat1'
points=list((root/'outputs').glob('*point01.json'));assert len(points)==1
p=points[0];d=json.loads(p.read_text());r=d['router']['runs'][0];rows=r['per_request'];assert len(rows)==16000 and r['utility']=='hard'
assert len({x['request_id'] for x in rows})==16000 and not any(x.get('error') for x in rows)
assert d['config']['request_rate_qps']==8.6
assert r['summary']['failed_requests']==0 and r['summary']['succeeded_requests']==16000
for x in rows:
 assert math.isfinite(x['system_entry_e2e_ttft_ms'])
 assert x['system_entry_e2e_ttft_slo_met']==(x['system_entry_e2e_ttft_ms']<=x['ttft_slo_ms'])
with (base/'vllm_utils/bucketed_prompt_outputs/req_map_qps_seed69_holdout4000_n16000.csv').open() as f:m={x['req_id']:x for x in csv.DictReader(f)}
scores={}
for line in (base/'sfs_model_family/experiments/paper_ablation_20260907/judge_flash/paired_scores.jsonl').open():
 x=json.loads(line);scores[(x['bucket'],str(x['example_id']),x['model_label'])]=x
common={k[:2] for k in scores if all((*k[:2],model) in scores for model in ('qwen3-0.6b','qwen3-8b','qwen3-32b'))};assert len(common)==15996
utility={'pro':[],'flash':[]}
for x in rows:
 q=m[x['request_id']];key=(q['bucket'],q['example_id']);assert q['bucket']==x['bucket']
 if key not in common:continue
 score=scores[(*key,x['response_model'])]
 for judge in utility:utility[judge].append((score[judge]-d['config']['lambda_weight']*x['actual_cost'])*x['system_entry_e2e_ttft_slo_met'])
out={'status':'PASS_INDEPENDENT_CONTROLLER_AUDIT','raw_point':str(p),'raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'requests':16000,'succeeded':16000,'failed':0,'scored_queries':len(common),'qps':8.6,'ttft_slo_attainment_pct':100*statistics.mean(x['system_entry_e2e_ttft_slo_met'] for x in rows),'ontimeutility':{k:statistics.mean(v) for k,v in utility.items()},'arrival_quarters':[]}
a=sorted(rows,key=lambda x:x['system_entry_offset_s'])
for i in range(4):
 q=a[i*4000:(i+1)*4000];out['arrival_quarters'].append({'attainment_pct':100*statistics.mean(x['system_entry_e2e_ttft_slo_met'] for x in q),'engine_ttft_median_ms':statistics.median(x['ttft_ms'] for x in q),'router_delay_mean_ms':statistics.mean(x['system_entry_to_dispatch_ms'] for x in q)})
summary=json.loads((root/'result_summary.json').read_text());assert out['raw_sha256']==summary['raw_sha256'];assert abs(out['ttft_slo_attainment_pct']-summary['ttft_slo_attainment_pct'])<1e-9
for judge in utility:assert abs(out['ontimeutility'][judge]-summary['ontimeutility'][judge])<1e-12
(root/'controller_independent_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

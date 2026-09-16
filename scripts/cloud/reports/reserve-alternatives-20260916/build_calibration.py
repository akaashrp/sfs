import csv,hashlib,json
from pathlib import Path
base=Path('/ocean/projects/cis250162p/aparthas/vllm_utils/bucketed_prompt_outputs')
with (base/'req_map_qps_seed69_holdout4000_n16000.csv').open() as f:heldout={(r['bucket'],r['example_id']) for r in csv.DictReader(f)}
summary=json.loads((base/'model_dataset_input_output_lengths.json').read_text());out={'models':{},'sources':[],'heldout_overlap':0}
for model,m in summary['models'].items():
 rows=[]
 for bucket,b in m['datasets'].items():
  path=base/model/'outputs'/f'{bucket}.jsonl';lengths=[];indices=set()
  for line in path.open():
   r=json.loads(line);assert not r.get('error');assert r['model_label']==model
   assert (bucket,str(r['prompt_metadata']['example_id'])) not in heldout
   indices.add(r['prompt_index']);n=r['response']['completion_tokens'];assert 0<=n<=r['max_completion_tokens'];lengths.append(n)
  assert indices==set(range(2500)) and lengths==b['output_lengths']
  rows.extend(lengths);out['sources'].append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'records':len(lengths)})
 out['models'][model]=sorted(rows);print(model,len(rows),flush=True)
Path('.scratch/reserve-analysis-20260916/calibration-lengths.json').write_text(json.dumps(out)+'\n')

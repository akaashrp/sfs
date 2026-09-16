import json,re,time
from pathlib import Path
root=Path('/workspace/sfs/state/controls/qps8p6-repeat1')
status=json.loads((root/'status.json').read_text())
monitor=json.loads(Path('/workspace/sfs/state/monitor/latest.json').read_text())
result={'time':time.time(),'state':status['state'],'phase_elapsed_minutes':round((time.time()-status['time'])/60,1),'health':monitor.get('controls',{}).get(root.name,{}).get('health'),'alerts':monitor.get('attention',[]),'engines':{}}
for p in root.glob('server_qwen*.log'):
    with p.open('rb') as f:
        f.seek(max(0,p.stat().st_size-16384));tail=f.read().decode(errors='replace')
    rows=re.findall(r'(\d\d:\d\d:\d\d).*?Running: (\d+) reqs, Waiting: (\d+) reqs, GPU util: ([\d.]+)%, GPU KV cache usage: ([\d.]+)%',tail)
    if rows:
        t,r,w,g,k=rows[-1];result['engines'][p.stem.removeprefix('server_')]={'logged_at':t,'running':int(r),'waiting':int(w),'gpu_pct':float(g),'kv_pct':float(k)}
for p in (root/'outputs').glob('*predicted_waits_router_hard.log'):
    with p.open('rb') as f:
        f.seek(max(0,p.stat().st_size-65536));tail=f.read().decode(errors='replace')
    ids=re.findall(r"'request_id': 'hard-req-(\d+)'",tail)
    if ids:result['last_routed_request_index']=max(map(int,ids))
if status['state']=='RUNNING_16000':
    offset=time.time()-time.monotonic();done=set()
    for p in root.glob('wait_qwen*.log'):
        for line in p.open():
            rid=re.search(r'request_id=chatcmpl-hard-req-(\d+) ',line)
            ready=re.search(r'ready_ts_s=([\d.]+)',line)
            if rid and ready and float(ready[1])+offset>=status['time']:done.add(int(rid[1]))
    result['logged_completions']=len(done)
if (root/'result_summary.json').exists():result['summary']=json.loads((root/'result_summary.json').read_text())
print(json.dumps(result),flush=True)

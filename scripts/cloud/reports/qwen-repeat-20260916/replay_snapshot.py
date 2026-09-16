import argparse,hashlib,importlib.util,json,struct,time
from pathlib import Path
import msgspec
import torch
parser=argparse.ArgumentParser(description='Read-only native replay with hindsight lengths for running overruns')
for name in ('extension','point','snapshots','output'):parser.add_argument('--'+name,type=Path,required=True)
a=parser.parse_args()
lib=a.extension
spec=importlib.util.spec_from_file_location('_scheduler_sim',lib);native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
point=a.point
rows=json.loads(point.read_text())['router']['runs'][0]['per_request'];lookup={r['response_id']:r for r in rows}
root=a.snapshots
h=struct.Struct('<8sIIQQdQdddddddd');results=[]
for name in ('1789548216429795210-qwen3-0.6b.bin','1789548919290306853-qwen3-0.6b.bin'):
 raw=(root/name).read_bytes();hdr=h.unpack(raw[:h.size]);payload=raw[h.size:];d=msgspec.msgpack.decode(payload)
 w=native.SchedulerSimulationWorker(1.,hdr[9],hdr[10],hdr[12],hdr[13],hdr[11],hdr[14])
 w.update_snapshot(payload);parsed=w.parsed_snapshot()
 assert all(parsed['requests'][rid]['num_output_target_tokens']==r['num_output_target_tokens'] for rid,r in d['requests'].items())
 baseline=w.run_simulation_for_test(payload,4096,'prefill_done',(),d['created_at'],0.)
 changed=[]
 for rid in d['running_request_ids']:
  r=d['requests'][rid];match=lookup.get(rid)
  if match and r['num_output_target_tokens']-r['num_output_processed_tokens']<=1 and match['usage_completion_tokens']-r['num_output_processed_tokens']>=128:
   r['num_output_target_tokens']=match['usage_completion_tokens'];changed.append(rid)
 assert changed, 'Expected matching running overrun requests'
 oracle=w.run_simulation_for_test(msgspec.msgpack.encode(d),4096,'prefill_done',(),d['created_at'],0.)
 result=dict(extension_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),raw_point_sha256=hashlib.sha256(point.read_bytes()).hexdigest(),snapshot=name,changed_running_requests=changed,probe_prompt_tokens=4096,original=baseline,oracle_only_for_running_overruns=oracle)
 results.append(result);print(json.dumps(result),flush=True)
a.output.write_text(json.dumps(results,indent=2)+'\n')

"""Read coherent published telemetry through read-only mmap; never register SHM ownership."""
import json,mmap,struct,time
from pathlib import Path
root=Path('/workspace/sfs/state/controls/qps8p6-repeat1')
if json.loads((root/'status.json').read_text())['state']!='RUNNING_16000':raise SystemExit(0)
header=struct.Struct('<8sIIQQdQdddddddd')
out=Path('/workspace/sfs/state/diagnostics/qwen-repeat1-snapshots');out.mkdir(exist_ok=True)
now=time.time_ns();record={'time_ns':now,'monotonic':time.monotonic(),'scope':'Read-only seqlock snapshot capture; no SHM creation, registration, writes or unlink','models':{}}
for row in json.loads((root/'instances.json').read_text())['instances']:
    with Path('/dev/shm',row['snapshot_shm_name']).open('rb') as stream:
        with mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as view:
            for attempt in range(32):
                before=view[:header.size];v=header.unpack(before)
                assert v[0]==b'VLLMSHM1' and v[1]==1 and v[2]==header.size
                if v[3]%2:continue
                assert 0<=v[6]<=len(view)-header.size
                payload=view[header.size:header.size+v[6]]
                if before!=view[:header.size]:continue
                name=f'{now}-{row["model_id"]}.bin';(out/name).write_bytes(before+payload)
                record['models'][row['model_id']]={'file':name,'snapshot_version':v[4],'created_at':v[5],'payload_bytes':v[6]};break
            else:record['models'][row['model_id']]={'coherent_read_unavailable':True}
with (out/'index.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
print(json.dumps(record))

"""Read-only campaign progress monitor, separate from routing decisions."""
import json
import mmap
from pathlib import Path
import struct
import subprocess
import time

ROOT = Path('/workspace/sfs/state')
OUT = ROOT/'monitor/baselines'
HEADER = struct.Struct('<8sIIQQdQ')


def observe():
    record={'time':time.time(), 'families':{}, 'attention':[]}
    record['gpus']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True,timeout=15).splitlines()
    for folder in sorted((ROOT/'baselines').glob('*')):
        status=folder/'status.json'
        if not status.exists(): continue
        row=json.loads(status.read_text())
        for name in ('phase','active_cell','heartbeat'):
            p=folder/(name+'.json')
            if p.exists(): row[name]=json.loads(p.read_text())
        row['completed_cells']=len(list((folder/'cells').glob('*/audit.json')))
        row['batch_traces']={p.name:{'bytes':p.stat().st_size,'mtime_age_s':time.time()-p.stat().st_mtime} for p in folder.glob('batch_stats_*.csv')}
        row['snapshots']={}
        cfg=folder/'instances.json'
        if cfg.exists() and row['state']=='RUNNING':
            for instance in json.loads(cfg.read_text())['instances']:
                try:
                    # mmap does not register as a shared-memory owner or unlink.
                    with (Path('/dev/shm')/instance['snapshot_shm_name']).open('rb') as f:
                        with mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as view:
                            a=bytes(view[:HEADER.size]); b=bytes(view[:HEADER.size])
                    magic,version,size,seq,snap,created,payload=HEADER.unpack(a)
                    if a != b or seq%2: raise ValueError('Concurrent publication; retry next observation')
                    row['snapshots'][instance['model_id']]={'version':snap,'publication_age_s':max(0,time.monotonic()-created),'payload_bytes':payload}
                except (OSError,ValueError) as error:
                    row['snapshots'][instance['model_id']]={'error':str(error)}
        if row['state']=='FAILED': record['attention'].append(folder.name+': '+row.get('error','failed'))
        if row['state']=='RUNNING' and row.get('heartbeat') and time.time()-row['heartbeat']['time']>90:
            record['attention'].append(folder.name+': controller heartbeat is not advancing')
        record['families'][folder.name]=row
    return record


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    while True:
        try: row=observe()
        except Exception as error: row={'time':time.time(),'attention':[str(error)]}
        raw=json.dumps(row,allow_nan=False)
        with (OUT/'observations.jsonl').open('a') as f:f.write(raw+'\n')
        pending=OUT/'latest.pending';pending.write_text(raw+'\n');pending.replace(OUT/'latest.json')
        print(json.dumps({'time':row['time'],'attention':row['attention'],'phases':{k:v.get('phase',v['state']) for k,v in row.get('families',{}).items()}}),flush=True)
        time.sleep(30)

if __name__=='__main__':main()

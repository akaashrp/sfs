import json,subprocess,time
from pathlib import Path
out=Path(__file__).with_name('observations.jsonl')
last_capture={}
while True:
    finished=0
    for lane in ('a','b'):
        root='/workspace/sfs/state/controls/qps8-lane-'+lane
        try:
            p=subprocess.run(['ssh','-o','ConnectTimeout=15','sfs-vast','python3 /workspace/sfs/setup/inspect_qps8_control.py '+root],capture_output=True,text=True,timeout=45)
            d=json.loads(p.stdout);d['lane']=lane
            if d['state']=='RUNNING_16000' and time.time()-last_capture.get(lane,0)>300:
                c=subprocess.run(['ssh','-o','ConnectTimeout=15','sfs-vast','python3 /workspace/sfs/setup/capture_qps8_snapshots.py '+root],capture_output=True,text=True,timeout=45)
                d['snapshot_capture_returncode']=c.returncode
                last_capture[lane]=time.time()
        except Exception as e:d={'time':time.time(),'lane':lane,'state':'MONITOR_ERROR','error':str(e)}
        text=json.dumps(d)
        with out.open('a') as f:f.write(text+'\n')
        print(text,flush=True)
        if d['state'] in ('COMPLETE','FAILED'):finished+=1
    if finished==2:break
    time.sleep(45)

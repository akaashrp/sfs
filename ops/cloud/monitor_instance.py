#!/usr/bin/env python3
"""Read-only cloud job/GPU monitor. Never starts, stops, or modifies a job."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request


def snapshot(state):
    record = {'time': time.time(), 'controls': {}, 'attention': []}
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu,temperature.gpu,power.draw',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
    record['gpu_rows'] = result.stdout.strip().splitlines()
    if result.returncode:
        record['attention'].append('nvidia-smi failed')
    for directory in sorted((state/'controls').glob('qps*')):
        path = directory/'status.json'
        if not path.exists():
            continue
        status = json.loads(path.read_text())
        health = {}
        config = directory/'instances.json'
        if config.exists():
            for model in json.loads(config.read_text())['instances']:
                try:
                    with urllib.request.urlopen(model['address']+'/health', timeout=2) as response:
                        health[model['model_id']] = response.status
                except Exception as error:
                    health[model['model_id']] = type(error).__name__
        status['health'] = health
        if status['state'] == 'FAILED':
            record['attention'].append(directory.name+' failed')
        if status['state'] in ('SMOKE', 'RUNNING_16000'):
            if any(value != 200 for value in health.values()):
                record['attention'].append(directory.name+' health probe failed')
            try:
                os.kill(status['pid'], 0)
            except ProcessLookupError:
                record['attention'].append(directory.name+' driver exited without final status')
        summary = directory/'result_summary.json'
        if summary.exists():
            status['result'] = json.loads(summary.read_text())
        record['controls'][directory.name] = status
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=60)
    options = parser.parse_args()
    output = options.state/'monitor'; output.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            record = snapshot(options.state)
        except Exception as error:
            record = {'time': time.time(), 'attention': [str(error)]}
        text = json.dumps(record, allow_nan=False)
        with (output/'observations.jsonl').open('a') as stream:
            stream.write(text+'\n')
        pending = output/'latest.pending'; pending.write_text(text+'\n'); pending.replace(output/'latest.json')
        print(json.dumps({'time': record['time'], 'attention': record['attention'],
                          'states': {key: value['state'] for key,value in record.get('controls',{}).items()}}), flush=True)
        time.sleep(options.interval)


if __name__ == '__main__':
    main()

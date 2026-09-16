#!/usr/bin/env python3
"""Poll one authorized Vast rental, saving only non-secret status fields."""
import argparse
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance', type=int, required=True)
    parser.add_argument('--key-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=60)
    options = parser.parse_args(); options.output.mkdir(parents=True, exist_ok=True)
    while True:
        record = {'time': time.time(), 'instance_id': options.instance}
        try:
            request = urllib.request.Request('https://console.vast.ai/api/v1/instances/?owner=me',
                headers={'Authorization': 'Bearer '+options.key_file.read_text().strip()})
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = json.load(response)
            instances = payload['instances'] if isinstance(payload, dict) else payload
            matches = [row for row in instances if row['id'] == options.instance]
            record['present'] = len(matches) == 1
            if matches:
                record.update({key: matches[0].get(key) for key in
                    ('actual_status', 'cur_state', 'intended_status', 'gpu_name', 'num_gpus', 'dph_total', 'disk_space', 'workspace_is_volume')})
        except Exception as error:
            record['error'] = type(error).__name__
        text = json.dumps(record)
        with (options.output/'vast-account-status.jsonl').open('a') as stream:
            stream.write(text+'\n')
        pending = options.output/'vast-account-latest.pending'; pending.write_text(text+'\n')
        pending.replace(options.output/'vast-account-latest.json')
        print(text, flush=True)
        time.sleep(options.interval)


if __name__ == '__main__':
    main()

"""Frozen calibration-only warm-up contract (32 prompts, 96 completions)."""
import json
from collections import Counter
from pathlib import Path

BUCKETS = ('alpaca','govreport-summarization','hotpot_qa','writingprompts')

def load_warmup(path):
    if not path: raise ValueError('vllm_sr_latency requires a frozen calibration warm-up file')
    data = json.loads(Path(path).read_text())
    rows = data.get('requests',[])
    if data.get('data_role') != 'calibration' or len(rows)!=32 or Counter(r.get('bucket') for r in rows)!=Counter({b:8 for b in BUCKETS}):
        raise ValueError('Warm-up requires 32 balanced calibration requests')
    if len({r['request_id'] for r in rows})!=32 or any(not isinstance(r.get('prompt'),str) or not r['prompt'] or r.get('prompt_tokens',0)<=0 for r in rows):
        raise ValueError('Invalid warm-up prompt identities/features')
    return rows

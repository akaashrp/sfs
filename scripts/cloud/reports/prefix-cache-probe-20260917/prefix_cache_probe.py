"""Bounded GPU-7 prefix-cache hit-rate probe on the frozen Qwen evaluation prompts (diagnostic, not a cell).

Serves ONE canonical Qwen3-0.6B engine (campaign argv from scripts.cloud.pool.server_argv, with only
--no-enable-prefix-caching flipped to --enable-prefix-caching and --enable-prompt-tokens-details added so
usage reports cached tokens), then streams the campaign's frozen request set (scripts.runs.experiments
._build_request_set with the bundle's canonical argv) with max_completion_tokens=1 so only prefill runs.

Phases: warm-up -> reset cache -> template-isolation (first eval request of each bucket, twice) -> reset
-> 512 calibration smoke prompts (warm-up, concurrency 128) -> 16,000 evaluation requests in frozen
arrival order (concurrency 128) -> final metrics. Records /metrics prefix-cache counters per phase, per-request
prompt/cached tokens, and the tokenizer-level common prefix (template vs dataset header) for attribution.

Waits (inside this process) until the blocking supervisor program is not RUNNING and the GPU reads 0 MiB;
holds no pool lock (the pool lock is non-blocking and would fail a production pool); exits leaving the GPU
empty. Never touches other GPUs, the repository checkouts, or campaign state.
"""
import argparse
import asyncio
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

from scripts.cloud.common import ROOT, read, write, expand
from scripts.cloud.canonical_control import preflight
from scripts.cloud.pool import server_argv, hardware
from scripts.cloud.worker import parse

BLOCK = 16
METRIC_RE = re.compile(r'^(vllm:[A-Za-z0-9_:]+)(\{[^}]*\})?\s+(\S+)$')
KEEP = ('vllm:prefix_cache_queries_total', 'vllm:prefix_cache_hits_total', 'vllm:prefix_cache_queries',
        'vllm:prefix_cache_hits', 'vllm:kv_cache_lookups_total', 'vllm:kv_cache_hit_rate_perc',
        'vllm:prompt_tokens_total', 'vllm:request_success_total', 'vllm:num_requests_running',
        'vllm:num_requests_waiting', 'vllm:kv_cache_usage_perc', 'vllm:num_preemptions_total')
HIT_LINE = re.compile(r'Prefix cache hit rate: ([\d.]+)% \(step: ([\d.]+)%\)')


def sh(argv):
    return subprocess.run(argv, capture_output=True, text=True, timeout=60).stdout


def gpu_memory_used(gpu):
    return int(sh(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-i', gpu]).strip().splitlines()[0])


def gpu_compute_pids(gpu):
    out = sh(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits', '-i', gpu])
    pids = []
    for line in out.strip().splitlines():
        try:
            pids.append(int(line.split(',')[0]))
        except ValueError:
            pass
    return pids


def program_running(name):
    try:
        return 'RUNNING' in sh(['supervisorctl', 'status', name])
    except Exception:
        return True  # be conservative: an unreadable supervisor state keeps us waiting


def scrape(address):
    with urllib.request.urlopen(address + '/metrics', timeout=10) as response:
        text = response.read().decode()
    values = collections.defaultdict(float)
    for line in text.splitlines():
        m = METRIC_RE.match(line)
        if m and m.group(1) in KEEP:
            values[m.group(1)] += float(m.group(3))
    return dict(values)


def delta(before, after):
    q = after.get('vllm:prefix_cache_queries_total', 0) - before.get('vllm:prefix_cache_queries_total', 0)
    h = after.get('vllm:prefix_cache_hits_total', 0) - before.get('vllm:prefix_cache_hits_total', 0)
    return {'queries': q, 'hits': h, 'hit_rate': (h / q) if q else None,
            'prompt_tokens': after.get('vllm:prompt_tokens_total', 0) - before.get('vllm:prompt_tokens_total', 0),
            'requests': after.get('vllm:request_success_total', 0) - before.get('vllm:request_success_total', 0),
            'preemptions': after.get('vllm:num_preemptions_total', 0) - before.get('vllm:num_preemptions_total', 0)}


def common_prefix_len(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu', required=True)
    ap.add_argument('--cpus', default='84-95')
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--models', required=True)
    ap.add_argument('--state', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--wait-program', default='sfs-fcfs-run-lane-b')
    ap.add_argument('--model', default='qwen3-0.6b')
    ap.add_argument('--concurrency', type=int, default=128)
    ap.add_argument('--smoke-per-bucket', type=int, default=128)
    ap.add_argument('--gpu-budget-s', type=float, default=840, help='hard wall-clock budget from server launch to server stop')
    ap.add_argument('--poll-s', type=float, default=60)
    ap.add_argument('--dry-run', action='store_true', help='stop after CPU-only preparation (no GPU wait, no server)')
    o = ap.parse_args()
    lo, hi = (int(x) for x in o.cpus.split('-'))
    os.sched_setaffinity(0, set(range(lo, hi + 1)))
    out = Path(o.output).resolve(); out.mkdir(parents=True, exist_ok=False)
    status = lambda state, **extra: write(out / 'status.json', dict(state=state, time=time.time(), pid=os.getpid(), **extra))
    logf = (out / 'run.log').open('a')
    def log(msg):
        logf.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}\n'); logf.flush(); print(msg, flush=True)

    # ---- CPU-only preparation (before touching the GPU) ----
    bundle, state = Path(o.bundle).resolve(), Path(o.state).resolve()
    manifest = read(bundle / 'bundle.json')
    definition = manifest['families']['qwen']
    status('PREFLIGHT')
    argv, evidence = preflight(bundle, manifest)
    from scripts.runs import experiments as exp
    from scripts.runs.ministral3_methodology_stage import smoke_requests
    from sfs_core.shared.shared_experiment_helpers import build_messages, resolve_chat_template_kwargs
    args = parse(argv)
    requests, request_manifest, prompt_source = exp._build_request_set(args)
    system_prompt = args.system_prompt
    template_kwargs = resolve_chat_template_kwargs(args.chat_template_kwargs)
    calibration = [exp.ExperimentRequest(**json.loads(line)) for line in
                   Path(expand(definition['calibration_requests'], bundle, {})).read_text().splitlines()]
    smoke = smoke_requests(calibration, per_bucket=o.smoke_per_bucket)
    write(out / 'preflight.json', dict(evidence, system_prompt=system_prompt, chat_template_kwargs=template_kwargs,
                                       smoke_requests=len(smoke), eval_requests=len(requests),
                                       request_order_sha256=hashlib.sha256(json.dumps([r.request_id for r in requests]).encode()).hexdigest(),
                                       prompt_source={k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v))
                                                      for k, v in prompt_source.items() if not isinstance(v, (list, dict))}))
    log(f'preflight passed: {len(requests)} eval requests, {len(smoke)} smoke requests')

    # Tokenizer-level attribution: rendered chat text -> token ids, common prefix globally and per bucket.
    status('TOKENIZING')
    model_paths = read(o.models)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_paths[o.model])
    jinja = (ROOT / 'src/assets/templates/chat_template_qwen3.jinja').read_text()
    def render_ids(prompt):
        text = tokenizer.apply_chat_template(build_messages(prompt, system_prompt), tokenize=False,
                                             add_generation_prompt=True, chat_template=jinja, **template_kwargs)
        return tokenizer(text, add_special_tokens=False)['input_ids']
    local_tokens, global_prefix, bucket_prefix = {}, None, {}
    seen_hash, duplicates = {}, collections.Counter()
    for phase, rows in (('smoke', smoke), ('eval', requests)):
        for req in rows:
            ids = render_ids(req.prompt)
            local_tokens[(phase, req.request_id)] = len(ids)
            head = ids[:256]
            global_prefix = head if global_prefix is None else global_prefix[:common_prefix_len(global_prefix, head)]
            bucket_prefix[req.bucket] = head if req.bucket not in bucket_prefix else bucket_prefix[req.bucket][:common_prefix_len(bucket_prefix[req.bucket], head)]
            digest = hashlib.sha256(req.prompt.encode()).hexdigest()
            if digest in seen_hash:
                duplicates[(phase, req.bucket)] += 1
            else:
                seen_hash[digest] = (phase, req.request_id)
    template_tokens = len(global_prefix)
    template_block_tokens = (template_tokens // BLOCK) * BLOCK
    header = {b: {'common_prefix_tokens': len(p), 'block_tokens': (len(p) // BLOCK) * BLOCK,
                  'text': tokenizer.decode(p)} for b, p in bucket_prefix.items()}
    attribution = {'block_size': BLOCK, 'template_common_prefix_tokens': template_tokens,
                   'template_common_prefix_text': tokenizer.decode(global_prefix),
                   'template_block_tokens': template_block_tokens, 'per_bucket_common_prefix': header,
                   'duplicate_prompts': {f'{k[0]}:{k[1]}': v for k, v in duplicates.items()},
                   'note': 'vLLM caches full 16-token blocks and never the last prompt token; hits <= ((prompt_tokens-1)//16)*16'}
    write(out / 'attribution.json', attribution)
    log(f'template common prefix {template_tokens} tokens ({template_block_tokens} block-aligned); duplicates {dict(duplicates)}')

    # ---- Server argv: canonical, with ONLY prefix caching flipped (+ usage detail reporting) ----
    from scripts.runs.qwen_baselines import pool_config
    from scripts.runs.serving_ipc import qwen_ipc_environment
    tag = 'pcprobe' + uuid.uuid4().hex[:8]
    local = state / 'ipc' / tag
    env = qwen_ipc_environment(local, os.environ)
    sock = socket.socket(); sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    cfg = pool_config({}, (port, port, port), tag)
    index = definition['models'].index(o.model)
    row = cfg['instances'][index]
    argv_server = server_argv('qwen', model_paths[o.model], row, index, out, bundle / 'qwen/length')
    if argv_server.count('--no-enable-prefix-caching') != 1:
        raise ValueError('Expected exactly one --no-enable-prefix-caching in the canonical argv')
    argv_server = ['--enable-prefix-caching' if a == '--no-enable-prefix-caching' else a for a in argv_server]
    argv_server = [*argv_server[:-2], '--enable-prompt-tokens-details', *argv_server[-2:]]  # keep '--host 127.0.0.1' last
    write(out / f'server_argv_{o.model}.json', argv_server)
    write(out / 'instances.json', {'instances': [row], 'serving_profile': dict(cfg['serving_profile'], prefix_caching=True),
                                   'diagnostic': 'single-model prefix-cache probe; not a control'})
    server_env = dict(env, CUDA_VISIBLE_DEVICES=o.gpu, VLLM_USE_V1='1', VLLM_ATTENTION_BACKEND='FLASH_ATTN',
                      VLLM_USE_FLASHINFER_SAMPLER='0', VLLM_PER_REQUEST_WAIT_LOG_PATH=str(out / f'wait_{o.model}.log'),
                      XDG_CACHE_HOME=str(state / 'cache'), TRITON_CACHE_DIR=str(local / 'triton'), CUDA_CACHE_PATH=str(local / 'nv'))
    sock.close()
    if o.dry_run:
        status('DRY_RUN_COMPLETE'); log('dry run complete: preflight, tokenizer attribution and server argv written'); return

    # ---- Wait for the GPU window ----
    polls = 0
    while True:
        running, used = program_running(o.wait_program), gpu_memory_used(o.gpu)
        status('WAITING_GPU', polls=polls, blocker=o.wait_program, blocker_running=running, gpu_memory_used_mib=used)
        if not running and used == 0:
            time.sleep(20)  # confirm the free window is stable, not a between-cell gap
            if not program_running(o.wait_program) and gpu_memory_used(o.gpu) == 0:
                break
        if polls % 30 == 0:
            log(f'waiting: {o.wait_program} running={running} gpu{o.gpu} used={used} MiB')
        polls += 1
        time.sleep(o.poll_s)
    log(f'GPU {o.gpu} free and {o.wait_program} not RUNNING after {polls} polls; launching')
    write(out / 'hardware.json', dict(hardware([o.gpu]), cpu_affinity=sorted(os.sched_getaffinity(0))))

    server_log = (out / f'server_{o.model}.log').open('x')
    status('STARTING_SERVER')
    launched = time.monotonic()
    proc = subprocess.Popen(argv_server, cwd=ROOT / 'src', env=server_env, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
    budget_end = launched + o.gpu_budget_s
    stop = threading.Event(); foreign = threading.Event(); sampler = None
    address = row['address']
    phases, records, own_pids = {}, [], set()
    sampler_started = time.monotonic()

    def sample_loop():
        with (out / 'timeline.jsonl').open('a') as f:
            while not stop.is_set():
                entry = {'time': time.time(), 'since_launch_s': time.monotonic() - launched,
                         'gpu_memory_used_mib': None, 'metrics': None}
                try:
                    entry['gpu_memory_used_mib'] = gpu_memory_used(o.gpu)
                    # nvidia-smi reports host-namespace PIDs (not resolvable here), so "ours" is the set seen
                    # during the first samples after readiness; any later new PID means another job took the GPU.
                    pids = set(gpu_compute_pids(o.gpu))
                    entry['compute_pids'] = sorted(pids)
                    if time.monotonic() - sampler_started < 20:
                        own_pids.update(pids)
                    elif pids - own_pids:
                        entry['foreign_pids'] = sorted(pids - own_pids); foreign.set()
                except Exception as error:
                    entry['error'] = repr(error)
                try:
                    entry['metrics'] = scrape(address)
                except Exception:
                    pass
                f.write(json.dumps(entry) + '\n'); f.flush()
                stop.wait(5)

    try:
        deadline = time.monotonic() + 600
        while True:
            if proc.poll() is not None:
                raise RuntimeError('Server exited before readiness')
            try:
                with urllib.request.urlopen(address + '/v1/models', timeout=3) as response:
                    if o.model in {r['id'] for r in json.load(response)['data']}:
                        break
            except (OSError, ValueError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('readiness')
            time.sleep(2)
        ready_s = time.monotonic() - launched
        log(f'server ready after {ready_s:.0f}s')
        sampler_started = time.monotonic()
        sampler = threading.Thread(target=sample_loop, daemon=True); sampler.start()

        from openai import AsyncOpenAI
        client = AsyncOpenAI(base_url=address + '/v1', api_key='EMPTY', timeout=900, max_retries=0)

        def reset_cache():
            req = urllib.request.Request(address + '/reset_prefix_cache', method='POST')
            with urllib.request.urlopen(req, timeout=30) as response:
                return response.status

        async def one(req, phase, seq, dispatch_deadline=None):
            if foreign.is_set():
                return {'phase': phase, 'seq': seq, 'request_id': req.request_id, 'bucket': req.bucket, 'skipped': 'foreign_process'}
            if dispatch_deadline is not None and time.monotonic() > dispatch_deadline:
                return {'phase': phase, 'seq': seq, 'request_id': req.request_id, 'bucket': req.bucket, 'skipped': 'budget'}
            t0 = time.monotonic()
            reply = await client.chat.completions.create(
                model=o.model, messages=build_messages(req.prompt, system_prompt), temperature=0, top_p=1,
                max_completion_tokens=1,
                extra_body={'chat_template_kwargs': dict(template_kwargs), 'request_id': f'{phase}-{req.request_id}'})
            usage = reply.usage
            details = getattr(usage, 'prompt_tokens_details', None)
            cached = int(getattr(details, 'cached_tokens', 0) or 0)
            n = int(usage.prompt_tokens)
            return {'phase': phase, 'seq': seq, 'request_id': req.request_id, 'bucket': req.bucket,
                    'prompt_tokens': n, 'cached_tokens': cached, 'max_cacheable': ((n - 1) // BLOCK) * BLOCK,
                    'local_tokens': local_tokens.get((phase if phase in ('smoke', 'eval') else 'eval', req.request_id)),
                    'manifest_prompt_tokens': req.prompt_tokens, 'dispatched_s': t0 - launched,
                    'latency_s': time.monotonic() - t0}

        async def stream(rows, phase, concurrency, dispatch_deadline):
            sem = asyncio.Semaphore(concurrency)
            async def guarded(seq, req):
                async with sem:
                    return await one(req, phase, seq, dispatch_deadline)
            results = await asyncio.gather(*(guarded(i, r) for i, r in enumerate(rows)))
            records.extend(results)
            with (out / 'records.jsonl').open('a') as f:
                for r in results:
                    f.write(json.dumps(r) + '\n')
            return results

        def phase_metrics(name, before, started):
            after = scrape(address)
            phases[name] = dict(delta(before, after), duration_s=time.monotonic() - started, counters_after=after)
            log(f'{name}: {json.dumps({k: phases[name][k] for k in ("requests", "queries", "hits", "hit_rate", "duration_s")})}')
            return after

        async def gpu_phases():
            # warm-up (as warm_up_instances does), then start clean
            status('RUNNING', phase='warmup')
            before = scrape(address); t = time.monotonic()
            await client.chat.completions.create(model=o.model, messages=[{'role': 'user', 'content': 'warm-up'}],
                                                 temperature=0, top_p=1, max_completion_tokens=1)
            before = phase_metrics('warmup', before, t)

            # template isolation: cold cache, first eval request of each bucket in stream order, sequentially, then repeat
            status('RUNNING', phase='isolation')
            reset_cache(); await asyncio.sleep(1)
            first_of_bucket = {}
            for r in requests:
                first_of_bucket.setdefault(r.bucket, r)
            isolation_rows = list(first_of_bucket.values())
            before = scrape(address); t = time.monotonic()
            await stream(isolation_rows, 'isolation-cold', 1, None)
            await stream(isolation_rows, 'isolation-repeat', 1, None)
            before = phase_metrics('isolation', before, t)

            # calibration smoke warm-up (512 prompts), concurrency 128, then the frozen evaluation stream
            status('RUNNING', phase='smoke')
            reset_cache(); await asyncio.sleep(1)
            before = scrape(address); t = time.monotonic()
            await stream(smoke, 'smoke', o.concurrency, budget_end - 60)
            before = phase_metrics('smoke', before, t)
            status('RUNNING', phase='eval', requests=len(requests))
            t = time.monotonic()
            await stream(requests, 'eval', o.concurrency, budget_end - 45)
            phase_metrics('eval', before, t)
            await client.close()

        asyncio.run(gpu_phases())
        gpu_seconds = time.monotonic() - launched
    except BaseException as error:
        status('FAILED', error=repr(error)); log(f'FAILED {error!r}')
        raise
    finally:
        stop.set()
        if sampler is not None:
            sampler.join(timeout=15)
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL); proc.wait()
        server_log.close()
        time.sleep(5)
        log(f'server stopped; gpu{o.gpu} memory used now {gpu_memory_used(o.gpu)} MiB')

    # ---- Analysis ----
    status('ANALYZING', gpu_seconds=gpu_seconds)
    try:
        analyze(out, o, records, phases, attribution, header, template_tokens, template_block_tokens, model_paths, argv_server, gpu_seconds, ready_s, status, log)
    except Exception as error:
        status('ANALYSIS_FAILED', error=repr(error), gpu_seconds=gpu_seconds); log(f'ANALYSIS_FAILED {error!r} (records.jsonl/timeline.jsonl/server log are intact)')
        raise


def analyze(out, o, records, phases, attribution, header, template_tokens, template_block_tokens, model_paths, argv_server, gpu_seconds, ready_s, status, log):
    log_lines = []
    for line in (out / f'server_{o.model}.log').read_text(errors='replace').splitlines():
        m = HIT_LINE.search(line)
        if m:
            stamp = re.search(r'\d\d-\d\d \d\d:\d\d:\d\d', line)
            log_lines.append({'time': stamp.group(0) if stamp else None, 'rolling_pct': float(m.group(1)), 'step_pct': float(m.group(2))})

    def summarize(rows):
        done = [r for r in rows if 'prompt_tokens' in r]
        p = sum(r['prompt_tokens'] for r in done); c = sum(r['cached_tokens'] for r in done)
        mx = sum(r['max_cacheable'] for r in done)
        tb = sum(min(r['cached_tokens'], template_block_tokens) for r in done)
        hb = sum(min(r['cached_tokens'], header[r['bucket']]['block_tokens']) for r in done)
        return {'requests': len(done), 'skipped': len(rows) - len(done), 'prompt_tokens': p, 'cached_tokens': c,
                'token_hit_rate': (c / p) if p else None, 'hit_rate_of_cacheable': (c / mx) if mx else None,
                'requests_with_any_hit': sum(1 for r in done if r['cached_tokens'] > 0),
                'mean_cached_tokens': (c / len(done)) if done else None,
                'hits_template_blocks': tb, 'hits_dataset_header_blocks_beyond_template': hb - tb,
                'hits_beyond_dataset_header': c - hb,
                'token_hit_rate_excluding_template_blocks': ((c - tb) / p) if p else None,
                'token_hit_rate_excluding_dataset_header_blocks': ((c - hb) / p) if p else None,
                'usage_vs_local_token_mismatches': sum(1 for r in done if r.get('local_tokens') not in (None, r['prompt_tokens']))}

    eval_rows = [r for r in records if r['phase'] == 'eval']
    by_bucket = {}
    for bucket in sorted({r['bucket'] for r in eval_rows}):
        rows = [r for r in eval_rows if r['bucket'] == bucket]
        done = [r for r in rows if 'prompt_tokens' in r]
        first, later = (done[0] if done else None), done[1:]
        by_bucket[bucket] = dict(summarize(rows),
                                 first_request={'request_id': first['request_id'], 'prompt_tokens': first['prompt_tokens'],
                                                'cached_tokens': first['cached_tokens']} if first else None,
                                 later_mean_cached_tokens=(sum(r['cached_tokens'] for r in later) / len(later)) if later else None,
                                 later_token_hit_rate=(sum(r['cached_tokens'] for r in later) / sum(r['prompt_tokens'] for r in later)) if later else None)
    isolation = {phase: [{k: r.get(k) for k in ('request_id', 'bucket', 'prompt_tokens', 'cached_tokens', 'max_cacheable')}
                         for r in records if r['phase'] == phase] for phase in ('isolation-cold', 'isolation-repeat')}
    analysis = {
        'probe': 'prefix-cache-probe-20260917', 'model': o.model, 'gpu': o.gpu, 'cpus': o.cpus,
        'served_model_path': model_paths[o.model], 'server_argv': argv_server,
        'serving_config': 'canonical Qwen argv with --enable-prefix-caching (chunked prefill and 32768/512 budgets unchanged) + --enable-prompt-tokens-details',
        'request_set': 'frozen bundle request set (scripts.runs.experiments._build_request_set, canonical Qwen argv, holdout indices from bundle) in frozen arrival order; max_completion_tokens=1',
        'concurrency': o.concurrency, 'gpu_seconds': gpu_seconds, 'server_ready_s': ready_s,
        'phases_metrics': {k: {kk: vv for kk, vv in v.items() if kk != 'counters_after'} for k, v in phases.items()},
        'final_counters': phases['eval']['counters_after'],
        'eval': summarize(eval_rows), 'eval_by_bucket': by_bucket,
        'smoke': summarize([r for r in records if r['phase'] == 'smoke']),
        'isolation': isolation, 'attribution': attribution,
        'server_log_hit_rate_lines': log_lines,
        'kv_cache_note': 'the cache window is the single 0.6B engine KV pool (~39k blocks x 16 tokens); stream is prefill-only at concurrency ' + str(o.concurrency),
    }
    write(out / 'analysis.json', analysis)
    ev = analysis['eval']
    lines = ['# Prefix-cache hit-rate probe (Qwen3-0.6B, frozen evaluation prompts)', '',
             f'GPU {o.gpu} on Vast, {gpu_seconds/60:.1f} GPU-minutes, prefill-only (max_completion_tokens=1), concurrency {o.concurrency}.',
             'Canonical campaign argv with only `--no-enable-prefix-caching` -> `--enable-prefix-caching` (plus `--enable-prompt-tokens-details` for usage reporting).', '',
             '## Evaluation stream (16,000 frozen requests, after 512 calibration smoke prompts)', '',
             f"- completed {ev['requests']} (skipped {ev['skipped']}); prompt tokens {ev['prompt_tokens']:,}; cached tokens {ev['cached_tokens']:,}",
             f"- overall token hit rate: {100*(ev['token_hit_rate'] or 0):.2f}% (of cacheable tokens: {100*(ev['hit_rate_of_cacheable'] or 0):.2f}%)",
             f"- excluding template blocks ({template_block_tokens} tokens/request): {100*(ev['token_hit_rate_excluding_template_blocks'] or 0):.2f}%; excluding dataset-header blocks: {100*(ev['token_hit_rate_excluding_dataset_header_blocks'] or 0):.2f}%",
             f"- /metrics delta over the eval phase: queries {phases['eval']['queries']:,.0f}, hits {phases['eval']['hits']:,.0f}, hit rate {100*(phases['eval']['hit_rate'] or 0):.2f}%",
             '', '| bucket | requests | prompt tokens | cached tokens | hit rate | first-request cached | later mean cached |', '|---|---|---|---|---|---|---|']
    for b, v in by_bucket.items():
        fr = v['first_request']
        lines.append(f"| {b} | {v['requests']} | {v['prompt_tokens']:,} | {v['cached_tokens']:,} | {100*(v['token_hit_rate'] or 0):.2f}% | {fr['cached_tokens'] if fr else 'n/a'} | {v['later_mean_cached_tokens']:.1f} |" if v['later_mean_cached_tokens'] is not None else f"| {b} | {v['requests']} | - | - | - | - | - |")
    lines += ['', '## Template / dataset-header attribution', '',
              f"- shared chat-template+system-prompt prefix: {template_tokens} tokens ({template_block_tokens} block-aligned cacheable)",
              *[f"- {b}: common prefix {h['common_prefix_tokens']} tokens ({h['block_tokens']} block-aligned): {json.dumps(h['text'][-80:])}" for b, h in header.items()],
              f"- duplicate prompts within the streams: {attribution['duplicate_prompts'] or 'none'}",
              '', '## Isolation phase (cold cache, first eval request of each bucket, then repeated)', '',
              *[f"- cold: {r['bucket']} cached {r['cached_tokens']}/{r['prompt_tokens']}" for r in isolation['isolation-cold']],
              *[f"- repeat: {r['bucket']} cached {r['cached_tokens']}/{r['prompt_tokens']} (max cacheable {r['max_cacheable']})" for r in isolation['isolation-repeat']],
              '', 'Files: analysis.json (all numbers), records.jsonl (per request), timeline.jsonl (5 s /metrics + GPU samples), attribution.json, server log.']
    (out / 'README.md').write_text('\n'.join(lines) + '\n')
    status('COMPLETE', gpu_seconds=gpu_seconds, token_hit_rate=ev['token_hit_rate'])
    log(f"COMPLETE: eval token hit rate {ev['token_hit_rate']}")


if __name__ == '__main__':
    main()

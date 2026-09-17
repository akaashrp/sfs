"""Native-simulator replay of candidate remaining-decode targets on saved snapshots (read-only).

Two modes:
  hindsight  : for each saved snapshot, replace running-decode targets by each candidate and by
               the hindsight oracle (eventual lengths); simulate hypothetical probes (128/4096/16384
               prompt tokens) and report each candidate's wait-estimate difference from the oracle.
  dispatch   : for a dense-snapshot single-model diagnostic run, align every real request with the
               latest snapshot published before its engine enqueue time, simulate that request's own
               prompt under the current targets and each candidate, and compare against the engine's
               measured TTFT. This is measured evidence (real TTFT), not hindsight.

Candidate tables are fitted exactly as in evaluate_tail.py: calibration-only tables (prompts
disjoint from the evaluation set) and prompt-fold held-out online tables.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import struct
import sys
from collections import defaultdict
from pathlib import Path

import multiprocessing as mp
import msgspec
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_tail import MODELS, FOLDS, SurvivalTable, Backoff, build_tables, prompt_bin, pred_bin, fold_of  # noqa: E402

HEADER = struct.Struct('<8sIIQQdQdddddddd')
PROBES = (128, 4096, 16384)


def load_native(extension):
    import torch  # noqa: F401  (libtorch symbols before the extension; no GPU work)
    spec = importlib.util.spec_from_file_location('_scheduler_sim', extension)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class Estimators:
    def __init__(self, online_rows, calibration_rows):
        self.cal = {m: build_tables([r for r in calibration_rows if r['model'] == m], [lambda r: prompt_bin(r['pt']), lambda r: 'all']) for m in MODELS}
        self.online = {}
        for m in MODELS:
            for f in range(FOLDS):
                train = [r for r in online_rows if r['model'] == m and r['fold'] != f]
                self.online[(m, f)] = build_tables(train, [lambda r: (prompt_bin(r['pt']), pred_bin(r['p'])), lambda r: prompt_bin(r['pt']), lambda r: 'all'])

    def targets(self, model, g, cap, p, pt, fold, current_T, quantiles=(0.5, 0.65, 0.8)):
        out = {'current': current_T}
        def put(name, T):
            T = g + 1 if T is None else T
            T = min(max(int(T), g + 1), cap)
            out[name] = T
            out['exhausted_only_' + name] = T if (current_T - g) <= 1 else current_T
        for qq in quantiles:
            tag = f'q{int(qq*100)}'
            put(f'cal_model_{tag}', self.cal[model].levels[-1]['all'].estimate(g, qq))
            put(f'cal_prompt_{tag}', self.cal[model].estimate([prompt_bin(pt), 'all'], g, qq)[0])
            put(f'online_prompt_pred_{tag}', self.online[(model, fold)].estimate([(prompt_bin(pt), pred_bin(p)), prompt_bin(pt), 'all'], g, qq)[0])
        return out


def decode(path):
    raw = path.read_bytes(); h = HEADER.unpack(raw[:HEADER.size])
    assert h[0] == b'VLLMSHM1' and h[2] == HEADER.size and len(raw) == HEADER.size + h[6]
    return raw, h, msgspec.msgpack.decode(raw[HEADER.size:])


VARIANTS = None  # optional set of variant names to simulate (oracle and current always kept)


def variants_for(state, model, lookup, est, run_lookup_key='response_id'):
    """Return (variant states, rows) where variant states carry candidate targets for running decode requests."""
    variants = {}; rows = []
    reserve = int(state['decode_reserve_tokens'])
    for rid in state['running_request_ids']:
        r = state['requests'][rid]
        if r['num_prompt_processed_tokens'] < r['num_prompt_tokens']:
            continue
        actual = lookup.get(rid)
        if actual is None:
            continue
        g = int(r['num_output_processed_tokens']); T = int(r['num_output_target_tokens'])
        cap = min(int(r['max_tokens']), int(state['config']['max_model_len']) - int(r['num_prompt_tokens']))
        L = int(actual['L'])
        assert 0 <= g <= L <= cap
        targets = est.targets(model, g, cap, float(actual['p']), int(r['num_prompt_tokens']), actual['fold'], T)
        assert targets['current'] == T
        targets['oracle'] = max(g + 1, L)
        for name, target in targets.items():
            if VARIANTS is not None and name not in VARIANTS and name not in ('current', 'oracle'):
                continue
            if name not in variants:
                variants[name] = msgspec.msgpack.decode(msgspec.msgpack.encode(state))
            variants[name]['requests'][rid]['num_output_target_tokens'] = int(target)
        rows.append({'id': rid, 'g': g, 'L': L, 'reserve': reserve, 'targets': targets})
    # Waiting requests keep prediction + reserve in every variant above (the design only varies running-decode
    # targets). Two extra variants also replace the waiting requests' targets: with their eventual length
    # (oracle_all: everything the simulator can know about lengths) and with the calibration prompt-bin median
    # at zero generated tokens (cal_prompt_q50_all: a deployable rule that ignores the router prediction).
    max_len = int(state['config']['max_model_len'])
    waiting = [(rid, state['requests'][rid], lookup[rid]) for rid in state.get('waiting_request_ids', ())
               if rid in state['requests'] and rid in lookup]
    if rows and waiting:
        for name, base, fn in (('oracle_all', 'oracle', lambda r, a: int(a['L'])),
                               ('cal_prompt_q50_all', 'cal_prompt_q50', lambda r, a: est.cal[model].estimate([prompt_bin(int(r['num_prompt_tokens'])), 'all'], 0, 0.5)[0])):
            if base not in variants or (VARIANTS is not None and name not in VARIANTS):
                continue
            v = msgspec.msgpack.decode(msgspec.msgpack.encode(variants[base]))
            for rid, r, a in waiting:
                cap = min(int(r['max_tokens']), max_len - int(r['num_prompt_tokens']))
                T = fn(r, a)
                v['requests'][rid]['num_output_target_tokens'] = int(min(max(T or 1, 1), cap))
            variants[name] = v
    return variants, rows


def worker_for(native, h):
    return native.SchedulerSimulationWorker(1., h[9], h[10], h[12], h[13], h[11], h[14])


_G = {}


def _hindsight_task(item):
    path, run = item
    native, est, lookups = _G['native'], _G['est'], _G['lookups']
    raw, h, state = decode(path)
    model = path.stem.split('-', 1)[1]
    variants, rows = variants_for(state, model, lookups[run], est)
    if not rows:
        return []
    w = worker_for(native, h)
    out = []
    for prompt in PROBES:
        estimates = {name: w.run_simulation_for_test(msgspec.msgpack.encode(v), prompt, 'prefill_done', (), state['created_at'], 0.)['estimated_wait_ms'] for name, v in variants.items()}
        out.append({'run': run, 'snapshot': path.name, 'sha256': hashlib.sha256(raw).hexdigest(), 'model': model, 'prompt_tokens': prompt,
                    'running_decode': len(rows), 'num_waiting': state['num_waiting'], 'estimates_ms': estimates})
    print(path.name, len(rows), flush=True)
    return out


def hindsight(args, native, est, lookups):
    _G.update(native=native, est=est, lookups=lookups)
    items = [(path, run) for snap_dir, run in args.snapshots for path in sorted(Path(snap_dir).glob('*.bin'))]
    out = []
    with mp.get_context('fork').Pool(args.workers) as pool:
        for part in pool.imap_unordered(_hindsight_task, items):
            out.extend(part)
    out.sort(key=lambda r: (r['run'], r['snapshot'], r['prompt_tokens']))
    summary = defaultdict(lambda: defaultdict(list))
    for r in out:
        o = r['estimates_ms']['oracle']
        for k, v in r['estimates_ms'].items():
            summary[r['model']][k].append(v - o)
    table = {m: {k: {'n': len(v), 'mae_ms': float(np.mean(np.abs(v))), 'p90_abs_ms': float(np.quantile(np.abs(v), .9)), 'mean_under_ms': float(np.mean(np.clip(-np.array(v), 0, None))), 'mean_over_ms': float(np.mean(np.clip(v, 0, None)))}
                  for k, v in d.items()} for m, d in summary.items()}
    return {'mode': 'hindsight', 'probes': PROBES, 'rows': out, 'summary': table}


def _dispatch_task(item):
    i, reqs = item
    created_at, path, h, state = _G['snaps'][i]
    variants, vrows = variants_for(state, _G['model'], _G['lookup'], _G['est'])
    if not variants:
        variants = {'current': state}
    w = worker_for(_G['native'], h)
    encoded = {name: msgspec.msgpack.encode(v) for name, v in variants.items()}
    out = []
    for r, staleness in reqs:
        estimates = {name: w.run_simulation_for_test(payload, int(r['prompt_tokens']), 'prefill_done', (), state['created_at'], 0.)['estimated_wait_ms'] for name, payload in encoded.items()}
        out.append({'request_id': r['request_id'], 'snapshot': path.name, 'staleness_s': staleness, 'prompt_tokens': r['prompt_tokens'],
                    'ttft_ms': r['ttft_ms'], 'e2e_ttft_ms': r.get('e2e_ttft_ms'), 'router_wait_ms': r['wait_time_ms'], 'ttft_slo_ms': r['ttft_slo_ms'],
                    'running_decode': len(vrows), 'num_waiting': state['num_waiting'], 'kv_usage': state['kv_cache_config'].get('kv_cache_usage'), 'estimates_ms': estimates})
    return out


def dispatch(args, native, est, lookups):
    """Align each real request of the dense-snapshot run to the last snapshot before its engine enqueue."""
    run = args.snapshots[0][1]
    snap_dir = Path(args.snapshots[0][0])
    index = [json.loads(l) for l in (snap_dir / 'index.jsonl').open()]
    files = sorted(snap_dir.glob('*.bin'))
    snaps = []
    for path in files:
        raw, h, state = decode(path)
        snaps.append((state['created_at'], path, h, state))
    snaps.sort(key=lambda t: t[0])
    created = np.array([s[0] for s in snaps])
    per_request = json.load(open(args.point))['router']['runs'][0]['per_request']
    model = per_request[0]['response_model']
    lookup = lookups[run]
    groups = defaultdict(list)
    for r in sorted(per_request, key=lambda r: r['queued_ts_s']):
        if r.get('error'):
            continue
        t = float(r['queued_ts_s'])
        i = int(np.searchsorted(created, t, side='right')) - 1
        if i < 0:
            continue
        staleness = t - snaps[i][0]
        if staleness > args.max_staleness_s:
            continue
        groups[i].append((r, staleness))
    _G.update(native=native, est=est, lookup=lookup, model=model, snaps=snaps)
    rows = []
    with mp.get_context('fork').Pool(args.workers) as pool:
        for part in pool.imap_unordered(_dispatch_task, list(groups.items())):
            rows.extend(part)
    rows.sort(key=lambda r: r['request_id'])
    names = sorted({k for r in rows for k in r['estimates_ms']})
    summary = {}
    ttft_all = np.array([r['ttft_ms'] for r in rows]); router = np.array([r['router_wait_ms'] for r in rows])
    slo_all = np.array([r['ttft_slo_ms'] for r in rows])
    def stats(est):
        # A request aligned to a snapshot with no running decode request has only the 'current' variant; skip it.
        keep = np.isfinite(est); est = est[keep]; ttft = ttft_all[keep]; slo = slo_all[keep]
        err = est - ttft; ae = np.abs(err)
        return {'n': int(len(err)), 'mae_ms': float(ae.mean()), 'median_ae_ms': float(np.median(ae)), 'p90_ae_ms': float(np.quantile(ae, .9)), 'p95_ae_ms': float(np.quantile(ae, .95)),
                'p99_ae_ms': float(np.quantile(ae, .99)), 'mean_under_ms': float(np.clip(-err, 0, None).mean()), 'mean_over_ms': float(np.clip(err, 0, None).mean()),
                'under_by_gt_1s_pct': float((err < -1000).mean() * 100), 'over_by_gt_1s_pct': float((err > 1000).mean() * 100),
                'slo_decision_agreement_pct': float(((est <= slo) == (ttft <= slo)).mean() * 100)}
    summary['router_logged'] = stats(router)
    for k in names:
        summary[k] = stats(np.array([r['estimates_ms'].get(k, np.nan) for r in rows], dtype=float))
    summary['replayed_current_vs_router_logged'] = {'mae_ms': float(np.mean(np.abs(np.array([r['estimates_ms']['current'] for r in rows]) - router))),
                                                    'median_abs_ms': float(np.median(np.abs(np.array([r['estimates_ms']['current'] for r in rows]) - router)))}
    return {'mode': 'dispatch', 'model': model, 'aligned_requests': len(rows), 'total_requests': len(per_request), 'max_staleness_s': args.max_staleness_s,
            'snapshots': len(snaps), 'rows': rows, 'summary': summary}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=['hindsight', 'dispatch'], required=True)
    ap.add_argument('--extension', type=Path, required=True)
    ap.add_argument('--online', type=Path, required=True, help='slim online rows jsonl (model, pt, p, L, fold, run, response_id)')
    ap.add_argument('--calibration', type=Path, required=True)
    ap.add_argument('--snapshots', nargs='+', required=True, help='dir=run entries')
    ap.add_argument('--point', type=Path, help='dispatch mode: the diagnostic run point.json')
    ap.add_argument('--max-staleness-s', type=float, default=3.0)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--variants', help='comma-separated variant names to simulate (current and oracle always included)')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.snapshots = [tuple(s.split('=')) for s in args.snapshots]
    global VARIANTS
    VARIANTS = set(args.variants.split(',')) if args.variants else None
    native = load_native(args.extension)
    online = [json.loads(l) for l in args.online.open()]
    calibration = [json.loads(l) for l in args.calibration.open()]
    for r in calibration:
        r['L'] = int(r['completion_tokens']); r['pt'] = int(r['prompt_tokens'])
    est = Estimators(online, calibration)
    lookups = defaultdict(dict)
    for r in online:
        lookups[r['run']][r['response_id']] = r
    if args.mode == 'dispatch':
        # The diagnostic run replays repeat1's request identities; their eventual lengths come from its own point.
        run = args.snapshots[0][1]
        for r in json.load(open(args.point))['router']['runs'][0]['per_request']:
            base = lookups['repeat1'].get(r['response_id'])
            lookups[run][r['response_id']] = {'L': int(r['usage_completion_tokens']), 'p': float(r['predicted_output_tokens']),
                                              'fold': base['fold'] if base else fold_of(r['request_id']), 'model': r['response_model']}
        result = dispatch(args, native, est, lookups)
    else:
        result = hindsight(args, native, est, lookups)
    result['extension_sha256'] = hashlib.sha256(args.extension.read_bytes()).hexdigest()
    result['scope'] = ('Read-only replay through the deployed native simulator; hypothetical probes and hindsight lengths are not measured latency. '
                       'dispatch mode compares simulator estimates from the last published snapshot (no pending-dispatch overlay) with measured engine TTFT.')
    args.output.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result['summary'], indent=1)[:6000])


if __name__ == '__main__':
    main()
